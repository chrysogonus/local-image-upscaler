from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

DATASET_SCHEMA_VERSION = 1
PREPARED_SCHEMA_VERSION = 1
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
MAX_SOURCE_PIXELS = 100_000_000
ALLOWED_TRACKS = {"paired", "authentic"}
DEGRADATIONS = {"clean", "optical", "noise-jpeg", "combined"}


class BenchmarkDataError(ValueError):
    """The benchmark definition or its downloaded data is invalid."""


@dataclass(frozen=True, slots=True)
class BenchmarkSource:
    url: str
    sha256: str
    width: int
    height: int
    source_page: str
    creator: str
    license: str


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    id: str
    title: str
    track: str
    tags: tuple[str, ...]
    source: BenchmarkSource
    crop: tuple[int, int, int, int]
    degradation: str | None


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    id: str
    name: str
    description: str
    license_policy: str
    rights_url: str
    cases: tuple[BenchmarkCase, ...]
    digest: str


def default_manifest_path() -> Path:
    return Path(__file__).with_name("dataset.json")


def benchmark_root(value: Path | None = None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    configured = os.getenv("UPSCALER_BENCHMARK_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.cwd() / ".upscaler" / "benchmarks").resolve()


def _require_string(mapping: dict[str, Any], name: str, context: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkDataError(f"{context}.{name} must be a non-empty string")
    return value


def _load_case(raw: Any) -> BenchmarkCase:
    if not isinstance(raw, dict):
        raise BenchmarkDataError("every dataset case must be an object")
    case_id = _require_string(raw, "id", "case")
    context = f"case {case_id!r}"
    track = _require_string(raw, "track", context)
    if track not in ALLOWED_TRACKS:
        raise BenchmarkDataError(f"{context}.track must be paired or authentic")
    tags = raw.get("tags")
    if (
        not isinstance(tags, list)
        or not tags
        or any(not isinstance(tag, str) or not tag for tag in tags)
    ):
        raise BenchmarkDataError(f"{context}.tags must contain non-empty strings")
    source_raw = raw.get("source")
    if not isinstance(source_raw, dict):
        raise BenchmarkDataError(f"{context}.source must be an object")
    source_width = source_raw.get("width")
    source_height = source_raw.get("height")
    if not isinstance(source_width, int) or not isinstance(source_height, int):
        raise BenchmarkDataError(f"{context} source dimensions must be integers")
    source = BenchmarkSource(
        url=_require_string(source_raw, "url", f"{context}.source"),
        sha256=_require_string(source_raw, "sha256", f"{context}.source"),
        width=source_width,
        height=source_height,
        source_page=_require_string(source_raw, "source_page", f"{context}.source"),
        creator=_require_string(source_raw, "creator", f"{context}.source"),
        license=_require_string(source_raw, "license", f"{context}.source"),
    )
    if not source.url.startswith("https://") or not source.source_page.startswith("https://"):
        raise BenchmarkDataError(f"{context} source URLs must use HTTPS")
    if len(source.sha256) != 64 or any(char not in "0123456789abcdef" for char in source.sha256):
        raise BenchmarkDataError(f"{context}.source.sha256 must be lowercase SHA-256")
    if source.width <= 0 or source.height <= 0 or source.width * source.height > MAX_SOURCE_PIXELS:
        raise BenchmarkDataError(f"{context} source dimensions are outside the safe limit")
    if "public domain" not in source.license.lower():
        raise BenchmarkDataError(f"{context} is not explicitly marked public domain")
    crop_raw = raw.get("crop")
    if not isinstance(crop_raw, dict):
        raise BenchmarkDataError(f"{context}.crop must be an object")
    crop_values_raw = tuple(crop_raw.get(key) for key in ("left", "top", "width", "height"))
    if any(not isinstance(value, int) for value in crop_values_raw):
        raise BenchmarkDataError(f"{context} crop coordinates must be integers")
    left, top, width, height = (int(value) for value in crop_values_raw if isinstance(value, int))
    if left < 0 or top < 0 or width <= 0 or height <= 0:
        raise BenchmarkDataError(f"{context} crop coordinates are invalid")
    if left + width > source.width or top + height > source.height:
        raise BenchmarkDataError(f"{context} crop extends beyond the source")
    degradation = raw.get("degradation")
    if track == "paired" and degradation not in DEGRADATIONS:
        raise BenchmarkDataError(f"{context} must select a supported degradation")
    if track == "authentic" and degradation is not None:
        raise BenchmarkDataError(f"{context} authentic inputs cannot add a degradation")
    return BenchmarkCase(
        id=case_id,
        title=_require_string(raw, "title", context),
        track=track,
        tags=tuple(tags),
        source=source,
        crop=(left, top, width, height),
        degradation=degradation,
    )


def load_dataset(path: Path | None = None) -> BenchmarkDataset:
    manifest_path = path or default_manifest_path()
    payload_bytes = manifest_path.read_bytes()
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError as exc:
        raise BenchmarkDataError(f"Could not parse {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkDataError("dataset manifest must be an object")
    if payload.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise BenchmarkDataError(
            f"unsupported dataset schema {payload.get('schema_version')!r}; "
            f"expected {DATASET_SCHEMA_VERSION}"
        )
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise BenchmarkDataError("dataset cases must be a list")
    cases = tuple(_load_case(raw) for raw in raw_cases)
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise BenchmarkDataError("dataset case ids must be unique")
    return BenchmarkDataset(
        id=_require_string(payload, "id", "dataset"),
        name=_require_string(payload, "name", "dataset"),
        description=_require_string(payload, "description", "dataset"),
        license_policy=_require_string(payload, "license_policy", "dataset"),
        rights_url=_require_string(payload, "rights_url", "dataset"),
        cases=cases,
        digest=hashlib.sha256(payload_bytes).hexdigest(),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download(source: BenchmarkSource, destination: Path) -> None:
    if destination.is_file() and _sha256(destination) == source.sha256:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        source.url,
        headers={"User-Agent": "local-image-upscaler-benchmark/1"},
    )
    temporary: Path | None = None
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,  # noqa: S310 - pinned HTTPS
            tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix="download-", delete=False
            ) as output,
        ):
            if not response.geturl().startswith("https://"):
                raise BenchmarkDataError(f"download redirected outside HTTPS: {source.url}")
            temporary = Path(output.name)
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    limit_mib = MAX_DOWNLOAD_BYTES // 1024 // 1024
                    raise BenchmarkDataError(f"{source.url} exceeds the {limit_mib} MiB limit")
                output.write(chunk)
        actual = _sha256(temporary)
        if actual != source.sha256:
            raise BenchmarkDataError(
                f"checksum mismatch for {source.url}: expected {source.sha256}, got {actual}"
            )
        temporary.replace(destination)
    except BenchmarkDataError:
        raise
    except (OSError, urllib.error.URLError) as exc:
        raise BenchmarkDataError(f"could not download {source.url}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _decoded_crop(source_path: Path, case: BenchmarkCase) -> Image.Image:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            with Image.open(source_path) as opened:
                if opened.format not in {"JPEG", "PNG"}:
                    raise BenchmarkDataError(
                        f"{case.id} decoded as unsupported format {opened.format!r}"
                    )
                if getattr(opened, "is_animated", False):
                    raise BenchmarkDataError(f"{case.id} source must not be animated")
                oriented = ImageOps.exif_transpose(opened)
                oriented.load()
        except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise BenchmarkDataError(f"could not safely decode {case.id}: {exc}") from exc
    if oriented.size != (case.source.width, case.source.height):
        raise BenchmarkDataError(
            f"{case.id} dimensions changed: expected "
            f"{case.source.width}x{case.source.height}, got {oriented.width}x{oriented.height}"
        )
    left, top, width, height = case.crop
    return oriented.convert("RGB").crop((left, top, left + width, top + height))


def degrade_reference(reference: Image.Image, recipe: str, seed: int) -> tuple[Image.Image, str]:
    """Create one deterministic 4x input while retaining actual JPEG artifacts."""
    if recipe not in DEGRADATIONS:
        raise BenchmarkDataError(f"unknown degradation recipe {recipe!r}")
    if reference.width % 4 or reference.height % 4:
        raise BenchmarkDataError("paired reference dimensions must be divisible by four")
    working = reference.convert("RGB")
    if recipe in {"optical", "combined"}:
        radius = 1.2 if recipe == "optical" else 1.6
        working = working.filter(ImageFilter.GaussianBlur(radius=radius))
    working = working.resize(
        (reference.width // 4, reference.height // 4), Image.Resampling.LANCZOS
    )
    if recipe in {"noise-jpeg", "combined"}:
        sigma = 4.0 if recipe == "noise-jpeg" else 6.0
        values = np.asarray(working, dtype=np.float32)
        noise = np.random.default_rng(seed).normal(0.0, sigma, values.shape)
        values = np.clip(np.rint(values + noise), 0, 255).astype(np.uint8)
        working = Image.fromarray(values, "RGB")
        buffer = io.BytesIO()
        working.save(
            buffer,
            format="JPEG",
            quality=76 if recipe == "noise-jpeg" else 60,
            subsampling=2,
            optimize=False,
            progressive=False,
        )
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            encoded.load()
            working = encoded.convert("RGB")
        # Store the decoded pixels losslessly. The JPEG ringing/blocking is now
        # part of those pixels; a second JPEG encode would add an unrecorded
        # degradation on top of the named recipe.
        return working, "PNG"
    return working, "PNG"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_dataset(
    root: Path | None = None,
    *,
    manifest_path: Path | None = None,
    fetch: bool = True,
) -> Path:
    dataset = load_dataset(manifest_path)
    destination = benchmark_root(root) / "data" / dataset.id
    cases_root = destination / "cases"
    prepared_cases: list[dict[str, Any]] = []
    for index, case in enumerate(dataset.cases):
        case_root = cases_root / case.id
        source_path = case_root / "source.jpg"
        if fetch:
            _download(case.source, source_path)
        elif not source_path.is_file() or _sha256(source_path) != case.source.sha256:
            raise BenchmarkDataError(f"{case.id} is not downloaded or has the wrong checksum")
        crop = _decoded_crop(source_path, case)
        input_format = "PNG"
        reference_path: Path | None = None
        if case.track == "paired":
            reference_path = case_root / "reference.png"
            crop.save(reference_path, format="PNG")
            materialized, input_format = degrade_reference(
                crop, case.degradation or "", seed=10_000 + index
            )
        else:
            materialized = crop
        input_suffix = ".jpg" if input_format == "JPEG" else ".png"
        input_path = case_root / f"input{input_suffix}"
        for obsolete in (case_root / "input.jpg", case_root / "input.png"):
            if obsolete != input_path:
                obsolete.unlink(missing_ok=True)
        if input_format == "JPEG":
            materialized.save(
                input_path,
                format="JPEG",
                quality=95,
                subsampling=2,
                optimize=False,
                progressive=False,
            )
        else:
            materialized.save(input_path, format="PNG")
        prepared_cases.append(
            {
                "id": case.id,
                "title": case.title,
                "track": case.track,
                "tags": list(case.tags),
                "degradation": case.degradation,
                "input": str(input_path.relative_to(destination)),
                "input_sha256": _sha256(input_path),
                "input_width": materialized.width,
                "input_height": materialized.height,
                "reference": (
                    str(reference_path.relative_to(destination)) if reference_path else None
                ),
                "reference_sha256": _sha256(reference_path) if reference_path else None,
                "source_page": case.source.source_page,
                "creator": case.source.creator,
                "license": case.source.license,
            }
        )
    prepared = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "dataset_id": dataset.id,
        "dataset_digest": dataset.digest,
        "name": dataset.name,
        "description": dataset.description,
        "license_policy": dataset.license_policy,
        "rights_url": dataset.rights_url,
        "cases": prepared_cases,
    }
    prepared_path = destination / "prepared.json"
    _write_json(prepared_path, prepared)
    return prepared_path


def load_prepared(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkDataError(f"could not read prepared dataset {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != PREPARED_SCHEMA_VERSION:
        raise BenchmarkDataError("unsupported prepared dataset schema")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise BenchmarkDataError("prepared dataset has no cases")
    return payload
