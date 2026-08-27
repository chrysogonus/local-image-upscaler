from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
from PIL import Image

from upscaler.config import load_config
from upscaler.imaging.pipeline import ProcessResult, process_image
from upscaler.models import ModelRegistry, realesrgan_cuda, realesrgan_ncnn, spandrel_sr
from upscaler.models.base import ModelAdapter
from upscaler.models.tiled import DEFAULT_OVERLAP
from upscaler.schemas import JobSettings, ProcessingMode

from .dataset import benchmark_root, load_prepared, prepare_dataset

RUN_SCHEMA_VERSION = 1
STANDARD_CANDIDATES = ("classical", "swinir", "realesrgan")
CANDIDATE_NAMES = {
    "classical": "Lanczos baseline",
    "swinir": "SwinIR-L",
    "realesrgan": "Real-ESRGAN",
}


class BenchmarkRunError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_revision() -> str | None:
    repository = Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if len(value) == 40 else None


def _fingerprint_files(adapter: ModelAdapter) -> list[dict[str, Any]]:
    paths: list[Path] = []
    if adapter.id == "spandrel-sr":
        paths.append(spandrel_sr.checkpoint_path())
    elif adapter.id == "realesrgan-cuda":
        paths.extend(
            realesrgan_cuda.weights_dir() / name for name in realesrgan_cuda.REQUIRED_WEIGHTS
        )
    elif adapter.id == "realesrgan":
        binary = adapter.binary  # type: ignore[attr-defined]
        if binary:
            paths.append(binary)
            paths.extend(
                binary.parent / "models" / name for name in realesrgan_ncnn.REQUIRED_MODEL_FILES
            )
    fingerprints = []
    for path in paths:
        if not path.is_file():
            raise BenchmarkRunError(f"benchmark dependency disappeared during preflight: {path}")
        fingerprints.append(
            {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    return fingerprints


def _resolved_precision(adapter: ModelAdapter) -> str:
    if adapter.id == "classical":
        return "uint8 RGB / Pillow"
    if adapter.id == "realesrgan":
        return "NCNN runtime-managed"
    if adapter.id == "realesrgan-cuda":
        import torch

        return str(realesrgan_cuda._resolve_dtype(torch)).removeprefix("torch.")
    if adapter.id == "spandrel-sr":
        model = getattr(adapter, "_model", None)
        if model is not None:
            return str(next(model.model.parameters()).dtype).removeprefix("torch.")
        return "recorded after model load"
    return "unknown"


def _resolved_overlap(adapter: ModelAdapter) -> int | str:
    if adapter.id in {"spandrel-sr", "realesrgan-cuda"}:
        return DEFAULT_OVERLAP
    if adapter.id == "realesrgan":
        return "NCNN runtime-managed"
    return 0


def _candidate_fingerprint(candidate_id: str, adapter: ModelAdapter) -> dict[str, Any]:
    return {
        "id": candidate_id,
        "name": CANDIDATE_NAMES[candidate_id],
        "adapter_id": adapter.id,
        "adapter_name": adapter.name,
        "device": adapter.device,
        "precision": _resolved_precision(adapter),
        "license": adapter.license,
        "files": _fingerprint_files(adapter),
    }


def _diagnostics(output: Path, reference: Path | None) -> dict[str, float] | None:
    if reference is None:
        return None
    with Image.open(output) as output_image, Image.open(reference) as reference_image:
        actual = np.asarray(output_image.convert("RGB"), dtype=np.float32)
        expected = np.asarray(reference_image.convert("RGB"), dtype=np.float32)
    if actual.shape != expected.shape:
        raise BenchmarkRunError(
            f"paired output shape {actual.shape} does not match reference {expected.shape}"
        )
    difference = actual - expected
    mae = float(np.abs(difference).mean())
    mse = float(np.square(difference).mean())
    psnr = 20.0 * math.log10(255.0) - 10.0 * math.log10(mse) if mse else 99.0
    return {"mae": round(mae, 6), "psnr": round(psnr, 6)}


def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _load_existing(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkRunError(f"could not read existing run manifest: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != RUN_SCHEMA_VERSION:
        raise BenchmarkRunError("existing run uses an unsupported schema")
    return payload


def _valid_completed_outputs(run_dir: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    valid = []
    for item in payload.get("outputs", []):
        if not isinstance(item, dict):
            continue
        relative_path = item.get("path")
        expected_hash = item.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            continue
        output = run_dir / relative_path
        if output.is_file() and _sha256(output) == expected_hash:
            valid.append(item)
    return valid


def run_benchmark(
    root: Path | None = None,
    *,
    prepared_path: Path | None = None,
    run_dir: Path | None = None,
    registry: ModelRegistry | None = None,
    processor: Callable[..., ProcessResult] = process_image,
) -> Path:
    resolved_root = benchmark_root(root)
    prepared_path = prepared_path or resolved_root / "data" / "faithful-photo-v1" / "prepared.json"
    if not prepared_path.is_file():
        prepared_path = prepare_dataset(resolved_root)
    prepared = load_prepared(prepared_path)
    prepared_root = prepared_path.parent
    cases = prepared["cases"]
    max_width = max(int(case["input_width"]) for case in cases)
    max_height = max(int(case["input_height"]) for case in cases)
    settings = JobSettings(
        target_edge=max(max_width, max_height) * 4,
        processing_mode=ProcessingMode.upscale,
        sharpen=0,
        tile_size=0,
        tta=False,
        restore_large=False,
        max_neural_passes=1,
    )
    config = load_config()
    registry = registry or ModelRegistry(config)
    plans = {
        candidate: registry.resolve_benchmark_candidate(
            candidate, settings, width=max_width, height=max_height
        )
        for candidate in STANDARD_CANDIDATES
    }
    if spandrel_sr.checkpoint_path().name != spandrel_sr.WEIGHTS_FILENAME:
        raise BenchmarkRunError(
            "the standard benchmark requires the pinned SwinIR-L checkpoint; "
            "unset UPSCALER_SR_MODEL before running it"
        )
    candidate_fingerprints = [
        _candidate_fingerprint(candidate, plans[candidate].adapter)
        for candidate in STANDARD_CANDIDATES
    ]

    if run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = resolved_root / "runs" / f"{stamp}-{uuid.uuid4().hex[:8]}"
    else:
        run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run.json"
    if manifest_path.exists():
        run = _load_existing(manifest_path)
        if run.get("dataset", {}).get("digest") != prepared.get("dataset_digest"):
            raise BenchmarkRunError("cannot resume: prepared dataset digest changed")
        if run.get("dataset", {}).get("prepared_digest") != _sha256(prepared_path):
            raise BenchmarkRunError("cannot resume: materialized benchmark inputs changed")
        prior_candidates = {item.get("id"): item for item in run.get("candidates", [])}
        for current in candidate_fingerprints:
            prior = prior_candidates.get(current["id"])
            if not isinstance(prior, dict) or (
                prior.get("adapter_id") != current["adapter_id"]
                or prior.get("files") != current["files"]
            ):
                raise BenchmarkRunError(
                    "cannot resume: resolved candidate runtime or model files changed"
                )
            if current["precision"] == "recorded after model load":
                current["precision"] = prior.get("precision", current["precision"])
        outputs = _valid_completed_outputs(run_dir, run)
        run["outputs"] = outputs
        run["state"] = "incomplete"
        run["updated_at"] = _utc_now()
        run["candidates"] = candidate_fingerprints
    else:
        run = {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": run_dir.name,
            "state": "incomplete",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "git_revision": _git_revision(),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "pillow": _package_version("pillow"),
                "numpy": _package_version("numpy"),
                "torch": _package_version("torch"),
                "spandrel": _package_version("spandrel"),
            },
            "dataset": {
                "id": prepared["dataset_id"],
                "digest": prepared["dataset_digest"],
                "prepared_digest": _sha256(prepared_path),
                "prepared_manifest": _relative(prepared_path, run_dir),
                "name": prepared["name"],
                "license_policy": prepared["license_policy"],
                "rights_url": prepared["rights_url"],
            },
            "settings": {
                "scale": 4,
                "processing_mode": "upscale",
                "sharpen": 0,
                "tta": False,
                "restore_large": False,
                "max_neural_passes": 1,
                "requested_tile_size": 0,
                "preprocessing": "dataset-manifest degradation; production normalize_input",
                "finishing": "single exact resize; sharpening disabled",
                "encoder": "PNG",
            },
            "candidates": candidate_fingerprints,
            "cases": cases,
            "outputs": [],
        }
        outputs = []
    _write_json(manifest_path, run)

    completed = {(item["case_id"], item["candidate_id"]) for item in outputs}
    cancel = Event()
    try:
        for case in cases:
            input_path = prepared_root / case["input"]
            reference_path = prepared_root / case["reference"] if case["reference"] else None
            target_edge = max(int(case["input_width"]), int(case["input_height"])) * 4
            for candidate in STANDARD_CANDIDATES:
                if (case["id"], candidate) in completed:
                    continue
                plan = plans[candidate]
                output = run_dir / "outputs" / case["id"] / f"{candidate}.png"
                workspace = run_dir / "work" / case["id"] / candidate
                output.parent.mkdir(parents=True, exist_ok=True)
                workspace.mkdir(parents=True, exist_ok=True)
                case_settings = settings.model_copy(
                    update={"target_edge": target_edge, "tile_size": plan.tile_size}
                )
                started = time.monotonic()
                try:
                    result = processor(
                        input_path,
                        output,
                        workspace,
                        input_path.name,
                        case_settings,
                        plan.adapter,
                        config.max_input_pixels,
                        cancel,
                        lambda _phase, _message, _progress: None,
                        None,
                        resolved_tile_size=plan.tile_size,
                    )
                    expected_size = (
                        int(case["input_width"]) * 4,
                        int(case["input_height"]) * 4,
                    )
                    if (result.result.width, result.result.height) != expected_size:
                        raise BenchmarkRunError(
                            f"{case['id']} / {candidate} returned "
                            f"{result.result.width}x{result.result.height}; expected "
                            f"{expected_size[0]}x{expected_size[1]}"
                        )
                    elapsed = time.monotonic() - started
                    item = {
                        "case_id": case["id"],
                        "candidate_id": candidate,
                        "adapter_id": plan.adapter.id,
                        "engine_id": result.result.engine,
                        "path": _relative(output, run_dir),
                        "sha256": _sha256(output),
                        "bytes": output.stat().st_size,
                        "width": result.result.width,
                        "height": result.result.height,
                        "tile_size": plan.tile_size,
                        "overlap": _resolved_overlap(plan.adapter),
                        "precision": _resolved_precision(plan.adapter),
                        "elapsed_seconds": round(elapsed, 6),
                        "warnings": result.result.warnings,
                        "diagnostics": _diagnostics(output, reference_path),
                    }
                except BaseException:
                    output.unlink(missing_ok=True)
                    raise
                finally:
                    shutil.rmtree(workspace, ignore_errors=True)
                outputs.append(item)
                completed.add((case["id"], candidate))
                for fingerprint in candidate_fingerprints:
                    if fingerprint["id"] == candidate:
                        fingerprint["precision"] = item["precision"]
                run["candidates"] = candidate_fingerprints
                run["outputs"] = outputs
                run["updated_at"] = _utc_now()
                _write_json(manifest_path, run)
    except KeyboardInterrupt:
        cancel.set()
        run["updated_at"] = _utc_now()
        _write_json(manifest_path, run)
        raise
    expected = len(cases) * len(STANDARD_CANDIDATES)
    if len(outputs) != expected:
        raise BenchmarkRunError(f"run has {len(outputs)} valid outputs; expected {expected}")
    run["state"] = "complete"
    run["updated_at"] = _utc_now()
    _write_json(manifest_path, run)
    empty_work = run_dir / "work"
    if empty_work.is_dir():
        shutil.rmtree(empty_work, ignore_errors=True)
    return manifest_path


def load_run(path: Path) -> dict[str, Any]:
    manifest = path / "run.json" if path.is_dir() else path
    payload = _load_existing(manifest)
    if payload.get("state") != "complete":
        raise BenchmarkRunError("benchmark run is incomplete; resume it before review")
    run_dir = manifest.parent
    valid = _valid_completed_outputs(run_dir, payload)
    expected_keys = {
        (case["id"], candidate["id"])
        for case in payload.get("cases", [])
        for candidate in payload.get("candidates", [])
    }
    actual_keys = {(output.get("case_id"), output.get("candidate_id")) for output in valid}
    if len(valid) != len(actual_keys) or actual_keys != expected_keys:
        raise BenchmarkRunError("one or more benchmark outputs are missing or changed")
    return payload
