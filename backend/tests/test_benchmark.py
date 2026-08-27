from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from upscaler.benchmark import cli
from upscaler.benchmark.dataset import (
    BenchmarkDataError,
    degrade_reference,
    load_dataset,
    prepare_dataset,
)
from upscaler.benchmark.review import (
    ARTIFACT_TAGS,
    BenchmarkReviewError,
    generate_report,
    generate_review,
    output_token,
    validate_session,
)
from upscaler.benchmark.runner import CANDIDATE_NAMES, load_run, run_benchmark
from upscaler.config import load_config
from upscaler.imaging.pipeline import ProcessResult, process_image
from upscaler.models import ModelRegistry
from upscaler.schemas import JobSettings, ProcessingMode, ResultInfo, SourceInfo


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_curated_dataset_has_the_required_public_domain_tracks() -> None:
    dataset = load_dataset()

    assert len(dataset.cases) == 12
    assert sum(case.track == "paired" for case in dataset.cases) == 8
    assert sum(case.track == "authentic" for case in dataset.cases) == 4
    assert {case.degradation for case in dataset.cases if case.track == "paired"} == {
        "clean",
        "optical",
        "noise-jpeg",
        "combined",
    }
    for case in dataset.cases:
        assert case.source.url.startswith("https://images-assets.nasa.gov/")
        assert case.source.source_page.startswith("https://images.nasa.gov/details/")
        assert "public domain" in case.source.license.lower()


@pytest.mark.parametrize("recipe", ["clean", "optical", "noise-jpeg", "combined"])
def test_degradations_are_deterministic_and_exactly_four_times(recipe: str) -> None:
    y, x = np.mgrid[:64, :64]
    pixels = np.stack((x * 4, y * 4, (x + y) * 2), axis=2).astype(np.uint8)
    reference = Image.fromarray(pixels, "RGB")

    first, first_format = degrade_reference(reference, recipe, seed=123)
    second, second_format = degrade_reference(reference, recipe, seed=123)

    assert first.size == second.size == (16, 16)
    assert first_format == second_format == "PNG"
    assert np.array_equal(np.asarray(first), np.asarray(second))


def _custom_manifest(path: Path, source: Path) -> None:
    payload = {
        "schema_version": 1,
        "id": "fixture-v1",
        "name": "Fixture",
        "description": "A deterministic test fixture.",
        "license_policy": "public-domain-only",
        "rights_url": "https://example.test/rights",
        "cases": [
            {
                "id": "paired",
                "title": "Paired fixture",
                "track": "paired",
                "tags": ["clean-resize"],
                "source": {
                    "url": "https://example.test/paired.png",
                    "sha256": sha256(source),
                    "width": 64,
                    "height": 64,
                    "source_page": "https://example.test/paired",
                    "creator": "Fixture author",
                    "license": "Public domain",
                },
                "crop": {"left": 0, "top": 0, "width": 32, "height": 32},
                "degradation": "clean",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prepare_validates_and_materializes_without_network(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64), (20, 80, 140)).save(source)
    manifest = tmp_path / "dataset.json"
    _custom_manifest(manifest, source)
    case_root = tmp_path / "root" / "data" / "fixture-v1" / "cases" / "paired"
    case_root.mkdir(parents=True)
    (case_root / "source.jpg").write_bytes(source.read_bytes())

    prepared_path = prepare_dataset(tmp_path / "root", manifest_path=manifest, fetch=False)
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))

    assert prepared["dataset_id"] == "fixture-v1"
    assert prepared["cases"][0]["input_width"] == 8
    assert prepared["cases"][0]["reference_sha256"]
    with Image.open(case_root / "reference.png") as reference:
        assert reference.size == (32, 32)
    with Image.open(case_root / "input.png") as benchmark_input:
        assert benchmark_input.size == (8, 8)


def test_prepare_rejects_a_changed_download(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64), "navy").save(source)
    manifest = tmp_path / "dataset.json"
    _custom_manifest(manifest, source)
    case_root = tmp_path / "root" / "data" / "fixture-v1" / "cases" / "paired"
    case_root.mkdir(parents=True)
    Image.new("RGB", (64, 64), "red").save(case_root / "source.jpg", format="PNG")

    with pytest.raises(BenchmarkDataError, match="not downloaded or has the wrong checksum"):
        prepare_dataset(tmp_path / "root", manifest_path=manifest, fetch=False)


def test_cli_reports_the_created_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = tmp_path / "prepared.json"
    monkeypatch.setattr(cli, "prepare_dataset", lambda _root: result)

    assert cli.main(["--root", str(tmp_path), "prepare"]) == 0
    assert capsys.readouterr().out.strip() == str(result)


def test_cli_turns_validation_failures_into_an_actionable_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(_root: Path) -> Path:
        raise BenchmarkDataError("fixture checksum changed")

    monkeypatch.setattr(cli, "prepare_dataset", fail)

    assert cli.main(["--root", str(tmp_path), "prepare"]) == 2
    assert "fixture checksum changed" in capsys.readouterr().err


class FakeAdapter:
    neural = True
    generative = False
    max_passes = 1
    native_scales = (4,)
    license = "Test license"
    noncommercial = False
    restores_without_enlarging = False
    restores_faces = False
    available = True
    unavailable_reason = None
    device = "Test device"

    def __init__(self, candidate: str) -> None:
        self.id = f"fixture-{candidate}"
        self.name = f"Fixture {candidate}"


class FakeRegistry:
    def __init__(self) -> None:
        self.adapters: dict[str, FakeAdapter] = {}

    def resolve_benchmark_candidate(self, candidate: str, *_args: Any, **_kwargs: Any) -> Any:
        adapter = self.adapters.setdefault(candidate, FakeAdapter(candidate))
        return SimpleNamespace(adapter=adapter, tile_size=32)


def _prepared_fixture(tmp_path: Path, case_count: int = 2) -> Path:
    data_root = tmp_path / "data"
    cases = []
    for index in range(case_count):
        case_id = f"case-{index}"
        case_root = data_root / "cases" / case_id
        case_root.mkdir(parents=True)
        benchmark_input = case_root / "input.png"
        Image.new("RGB", (64, 64), (20 + index, 40, 60)).save(benchmark_input)
        reference = case_root / "reference.png" if index == 0 else None
        if reference:
            Image.new("RGB", (256, 256), (20, 40, 60)).save(reference)
        cases.append(
            {
                "id": case_id,
                "title": f"Case {index}",
                "track": "paired" if reference else "authentic",
                "tags": ["fixture", "paired" if reference else "authentic"],
                "degradation": "clean" if reference else None,
                "input": str(benchmark_input.relative_to(data_root)),
                "input_sha256": sha256(benchmark_input),
                "input_width": 64,
                "input_height": 64,
                "reference": str(reference.relative_to(data_root)) if reference else None,
                "reference_sha256": sha256(reference) if reference else None,
                "source_page": f"https://example.test/{case_id}",
                "creator": "Fixture author",
                "license": "Public domain",
            }
        )
    prepared = {
        "schema_version": 1,
        "dataset_id": "fixture-v1",
        "dataset_digest": "a" * 64,
        "name": "Fixture",
        "description": "Fixture",
        "license_policy": "public-domain-only",
        "rights_url": "https://example.test/rights",
        "cases": cases,
    }
    path = data_root / "prepared.json"
    path.write_text(json.dumps(prepared), encoding="utf-8")
    return path


def _fake_processor(calls: list[tuple[str, str]]) -> Any:
    colors = {
        "fixture-classical": (20, 40, 60),
        "fixture-swinir": (30, 50, 70),
        "fixture-realesrgan": (40, 60, 80),
    }

    def process(
        source_path: Path,
        output_path: Path,
        _workspace: Path,
        _original_filename: str,
        settings: Any,
        adapter: FakeAdapter,
        *_args: Any,
        **_kwargs: Any,
    ) -> ProcessResult:
        calls.append((source_path.parent.name, adapter.id))
        with Image.open(source_path) as source:
            source_width, source_height = source.size
            size = (source_width * 4, source_height * 4)
        Image.new("RGB", size, colors[adapter.id]).save(output_path)
        return ProcessResult(
            source=SourceInfo(
                filename=source_path.name,
                width=source_width,
                height=source_height,
                mode="RGB",
            ),
            result=ResultInfo(
                width=size[0],
                height=size[1],
                bytes=output_path.stat().st_size,
                engine=adapter.id,
                processing_mode=ProcessingMode.upscale,
                filename=output_path.name,
                resolved_tile_size=settings.tile_size,
            ),
            output_path=output_path,
        )

    return process


def test_runner_checkpoints_real_files_and_resumes_valid_outputs(tmp_path: Path) -> None:
    prepared = _prepared_fixture(tmp_path)
    calls: list[tuple[str, str]] = []
    run_dir = tmp_path / "run"

    manifest = run_benchmark(
        tmp_path,
        prepared_path=prepared,
        run_dir=run_dir,
        registry=FakeRegistry(),  # type: ignore[arg-type]
        processor=_fake_processor(calls),
    )
    run = load_run(manifest)

    assert run["state"] == "complete"
    assert len(run["outputs"]) == 6
    assert len(calls) == 6
    paired = [output for output in run["outputs"] if output["case_id"] == "case-0"]
    authentic = [output for output in run["outputs"] if output["case_id"] == "case-1"]
    assert all(output["diagnostics"] for output in paired)
    assert all(output["diagnostics"] is None for output in authentic)

    run_benchmark(
        tmp_path,
        prepared_path=prepared,
        run_dir=run_dir,
        registry=FakeRegistry(),  # type: ignore[arg-type]
        processor=_fake_processor(calls),
    )
    assert len(calls) == 6


def test_runner_removes_partial_output_and_workspace_after_failure(tmp_path: Path) -> None:
    prepared = _prepared_fixture(tmp_path, case_count=1)
    run_dir = tmp_path / "run"
    base = _fake_processor([])

    def fail_after_output(*args: Any, **kwargs: Any) -> ProcessResult:
        base(*args, **kwargs)
        raise RuntimeError("fixture inference failed")

    with pytest.raises(RuntimeError, match="fixture inference failed"):
        run_benchmark(
            tmp_path,
            prepared_path=prepared,
            run_dir=run_dir,
            registry=FakeRegistry(),  # type: ignore[arg-type]
            processor=fail_after_output,
        )

    assert not list((run_dir / "outputs").rglob("*.png"))
    assert not list((run_dir / "work").rglob("*.*"))
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["state"] == ("incomplete")


@pytest.mark.parametrize("candidate", ["swinir", "realesrgan"])
def test_installed_neural_benchmark_candidate_runs_the_production_pipeline(
    candidate: str, tmp_path: Path
) -> None:
    settings = JobSettings(target_edge=256, sharpen=0, max_neural_passes=1)
    registry = ModelRegistry(load_config())
    try:
        plan = registry.resolve_benchmark_candidate(candidate, settings, width=64, height=64)
    except ValueError as exc:
        pytest.skip(str(exc))
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    Image.new("RGB", (64, 64), (32, 96, 160)).save(source)
    resolved = settings.model_copy(update={"tile_size": plan.tile_size})

    result = process_image(
        source,
        output,
        tmp_path,
        source.name,
        resolved,
        plan.adapter,
        1_000_000,
        Event(),
        lambda _phase, _message, _progress: None,
        resolved_tile_size=plan.tile_size,
    )

    assert output.is_file()
    assert result.result.engine != "classical"
    assert result.result.width == result.result.height == 256


def _complete_session(run: dict[str, Any], session_id: str) -> dict[str, Any]:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for output in run["outputs"]:
        by_case.setdefault(output["case_id"], {})[output["candidate_id"]] = output
    judgments = []
    for case in run["cases"]:
        for left_id, right_id in itertools.combinations(CANDIDATE_NAMES, 2):
            left = by_case[case["id"]][left_id]
            right = by_case[case["id"]][right_id]
            judgments.append(
                {
                    "case_id": case["id"],
                    "left_output_hash": left["sha256"],
                    "right_output_hash": right["sha256"],
                    "left_output_token": output_token(run["run_id"], left),
                    "right_output_token": output_token(run["run_id"], right),
                    "choice": "tie",
                    "artifacts": {"left": [ARTIFACT_TAGS[0]], "right": []},
                }
            )
    return {
        "schema_version": 1,
        "run_id": run["run_id"],
        "session_id": session_id,
        "created_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T01:00:00+00:00",
        "judgments": judgments,
    }


def test_review_is_blinded_and_complete_sessions_merge_into_a_report(tmp_path: Path) -> None:
    prepared = _prepared_fixture(tmp_path)
    run_dir = tmp_path / "run"
    manifest = run_benchmark(
        tmp_path,
        prepared_path=prepared,
        run_dir=run_dir,
        registry=FakeRegistry(),  # type: ignore[arg-type]
        processor=_fake_processor([]),
    )
    run = load_run(manifest)
    review_path = generate_review(run_dir, session_id="11111111-1111-4111-8111-111111111111")
    review_html = review_path.read_text(encoding="utf-8")

    assert "Lanczos baseline" not in review_html
    assert "SwinIR-L" not in review_html
    assert len(list((run_dir / "review-assets").rglob("*.png"))) >= 6
    assert "prefers-reduced-motion" in review_html
    assert 'aria-live="polite"' in review_html
    payload_text = review_html.split('<script id="review-data" type="application/json">', 1)[
        1
    ].split("</script>", 1)[0]
    review_payload = json.loads(payload_text)
    token_candidates = {
        output_token(run["run_id"], output): output["candidate_id"] for output in run["outputs"]
    }
    placements = {candidate: {"a": 0, "b": 0} for candidate in CANDIDATE_NAMES}
    for comparison in review_payload["comparisons"]:
        placements[token_candidates[comparison["a"]["token"]]]["a"] += 1
        placements[token_candidates[comparison["b"]["token"]]]["b"] += 1
    assert all(counts["a"] == counts["b"] for counts in placements.values())

    session = _complete_session(run, "22222222-2222-4222-8222-222222222222")
    assert len(validate_session(run, session)) == 6
    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps(session), encoding="utf-8")
    report = generate_report(run_dir, [session_path])
    report_html = report.read_text(encoding="utf-8")
    aggregate = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))

    assert "Real outputs" in report_html
    assert "50.0%" in report_html
    assert aggregate["review_sessions"] == 1
    assert aggregate["judgments"] == 6


def test_review_rejects_tampered_or_incomplete_sessions(tmp_path: Path) -> None:
    prepared = _prepared_fixture(tmp_path, case_count=1)
    run_dir = tmp_path / "run"
    manifest = run_benchmark(
        tmp_path,
        prepared_path=prepared,
        run_dir=run_dir,
        registry=FakeRegistry(),  # type: ignore[arg-type]
        processor=_fake_processor([]),
    )
    run = load_run(manifest)
    session = _complete_session(run, "33333333-3333-4333-8333-333333333333")
    session["judgments"].pop()
    with pytest.raises(BenchmarkReviewError, match="incomplete"):
        validate_session(run, session)

    session = _complete_session(run, "44444444-4444-4444-8444-444444444444")
    session["judgments"][0]["left_output_token"] = "0" * 64
    with pytest.raises(BenchmarkReviewError, match="unknown"):
        validate_session(run, session)
