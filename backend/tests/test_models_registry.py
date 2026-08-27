from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest

from upscaler.config import AppConfig
from upscaler.models.base import ModelExecutionError, ModelRequest
from upscaler.models.realesrgan_cuda import (
    REQUIRED_WEIGHTS,
    WEIGHTS_FILENAME,
    RealEsrganCudaAdapter,
    weights_dir,
)
from upscaler.models.registry import (
    MODE_DESCRIPTIONS,
    MODE_ENGINES,
    MODE_NAMES,
    ModelRegistry,
)
from upscaler.schemas import GENERATIVE_MODES, JobSettings, ProcessingMode


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    # These tests isolate runtime/model ordering. Hardware-policy behavior has
    # its own fixtures below, so disable it here rather than coupling ordering
    # assertions to the GPU of the test runner.
    return AppConfig(work_root=tmp_path, hardware_policy="off")


class StubAdapter:
    max_passes = 3
    native_scales = (4,)
    supports_tta = True
    license = "MIT"

    def __init__(
        self,
        adapter_id: str,
        available: bool,
        generative: bool = False,
        neural: bool = True,
    ) -> None:
        self.id = adapter_id
        self.name = adapter_id
        self.generative = generative
        self.neural = neural
        self._available = available

    @property
    def available(self) -> bool:
        return self._available

    @property
    def unavailable_reason(self) -> str | None:
        return None if self._available else f"{self.id} is off"

    @property
    def device(self) -> str:
        return "stub"


def _registry_with(config: AppConfig, **availability: bool) -> ModelRegistry:
    registry = ModelRegistry(config)
    registry._adapters = {
        adapter_id: StubAdapter(adapter_id, available)
        for adapter_id, available in availability.items()
    }
    return registry


def test_upscale_prefers_cuda_then_ncnn_then_the_resampler(config: AppConfig) -> None:
    registry = _registry_with(
        config, **{"realesrgan-cuda": True, "realesrgan": True, "classical": True}
    )
    assert registry.adapter_for(ProcessingMode.upscale).id == "realesrgan-cuda"

    registry = _registry_with(
        config, **{"realesrgan-cuda": False, "realesrgan": True, "classical": True}
    )
    assert registry.adapter_for(ProcessingMode.upscale).id == "realesrgan"

    registry = _registry_with(
        config, **{"realesrgan-cuda": False, "realesrgan": False, "classical": True}
    )
    assert registry.adapter_for(ProcessingMode.upscale).id == "classical"


def test_benchmark_treats_both_realesrgan_runtimes_as_one_candidate(
    config: AppConfig,
) -> None:
    settings = JobSettings(target_edge=2048, sharpen=0, max_neural_passes=1)
    registry = _registry_with(
        config,
        **{
            "classical": True,
            "spandrel-sr": True,
            "realesrgan-cuda": True,
            "realesrgan": True,
        },
    )
    resolved = registry.resolve_benchmark_candidate("realesrgan", settings, width=512, height=512)
    assert resolved.adapter.id == "realesrgan-cuda"

    registry._adapters["realesrgan-cuda"] = StubAdapter("realesrgan-cuda", available=False)
    resolved = registry.resolve_benchmark_candidate("realesrgan", settings, width=512, height=512)
    assert resolved.adapter.id == "realesrgan"


def test_the_benchmark_accepts_only_its_own_named_candidates(config: AppConfig) -> None:
    """The benchmark compares faithful engines; it is not a back door engine picker."""
    registry = ModelRegistry(config)
    with pytest.raises(ValueError, match="Unknown benchmark candidate"):
        registry.resolve_benchmark_candidate(
            "comfyui-illustration", JobSettings(), width=512, height=512
        )


def test_every_mode_is_named_described_and_wired_to_real_engines(config: AppConfig) -> None:
    registry = ModelRegistry(config)
    assert set(MODE_ENGINES) == set(ProcessingMode)
    assert set(MODE_NAMES) == set(ProcessingMode)
    assert set(MODE_DESCRIPTIONS) == set(ProcessingMode)
    for engine_ids in MODE_ENGINES.values():
        assert engine_ids
        assert set(engine_ids) <= set(registry._adapters)


def test_no_engine_in_this_repository_claims_to_be_generative(config: AppConfig) -> None:
    """Every mode here reconstructs, and this is the assertion that keeps it so.

    The `generative` flag is not dead weight because it is empty: an engine
    added here that invented detail would fail this test rather than reach a
    mode that promises the opposite. See ACCEPTABLE_USE.md.
    """
    registry = ModelRegistry(config)
    generative = {
        adapter_id for adapter_id, adapter in registry._adapters.items() if adapter.generative
    }
    assert generative == set()
    assert frozenset() == GENERATIVE_MODES
    for mode, engine_ids in MODE_ENGINES.items():
        assert generative.isdisjoint(engine_ids), f"{mode.value} can reach a generative engine"


def test_upscale_always_has_a_working_engine(config: AppConfig) -> None:
    """The default mode may never be unusable, whatever is installed."""
    capability = next(
        entry
        for entry in ModelRegistry(config).capabilities()
        if entry.mode == ProcessingMode.upscale
    )
    assert capability.available is True


def test_capabilities_cover_every_mode_without_raising(config: AppConfig) -> None:
    capabilities = ModelRegistry(config).capabilities()
    assert [entry.mode for entry in capabilities] == list(ProcessingMode)
    for entry in capabilities:
        assert entry.description
        assert entry.available or entry.unavailable_reason
        assert entry.generative == (entry.mode in GENERATIVE_MODES)
        assert entry.max_passes >= 1


def _upscale_capability(registry: ModelRegistry):
    return registry._capability(ProcessingMode.upscale)


def test_falling_back_to_the_resampler_says_why(config: AppConfig) -> None:
    """A plain enlargement must never be presented as completed neural upscaling."""
    registry = ModelRegistry(config)
    registry._adapters = {
        "realesrgan-cuda": StubAdapter("realesrgan-cuda", available=False),
        "realesrgan": StubAdapter("realesrgan", available=False),
        "classical": StubAdapter("classical", available=True, neural=False),
    }
    capability = _upscale_capability(registry)

    assert capability.available is True
    assert capability.fallback_reason == "realesrgan-cuda is off"


def test_a_working_neural_engine_does_not_nag_about_a_better_one(config: AppConfig) -> None:
    registry = ModelRegistry(config)
    registry._adapters = {
        "realesrgan-cuda": StubAdapter("realesrgan-cuda", available=False),
        "realesrgan": StubAdapter("realesrgan", available=True),
        "classical": StubAdapter("classical", available=True, neural=False),
    }

    assert _upscale_capability(registry).fallback_reason is None


def test_sharpen_never_reports_a_fallback(config: AppConfig) -> None:
    """The resampler is that mode's intended engine, not a degradation."""
    capability = next(
        entry
        for entry in ModelRegistry(config).capabilities()
        if entry.mode == ProcessingMode.sharpen_only
    )
    assert capability.fallback_reason is None


def test_an_unavailable_mode_is_refused_with_an_actionable_reason(config: AppConfig) -> None:
    registry = _registry_with(config, **{"comfyui-illustration": False})
    with pytest.raises(ValueError, match="comfyui-illustration is off"):
        registry.adapter_for(ProcessingMode.illustration)


def test_the_ncnn_runtime_is_restricted_to_its_native_scale(config: AppConfig) -> None:
    """-s 2 and -s 3 against a 4x model return a correctly sized, wrongly cropped image."""
    adapter = ModelRegistry(config)._adapters["realesrgan"]
    assert adapter.native_scales == (4,)


def test_weights_dir_honours_the_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UPSCALER_REALESRGAN_WEIGHTS_DIR", str(tmp_path))
    assert weights_dir() == tmp_path.resolve()


def test_cuda_adapter_reports_a_reason_instead_of_raising_on_import(
    monkeypatch, tmp_path: Path
) -> None:
    """The module must be importable and honest on a machine with no torch."""
    monkeypatch.setenv("UPSCALER_REALESRGAN_WEIGHTS_DIR", str(tmp_path))
    adapter = RealEsrganCudaAdapter()
    monkeypatch.setattr(adapter, "_probe", (False, "Unavailable", "PyTorch is not installed."))

    assert adapter.available is False
    assert adapter.device == "Unavailable"
    assert adapter.unavailable_reason == "PyTorch is not installed."


def test_cuda_adapter_distinguishes_missing_weights_from_missing_cuda(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("UPSCALER_REALESRGAN_WEIGHTS_DIR", str(tmp_path))
    adapter = RealEsrganCudaAdapter()
    monkeypatch.setattr(adapter, "_probe", (True, "CUDA (Test GPU, sm_121)", None))

    assert adapter.available is False
    reason = adapter.unavailable_reason
    assert reason is not None
    # The message must point at the weights, not at CUDA, or the user chases the wrong problem.
    assert WEIGHTS_FILENAME in reason
    assert "missing" in reason.lower()
    assert adapter.device == "CUDA (Test GPU, sm_121)"

    for name in REQUIRED_WEIGHTS:
        (tmp_path / name).write_bytes(b"stub")
    assert adapter.available is True
    assert adapter.unavailable_reason is None


def test_cuda_enhance_refuses_when_unavailable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UPSCALER_REALESRGAN_WEIGHTS_DIR", str(tmp_path))
    adapter = RealEsrganCudaAdapter()
    monkeypatch.setattr(adapter, "_probe", (False, "Unavailable", "no CUDA here"))

    request = ModelRequest(
        source_path=tmp_path / "in.png",
        output_path=tmp_path / "out.png",
        workspace=tmp_path,
        native_scale=4,
        target_width=1024,
        target_height=1024,
        tile_size=0,
    )
    with pytest.raises(ModelExecutionError, match="no CUDA here"):
        adapter.enhance(request, Event(), lambda *_: None)
