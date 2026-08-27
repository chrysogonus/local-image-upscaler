from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from upscaler.config import AppConfig
from upscaler.hardware import HardwareSnapshot
from upscaler.models.comfyui import load_catalog
from upscaler.models.registry import ModelRegistry
from upscaler.resource_policy import (
    ENGINE_PROFILES,
    WORKFLOW_PROFILES,
    HardwarePolicyError,
    admission_reason,
    capacity_reason,
    resolve_tile,
    safe_tile_sizes,
)
from upscaler.schemas import JobSettings, ProcessingMode


def _catalog():
    """The checked-in graphs, as the ComfyUI adapter would report them."""
    return list(load_catalog())


def hardware(ram_gib: int, vram_gib: int | None, *, kind: str = "dedicated"):
    ram = ram_gib * 1024
    vram = vram_gib * 1024 if vram_gib is not None else None
    return HardwareSnapshot(
        scope="backend",
        ram_physical_mib=ram,
        ram_effective_mib=ram,
        ram_available_mib=ram,
        gpu_name="Test GPU" if vram_gib is not None else None,
        vram_total_mib=vram,
        vram_available_mib=vram,
        memory_kind=kind,
        source="test",
    )


@pytest.mark.parametrize(
    ("snapshot", "profile_ids", "excluded"),
    [
        (hardware(8, None), ["classical"], ["realesrgan", "spandrel-sr", "realesrgan-cuda"]),
        (
            hardware(64, 16),
            [
                "classical",
                "realesrgan",
                "spandrel-sr",
                "realesrgan-cuda",
                "comfyui-illustration",
            ],
            [],
        ),
        (hardware(128, 128, kind="unified"), list(ENGINE_PROFILES), []),
    ],
)
def test_acceptance_matrix_for_native_engines(snapshot, profile_ids, excluded) -> None:
    for profile_id in profile_ids:
        assert capacity_reason(ENGINE_PROFILES[profile_id], snapshot) is None
    for profile_id in excluded:
        assert capacity_reason(ENGINE_PROFILES[profile_id], snapshot)


def test_workflow_acceptance_matrix() -> None:
    """A graph is offered on hardware that can run it and hidden where it cannot."""
    for profile in WORKFLOW_PROFILES.values():
        assert capacity_reason(profile, hardware(64, 16)) is None
        assert capacity_reason(profile, hardware(128, 128, kind="unified")) is None
        # Undetectable GPU memory fails closed rather than offering the graph.
        assert capacity_reason(profile, hardware(8, None))


def test_unknown_vram_fails_closed_with_override_hint() -> None:
    reason = capacity_reason(ENGINE_PROFILES["spandrel-sr"], hardware(64, None))
    assert reason is not None
    assert "UPSCALER_VRAM_MIB" in reason


def test_automatic_and_explicit_tiles_follow_capacity(tmp_path: Path) -> None:
    config = AppConfig(work_root=tmp_path)
    profile = ENGINE_PROFILES["spandrel-sr"]

    assert safe_tile_sizes(profile, hardware(64, 16), config) == (0, 128, 256, 512, 768)
    assert safe_tile_sizes(profile, hardware(64, 8), config) == (0, 128, 256)
    assert resolve_tile(profile, 0, hardware(64, 8), config) == 256
    with pytest.raises(HardwarePolicyError, match="512px"):
        resolve_tile(profile, 512, hardware(64, 8), config)


def test_live_admission_uses_reserves(tmp_path: Path) -> None:
    config = AppConfig(work_root=tmp_path)
    snapshot = hardware(64, 16)
    busy = replace(
        snapshot,
        ram_available_mib=63 * 1024,
        vram_available_mib=8 * 1024,
    )
    reason = admission_reason(
        ENGINE_PROFILES["spandrel-sr"], busy, working_mib=512, resolved_tile=512, config=config
    )
    assert reason is not None and "free VRAM" in reason


def test_unified_admission_uses_one_shared_available_pool(tmp_path: Path) -> None:
    config = AppConfig(work_root=tmp_path)
    snapshot = replace(
        hardware(128, 128, kind="unified"),
        ram_available_mib=5 * 1024,
        vram_available_mib=5 * 1024,
    )

    reason = admission_reason(
        WORKFLOW_PROFILES["illustration-upscale"],
        snapshot,
        working_mib=1024,
        resolved_tile=0,
        config=config,
    )

    assert reason is not None and "shared memory pool" in reason


class StubAdapter:
    max_passes = 1
    native_scales = (1,)
    license = "test"
    neural = True
    generative = False

    def __init__(
        self,
        adapter_id: str,
        workflows=(),
        *,
        stats=None,
        reclaimed_stats=None,
        reclaim_result: bool = False,
    ) -> None:
        self.id = adapter_id
        self.name = adapter_id
        self.available = True
        self.unavailable_reason = None
        self.device = "stub"
        self._workflows = tuple(workflows)
        self._stats = stats
        self._reclaimed_stats = reclaimed_stats
        self._reclaim_result = reclaim_result
        self.reclaim_calls = 0

    def workflows(self):
        return self._workflows

    def hardware_stats(self, *, refresh: bool = False):
        if self.reclaim_calls and self._reclaimed_stats is not None:
            return self._reclaimed_stats
        return self._stats

    def reclaim_memory_if_idle(self) -> bool:
        self.reclaim_calls += 1
        return self._reclaim_result


def test_a_mode_whose_only_workflow_does_not_fit_is_unavailable(tmp_path: Path) -> None:
    """Illustration has no second engine, so it reports why instead of degrading."""
    registry = ModelRegistry(AppConfig(work_root=tmp_path))
    small = {
        "system": {"ram_total": 64 * 1024**3, "ram_free": 48 * 1024**3},
        "devices": [{"name": "Test GPU", "vram_total": 2 * 1024**3, "vram_free": 2 * 1024**3}],
    }
    registry._adapters = {
        "comfyui-illustration": StubAdapter("comfyui-illustration", _catalog(), stats=small)
    }

    with pytest.raises(ValueError, match="GiB VRAM"):
        registry.adapter_for(ProcessingMode.illustration)


def test_registry_returns_only_capacity_checked_comfy_workflows(tmp_path: Path) -> None:
    registry = ModelRegistry(AppConfig(work_root=tmp_path))
    stats = {
        "system": {"ram_total": 64 * 1024**3, "ram_free": 48 * 1024**3},
        "devices": [
            {
                "name": "Test GPU",
                "vram_total": 16 * 1024**3,
                "vram_free": 14 * 1024**3,
            }
        ],
    }
    registry._adapters = {
        "comfyui-illustration": StubAdapter("comfyui-illustration", _catalog(), stats=stats)
    }

    assert {workflow.id for workflow in registry.workflows()} == {"illustration-upscale"}


def test_policy_off_preserves_runtime_only_resolution(tmp_path: Path) -> None:
    registry = ModelRegistry(AppConfig(work_root=tmp_path, hardware_policy="off"))
    registry._adapters = {
        "spandrel-sr": StubAdapter("spandrel-sr"),
        "realesrgan-cuda": StubAdapter("realesrgan-cuda"),
    }
    registry.hardware._stable = hardware(2, None)

    assert registry.adapter_for(ProcessingMode.upscale).id == "spandrel-sr"


def test_cpu_fallback_discloses_hardware_exclusion(tmp_path: Path) -> None:
    registry = ModelRegistry(AppConfig(work_root=tmp_path))
    classical = registry._adapters["classical"]
    registry._adapters = {
        "spandrel-sr": StubAdapter("spandrel-sr"),
        "classical": classical,
    }
    registry.hardware._stable = hardware(64, 4)

    capability = registry._capability(ProcessingMode.upscale)

    assert capability.engine == classical.name
    assert capability.fallback_reason is not None
    assert "reserve" in capability.fallback_reason


def test_every_checked_in_engine_and_workflow_has_a_profile(tmp_path: Path) -> None:
    registry_ids = set(ModelRegistry(AppConfig(work_root=tmp_path))._adapters)
    assert registry_ids == set(ENGINE_PROFILES)
    assert {workflow.id for workflow in load_catalog()} == set(WORKFLOW_PROFILES)


def test_manual_unsafe_workflow_is_refused(tmp_path: Path) -> None:
    """Naming a graph by hand does not bypass the capacity check the picker applies."""
    registry = ModelRegistry(AppConfig(work_root=tmp_path))
    registry._adapters = {"comfyui-illustration": StubAdapter("comfyui-illustration", _catalog())}
    registry.hardware._stable = hardware(8, None)

    with pytest.raises(HardwarePolicyError, match="excluded"):
        registry.resolve_job(
            JobSettings(
                processing_mode=ProcessingMode.illustration, workflow="illustration-upscale"
            )
        )


def test_a_visible_workflow_is_rechecked_against_free_vram(tmp_path: Path) -> None:
    """Fitting the GPU is not the same as fitting it while something else runs."""
    registry = ModelRegistry(AppConfig(work_root=tmp_path))
    stats = {
        "system": {"ram_total": 80 * 1024**3, "ram_free": 70 * 1024**3},
        "devices": [
            {
                "name": "Test GPU",
                "vram_total": 16 * 1024**3,
                "vram_free": 4 * 1024**3,
            }
        ],
    }
    registry._adapters = {
        "comfyui-illustration": StubAdapter("comfyui-illustration", _catalog(), stats=stats)
    }
    settings = JobSettings(
        processing_mode=ProcessingMode.illustration,
        workflow="illustration-upscale",
    )

    assert registry.resolve_job(settings).workflow_id == "illustration-upscale"
    with pytest.raises(HardwarePolicyError, match="free VRAM"):
        registry.resolve_job(settings, width=32, height=16)


def test_comfy_admission_retries_after_idle_cache_release(tmp_path: Path) -> None:
    registry = ModelRegistry(AppConfig(work_root=tmp_path))
    initial = {
        "system": {"ram_total": 128 * 1024**3, "ram_free": 5 * 1024**3},
        "devices": [
            {
                "name": "NVIDIA GB10",
                "vram_total": 128 * 1024**3,
                "vram_free": 5 * 1024**3,
            }
        ],
    }
    reclaimed = {
        **initial,
        "system": {"ram_total": 128 * 1024**3, "ram_free": 96 * 1024**3},
        "devices": [{**initial["devices"][0], "vram_free": 96 * 1024**3}],
    }
    adapter = StubAdapter(
        "comfyui-illustration",
        _catalog(),
        stats=initial,
        reclaimed_stats=reclaimed,
        reclaim_result=True,
    )
    registry._adapters = {"comfyui-illustration": adapter}
    settings = JobSettings(
        processing_mode=ProcessingMode.illustration,
        workflow="illustration-upscale",
    )

    plan = registry.resolve_job(settings, width=32, height=16)

    assert plan.workflow_id == "illustration-upscale"
    assert adapter.reclaim_calls == 1
