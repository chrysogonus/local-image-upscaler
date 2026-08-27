from __future__ import annotations

import sys
from types import SimpleNamespace

from upscaler.config import AppConfig
from upscaler.hardware import (
    MIB,
    HardwareService,
    cgroup_memory,
    classify_memory,
    hardware_from_comfy_stats,
    parse_meminfo,
    parse_nvidia_smi,
    parse_vulkan_name,
)


def test_meminfo_and_cgroup_limit_choose_effective_ram() -> None:
    physical, available = parse_meminfo(
        "MemTotal:       131072000 kB\nMemAvailable:    65536000 kB\n"
    )
    limit, remaining = cgroup_memory(32 * 1024 * MIB, 7 * 1024 * MIB, physical)

    assert physical == 125 * 1024 * MIB
    assert available == 64000 * MIB
    assert limit == 32 * 1024 * MIB
    assert remaining == 25 * 1024 * MIB


def test_cgroup_v1_unlimited_sentinel_is_ignored() -> None:
    assert cgroup_memory(1 << 62, 100, 16 * 1024 * MIB) == (None, None)


def test_nvidia_and_vulkan_parsers() -> None:
    assert parse_nvidia_smi("NVIDIA GeForce RTX 5070 Ti, 16303, 13180\n") == (
        "NVIDIA GeForce RTX 5070 Ti",
        16303,
        13180,
    )
    assert parse_vulkan_name("GPU0:\n\tdeviceName = NVIDIA GeForce RTX 5070 Ti\n") == (
        "NVIDIA GeForce RTX 5070 Ti"
    )


def test_torch_cuda_probe_reads_total_and_free_memory(monkeypatch, tmp_path) -> None:
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda _index: SimpleNamespace(name="Test CUDA"),
        mem_get_info=lambda _index: (12 * 1024 * MIB, 16 * 1024 * MIB),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    service = object.__new__(HardwareService)
    service.config = AppConfig(work_root=tmp_path)

    assert service._torch_gpu() == ("Test CUDA", 16384, 12288)


def test_comfyui_stats_and_unified_classification() -> None:
    stats = {
        "system": {"ram_total": 128 * 1024 * MIB, "ram_free": 96 * 1024 * MIB},
        "devices": [
            {
                "name": "NVIDIA GB10",
                "vram_total": 128 * 1024 * MIB,
                "vram_free": 96 * 1024 * MIB,
            }
        ],
    }
    report = hardware_from_comfy_stats(stats)

    assert report.memory_kind == "unified"
    assert report.ram_effective_mib == 128 * 1024
    assert report.vram_available_mib == 96 * 1024
    assert classify_memory("ordinary discrete GPU", 16384, 65536) == "dedicated"


def test_detection_overrides_are_authoritative(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(HardwareService, "_ram", lambda _self: (64 * 1024 * MIB,) * 3)
    monkeypatch.setattr(
        HardwareService,
        "_torch_gpu",
        lambda _self: ("Detected GPU", 8192, 8192),
    )
    service = HardwareService(
        AppConfig(
            work_root=tmp_path,
            ram_mib_override=96 * 1024,
            vram_mib_override=24 * 1024,
            gpu_name_override="Corrected GPU",
            memory_kind_override="unified",
        )
    )

    assert service.stable.ram_effective_mib == 96 * 1024
    assert service.stable.vram_total_mib == 24 * 1024
    assert service.stable.gpu_name == "Corrected GPU"
    assert service.stable.memory_kind == "unified"
