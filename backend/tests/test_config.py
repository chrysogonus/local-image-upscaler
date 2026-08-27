from __future__ import annotations

import pytest

from upscaler.config import load_config


def test_hardware_policy_defaults_to_safe(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("UPSCALER_WORK_ROOT", str(tmp_path))
    for name in (
        "UPSCALER_HARDWARE_POLICY",
        "UPSCALER_RAM_RESERVE_MIB",
        "UPSCALER_VRAM_RESERVE_MIB",
    ):
        monkeypatch.delenv(name, raising=False)

    config = load_config()

    assert config.hardware_policy == "safe"
    assert config.ram_reserve_mib == 4096
    assert config.vram_reserve_mib == 1536


def test_hardware_overrides_are_loaded(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("UPSCALER_WORK_ROOT", str(tmp_path))
    monkeypatch.setenv("UPSCALER_HARDWARE_POLICY", "off")
    monkeypatch.setenv("UPSCALER_RAM_MIB", "98304")
    monkeypatch.setenv("UPSCALER_VRAM_MIB", "24576")
    monkeypatch.setenv("UPSCALER_GPU_NAME", "Corrected GPU")
    monkeypatch.setenv("UPSCALER_MEMORY_KIND", "unified")

    config = load_config()

    assert config.hardware_policy == "off"
    assert config.ram_mib_override == 98304
    assert config.vram_mib_override == 24576
    assert config.gpu_name_override == "Corrected GPU"
    assert config.memory_kind_override == "unified"


def test_invalid_hardware_policy_fails_startup(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("UPSCALER_WORK_ROOT", str(tmp_path))
    monkeypatch.setenv("UPSCALER_HARDWARE_POLICY", "hopeful")

    with pytest.raises(ValueError, match="UPSCALER_HARDWARE_POLICY"):
        load_config()
