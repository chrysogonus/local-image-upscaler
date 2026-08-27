from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in choices:
        expected = "|".join(sorted(choices))
        raise ValueError(f"{name} must be one of {expected}")
    return value


@dataclass(frozen=True, slots=True)
class AppConfig:
    work_root: Path
    max_upload_bytes: int = 100 * 1024 * 1024
    max_input_pixels: int = 120_000_000
    max_jobs: int = 1
    job_retention_seconds: int = 60 * 60
    realesrgan_binary: Path | None = None
    hardware_policy: Literal["safe", "off"] = "safe"
    ram_reserve_mib: int = 4096
    vram_reserve_mib: int = 1536
    ram_mib_override: int | None = None
    vram_mib_override: int | None = None
    gpu_name_override: str | None = None
    memory_kind_override: Literal["dedicated", "unified"] | None = None


def load_config() -> AppConfig:
    configured_root = os.getenv("UPSCALER_WORK_ROOT")
    work_root = (
        Path(configured_root).expanduser().resolve()
        if configured_root
        else Path(tempfile.gettempdir()) / "local-image-upscaler"
    )
    binary = os.getenv("UPSCALER_REALESRGAN_BIN")
    memory_kind = os.getenv("UPSCALER_MEMORY_KIND")
    if memory_kind:
        memory_kind = _env_choice("UPSCALER_MEMORY_KIND", "dedicated", {"dedicated", "unified"})
    return AppConfig(
        work_root=work_root,
        max_upload_bytes=_env_int("UPSCALER_MAX_UPLOAD_BYTES", 100 * 1024 * 1024),
        max_input_pixels=_env_int("UPSCALER_MAX_INPUT_PIXELS", 120_000_000),
        max_jobs=max(1, _env_int("UPSCALER_MAX_JOBS", 1)),
        job_retention_seconds=max(60, _env_int("UPSCALER_JOB_RETENTION_SECONDS", 3600)),
        realesrgan_binary=Path(binary).expanduser().resolve() if binary else None,
        hardware_policy=_env_choice(  # type: ignore[arg-type]
            "UPSCALER_HARDWARE_POLICY", "safe", {"safe", "off"}
        ),
        ram_reserve_mib=max(0, _env_int("UPSCALER_RAM_RESERVE_MIB", 4096)),
        vram_reserve_mib=max(0, _env_int("UPSCALER_VRAM_RESERVE_MIB", 1536)),
        ram_mib_override=_env_optional_int("UPSCALER_RAM_MIB"),
        vram_mib_override=_env_optional_int("UPSCALER_VRAM_MIB"),
        gpu_name_override=os.getenv("UPSCALER_GPU_NAME") or None,
        memory_kind_override=memory_kind,  # type: ignore[arg-type]
    )
