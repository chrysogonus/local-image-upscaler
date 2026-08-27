from __future__ import annotations

import math
from dataclasses import dataclass

from upscaler.config import AppConfig
from upscaler.geometry import estimate_working_bytes, plan_native_scales, target_dimensions
from upscaler.hardware import HardwareSnapshot
from upscaler.schemas import JobSettings, ProcessingMode

MIB = 1024 * 1024
TILE_CHOICES = (0, 128, 256, 512, 768)
POLICY_VERSION = 1


class HardwarePolicyError(ValueError):
    """A valid feature or setting is unsafe under the configured hardware policy."""


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    id: str
    ram_mib: int
    vram_mib: int
    unified_mib: int
    tile_vram_mib: tuple[tuple[int, int], ...] = ()
    default_tile: int = 0
    app_tiles: bool = True

    def tile_peak(self, tile: int) -> int:
        peaks = dict(self.tile_vram_mib)
        return peaks.get(tile, self.vram_mib)


ENGINE_PROFILES: dict[str, ResourceProfile] = {
    "classical": ResourceProfile("cpu-resample-sharpen", 2048, 0, 2048, app_tiles=False),
    "realesrgan": ResourceProfile(
        "ncnn-realesrgan",
        4096,
        2048,
        6144,
        ((128, 2048), (256, 2048), (512, 4096), (768, 6144)),
        256,
    ),
    "spandrel-sr": ResourceProfile(
        "swinir-l",
        8192,
        4096,
        10240,
        ((128, 4096), (256, 4096), (512, 8192), (768, 12288)),
        256,
    ),
    "realesrgan-cuda": ResourceProfile(
        "cuda-realesrgan",
        8192,
        4096,
        10240,
        ((128, 4096), (256, 4096), (512, 6144), (768, 10240)),
        512,
    ),
    "comfyui-illustration": ResourceProfile("illustration", 8192, 4096, 10240, app_tiles=False),
}

WORKFLOW_PROFILES: dict[str, ResourceProfile] = {
    "illustration-upscale": ENGINE_PROFILES["comfyui-illustration"],
}


def capacity_reason(profile: ResourceProfile, hardware: HardwareSnapshot) -> str | None:
    if hardware.memory_kind == "unified":
        capacity = hardware.ram_effective_mib or hardware.vram_total_mib
        if capacity is None:
            return "Unified-memory capacity could not be detected. Set UPSCALER_RAM_MIB."
        if capacity < profile.unified_mib:
            return (
                f"Requires {profile.unified_mib / 1024:g} GiB unified memory; "
                f"{capacity / 1024:.1f} GiB is available to this deployment."
            )
        return None

    if hardware.ram_effective_mib is None:
        return "RAM capacity could not be detected. Set UPSCALER_RAM_MIB."
    if hardware.ram_effective_mib < profile.ram_mib:
        return (
            f"Requires {profile.ram_mib / 1024:g} GiB RAM; this deployment can use "
            f"{hardware.ram_effective_mib / 1024:.1f} GiB."
        )
    if profile.vram_mib:
        if hardware.vram_total_mib is None:
            return (
                "GPU memory capacity is unknown. Safe mode keeps this feature hidden; set "
                "UPSCALER_VRAM_MIB if detection is wrong."
            )
        if hardware.vram_total_mib < profile.vram_mib:
            return (
                f"Requires {profile.vram_mib / 1024:g} GiB VRAM; the detected GPU has "
                f"{hardware.vram_total_mib / 1024:.1f} GiB."
            )
    return None


def safe_tile_sizes(
    profile: ResourceProfile,
    hardware: HardwareSnapshot,
    config: AppConfig,
) -> tuple[int, ...]:
    if not profile.app_tiles:
        return ()
    if config.hardware_policy == "off":
        return tuple(
            tile for tile in TILE_CHOICES if tile == 0 or tile in dict(profile.tile_vram_mib)
        )
    if capacity_reason(profile, hardware):
        return ()
    if hardware.memory_kind == "unified":
        capacity = hardware.ram_effective_mib or hardware.vram_total_mib or 0
        explicit = [
            tile
            for tile, peak in profile.tile_vram_mib
            if max(profile.unified_mib, peak + config.ram_reserve_mib) <= capacity
        ]
    else:
        capacity = hardware.vram_total_mib or 0
        explicit = [
            tile
            for tile, peak in profile.tile_vram_mib
            if peak + config.vram_reserve_mib <= capacity
        ]
    return (0, *explicit) if explicit else ()


def resolve_tile(
    profile: ResourceProfile,
    requested: int,
    hardware: HardwareSnapshot,
    config: AppConfig,
) -> int:
    if config.hardware_policy == "off":
        if requested:
            return requested
        return profile.default_tile
    if not profile.app_tiles:
        if requested:
            raise HardwarePolicyError(
                "This workflow controls tiling internally and does not accept an app tile size."
            )
        return 0
    choices = safe_tile_sizes(profile, hardware, config)
    if requested:
        if requested not in choices:
            shown = ", ".join(str(tile) for tile in choices if tile) or "none"
            raise HardwarePolicyError(
                f"A {requested}px tile is not safe for this engine on the detected hardware. "
                f"Safe explicit sizes: {shown}."
            )
        return requested
    explicit = [tile for tile in choices if tile]
    if not explicit:
        return profile.default_tile
    preferred = profile.default_tile
    return max((tile for tile in explicit if tile <= preferred), default=min(explicit))


def estimate_job_working_mib(
    settings: JobSettings,
    adapter: object,
    width: int,
    height: int,
    resolved_tile: int,
) -> int:
    sharpen_only = settings.processing_mode == ProcessingMode.sharpen_only
    if sharpen_only:
        target_width, target_height, scale = width, height, 1.0
    else:
        target_width, target_height, scale = target_dimensions(width, height, settings.target_edge)
    neural = bool(
        not sharpen_only
        and getattr(adapter, "neural", False)
        and (scale > 1 or settings.restore_large)
    )
    passes: tuple[int, ...] = ()
    if neural:
        effective_scale = max(scale, 2.0) if settings.restore_large else scale
        passes = plan_native_scales(
            effective_scale,
            min(settings.max_neural_passes, getattr(adapter, "max_passes", 1)),
            tuple(getattr(adapter, "native_scales", (1,))),
        ) or (1,)
    estimate = estimate_working_bytes(
        width,
        height,
        target_width,
        target_height,
        neural=neural,
        tile_size=resolved_tile,
        passes=passes,
    )
    return math.ceil(estimate / MIB)


def admission_reason(
    profile: ResourceProfile,
    hardware: HardwareSnapshot,
    working_mib: int,
    resolved_tile: int,
    config: AppConfig,
) -> str | None:
    if config.hardware_policy == "off":
        return None
    peak = profile.tile_peak(resolved_tile)
    if hardware.memory_kind == "unified":
        available = (
            hardware.ram_available_mib
            if hardware.ram_available_mib is not None
            else hardware.vram_available_mib
        )
        needed = working_mib + peak + max(config.ram_reserve_mib, config.vram_reserve_mib)
        if available is None or available < needed:
            found = "unknown" if available is None else f"{available / 1024:.1f} GiB"
            return (
                f"This job needs about {needed / 1024:.1f} GiB from the shared memory pool, "
                f"including the reserve; currently available: {found}."
            )
        return None
    available_ram = hardware.ram_available_mib
    needed_ram = working_mib + config.ram_reserve_mib
    if available_ram is None or available_ram < needed_ram:
        found = "unknown" if available_ram is None else f"{available_ram / 1024:.1f} GiB"
        return (
            f"This job needs about {needed_ram / 1024:.1f} GiB available RAM including the "
            f"reserve; currently available: {found}."
        )
    if peak:
        available_vram = hardware.vram_available_mib
        needed_vram = peak + config.vram_reserve_mib
        if available_vram is None or available_vram < needed_vram:
            found = "unknown" if available_vram is None else f"{available_vram / 1024:.1f} GiB"
            return (
                f"This job needs about {needed_vram / 1024:.1f} GiB free VRAM including the "
                f"reserve; currently available: {found}."
            )
    return None
