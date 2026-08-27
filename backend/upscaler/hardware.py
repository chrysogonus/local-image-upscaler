from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from upscaler.config import AppConfig

MIB = 1024 * 1024
_CGROUP_V2_MAX = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V2_CURRENT = Path("/sys/fs/cgroup/memory.current")
_CGROUP_V1_MAX = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
_CGROUP_V1_CURRENT = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")


@dataclass(frozen=True, slots=True)
class HardwareSnapshot:
    scope: str
    ram_physical_mib: int | None
    ram_effective_mib: int | None
    ram_available_mib: int | None
    gpu_name: str | None
    vram_total_mib: int | None
    vram_available_mib: int | None
    memory_kind: str
    source: str
    warnings: tuple[str, ...] = ()


def _read_int(path: Path) -> int | None:
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if not raw or raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def parse_meminfo(text: str) -> tuple[int | None, int | None]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"^(MemTotal|MemAvailable):\s+(\d+)\s+kB$", line.strip())
        if match:
            values[match.group(1)] = int(match.group(2)) * 1024
    return values.get("MemTotal"), values.get("MemAvailable")


def cgroup_memory(
    limit: int | None,
    current: int | None,
    physical: int | None,
) -> tuple[int | None, int | None]:
    """Return a real cgroup limit and its remaining bytes, ignoring sentinel maxima."""
    if limit is None or limit <= 0:
        return None, None
    # cgroup v1 represents "unlimited" with a huge page-aligned integer.
    if limit >= (1 << 60) or (physical is not None and limit > physical * 16):
        return None, None
    remaining = max(0, limit - current) if current is not None else None
    return limit, remaining


def parse_nvidia_smi(text: str) -> tuple[str | None, int | None, int | None]:
    line = next((item.strip() for item in text.splitlines() if item.strip()), "")
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 3:
        return None, None, None
    try:
        return parts[0] or None, int(float(parts[1])), int(float(parts[2]))
    except ValueError:
        return None, None, None


def parse_vulkan_name(text: str) -> str | None:
    for pattern in (r"deviceName\s*=\s*(.+)", r"GPU\d+:\s*(.+)"):
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def classify_memory(
    gpu_name: str | None,
    vram_total_mib: int | None,
    ram_effective_mib: int | None,
    override: str | None = None,
) -> str:
    if override:
        return override
    name = (gpu_name or "").lower()
    if any(token in name for token in ("gb10", "grace blackwell", "jetson", "tegra", "apple")):
        return "unified"
    if (
        vram_total_mib is not None
        and ram_effective_mib is not None
        and ram_effective_mib >= 32 * 1024
        and abs(vram_total_mib - ram_effective_mib) <= ram_effective_mib * 0.05
    ):
        return "unified"
    return "dedicated"


def hardware_from_comfy_stats(
    stats: dict[str, Any],
    *,
    memory_kind_override: str | None = None,
) -> HardwareSnapshot:
    system_value = stats.get("system")
    system: dict[str, Any] = system_value if isinstance(system_value, dict) else {}
    devices = stats.get("devices") if isinstance(stats.get("devices"), list) else []
    device: dict[str, Any] = devices[0] if devices and isinstance(devices[0], dict) else {}

    def mib(value: Any) -> int | None:
        return int(value) // MIB if isinstance(value, (int, float)) and value >= 0 else None

    ram_total = mib(system.get("ram_total"))
    ram_free = mib(system.get("ram_free"))
    vram_total = mib(
        device.get("vram_total")
        if device.get("vram_total") is not None
        else device.get("torch_vram_total")
    )
    vram_free = mib(
        device.get("vram_free")
        if device.get("vram_free") is not None
        else device.get("torch_vram_free")
    )
    name = device.get("name") if isinstance(device.get("name"), str) else None
    kind = classify_memory(name, vram_total, ram_total, memory_kind_override)
    return HardwareSnapshot(
        scope="comfyui",
        ram_physical_mib=ram_total,
        ram_effective_mib=ram_total,
        ram_available_mib=ram_free,
        gpu_name=name,
        vram_total_mib=vram_total,
        vram_available_mib=vram_free,
        memory_kind=kind,
        source="ComfyUI /system_stats",
    )


class HardwareService:
    """Detect stable capacity once and refresh only currently available memory."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._stable = self._detect()

    def _ram(self) -> tuple[int | None, int | None, int | None]:
        try:
            physical, available = parse_meminfo(Path("/proc/meminfo").read_text())
        except OSError:
            pages = os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else 0
            page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 0
            physical = pages * page_size if pages and page_size else None
            available = None
        limit_raw = _read_int(_CGROUP_V2_MAX)
        current = _read_int(_CGROUP_V2_CURRENT)
        if limit_raw is None:
            limit_raw = _read_int(_CGROUP_V1_MAX)
            current = _read_int(_CGROUP_V1_CURRENT)
        limit, remaining = cgroup_memory(limit_raw, current, physical)
        effective_values = [value for value in (physical, limit) if value is not None]
        effective = min(effective_values) if effective_values else None
        available_values = [value for value in (available, remaining) if value is not None]
        effective_available = min(available_values) if available_values else None
        return physical, effective, effective_available

    def _torch_gpu(self) -> tuple[str | None, int | None, int | None]:
        try:
            import torch

            if not torch.cuda.is_available():
                return None, None, None
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            return str(props.name), int(total) // MIB, int(free) // MIB
        except Exception:  # noqa: BLE001 - optional runtimes must not break CPU startup
            return None, None, None

    def _nvidia_gpu(self) -> tuple[str | None, int | None, int | None]:
        binary = shutil.which("nvidia-smi")
        if not binary:
            return None, None, None
        try:
            completed = subprocess.run(  # noqa: S603 - fixed system utility and arguments
                [
                    binary,
                    "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return None, None, None
        return parse_nvidia_smi(completed.stdout)

    def _vulkan_gpu(self) -> tuple[str | None, int | None, int | None]:
        total = used = None
        for candidate in sorted(Path("/sys/class/drm").glob("card*/device")):
            candidate_total = _read_int(candidate / "mem_info_vram_total")
            if candidate_total:
                total = candidate_total // MIB
                used_raw = _read_int(candidate / "mem_info_vram_used")
                used = used_raw // MIB if used_raw is not None else None
                break
        name = None
        binary = shutil.which("vulkaninfo")
        if binary:
            try:
                completed = subprocess.run(  # noqa: S603 - fixed system utility and arguments
                    [binary, "--summary"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                name = parse_vulkan_name(completed.stdout)
            except (OSError, subprocess.SubprocessError):
                pass
        return name, total, max(0, total - used) if total is not None and used is not None else None

    def _detect(self) -> HardwareSnapshot:
        physical, effective, available = self._ram()
        name, vram, vram_free = self._torch_gpu()
        source = "PyTorch CUDA"
        if vram is None:
            name, vram, vram_free = self._nvidia_gpu()
            source = "nvidia-smi"
        if vram is None:
            name, vram, vram_free = self._vulkan_gpu()
            source = "Vulkan/sysfs" if name or vram else "system"

        physical_mib = physical // MIB if physical is not None else None
        effective_mib = effective // MIB if effective is not None else None
        available_mib = available // MIB if available is not None else None
        if self.config.ram_mib_override is not None:
            effective_mib = self.config.ram_mib_override
            available_mib = (
                min(available_mib, effective_mib) if available_mib is not None else effective_mib
            )
        if self.config.vram_mib_override is not None:
            vram = self.config.vram_mib_override
            vram_free = min(vram_free, vram) if vram_free is not None else vram
        name = self.config.gpu_name_override or name
        kind = classify_memory(name, vram, effective_mib, self.config.memory_kind_override)
        warnings: list[str] = []
        if name and vram is None:
            warnings.append(
                "GPU memory capacity is unknown; safe mode hides GPU features. Set "
                "UPSCALER_VRAM_MIB if detection is wrong."
            )
        return HardwareSnapshot(
            scope="backend",
            ram_physical_mib=physical_mib,
            ram_effective_mib=effective_mib,
            ram_available_mib=available_mib,
            gpu_name=name,
            vram_total_mib=vram,
            vram_available_mib=vram_free,
            memory_kind=kind,
            source=source,
            warnings=tuple(warnings),
        )

    def snapshot(self) -> HardwareSnapshot:
        """Refresh free memory without changing totals used for stable UI visibility."""
        _, _, available = self._ram()
        available_mib = available // MIB if available is not None else None
        if self.config.ram_mib_override is not None:
            available_mib = (
                min(available_mib, self.config.ram_mib_override)
                if available_mib is not None
                else self.config.ram_mib_override
            )
        _, _, vram_free = self._torch_gpu()
        if vram_free is None:
            _, _, vram_free = self._nvidia_gpu()
        if vram_free is None:
            _, _, vram_free = self._vulkan_gpu()
        if self.config.vram_mib_override is not None:
            vram_free = (
                min(vram_free, self.config.vram_mib_override)
                if vram_free is not None
                else self.config.vram_mib_override
            )
        return replace(
            self._stable,
            ram_available_mib=available_mib,
            vram_available_mib=vram_free,
        )

    @property
    def stable(self) -> HardwareSnapshot:
        return self._stable
