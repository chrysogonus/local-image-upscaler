from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

from upscaler.config import AppConfig
from upscaler.hardware import HardwareService, HardwareSnapshot, hardware_from_comfy_stats
from upscaler.models.base import ModelAdapter
from upscaler.models.classical import ClassicalAdapter
from upscaler.models.comfyui import ComfyUiIllustrationAdapter, Workflow
from upscaler.models.realesrgan_cuda import RealEsrganCudaAdapter
from upscaler.models.realesrgan_ncnn import RealEsrganNcnnAdapter
from upscaler.models.spandrel_sr import SpandrelSrAdapter
from upscaler.resource_policy import (
    ENGINE_PROFILES,
    WORKFLOW_PROFILES,
    HardwarePolicyError,
    ResourceProfile,
    admission_reason,
    capacity_reason,
    estimate_job_working_mib,
    resolve_tile,
    safe_tile_sizes,
)
from upscaler.schemas import (
    GENERATIVE_MODES,
    FeatureExclusion,
    HardwareReport,
    JobSettings,
    ModeCapability,
    ProcessingMode,
    ResourceRequirement,
    WorkflowInfo,
)

MODE_NAMES = {
    ProcessingMode.upscale: "Upscale",
    ProcessingMode.illustration: "Illustration",
    ProcessingMode.sharpen_only: "Sharpen",
}
MODE_DESCRIPTIONS = {
    ProcessingMode.upscale: "Photos and general images. Recovers detail the pixels imply.",
    ProcessingMode.illustration: (
        "Anime, line art, and digital drawing. Pixel-space model, no redraw."
    ),
    ProcessingMode.sharpen_only: "Sharpens edges at the original size. Adds no resolution.",
}

# The engines each mode may run, best first; the first available one wins.
#
# The transformer leads: SwinIR-L trained on the BSRGAN degradation pipeline
# reconstructs real photographs better than the 2018 RRDBNet generator behind
# Real-ESRGAN, which stays as the fallback because it is the one that installs
# without CUDA-only weights. Upscale ends at the always-available resampler so
# the mode is never unusable.
#
# Illustration has exactly one engine and no fallback. Its claim is that a model
# trained on drawn edges did the enlargement; standing a photographic model in
# for it would quietly answer a different question, so the mode reports itself
# unavailable instead.
MODE_ENGINES = {
    ProcessingMode.upscale: ("spandrel-sr", "realesrgan-cuda", "realesrgan", "classical"),
    ProcessingMode.illustration: ("comfyui-illustration",),
    ProcessingMode.sharpen_only: ("classical",),
}

# Logical candidates exposed only to the developer benchmark. Keeping this
# separate from MODE_ENGINES prevents a benchmark comparison from becoming an
# engine picker in the product API.
BENCHMARK_CANDIDATE_ENGINES = {
    "classical": ("classical",),
    "swinir": ("spandrel-sr",),
    "realesrgan": ("realesrgan-cuda", "realesrgan"),
}


@dataclass(frozen=True, slots=True)
class ResolvedJobPlan:
    adapter: ModelAdapter
    workflow_id: str | None
    tile_size: int
    profile: ResourceProfile
    estimated_working_mib: int


class ModelRegistry:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.hardware = HardwareService(config)
        self._comfy_stable_hardware: HardwareSnapshot | None = None
        adapters: list[ModelAdapter] = [
            ClassicalAdapter(),
            ComfyUiIllustrationAdapter(),
            SpandrelSrAdapter(),
            RealEsrganCudaAdapter(),
            RealEsrganNcnnAdapter(config.realesrgan_binary),
        ]
        self._adapters = {adapter.id: adapter for adapter in adapters}

    def adapter_for(self, mode: ProcessingMode) -> ModelAdapter:
        adapter = self._first_available(mode)
        if adapter is not None:
            return adapter
        raise ValueError(self._unavailable_reason(mode))

    def resolve_benchmark_candidate(
        self,
        candidate: str,
        settings: JobSettings,
        *,
        width: int,
        height: int,
    ) -> ResolvedJobPlan:
        """Resolve one explicit faithful candidate for the local benchmark.

        This deliberately accepts logical benchmark candidates rather than
        arbitrary adapter ids. In particular, both Real-ESRGAN runtimes are one
        model candidate: CUDA is preferred and NCNN is the fallback.
        """
        engine_ids = BENCHMARK_CANDIDATE_ENGINES.get(candidate)
        if engine_ids is None:
            raise ValueError(f"Unknown benchmark candidate {candidate!r}.")
        reasons: list[str] = []
        for engine_id in engine_ids:
            adapter = self._adapters[engine_id]
            if not adapter.available:
                if adapter.unavailable_reason:
                    reasons.append(adapter.unavailable_reason)
                continue
            profile = self._profile(adapter)
            reason = self._capacity_reason(adapter, profile)
            if reason:
                reasons.append(reason)
                continue
            hardware = self._hardware_for(adapter, live=True)
            try:
                tile_size = resolve_tile(profile, settings.tile_size, hardware, self.config)
                estimated = estimate_job_working_mib(settings, adapter, width, height, tile_size)
                reason = admission_reason(profile, hardware, estimated, tile_size, self.config)
            except HardwarePolicyError as exc:
                reasons.append(str(exc))
                continue
            if reason:
                reasons.append(reason)
                continue
            return ResolvedJobPlan(
                adapter=adapter,
                workflow_id=None,
                tile_size=tile_size,
                profile=profile,
                estimated_working_mib=estimated,
            )
        detail = reasons[0] if reasons else "No compatible engine is installed."
        raise ValueError(f"Benchmark candidate {candidate!r} is unavailable: {detail}")

    def resolve_job(
        self,
        settings: JobSettings,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> ResolvedJobPlan:
        """Resolve one immutable plan; supplied dimensions also perform live admission."""
        workflow = None
        if settings.workflow:
            if settings.processing_mode != ProcessingMode.illustration:
                raise ValueError("Only illustration mode accepts a workflow.")
            adapter = self._adapters.get("comfyui-illustration")
            if adapter is None or not adapter.available:
                reason = adapter.unavailable_reason if adapter is not None else None
                raise ValueError(reason or "ComfyUI is unavailable.")
            workflow = next(
                (
                    item
                    for item in self._comfy_adapter(adapter).workflows()
                    if item.id == settings.workflow
                ),
                None,
            )
            if workflow is None:
                raise ValueError(f"Unknown ComfyUI workflow {settings.workflow!r}.")
            profile = WORKFLOW_PROFILES[workflow.id]
            reason = self._capacity_reason(adapter, profile)
            if reason:
                raise HardwarePolicyError(f"{workflow.name} is excluded: {reason}")
        else:
            adapter = self._first_available(settings.processing_mode)
            if adapter is None:
                reason = self._unavailable_reason(settings.processing_mode)
                if self._has_installed_engine(settings.processing_mode):
                    raise HardwarePolicyError(reason)
                raise ValueError(reason)
            if adapter.id == "comfyui-illustration":
                safe_workflows = self._safe_workflows(adapter)
                workflow = safe_workflows[0] if safe_workflows else None
            profile = self._profile(adapter, workflow.id if workflow else None)

        hardware = self._hardware_for(adapter, live=width is not None)
        tile_size = resolve_tile(profile, settings.tile_size, hardware, self.config)
        estimated = 0
        if width is not None and height is not None:
            estimated = estimate_job_working_mib(settings, adapter, width, height, tile_size)
            reason = admission_reason(profile, hardware, estimated, tile_size, self.config)
            reclaim = getattr(adapter, "reclaim_memory_if_idle", None)
            if reason and adapter.id == "comfyui-illustration" and callable(reclaim) and reclaim():
                hardware = self._hardware_for(adapter, live=True)
                reason = admission_reason(profile, hardware, estimated, tile_size, self.config)
            if reason:
                raise HardwarePolicyError(reason)
        return ResolvedJobPlan(
            adapter=adapter,
            workflow_id=workflow.id if workflow else None,
            tile_size=tile_size,
            profile=profile,
            estimated_working_mib=estimated,
        )

    def capabilities(self) -> list[ModeCapability]:
        return [self._capability(mode) for mode in ProcessingMode]

    def workflows(self) -> list[WorkflowInfo]:
        """The external graphs a mode would actually offer right now.

        Empty unless an engine that runs them is the one a mode resolved to, so
        the interface never shows a choice that nothing would act on.
        """
        adapter = self._first_available(ProcessingMode.illustration)
        if adapter is None or not hasattr(adapter, "workflows"):
            return []
        return [self._workflow_info(workflow) for workflow in self._safe_workflows(adapter)]

    def hardware_reports(self) -> list[HardwareReport]:
        reports = [self._hardware_report(self.hardware.snapshot())]
        adapter = self._adapters.get("comfyui-illustration")
        if adapter is not None and adapter.available:
            report = self._hardware_for(adapter, live=True)
            reports.append(self._hardware_report(report))
        return reports

    def excluded_features(self) -> list[FeatureExclusion]:
        if self.config.hardware_policy == "off":
            return []
        excluded: list[FeatureExclusion] = []
        for mode in ProcessingMode:
            capability = self._capability(mode)
            if not capability.available and capability.unavailable_reason:
                excluded.append(
                    FeatureExclusion(
                        id=mode.value,
                        name=MODE_NAMES[mode],
                        reason=capability.unavailable_reason,
                    )
                )
        adapter = self._adapters.get("comfyui-illustration")
        if adapter is not None and adapter.available:
            safe_ids = {workflow.id for workflow in self._safe_workflows(adapter)}
            for workflow in self._comfy_adapter(adapter).workflows():
                if workflow.id in safe_ids:
                    continue
                reason = self._capacity_reason(adapter, WORKFLOW_PROFILES[workflow.id])
                if reason:
                    excluded.append(
                        FeatureExclusion(id=workflow.id, name=workflow.name, reason=reason)
                    )
        return excluded

    def _first_available(self, mode: ProcessingMode) -> ModelAdapter | None:
        for engine_id in MODE_ENGINES[mode]:
            adapter = self._adapters.get(engine_id)
            if adapter is not None and adapter.available and self._adapter_fits(adapter):
                return adapter
        return None

    def _has_installed_engine(self, mode: ProcessingMode) -> bool:
        return any(
            (adapter := self._adapters.get(engine_id)) is not None and adapter.available
            for engine_id in MODE_ENGINES[mode]
        )

    def _adapter_fits(self, adapter: ModelAdapter) -> bool:
        if self.config.hardware_policy == "off":
            return True
        if adapter.id == "comfyui-illustration":
            return bool(self._safe_workflows(adapter))
        profile = self._profile(adapter)
        if self._capacity_reason(adapter, profile) is not None:
            return False
        if profile.app_tiles:
            return bool(safe_tile_sizes(profile, self._hardware_for(adapter), self.config))
        return True

    def _safe_workflows(self, adapter: ModelAdapter) -> tuple[Workflow, ...]:
        workflows = self._comfy_adapter(adapter).workflows()
        if self.config.hardware_policy == "off":
            return workflows
        return tuple(
            workflow
            for workflow in workflows
            if self._capacity_reason(adapter, WORKFLOW_PROFILES[workflow.id]) is None
        )

    def _profile(self, adapter: ModelAdapter, workflow_id: str | None = None) -> ResourceProfile:
        if workflow_id:
            return WORKFLOW_PROFILES[workflow_id]
        return ENGINE_PROFILES[adapter.id]

    def _hardware_for(self, adapter: ModelAdapter, *, live: bool = False) -> HardwareSnapshot:
        if adapter.id != "comfyui-illustration":
            return self.hardware.snapshot() if live else self.hardware.stable
        stats = self._comfy_adapter(adapter).hardware_stats(refresh=live)
        if stats is None:
            return HardwareSnapshot(
                scope="comfyui",
                ram_physical_mib=None,
                ram_effective_mib=None,
                ram_available_mib=None,
                gpu_name=None,
                vram_total_mib=None,
                vram_available_mib=None,
                memory_kind=self.config.memory_kind_override or "dedicated",
                source="ComfyUI /system_stats",
            )
        report = hardware_from_comfy_stats(
            stats, memory_kind_override=self.config.memory_kind_override
        )
        if self.config.ram_mib_override is not None:
            report = replace(
                report,
                ram_effective_mib=self.config.ram_mib_override,
                ram_available_mib=min(
                    report.ram_available_mib
                    if report.ram_available_mib is not None
                    else self.config.ram_mib_override,
                    self.config.ram_mib_override,
                ),
            )
        if self.config.vram_mib_override is not None:
            report = replace(
                report,
                vram_total_mib=self.config.vram_mib_override,
                vram_available_mib=min(
                    report.vram_available_mib
                    if report.vram_available_mib is not None
                    else self.config.vram_mib_override,
                    self.config.vram_mib_override,
                ),
            )
        if self.config.gpu_name_override:
            report = replace(report, gpu_name=self.config.gpu_name_override)
        if self._comfy_stable_hardware is None:
            self._comfy_stable_hardware = report
        if live:
            return replace(
                self._comfy_stable_hardware,
                ram_available_mib=report.ram_available_mib,
                vram_available_mib=report.vram_available_mib,
                warnings=report.warnings,
            )
        return self._comfy_stable_hardware

    def _capacity_reason(self, adapter: ModelAdapter, profile: ResourceProfile) -> str | None:
        if self.config.hardware_policy == "off":
            return None
        hardware = self._hardware_for(adapter)
        reason = capacity_reason(profile, hardware)
        if reason:
            return reason
        if (
            hardware.memory_kind == "dedicated"
            and profile.vram_mib
            and hardware.vram_total_mib is not None
            and hardware.vram_total_mib < profile.vram_mib + self.config.vram_reserve_mib
        ):
            needed = profile.vram_mib + self.config.vram_reserve_mib
            return (
                f"Requires {needed / 1024:.1f} GiB VRAM including the configured reserve; "
                f"the detected GPU has {hardware.vram_total_mib / 1024:.1f} GiB."
            )
        return None

    @staticmethod
    def _requirement(profile: ResourceProfile) -> ResourceRequirement:
        return ResourceRequirement(
            ram_mib=profile.ram_mib,
            vram_mib=profile.vram_mib,
            unified_mib=profile.unified_mib,
        )

    def _workflow_info(self, workflow: Workflow) -> WorkflowInfo:
        profile = WORKFLOW_PROFILES[workflow.id]
        return WorkflowInfo(
            id=workflow.id,
            name=workflow.name,
            description=workflow.description,
            warning=workflow.warning,
            enlarges=workflow.enlarges,
            resource_requirement=self._requirement(profile),
        )

    @staticmethod
    def _hardware_report(snapshot: HardwareSnapshot) -> HardwareReport:
        return HardwareReport(
            scope=snapshot.scope,
            ram_physical_mib=snapshot.ram_physical_mib,
            ram_effective_mib=snapshot.ram_effective_mib,
            ram_available_mib=snapshot.ram_available_mib,
            gpu_name=snapshot.gpu_name,
            vram_total_mib=snapshot.vram_total_mib,
            vram_available_mib=snapshot.vram_available_mib,
            memory_kind=snapshot.memory_kind,
            source=snapshot.source,
            warnings=list(snapshot.warnings),
        )

    def _unavailable_reason(self, mode: ProcessingMode) -> str:
        """The most actionable reason among the engines this mode could have used."""
        for engine_id in MODE_ENGINES[mode]:
            adapter = self._adapters.get(engine_id)
            if adapter is not None and adapter.available:
                if adapter.id == "comfyui-illustration":
                    reasons = [
                        self._capacity_reason(adapter, WORKFLOW_PROFILES[item.id])
                        for item in self._comfy_adapter(adapter).workflows()
                    ]
                    reason = next((item for item in reasons if item), None)
                else:
                    reason = self._capacity_reason(adapter, self._profile(adapter))
                if reason:
                    return reason
                profile = self._profile(adapter)
                if profile.app_tiles and not safe_tile_sizes(
                    profile, self._hardware_for(adapter), self.config
                ):
                    return (
                        "No tile leaves the configured VRAM reserve on this GPU. Reduce "
                        "UPSCALER_VRAM_RESERVE_MIB or use UPSCALER_HARDWARE_POLICY=off."
                    )
            if adapter is not None and adapter.unavailable_reason:
                return adapter.unavailable_reason
        return f"{MODE_NAMES[mode]} is unavailable on this machine."

    def _fallback_reason(self, mode: ProcessingMode, resolved: ModelAdapter) -> str | None:
        """Why this mode is running on the resampler instead of a neural engine.

        Only reported once the mode has actually degraded. Naming a missing CUDA
        engine while the Vulkan one is doing the work would be nagging, not
        information.
        """
        if resolved.neural:
            return None
        for engine_id in MODE_ENGINES[mode]:
            if engine_id == resolved.id:
                return None
            adapter = self._adapters.get(engine_id)
            if adapter is not None and adapter.available and not self._adapter_fits(adapter):
                if adapter.id == "comfyui-illustration":
                    workflow = self._comfy_adapter(adapter).workflows()[0]
                    reason = self._capacity_reason(adapter, WORKFLOW_PROFILES[workflow.id])
                else:
                    profile = self._profile(adapter)
                    reason = self._capacity_reason(adapter, profile)
                    if reason is None and profile.app_tiles:
                        reason = "No tile leaves the configured VRAM reserve on this GPU."
                if reason:
                    return reason
            if adapter is not None and adapter.unavailable_reason:
                return adapter.unavailable_reason
        return None

    @staticmethod
    def _comfy_adapter(adapter: ModelAdapter) -> ComfyUiIllustrationAdapter:
        # Registry membership, rather than concrete inheritance, is the
        # interface boundary here. Tests and third-party registries may supply
        # another structurally compatible adapter under the ComfyUI id.
        return cast(ComfyUiIllustrationAdapter, adapter)

    def _capability(self, mode: ProcessingMode) -> ModeCapability:
        description = MODE_DESCRIPTIONS[mode]
        adapter = self._first_available(mode)
        if adapter is None:
            return ModeCapability(
                mode=mode,
                name=MODE_NAMES[mode],
                description=description,
                available=False,
                generative=mode in GENERATIVE_MODES,
                engine=self._adapters[MODE_ENGINES[mode][0]].name,
                device="Unavailable",
                unavailable_reason=self._unavailable_reason(mode),
            )
        workflow = (
            self._safe_workflows(adapter)[0] if adapter.id == "comfyui-illustration" else None
        )
        profile = self._profile(adapter, workflow.id if workflow else None)
        hardware = self._hardware_for(adapter)
        return ModeCapability(
            mode=mode,
            name=MODE_NAMES[mode],
            description=description,
            available=True,
            generative=mode in GENERATIVE_MODES,
            engine=adapter.name,
            device=adapter.device,
            fallback_reason=self._fallback_reason(mode, adapter),
            max_passes=adapter.max_passes,
            native_scales=list(adapter.native_scales),
            supports_tta=adapter.supports_tta,
            resource_requirement=self._requirement(profile),
            safe_tile_sizes=list(safe_tile_sizes(profile, hardware, self.config)),
        )
