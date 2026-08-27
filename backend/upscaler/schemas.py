from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProcessingMode(str, Enum):
    """What a job claims about its output.

    Every mode here reconstructs: it recovers detail the pixels still imply.
    ``upscale`` does that with a general photographic model, ``illustration``
    with one trained specifically for drawn edges and flat colour, and
    ``sharpen_only`` changes no dimensions at all. None of them synthesises
    detail that was never in the source; see ACCEPTABLE_USE.md.
    """

    upscale = "upscale"
    illustration = "illustration"
    sharpen_only = "sharpen_only"


# Modes whose output contains invented detail. Empty here, and the test suite
# keeps it that way: this application only reconstructs. The label survives the
# emptiness deliberately, so that an engine which ever did invent detail would
# be announced by the interface rather than quietly indistinguishable from one
# that did not.
GENERATIVE_MODES: frozenset[ProcessingMode] = frozenset()


class JobState(str, Enum):
    queued = "queued"
    analyzing = "analyzing"
    loading_model = "loading_model"
    enhancing = "enhancing"
    finishing = "finishing"
    encoding = "encoding"
    cancelling = "cancelling"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


TERMINAL_STATES = {JobState.completed, JobState.failed, JobState.cancelled}


class JobSettings(BaseModel):
    """Everything a job needs from the user.

    The engine, model weights, and output encoding are all derived from
    ``processing_mode`` rather than exposed: each mode has one
    correct answer for them, and offering the combinations invited choices that
    could only make the result worse.
    """

    model_config = ConfigDict(extra="forbid")

    target_edge: int = Field(default=3840, ge=256, le=7680)
    processing_mode: ProcessingMode = ProcessingMode.upscale
    sharpen: int = Field(default=15, ge=0, le=100)
    tile_size: int = Field(default=0, ge=0, le=2048)
    tta: bool = False
    restore_large: bool = False
    max_neural_passes: int = Field(default=3, ge=1, le=4)
    # Only meaningful when the resolved engine runs external graphs. Left unset,
    # the engine takes the first workflow it offers.
    workflow: str | None = Field(default=None, max_length=64)

    @field_validator("tile_size")
    @classmethod
    def valid_tile_size(cls, value: int) -> int:
        if value and value < 32:
            raise ValueError("tile size must be zero (automatic) or at least 32")
        return value

    @model_validator(mode="after")
    def valid_processing_combination(self) -> JobSettings:
        if self.processing_mode != ProcessingMode.illustration and self.workflow:
            raise ValueError(
                "a workflow only applies to illustration mode, which is the only mode that "
                "runs an external graph"
            )
        if self.processing_mode == ProcessingMode.sharpen_only:
            if not self.sharpen:
                raise ValueError("sharpen-only mode requires a non-zero sharpen strength")
            if self.tile_size or self.tta or self.restore_large:
                raise ValueError(
                    "sharpen-only mode does not accept neural tile, augmentation, or "
                    "restore-large settings"
                )
        return self


class SourceInfo(BaseModel):
    filename: str
    width: int
    height: int
    mode: str
    format: str | None = None
    animated: bool = False
    frames: int = 1
    has_alpha: bool = False
    has_icc: bool = False
    bit_depth: int = 8
    warnings: list[str] = Field(default_factory=list)


class ResultInfo(BaseModel):
    width: int
    height: int
    bytes: int
    engine: str
    processing_mode: ProcessingMode
    filename: str
    neural_passes: list[int] = Field(default_factory=list)
    resolved_tile_size: int = 0
    generative: bool = False
    warnings: list[str] = Field(default_factory=list)


class JobSnapshot(BaseModel):
    id: str
    state: JobState
    phase: str
    message: str
    progress: float | None = Field(default=None, ge=0, le=1)
    settings: JobSettings
    source: SourceInfo | None = None
    result: ResultInfo | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    revision: int


class ModeCapability(BaseModel):
    """Whether one mode can run here, and on what.

    Reported per mode rather than per engine: the interface offers modes, so
    anything it cannot act on is noise. ``unavailable_reason`` is what the user
    is shown in place of the mode, and has to be actionable.
    """

    mode: ProcessingMode
    name: str
    description: str
    available: bool
    generative: bool
    engine: str
    device: str
    unavailable_reason: str | None = None
    # Set when the mode runs, but on a materially weaker engine than it could -
    # currently only the deterministic resampler standing in for a neural one.
    # Silently degrading would present a plain enlargement as AI upscaling.
    fallback_reason: str | None = None
    max_passes: int = 1
    native_scales: list[int] = Field(default_factory=list)
    # Whether test-time augmentation reaches this mode's engine. Reported so the
    # interface can drop the control rather than disable it: a setting the
    # engine discards is not a tradeoff the user can make.
    supports_tta: bool = False
    resource_requirement: ResourceRequirement | None = None
    safe_tile_sizes: list[int] = Field(default_factory=list)
    safe_targets: list[int] = Field(default_factory=lambda: [3840, 7680])


class ResourceRequirement(BaseModel):
    ram_mib: int
    vram_mib: int
    unified_mib: int


class HardwareReport(BaseModel):
    scope: str
    ram_physical_mib: int | None = None
    ram_effective_mib: int | None = None
    ram_available_mib: int | None = None
    gpu_name: str | None = None
    vram_total_mib: int | None = None
    vram_available_mib: int | None = None
    memory_kind: str
    source: str
    warnings: list[str] = Field(default_factory=list)


class HardwarePolicyInfo(BaseModel):
    mode: str
    version: int
    ram_reserve_mib: int
    vram_reserve_mib: int
    visibility_basis: str = "stable total capacity"
    admission_basis: str = "currently available memory"


class FeatureExclusion(BaseModel):
    id: str
    name: str
    reason: str


class WorkflowInfo(BaseModel):
    """One external graph the user can pick, and what it honestly does.

    ``enlarges`` is the claim, not a feature flag: a workflow that edits at its
    own working resolution says so, so that the plain resample which takes its
    output up to 4K is never presented as detail the model generated.
    """

    id: str
    name: str
    description: str
    warning: str
    enlarges: bool
    resource_requirement: ResourceRequirement | None = None
    safe_targets: list[int] = Field(default_factory=lambda: [3840, 7680])


class Capabilities(BaseModel):
    version: str
    modes: list[ModeCapability]
    # Non-empty only when the resolved engine for a mode runs external graphs.
    workflows: list[WorkflowInfo] = Field(default_factory=list)
    targets: list[int] = [3840, 7680]
    max_upload_bytes: int
    max_input_pixels: int
    platform: dict[str, Any]
    hardware: list[HardwareReport] = Field(default_factory=list)
    hardware_policy: HardwarePolicyInfo
    excluded_features: list[FeatureExclusion] = Field(default_factory=list)
