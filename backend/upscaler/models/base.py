from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Protocol

ProgressCallback = Callable[[str, str, float | None], None]


class ProcessingCancelled(RuntimeError):
    pass


class ModelExecutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelRequest:
    source_path: Path
    output_path: Path
    workspace: Path
    native_scale: int
    # The job's final output dimensions. An engine with a fixed enlargement
    # factor ignores these and multiplies by native_scale; an engine that
    # renders to whatever size it is asked for uses these instead, so that a
    # small source is not rendered at 4x and then resampled the rest of the way.
    target_width: int
    target_height: int
    tile_size: int
    tta: bool = False
    # Set only by the mode that offers it. An engine that runs someone else's
    # graph needs to know which graph; that has no meaning for a fixed-weight
    # upscaler.
    workflow: str | None = None


@dataclass(frozen=True, slots=True)
class ModelResult:
    output_path: Path
    engine_id: str
    warnings: tuple[str, ...] = ()
    # The width at which this engine's model stage actually produced detail,
    # when that is not the width of the file it returned. An engine that resizes
    # to the target itself has already absorbed part of the enlargement, and the
    # finishing sharpen sizes its radii to how far the image was stretched: left
    # unreported, that stretch is invisible and the filter runs too fine to undo
    # the softness. Zero means the output is its own answer.
    detail_width: int = 0


class ModelAdapter(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def neural(self) -> bool: ...

    # Synthesises detail the source never contained. Never true by default: an
    # adapter has to claim it, and the interface labels it wherever it appears.
    @property
    def generative(self) -> bool: ...

    # How many times this engine may be chained to cover a large factor. A
    # diffusion pass costs minutes, so it caps lower than a feed-forward one.
    @property
    def max_passes(self) -> int: ...

    # The enlargement factors this engine actually produces correctly. Asking
    # for anything else is a silent correctness bug, not a slow path: the NCNN
    # runtime accepts -s 2 against a 4x model and returns a cropped image at the
    # right dimensions, which looks like a working result until it is compared.
    @property
    def native_scales(self) -> tuple[int, ...]: ...

    # Whether test-time augmentation reaches this engine's inference at all. An
    # engine that runs someone else's graph, or a one-step restorer with no
    # ensemble to average, cannot honour it - and a control that costs eight
    # inferences for an identical result must not be offered for it.
    @property
    def supports_tta(self) -> bool: ...

    # The weights' licence. Every engine here is permissively licensed; the
    # string is reported so that stays checkable rather than remembered.
    @property
    def license(self) -> str: ...

    @property
    def available(self) -> bool: ...

    @property
    def unavailable_reason(self) -> str | None: ...

    @property
    def device(self) -> str: ...

    def enhance(
        self,
        request: ModelRequest,
        cancel: Event,
        progress: ProgressCallback,
    ) -> ModelResult: ...
