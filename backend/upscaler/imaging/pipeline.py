from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from PIL import Image

from upscaler.geometry import plan_native_scales, target_dimensions
from upscaler.imaging.finishing import sharpen_luminance
from upscaler.imaging.io import normalize_input, resize_exact, save_output
from upscaler.models.base import (
    ModelAdapter,
    ModelRequest,
    ProcessingCancelled,
    ProgressCallback,
)
from upscaler.schemas import JobSettings, ProcessingMode, ResultInfo, SourceInfo


@dataclass(frozen=True, slots=True)
class ProcessResult:
    source: SourceInfo
    result: ResultInfo
    output_path: Path


def _check_cancel(cancel: Event) -> None:
    if cancel.is_set():
        raise ProcessingCancelled("processing was cancelled")


def _pass_progress(progress: ProgressCallback, index: int, count: int) -> ProgressCallback:
    """Fold one pass's own progress into its slice of the whole chain.

    Without this the bar restarts at zero on every pass, which reads as the job
    having failed and started over.
    """
    if count == 1:
        return progress

    def report(phase: str, message: str, fraction: float | None) -> None:
        if fraction is not None:
            fraction = (index + fraction) / count
        progress(phase, f"{message} (pass {index + 1} of {count})", fraction)

    return report


def process_image(
    source_path: Path,
    output_path: Path,
    workspace: Path,
    original_filename: str,
    settings: JobSettings,
    adapter: ModelAdapter,
    max_input_pixels: int,
    cancel: Event,
    progress: ProgressCallback,
    *,
    resolved_tile_size: int | None = None,
) -> ProcessResult:
    progress("analyzing", "Validating and decoding the source image", None)
    normalized = normalize_input(
        source_path,
        workspace / "normalized.png",
        original_filename,
        max_input_pixels,
    )
    _check_cancel(cancel)

    width, height = normalized.source.width, normalized.source.height
    sharpen_only = settings.processing_mode == ProcessingMode.sharpen_only
    if sharpen_only:
        target_width, target_height, requested_scale = width, height, 1.0
    else:
        target_width, target_height, requested_scale = target_dimensions(
            width, height, settings.target_edge
        )
    use_neural = (
        not sharpen_only and adapter.neural and (requested_scale > 1 or settings.restore_large)
    )
    warnings = list(normalized.source.warnings)
    if sharpen_only:
        warnings.append(
            "Sharpen-only processing preserved the source dimensions; it did not create new "
            "resolution."
        )
    elif adapter.neural and not use_neural:
        warnings.append(
            "The source already meets the target; neural enlargement was skipped and the image "
            "was reduced faithfully."
        )

    model_input = normalized.path
    model_output = workspace / "enhanced.png"
    engine_id = "classical"
    plan: tuple[int, ...] = ()
    # Where the last engine's detail was actually made, when that is not the size
    # of the file it returned. Zero until an engine says otherwise.
    detail_width = 0
    if sharpen_only:
        working_path = model_input
        engine_id = "classical:sharpen-only"
    elif use_neural:
        effective_scale = max(requested_scale, 2.0) if settings.restore_large else requested_scale
        plan = plan_native_scales(
            effective_scale,
            min(settings.max_neural_passes, adapter.max_passes),
            adapter.native_scales,
        )
        if not plan:
            # The source already meets the target, but this engine was kept
            # because its job is to change the picture rather than its size.
            # Run it once at the source's own dimensions and let the single
            # exact resize below reach the target as it always does.
            plan = (1,)
        working_path = model_input
        # Chaining keeps the model responsible for the whole enlargement. A
        # single pass caps at 4x, so anything beyond that used to be handed to
        # Lanczos, which is what made small sources come back soft.
        seen: dict[str, None] = {}
        for index, native_scale in enumerate(plan):
            _check_cancel(cancel)
            model_result = adapter.enhance(
                ModelRequest(
                    source_path=working_path,
                    output_path=workspace / f"enhanced-{index + 1}.png",
                    workspace=workspace,
                    native_scale=native_scale,
                    target_width=target_width,
                    target_height=target_height,
                    tile_size=settings.tile_size,
                    tta=settings.tta,
                    workflow=settings.workflow,
                ),
                cancel,
                _pass_progress(progress, index, len(plan)),
            )
            working_path = model_result.output_path
            engine_id = model_result.engine_id
            detail_width = model_result.detail_width
            seen.update(dict.fromkeys(model_result.warnings))
        warnings.extend(seen)
        if len(plan) > 1:
            chain = " -> ".join(f"{scale}x" for scale in plan)
            warnings.append(
                f"Enlargement used {len(plan)} chained neural passes ({chain}); each pass can "
                "compound the previous one's artifacts."
            )
    elif not adapter.neural:
        model_result = adapter.enhance(
            ModelRequest(
                source_path=model_input,
                output_path=model_output,
                workspace=workspace,
                native_scale=1,
                target_width=target_width,
                target_height=target_height,
                tile_size=0,
                workflow=settings.workflow,
            ),
            cancel,
            progress,
        )
        working_path = model_result.output_path
        engine_id = model_result.engine_id
        detail_width = model_result.detail_width
    else:
        # A source larger than the target should not be needlessly enlarged by a
        # neural model and then reduced again unless restoration was requested.
        shutil.copyfile(model_input, model_output)
        working_path = model_output
        engine_id = "classical:neural-skipped"

    _check_cancel(cancel)
    finish_message = (
        "Preparing the source at its original dimensions"
        if sharpen_only
        else "Resizing once to the exact target dimensions"
    )
    progress("finishing", finish_message, None)
    with Image.open(working_path) as opened:
        opened.load()
        # How much this last resize stretched the image is how wide the softness
        # the finishing pass has to undo actually is. A neural engine synthesises
        # detail at its own output size, so only the remainder after it counts -
        # and an engine that resized to the target itself has to report that
        # size, or the stretch it did internally is invisible here and the
        # filter runs too fine to undo the softness it left.
        detail_scale = min(4.0, max(1.0, target_width / (detail_width or opened.width)))
        finished = (
            opened.copy() if sharpen_only else resize_exact(opened, (target_width, target_height))
        )

    _check_cancel(cancel)

    if settings.sharpen:
        progress("finishing", "Sharpening edge and local detail", None)
        finished = sharpen_luminance(finished, settings.sharpen, detail_scale=detail_scale)

    _check_cancel(cancel)
    progress("encoding", "Encoding PNG output", None)
    save_output(finished, output_path, normalized.icc_profile)
    _check_cancel(cancel)

    base = Path(original_filename).stem or "image"
    if sharpen_only:
        label = "sharpened"
    else:
        label = (
            f"{settings.target_edge // 960}k"
            if settings.target_edge in {3840, 7680}
            else str(settings.target_edge)
        )
    return ProcessResult(
        source=normalized.source,
        result=ResultInfo(
            width=target_width,
            height=target_height,
            bytes=output_path.stat().st_size,
            engine=engine_id,
            processing_mode=settings.processing_mode,
            filename=f"{base}-{label}.png",
            neural_passes=list(plan),
            resolved_tile_size=(
                settings.tile_size if resolved_tile_size is None else resolved_tile_size
            ),
            generative=bool(adapter.generative and plan),
            warnings=warnings,
        ),
        output_path=output_path,
    )
