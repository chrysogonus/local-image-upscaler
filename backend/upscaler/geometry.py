from __future__ import annotations

import math


def target_dimensions(width: int, height: int, target_edge: int) -> tuple[int, int, float]:
    if width < 1 or height < 1 or target_edge < 1:
        raise ValueError("image and target dimensions must be positive")
    scale = target_edge / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale)), scale


NATIVE_SCALES = (2, 3, 4)
MAX_NATIVE_SCALE = NATIVE_SCALES[-1]

# A leftover factor this small is a gentle resample rather than an enlargement,
# so spending a whole extra neural pass on it costs far more time than the
# softness it would remove.
RESIDUAL_TOLERANCE = 1.15

DEFAULT_MAX_PASSES = 3


def choose_native_scale(requested_scale: float) -> int:
    """Select a supported Real-ESRGAN scale for a single pass."""
    if requested_scale <= 2:
        return 2
    if requested_scale <= 3:
        return 3
    return MAX_NATIVE_SCALE


def plan_native_scales(
    requested_scale: float,
    max_passes: int = DEFAULT_MAX_PASSES,
    native_scales: tuple[int, ...] = NATIVE_SCALES,
) -> tuple[int, ...]:
    """Native model scales whose product covers ``requested_scale``.

    One Real-ESRGAN pass tops out at 4x, so a larger factor used to be finished
    with Lanczos - precisely the enlargement the model was supposed to perform.
    A 480x270 source bound for 4K needs 8x, of which only 4x was neural and the
    remaining 2x was a plain resample, which is why small sources came back
    soft. Chaining passes keeps the model responsible for the whole factor, up
    to ``max_passes``; the remainder is left to the single exact resize that
    already ends the pipeline.
    """
    if max_passes < 1:
        raise ValueError("at least one neural pass must be allowed")
    if not native_scales:
        raise ValueError("an engine must support at least one native scale")
    if requested_scale <= 1:
        return ()

    ordered = tuple(sorted(native_scales))
    plan: list[int] = []
    remaining = requested_scale
    while remaining > RESIDUAL_TOLERANCE and len(plan) < max_passes:
        step = _smallest_native_scale_at_least(remaining, ordered)
        plan.append(step)
        remaining /= step
    if not plan:
        # Between 1x and the tolerance: still worth one restorative pass, since
        # the caller only asks for a plan when neural processing was requested.
        plan.append(_smallest_native_scale_at_least(requested_scale, ordered))
    return tuple(plan)


def _smallest_native_scale_at_least(factor: float, native_scales: tuple[int, ...]) -> int:
    for scale in native_scales:
        if factor <= scale:
            return scale
    return native_scales[-1]


def decoded_rgba_bytes(width: int, height: int) -> int:
    return width * height * 4


def estimate_working_bytes(
    width: int,
    height: int,
    target_width: int,
    target_height: int,
    *,
    neural: bool,
    tile_size: int,
    native_scale: int = 4,
    passes: tuple[int, ...] = (),
) -> int:
    source_and_output = decoded_rgba_bytes(width, height) + decoded_rgba_bytes(
        target_width, target_height
    )
    if not neural:
        return math.ceil(source_and_output * 2.5)
    if passes:
        native_scale = max(passes)
    tile = tile_size or min(512, max(width, height))
    tile_working = tile * tile * (4 + 32 + 32 * native_scale * native_scale)
    # A chained plan overshoots the target on purpose, so the largest buffer the
    # run ever holds is the last pass's output rather than the encoded result.
    chained = 0
    if passes:
        total = math.prod(passes)
        chained = decoded_rgba_bytes(width * total, height * total)
    return math.ceil(source_and_output * 1.5 + tile_working + chained)
