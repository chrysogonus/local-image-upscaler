"""Shared tiled inference and seam-safe blending.

Every neural engine here processes an image in overlapping tiles, and every one
of them used to carry its own copy of the accumulate-and-divide loop. Blending
correctness is not a per-engine detail - a mistake in the weights shows up as a
soft band or a visible seam whichever model produced the pixels - so it lives
once, here.

The window is the important part. A linear ramp across the overlap is what the
engines used to do, and it leaves a faint band because its derivative steps at
both ends of the ramp: the eye reads the discontinuity in the rate of change,
not in the value. A raised cosine has no such step, so the transition is
genuinely invisible.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Event
from typing import Any

from upscaler.models.base import ProcessingCancelled, ProgressCallback
from upscaler.tiles import Tile, axis_overlaps, feather_axis, plan_tiles

# The overlap each engine gets by default, in *source* pixels. Deliberately
# wider than the 16px that just hides a hard edge: neighbouring tiles resolve
# an ambiguous texture slightly differently, and the raised-cosine ramp needs a
# long enough runway to cross fade that disagreement below visibility.
DEFAULT_OVERLAP = 32


def cosine_window(tile: Tile, ramps: tuple[int, int, int, int], scale: int, np: Any) -> Any:
    """Raised-cosine blend weights for one tile, in output-pixel space.

    ``ramps`` is this tile's real (top, bottom, left, right) overlap with its
    neighbours in source pixels, which is not one constant: the tile flush
    against each far edge overlaps by whatever was left over.

    Edges that touch the image border keep full weight: there is no neighbour
    to blend with there, and ramping down into the frame edge would darken it.

    The cosine is applied to each axis *before* the outer product, not to the
    product afterwards. That ordering is the whole correctness argument. Two
    tiles crossfading along one axis contribute ``c(t)`` and ``c(1-t)``, and the
    raised cosine satisfies ``c(t) + c(1-t) == 1``, so a separable window sums
    to exactly one everywhere - including the corners where four tiles meet,
    where the total factorises as ``(c(v)+c(1-v)) * (c(h)+c(1-h))``. Lifting the
    product instead breaks that: four tiles at a corner sum to about 0.59
    rather than 1. ``run_tiled`` divides by the accumulated weight and so hides
    the error, but only by rescaling each tile's contribution away from the
    share the ramp was supposed to give it, which is a blend nobody designed.
    """
    top, bottom, left, right = ramps
    rows = feather_axis(
        tile.height * scale,
        top * scale,
        tile.touches_top,
        tile.touches_bottom,
        end_overlap=bottom * scale,
    )
    columns = feather_axis(
        tile.width * scale,
        left * scale,
        tile.touches_left,
        tile.touches_right,
        end_overlap=right * scale,
    )
    vertical = _raised_cosine(np.asarray(rows, dtype=np.float32), np).reshape(-1, 1, 1)
    horizontal = _raised_cosine(np.asarray(columns, dtype=np.float32), np).reshape(1, -1, 1)
    return (vertical * horizontal).astype(np.float32)


def _raised_cosine(ramp: Any, np: Any) -> Any:
    """Smooth a linear ramp without moving its endpoints or breaking ``c(t)+c(1-t)==1``.

    A linear crossfade leaves a faint band because its derivative steps at both
    ends of the ramp; the eye reads the discontinuity in the rate of change even
    though the values are continuous.
    """
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(ramp, 0.0, 1.0))


def run_tiled(
    source: Any,
    *,
    scale: int,
    tile_size: int,
    overlap: int,
    infer: Callable[[Any], Any],
    cancel: Event,
    progress: ProgressCallback,
    message: str,
    np: Any,
) -> Any:
    """Run ``infer`` over overlapping tiles and blend the results.

    ``source`` is an HWC array. ``infer`` receives one HWC patch and returns its
    HWC float32 result, already enlarged by ``scale``. Progress is reported per
    completed tile; engines whose single tile takes tens of seconds report from
    inside ``infer`` instead and pass the fraction through themselves.
    """
    height, width = source.shape[:2]
    tiles = plan_tiles(width, height, tile_size, overlap)
    horizontal = axis_overlaps([tile.x0 for tile in tiles], tile_size, width)
    vertical = axis_overlaps([tile.y0 for tile in tiles], tile_size, height)

    accumulated = np.zeros((height * scale, width * scale, 3), dtype=np.float32)
    weights = np.zeros((height * scale, width * scale, 1), dtype=np.float32)

    progress("enhancing", f"{message} across {len(tiles)} tile(s)", 0.0)
    for index, tile in enumerate(tiles):
        if cancel.is_set():
            raise ProcessingCancelled("processing was cancelled")
        patch = source[tile.y0 : tile.y1, tile.x0 : tile.x1, :]
        result = infer(patch)

        top, bottom = vertical[tile.y0]
        left, right = horizontal[tile.x0]
        window = cosine_window(tile, (top, bottom, left, right), scale, np)
        y0, x0 = tile.y0 * scale, tile.x0 * scale
        y1, x1 = y0 + result.shape[0], x0 + result.shape[1]
        accumulated[y0:y1, x0:x1, :] += result * window
        weights[y0:y1, x0:x1, :] += window
        progress("enhancing", message, (index + 1) / len(tiles))

    # Every output pixel belongs to at least one tile, so this only guards
    # against a division by zero from a degenerate plan.
    np.divide(accumulated, np.maximum(weights, 1e-6), out=accumulated)
    return accumulated


def self_ensemble(call: Callable[[Any], Any], tensor: Any, torch: Any) -> Any:
    """Average a model over the dihedral group of the square.

    Eight-way geometric self-ensemble. Still standard practice at the top of the
    NTIRE super-resolution leaderboards, and it costs exactly what it says:
    eight forward passes for one tile.
    """
    total = None
    for rotation in range(4):
        for flip in (False, True):
            variant = torch.rot90(tensor, rotation, dims=(2, 3))
            if flip:
                variant = torch.flip(variant, dims=(3,))
            out = call(variant)
            if flip:
                out = torch.flip(out, dims=(3,))
            out = torch.rot90(out, -rotation, dims=(2, 3))
            total = out if total is None else total + out
    assert total is not None
    return total / 8.0


def resolve_tiling(
    requested_tile: int,
    width: int,
    height: int,
    *,
    default: int,
    minimum: int = 64,
    maximum: int | None = None,
) -> tuple[int, int]:
    """Return the (tile_size, overlap) an engine should actually use."""
    tile = requested_tile or default
    if maximum is not None:
        tile = min(tile, maximum)
    tile = max(minimum, min(tile, max(width, height)))
    return tile, min(DEFAULT_OVERLAP, max(0, tile // 4))
