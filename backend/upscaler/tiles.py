from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int
    touches_left: bool
    touches_top: bool
    touches_right: bool
    touches_bottom: bool

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def _starts(length: int, tile_size: int, overlap: int) -> list[int]:
    """Evenly spaced tile origins covering the axis.

    Spacing is distributed rather than fixed. Striding by ``tile_size -
    overlap`` and then pushing one last tile flush against the far edge leaves
    that final tile overlapping its neighbour by an arbitrary leftover - often
    far more than the requested overlap, and sometimes so much that three tiles
    cover the same pixels. Blending is defined pairwise across a known overlap,
    so an irregular one shows up as a seam in the last row and column of every
    image whose dimensions happen not to divide evenly.

    Spreading the same number of tiles evenly keeps every overlap identical and
    never below the requested amount, at no extra cost: the tile count is the
    one the stride implies.
    """
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    span = length - tile_size
    count = math.ceil(span / stride) + 1
    return [round(index * span / (count - 1)) for index in range(count)]


def plan_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[Tile]:
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    if tile_size < 1:
        raise ValueError("tile size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be non-negative and smaller than the tile")

    result: list[Tile] = []
    for y0 in _starts(height, tile_size, overlap):
        for x0 in _starts(width, tile_size, overlap):
            x1, y1 = min(width, x0 + tile_size), min(height, y0 + tile_size)
            result.append(
                Tile(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    touches_left=x0 == 0,
                    touches_top=y0 == 0,
                    touches_right=x1 == width,
                    touches_bottom=y1 == height,
                )
            )
    return result


def feather_axis(
    length: int,
    overlap: int,
    touches_start: bool,
    touches_end: bool,
    end_overlap: int | None = None,
) -> tuple[float, ...]:
    """Crossfade weights along one axis of a tile.

    ``end_overlap`` defaults to ``overlap``, but the two sides genuinely differ:
    ``plan_tiles`` finishes each axis with a tile pushed flush against the edge,
    which can overlap its neighbour by much more than the requested amount. A
    ramp narrower than the real overlap leaves both tiles at full weight across
    the middle of the shared strip, so the crossfade there collapses into a flat
    average - a seam in exactly the last row and column of every image whose
    dimensions do not divide evenly.
    """
    if length < 1:
        raise ValueError("axis length must be positive")
    end = overlap if end_overlap is None else end_overlap
    weights = [1.0] * length
    if overlap > 0 and not touches_start:
        ramp = min(overlap, length)
        for index in range(ramp):
            weights[index] = min(weights[index], (index + 1) / (ramp + 1))
    if end > 0 and not touches_end:
        ramp = min(end, length)
        for offset in range(ramp):
            index = length - 1 - offset
            weights[index] = min(weights[index], (offset + 1) / (ramp + 1))
    return tuple(weights)


def axis_overlaps(starts: list[int], tile_size: int, length: int) -> dict[int, tuple[int, int]]:
    """Each start's real (leading, trailing) overlap with its neighbours.

    Derived from the layout rather than assumed, because the flush final tile
    does not honour the requested stride. Callers pass one start per tile and
    tiles form a grid, so the same coordinate arrives once per row: deduplicate
    before pairing, or a start ends up compared against itself and reports an
    overlap of a whole tile.
    """
    ordered = sorted(set(starts))
    result: dict[int, tuple[int, int]] = {}
    for index, start in enumerate(ordered):
        end = min(length, start + tile_size)
        leading = 0 if index == 0 else min(length, ordered[index - 1] + tile_size) - start
        trailing = 0 if index == len(ordered) - 1 else end - ordered[index + 1]
        result[start] = (max(0, leading), max(0, trailing))
    return result
