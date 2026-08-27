import math

import pytest

from upscaler.tiles import _starts, axis_overlaps, feather_axis, plan_tiles


def test_tiles_cover_edges_and_overlap():
    tiles = plan_tiles(1000, 700, tile_size=256, overlap=32)
    assert any(tile.touches_left and tile.touches_top for tile in tiles)
    assert any(tile.touches_right and tile.touches_bottom for tile in tiles)
    assert min(tile.x0 for tile in tiles) == 0
    assert max(tile.x1 for tile in tiles) == 1000
    assert min(tile.y0 for tile in tiles) == 0
    assert max(tile.y1 for tile in tiles) == 700


def test_small_image_is_one_tile():
    tiles = plan_tiles(40, 20, tile_size=128, overlap=16)
    assert len(tiles) == 1
    assert tiles[0].width == 40
    assert tiles[0].height == 20
    assert tiles[0].touches_left
    assert tiles[0].touches_top
    assert tiles[0].touches_right
    assert tiles[0].touches_bottom


@pytest.mark.parametrize(
    ("length", "tile_size", "overlap"),
    [(1000, 256, 32), (3840, 256, 32), (1645, 256, 32), (137, 37, 9), (100, 100, 16)],
)
def test_tile_spacing_is_even_and_never_tighter_than_requested(length, tile_size, overlap):
    """Blending is defined across a known overlap, so an irregular one is a seam.

    The old layout strode by a fixed amount and then pushed a final tile flush
    against the edge, leaving that one overlapping by whatever was left over.
    """
    starts = _starts(length, tile_size, overlap)
    assert starts[0] == 0
    if length > tile_size:
        assert starts[-1] == length - tile_size

    gaps = [second - first for first, second in zip(starts[:-1], starts[1:], strict=True)]
    for gap in gaps:
        assert tile_size - gap >= overlap, "an overlap tighter than requested"
    if gaps:
        assert max(gaps) - min(gaps) <= 1, "spacing should be even to within rounding"


@pytest.mark.parametrize(
    ("length", "tile_size", "overlap"), [(1000, 256, 32), (3840, 256, 32), (137, 37, 9)]
)
def test_tile_spacing_adds_no_extra_tiles(length, tile_size, overlap):
    """Even spacing must not cost more inference than the stride implied."""
    stride = tile_size - overlap
    expected = math.ceil((length - tile_size) / stride) + 1
    assert len(_starts(length, tile_size, overlap)) == expected


def test_feather_axis_ramps_each_side_over_its_own_overlap():
    weights = feather_axis(20, overlap=4, touches_start=False, touches_end=False, end_overlap=8)
    assert weights[0] < weights[1] < weights[2] < weights[3]
    assert weights[4] == 1
    assert weights[-1] < weights[-2]
    # The trailing ramp is twice as long, so it starts falling twice as early.
    assert weights[-8] < 1


def test_axis_overlaps_reports_the_real_neighbour_distance():
    starts = _starts(137, 37, 9)
    overlaps = axis_overlaps(starts, 37, 137)
    assert overlaps[starts[0]][0] == 0, "the first tile has no leading neighbour"
    assert overlaps[starts[-1]][1] == 0, "the last tile has no trailing neighbour"
    for start, (leading, trailing) in overlaps.items():
        assert leading >= 0 and trailing >= 0
        del start


def test_axis_overlaps_ignores_duplicate_starts():
    """Callers pass one start per grid tile, so each column repeats per row."""
    starts = _starts(137, 37, 9)
    assert axis_overlaps(starts * 3, 37, 137) == axis_overlaps(starts, 37, 137)


def test_feather_keeps_outer_border_and_tapers_internal_border():
    weights = feather_axis(10, overlap=3, touches_start=True, touches_end=False)
    assert weights[0] == 1
    assert weights[-1] < weights[-2] < weights[-3] < weights[-4]


@pytest.mark.parametrize("overlap", [-1, 32])
def test_invalid_overlap(overlap):
    with pytest.raises(ValueError):
        plan_tiles(100, 100, tile_size=32, overlap=overlap)
