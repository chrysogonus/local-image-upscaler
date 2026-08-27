"""Seam-safety of the shared tiled inference.

These are the tests that make a blending mistake detectable. A bad window does
not crash and does not look broken in a thumbnail; it leaves a faint band that
only shows at 1:1, which is exactly the kind of defect a test has to catch
because a glance will not.
"""

from __future__ import annotations

from threading import Event

import numpy as np
import pytest

from upscaler.models.base import ProcessingCancelled
from upscaler.models.tiled import cosine_window, resolve_tiling, run_tiled
from upscaler.tiles import Tile


def _identity_infer(scale: int):
    """Nearest-neighbour enlargement: whatever comes back is the blend's doing."""

    def infer(patch: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(patch, scale, axis=0), scale, axis=1)

    return infer


def _run(source: np.ndarray, *, scale: int = 2, tile_size: int = 32, overlap: int = 8):
    return run_tiled(
        source,
        scale=scale,
        tile_size=tile_size,
        overlap=overlap,
        infer=_identity_infer(scale),
        cancel=Event(),
        progress=lambda *_: None,
        message="testing",
        np=np,
    )


def test_a_uniform_image_survives_tiling_exactly() -> None:
    """The blend must be a partition of unity. If the weights do not sum to one
    everywhere, a flat field comes back with visible banding."""
    source = np.full((100, 140, 3), 0.5, dtype=np.float32)
    result = _run(source)

    assert result.shape == (200, 280, 3)
    assert np.allclose(result, 0.5, atol=1e-5)


@pytest.mark.parametrize(
    ("width", "height", "tile_size", "overlap"),
    [(160, 96, 48, 12), (160, 96, 32, 8), (137, 91, 37, 9), (3840, 1645, 256, 32)],
)
def test_the_window_sums_to_one_before_normalisation(
    width: int, height: int, tile_size: int, overlap: int
) -> None:
    """The property that makes the blend the one that was designed.

    ``run_tiled`` divides by the accumulated weight, so a window that does not
    sum to one still produces a plausible image - which is why this needs its
    own test rather than being caught by comparing outputs. What the division
    cannot restore is each tile's intended *share* at a corner where four tiles
    meet; getting that wrong silently reweights the crossfade.
    """
    from upscaler.tiles import axis_overlaps, plan_tiles

    tiles = plan_tiles(width, height, tile_size, overlap)
    horizontal = axis_overlaps([tile.x0 for tile in tiles], tile_size, width)
    vertical = axis_overlaps([tile.y0 for tile in tiles], tile_size, height)

    accumulated = np.zeros((height, width, 1), dtype=np.float32)
    for tile in tiles:
        top, bottom = vertical[tile.y0]
        left, right = horizontal[tile.x0]
        accumulated[tile.y0 : tile.y1, tile.x0 : tile.x1, :] += cosine_window(
            tile, (top, bottom, left, right), 1, np
        )

    assert np.abs(accumulated - 1.0).max() < 1e-5


@pytest.mark.parametrize("tile_size,overlap", [(48, 12), (32, 8), (64, 32), (37, 9)])
def test_tiling_reproduces_untiled_inference_exactly(tile_size: int, overlap: int) -> None:
    """Guards the accumulate-and-divide machinery: indexing, offsets, edge tiles.

    Note this passes for any strictly positive window, because the division by
    accumulated weight normalises whatever the window happened to be. The window
    itself is pinned by the partition-of-unity test above.
    """
    rng = np.random.default_rng(0)
    source = rng.random((96, 160, 3), dtype=np.float32)
    expected = _identity_infer(2)(source)
    result = _run(source, tile_size=tile_size, overlap=overlap)

    assert result.shape == expected.shape
    assert np.abs(result - expected).max() < 1e-5


def test_a_gradient_survives_tiling_without_seams() -> None:
    """A gradient is where a seam is most visible: a step in a signal the eye
    expects to be smooth. Compared against the untiled result so the test does
    not measure the nearest-neighbour fixture's own staircase."""
    ramp = np.linspace(0.0, 1.0, 160, dtype=np.float32)
    source = np.repeat(ramp[None, :, None], 96, axis=0).repeat(3, axis=2)
    expected = _identity_infer(2)(source)
    result = _run(source, tile_size=48, overlap=12)

    assert np.abs(result - expected).max() < 1e-5


def test_a_single_tile_image_is_untouched_by_the_window() -> None:
    """No neighbours means no blending; ramping here would darken the border."""
    source = np.random.default_rng(0).random((20, 20, 3), dtype=np.float32)
    result = _run(source, scale=1, tile_size=32, overlap=8)

    assert np.allclose(result, source, atol=1e-6)


def test_borders_keep_full_weight() -> None:
    """A tile edge that touches the frame has nothing to blend with."""
    single = Tile(0, 0, 10, 10, True, True, True, True)
    window = cosine_window(single, (4, 4, 4, 4), 1, np)
    assert np.allclose(window, 1.0)

    interior = Tile(10, 10, 20, 20, False, False, False, False)
    window = cosine_window(interior, (4, 4, 4, 4), 1, np)
    assert window[0, 0, 0] < 0.5
    assert window[-1, -1, 0] < 0.5
    assert window[5, 5, 0] == pytest.approx(1.0)


def test_each_side_ramps_over_its_own_overlap() -> None:
    """The flush final tile overlaps by more than the requested stride, and a
    ramp that ignores that leaves both tiles at full weight across the middle
    of the shared strip - a flat average where a crossfade was intended."""
    tile = Tile(0, 0, 40, 40, False, False, False, False)
    window = cosine_window(tile, (4, 20, 4, 20), 1, np)

    profile = window[:, 20, 0]
    # Four pixels of ramp at the leading edge, twenty at the trailing one.
    assert profile[4] > 0.9
    assert profile[-20] < 0.999
    assert profile[-1] < 0.05


def test_the_window_has_no_derivative_step() -> None:
    """A linear ramp leaves a faint band because its slope jumps at each end of
    the ramp. The raised cosine is what removes it, so the second difference
    across the transition has to stay bounded."""
    tile = Tile(0, 0, 64, 64, False, True, False, True)
    window = cosine_window(tile, (16, 16, 16, 16), 1, np)

    profile = window[0, :, 0]
    curvature = np.abs(np.diff(profile, n=2))
    assert curvature.max() < 0.02


def test_cancellation_is_checked_between_tiles() -> None:
    cancel = Event()
    cancel.set()
    with pytest.raises(ProcessingCancelled):
        run_tiled(
            np.zeros((64, 64, 3), dtype=np.float32),
            scale=1,
            tile_size=32,
            overlap=8,
            infer=_identity_infer(1),
            cancel=cancel,
            progress=lambda *_: None,
            message="testing",
            np=np,
        )


def test_progress_reaches_one() -> None:
    seen: list[float | None] = []
    _ = run_tiled(
        np.zeros((80, 80, 3), dtype=np.float32),
        scale=1,
        tile_size=32,
        overlap=8,
        infer=_identity_infer(1),
        cancel=Event(),
        progress=lambda _phase, _message, fraction: seen.append(fraction),
        message="testing",
        np=np,
    )
    assert seen[0] == 0.0
    assert seen[-1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("requested", "width", "height", "expected_tile"),
    [
        (0, 4000, 2000, 256),  # automatic falls back to the engine default
        (128, 4000, 2000, 128),  # an explicit choice is honoured
        (2048, 300, 200, 300),  # never larger than the image's long edge
        (16, 4000, 2000, 64),  # never below the floor
    ],
)
def test_tile_resolution_stays_inside_its_bounds(
    requested: int, width: int, height: int, expected_tile: int
) -> None:
    tile, overlap = resolve_tiling(requested, width, height, default=256)
    assert tile == expected_tile
    assert 0 <= overlap < tile
