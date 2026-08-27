import numpy as np
import pytest
from PIL import Image, ImageFilter

from upscaler.imaging.finishing import (
    MAX_ENVELOPE_KERNEL,
    OVERSHOOT_TOLERANCE,
    sharpen_luminance,
)


def _luminance(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB").convert("YCbCr").split()[0], dtype=np.float32)


def _upscaled_edge(scale: int = 4) -> Image.Image:
    """A step edge plus fine texture, enlarged the way a real job enlarges it.

    The softness a finishing pass has to undo is created by the resize, so a
    fixture that is already at its final size cannot detect a sharpener whose
    radius is too narrow for the image it is handed.
    """
    small = np.full((200, 200), 60, dtype=np.int16)
    small[:, 100:] = 140
    small[::3, ::3] += 18
    source = Image.fromarray(np.clip(small, 0, 255).astype(np.uint8)).convert("RGB")
    size = (200 * scale, 200 * scale)
    return source.resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)


def _edge_gradient(image: Image.Image) -> float:
    """Peak luminance gradient across the fixture's step, i.e. its acutance."""
    y = _luminance(image)
    band = y[10:-10, y.shape[1] // 2 - 20 : y.shape[1] // 2 + 20]
    return float(np.abs(np.diff(band, axis=1)).max())


def test_zero_strength_returns_the_image_unchanged():
    source = _upscaled_edge()
    assert np.array_equal(_luminance(sharpen_luminance(source, 0)), _luminance(source))


def test_response_grows_with_strength():
    source = _upscaled_edge()
    base = _luminance(source)
    deltas = [
        float(np.abs(_luminance(sharpen_luminance(source, s, detail_scale=4.0)) - base).mean())
        for s in (15, 35, 60, 100)
    ]
    assert deltas == sorted(deltas)
    assert len(set(deltas)) == len(deltas)


def test_the_default_strength_visibly_sharpens_an_upscaled_edge():
    """The regression guard for the bug this filter replaced.

    A sub-pixel radius applied after a 4x resize left the edge gradient
    numerically identical to the unsharpened one at the shipped default.
    """
    source = _upscaled_edge()
    before = _edge_gradient(source)
    after = _edge_gradient(sharpen_luminance(source, 35, detail_scale=4.0))
    assert after >= before * 1.25


def test_overshoot_stays_inside_the_local_envelope():
    source = _upscaled_edge()
    y = _luminance(source)
    channel = source.convert("RGB").convert("YCbCr").split()[0]
    low = np.asarray(channel.filter(ImageFilter.MinFilter(MAX_ENVELOPE_KERNEL)), dtype=np.float32)
    high = np.asarray(channel.filter(ImageFilter.MaxFilter(MAX_ENVELOPE_KERNEL)), dtype=np.float32)
    slack = OVERSHOOT_TOLERANCE * (high - low) + 1.0  # one level for 8-bit rounding

    sharpened = _luminance(sharpen_luminance(source, 100, detail_scale=4.0))
    assert (sharpened <= high + slack).all()
    assert (sharpened >= low - slack).all()
    assert np.abs(sharpened - y).max() > 20  # and it is doing real work while bounded


def test_a_flat_field_is_returned_untouched():
    """The edge gate and the clamp must both collapse to nothing without detail."""
    flat = Image.new("RGB", (128, 128), (128, 128, 128))
    sharpened = sharpen_luminance(flat, 100, detail_scale=4.0)
    assert np.array_equal(_luminance(sharpened), _luminance(flat))


def test_chroma_is_never_sharpened():
    source = _upscaled_edge()
    _, cb, cr = source.convert("YCbCr").split()
    _, out_cb, out_cr = sharpen_luminance(source, 100, detail_scale=4.0).convert("YCbCr").split()
    # Only the YCbCr round trip separates them; a sharpened chroma plane would
    # differ by far more than its rounding.
    for original, finished in ((cb, out_cb), (cr, out_cr)):
        drift = np.abs(
            np.asarray(original, dtype=np.float32) - np.asarray(finished, dtype=np.float32)
        )
        assert drift.max() <= 2


def test_alpha_survives_bit_for_bit():
    source = Image.new("RGBA", (64, 64), (200, 40, 20, 0))
    source.paste((20, 120, 220, 255), (8, 8, 56, 56))
    sharpened = sharpen_luminance(source, 60, detail_scale=2.0)
    assert sharpened.mode == "RGBA"
    assert np.array_equal(np.asarray(sharpened.getchannel("A")), np.asarray(source.getchannel("A")))


def test_a_wider_detail_scale_works_over_a_wider_band():
    """The radii track the enlargement, so a 4x job is corrected wider than a 1x one."""
    source = _upscaled_edge()
    base = _luminance(source)
    narrow = np.abs(_luminance(sharpen_luminance(source, 60, detail_scale=1.0)) - base)
    wide = np.abs(_luminance(sharpen_luminance(source, 60, detail_scale=4.0)) - base)
    assert (wide > 0.5).sum() > (narrow > 0.5).sum()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_every_supported_mode_is_accepted(mode: str):
    source = Image.new(mode, (48, 48), 90 if mode == "L" else (90, 110, 130, 255)[: len(mode)])
    sharpened = sharpen_luminance(source, 45, detail_scale=2.0)
    assert sharpened.size == source.size
    assert sharpened.mode == ("RGBA" if mode == "RGBA" else "RGB")
