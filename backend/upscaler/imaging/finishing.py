from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

# Radii are multiples of the blur the previous stage left behind, never fixed
# output pixels. Sharpening runs after the exact resize, so a 4x enlargement has
# edge ramps roughly four pixels wide; a sub-pixel radius there is arithmetically
# almost a no-op, which is exactly what a single fixed-radius unsharp mask was.
FINE_RADIUS = 0.6
MID_RADIUS = 1.8
BROAD_RADIUS = 6.0

# One radius only ever restores acutance, which is visible at 1:1 and nowhere
# else. The broad layer is local contrast: a tonal change, not only an edge one,
# and the reason a sharpened result reads as improved at fit-to-screen too.
FINE_WEIGHT = 1.15
MID_WEIGHT = 0.60
BROAD_WEIGHT = 0.35

# A soft gate in place of a hard threshold. A binary cutoff both blocks fine
# texture outright and leaves a visible sharpened/unsharpened boundary; ramping
# the boost over a blurred energy map suppresses noise without either.
GATE_KNEE = 2.5
GATE_FLOOR = 0.35
# The energy map is blurred through Pillow's 8-bit filter, so it is scaled first
# to resolve the knee in quarter-level steps instead of whole ones.
GATE_PRECISION = 4.0

# Halos are what makes sharpening look cheap, and fear of them is what kept the
# amounts too low to see. Bounding the result to the local envelope plus a
# fraction of local contrast caps the overshoot instead, which is what lets the
# weights above be strong enough to matter.
OVERSHOOT_TOLERANCE = 0.18
MAX_ENVELOPE_KERNEL = 9

# Past this the radii stop tracking real edge width and only widen the halo.
MAX_DETAIL_SCALE = 4.0


def _blurred(channel: Image.Image, radius: float) -> np.ndarray:
    """Gaussian blur via Pillow's C filter, which rejects float images.

    Blurring at 8-bit costs under half a level in the detail layer, which even
    the strongest weight above keeps below one level of the result.
    """
    return np.asarray(channel.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)


def _edge_gate(mid: np.ndarray, radius: float) -> np.ndarray:
    """How much of the boost each pixel earns, from local mid-scale energy."""
    energy = np.clip(np.abs(mid) * GATE_PRECISION, 0.0, 255.0).astype(np.uint8)
    spread = _blurred(Image.fromarray(energy), radius) / GATE_PRECISION
    mask = np.clip(spread / GATE_KNEE, 0.0, 1.0)
    mask = mask * mask * (3.0 - 2.0 * mask)  # smoothstep, so the gate has no seam
    return GATE_FLOOR + (1.0 - GATE_FLOOR) * mask


def _envelope(channel: Image.Image, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """The local min and max a sharpened pixel may overshoot, and by how much."""
    kernel = min(MAX_ENVELOPE_KERNEL, max(3, round(radius) * 2 + 1))
    low = np.asarray(channel.filter(ImageFilter.MinFilter(kernel)), dtype=np.float32)
    high = np.asarray(channel.filter(ImageFilter.MaxFilter(kernel)), dtype=np.float32)
    tolerance = OVERSHOOT_TOLERANCE * (high - low)
    return low - tolerance, high + tolerance


def sharpen_luminance(
    image: Image.Image,
    strength: int,
    *,
    detail_scale: float = 1.0,
) -> Image.Image:
    """Sharpen luminance at three scales, gated on edges and clamped against halos.

    ``detail_scale`` is how much the last stage enlarged the image, which is the
    width the softness actually has. Chroma is never touched, so no amount of
    sharpening can produce a coloured fringe, and alpha is carried across intact.
    """
    if strength <= 0:
        return image.copy()

    alpha = image.getchannel("A") if image.mode == "RGBA" else None
    luminance, cb, cr = image.convert("RGB").convert("YCbCr").split()
    y = np.asarray(luminance, dtype=np.float32)

    scale = min(MAX_DETAIL_SCALE, max(1.0, detail_scale))
    fine_radius = FINE_RADIUS * scale
    fine = y - _blurred(luminance, fine_radius)
    mid = y - _blurred(luminance, MID_RADIUS * scale)
    broad = y - _blurred(luminance, BROAD_RADIUS * scale)

    amount = strength / 100.0
    boost = amount * (FINE_WEIGHT * fine + MID_WEIGHT * mid + BROAD_WEIGHT * broad)
    sharpened = y + boost * _edge_gate(mid, fine_radius)

    low, high = _envelope(luminance, fine_radius)
    sharpened = np.clip(sharpened, low, high)

    merged = Image.fromarray(np.rint(np.clip(sharpened, 0.0, 255.0)).astype(np.uint8))
    result = Image.merge("YCbCr", (merged, cb, cr)).convert("RGB")
    if alpha is not None:
        result.putalpha(alpha)
    return result
