from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

from upscaler.schemas import SourceInfo


class InvalidImageError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NormalizedImage:
    path: Path
    source: SourceInfo
    icc_profile: bytes | None


def inspect_input_dimensions(source_path: Path, max_pixels: int) -> tuple[int, int]:
    """Validate an upload cheaply and return its dimensions after EXIF orientation."""
    previous_pixel_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source_path) as probe:
                width, height = probe.size
                if width * height > max_pixels:
                    raise InvalidImageError(
                        f"Decoded image has {width * height:,} pixels; the limit is {max_pixels:,}."
                    )
                probe.verify()
            with Image.open(source_path) as metadata:
                orientation = metadata.getexif().get(274)
    except Image.DecompressionBombError as exc:
        raise InvalidImageError("Image dimensions exceed the decoder safety limit.") from exc
    except Image.DecompressionBombWarning as exc:
        raise InvalidImageError("Image dimensions exceed the decoder safety limit.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise InvalidImageError("The file is not a supported or valid image.") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_pixel_limit
    return (height, width) if orientation in {5, 6, 7, 8} else (width, height)


def _bit_depth(mode: str) -> int:
    if "16" in mode or mode in {"I", "F"}:
        return 16
    return 8


def _has_alpha(image: Image.Image) -> bool:
    return image.mode in {"RGBA", "LA", "PA"} or "transparency" in image.info


def _convert_to_working_srgb(
    image: Image.Image,
    source_icc: bytes | None,
) -> tuple[Image.Image, bytes | None, list[str]]:
    messages: list[str] = []
    has_alpha = _has_alpha(image)
    alpha = image.convert("RGBA").getchannel("A") if has_alpha else None
    color = image.convert("RGB")
    srgb_profile = ImageCms.createProfile("sRGB")
    output_icc: bytes | None = None

    if source_icc:
        try:
            input_profile = ImageCms.ImageCmsProfile(io.BytesIO(source_icc))
            profiled = ImageCms.profileToProfile(
                color,
                input_profile,
                srgb_profile,
                outputMode="RGB",
                renderingIntent=ImageCms.Intent.PERCEPTUAL,
            )
            if profiled is None:
                raise ImageCms.PyCMSError("ICC conversion returned no image")
            color = profiled
            output_icc = ImageCms.ImageCmsProfile(srgb_profile).tobytes()
        except (OSError, ValueError, ImageCms.PyCMSError):
            messages.append("The embedded ICC profile was invalid; pixels were treated as sRGB.")
    else:
        output_icc = ImageCms.ImageCmsProfile(srgb_profile).tobytes()

    if alpha is not None:
        color.putalpha(alpha)
    return color, output_icc, messages


def normalize_input(
    source_path: Path,
    normalized_path: Path,
    original_filename: str,
    max_pixels: int,
) -> NormalizedImage:
    inspect_input_dimensions(source_path, max_pixels)

    with Image.open(source_path) as opened:
        source_format = opened.format
        original_mode = opened.mode
        animated = bool(getattr(opened, "is_animated", False))
        frames = int(getattr(opened, "n_frames", 1))
        has_alpha = _has_alpha(opened)
        source_icc = opened.info.get("icc_profile")
        opened.seek(0)
        opened.load()
        image = ImageOps.exif_transpose(opened)

    messages: list[str] = []
    if animated and frames > 1:
        messages.append(f"Only the first of {frames} animation frames was processed.")
    depth = _bit_depth(original_mode)
    if depth > 8:
        messages.append(
            "The current neural working path converts high-bit-depth pixels to 8-bit sRGB."
        )

    converted, working_icc, color_messages = _convert_to_working_srgb(image, source_icc)
    messages.extend(color_messages)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    if working_icc:
        converted.save(normalized_path, format="PNG", compress_level=2, icc_profile=working_icc)
    else:
        converted.save(normalized_path, format="PNG", compress_level=2)

    return NormalizedImage(
        path=normalized_path,
        source=SourceInfo(
            filename=original_filename,
            width=converted.width,
            height=converted.height,
            mode=original_mode,
            format=source_format,
            animated=animated,
            frames=frames,
            has_alpha=has_alpha,
            has_icc=bool(source_icc),
            bit_depth=depth,
            warnings=messages.copy(),
        ),
        icc_profile=working_icc,
    )


def resize_exact(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if image.size == size:
        return image.copy()
    # Pillow's RGBA resampler accounts for alpha during convolution; keeping the
    # image in RGBA avoids compositing transparent edges against an arbitrary matte.
    mode = "RGBA" if _has_alpha(image) else "RGB"
    return image.convert(mode).resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)


def save_output(image: Image.Image, path: Path, icc_profile: bytes | None) -> None:
    """Encode the finished image as PNG.

    The only output format, deliberately. A tool whose whole purpose is recovering
    detail should not end by discarding some to a lossy encoder, and PNG is also
    the only one of the candidates that carries alpha without compositing it
    against an arbitrary matte colour.
    """
    if icc_profile:
        image.save(path, format="PNG", compress_level=6, icc_profile=icc_profile)
    else:
        image.save(path, format="PNG", compress_level=6)
