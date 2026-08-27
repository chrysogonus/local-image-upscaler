from pathlib import Path
from threading import Event

import pytest
from PIL import Image, ImageChops, ImageCms

from upscaler.imaging import pipeline
from upscaler.imaging.io import InvalidImageError, inspect_input_dimensions, normalize_input
from upscaler.imaging.pipeline import process_image
from upscaler.models.base import ModelRequest, ModelResult
from upscaler.models.classical import ClassicalAdapter
from upscaler.schemas import JobSettings, ProcessingMode


class FakeNeuralAdapter:
    """A neural engine that only enlarges, so pass geometry is what is measured."""

    id = "fake-neural"
    name = "Fake neural"
    neural = True
    generative = False

    def __init__(
        self,
        max_passes: int = 3,
        native_scales: tuple[int, ...] = (2, 3, 4),
    ) -> None:
        self.max_passes = max_passes
        self.native_scales = native_scales
        self.calls: list[tuple[int, tuple[int, int]]] = []

    @property
    def available(self) -> bool:
        return True

    @property
    def unavailable_reason(self) -> None:
        return None

    @property
    def device(self) -> str:
        return "fake"

    def enhance(self, request: ModelRequest, cancel: Event, progress) -> ModelResult:
        with Image.open(request.source_path) as opened:
            opened.load()
            self.calls.append((request.native_scale, opened.size))
            enlarged = opened.resize(
                (opened.width * request.native_scale, opened.height * request.native_scale),
                Image.NEAREST,
            )
        progress("enhancing", "fake pass", 1.0)
        enlarged.save(request.output_path, format="PNG")
        return ModelResult(
            output_path=request.output_path,
            engine_id=f"{self.id}:x{request.native_scale}",
            warnings=("Inspect fine textures at 1:1.",),
        )


def test_normalize_applies_exif_orientation(tmp_path: Path):
    source = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (30, 10), "red")
    exif = Image.Exif()
    exif[274] = 6
    image.save(source, exif=exif)

    normalized = normalize_input(source, tmp_path / "normalized.png", source.name, 10_000)

    assert inspect_input_dimensions(source, 10_000) == (10, 30)
    assert (normalized.source.width, normalized.source.height) == (10, 30)


def test_normalize_rejects_non_image(tmp_path: Path):
    source = tmp_path / "bad.bin"
    source.write_bytes(b"not an image")
    with pytest.raises(InvalidImageError):
        normalize_input(source, tmp_path / "normalized.png", source.name, 10_000)


def test_normalize_converts_grayscale_and_records_the_source_mode(tmp_path: Path):
    source = tmp_path / "grey.png"
    Image.new("L", (17, 11), 96).save(source)

    normalized = normalize_input(source, tmp_path / "normalized.png", source.name, 10_000)

    assert normalized.source.mode == "L"
    assert normalized.source.has_alpha is False
    with Image.open(normalized.path) as image:
        assert image.mode == "RGB"
        assert image.size == (17, 11)


def test_normalize_preserves_a_valid_icc_profile_as_srgb(tmp_path: Path):
    source = tmp_path / "profiled.png"
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.new("RGB", (17, 11), "#506070").save(source, icc_profile=profile)

    normalized = normalize_input(source, tmp_path / "normalized.png", source.name, 10_000)

    assert normalized.source.has_icc is True
    assert normalized.icc_profile
    assert normalized.source.warnings == []
    with Image.open(normalized.path) as image:
        assert image.info.get("icc_profile") == normalized.icc_profile


def test_normalize_warns_when_an_icc_profile_is_invalid(tmp_path: Path):
    source = tmp_path / "bad-profile.png"
    Image.new("RGB", (17, 11), "#506070").save(source, icc_profile=b"not an ICC profile")

    normalized = normalize_input(source, tmp_path / "normalized.png", source.name, 10_000)

    assert normalized.source.has_icc is True
    assert normalized.icc_profile is None
    assert normalized.source.warnings == [
        "The embedded ICC profile was invalid; pixels were treated as sRGB."
    ]


def test_normalize_warns_about_high_bit_depth_conversion(tmp_path: Path):
    source = tmp_path / "sixteen-bit.png"
    Image.new("I;16", (17, 11), 4096).save(source)

    normalized = normalize_input(source, tmp_path / "normalized.png", source.name, 10_000)

    assert normalized.source.bit_depth == 16
    assert normalized.source.warnings == [
        "The current neural working path converts high-bit-depth pixels to 8-bit sRGB."
    ]


def test_normalize_processes_only_the_first_animation_frame_and_warns(tmp_path: Path):
    source = tmp_path / "animated.gif"
    frames = [Image.new("RGB", (17, 11), color) for color in ("red", "blue")]
    frames[0].save(source, save_all=True, append_images=frames[1:], duration=20, loop=0)

    normalized = normalize_input(source, tmp_path / "normalized.png", source.name, 10_000)

    assert normalized.source.animated is True
    assert normalized.source.frames == 2
    assert normalized.source.warnings == ["Only the first of 2 animation frames was processed."]


def test_classical_pipeline_preserves_alpha_and_exact_size(tmp_path: Path):
    source = tmp_path / "source.png"
    output = tmp_path / "result.png"
    image = Image.new("RGBA", (64, 32), (220, 20, 10, 0))
    image.paste((20, 120, 220, 255), (8, 4, 56, 28))
    image.save(source)
    events: list[tuple[str, float | None]] = []

    result = process_image(
        source,
        output,
        tmp_path,
        source.name,
        JobSettings(target_edge=256, sharpen=10),
        ClassicalAdapter(),
        1_000_000,
        Event(),
        lambda phase, _message, progress: events.append((phase, progress)),
    )

    assert (result.result.width, result.result.height) == (256, 128)
    with Image.open(output) as processed:
        assert processed.mode == "RGBA"
        assert processed.getchannel("A").getextrema() == (0, 255)
    assert events[-1][0] == "encoding"


def test_sharpen_only_preserves_dimensions_and_skips_enhancement(tmp_path: Path):
    source = tmp_path / "soft.png"
    output = tmp_path / "result.png"
    image = Image.new("RGB", (37, 19), (80, 80, 80))
    image.paste((160, 160, 160), (10, 4, 27, 15))
    image.save(source)
    events: list[str] = []

    result = process_image(
        source,
        output,
        tmp_path,
        source.name,
        JobSettings(
            target_edge=7680,
            processing_mode=ProcessingMode.sharpen_only,
            sharpen=35,
        ),
        ClassicalAdapter(),
        1_000_000,
        Event(),
        lambda phase, _message, _progress: events.append(phase),
    )

    assert (result.result.width, result.result.height) == image.size
    assert result.result.processing_mode == ProcessingMode.sharpen_only
    assert result.result.engine == "classical:sharpen-only"
    assert result.result.filename == "soft-sharpened.png"
    assert "enhancing" not in events
    with Image.open(output) as processed:
        assert processed.size == image.size
        assert ImageChops.difference(processed.convert("RGB"), image).getbbox() is not None


def test_combined_mode_sharpens_after_exact_resize(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.png"
    output = tmp_path / "result.png"
    Image.new("RGB", (64, 32), "#507090").save(source)
    calls: list[tuple[str, tuple[int, int]]] = []
    scales: list[float] = []
    resize_exact = pipeline.resize_exact
    sharpen_luminance = pipeline.sharpen_luminance

    def record_resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
        calls.append(("resize", size))
        return resize_exact(image, size)

    def record_sharpen(image: Image.Image, strength: int, *, detail_scale: float) -> Image.Image:
        calls.append(("sharpen", image.size))
        scales.append(detail_scale)
        return sharpen_luminance(image, strength, detail_scale=detail_scale)

    monkeypatch.setattr(pipeline, "resize_exact", record_resize)
    monkeypatch.setattr(pipeline, "sharpen_luminance", record_sharpen)

    process_image(
        source,
        output,
        tmp_path,
        source.name,
        JobSettings(target_edge=256, processing_mode=ProcessingMode.upscale, sharpen=20),
        ClassicalAdapter(),
        1_000_000,
        Event(),
        lambda _phase, _message, _progress: None,
    )

    assert calls == [("resize", (256, 128)), ("sharpen", (256, 128))]
    # The sharpener is told how far the resize stretched the image, because that
    # is the width of the softness it has to undo.
    assert scales == [4.0]


class FakeSelfResizingAdapter(FakeNeuralAdapter):
    """An engine that reaches the target itself, as the ComfyUI graphs do.

    Its model stage runs at a fixed factor and its own resize covers whatever is
    left, so the file it returns is the target size while its detail is not.
    """

    id = "fake-self-resizing"

    def __init__(self, model_scale: int = 4) -> None:
        super().__init__(max_passes=1, native_scales=(1,))
        self.model_scale = model_scale

    def enhance(self, request: ModelRequest, cancel: Event, progress) -> ModelResult:
        with Image.open(request.source_path) as opened:
            opened.load()
            detail_width = opened.width * self.model_scale
            enhanced = opened.resize(
                (detail_width, opened.height * self.model_scale), Image.NEAREST
            ).resize((request.target_width, request.target_height), Image.LANCZOS)
        enhanced.save(request.output_path, format="PNG")
        return ModelResult(
            output_path=request.output_path,
            engine_id=self.id,
            detail_width=detail_width,
        )


def test_an_engine_that_resizes_itself_still_sizes_the_sharpener(tmp_path: Path, monkeypatch):
    """The stretch inside the engine is invisible in the file it returns.

    A 64px source at a 512px target is 8x, of which the model does 4x and the
    engine's own resize does the remaining 2x. Measuring the returned file would
    report no enlargement at all and run the filter at its finest radii, which is
    the softness it exists to undo.
    """
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 32), "#507090").save(source)
    scales: list[float] = []
    sharpen_luminance = pipeline.sharpen_luminance

    def record_sharpen(image: Image.Image, strength: int, *, detail_scale: float) -> Image.Image:
        scales.append(detail_scale)
        return sharpen_luminance(image, strength, detail_scale=detail_scale)

    monkeypatch.setattr(pipeline, "sharpen_luminance", record_sharpen)

    result = process_image(
        source,
        tmp_path / "result.png",
        tmp_path,
        source.name,
        JobSettings(target_edge=512, processing_mode=ProcessingMode.illustration, sharpen=35),
        FakeSelfResizingAdapter(),
        1_000_000,
        Event(),
        lambda _phase, _message, _progress: None,
    )

    assert (result.result.width, result.result.height) == (512, 256)
    assert scales == [2.0]


def test_a_source_the_engine_overshoots_asks_for_no_extra_sharpening(tmp_path, monkeypatch):
    """The same report must not turn a reduction into a claimed enlargement."""
    source = tmp_path / "source.png"
    Image.new("RGB", (400, 200), "#507090").save(source)
    scales: list[float] = []
    sharpen_luminance = pipeline.sharpen_luminance

    def record_sharpen(image: Image.Image, strength: int, *, detail_scale: float) -> Image.Image:
        scales.append(detail_scale)
        return sharpen_luminance(image, strength, detail_scale=detail_scale)

    monkeypatch.setattr(pipeline, "sharpen_luminance", record_sharpen)

    process_image(
        source,
        tmp_path / "result.png",
        tmp_path,
        source.name,
        JobSettings(target_edge=512, processing_mode=ProcessingMode.illustration, sharpen=35),
        FakeSelfResizingAdapter(),
        1_000_000,
        Event(),
        lambda _phase, _message, _progress: None,
    )

    # The model reached 1600px and the graph reduced to 512, which leaves no
    # resize softness behind.
    assert scales == [1.0]


def test_sharpen_only_reports_no_enlargement_to_the_sharpener(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 32), "#507090").save(source)
    scales: list[float] = []
    sharpen_luminance = pipeline.sharpen_luminance

    def record_sharpen(image: Image.Image, strength: int, *, detail_scale: float) -> Image.Image:
        scales.append(detail_scale)
        return sharpen_luminance(image, strength, detail_scale=detail_scale)

    monkeypatch.setattr(pipeline, "sharpen_luminance", record_sharpen)

    process_image(
        source,
        tmp_path / "result.png",
        tmp_path,
        source.name,
        JobSettings(
            target_edge=7680,
            processing_mode=ProcessingMode.sharpen_only,
            sharpen=45,
        ),
        ClassicalAdapter(),
        1_000_000,
        Event(),
        lambda _phase, _message, _progress: None,
    )

    assert scales == [1.0]


def _source(tmp_path: Path, size: tuple[int, int]) -> Path:
    path = tmp_path / "source.png"
    image = Image.new("RGB", size, "#3a5f7d")
    image.paste((230, 220, 200), (2, 2, size[0] // 2, size[1] // 2))
    image.save(path)
    return path


def test_large_factor_is_covered_by_chained_neural_passes(tmp_path: Path):
    """A 64x36 source needs 60x; one 4x pass would leave 15x to Lanczos."""
    source = _source(tmp_path, (64, 36))
    adapter = FakeNeuralAdapter(max_passes=3)

    result = process_image(
        source,
        tmp_path / "result.png",
        tmp_path,
        source.name,
        JobSettings(target_edge=3840, processing_mode=ProcessingMode.upscale, sharpen=0),
        adapter,
        10_000_000,
        Event(),
        lambda *_: None,
    )

    assert [scale for scale, _ in adapter.calls] == [4, 4, 4]
    # Each pass must consume the previous pass's output, not the original.
    assert [size for _, size in adapter.calls] == [(64, 36), (256, 144), (1024, 576)]
    assert result.result.neural_passes == [4, 4, 4]
    assert (result.result.width, result.result.height) == (3840, 2160)


def test_engine_is_never_asked_for_a_scale_it_cannot_produce(tmp_path: Path):
    """A 4x-only engine must be driven at 4x even when 2x would fit the target."""
    source = _source(tmp_path, (400, 225))  # 9.6x to 4K; a 2x tail would fit
    adapter = FakeNeuralAdapter(native_scales=(4,))

    result = process_image(
        source,
        tmp_path / "result.png",
        tmp_path,
        source.name,
        JobSettings(target_edge=3840, processing_mode=ProcessingMode.upscale, sharpen=0),
        adapter,
        50_000_000,
        Event(),
        lambda *_: None,
    )

    assert {scale for scale, _ in adapter.calls} == {4}
    assert result.result.neural_passes == [4, 4]
    assert (result.result.width, result.result.height) == (3840, 2160)


def test_pass_budget_is_capped_by_the_engine(tmp_path: Path):
    source = _source(tmp_path, (64, 36))
    adapter = FakeNeuralAdapter(max_passes=1)

    result = process_image(
        source,
        tmp_path / "result.png",
        tmp_path,
        source.name,
        JobSettings(target_edge=3840, processing_mode=ProcessingMode.upscale, sharpen=0),
        adapter,
        10_000_000,
        Event(),
        lambda *_: None,
    )

    assert result.result.neural_passes == [4]
    # The remainder still has to land on the exact requested target.
    assert (result.result.width, result.result.height) == (3840, 2160)


def test_single_pass_reports_no_chaining_warning(tmp_path: Path):
    source = _source(tmp_path, (1000, 1000))
    adapter = FakeNeuralAdapter()

    result = process_image(
        source,
        tmp_path / "result.png",
        tmp_path,
        source.name,
        JobSettings(target_edge=3840, processing_mode=ProcessingMode.upscale, sharpen=0),
        adapter,
        50_000_000,
        Event(),
        lambda *_: None,
    )

    assert result.result.neural_passes == [4]
    assert not any("chained" in warning for warning in result.result.warnings)


def test_chained_progress_never_runs_backwards(tmp_path: Path):
    source = _source(tmp_path, (64, 36))
    fractions: list[float] = []

    process_image(
        source,
        tmp_path / "result.png",
        tmp_path,
        source.name,
        JobSettings(target_edge=3840, processing_mode=ProcessingMode.upscale, sharpen=0),
        FakeNeuralAdapter(),
        10_000_000,
        Event(),
        lambda phase, _message, fraction: (
            fractions.append(fraction) if phase == "enhancing" and fraction is not None else None
        ),
    )

    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(1.0)


def test_an_already_large_source_skips_an_engine_that_only_enlarges(tmp_path: Path):
    """Enlarging past the target only to shrink again would cost time and detail."""
    source = _source(tmp_path, (5000, 2812))
    adapter = FakeNeuralAdapter()

    result = process_image(
        source,
        tmp_path / "result.png",
        tmp_path,
        source.name,
        JobSettings(target_edge=3840, processing_mode=ProcessingMode.upscale, sharpen=0),
        adapter,
        30_000_000,
        Event(),
        lambda *_: None,
    )

    assert adapter.calls == []
    assert result.result.neural_passes == []
    assert any("already meets the target" in warning for warning in result.result.warnings)
    assert (result.result.width, result.result.height) == (3840, 2160)
