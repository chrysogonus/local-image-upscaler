from pathlib import Path
from threading import Event

import numpy as np
import pytest
from PIL import Image

from upscaler.imaging.pipeline import process_image
from upscaler.models import realesrgan_ncnn
from upscaler.models.realesrgan_ncnn import RealEsrganNcnnAdapter
from upscaler.schemas import JobSettings, ProcessingMode


def test_adapter_detects_runtime_installed_after_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    binary = tmp_path / "runtime" / "realesrgan-ncnn-vulkan"

    def find_configured(configured: Path | None) -> Path | None:
        return configured if configured and configured.is_file() else None

    monkeypatch.setattr(realesrgan_ncnn, "find_binary", find_configured)
    adapter = RealEsrganNcnnAdapter(binary)
    assert not adapter.available

    binary.parent.mkdir()
    binary.touch()
    models = binary.parent / "models"
    models.mkdir()
    for name in realesrgan_ncnn.REQUIRED_MODEL_FILES:
        (models / name).touch()

    assert adapter.available
    assert adapter.binary == binary


def test_realesrgan_runtime_smoke(tmp_path: Path):
    adapter = RealEsrganNcnnAdapter()
    if not adapter.available:
        pytest.skip("optional Real-ESRGAN runtime is not installed")

    source = tmp_path / "source.png"
    output = tmp_path / "result.png"
    Image.new("RGB", (32, 16), (38, 91, 160)).save(source)

    result = process_image(
        source,
        output,
        tmp_path,
        source.name,
        JobSettings(
            target_edge=256,
            processing_mode=ProcessingMode.upscale,
            tile_size=32,
            sharpen=0,
        ),
        adapter,
        1_000_000,
        Event(),
        lambda _phase, _message, _progress: None,
    )

    assert result.result.engine.startswith("realesrgan:")
    with Image.open(output) as image:
        assert image.size == (256, 128)
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)

    # A flat field is a small but useful real-model regression fixture: a broken
    # channel order, range conversion, tile boundary, or incompatible runtime
    # turns it dark, shifts its colour, or invents spatial texture. The bounds
    # allow the model's restrained processing while rejecting those failures.
    expected = np.array((38, 91, 160), dtype=np.float32)
    assert np.abs(pixels.mean(axis=(0, 1)) - expected).max() < 25
    assert pixels.std(axis=(0, 1)).max() < 20
