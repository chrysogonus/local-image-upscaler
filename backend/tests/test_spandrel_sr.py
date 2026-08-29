"""The transformer engine must be honest about being unavailable.

It cannot run in CI - no CUDA, no spandrel, no weights - so what is tested here
is the contract that matters when it cannot run anywhere: it imports without
torch, reports an actionable reason rather than raising, and refuses to process
instead of quietly handing back the input.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from upscaler.models.base import ModelExecutionError, ModelRequest
from upscaler.models.spandrel_sr import (
    PROBE_SHAPE,
    WEIGHTS_FILENAME,
    SpandrelSrAdapter,
    bfloat16_survives_a_tile,
    checkpoint_path,
    weights_dir,
)


def test_weights_dir_honours_the_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UPSCALER_SR_WEIGHTS_DIR", str(tmp_path))
    assert weights_dir() == tmp_path.resolve()
    assert checkpoint_path() == tmp_path.resolve() / WEIGHTS_FILENAME


def test_a_user_supplied_checkpoint_overrides_the_pinned_one(monkeypatch, tmp_path: Path) -> None:
    """Any spandrel-loadable architecture is a file, not a code change."""
    custom = tmp_path / "4xNomos8kSCHAT-L.pth"
    monkeypatch.setenv("UPSCALER_SR_MODEL", str(custom))
    assert checkpoint_path() == custom.resolve()
    assert SpandrelSrAdapter().name == "4xNomos8kSCHAT-L (spandrel)"


def test_the_pinned_checkpoint_reports_its_real_name(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("UPSCALER_SR_MODEL", raising=False)
    monkeypatch.setenv("UPSCALER_SR_WEIGHTS_DIR", str(tmp_path))
    assert SpandrelSrAdapter().name == "SwinIR-L (real-world)"


def test_reports_a_reason_instead_of_raising(monkeypatch) -> None:
    adapter = SpandrelSrAdapter()
    monkeypatch.setattr(adapter, "_probe", (False, "Unavailable", "no CUDA device here"))

    assert adapter.available is False
    assert adapter.device == "Unavailable"
    assert adapter.unavailable_reason == "no CUDA device here"


def test_missing_weights_are_distinguished_from_a_missing_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("UPSCALER_SR_MODEL", raising=False)
    monkeypatch.setenv("UPSCALER_SR_WEIGHTS_DIR", str(tmp_path))
    adapter = SpandrelSrAdapter()
    monkeypatch.setattr(adapter, "_probe", (True, "CUDA (Test GPU)", None))
    monkeypatch.setattr(adapter, "_runtime_reason", lambda: None)

    reason = adapter.unavailable_reason
    assert adapter.available is False
    assert reason is not None and "UPSCALER_FETCH_SWINIR" in reason

    (tmp_path / WEIGHTS_FILENAME).write_bytes(b"stub")
    assert adapter.available is True
    assert adapter.unavailable_reason is None


def test_a_missing_spandrel_names_its_own_setup_target(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UPSCALER_SR_WEIGHTS_DIR", str(tmp_path))
    (tmp_path / WEIGHTS_FILENAME).write_bytes(b"stub")
    adapter = SpandrelSrAdapter()
    monkeypatch.setattr(adapter, "_probe", (True, "CUDA (Test GPU)", None))
    monkeypatch.setattr(adapter, "_runtime_reason", lambda: "spandrel is not installed.")

    assert adapter.available is False
    assert adapter.unavailable_reason == "spandrel is not installed."


def test_enhance_refuses_when_unavailable(monkeypatch, tmp_path: Path) -> None:
    adapter = SpandrelSrAdapter()
    monkeypatch.setattr(adapter, "_probe", (False, "Unavailable", "no CUDA here"))

    request = ModelRequest(
        source_path=tmp_path / "in.png",
        output_path=tmp_path / "out.png",
        workspace=tmp_path,
        native_scale=4,
        target_width=1024,
        target_height=1024,
        tile_size=0,
    )
    with pytest.raises(ModelExecutionError, match="no CUDA here"):
        adapter.enhance(request, Event(), lambda *_: None)


def test_the_transformer_reconstructs_rather_than_generates() -> None:
    """Upscale's whole promise. A generative engine here would break it silently."""
    assert SpandrelSrAdapter.generative is False
    assert SpandrelSrAdapter.neural is True


class _FakeTorch:
    """Enough of torch to drive the bfloat16 probe without a GPU."""

    bfloat16 = "bfloat16"

    def __init__(self) -> None:
        self.requested: tuple[Any, ...] | None = None

    def zeros(self, shape: tuple[int, ...], *, device: str, dtype: str) -> tuple[Any, ...]:
        self.requested = (shape, device, dtype)
        return shape

    @staticmethod
    def inference_mode() -> AbstractContextManager[None]:
        return nullcontext()


class _FakeDescriptor:
    input_channels = 3

    def __init__(self, explode: Exception | None = None) -> None:
        self.explode = explode
        self.seen: tuple[int, ...] | None = None

    def __call__(self, probe: tuple[int, ...]) -> tuple[int, ...]:
        self.seen = probe
        if self.explode is not None:
            raise self.explode
        return probe


def test_a_checkpoint_that_survives_the_probe_keeps_bfloat16() -> None:
    torch = _FakeTorch()
    descriptor = _FakeDescriptor()

    assert bfloat16_survives_a_tile(descriptor, torch) is True
    assert torch.requested == ((1, 3, *PROBE_SHAPE), "cuda", "bfloat16")


def test_swinirs_rebuilt_attention_mask_demotes_the_engine_to_float32() -> None:
    """The exact failure a user hits: fine at the model's own size, dead at any other."""
    torch = _FakeTorch()
    descriptor = _FakeDescriptor(RuntimeError("expected scalar type Float but found BFloat16"))

    assert bfloat16_survives_a_tile(descriptor, torch) is False


def test_an_unexpected_probe_failure_also_demotes_rather_than_raising() -> None:
    assert bfloat16_survives_a_tile(_FakeDescriptor(ValueError("odd")), _FakeTorch()) is False


def test_the_probe_uses_a_shape_the_model_cannot_have_precomputed() -> None:
    """A square power-of-two probe would pass on SwinIR and hide the bug it exists to catch."""
    height, width = PROBE_SHAPE
    assert height != width
    assert height % 64 and width % 64
