from __future__ import annotations

from pathlib import Path

import pytest

from upscaler.__main__ import server_options
from upscaler.app import frontend_directory


def test_defaults_to_loopback(monkeypatch) -> None:
    """The default must stay loopback; only a container opts out explicitly."""
    monkeypatch.delenv("UPSCALER_HOST", raising=False)
    monkeypatch.delenv("UPSCALER_PORT", raising=False)
    assert server_options() == ("127.0.0.1", 8000)


def test_environment_overrides_host_and_port(monkeypatch) -> None:
    monkeypatch.setenv("UPSCALER_HOST", "0.0.0.0")
    monkeypatch.setenv("UPSCALER_PORT", "9001")
    assert server_options() == ("0.0.0.0", 9001)


def test_blank_host_falls_back_to_loopback(monkeypatch) -> None:
    monkeypatch.setenv("UPSCALER_HOST", "   ")
    monkeypatch.delenv("UPSCALER_PORT", raising=False)
    assert server_options() == ("127.0.0.1", 8000)


@pytest.mark.parametrize("value", ["not-a-port", "0", "65536", "-1"])
def test_invalid_port_is_rejected(monkeypatch, value: str) -> None:
    monkeypatch.delenv("UPSCALER_HOST", raising=False)
    monkeypatch.setenv("UPSCALER_PORT", value)
    with pytest.raises(ValueError, match="UPSCALER_PORT"):
        server_options()


def test_frontend_directory_defaults_to_the_source_layout(monkeypatch) -> None:
    monkeypatch.delenv("UPSCALER_FRONTEND_DIST", raising=False)
    assert frontend_directory().name == "dist"
    assert frontend_directory().parent.name == "frontend"


def test_frontend_directory_honours_the_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UPSCALER_FRONTEND_DIST", str(tmp_path))
    assert frontend_directory() == tmp_path.resolve()
