"""Real-ESRGAN through PyTorch on a CUDA accelerator.

Runs in-process rather than shelling out to the NCNN binary, which lets tiles be
cancelled individually and progress be reported for real instead of estimated.
Availability is probed, never assumed: without torch, without a CUDA device, or
without weights this adapter reports itself unavailable and the registry falls
through to the NCNN or classical engines.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Event, Lock
from typing import Any

from PIL import Image

from upscaler.models.base import (
    ModelExecutionError,
    ModelRequest,
    ModelResult,
    ProcessingCancelled,
    ProgressCallback,
)
from upscaler.models.tiled import resolve_tiling, run_tiled, self_ensemble

NATIVE_SCALE = 4
DEFAULT_TILE = 512

# The general-purpose photo model, and the only one this engine loads. One
# well-chosen model beats a choice the user cannot evaluate before running.
WEIGHTS_FILENAME = "RealESRGAN_x4plus.pth"
REQUIRED_WEIGHTS = (WEIGHTS_FILENAME,)


def weights_dir() -> Path:
    configured = os.getenv("UPSCALER_REALESRGAN_WEIGHTS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / ".upscaler" / "realesrgan-torch"


def probe_cuda() -> tuple[bool, str, str | None]:
    """Return (usable, device_label, reason_if_unusable).

    A successful ``torch.cuda.is_available()`` is not proof the wheel carries
    kernels for this GPU, so a tiny convolution is executed. On a very new
    architecture the mismatch only surfaces when a kernel actually launches.
    """
    try:
        import torch
    except ImportError:
        return (
            False,
            "Unavailable",
            ("PyTorch is not installed. Install the CUDA extra to enable GPU restoration."),
        )
    try:
        if not torch.cuda.is_available():
            return False, "Unavailable", "PyTorch is installed but reports no usable CUDA device."
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        probe = torch.zeros(1, 3, 8, 8, device="cuda")
        torch.nn.functional.conv2d(probe, torch.zeros(3, 3, 3, 3, device="cuda"), padding=1)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - any CUDA fault means "not usable here"
        return (
            False,
            "Unavailable",
            (
                f"A CUDA device was found but PyTorch could not run on it: {exc}. "
                "This usually means the installed build lacks kernels for this GPU."
            ),
        )
    return True, f"CUDA ({name}, sm_{major}{minor})", None


def _resolve_dtype(torch: Any) -> Any:
    requested = os.getenv("UPSCALER_CUDA_PRECISION", "auto").strip().lower()
    if requested == "fp32":
        return torch.float32
    if requested == "fp16":
        return torch.float16
    if requested == "bf16":
        return torch.bfloat16
    # bfloat16 carries float32's exponent range, so it avoids the overflow that
    # float16 can hit in the dense blocks while still using tensor cores.
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32


class RealEsrganCudaAdapter:
    id = "realesrgan-cuda"
    name = "Real-ESRGAN (CUDA)"
    neural = True
    generative = False
    max_passes = 3
    # The network is 4x; this adapter resamples its own output down to a smaller
    # requested scale, so 2x and 3x are genuinely supported here.
    native_scales = (2, 3, 4)
    supports_tta = True
    license = "BSD-3-Clause"

    def __init__(self) -> None:
        self._probe: tuple[bool, str, str | None] | None = None
        self._models: dict[tuple[str, str], Any] = {}
        self._lock = Lock()

    # Probing initialises a CUDA context, so it is done once and cached. Weight
    # files are re-checked on every call so installing them does not require a
    # backend restart, matching the NCNN adapter's behaviour.
    def _cuda(self) -> tuple[bool, str, str | None]:
        if self._probe is None:
            self._probe = probe_cuda()
        return self._probe

    def _missing_weights(self) -> list[str]:
        directory = weights_dir()
        return [name for name in REQUIRED_WEIGHTS if not (directory / name).is_file()]

    @property
    def available(self) -> bool:
        usable, _, _ = self._cuda()
        return usable and not self._missing_weights()

    @property
    def unavailable_reason(self) -> str | None:
        usable, _, reason = self._cuda()
        if not usable:
            return reason
        missing = self._missing_weights()
        if missing:
            return (
                f"CUDA is ready but {', '.join(missing)} is missing. "
                "Run `make setup-model-cuda` to download the weights."
            )
        return None

    @property
    def device(self) -> str:
        usable, label, _ = self._cuda()
        return label if usable else "Unavailable"

    def _generator(self, filename: str) -> Any:
        import torch

        from upscaler.models.rrdbnet import load_generator

        dtype = _resolve_dtype(torch)
        key = (filename, str(dtype))
        with self._lock:
            model = self._models.get(key)
            if model is None:
                path = weights_dir() / filename
                try:
                    model = load_generator(path, torch.device("cuda"), dtype)
                except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
                    raise ModelExecutionError(f"Could not load {filename}: {exc}") from exc
                self._models[key] = model
            return model

    def enhance(
        self,
        request: ModelRequest,
        cancel: Event,
        progress: ProgressCallback,
    ) -> ModelResult:
        if not self.available:
            raise ModelExecutionError(self.unavailable_reason or "The CUDA engine is unavailable")
        _check(cancel)

        import numpy as np
        import torch

        progress("loading_model", f"Loading {self.name} ({Path(WEIGHTS_FILENAME).stem})", None)
        model = self._generator(WEIGHTS_FILENAME)
        dtype = _resolve_dtype(torch)

        with Image.open(request.source_path) as opened:
            source = opened.convert("RGBA") if _has_alpha(opened) else opened.convert("RGB")
            source.load()
        alpha = source.getchannel("A") if source.mode == "RGBA" else None
        rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0

        height, width = rgb.shape[:2]
        tile_size, overlap = resolve_tiling(request.tile_size, width, height, default=DEFAULT_TILE)

        def infer(patch: Any) -> Any:
            tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.to(device="cuda", dtype=dtype, non_blocking=True)
            with torch.inference_mode():
                out = self_ensemble(model, tensor, torch) if request.tta else model(tensor)
                out = out.clamp_(0.0, 1.0).float()
            return out.squeeze(0).permute(1, 2, 0).cpu().numpy()

        try:
            accum = run_tiled(
                rgb,
                scale=NATIVE_SCALE,
                tile_size=tile_size,
                overlap=overlap,
                infer=infer,
                cancel=cancel,
                progress=progress,
                message="Restoring detail with tiled neural inference",
                np=np,
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise ModelExecutionError(
                "The GPU ran out of memory. Choose a smaller tile size and retry."
            ) from exc
        except ProcessingCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            raise ModelExecutionError(f"CUDA inference failed: {exc}") from exc

        enhanced = Image.fromarray((accum * 255.0 + 0.5).clip(0, 255).astype(np.uint8), "RGB")

        progress("finishing", "Assembling the enhanced image", None)
        if request.native_scale != NATIVE_SCALE:
            enhanced = enhanced.resize(
                (width * request.native_scale, height * request.native_scale),
                Image.Resampling.LANCZOS,
            )

        warnings = ["Inspect fine textures at 1:1 for neural or tile-generated artifacts."]
        if alpha is not None:
            enhanced = enhanced.convert("RGBA")
            enhanced.putalpha(alpha.resize(enhanced.size, Image.Resampling.LANCZOS))
            warnings.append(
                "Transparency was resampled rather than neurally restored; the model processes "
                "color channels only."
            )
        enhanced.save(request.output_path, format="PNG")
        return ModelResult(
            output_path=request.output_path,
            engine_id=f"{self.id}:{Path(WEIGHTS_FILENAME).stem}",
            warnings=tuple(warnings),
        )


def _check(cancel: Event) -> None:
    if cancel.is_set():
        raise ProcessingCancelled("processing was cancelled")


def _has_alpha(image: Image.Image) -> bool:
    return image.mode in {"RGBA", "LA", "PA"} or "transparency" in image.info
