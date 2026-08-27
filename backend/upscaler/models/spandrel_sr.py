"""Transformer super-resolution through any spandrel-supported checkpoint.

Real-ESRGAN's RRDBNet is a 2018 convolutional generator, and the field moved to
transformers years ago: the NTIRE 2026 x4 report puts HAT, SwinIR and HMANet at
the foundation of every winning entry. This adapter is the way in. It drives
whatever checkpoint it is given through ``spandrel``, which already recognises
HAT, DAT, DRCT, SwinIR, ATD, SPAN, RGT and PLKSR, so a newer architecture is a
file in a directory rather than a code change.

The pinned default is SwinIR-L trained with the BSRGAN degradation pipeline,
and the degradation pipeline is the whole reason for that choice. The headline
transformer checkpoints - HAT-L, DRCT-L, the classical SwinIR variants - are
trained on bicubic downsampling only. Fed a real photograph, whose softness
comes from compression, sensor noise and resampling rather than from clean
bicubic, they sharpen the artifacts along with the detail and land behind
Real-ESRGAN on the images this application actually receives. Picking the
architecture that tops a benchmark while ignoring what it was trained to invert
is precisely the trap ``AGENTS.md`` warns about, so the benchmark leader is not
what ships here.

Any other spandrel-loadable checkpoint can be used instead by pointing
``UPSCALER_SR_MODEL`` at it; the model's own scale and architecture are then
read from the checkpoint and reported in the job result.
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
from upscaler.models.realesrgan_cuda import probe_cuda
from upscaler.models.tiled import resolve_tiling, run_tiled, self_ensemble

WEIGHTS_FILENAME = "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
DEFAULT_NAME = "SwinIR-L (real-world)"

# Transformer attention is quadratic in the tile area and SwinIR-L is deep, so
# this sits below the convolutional engine's 512. Larger tiles buy nothing:
# the receptive field is already far smaller than the tile.
DEFAULT_TILE = 256


def weights_dir() -> Path:
    configured = os.getenv("UPSCALER_SR_WEIGHTS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / ".upscaler" / "spandrel-sr"


def checkpoint_path() -> Path:
    """The checkpoint this adapter drives, pinned default or user override."""
    configured = os.getenv("UPSCALER_SR_MODEL")
    if configured:
        return Path(configured).expanduser().resolve()
    return weights_dir() / WEIGHTS_FILENAME


class SpandrelSrAdapter:
    id = "spandrel-sr"
    neural = True
    generative = False
    max_passes = 3
    # The pinned checkpoint is 4x. Like the CUDA engine, this adapter resamples
    # its own output down for a smaller request, so 2x and 3x are genuine here.
    native_scales = (2, 3, 4)
    supports_tta = True
    license = "Apache-2.0"

    def __init__(self) -> None:
        self._probe: tuple[bool, str, str | None] | None = None
        self._model: Any = None
        self._loaded_from: Path | None = None
        self._lock = Lock()

    @property
    def name(self) -> str:
        path = checkpoint_path()
        if path.name == WEIGHTS_FILENAME:
            return DEFAULT_NAME
        return f"{path.stem} (spandrel)"

    def _cuda(self) -> tuple[bool, str, str | None]:
        if self._probe is None:
            self._probe = probe_cuda()
        return self._probe

    def _runtime_reason(self) -> str | None:
        try:
            import spandrel  # noqa: F401
        except ImportError:
            return (
                "spandrel is not installed; it supplies the transformer architectures. "
                "Run `make setup-swinir`."
            )
        return None

    @property
    def available(self) -> bool:
        usable, _, _ = self._cuda()
        return usable and self._runtime_reason() is None and checkpoint_path().is_file()

    @property
    def unavailable_reason(self) -> str | None:
        usable, _, reason = self._cuda()
        if not usable:
            return reason
        runtime = self._runtime_reason()
        if runtime:
            return runtime
        path = checkpoint_path()
        if not path.is_file():
            return (
                f"The transformer runtime is ready but {path.name} is missing. "
                "Run `make setup-model-swinir` to download the weights."
            )
        return None

    @property
    def device(self) -> str:
        usable, label, _ = self._cuda()
        return label if usable else "Unavailable"

    def _load(self, path: Path) -> Any:
        from spandrel import ImageModelDescriptor, ModelLoader

        try:
            descriptor = ModelLoader().load_from_file(path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            raise ModelExecutionError(f"Could not load {path.name}: {exc}") from exc
        if not isinstance(descriptor, ImageModelDescriptor):
            raise ModelExecutionError(f"{path.name} is not a single-image model")
        if descriptor.purpose != "SR":
            raise ModelExecutionError(
                f"{path.name} loaded as a {descriptor.purpose} model, not an upscaler"
            )
        return descriptor

    def _descriptor(self) -> Any:
        import torch

        path = checkpoint_path()
        with self._lock:
            if self._model is None or self._loaded_from != path:
                descriptor = self._load(path)
                if descriptor.supports_bfloat16:
                    descriptor.to(torch.device("cuda"), torch.bfloat16).eval()
                    if not bfloat16_survives_a_tile(descriptor, torch):
                        # Reloaded rather than cast back up: bfloat16 has
                        # already dropped the mantissa bits, and widening those
                        # weights again would keep the loss without the speed.
                        descriptor = self._load(path)
                        descriptor.to(torch.device("cuda"), torch.float32).eval()
                else:
                    descriptor.to(torch.device("cuda"), torch.float32).eval()
                self._model = descriptor
                self._loaded_from = path
            return self._model

    def enhance(
        self,
        request: ModelRequest,
        cancel: Event,
        progress: ProgressCallback,
    ) -> ModelResult:
        if not self.available:
            raise ModelExecutionError(
                self.unavailable_reason or "The transformer engine is unavailable"
            )
        _check(cancel)

        import numpy as np
        import torch

        progress("loading_model", f"Loading {self.name}", None)
        descriptor = self._descriptor()
        model_scale = int(descriptor.scale)
        dtype = next(descriptor.model.parameters()).dtype

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
                # Through the descriptor, not the raw module: spandrel's [0, 1]
                # convention is the one these SR checkpoints were trained on,
                # and it pads the tile up to the window size the architecture
                # requires. That padding is why a transformer works here at all.
                if request.tta:
                    out = self_ensemble(descriptor, tensor, torch)
                else:
                    out = descriptor(tensor)
                out = out.clamp_(0.0, 1.0).float()
            return out.squeeze(0).permute(1, 2, 0).cpu().numpy()

        try:
            enhanced_array = run_tiled(
                rgb,
                scale=model_scale,
                tile_size=tile_size,
                overlap=overlap,
                infer=infer,
                cancel=cancel,
                progress=progress,
                message="Restoring detail with tiled transformer inference",
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
            raise ModelExecutionError(f"Transformer inference failed: {exc}") from exc

        enhanced = Image.fromarray(
            (enhanced_array * 255.0 + 0.5).clip(0, 255).astype(np.uint8), "RGB"
        )

        progress("finishing", "Assembling the enhanced image", None)
        if request.native_scale != model_scale:
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
            engine_id=f"{self.id}:{checkpoint_path().stem}",
            warnings=tuple(warnings),
        )


# Deliberately not a multiple of the 64-pixel window SwinIR was built around,
# and not square: the shapes this engine actually sends are arbitrary tiles.
PROBE_SHAPE = (40, 48)


def bfloat16_survives_a_tile(descriptor: Any, torch: Any) -> bool:
    """Whether this checkpoint really runs in bfloat16 at an arbitrary tile size.

    ``spandrel.ImageModelDescriptor.supports_bfloat16`` is a claim about the
    architecture, not a guarantee about every input. SwinIR is the case in
    point. It precomputes the shifted-window attention mask as a registered
    buffer, which ``.to(dtype)`` converts along with the weights, but for any
    input whose padded size differs from the one it was constructed with it
    rebuilds that mask on the fly - in float32, moved to the right device and
    no further. The mixed-dtype attention then dies with "expected scalar type
    Float but found BFloat16", and only for tiles that are not the model's own
    ``img_size``. A single tiny forward pass at a size nothing precomputed
    catches it before a job does.

    Failure means float32, never a crash: an engine that cannot answer this
    question is an engine we do not run in half precision.
    """
    probe = torch.zeros(
        (1, descriptor.input_channels, *PROBE_SHAPE),
        device="cuda",
        dtype=torch.bfloat16,
    )
    try:
        with torch.inference_mode():
            descriptor(probe)
    except Exception:  # noqa: BLE001 - any failure here means float32, not a crash
        return False
    return True


def _check(cancel: Event) -> None:
    if cancel.is_set():
        raise ProcessingCancelled("processing was cancelled")


def _has_alpha(image: Image.Image) -> bool:
    return image.mode in {"RGBA", "LA", "PA"} or "transparency" in image.info
