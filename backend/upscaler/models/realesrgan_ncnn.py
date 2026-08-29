from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from threading import Event

from upscaler.models.base import (
    ModelExecutionError,
    ModelRequest,
    ModelResult,
    ProcessingCancelled,
    ProgressCallback,
)

# The general-purpose photo model. The archive also bundles an illustration
# variant, but one well-chosen model beats a choice the user has no way to
# evaluate before running the job, so only this one is required or used.
MODEL_NAME = "realesrgan-x4plus"
REQUIRED_MODEL_FILES = (f"{MODEL_NAME}.bin", f"{MODEL_NAME}.param")


def find_binary(configured: Path | None) -> Path | None:
    candidates: list[Path] = []
    if configured:
        candidates.append(configured)
    for name in ("realesrgan-ncnn-vulkan", "realesrgan-ncnn-vulkan.exe"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    repository_runtime = (
        Path(__file__).resolve().parents[3]
        / ".upscaler"
        / "realesrgan-ncnn-vulkan"
        / "realesrgan-ncnn-vulkan"
    )
    candidates.append(repository_runtime)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


class RealEsrganNcnnAdapter:
    id = "realesrgan"
    name = "Real-ESRGAN"
    neural = True
    generative = False
    max_passes = 3
    # 4x only. Both bundled models are 4x, and the runtime's -s 2 and -s 3 paths
    # are meant for the animevideov3 weights that ship real 2x and 3x variants:
    # against x4plus they return a correctly sized but wrongly cropped image.
    # A smaller factor is reached by enlarging 4x and letting the pipeline's one
    # exact resize bring it down.
    native_scales = (4,)
    supports_tta = True
    license = "BSD-3-Clause"

    def __init__(self, configured_binary: Path | None = None) -> None:
        # Resolve on demand so a runtime installed while the local service is
        # running becomes available without restarting the backend.
        self._configured_binary = configured_binary

    @property
    def binary(self) -> Path | None:
        return find_binary(self._configured_binary)

    @property
    def available(self) -> bool:
        return bool(
            self.binary
            and all(
                (self.binary.parent / "models" / name).is_file() for name in REQUIRED_MODEL_FILES
            )
        )

    @property
    def unavailable_reason(self) -> str | None:
        if self.available:
            return None
        if self.binary:
            return "Real-ESRGAN was found, but its required model files are missing."
        return (
            "Real-ESRGAN NCNN/Vulkan is not installed. The released image does not "
            "carry the Vulkan binary, so this engine needs one mounted into the "
            "container and named by UPSCALER_REALESRGAN_BIN."
        )

    @property
    def device(self) -> str:
        return "Vulkan (GPU or software)" if self.available else "Unavailable"

    def enhance(
        self,
        request: ModelRequest,
        cancel: Event,
        progress: ProgressCallback,
    ) -> ModelResult:
        binary = self.binary
        if not binary or not all(
            (binary.parent / "models" / name).is_file() for name in REQUIRED_MODEL_FILES
        ):
            raise ModelExecutionError(self.unavailable_reason or "Real-ESRGAN is unavailable")
        if cancel.is_set():
            raise ProcessingCancelled("processing was cancelled")

        command = [
            str(binary),
            "-i",
            str(request.source_path),
            "-o",
            str(request.output_path),
            "-n",
            MODEL_NAME,
            "-s",
            str(request.native_scale),
            "-t",
            str(request.tile_size),
            "-f",
            "png",
        ]
        if request.tta:
            command.append("-x")

        progress("loading_model", f"Loading {self.name} ({MODEL_NAME})", None)
        with tempfile.TemporaryFile() as log:
            try:
                process = subprocess.Popen(  # noqa: S603 - fixed executable and validated arguments
                    command,
                    cwd=binary.parent,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                )
            except OSError as exc:
                raise ModelExecutionError(f"Could not start Real-ESRGAN: {exc}") from exc

            progress("enhancing", "Restoring detail with tiled neural inference", None)
            while process.poll() is None:
                if cancel.wait(0.2):
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
                    raise ProcessingCancelled("processing was cancelled")
                time.sleep(0.05)

            if process.returncode != 0:
                log.seek(0)
                details = log.read().decode("utf-8", errors="replace")[-4000:].strip()
                suffix = f": {details}" if details else ""
                raise ModelExecutionError(
                    f"Real-ESRGAN exited with status {process.returncode}{suffix}"
                )

        if not request.output_path.is_file():
            raise ModelExecutionError("Real-ESRGAN finished without creating an output image")
        return ModelResult(
            output_path=request.output_path,
            engine_id=f"{self.id}:{MODEL_NAME}",
            warnings=("Inspect fine textures at 1:1 for neural or tile-generated artifacts.",),
        )
