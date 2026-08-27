from __future__ import annotations

import shutil
from threading import Event

from upscaler.models.base import ModelRequest, ModelResult, ProcessingCancelled, ProgressCallback


class ClassicalAdapter:
    id = "classical"
    name = "Faithful resample"
    neural = False
    generative = False
    max_passes = 1
    # Never planned for: enlargement here is the pipeline's exact resize.
    native_scales = ()
    # Not an inference engine: there is nothing to average.
    supports_tta = False
    license = "Pillow (MIT-CMU)"

    @property
    def available(self) -> bool:
        return True

    @property
    def unavailable_reason(self) -> None:
        return None

    @property
    def device(self) -> str:
        return "CPU"

    def enhance(
        self,
        request: ModelRequest,
        cancel: Event,
        progress: ProgressCallback,
    ) -> ModelResult:
        if cancel.is_set():
            raise ProcessingCancelled("processing was cancelled")
        progress("enhancing", "Preparing a faithful high-quality resample", None)
        shutil.copyfile(request.source_path, request.output_path)
        return ModelResult(output_path=request.output_path, engine_id=self.id)
