from __future__ import annotations

import asyncio
import re
import shutil
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile

from upscaler.config import AppConfig
from upscaler.imaging import inspect_input_dimensions, process_image
from upscaler.models import ModelRegistry
from upscaler.models.base import ProcessingCancelled
from upscaler.models.registry import ResolvedJobPlan
from upscaler.schemas import (
    TERMINAL_STATES,
    JobSettings,
    JobSnapshot,
    JobState,
    ResultInfo,
    SourceInfo,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def safe_filename(value: str) -> str:
    name = Path(value).name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    return cleaned[:160] or "image"


@dataclass(slots=True)
class _Job:
    id: str
    settings: JobSettings
    original_filename: str
    workdir: Path
    input_path: Path
    output_path: Path
    plan: ResolvedJobPlan
    state: JobState = JobState.queued
    phase: str = "queued"
    message: str = "Waiting for the local processor"
    progress: float | None = 0.0
    source: SourceInfo | None = None
    result: ResultInfo | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    revision: int = 0
    cancel: threading.Event = field(default_factory=threading.Event)
    task: asyncio.Task[None] | None = None
    delete_when_done: bool = False

    def snapshot(self) -> JobSnapshot:
        return JobSnapshot(
            id=self.id,
            state=self.state,
            phase=self.phase,
            message=self.message,
            progress=self.progress,
            settings=self.settings,
            source=self.source,
            result=self.result,
            error=self.error,
            created_at=self.created_at,
            updated_at=self.updated_at,
            revision=self.revision,
        )


class JobManager:
    def __init__(self, config: AppConfig, models: ModelRegistry) -> None:
        self.config = config
        self.models = models
        self.config.work_root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.RLock()
        self._semaphore = asyncio.Semaphore(config.max_jobs)
        self._closed = False
        self._sweeper = asyncio.create_task(self._sweep_loop(), name="upscaler-job-sweeper")

    async def create(self, upload: UploadFile, settings: JobSettings) -> JobSnapshot:
        if self._closed:
            raise RuntimeError("job manager is closed")
        self.sweep_expired()
        # Resolved here rather than mid-job so an unavailable mode is refused up
        # front, instead of after the user waits through an upload.
        preliminary_plan = self.models.resolve_job(settings)

        job_id = str(uuid.uuid4())
        workdir = self.config.work_root / job_id
        workdir.mkdir(mode=0o700)
        job = _Job(
            id=job_id,
            settings=settings,
            original_filename=safe_filename(upload.filename or "image"),
            workdir=workdir,
            input_path=workdir / "source.upload",
            output_path=workdir / "result.png",
            plan=preliminary_plan,
        )

        try:
            await self._receive(upload, job.input_path)
            width, height = inspect_input_dimensions(job.input_path, self.config.max_input_pixels)
            # Totals drive stable UI visibility, but current free memory is
            # deliberately re-read here. The resulting plan is never resolved
            # again, so a queued job cannot silently switch engine or tile.
            job.plan = self.models.resolve_job(
                settings,
                width=width,
                height=height,
            )
        except Exception:
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        finally:
            await upload.close()

        with self._lock:
            self._jobs[job.id] = job
            job.task = asyncio.create_task(self._execute(job.id), name=f"upscale-{job.id}")
            return job.snapshot()

    async def _receive(self, upload: UploadFile, destination: Path) -> None:
        """Stream an upload to disk, enforcing the configured size limit."""
        total = 0
        with destination.open("xb") as handle:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > self.config.max_upload_bytes:
                    limit_mib = self.config.max_upload_bytes // (1024 * 1024)
                    raise ValueError(f"Upload exceeds the {limit_mib} MiB limit.")
                handle.write(chunk)
        if total == 0:
            raise ValueError("The uploaded file is empty.")

    def get(self, job_id: str) -> JobSnapshot:
        with self._lock:
            return self._get_job(job_id).snapshot()

    def result_path(self, job_id: str) -> tuple[Path, ResultInfo]:
        with self._lock:
            job = self._get_job(job_id)
            if job.state != JobState.completed or not job.result or not job.output_path.is_file():
                raise ValueError("result is not ready")
            return job.output_path, job.result

    def cancel(self, job_id: str, *, delete: bool = False) -> JobSnapshot:
        remove_now = False
        with self._lock:
            job = self._get_job(job_id)
            if job.state in TERMINAL_STATES:
                remove_now = delete
            else:
                job.cancel.set()
                job.delete_when_done = job.delete_when_done or delete
                self._mutate(
                    job,
                    state=JobState.cancelling,
                    phase="cancelling",
                    message="Stopping safely after the current operation",
                    progress=None,
                )
            snapshot = job.snapshot()
        if remove_now:
            self._remove(job_id)
        return snapshot

    def delete(self, job_id: str) -> JobSnapshot:
        with self._lock:
            self._get_job(job_id)
        return self.cancel(job_id, delete=True)

    async def _execute(self, job_id: str) -> None:
        try:
            async with self._semaphore:
                with self._lock:
                    job = self._get_job(job_id)
                    if job.cancel.is_set():
                        raise ProcessingCancelled("processing was cancelled")
                    plan = job.plan
                    adapter = plan.adapter
                    resolved_settings = job.settings.model_copy(
                        update={
                            "tile_size": plan.tile_size,
                            "workflow": plan.workflow_id,
                        }
                    )

                def report(phase: str, message: str, progress: float | None) -> None:
                    state = JobState(phase)
                    with self._lock:
                        current = self._jobs.get(job_id)
                        if current and current.state not in TERMINAL_STATES:
                            self._mutate(
                                current,
                                state=state,
                                phase=phase,
                                message=message,
                                progress=progress,
                            )

                result = await self._run_worker(
                    lambda: process_image(
                        job.input_path,
                        job.output_path,
                        job.workdir,
                        job.original_filename,
                        resolved_settings,
                        adapter,
                        self.config.max_input_pixels,
                        job.cancel,
                        report,
                        resolved_tile_size=plan.tile_size,
                    )
                )
                with self._lock:
                    current = self._jobs.get(job_id)
                    if current:
                        current.source = result.source
                        current.result = result.result
                        self._mutate(
                            current,
                            state=JobState.completed,
                            phase="completed",
                            message="Full-resolution result is ready",
                            progress=1.0,
                        )
        except ProcessingCancelled:
            with self._lock:
                current = self._jobs.get(job_id)
                if current:
                    self._mutate(
                        current,
                        state=JobState.cancelled,
                        phase="cancelled",
                        message="Processing was cancelled",
                        progress=None,
                    )
        except Exception as exc:
            with self._lock:
                current = self._jobs.get(job_id)
                if current:
                    self._mutate(
                        current,
                        state=JobState.failed,
                        phase="failed",
                        message="Processing failed",
                        progress=None,
                        error=str(exc),
                    )
        finally:
            with self._lock:
                current = self._jobs.get(job_id)
                delete = bool(current and current.delete_when_done)
                discard_workspace = bool(
                    current and current.state in {JobState.cancelled, JobState.failed}
                )
            if delete:
                self._remove(job_id)
            elif discard_workspace:
                shutil.rmtree(job.workdir, ignore_errors=True)

    async def _run_worker(self, operation: Callable[[], object]):
        """Run blocking imaging without relying on the server's AnyIO worker pool."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[object] = loop.create_future()

        def resolve_result(value: object) -> None:
            if not future.done():
                future.set_result(value)

        def resolve_error(error: BaseException) -> None:
            if not future.done():
                future.set_exception(error)

        def run() -> None:
            try:
                value = operation()
            except BaseException as exc:
                loop.call_soon_threadsafe(resolve_error, exc)
            else:
                loop.call_soon_threadsafe(resolve_result, value)

        threading.Thread(target=run, name="upscaler-image-worker", daemon=True).start()
        return await future

    def _get_job(self, job_id: str) -> _Job:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise KeyError("job not found") from exc

    def _mutate(
        self,
        job: _Job,
        *,
        state: JobState,
        phase: str,
        message: str,
        progress: float | None,
        error: str | None = None,
    ) -> None:
        job.state = state
        job.phase = phase
        job.message = message
        job.progress = progress
        job.error = error
        job.updated_at = _now()
        job.revision += 1

    def _remove(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job:
            shutil.rmtree(job.workdir, ignore_errors=True)

    def sweep_expired(self) -> None:
        cutoff = _now().timestamp() - self.config.job_retention_seconds
        with self._lock:
            expired = [
                job.id
                for job in self._jobs.values()
                if job.state in TERMINAL_STATES and job.updated_at.timestamp() < cutoff
            ]
        for job_id in expired:
            self._remove(job_id)

    async def _sweep_loop(self) -> None:
        interval = min(60, max(10, self.config.job_retention_seconds // 2))
        try:
            while True:
                await asyncio.sleep(interval)
                self.sweep_expired()
        except asyncio.CancelledError:
            return

    async def close(self) -> None:
        self._closed = True
        self._sweeper.cancel()
        await asyncio.gather(self._sweeper, return_exceptions=True)
        with self._lock:
            jobs = list(self._jobs.values())
            for job in jobs:
                job.cancel.set()
            tasks = [job.task for job in jobs if job.task and not job.task.done()]
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10)
            except asyncio.TimeoutError:
                for task in tasks:
                    task.cancel()
        for job in jobs:
            self._remove(job.id)
