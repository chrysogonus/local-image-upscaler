from __future__ import annotations

import asyncio
import json
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from starlette.background import BackgroundTask

from upscaler import __version__
from upscaler.devices import platform_capabilities
from upscaler.jobs import JobManager, JobQueueFull
from upscaler.resource_policy import POLICY_VERSION, HardwarePolicyError
from upscaler.schemas import (
    TERMINAL_STATES,
    Capabilities,
    HardwarePolicyInfo,
    JobSettings,
    JobSnapshot,
    ProcessingMode,
)

router = APIRouter(prefix="/api/v1")


def _manager(request: Request) -> JobManager:
    return request.app.state.job_manager


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/capabilities", response_model=Capabilities)
async def capabilities(request: Request) -> Capabilities:
    manager = _manager(request)
    return Capabilities(
        version=__version__,
        modes=manager.models.capabilities(),
        workflows=manager.models.workflows(),
        max_upload_bytes=manager.config.max_upload_bytes,
        max_input_pixels=manager.config.max_input_pixels,
        platform=platform_capabilities(),
        hardware=manager.models.hardware_reports(),
        hardware_policy=HardwarePolicyInfo(
            mode=manager.config.hardware_policy,
            version=POLICY_VERSION,
            ram_reserve_mib=manager.config.ram_reserve_mib,
            vram_reserve_mib=manager.config.vram_reserve_mib,
        ),
        excluded_features=manager.models.excluded_features(),
    )


@router.post("/jobs", response_model=JobSnapshot, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    request: Request,
    file: Annotated[UploadFile, File()],
    target_edge: Annotated[int, Form()] = 3840,
    processing_mode: Annotated[ProcessingMode, Form()] = ProcessingMode.upscale,
    sharpen: Annotated[int, Form()] = 15,
    tile_size: Annotated[int, Form()] = 0,
    tta: Annotated[bool, Form()] = False,
    restore_large: Annotated[bool, Form()] = False,
    max_neural_passes: Annotated[int, Form()] = 3,
    workflow: Annotated[str | None, Form()] = None,
) -> JobSnapshot:
    try:
        settings = JobSettings(
            target_edge=target_edge,
            processing_mode=processing_mode,
            sharpen=sharpen,
            tile_size=tile_size,
            tta=tta,
            restore_large=restore_large,
            max_neural_passes=max_neural_passes,
            # An empty form field is the browser saying "not set": pydantic
            # would otherwise reject it for a mode that forbids the field.
            workflow=workflow or None,
        )
        return await _manager(request).create(file, settings)
    except JobQueueFull as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except HardwarePolicyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/jobs/{job_id}", response_model=JobSnapshot)
async def get_job(job_id: str, request: Request) -> JobSnapshot:
    try:
        return _manager(request).get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@router.delete("/jobs/{job_id}", response_model=JobSnapshot, status_code=status.HTTP_202_ACCEPTED)
async def delete_job(job_id: str, request: Request) -> JobSnapshot:
    try:
        return _manager(request).delete(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    manager = _manager(request)
    try:
        manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    async def events():
        last_revision = -1
        heartbeat = 0
        while not await request.is_disconnected():
            try:
                snapshot = manager.get(job_id)
            except KeyError:
                yield "event: removed\ndata: {}\n\n"
                return
            if snapshot.revision != last_revision:
                payload = json.dumps(snapshot.model_dump(mode="json"), separators=(",", ":"))
                yield f"event: job\ndata: {payload}\n\n"
                last_revision = snapshot.revision
                heartbeat = 0
            elif heartbeat >= 60:
                yield ": keep-alive\n\n"
                heartbeat = 0
            if snapshot.state in TERMINAL_STATES:
                return
            heartbeat += 1
            await asyncio.sleep(0.25)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/jobs/{job_id}/result")
async def job_result(job_id: str, request: Request, download: bool = False) -> StreamingResponse:
    manager = _manager(request)
    try:
        path, result = manager.result_path(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail="Result is not ready") from exc

    async def chunks():
        with path.open("rb") as source:
            while data := source.read(1024 * 1024):
                yield data
                await asyncio.sleep(0)

    disposition = "attachment" if download else "inline"
    headers = {
        "Content-Length": str(path.stat().st_size),
        "Content-Disposition": f"{disposition}; filename*=UTF-8''{quote(result.filename)}",
    }
    return StreamingResponse(
        chunks(),
        media_type="image/png",
        headers=headers,
        background=BackgroundTask(manager.delete, job_id) if download else None,
    )
