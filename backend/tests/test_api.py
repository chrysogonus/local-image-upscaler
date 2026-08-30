import asyncio
import io
import threading
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from upscaler.app import create_app
from upscaler.config import AppConfig
from upscaler.hardware import HardwareSnapshot
from upscaler.jobs import manager as job_manager_module
from upscaler.jobs.manager import safe_filename
from upscaler.models.base import ProcessingCancelled


def png_bytes(size=(32, 16)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, "#3866a3").save(output, format="PNG")
    return output.getvalue()


async def wait_for_terminal(client: httpx.AsyncClient, job_id: str) -> dict:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        snapshot = (await client.get(f"/api/v1/jobs/{job_id}")).json()
        if snapshot["state"] in {"completed", "failed", "cancelled"}:
            return snapshot
        await asyncio.sleep(0.05)
    raise AssertionError("job did not finish")


async def wait_for_removal(client: httpx.AsyncClient, job_id: str) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if (await client.get(f"/api/v1/jobs/{job_id}")).status_code == 404:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("job was not removed")


def app_client(config: AppConfig):
    app = create_app(config)
    transport = httpx.ASGITransport(app=app)
    # Addressed as loopback rather than httpx's "testserver" convention: the
    # host allowlist is a security boundary, so the suite exercises it with a
    # name a real deployment actually uses.
    return app, httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1")


@pytest.mark.asyncio
async def test_capabilities_are_reported_per_mode(tmp_path: Path):
    config = AppConfig(work_root=tmp_path / "jobs")
    app, client = app_client(config)
    async with app.router.lifespan_context(app), client:
        payload = (await client.get("/api/v1/capabilities")).json()

        modes = {entry["mode"]: entry for entry in payload["modes"]}
        assert set(modes) == {"upscale", "illustration", "sharpen_only"}
        # Upscale falls back to the resampler, so it is usable on any machine.
        assert modes["upscale"]["available"] is True
        # Nothing here invents detail, and the reported flag has to say so.
        for entry in modes.values():
            assert entry["generative"] is False
        for entry in modes.values():
            assert entry["name"] and entry["description"]
            assert entry["available"] or entry["unavailable_reason"]
        assert payload["hardware_policy"]["mode"] == "safe"
        assert payload["hardware_policy"]["version"] == 1
        assert payload["hardware"][0]["scope"] == "backend"
        assert isinstance(payload["excluded_features"], list)


@pytest.mark.asyncio
async def test_an_unavailable_mode_is_refused_before_the_upload_is_processed(tmp_path: Path):
    """Unavailable means rejected up front, not a job that runs and returns something weaker."""
    config = AppConfig(work_root=tmp_path / "jobs")
    app, client = app_client(config)
    async with app.router.lifespan_context(app), client:
        modes = {
            entry["mode"]: entry
            for entry in (await client.get("/api/v1/capabilities")).json()["modes"]
        }
        if modes["illustration"]["available"]:
            pytest.skip("a ComfyUI with the illustration model is reachable from this machine")

        response = await client.post(
            "/api/v1/jobs",
            files={"file": ("small.png", png_bytes(), "image/png")},
            data={"target_edge": "256", "processing_mode": "illustration"},
        )

        assert response.status_code == 400
        assert response.json()["detail"]
        assert list((tmp_path / "jobs").iterdir()) == []


@pytest.mark.asyncio
async def test_upscale_job_lifecycle(tmp_path: Path):
    config = AppConfig(work_root=tmp_path / "jobs", max_input_pixels=1_000_000)
    app, client = app_client(config)
    async with app.router.lifespan_context(app), client:
        response = await client.post(
            "/api/v1/jobs",
            files={"file": ("small.png", png_bytes(), "image/png")},
            data={"target_edge": "256"},
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        snapshot = await wait_for_terminal(client, job_id)
        assert snapshot["state"] == "completed"
        assert snapshot["settings"]["processing_mode"] == "upscale"
        assert snapshot["result"]["processing_mode"] == "upscale"
        assert snapshot["result"]["width"] == 256
        assert snapshot["result"]["height"] == 128
        assert snapshot["result"]["filename"].endswith(".png")

        # The upload asked for automatic tiling, so the size the run resolved to
        # depends on the engine actually installed: zero where the engine tiles
        # internally, otherwise one of the sizes this machine advertises as safe.
        # resolve_tile itself is pinned against fixed hardware in
        # test_resource_policy; what matters here is that the completed result
        # reports the tile the job really used rather than the one it requested.
        modes = (await client.get("/api/v1/capabilities")).json()["modes"]
        upscale = next(mode for mode in modes if mode["mode"] == "upscale")
        resolved_tile = snapshot["result"]["resolved_tile_size"]
        assert snapshot["settings"]["tile_size"] == 0
        assert resolved_tile == 0 or resolved_tile in upscale["safe_tile_sizes"]

        result = await client.get(f"/api/v1/jobs/{job_id}/result")
        assert result.status_code == 200
        with Image.open(io.BytesIO(result.content)) as image:
            assert image.size == (256, 128)

        assert (await client.delete(f"/api/v1/jobs/{job_id}")).status_code == 202
        assert (await client.get(f"/api/v1/jobs/{job_id}")).status_code == 404


@pytest.mark.asyncio
async def test_sharpen_only_job_keeps_source_dimensions(tmp_path: Path):
    config = AppConfig(work_root=tmp_path / "jobs", max_input_pixels=1_000_000)
    app, client = app_client(config)
    async with app.router.lifespan_context(app), client:
        response = await client.post(
            "/api/v1/jobs",
            files={"file": ("small.png", png_bytes((41, 23)), "image/png")},
            data={"target_edge": "7680", "processing_mode": "sharpen_only", "sharpen": "25"},
        )

        assert response.status_code == 202
        snapshot = await wait_for_terminal(client, response.json()["id"])
        assert snapshot["state"] == "completed"
        assert snapshot["result"]["processing_mode"] == "sharpen_only"
        assert snapshot["result"]["engine"] == "classical:sharpen-only"
        assert (snapshot["result"]["width"], snapshot["result"]["height"]) == (41, 23)
        assert snapshot["result"]["filename"] == "small-sharpened.png"


@pytest.mark.asyncio
async def test_rejects_incompatible_processing_settings_before_job_creation(tmp_path: Path):
    config = AppConfig(work_root=tmp_path / "jobs", max_input_pixels=1_000_000)
    app, client = app_client(config)
    async with app.router.lifespan_context(app), client:
        response = await client.post(
            "/api/v1/jobs",
            files={"file": ("small.png", png_bytes(), "image/png")},
            data={"processing_mode": "sharpen_only", "sharpen": "20", "tta": "true"},
        )

        assert response.status_code == 400
        assert "does not accept neural" in response.json()["detail"]
        assert list((tmp_path / "jobs").iterdir()) == []


@pytest.mark.asyncio
async def test_invalid_image_is_rejected_and_workspace_is_cleaned(tmp_path: Path):
    config = AppConfig(work_root=tmp_path / "jobs", max_input_pixels=1_000_000)
    app, client = app_client(config)
    async with app.router.lifespan_context(app), client:
        response = await client.post(
            "/api/v1/jobs",
            files={"file": ("fake.png", b"definitely not png", "image/png")},
            data={"target_edge": "256"},
        )
        assert response.status_code == 400
        assert "supported or valid image" in response.json()["detail"]
        assert "Traceback" not in response.json()["detail"]
        assert list((tmp_path / "jobs").iterdir()) == []


@pytest.mark.asyncio
async def test_live_memory_admission_returns_409_and_cleans_upload(tmp_path: Path):
    config = AppConfig(work_root=tmp_path / "jobs", max_input_pixels=1_000_000)
    app, client = app_client(config)
    async with app.router.lifespan_context(app), client:
        manager = app.state.job_manager
        manager.models._adapters["spandrel-sr"] = SimpleNamespace(
            id="spandrel-sr",
            name="Test SwinIR",
            neural=True,
            generative=False,
            max_passes=3,
            native_scales=(2, 3, 4),
            license="test",
            noncommercial=False,
            restores_without_enlarging=False,
            restores_faces=False,
            available=True,
            unavailable_reason=None,
            device="Test GPU",
        )
        stable = HardwareSnapshot(
            scope="backend",
            ram_physical_mib=64 * 1024,
            ram_effective_mib=64 * 1024,
            ram_available_mib=64 * 1024,
            gpu_name="Test GPU",
            vram_total_mib=16 * 1024,
            vram_available_mib=16 * 1024,
            memory_kind="dedicated",
            source="test",
        )
        busy = HardwareSnapshot(
            scope="backend",
            ram_physical_mib=stable.ram_physical_mib,
            ram_effective_mib=stable.ram_effective_mib,
            ram_available_mib=1024,
            gpu_name=stable.gpu_name,
            vram_total_mib=stable.vram_total_mib,
            vram_available_mib=1024,
            memory_kind=stable.memory_kind,
            source=stable.source,
        )
        manager.models.hardware._stable = stable
        manager.models.hardware.snapshot = lambda: busy

        response = await client.post(
            "/api/v1/jobs",
            files={"file": ("small.png", png_bytes(), "image/png")},
            data={"target_edge": "256", "tile_size": "128"},
        )

        assert response.status_code == 409
        assert "available RAM" in response.json()["detail"]
        assert list((tmp_path / "jobs").iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host",
    [
        # The ASGI test-client convention. It was once in the allowlist and is
        # not a name any deployment binds, so it must be refused like any other.
        "testserver",
        # The classic DNS-rebinding suffix: a name an attacker controls that
        # merely starts with a loopback address.
        "127.0.0.1.attacker.example",
        "localhost.attacker.example",
        # And the reverse, in case the check is ever loosened to a substring.
        "attacker.example",
    ],
)
async def test_rejects_every_host_that_is_not_loopback(tmp_path: Path, host: str):
    config = AppConfig(work_root=tmp_path / "jobs")
    app, client = app_client(config)
    async with app.router.lifespan_context(app), client:
        response = await client.get("/api/v1/health", headers={"host": host})
        assert response.status_code == 421, f"{host} was accepted as a local host"


@pytest.mark.asyncio
async def test_accepts_a_state_change_that_carries_no_origin(tmp_path: Path):
    """The absent-Origin branch, pinned so it is a decision rather than a gap.

    A browser always attaches Origin to a cross-origin POST, so no header means
    a local non-browser client, which SECURITY.md places outside the trust
    boundary. Requiring it would break every local API client and close nothing.
    """
    config = AppConfig(work_root=tmp_path / "jobs")
    app, client = app_client(config)
    async with app.router.lifespan_context(app), client:
        response = await client.post(
            "/api/v1/jobs",
            files={"file": ("small.png", png_bytes(), "image/png")},
        )

        assert response.status_code == 202


@pytest.mark.asyncio
async def test_serves_the_schema_locally_and_no_cdn_backed_documentation(tmp_path: Path):
    """FastAPI's Swagger UI and ReDoc pages load from cdn.jsdelivr.net.

    Serving either would make the browser call a third party and would render
    blank offline, so neither is mounted. The schema they read is local and
    stays available.
    """
    config = AppConfig(work_root=tmp_path / "jobs")
    app, client = app_client(config)
    async with app.router.lifespan_context(app), client:
        schema = await client.get("/api/v1/openapi.json")
        assert schema.status_code == 200
        assert "/api/v1/jobs" in schema.json()["paths"]

        for path in ("/api/docs", "/docs", "/redoc"):
            assert (await client.get(path)).status_code == 404, f"{path} is still served"


@pytest.mark.asyncio
async def test_rejects_cross_origin_state_changes(tmp_path: Path):
    config = AppConfig(work_root=tmp_path / "jobs")
    app, client = app_client(config)
    async with app.router.lifespan_context(app), client:
        response = await client.post(
            "/api/v1/jobs",
            headers={"origin": "https://attacker.example"},
            files={"file": ("small.png", png_bytes(), "image/png")},
        )

        assert response.status_code == 403
        assert response.json()["detail"] == "Cross-origin state changes are not allowed."
        assert list(config.work_root.iterdir()) == []


@pytest.mark.asyncio
async def test_accepts_the_vite_dev_server_origin(tmp_path: Path):
    """The dev server's origin never matches the request's own host.

    Vite rewrites Host to the proxy target, so a job submitted from
    `make dev-frontend` arrives with the backend's own Host and an Origin of
    `http://127.0.0.1:5173`. Dropping that allowance rejects every submission
    from the documented development workflow.
    """
    config = AppConfig(work_root=tmp_path / "jobs")
    app, client = app_client(config)
    async with app.router.lifespan_context(app), client:
        response = await client.post(
            "/api/v1/jobs",
            headers={"origin": "http://127.0.0.1:5173"},
            files={"file": ("small.png", png_bytes(), "image/png")},
        )

        assert response.status_code == 202


@pytest.mark.asyncio
async def test_upload_limit_is_enforced_and_workspace_is_cleaned(tmp_path: Path):
    payload = png_bytes()
    config = AppConfig(
        work_root=tmp_path / "jobs",
        max_upload_bytes=len(payload) - 1,
        max_input_pixels=1_000_000,
    )
    app, client = app_client(config)
    async with app.router.lifespan_context(app), client:
        response = await client.post(
            "/api/v1/jobs",
            files={"file": ("small.png", payload, "image/png")},
        )

        assert response.status_code == 400
        assert "Upload exceeds" in response.json()["detail"]
        assert list(config.work_root.iterdir()) == []


@pytest.mark.asyncio
async def test_decoded_pixel_limit_is_enforced_and_workspace_is_cleaned(tmp_path: Path):
    config = AppConfig(work_root=tmp_path / "jobs", max_input_pixels=100)
    app, client = app_client(config)
    async with app.router.lifespan_context(app), client:
        response = await client.post(
            "/api/v1/jobs",
            files={"file": ("small.png", png_bytes(), "image/png")},
        )

        assert response.status_code == 400
        assert "decoder safety limit" in response.json()["detail"]
        assert list(config.work_root.iterdir()) == []


@pytest.mark.asyncio
async def test_deleting_an_active_job_cancels_it_and_removes_its_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    started = threading.Event()

    def wait_for_cancel(*args, **_kwargs):
        cancel = args[7]
        started.set()
        if not cancel.wait(timeout=2):
            raise AssertionError("the active worker was never cancelled")
        raise ProcessingCancelled("cancelled by test")

    monkeypatch.setattr(job_manager_module, "process_image", wait_for_cancel)
    config = AppConfig(work_root=tmp_path / "jobs", max_input_pixels=1_000_000)
    app, client = app_client(config)
    async with app.router.lifespan_context(app), client:
        created = await client.post(
            "/api/v1/jobs",
            files={"file": ("small.png", png_bytes(), "image/png")},
        )
        job_id = created.json()["id"]
        assert await asyncio.to_thread(started.wait, 2)
        assert (await client.get(f"/api/v1/jobs/{job_id}/result")).status_code == 409

        deleted = await client.delete(f"/api/v1/jobs/{job_id}")

        assert deleted.status_code == 202
        assert deleted.json()["state"] == "cancelling"
        await wait_for_removal(client, job_id)
        assert list(config.work_root.iterdir()) == []


@pytest.mark.asyncio
async def test_a_failed_job_discards_files_but_keeps_the_error_until_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def fail_processing(*_args, **_kwargs):
        raise RuntimeError("deliberate worker failure")

    monkeypatch.setattr(job_manager_module, "process_image", fail_processing)
    config = AppConfig(work_root=tmp_path / "jobs", max_input_pixels=1_000_000)
    app, client = app_client(config)
    async with app.router.lifespan_context(app), client:
        created = await client.post(
            "/api/v1/jobs",
            files={"file": ("small.png", png_bytes(), "image/png")},
        )
        job_id = created.json()["id"]

        snapshot = await wait_for_terminal(client, job_id)

        assert snapshot["state"] == "failed"
        assert snapshot["error"] == "deliberate worker failure"
        assert not (config.work_root / job_id).exists()
        assert (await client.delete(f"/api/v1/jobs/{job_id}")).status_code == 202
        assert (await client.get(f"/api/v1/jobs/{job_id}")).status_code == 404


@pytest.mark.asyncio
async def test_jobs_are_queued_to_the_configured_concurrency_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    first_started = threading.Event()
    first_release = threading.Event()
    second_started = threading.Event()
    process_image = job_manager_module.process_image

    def controlled_processing(*args, **kwargs):
        if args[3] == "first.png":
            first_started.set()
            if not first_release.wait(timeout=3):
                raise AssertionError("first job was never released")
        else:
            second_started.set()
        return process_image(*args, **kwargs)

    monkeypatch.setattr(job_manager_module, "process_image", controlled_processing)
    config = AppConfig(work_root=tmp_path / "jobs", max_input_pixels=1_000_000, max_jobs=1)
    app, client = app_client(config)
    async with app.router.lifespan_context(app), client:
        first = await client.post(
            "/api/v1/jobs",
            files={"file": ("first.png", png_bytes(), "image/png")},
            data={"target_edge": "256"},
        )
        assert await asyncio.to_thread(first_started.wait, 2)
        second = await client.post(
            "/api/v1/jobs",
            files={"file": ("second.png", png_bytes(), "image/png")},
            data={"target_edge": "256"},
        )
        second_id = second.json()["id"]

        await asyncio.sleep(0.05)
        assert not second_started.is_set()
        assert (await client.get(f"/api/v1/jobs/{second_id}")).json()["state"] == "queued"

        first_release.set()
        assert (await wait_for_terminal(client, first.json()["id"]))["state"] == "completed"
        assert (await wait_for_terminal(client, second_id))["state"] == "completed"
        assert second_started.is_set()


@pytest.mark.asyncio
async def test_queue_depth_is_bounded_and_refuses_before_writing_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """max_jobs bounds what runs; max_queued_jobs bounds what waits behind it.

    Without the second bound the queue grows without limit, and because the
    upload is streamed to disk during ``create`` every waiting job has already
    spent up to max_upload_bytes of the work root. The refusal therefore has to
    happen before the workspace is made, not after the bytes land.
    """
    first_started = threading.Event()
    first_release = threading.Event()
    process_image = job_manager_module.process_image

    def controlled_processing(*args, **kwargs):
        if args[3] == "first.png":
            first_started.set()
            if not first_release.wait(timeout=3):
                raise AssertionError("first job was never released")
        return process_image(*args, **kwargs)

    monkeypatch.setattr(job_manager_module, "process_image", controlled_processing)
    config = AppConfig(
        work_root=tmp_path / "jobs",
        max_input_pixels=1_000_000,
        max_jobs=1,
        max_queued_jobs=2,
    )
    app, client = app_client(config)
    async with app.router.lifespan_context(app), client:
        first = await client.post(
            "/api/v1/jobs",
            files={"file": ("first.png", png_bytes(), "image/png")},
            data={"target_edge": "256"},
        )
        assert first.status_code == 202
        assert await asyncio.to_thread(first_started.wait, 2)

        second = await client.post(
            "/api/v1/jobs",
            files={"file": ("second.png", png_bytes(), "image/png")},
            data={"target_edge": "256"},
        )
        assert second.status_code == 202

        third = await client.post(
            "/api/v1/jobs",
            files={"file": ("third.png", png_bytes(), "image/png")},
            data={"target_edge": "256"},
        )
        assert third.status_code == 429
        assert "limit" in third.json()["detail"]
        # The refused submission left nothing behind: one workspace for the
        # running job, one for the queued job, and none for the third.
        assert len(list(config.work_root.iterdir())) == 2

        first_release.set()
        assert (await wait_for_terminal(client, first.json()["id"]))["state"] == "completed"
        assert (await wait_for_terminal(client, second.json()["id"]))["state"] == "completed"

        # A finished job no longer occupies a slot, so submission resumes.
        fourth = await client.post(
            "/api/v1/jobs",
            files={"file": ("fourth.png", png_bytes(), "image/png")},
            data={"target_edge": "256"},
        )
        assert fourth.status_code == 202
        assert (await wait_for_terminal(client, fourth.json()["id"]))["state"] == "completed"


@pytest.mark.asyncio
async def test_event_stream_reports_the_terminal_snapshot(tmp_path: Path):
    config = AppConfig(work_root=tmp_path / "jobs", max_input_pixels=1_000_000)
    app, client = app_client(config)
    async with app.router.lifespan_context(app), client:
        created = await client.post(
            "/api/v1/jobs",
            files={"file": ("small.png", png_bytes(), "image/png")},
            data={"target_edge": "256"},
        )
        job_id = created.json()["id"]
        await wait_for_terminal(client, job_id)

        response = await client.get(f"/api/v1/jobs/{job_id}/events")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "event: job" in response.text
        assert '"state":"completed"' in response.text


@pytest.mark.asyncio
async def test_download_response_cleans_the_completed_job(tmp_path: Path):
    config = AppConfig(work_root=tmp_path / "jobs", max_input_pixels=1_000_000)
    app, client = app_client(config)
    async with app.router.lifespan_context(app), client:
        created = await client.post(
            "/api/v1/jobs",
            files={"file": ("small.png", png_bytes(), "image/png")},
            data={"target_edge": "256"},
        )
        job_id = created.json()["id"]
        await wait_for_terminal(client, job_id)

        response = await client.get(f"/api/v1/jobs/{job_id}/result?download=true")

        assert response.status_code == 200
        assert response.headers["content-disposition"].startswith("attachment;")
        assert int(response.headers["content-length"]) == len(response.content)
        assert (await client.get(f"/api/v1/jobs/{job_id}")).status_code == 404
        assert list(config.work_root.iterdir()) == []


@pytest.mark.asyncio
async def test_expired_terminal_job_is_removed_on_the_next_sweep(tmp_path: Path):
    config = AppConfig(
        work_root=tmp_path / "jobs",
        max_input_pixels=1_000_000,
        job_retention_seconds=60,
    )
    app, client = app_client(config)
    async with app.router.lifespan_context(app), client:
        created = await client.post(
            "/api/v1/jobs",
            files={"file": ("small.png", png_bytes(), "image/png")},
            data={"target_edge": "256"},
        )
        job_id = created.json()["id"]
        await wait_for_terminal(client, job_id)
        manager = app.state.job_manager
        manager._jobs[job_id].updated_at -= timedelta(seconds=61)

        manager.sweep_expired()

        assert (await client.get(f"/api/v1/jobs/{job_id}")).status_code == 404
        assert list(config.work_root.iterdir()) == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("../../private/photo?.png", "photo_.png"),
        ("...", "image"),
        ("plain-name.webp", "plain-name.webp"),
    ],
)
def test_uploaded_filename_is_reduced_to_safe_display_text(value: str, expected: str):
    assert safe_filename(value) == expected
