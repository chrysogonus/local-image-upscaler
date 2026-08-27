"""The ComfyUI engine's contract.

Standing rules for an engine module, the same ones the other engine tests hold:
it imports without its optional dependency, it reports an actionable reason
rather than raising when it cannot run, and it fails loudly rather than quietly
handing back the input.

Two rules are specific to this one. It must refuse to send images anywhere but
this machine unless told otherwise in as many words, and the workflow it runs
must be the graph checked into the repository rather than something assembled at
runtime - so the templates themselves are asserted here.
"""

from __future__ import annotations

import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event

import pytest
from PIL import Image

from upscaler.models import comfyui
from upscaler.models.base import ModelExecutionError, ModelRequest, ProcessingCancelled
from upscaler.models.comfyui import (
    ComfyClient,
    ComfyUiIllustrationAdapter,
    execute_graph,
    find_workflow,
    load_catalog,
    load_template,
    output_node,
    patch_graph,
)

TIMEOUT = object()


def png_bytes(size: tuple[int, int] = (16, 9), colour: str = "red") -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


def image_frame(payload: bytes) -> bytes:
    """A result image exactly as ComfyUI frames it: event type, format, data."""
    return struct.pack(">II", 1, 2) + payload


class FakeSocket:
    """A scripted websocket. ``TIMEOUT`` stands for a quiet half second."""

    def __init__(self, frames: list) -> None:
        self._frames = list(frames)
        self.closed = False

    def __enter__(self) -> FakeSocket:
        return self

    def __exit__(self, *exc: object) -> bool:
        self.closed = True
        return False

    def recv(self, timeout: float | None = None) -> str | bytes:
        if not self._frames:
            raise TimeoutError
        frame = self._frames.pop(0)
        if frame is TIMEOUT:
            raise TimeoutError
        return frame


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # keep the test output quiet
        pass

    def _reply(self, code: int, payload: object) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/system_stats":
            release_sequence = self.server.release_sequence  # type: ignore[attr-defined]
            if release_sequence:
                ram_free, vram_free = release_sequence.pop(0)
                self.server.ram_free = ram_free  # type: ignore[attr-defined]
                self.server.vram_free = vram_free  # type: ignore[attr-defined]
            self._reply(
                200,
                {
                    "system": {
                        "ram_total": self.server.ram_total,  # type: ignore[attr-defined]
                        "ram_free": self.server.ram_free,  # type: ignore[attr-defined]
                    },
                    "devices": [
                        {
                            "name": "cuda:0 Test GPU",
                            "vram_total": self.server.vram_total,  # type: ignore[attr-defined]
                            "vram_free": self.server.vram_free,  # type: ignore[attr-defined]
                        }
                    ],
                },
            )
        elif self.path == "/queue":
            self._reply(
                200,
                {
                    "queue_running": self.server.queue_running,  # type: ignore[attr-defined]
                    "queue_pending": self.server.queue_pending,  # type: ignore[attr-defined]
                },
            )
        elif self.path == "/object_info/UpscaleModelLoader":
            self._reply(
                200,
                {
                    "UpscaleModelLoader": {
                        "input": {
                            "required": {
                                "model_name": [
                                    "COMBO",
                                    {"options": self.server.upscale_models},  # type: ignore[attr-defined]
                                ]
                            }
                        }
                    }
                },
            )
        else:
            self._reply(404, {})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        calls = self.server.calls  # type: ignore[attr-defined]
        calls.append((self.path, body))
        if self.path == "/upload/image":
            self._reply(200, {"name": "stored.png", "subfolder": "upscaler", "type": "input"})
        elif self.path == "/prompt":
            if self.server.reject:  # type: ignore[attr-defined]
                self._reply(400, {"node_errors": {"76": {"errors": ["value not in list"]}}})
            else:
                self._reply(200, {"prompt_id": "job-1"})
        elif self.path == "/free":
            ram_after = self.server.ram_free_after_release  # type: ignore[attr-defined]
            vram_after = self.server.vram_free_after_release  # type: ignore[attr-defined]
            if ram_after is not None:
                self.server.ram_free = ram_after  # type: ignore[attr-defined]
            if vram_after is not None:
                self.server.vram_free = vram_after  # type: ignore[attr-defined]
            self._reply(200, {})
        else:
            self._reply(200, {})


@pytest.fixture
def fake_comfy():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.calls = []
    server.reject = False
    server.upscale_models = [comfyui.ILLUSTRATION_MODEL]
    server.ram_total = 64 * 1024**3
    server.ram_free = 8 * 1024**3
    server.vram_total = 16 * 1024**3
    server.vram_free = 4 * 1024**3
    server.ram_free_after_release = None
    server.vram_free_after_release = None
    server.release_sequence = []
    server.queue_running = []
    server.queue_pending = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server.base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (comfyui.ENV_URL, comfyui.ENV_ALLOW_REMOTE, comfyui.ENV_INPUT_DIR):
        monkeypatch.delenv(name, raising=False)


# --- availability -----------------------------------------------------------


def test_without_a_url_the_engine_is_absent_and_says_how_to_enable_it() -> None:
    adapter = ComfyUiIllustrationAdapter()
    assert adapter.available is False
    reason = adapter.unavailable_reason
    assert reason is not None
    assert comfyui.ENV_URL in reason
    assert adapter.device == "Unavailable"


def test_a_host_that_is_not_this_machine_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Posting someone's photographs to another host is the one thing this must not do."""
    monkeypatch.setenv(comfyui.ENV_URL, "http://images.example.com:8188")
    adapter = ComfyUiIllustrationAdapter()
    assert adapter.available is False
    assert "not this machine" in (adapter.unavailable_reason or "")


def test_a_remote_host_is_allowed_only_when_asked_for_in_as_many_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(comfyui.ENV_URL, "http://images.example.com:8188")
    monkeypatch.setenv(comfyui.ENV_ALLOW_REMOTE, "1")
    url, reason = comfyui.resolve_url()
    assert url == "http://images.example.com:8188"
    assert reason is None


def test_an_unreachable_server_reports_it_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(comfyui.ENV_URL, "http://127.0.0.1:1")
    adapter = ComfyUiIllustrationAdapter()
    assert adapter.available is False
    assert "did not answer" in (adapter.unavailable_reason or "")


def test_a_running_server_makes_the_engine_available(
    monkeypatch: pytest.MonkeyPatch, fake_comfy
) -> None:
    monkeypatch.setenv(comfyui.ENV_URL, fake_comfy.base)
    adapter = ComfyUiIllustrationAdapter()
    assert adapter.available is True
    assert "Test GPU" in adapter.device
    assert adapter.unavailable_reason is None


def test_the_illustration_engine_requires_its_exact_model(
    monkeypatch: pytest.MonkeyPatch, fake_comfy
) -> None:
    monkeypatch.setenv(comfyui.ENV_URL, fake_comfy.base)
    fake_comfy.upscale_models = []
    adapter = ComfyUiIllustrationAdapter()

    assert adapter.available is False
    assert comfyui.ILLUSTRATION_MODEL in (adapter.unavailable_reason or "")
    assert "setup-model-comfyui-illustration" in (adapter.unavailable_reason or "")


def test_the_illustration_engine_is_faithful_and_available_with_the_model(
    monkeypatch: pytest.MonkeyPatch, fake_comfy
) -> None:
    monkeypatch.setenv(comfyui.ENV_URL, fake_comfy.base)
    adapter = ComfyUiIllustrationAdapter()

    assert adapter.available is True
    assert adapter.generative is False
    assert adapter.workflows()[0].id == "illustration-upscale"


def test_reachability_is_cached_so_capabilities_never_waits_on_the_network(
    monkeypatch: pytest.MonkeyPatch, fake_comfy
) -> None:
    monkeypatch.setenv(comfyui.ENV_URL, fake_comfy.base)
    adapter = ComfyUiIllustrationAdapter()
    probes = 0
    original = adapter._run_probe

    def counted() -> tuple[bool, str, str | None]:
        nonlocal probes
        probes += 1
        return original()

    monkeypatch.setattr(adapter, "_run_probe", counted)
    for _ in range(10):
        assert adapter.available is True
    assert probes == 1


def test_idle_comfyui_releases_cached_models_for_an_admission_retry(
    monkeypatch: pytest.MonkeyPatch, fake_comfy
) -> None:
    monkeypatch.setenv(comfyui.ENV_URL, fake_comfy.base)
    monkeypatch.setattr(comfyui, "RECLAIM_POLL_SECONDS", 0)
    fake_comfy.ram_free_after_release = 48 * 1024**3
    fake_comfy.vram_free_after_release = 12 * 1024**3
    adapter = ComfyUiIllustrationAdapter()
    assert adapter.available is True

    assert adapter.reclaim_memory_if_idle() is True

    free_calls = [body for path, body in fake_comfy.calls if path == "/free"]
    assert [json.loads(body) for body in free_calls] == [
        {"unload_models": True, "free_memory": True}
    ]
    assert adapter.hardware_stats()["system"]["ram_free"] == 48 * 1024**3


def test_reclaim_waits_for_partial_memory_release_to_settle(
    monkeypatch: pytest.MonkeyPatch, fake_comfy
) -> None:
    monkeypatch.setenv(comfyui.ENV_URL, fake_comfy.base)
    monkeypatch.setattr(comfyui, "RECLAIM_POLL_SECONDS", 0)
    adapter = ComfyUiIllustrationAdapter()
    assert adapter.available is True
    initial_ram = fake_comfy.ram_free
    initial_vram = fake_comfy.vram_free
    fake_comfy.release_sequence = [
        (initial_ram + 32 * 1024**2, initial_vram + 32 * 1024**2),
        (initial_ram + 2 * 1024**3, initial_vram + 2 * 1024**3),
        (48 * 1024**3, 12 * 1024**3),
        (48 * 1024**3, 12 * 1024**3),
        (48 * 1024**3, 12 * 1024**3),
        (48 * 1024**3, 12 * 1024**3),
        (48 * 1024**3, 12 * 1024**3),
        (48 * 1024**3, 12 * 1024**3),
    ]

    assert adapter.reclaim_memory_if_idle() is True

    assert fake_comfy.release_sequence == []
    assert adapter.hardware_stats()["system"]["ram_free"] == 48 * 1024**3


def test_busy_comfyui_is_never_asked_to_release_memory(
    monkeypatch: pytest.MonkeyPatch, fake_comfy
) -> None:
    monkeypatch.setenv(comfyui.ENV_URL, fake_comfy.base)
    fake_comfy.queue_running = [[0, "another-job"]]
    adapter = ComfyUiIllustrationAdapter()
    assert adapter.available is True

    assert adapter.reclaim_memory_if_idle() is False
    assert all(path != "/free" for path, _body in fake_comfy.calls)


# --- the checked-in templates ----------------------------------------------


def test_every_catalogued_workflow_has_a_template_and_the_slots_it_claims() -> None:
    """Every node id the catalogue names has to exist, or a job fails mid-run."""
    for workflow in load_catalog():
        template = load_template(workflow)
        assert output_node(template) is not None, f"{workflow.id} has no websocket output"
        for slot, value in workflow.slots.items():
            for node_id in [value] if isinstance(value, str) else value:
                assert node_id in template, f"{workflow.id}.{slot} points at missing {node_id}"
        assert workflow.node("image"), f"{workflow.id} has nowhere to put the source"


def test_an_enlarging_workflow_declares_where_the_output_size_goes() -> None:
    for workflow in load_catalog():
        if not workflow.enlarges:
            continue
        assert workflow.nodes("target_scales"), f"{workflow.id} cannot be told the output size"


def test_templates_send_their_result_over_the_socket_rather_than_comfyuis_disk() -> None:
    """Nothing may accumulate in another application's output directory."""
    for workflow in load_catalog():
        classes = {node["class_type"] for node in load_template(workflow).values()}
        assert "SaveImage" not in classes
        assert "SaveImageWebsocket" in classes


def test_no_template_references_a_node_it_does_not_contain() -> None:
    for workflow in load_catalog():
        template = load_template(workflow)
        for node_id, node in template.items():
            for name, value in node["inputs"].items():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    assert value[0] in template, f"{workflow.id}: {node_id}.{name} dangles"


def test_an_unknown_workflow_is_rejected_by_name() -> None:
    with pytest.raises(ValueError, match="Unknown ComfyUI workflow"):
        find_workflow("no-such-workflow")


# --- writing a job into the graph ------------------------------------------


def test_the_illustration_graph_scales_colour_and_alpha_to_the_exact_target() -> None:
    workflow = find_workflow("illustration-upscale", comfyui.ProcessingMode.illustration)
    graph = patch_graph(
        load_template(workflow),
        workflow,
        image="upscaler/stored.png",
        target=(3200, 1800),
    )

    assert (graph["4"]["inputs"]["width"], graph["4"]["inputs"]["height"]) == (3200, 1800)
    assert (graph["6"]["inputs"]["width"], graph["6"]["inputs"]["height"]) == (3200, 1800)
    assert graph["2"]["inputs"]["model_name"] == comfyui.ILLUSTRATION_MODEL
    assert graph["3"]["class_type"] == "ImageUpscaleWithModel"
    assert graph["8"]["class_type"] == "JoinImageWithAlpha"


def test_patching_leaves_the_checked_in_template_untouched() -> None:
    workflow = find_workflow("illustration-upscale")
    template = load_template(workflow)
    before = json.dumps(template, sort_keys=True)
    patch_graph(template, workflow, image="x.png", target=(100, 200))
    assert json.dumps(template, sort_keys=True) == before


# --- running -----------------------------------------------------------------


def _request(tmp_path: Path, **kwargs) -> ModelRequest:
    source = tmp_path / "source.png"
    source.write_bytes(png_bytes((40, 20)))
    defaults = {
        "source_path": source,
        "output_path": tmp_path / "out.png",
        "workspace": tmp_path,
        "native_scale": 1,
        "target_width": 80,
        "target_height": 40,
        "tile_size": 0,
    }
    return ModelRequest(**{**defaults, **kwargs})


def test_a_rejected_graph_surfaces_comfyuis_own_node_errors(
    monkeypatch: pytest.MonkeyPatch, fake_comfy, tmp_path: Path
) -> None:
    monkeypatch.setenv(comfyui.ENV_URL, fake_comfy.base)
    fake_comfy.reject = True
    monkeypatch.setattr(comfyui, "_websocket", lambda *a, **k: FakeSocket([]))
    adapter = ComfyUiIllustrationAdapter()
    with pytest.raises(ModelExecutionError, match="value not in list"):
        adapter.enhance(
            _request(tmp_path, workflow="illustration-upscale"), Event(), lambda *a: None
        )


def test_a_finished_run_writes_the_image_that_came_back(
    monkeypatch: pytest.MonkeyPatch, fake_comfy, tmp_path: Path
) -> None:
    monkeypatch.setenv(comfyui.ENV_URL, fake_comfy.base)
    produced = png_bytes((80, 40), "blue")
    frames = [
        json.dumps({"type": "execution_start", "data": {"prompt_id": "job-1"}}),
        json.dumps({"type": "executing", "data": {"node": "9", "prompt_id": "job-1"}}),
        image_frame(produced),
        json.dumps({"type": "executing", "data": {"node": None, "prompt_id": "job-1"}}),
    ]
    monkeypatch.setattr(comfyui, "_websocket", lambda *a, **k: FakeSocket(frames))
    adapter = ComfyUiIllustrationAdapter()
    result = adapter.enhance(
        _request(tmp_path, workflow="illustration-upscale"), Event(), lambda *a: None
    )
    assert result.engine_id == "comfyui-illustration:illustration-upscale"
    with Image.open(result.output_path) as written:
        assert written.size == (80, 40)


def test_the_illustration_engine_reports_where_its_detail_was_made(
    monkeypatch: pytest.MonkeyPatch, fake_comfy, tmp_path: Path
) -> None:
    """The graph resizes to the target itself, so the file hides its own stretch.

    A 40px source reaches 640px through one x4 model pass and a Lanczos resize
    inside the graph. Only the first 4x is reconstruction; the rest is the
    softness the finishing pass is sized against.
    """
    monkeypatch.setenv(comfyui.ENV_URL, fake_comfy.base)
    frames = [
        json.dumps({"type": "executing", "data": {"node": "9", "prompt_id": "job-1"}}),
        image_frame(png_bytes((640, 320), "blue")),
        json.dumps({"type": "executing", "data": {"node": None, "prompt_id": "job-1"}}),
    ]
    monkeypatch.setattr(comfyui, "_websocket", lambda *a, **k: FakeSocket(frames))
    result = ComfyUiIllustrationAdapter().enhance(
        _request(tmp_path, workflow="illustration-upscale", target_width=640, target_height=320),
        Event(),
        lambda *a: None,
    )
    assert result.engine_id == "comfyui-illustration:illustration-upscale"
    assert result.detail_width == 160


def test_images_the_result_node_did_not_produce_are_ignored(
    monkeypatch: pytest.MonkeyPatch, fake_comfy
) -> None:
    """Sampler previews arrive on the same socket and must not become the result."""
    preview = image_frame(png_bytes((8, 8), "green"))
    wanted = image_frame(png_bytes((64, 64), "blue"))
    frames = [
        json.dumps({"type": "executing", "data": {"node": "64", "prompt_id": "job-1"}}),
        preview,
        json.dumps({"type": "executing", "data": {"node": "213", "prompt_id": "job-1"}}),
        wanted,
        json.dumps({"type": "executing", "data": {"node": None, "prompt_id": "job-1"}}),
    ]
    monkeypatch.setattr(comfyui, "_websocket", lambda *a, **k: FakeSocket(frames))
    client = ComfyClient(fake_comfy.base)
    encoded = execute_graph(client, {"213": {}}, "213", Event(), lambda *a: None, "test")
    assert encoded == wanted[8:]


def test_a_failure_inside_comfyui_names_the_node_that_failed(
    monkeypatch: pytest.MonkeyPatch, fake_comfy
) -> None:
    frames = [
        json.dumps(
            {
                "type": "execution_error",
                "data": {
                    "prompt_id": "job-1",
                    "node_type": "UNETLoader",
                    "exception_message": "checkpoint not found",
                },
            }
        )
    ]
    monkeypatch.setattr(comfyui, "_websocket", lambda *a, **k: FakeSocket(frames))
    client = ComfyClient(fake_comfy.base)
    with pytest.raises(ModelExecutionError, match="UNETLoader.*checkpoint not found"):
        execute_graph(client, {"213": {}}, "213", Event(), lambda *a: None, "test")


def test_finishing_without_an_image_is_an_error_not_a_silent_pass(
    monkeypatch: pytest.MonkeyPatch, fake_comfy
) -> None:
    frames = [json.dumps({"type": "executing", "data": {"node": None, "prompt_id": "job-1"}})]
    monkeypatch.setattr(comfyui, "_websocket", lambda *a, **k: FakeSocket(frames))
    client = ComfyClient(fake_comfy.base)
    with pytest.raises(ModelExecutionError, match="without returning an image"):
        execute_graph(client, {"213": {}}, "213", Event(), lambda *a: None, "test")


# --- cancellation ------------------------------------------------------------


def test_cancelling_a_running_job_interrupts_comfyui(
    monkeypatch: pytest.MonkeyPatch, fake_comfy
) -> None:
    cancel = Event()
    frames = [
        json.dumps({"type": "execution_start", "data": {"prompt_id": "job-1"}}),
        TIMEOUT,
    ]

    def progress(phase: str, message: str, fraction: float | None) -> None:
        cancel.set()

    monkeypatch.setattr(comfyui, "_websocket", lambda *a, **k: FakeSocket(frames))
    client = ComfyClient(fake_comfy.base)
    with pytest.raises(ProcessingCancelled):
        execute_graph(client, {"213": {}}, "213", cancel, progress, "test")
    posted = [path for path, _ in fake_comfy.calls]
    assert "/interrupt" in posted
    assert "/queue" in posted


def test_cancelling_a_job_still_queued_does_not_interrupt_somebody_elses(
    monkeypatch: pytest.MonkeyPatch, fake_comfy
) -> None:
    """/interrupt has no job id, so it may only be sent for a run known to be ours."""
    cancel = Event()
    cancel.set()
    monkeypatch.setattr(comfyui, "_websocket", lambda *a, **k: FakeSocket([]))
    client = ComfyClient(fake_comfy.base)
    with pytest.raises(ProcessingCancelled):
        execute_graph(client, {"213": {}}, "213", cancel, lambda *a: None, "test")
    posted = [path for path, _ in fake_comfy.calls]
    assert "/interrupt" not in posted
    assert "/queue" in posted


# --- cleanup -----------------------------------------------------------------


def test_the_uploaded_source_is_deleted_when_the_input_directory_is_known(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(comfyui.ENV_INPUT_DIR, str(tmp_path))
    uploaded = tmp_path / "upscaler" / "stored.png"
    uploaded.parent.mkdir()
    uploaded.write_bytes(b"x")
    comfyui._delete_upload("upscaler/stored.png")
    assert not uploaded.exists()


def test_cleanup_cannot_be_walked_out_of_the_input_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(comfyui.ENV_INPUT_DIR, str(tmp_path / "inputs"))
    (tmp_path / "inputs").mkdir()
    outside = tmp_path / "precious.png"
    outside.write_bytes(b"x")
    comfyui._delete_upload("../precious.png")
    assert outside.exists()


def test_progress_is_reported_from_comfyuis_own_node_states() -> None:
    states = {
        "1": {"state": "finished", "value": 1, "max": 1},
        "2": {"state": "running", "value": 5, "max": 10},
    }
    assert comfyui._overall_fraction(states, {"1", "2"}) == pytest.approx(0.75)
    assert comfyui._overall_fraction(states, {"1", "2", "3"}) == pytest.approx(0.5)
    assert comfyui._overall_fraction({}, set()) is None
