"""Run the checked-in illustration graph on a local ComfyUI.

ComfyUI is a separate application with its own models and its own queue. This
adapter treats it as a remote worker: upload the source, submit a checked-in
copy of the graph with the job's values written into it, follow the run over the
websocket, and take the finished pixels back off that same socket.

Three things this deliberately does *not* do.

It does not build graphs. The templates under ``workflows/`` are generated from
the saved ComfyUI workflows by ``scripts/comfy-export-workflow.py`` and are only
ever patched at a handful of named slots, so what runs stays the graph the user
authored and can inspect in ComfyUI.

It does not reach off the machine. ``UPSCALER_COMFYUI_URL`` has to be set for the
engine to exist at all, and a non-loopback host is refused unless the user also
sets ``UPSCALER_COMFYUI_ALLOW_REMOTE``: posting someone's photographs to another
host is exactly what this application promises not to do, so it cannot be
reached by a typo in a URL.

It does not write to ComfyUI's disk. The templates end in ``SaveImageWebsocket``
rather than ``SaveImage``, so results come back in memory and never land in a
directory the user then has to remember to clear. The one unavoidable exception
is the uploaded source, which ComfyUI's API only accepts as a file in its input
directory; ``UPSCALER_COMFYUI_INPUT_DIR`` lets the adapter delete it afterwards.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Event, Lock
from typing import Any

from PIL import Image

from upscaler.imaging.io import resize_exact
from upscaler.models.base import (
    ModelExecutionError,
    ModelRequest,
    ModelResult,
    ProcessingCancelled,
    ProgressCallback,
)
from upscaler.schemas import ProcessingMode

ENV_URL = "UPSCALER_COMFYUI_URL"
ENV_ALLOW_REMOTE = "UPSCALER_COMFYUI_ALLOW_REMOTE"
ENV_INPUT_DIR = "UPSCALER_COMFYUI_INPUT_DIR"

WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "workflows"
CATALOG_PATH = WORKFLOW_DIR / "catalog.json"

URL_HINT = "http://127.0.0.1:8188"
# host.docker.internal is the container's name for the machine it is running on,
# resolved through the host gateway declared in docker-compose.yml. The app ships
# as a container and ComfyUI does not, so this is how the supported deployment
# names its own host; treating it as remote would demand the off-device opt-in
# for a connection that never leaves the machine.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "host.docker.internal"})

# How long a reachability answer is reused. /capabilities is polled by the
# interface, and a request that waited on a network round trip every time would
# make an unreachable ComfyUI feel like a hung application rather than a missing
# one. Short enough that starting ComfyUI is picked up without a restart.
PROBE_TTL_SECONDS = 10.0
PROBE_TIMEOUT_SECONDS = 1.5
REQUEST_TIMEOUT_SECONDS = 30.0
RECLAIM_TIMEOUT_SECONDS = 5.0
RECLAIM_POLL_SECONDS = 0.1
RECLAIM_MIN_INCREASE_BYTES = 16 * 1024 * 1024
RECLAIM_SETTLE_POLLS = 5

# Keep concurrent admission fallbacks from racing each other between the idle
# check and the release request.
_RECLAIM_LOCK = Lock()

# ComfyUI frames binary websocket messages as a 4-byte event type followed by a
# 4-byte payload type. See ComfyUI's protocol.py and server.send_image.
BINARY_EVENT_PREVIEW_IMAGE = 1
BINARY_IMAGE_FORMATS = {1: "JPEG", 2: "PNG"}
BINARY_HEADER = struct.Struct(">II")

WEBSOCKET_HINT = (
    "The websockets package is required to talk to ComfyUI. The released image "
    "carries it; a host checkout needs `uv sync --extra comfyui`."
)
ILLUSTRATION_MODEL = "RealESRGAN_x4plus_anime_6B.pth"


@dataclass(frozen=True, slots=True)
class Workflow:
    """One ComfyUI graph, and where to write a job's values into it."""

    id: str
    mode: ProcessingMode
    name: str
    description: str
    warning: str
    template: str
    enlarges: bool
    # The enlargement the graph's model stage produces before the graph resizes
    # to the target, when the graph is a single fixed-factor model pass. None
    # where no single factor describes it.
    model_scale: int | None
    slots: dict[str, Any]

    def node(self, slot: str) -> str | None:
        value = self.slots.get(slot)
        return value if isinstance(value, str) else None

    def nodes(self, slot: str) -> tuple[str, ...]:
        value = self.slots.get(slot)
        return tuple(value) if isinstance(value, list) else ()


@lru_cache(maxsize=1)
def load_catalog() -> tuple[Workflow, ...]:
    """The workflows this build ships. Read once; the files are checked in."""
    try:
        raw = json.loads(CATALOG_PATH.read_text())
    except (OSError, ValueError) as exc:  # pragma: no cover - packaging error
        raise ModelExecutionError(f"The ComfyUI workflow catalog is unreadable: {exc}") from exc
    return tuple(
        Workflow(
            id=entry["id"],
            mode=ProcessingMode(entry.get("mode", ProcessingMode.illustration.value)),
            name=entry["name"],
            description=entry["description"],
            warning=entry["warning"],
            template=entry["template"],
            enlarges=bool(entry["enlarges"]),
            model_scale=entry.get("model_scale"),
            slots=entry["slots"],
        )
        for entry in raw["workflows"]
    )


def available_workflows(
    mode: ProcessingMode = ProcessingMode.illustration,
) -> tuple[Workflow, ...]:
    """The graphs this deployment will run, in catalogue order."""
    return tuple(workflow for workflow in load_catalog() if workflow.mode == mode)


def find_workflow(
    workflow_id: str | None,
    mode: ProcessingMode = ProcessingMode.illustration,
) -> Workflow:
    catalog = available_workflows(mode)
    if not catalog:
        raise ValueError(f"No ComfyUI workflows are available for {mode.value} mode.")
    if not workflow_id:
        return catalog[0]
    for workflow in catalog:
        if workflow.id == workflow_id:
            return workflow
    known = ", ".join(item.id for item in catalog)
    raise ValueError(f"Unknown ComfyUI workflow {workflow_id!r}. Available: {known}.")


def load_template(workflow: Workflow) -> dict[str, Any]:
    path = WORKFLOW_DIR / workflow.template
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:  # pragma: no cover - packaging error
        raise ModelExecutionError(f"Workflow template {path.name} is unreadable: {exc}") from exc


def resolve_url() -> tuple[str | None, str | None]:
    """The configured base URL, or why there is not a usable one."""
    raw = os.environ.get(ENV_URL, "").strip()
    if not raw:
        return None, (
            "ComfyUI is not configured. Set "
            f"{ENV_URL}={URL_HINT} to run Illustration on a local ComfyUI."
        )
    parsed = urllib.parse.urlsplit(raw if "://" in raw else f"http://{raw}")
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None, f"{ENV_URL} is not a usable http URL: {raw!r}."
    if parsed.hostname not in LOOPBACK_HOSTS and not os.environ.get(ENV_ALLOW_REMOTE):
        return None, (
            f"{ENV_URL} points at {parsed.hostname}, which is not this machine. Sending "
            "images there would take them off-device, so it is refused; set "
            f"{ENV_ALLOW_REMOTE}=1 if that is genuinely what you want."
        )
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", "")), None


def _fmt_http_error(exc: urllib.error.HTTPError) -> str:
    body = exc.read().decode("utf-8", errors="replace")
    try:
        parsed = json.loads(body)
    except ValueError:
        return body.strip()[:600] or f"HTTP {exc.code}"
    # ComfyUI rejects an invalid graph with per-node detail, which names the
    # actual problem (a missing checkpoint, an out-of-range value). Passing it
    # through beats "the job failed".
    node_errors = parsed.get("node_errors") if isinstance(parsed, dict) else None
    if node_errors:
        return json.dumps(node_errors)[:900]
    if isinstance(parsed, dict) and parsed.get("error"):
        return json.dumps(parsed["error"])[:600]
    return body.strip()[:600] or f"HTTP {exc.code}"


class ComfyClient:
    """The bits of ComfyUI's HTTP API this adapter needs."""

    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def _request(self, request: urllib.request.Request, timeout: float) -> Any:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read().decode("utf-8", errors="replace")
        return json.loads(raw) if raw.strip() else {}

    def get(self, path: str, timeout: float = REQUEST_TIMEOUT_SECONDS) -> Any:
        return self._request(urllib.request.Request(f"{self.base}{path}"), timeout)

    def post(
        self, path: str, payload: dict[str, Any], timeout: float = REQUEST_TIMEOUT_SECONDS
    ) -> Any:
        request = urllib.request.Request(
            f"{self.base}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._request(request, timeout)

    def upload_image(self, source: Path, name: str, subfolder: str) -> str:
        """Put the source in ComfyUI's input directory under a name we chose.

        The name is generated, never taken from the upload: it becomes a path in
        another application's directory.
        """
        boundary = f"----upscaler{secrets.token_hex(16)}"
        fields = {"subfolder": subfolder, "type": "input", "overwrite": "true"}
        parts: list[bytes] = []
        for key, value in fields.items():
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n'
                f"{value}\r\n".encode()
            )
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
            f'filename="{name}"\r\nContent-Type: image/png\r\n\r\n'.encode()
        )
        parts.append(source.read_bytes())
        parts.append(f"\r\n--{boundary}--\r\n".encode())
        body = b"".join(parts)
        request = urllib.request.Request(
            f"{self.base}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        result = self._request(request, REQUEST_TIMEOUT_SECONDS)
        uploaded = result.get("name", name)
        folder = result.get("subfolder") or ""
        return f"{folder}/{uploaded}" if folder else uploaded


def _queue_is_idle(queue: Any) -> bool:
    if not isinstance(queue, dict):
        return False
    running = queue.get("queue_running")
    pending = queue.get("queue_pending")
    return isinstance(running, list) and isinstance(pending, list) and not running and not pending


def _free_memory_values(stats: Any) -> dict[str, int]:
    if not isinstance(stats, dict):
        return {}
    values: dict[str, int] = {}
    system = stats.get("system")
    if isinstance(system, dict):
        ram_free = system.get("ram_free")
        if isinstance(ram_free, int | float) and ram_free >= 0:
            values["ram"] = int(ram_free)
    devices = stats.get("devices")
    if isinstance(devices, list) and devices and isinstance(devices[0], dict):
        vram_free = devices[0].get("vram_free")
        if isinstance(vram_free, int | float) and vram_free >= 0:
            values["vram"] = int(vram_free)
    return values


def _memory_release_visible(before: Any, after: Any) -> bool:
    previous = _free_memory_values(before)
    current = _free_memory_values(after)
    return any(
        current.get(kind, value) >= value + RECLAIM_MIN_INCREASE_BYTES
        for kind, value in previous.items()
    )


def _memory_still_moving(before: Any, after: Any) -> bool:
    previous = _free_memory_values(before)
    current = _free_memory_values(after)
    return any(
        abs(current.get(kind, value) - value) >= RECLAIM_MIN_INCREASE_BYTES
        for kind, value in previous.items()
    )


def patch_graph(
    template: dict[str, Any],
    workflow: Workflow,
    *,
    image: str,
    target: tuple[int, int] | None,
) -> dict[str, Any]:
    """Write this job's values into the named slots of a copy of the template."""
    graph = json.loads(json.dumps(template))

    def inputs(node_id: str | None) -> dict[str, Any] | None:
        if node_id is None:
            return None
        node = graph.get(node_id)
        return node["inputs"] if node else None

    image_inputs = inputs(workflow.node("image"))
    if image_inputs is None:
        raise ModelExecutionError(
            f"Workflow {workflow.id} has no image slot; its template and catalog disagree."
        )
    image_inputs["image"] = image

    if target is not None:
        width, height = target
        scale_nodes = workflow.nodes("target_scales")
        if not scale_nodes:
            raise ModelExecutionError(
                f"Workflow {workflow.id} claims to enlarge but has no target scale slot."
            )
        for node_id in scale_nodes:
            scale_inputs = inputs(node_id)
            if scale_inputs is None:
                raise ModelExecutionError(
                    f"Workflow {workflow.id} points at missing target scale node {node_id}."
                )
            scale_inputs["width"] = width
            scale_inputs["height"] = height
    return graph


def output_node(template: dict[str, Any]) -> str | None:
    for node_id, node in template.items():
        if node.get("class_type") == "SaveImageWebsocket":
            return node_id
    return None


def _websocket(base: str, client_id: str) -> Any:
    try:
        from websockets.sync.client import connect
    except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
        raise ModelExecutionError(WEBSOCKET_HINT) from exc
    scheme = "wss" if base.startswith("https://") else "ws"
    netloc = base.split("://", 1)[1]
    return connect(
        f"{scheme}://{netloc}/ws?clientId={client_id}",
        max_size=None,
        open_timeout=PROBE_TIMEOUT_SECONDS * 4,
    )


def _overall_fraction(states: dict[str, Any], expected: set[str]) -> float | None:
    """How much of the run is done, counted from ComfyUI's own per-node progress.

    Node-weighted rather than time-weighted, so it is a real measurement of the
    graph rather than an estimate of the clock. It advances unevenly because the
    samplers cost far more than the loaders, but it never runs backwards and it
    never invents motion the server did not report.
    """
    if not expected:
        return None
    done = 0.0
    for node_id in expected:
        state = states.get(node_id)
        if not state:
            continue
        if state.get("state") in {"finished", "error"}:
            done += 1.0
            continue
        maximum = state.get("max") or 0
        if maximum:
            done += min(1.0, (state.get("value") or 0) / maximum)
    return min(1.0, done / len(expected))


def execute_graph(
    client: ComfyClient,
    graph: dict[str, Any],
    result_node: str,
    cancel: Event,
    progress: ProgressCallback,
    label: str,
) -> bytes:
    """Submit the graph, follow it over the websocket, return the encoded result.

    The socket is opened before the prompt is submitted: ComfyUI starts sending
    the moment it accepts, and a socket opened afterwards misses the beginning of
    a run that can be the only thing worth reporting.
    """
    with _websocket(client.base, client.client_id) as socket:
        try:
            submitted = client.post("/prompt", {"prompt": graph, "client_id": client.client_id})
        except urllib.error.HTTPError as exc:
            raise ModelExecutionError(
                f"ComfyUI rejected the workflow: {_fmt_http_error(exc)}"
            ) from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ModelExecutionError(f"Could not submit the workflow to ComfyUI: {exc}") from exc
        prompt_id = submitted.get("prompt_id")
        if not prompt_id:
            raise ModelExecutionError("ComfyUI accepted the workflow without returning a job id.")

        expected = set(graph)
        current: str | None = None
        started = False
        frames: list[bytes] = []
        fraction: float | None = None

        while True:
            if cancel.is_set():
                _abandon(client, prompt_id, started)
                raise ProcessingCancelled("processing was cancelled")
            try:
                message = socket.recv(timeout=0.5)
            except TimeoutError:
                continue
            except Exception as exc:  # noqa: BLE001 - any socket failure ends the job
                raise ModelExecutionError(f"Lost the connection to ComfyUI: {exc}") from exc

            if isinstance(message, bytes | bytearray):
                # Only the result node's images are ours; anything else on this
                # socket is a sampler preview from some other node.
                if current == result_node and len(message) >= BINARY_HEADER.size:
                    event, image_format = BINARY_HEADER.unpack_from(message)
                    if event == BINARY_EVENT_PREVIEW_IMAGE and image_format in BINARY_IMAGE_FORMATS:
                        frames.append(bytes(message[BINARY_HEADER.size :]))
                continue

            try:
                event = json.loads(message)
            except ValueError:
                continue
            kind = event.get("type")
            data = event.get("data") or {}
            if data.get("prompt_id") not in (None, prompt_id):
                continue

            if kind == "status" and not started:
                remaining = (data.get("status") or {}).get("exec_info", {}).get("queue_remaining")
                if remaining:
                    progress(
                        "loading_model",
                        f"Waiting for ComfyUI: {remaining} job(s) in its queue",
                        None,
                    )
            elif kind == "execution_start":
                started = True
                progress("enhancing", f"ComfyUI is running {label}", 0.0)
            elif kind == "execution_cached":
                expected -= set(data.get("nodes") or [])
            elif kind == "executing":
                node = data.get("node")
                if node is None:
                    break
                started = True
                current = node
            elif kind == "progress_state":
                updated = _overall_fraction(data.get("nodes") or {}, expected)
                if updated is not None and (fraction is None or updated >= fraction):
                    fraction = updated
                    progress("enhancing", f"ComfyUI is running {label}", fraction)
            elif kind == "execution_error":
                node_type = data.get("node_type") or "a node"
                detail = data.get("exception_message") or "no detail given"
                raise ModelExecutionError(f"ComfyUI failed in {node_type}: {detail}")
            elif kind == "execution_interrupted":
                raise ProcessingCancelled("ComfyUI interrupted the job")
            elif kind == "execution_success":
                break

    if not frames:
        raise ModelExecutionError(
            "ComfyUI finished without returning an image. Check that the workflow's "
            "output node is still SaveImageWebsocket."
        )
    return frames[-1]


def _abandon(client: ComfyClient, prompt_id: str, started: bool) -> None:
    """Stop a run on the way out, without touching anybody else's.

    /interrupt has no job id and cancels whatever is running, so it is only safe
    once this prompt is known to be the running one. A prompt still queued is
    removed by id instead.
    """
    try:
        if started:
            client.post("/interrupt", {}, timeout=PROBE_TIMEOUT_SECONDS * 2)
        client.post("/queue", {"delete": [prompt_id]}, timeout=PROBE_TIMEOUT_SECONDS * 2)
    except (urllib.error.URLError, OSError, ValueError):
        # Cancellation must not fail the cancellation.
        pass


class ComfyUiIllustrationAdapter:
    """A faithful, pixel-space upscaler for drawings and animation."""

    id = "comfyui-illustration"
    name = "ComfyUI illustration upscaler"
    neural = True
    # The graph stays in pixel space: an upscale model, one exact resize, and
    # the source alpha restored. Nothing here invents detail.
    generative = False
    # The graph reaches the whole target in one submission.
    max_passes = 1
    # No fixed enlargement factor at this level: the graph resamples to whatever
    # size it is given, so the engine is driven by the job's target dimensions
    # rather than by a multiplier.
    native_scales = (1,)
    # The checked-in graph controls its own inference.
    supports_tta = False
    license = "BSD-3-Clause"
    workflow_mode = ProcessingMode.illustration

    def __init__(self) -> None:
        self._lock = Lock()
        self._probed_at = 0.0
        self._probe: tuple[bool, str, str | None] | None = None
        self._system_stats: dict[str, Any] | None = None

    def workflows(self) -> tuple[Workflow, ...]:
        return available_workflows(self.workflow_mode)

    def _reachability(self) -> tuple[bool, str, str | None]:
        """(usable, device, reason). Cached briefly; see PROBE_TTL_SECONDS."""
        with self._lock:
            now = time.monotonic()
            if self._probe is not None and now - self._probed_at < PROBE_TTL_SECONDS:
                return self._probe
            self._probe = self._run_probe()
            self._probed_at = now
            return self._probe

    def _run_probe(self) -> tuple[bool, str, str | None]:
        base, reason = resolve_url()
        if base is None:
            return False, "Unavailable", reason
        try:
            stats = ComfyClient(base).get("/system_stats", timeout=PROBE_TIMEOUT_SECONDS)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return (
                False,
                "Unavailable",
                (f"ComfyUI did not answer at {base}: {exc}. Start it, or correct {ENV_URL}."),
            )
        self._system_stats = stats
        devices = stats.get("devices") or []
        device = devices[0].get("name") if devices and isinstance(devices[0], dict) else None
        label = f"ComfyUI ({device})" if device else "ComfyUI"
        # The mode is unusable without the exact pinned weight, so say which one
        # is missing rather than letting the job fail inside ComfyUI.
        try:
            definition = ComfyClient(base).get(
                "/object_info/UpscaleModelLoader", timeout=PROBE_TIMEOUT_SECONDS
            )
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return False, label, f"Could not inspect ComfyUI's upscale models: {exc}."
        model_input = (
            definition.get("UpscaleModelLoader", {})
            .get("input", {})
            .get("required", {})
            .get("model_name", [])
        )
        choices: list[str] = []
        if isinstance(model_input, list) and model_input:
            if isinstance(model_input[0], list):
                choices = model_input[0]
            elif len(model_input) > 1 and isinstance(model_input[1], dict):
                choices = model_input[1].get("options") or []
        if ILLUSTRATION_MODEL not in choices:
            return (
                False,
                label,
                (
                    f"ComfyUI is missing {ILLUSTRATION_MODEL}. Install the checksum-pinned "
                    "model with `make setup-model-comfyui-illustration "
                    "UPSCALER_COMFYUI_UPSCALE_MODELS_DIR=/path/to/ComfyUI/models/upscale_models`."
                ),
            )
        return True, label, None

    def hardware_stats(self, *, refresh: bool = False) -> dict[str, Any] | None:
        """Return /system_stats, bypassing the UI reachability cache for admission."""
        if refresh:
            result = self._run_probe()
            with self._lock:
                self._probe = result
                self._probed_at = time.monotonic()
            if not result[0]:
                return None
        else:
            self._reachability()
        return self._system_stats

    def reclaim_memory_if_idle(self) -> bool:
        """Ask an idle ComfyUI to release cached models before one admission retry."""
        base, _ = resolve_url()
        if base is None:
            return False
        client = ComfyClient(base)
        with _RECLAIM_LOCK:
            try:
                queue = client.get("/queue", timeout=PROBE_TIMEOUT_SECONDS)
                if not _queue_is_idle(queue):
                    return False
                with self._lock:
                    before = self._system_stats
                if not _free_memory_values(before):
                    before = client.get("/system_stats", timeout=PROBE_TIMEOUT_SECONDS)
                if not _free_memory_values(before):
                    return False
                client.post(
                    "/free",
                    {"unload_models": True, "free_memory": True},
                    timeout=PROBE_TIMEOUT_SECONDS,
                )
                deadline = time.monotonic() + RECLAIM_TIMEOUT_SECONDS
                # ComfyUI acknowledges /free before its worker unloads anything.
                # A GB10 releases a large model in many visible steps, so the
                # first increase is only the start: wait for the pool to settle.
                previous = before
                release_visible = False
                settled_polls = 0
                while True:
                    current = client.get("/system_stats", timeout=PROBE_TIMEOUT_SECONDS)
                    with self._lock:
                        self._system_stats = current
                    release_visible = release_visible or _memory_release_visible(before, current)
                    if release_visible:
                        settled_polls = (
                            0 if _memory_still_moving(previous, current) else settled_polls + 1
                        )
                    if settled_polls >= RECLAIM_SETTLE_POLLS or time.monotonic() >= deadline:
                        return True
                    previous = current
                    time.sleep(RECLAIM_POLL_SECONDS)
            except (urllib.error.URLError, OSError, ValueError):
                # Admission keeps its original failure reason if ComfyUI cannot
                # prove that it is idle or cannot service the release request.
                return False

    @property
    def available(self) -> bool:
        return self._reachability()[0]

    @property
    def unavailable_reason(self) -> str | None:
        return self._reachability()[2]

    @property
    def device(self) -> str:
        return self._reachability()[1]

    def enhance(
        self,
        request: ModelRequest,
        cancel: Event,
        progress: ProgressCallback,
    ) -> ModelResult:
        usable, _, reason = self._reachability()
        if not usable:
            raise ModelExecutionError(reason or "ComfyUI is unavailable")
        base, url_reason = resolve_url()
        if base is None:
            raise ModelExecutionError(url_reason or "ComfyUI is unavailable")
        try:
            workflow = find_workflow(request.workflow, self.workflow_mode)
        except ValueError as exc:
            raise ModelExecutionError(str(exc)) from exc

        if cancel.is_set():
            raise ProcessingCancelled("processing was cancelled")

        template = load_template(workflow)
        result_node = output_node(template)
        if result_node is None:
            raise ModelExecutionError(
                f"Workflow template {workflow.template} has no SaveImageWebsocket output node."
            )

        # Render straight at the job's target. Driving this off native_scale
        # would cap a small source at four times its own size and leave the rest
        # of the enlargement to a plain resample, which is the softness the
        # tiled refine exists to avoid.
        target = (request.target_width, request.target_height)

        warnings: list[str] = [workflow.warning]
        if request.tile_size:
            warnings.append(
                "The tile size setting does not apply to ComfyUI: its tiling is part of the "
                "workflow, tuned to the resolution the model was trained at."
            )
        if request.tta:
            warnings.append(
                "Test-time augmentation does not apply to ComfyUI: the checked-in workflow "
                "controls its own inference."
            )

        client = ComfyClient(base)
        progress("loading_model", f"Sending the source to ComfyUI for {workflow.name}", None)
        upload_name = f"upscaler-{secrets.token_hex(12)}.png"
        try:
            reference = client.upload_image(request.source_path, upload_name, "upscaler")
        except urllib.error.HTTPError as exc:
            raise ModelExecutionError(
                f"ComfyUI rejected the upload: {_fmt_http_error(exc)}"
            ) from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ModelExecutionError(f"Could not send the image to ComfyUI: {exc}") from exc

        try:
            graph = patch_graph(
                template,
                workflow,
                image=reference,
                target=target if workflow.enlarges else None,
            )
            encoded = execute_graph(client, graph, result_node, cancel, progress, workflow.name)
        finally:
            _delete_upload(reference)

        progress("finishing", "Reading the result back from ComfyUI", None)
        request.output_path.write_bytes(encoded)
        # Every graph here returns the target size, so the width of the file is
        # not where its detail was made. Reporting that width keeps the
        # finishing sharpen sized to the stretch that actually happened, whether
        # this adapter did it below or the graph did it internally.
        detail_width = 0
        if not workflow.enlarges:
            # The graph worked at its own resolution, so the enlargement to the
            # requested size is an ordinary resample. Saying so keeps the result
            # from reading as generated detail at 4K.
            with Image.open(request.output_path) as produced:
                produced.load()
                detail_width = produced.width
                enlarged = resize_exact(produced, target)
            enlarged.save(request.output_path, format="PNG", compress_level=2)
            warnings.append(
                f"{workflow.name} works at its own resolution and does not enlarge; reaching "
                "the target was a plain Lanczos resample, not generated detail."
            )
        elif workflow.model_scale:
            with Image.open(request.source_path) as opened:
                detail_width = opened.width * workflow.model_scale
        return ModelResult(
            output_path=request.output_path,
            engine_id=f"{self.id}:{workflow.id}",
            warnings=tuple(warnings),
            detail_width=detail_width,
        )


def _delete_upload(reference: str) -> None:
    """Remove the source from ComfyUI's input directory when we can reach it.

    ComfyUI has no API for deleting an input, so this only works when the user
    points UPSCALER_COMFYUI_INPUT_DIR at the directory. Without it the file stays,
    which is why docs/engines.md says so rather than leaving it to be discovered.
    """
    configured = os.environ.get(ENV_INPUT_DIR, "").strip()
    if not configured:
        return
    root = Path(configured).resolve()
    candidate = (root / reference).resolve()
    if root not in candidate.parents:
        return
    with contextlib.suppress(OSError):
        candidate.unlink(missing_ok=True)
