"""Erase the pictures and prompts an ordinary session leaves behind.

Job workspaces outlive the process that made them: the retention sweep walks the
in-memory job dict, so a directory whose process has exited is invisible to it
forever. ComfyUI mode adds traces in another application's directories. This
module finds all of it, counts it, and removes it only once the caller has said
yes.

Collection and deletion share one list of targets so the preview and the removal
can never disagree about what is about to go.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from upscaler.config import load_config

# Job scratch space in the container build, named in docker-compose.yml. The
# weights volume beside it is deliberately never a target: those are models, not
# anything a user's picture went into.
DOCKER_JOB_VOLUME = "upscaler_work"

COMFYUI_WIPE_DIRS = ("input", "output", "temp")
# Shipped by ComfyUI rather than produced by a run. Removing them would only
# break its bundled examples without erasing anything of the user's.
COMFYUI_KEEP = frozenset({"example.png", "_output_images_will_be_put_here", "3d"})
COMFYUI_WORKFLOWS = Path("user") / "default" / "workflows"

REQUEST_TIMEOUT_SECONDS = 10.0

# The canvas, including prompts typed but never saved, lives in the browser. No
# command here can reach it, so the caller is told rather than left to assume.
BROWSER_NOTICE = (
    "ComfyUI also keeps the live canvas and unsaved prompts in your browser's local\n"
    "storage, which no command can reach from here. To clear that too, open ComfyUI\n"
    'and use the browser\'s "clear site data", or devtools > Application > Local\n'
    "Storage and remove the Comfy.Workflow.* , workflow and litegrapheditor_clipboard\n"
    "keys."
)

UNKNOWN = -1


class CleanupRefused(RuntimeError):
    """Raised when a precondition makes deletion unsafe."""


@dataclass(frozen=True, slots=True)
class Target:
    """One removable thing, measured before anything is deleted."""

    label: str
    detail: str
    files: int = UNKNOWN
    size: int = UNKNOWN
    paths: tuple[Path, ...] = field(default=())
    volume: str = ""
    endpoint: str = ""


def human_bytes(size: int) -> str:
    if size < 0:
        return "size needs root"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _contained(root: Path, candidate: Path) -> bool:
    """Whether candidate really sits under root once symlinks are resolved.

    A symlinked output directory must not let a wipe escape into the rest of the
    filesystem, which is the same reason the ComfyUI adapter checks its uploads.
    """
    try:
        resolved = candidate.resolve()
    except OSError:
        return False
    return root in resolved.parents


def _measure(paths: Iterable[Path]) -> tuple[int, int]:
    files = 0
    size = 0
    for path in paths:
        if path.is_symlink():
            files += 1
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and not child.is_symlink():
                    files += 1
                    size += child.stat().st_size
        elif path.is_file():
            files += 1
            size += path.stat().st_size
    return files, size


def _paths_target(label: str, directory: Path, children: Sequence[Path]) -> Target | None:
    if not children:
        return None
    files, size = _measure(children)
    return Target(
        label=label,
        detail=str(directory),
        files=files,
        size=size,
        paths=tuple(children),
    )


def app_address() -> tuple[str, int]:
    """The address the app serves on, resolved the way __main__ resolves it."""
    host = os.getenv("UPSCALER_HOST", "127.0.0.1").strip() or "127.0.0.1"
    raw_port = os.getenv("UPSCALER_PORT", "8000").strip() or "8000"
    try:
        port = int(raw_port)
    except ValueError:
        port = 8000
    return host, port


def app_is_running() -> bool:
    host, port = app_address()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((host, port)) == 0


def comfyui_root(override: str | None = None) -> Path | None:
    """Where ComfyUI is installed.

    The adapter's own variables are set on the command that launches the app, so
    a shell running `make clean-data` usually carries none of them. The explicit
    override exists so the cleanup does not depend on inheriting them.
    """
    configured = (override or "").strip() or os.getenv("UPSCALER_COMFYUI_ROOT", "").strip()
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        input_dir = os.getenv("UPSCALER_COMFYUI_INPUT_DIR", "").strip()
        if not input_dir:
            return None
        root = Path(input_dir).expanduser().resolve().parent
    return root if root.is_dir() else None


NO_COMFYUI_NOTICE = (
    "ComfyUI was NOT cleaned: no installation is configured, so its input, output,\n"
    "temp, saved workflows, history and queue were all left alone. Point at it with\n"
    "  make clean-data COMFYUI=/path/to/ComfyUI\n"
    "or export UPSCALER_COMFYUI_ROOT. Its temp directory is also emptied whenever\n"
    "ComfyUI itself restarts."
)


def is_comfyui_install(root: Path) -> bool:
    """Guard against a mistyped variable pointing the wipe at an ordinary folder."""
    return (root / "main.py").is_file() and (root / "comfy").is_dir()


def collect_workspaces(work_root: Path) -> Target | None:
    """Every per-job directory, leaving the root itself in place."""
    if not work_root.is_dir():
        return None
    root = work_root.resolve()
    children = [child for child in sorted(work_root.iterdir()) if _contained(root, child)]
    return _paths_target("Job workspaces", work_root, children)


def _wipe_directories(root: Path) -> list[tuple[str, Path]]:
    return [
        *((f"ComfyUI {name}", root / name) for name in COMFYUI_WIPE_DIRS),
        ("ComfyUI saved workflows", root / COMFYUI_WORKFLOWS),
    ]


def skipped_symlinks(root: Path) -> list[Path]:
    """Wipe directories that are themselves symlinks, which are never followed.

    Resolving one and deleting what is inside would empty whatever it points at,
    which is a plausible setup (a scratch directory moved to a faster disk) and a
    catastrophic mistake to get wrong. They are reported and left alone instead.
    """
    return [directory for _label, directory in _wipe_directories(root) if directory.is_symlink()]


def collect_comfyui(root: Path) -> list[Target]:
    if not is_comfyui_install(root):
        raise CleanupRefused(
            f"{root} does not look like a ComfyUI installation (no main.py and no comfy/). "
            "Set UPSCALER_COMFYUI_ROOT to the right directory."
        )
    targets: list[Target] = []
    for label, directory in _wipe_directories(root):
        # A real directory's own entries are inside it by construction, and both
        # the measure and the delete unlink a symlink rather than following it,
        # so a link pointing outward loses only the link.
        if directory.is_symlink() or not directory.is_dir():
            continue
        children = [
            child for child in sorted(directory.iterdir()) if child.name not in COMFYUI_KEEP
        ]
        target = _paths_target(label, directory, children)
        if target:
            targets.append(target)
    return targets


def collect_docker_volume(volume: str = DOCKER_JOB_VOLUME) -> Target | None:
    """The container build's job volume, which `make down` deliberately keeps."""
    if shutil.which("docker") is None:
        return None
    probe = subprocess.run(
        ["docker", "volume", "inspect", volume],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return None
    # Its mountpoint lives under /var/lib/docker and is unreadable without root,
    # so the size is reported as unknown rather than guessed.
    return Target(label="Docker job volume", detail=volume, volume=volume)


def collect_comfyui_state(base_url: str) -> Target | None:
    """ComfyUI's run history and queue, which are memory and not files."""
    if not base_url:
        return None
    return Target(label="ComfyUI history and queue", detail=base_url, endpoint=base_url)


def collect(*, include_docker: bool = True, root: Path | None = None) -> list[Target]:
    targets: list[Target] = []
    workspaces = collect_workspaces(load_config().work_root)
    if workspaces:
        targets.append(workspaces)

    comfyui_work_root = os.getenv("UPSCALER_COMFYUI_WORK_ROOT", "").strip()
    if comfyui_work_root:
        comfyui_workspaces = collect_workspaces(Path(comfyui_work_root).expanduser())
        if comfyui_workspaces:
            targets.append(
                Target(
                    label="ComfyUI container workspaces",
                    detail=comfyui_workspaces.detail,
                    files=comfyui_workspaces.files,
                    size=comfyui_workspaces.size,
                    paths=comfyui_workspaces.paths,
                )
            )

    if root is not None:
        targets.extend(collect_comfyui(root))
        base_url = os.getenv("UPSCALER_COMFYUI_URL", "").strip()
        state = collect_comfyui_state(base_url)
        if state:
            targets.append(state)

    if include_docker:
        volume = collect_docker_volume()
        if volume:
            targets.append(volume)
    return targets


def format_report(targets: Sequence[Target]) -> str:
    if not targets:
        return "Nothing to remove."
    width = max(len(target.label) for target in targets)
    lines = ["Would remove:"]
    total = 0
    for target in targets:
        if target.endpoint:
            amount = "cleared over the API"
        elif target.volume:
            amount = "removed whole, size needs root"
        else:
            amount = f"{target.files} files, {human_bytes(target.size)}"
            total += target.size
        lines.append(f"  {target.label.ljust(width)}  {target.detail}")
        lines.append(f"  {' '.ljust(width)}  {amount}")
    lines.append(f"\nTotal measurable: {human_bytes(total)}")
    return "\n".join(lines)


def clear_comfyui_state(base_url: str) -> bool:
    """Wipe ComfyUI's history and queue over its API.

    Both live only in memory, so an unreachable ComfyUI has already forgotten
    them and a failure here is worth reporting but not worth failing over.
    """
    payload = json.dumps({"clear": True}).encode()
    ok = True
    for path in ("/history", "/queue"):
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}{path}",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS):
                pass
        except (urllib.error.URLError, OSError, TimeoutError):
            ok = False
    return ok


def remove_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def remove_volume(volume: str) -> bool:
    result = subprocess.run(
        ["docker", "volume", "rm", volume],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def remove(targets: Sequence[Target]) -> list[str]:
    """Delete every target, returning the problems worth telling the user about."""
    problems: list[str] = []
    for target in targets:
        if target.endpoint:
            if not clear_comfyui_state(target.endpoint):
                problems.append(
                    f"Could not reach ComfyUI at {target.endpoint}; its history and queue "
                    "are held in memory, so they are already gone if it is not running."
                )
        elif target.volume:
            if not remove_volume(target.volume):
                problems.append(
                    f"Could not remove the Docker volume {target.volume}; it is still in "
                    "use if a container is running. Try `make down` first."
                )
        else:
            remove_paths(target.paths)
    return problems
