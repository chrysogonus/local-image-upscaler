#!/usr/bin/env python3
"""Install or adopt the ComfyUI the Illustration mode drives.

ComfyUI is a separate application with its own models and its own queue. This
project treats it as a local worker, but a worker nobody can start is not much
use, so this command makes one exist: it finds an installation already on the
machine, or clones the pinned one from ``models/manifest.json``, makes sure its
virtualenv can run it, installs the checksum-pinned illustration model, and
records where it all ended up.

That record, ``.upscaler/comfyui.conf``, is the point. Every later command -
``make up``, ``make down``, ``make clean-data`` - reads it instead of asking the
user to repeat a path they already chose here.

Adopting is deliberately conservative. An installation that already exists is
verified and recorded but never checked out to the pinned revision: moving
somebody's working tree, possibly onto an older commit than the custom nodes
they have installed expect, is not a decision this command gets to make. The pin
governs a fresh clone, and the record says which of the two happened.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPOSITORY_ROOT / "models" / "manifest.json"
STATE_PATH = REPOSITORY_ROOT / ".upscaler" / "comfyui.conf"
DEFAULT_PORT = 8188

# Searched in order. The first is where a fresh clone lands, the second is where
# ComfyUI's own installer has historically put one.
CANDIDATE_ROOTS = (Path.home() / "ComfyUI", Path.home() / "comfy" / "ComfyUI")


def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    """Run a command as an argument list, never through a shell."""
    printable = " ".join(command)
    print(f"  $ {printable}")
    return subprocess.run(command, check=True, text=True, **kwargs)  # type: ignore[call-overload]


def looks_like_comfyui(root: Path) -> bool:
    return (root / "main.py").is_file() and (root / "requirements.txt").is_file()


def head_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def read_state() -> dict[str, str]:
    """Parse the KEY=value record.

    Deliberately not JSON: comfyui-service.sh reads the same file from POSIX
    shell, and a format it can read with sed is one it never has to execute.
    """
    state: dict[str, str] = {}
    try:
        text = STATE_PATH.read_text(encoding="utf-8")
    except OSError:
        return state
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            state[key.strip()] = value.strip()
    return state


def find_existing(explicit: Path | None) -> Path | None:
    """The first usable installation: an explicit one, the recorded one, then the defaults."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(explicit)
    recorded = read_state().get("COMFYUI_ROOT")
    if isinstance(recorded, str) and recorded:
        candidates.append(Path(recorded))
    candidates.extend(CANDIDATE_ROOTS)

    for candidate in candidates:
        root = candidate.expanduser()
        if looks_like_comfyui(root):
            return root.resolve()
    if explicit and explicit.expanduser().exists():
        raise SystemExit(
            f"{explicit} exists but does not look like ComfyUI "
            "(no main.py and requirements.txt). Refusing to install into it."
        )
    return None


def clone(entry: dict[str, object], target: Path) -> Path:
    revision = str(entry["revision"])
    print(f"Cloning {entry['repo_url']} into {target}")
    print(f"  pinned revision {revision}")
    target.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", str(entry["repo_url"]), str(target)])
    run(["git", "-C", str(target), "checkout", "--quiet", revision])
    return target.resolve()


def ensure_venv(root: Path) -> Path:
    python = root / ".venv" / "bin" / "python"
    if python.is_file():
        return python
    print(f"Creating a virtualenv in {root / '.venv'}")
    run([sys.executable, "-m", "venv", str(root / ".venv")])
    if not python.is_file():
        raise SystemExit(f"virtualenv creation did not produce {python}")
    return python


def pip_install(python: Path, arguments: list[str]) -> None:
    """Install into ComfyUI's virtualenv, whatever tool created it.

    A virtualenv made by `uv venv` contains no pip at all, which is exactly what
    an adopted ComfyUI is likely to be, so `python -m pip` is not a safe
    assumption. uv can install into any interpreter and is already required to
    run this command; ensurepip is the last resort for a venv that has neither.
    """
    if shutil.which("uv"):
        run(["uv", "pip", "install", "--python", str(python), *arguments])
        return
    if not module_present(python, "pip"):
        run([str(python), "-m", "ensurepip", "--upgrade"])
    run([str(python), "-m", "pip", "install", *arguments])


def module_present(python: Path, module: str) -> bool:
    return (
        subprocess.run(
            [str(python), "-c", f"import {module}"],
            capture_output=True,
        ).returncode
        == 0
    )


def ensure_requirements(root: Path, python: Path, entry: dict[str, object]) -> None:
    """torch first from the pinned index, then ComfyUI's own requirements.

    ComfyUI's requirements.txt does not pin torch, on the reasoning that the
    right wheel depends on the machine. That leaves choosing one to whoever
    installs it, which is this command; the index in the manifest matches the
    CUDA runtime the application image itself is built on.
    """
    if module_present(python, "torch"):
        print("  torch is already installed, leaving it alone")
    else:
        index = str(entry["torch_index"])
        print(f"Installing torch from {index} (this is a multi-gigabyte download)")
        pip_install(python, ["--index-url", index, "torch", "torchvision"])

    print("Installing ComfyUI's requirements")
    pip_install(python, ["-r", str(root / "requirements.txt")])


def ensure_model(root: Path, manifest: dict[str, object]) -> str:
    """Reuse the checksum-pinned installer rather than repeating its verification."""
    weights = [
        entry
        for entry in manifest.get("weights", [])  # type: ignore[union-attr]
        if entry.get("group") == "comfyui-illustration"
    ]
    if not weights:
        raise SystemExit("models/manifest.json declares no comfyui-illustration weight")
    print("Installing the pinned illustration model")
    run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "install-weights.py"),
            "--group",
            "comfyui-illustration",
            "--dir",
            str(root / "models" / "upscale_models"),
        ]
    )
    return str(weights[0]["filename"])


def write_state(root: Path, adopted: bool, port: int, model: str) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "COMFYUI_ROOT": str(root),
        "COMFYUI_PORT": str(port),
        "COMFYUI_REVISION": head_revision(root) or "unknown",
        "COMFYUI_ADOPTED": "1" if adopted else "0",
        # Named here so the preflight can check for it without parsing JSON.
        "COMFYUI_MODEL": model,
    }
    STATE_PATH.write_text("".join(f"{k}={v}\n" for k, v in state.items()), encoding="utf-8")
    os.chmod(STATE_PATH, 0o600)
    print(f"\nRecorded in {STATE_PATH.relative_to(REPOSITORY_ROOT)}:")
    for key, value in state.items():
        print(f"  {key}={value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="where ComfyUI is, or where to clone it (default: search, then ~/ComfyUI)",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"port to run on (default: {DEFAULT_PORT})"
    )
    arguments = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = [e for e in manifest.get("applications", []) if e.get("id") == "comfyui"]
    if not entries:
        print("models/manifest.json declares no comfyui application", file=sys.stderr)
        return 1
    entry = entries[0]

    if shutil.which("git") is None:
        print("git is required to install ComfyUI and was not found", file=sys.stderr)
        return 1

    existing = find_existing(arguments.dir)
    if existing is not None:
        adopted = True
        root = existing
        print(f"Adopting the ComfyUI already installed at {root}")
        revision = head_revision(root)
        if revision and revision != entry["revision"]:
            print(
                f"  It is at {revision[:12]}, not the pinned {str(entry['revision'])[:12]}.\n"
                "  Leaving it there: moving your working tree is not this command's call."
            )
    else:
        adopted = False
        root = clone(entry, (arguments.dir or CANDIDATE_ROOTS[0]).expanduser())

    python = ensure_venv(root)
    ensure_requirements(root, python, entry)
    model = ensure_model(root, manifest)
    (root / "input").mkdir(exist_ok=True)
    write_state(root, adopted, arguments.port, model)

    print("\nComfyUI is ready. `make up` will start it with the application.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
