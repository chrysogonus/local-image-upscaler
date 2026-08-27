#!/usr/bin/env python3
"""Download the single-file model weights declared in the manifest.

Manifest-driven and platform independent: these are architecture-neutral
tensors, unlike the x86-64-only NCNN runtime installed by
``install-realesrgan-linux.sh``. Entries are grouped so each optional feature
installs only what it needs - ``realesrgan`` for the CUDA engine, ``swinir`` for
the transformer one - and each group has its own install directory, matching the
one its adapter reads.

Checksums live in ``models/manifest.json``. An entry with a recorded ``sha256``
is verified and the download is discarded on mismatch. An entry with ``null``
has never been pinned on this checkout; the digest is printed and the file is
rejected unless ``--pin`` is passed, which records it for every later install.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPOSITORY_ROOT / "models" / "manifest.json"
CHUNK = 1024 * 1024


# Each group's directory has to match what the corresponding adapter reads, or
# weights install successfully and the feature still reports itself missing.
GROUPS = {
    "comfyui-illustration": (
        "UPSCALER_COMFYUI_UPSCALE_MODELS_DIR",
        "comfyui-upscale-models",
    ),
    "realesrgan": ("UPSCALER_REALESRGAN_WEIGHTS_DIR", "realesrgan-torch"),
    "swinir": ("UPSCALER_SR_WEIGHTS_DIR", "spandrel-sr"),
}


def default_target(group: str) -> Path:
    variable, directory = GROUPS[group]
    configured = os.getenv(variable)
    if configured:
        return Path(configured).expanduser().resolve()
    return REPOSITORY_ROOT / ".upscaler" / directory


def download(url: str, destination: Path) -> str:
    if not url.startswith("https://"):
        raise ValueError(f"refusing to download over a non-HTTPS URL: {url}")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response:  # noqa: S310 - scheme checked above
        total = int(response.headers.get("Content-Length") or 0)
        read = 0
        with destination.open("wb") as handle:
            while chunk := response.read(CHUNK):
                handle.write(chunk)
                digest.update(chunk)
                read += len(chunk)
                if total:
                    print(f"\r  {read * 100 // total:3d}%  {read >> 20} MiB", end="", flush=True)
    print("\r" + " " * 32 + "\r", end="")
    return digest.hexdigest()


def install(entry: dict, target: Path, pin: bool) -> str | None:
    """Return the digest to pin, or None if nothing needs recording."""
    filename = entry["filename"]
    final = target / filename
    expected = entry.get("sha256")

    if final.is_file():
        actual = hashlib.sha256(final.read_bytes()).hexdigest()
        if expected and actual != expected:
            raise SystemExit(f"{filename} is already installed but its checksum does not match.")
        print(f"{filename}: already installed")
        return actual if not expected and pin else None

    print(f"{filename}: downloading {entry['url']}")
    target.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(dir=target, suffix=".partial")
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        actual = download(entry["url"], temporary)
        if expected:
            if actual != expected:
                raise SystemExit(
                    f"{filename}: checksum mismatch.\n"
                    f"  expected {expected}\n  actual   {actual}\n"
                    "The download was discarded."
                )
        elif not pin:
            raise SystemExit(
                f"{filename}: no checksum is pinned in models/manifest.json.\n"
                f"  computed sha256: {actual}\n"
                "Verify this against the upstream release, then re-run with --pin to record it."
            )
        temporary.replace(final)
        final.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"{filename}: installed and verified ({entry['license']})")
    return actual if not expected else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pin",
        action="store_true",
        help="record computed checksums for entries that have none yet",
    )
    parser.add_argument(
        "--group",
        default="realesrgan",
        choices=sorted(GROUPS),
        help="which manifest group to install (default: realesrgan)",
    )
    parser.add_argument("--dir", type=Path, default=None, help="install directory")
    arguments = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = [
        entry
        for entry in manifest.get("weights", [])
        if entry.get("group", "realesrgan") == arguments.group
    ]
    if not entries:
        print(
            f"No {arguments.group} weights are declared in models/manifest.json",
            file=sys.stderr,
        )
        return 1

    target = (arguments.dir or default_target(arguments.group)).expanduser().resolve()
    pinned = False
    for entry in entries:
        digest = install(entry, target, arguments.pin)
        if digest:
            entry["sha256"] = digest
            pinned = True
            print(f"{entry['filename']}: pinned sha256 {digest}")

    if pinned:
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("\nUpdated models/manifest.json. Commit it so later installs are verified.")

    print(f"\nWeights are in {target}")
    print("The backend picks them up without a restart.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
