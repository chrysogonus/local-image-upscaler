#!/usr/bin/env python3
"""Build the wheel and smoke-test the artifact in an isolated environment."""

from __future__ import annotations

import subprocess
import tempfile
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="upscaler-package-") as temporary:
        output = Path(temporary)
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(output)],
            cwd=REPOSITORY_ROOT,
            check=True,
        )
        wheels = list(output.glob("*.whl"))
        if len(wheels) != 1:
            raise SystemExit(f"expected one wheel, found {len(wheels)}")
        wheel = wheels[0]

        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
        required = {
            "upscaler/benchmark/dataset.json",
            "upscaler/workflows/catalog.json",
            "upscaler/workflows/illustration-upscale.api.json",
        }
        missing = required - names
        if missing:
            raise SystemExit(f"wheel is missing packaged runtime data: {sorted(missing)}")
        license_files = {Path(name).name for name in names if ".dist-info/licenses/" in name}
        missing_licenses = {"LICENSE", "NOTICE"} - license_files
        if missing_licenses:
            raise SystemExit(
                f"wheel is missing dist-info licence files: {sorted(missing_licenses)}"
            )

        subprocess.run(
            [
                "uv",
                "run",
                "--isolated",
                "--no-project",
                "--with",
                str(wheel),
                "python",
                "-c",
                (
                    "from upscaler import __version__; "
                    "from upscaler.app import create_app; "
                    "from upscaler.benchmark.dataset import load_dataset; "
                    "assert __version__ == '0.3.0'; "
                    "assert create_app().title == 'Local Image Upscaler'; "
                    "assert len(load_dataset().cases) == 12"
                ),
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
        )
        print(f"wheel smoke test passed: {wheel.name} ({len(names)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
