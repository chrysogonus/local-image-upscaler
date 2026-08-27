#!/usr/bin/env python3
"""Author the faithful illustration-upscale graph as a ComfyUI workflow."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

from comfy_workflow_builder import build_workflow

MODEL = "RealESRGAN_x4plus_anime_6B.pth"

# (id, class, title, {input: (source_id, slot)}, {widget: value})
GRAPH: list[tuple[int, str, str, dict[str, tuple[int, int]], dict[str, Any]]] = [
    (1, "LoadImage", "1. Your illustration", {}, {"image": "example.png"}),
    (2, "UpscaleModelLoader", "2. Illustration model", {}, {"model_name": MODEL}),
    (
        3,
        "ImageUpscaleWithModel",
        "3. Faithful pixel-space x4 reconstruction",
        {"upscale_model": (2, 0), "image": (1, 0)},
        {},
    ),
    (
        4,
        "ImageScale",
        "4. One exact resize to the requested dimensions",
        {"image": (3, 0)},
        {
            "upscale_method": "lanczos",
            "width": 3840,
            "height": 2160,
            "crop": "disabled",
        },
    ),
    (5, "MaskToImage", "Preserve the source transparency", {"mask": (1, 1)}, {}),
    (
        6,
        "ImageScale",
        "Scale transparency to the same exact dimensions",
        {"image": (5, 0)},
        {
            "upscale_method": "lanczos",
            "width": 3840,
            "height": 2160,
            "crop": "disabled",
        },
    ),
    (7, "ImageToMask", "Restore transparency mask", {"image": (6, 0)}, {"channel": "red"}),
    (
        8,
        "JoinImageWithAlpha",
        "Restore transparency without flattening",
        {"image": (4, 0), "alpha": (7, 0)},
        {},
    ),
    (9, "SaveImageWebsocket", "RESULT", {"images": (8, 0)}, {}),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    args = parser.parse_args()
    with urllib.request.urlopen(f"{args.url}/object_info", timeout=30) as response:  # noqa: S310
        object_info = json.load(response)
    workflow = build_workflow(GRAPH, object_info, "illustration-upscale")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(workflow, indent=1) + "\n")
    print(f"wrote {args.output} ({len(GRAPH)} nodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
