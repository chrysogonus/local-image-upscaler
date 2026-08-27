#!/usr/bin/env python3
"""Erase the pictures and prompts this app has left on disk.

Argument parsing and the confirmation only; the work lives in
upscaler.maintenance so it can be tested without a subprocess.
"""

from __future__ import annotations

import argparse
import sys

from upscaler.maintenance import (
    BROWSER_NOTICE,
    NO_COMFYUI_NOTICE,
    CleanupRefused,
    app_address,
    app_is_running,
    collect,
    comfyui_root,
    format_report,
    remove,
    skipped_symlinks,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be removed and exit without deleting anything",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the confirmation prompt (for scripted use)",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="leave the container build's job volume alone",
    )
    parser.add_argument(
        "--comfyui",
        metavar="PATH",
        default=None,
        help="the ComfyUI installation to clean, when the environment does not name one",
    )
    args = parser.parse_args(argv)

    if not args.dry_run and app_is_running():
        host, port = app_address()
        print(
            f"The app is serving on {host}:{port}. Stop it first, so nothing is deleted "
            "from under a running job.",
            file=sys.stderr,
        )
        return 1

    root = comfyui_root(args.comfyui)
    try:
        targets = collect(include_docker=not args.no_docker, root=root)
    except CleanupRefused as error:
        print(str(error), file=sys.stderr)
        return 2

    print(format_report(targets))

    # Skipping ComfyUI quietly would let the command report success while every
    # picture and prompt it holds stayed exactly where it was.
    if root is None:
        print(f"\n{NO_COMFYUI_NOTICE}", file=sys.stderr)

    for directory in skipped_symlinks(root) if root is not None else []:
        print(
            f"\nSkipped {directory}: it is a symlink, and emptying what it points at is "
            "not a decision this command should make. Remove its contents yourself if "
            "you want it gone.",
            file=sys.stderr,
        )

    if not targets:
        return 0
    if args.dry_run:
        return 0

    if not args.yes:
        answer = input("\nDelete these permanently? [y/N] ").strip().lower()
        if answer != "y":
            # Declining is a valid choice, not a failure: a non-zero exit here
            # only makes make report an alarming error over a deliberate no.
            print("Nothing was removed.")
            return 0

    problems = remove(targets)
    print("\nRemoved.")
    for problem in problems:
        print(f"  note: {problem}")
    print(f"\n{BROWSER_NOTICE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
