from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .dataset import BenchmarkDataError, benchmark_root, prepare_dataset
from .review import BenchmarkReviewError, generate_report, generate_review
from .runner import BenchmarkRunError, run_benchmark


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="upscaler-benchmark",
        description="Run the local faithful-upscaler perceptual-quality benchmark.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Benchmark data root (default: .upscaler/benchmarks).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("prepare", help="Download and materialize the pinned dataset.")

    run = commands.add_parser("run", help="Run all real benchmark candidates.")
    run.add_argument(
        "--run-dir",
        type=Path,
        help="Create or resume this run directory instead of generating an id.",
    )

    review = commands.add_parser("review", help="Generate a new blinded local review page.")
    review.add_argument("run_dir", type=Path)

    report = commands.add_parser("report", help="Merge complete sessions into a report.")
    report.add_argument("run_dir", type=Path)
    report.add_argument("sessions", type=Path, nargs="+")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = benchmark_root(args.root)
    try:
        if args.command == "prepare":
            result = prepare_dataset(root)
        elif args.command == "run":
            result = run_benchmark(root, run_dir=args.run_dir)
        elif args.command == "review":
            result = generate_review(args.run_dir)
        else:
            result = generate_report(args.run_dir, args.sessions)
    except (BenchmarkDataError, BenchmarkRunError, BenchmarkReviewError, ValueError) as exc:
        print(f"upscaler-benchmark: {exc}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover - console entry point is tested through main
    raise SystemExit(main())
