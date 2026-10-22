#!/usr/bin/env python
"""Measure the system.

Two modes:

``--synthetic``
    Runs the tracker over a generated sequence. The numbers verify the tracking
    logic; they do **not** measure a detector.

``--video <file>``
    Processes a real video. The runtime numbers are measured. Tracking metrics
    are only reported when an annotation file is supplied.

Examples::

    python scripts/benchmark.py --synthetic
    python scripts/benchmark.py --video clip.mp4 --annotations gt.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.evaluation.benchmark import (  # noqa: E402
    run_synthetic_benchmark,
    run_video_benchmark,
    write_report,
)
from src.utils.logging import setup_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/benchmark.py",
        description="Measure the tracking system.",
    )
    parser.add_argument("--config", default="")
    parser.add_argument("--synthetic", action="store_true", help="run the synthetic sequence")
    parser.add_argument("--video", default="", help="run on a real video file")
    parser.add_argument("--annotations", default="", help="ground truth JSON for --video")
    parser.add_argument("--frames", type=int, default=300, help="length of the synthetic sequence")
    parser.add_argument("--max-frames", type=int, default=0, help="limit the frames of the video run")
    parser.add_argument("--report", default="", help="write the result as JSON to this path")
    args = parser.parse_args(argv)

    config = load_config(args.config or None)
    setup_logging(config.logging.level)

    if args.video:
        result = run_video_benchmark(
            config,
            args.video,
            annotations_path=args.annotations or None,
            max_frames=args.max_frames,
        )
    elif args.synthetic:
        result = run_synthetic_benchmark(config, frames=args.frames)
    else:
        parser.error("either --synthetic or --video is required")

    print(result.summary())
    if args.report:
        write_report(result, args.report)
        print(f"\nreport written to {args.report}")
    else:
        print("\n" + json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
