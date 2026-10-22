"""Command line entry point.

Examples::

    # validate the configuration and print it
    python -m src.main --check-config

    # run with the default configuration (camera 0)
    python -m src.main

    # run on a video file with the Haar fallback (development only)
    python -m src.main --backend haar --source video --video clip.mp4

    # inspect a model before using it
    python -m src.inspect_model models/yolox_face.onnx
"""

from __future__ import annotations

import argparse
import logging
import sys

from .camera.capture import SourceError
from .config import ConfigError, load_config
from .detection.base import DetectorError
from .pipeline import Pipeline
from .recognition.arcface import IdentityError
from .utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="YOLOX persistent face tracking with a selectable target lock.",
    )
    parser.add_argument("--config", default="", help="path to the YAML configuration")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate the configuration, print it and exit",
    )
    parser.add_argument(
        "--inspect-model",
        default="",
        help="inspect an ONNX model and exit",
    )
    parser.add_argument(
        "--backend",
        default="",
        choices=("yolox_onnx", "haar"),
        help="detector backend (overrides the configuration)",
    )
    parser.add_argument("--model", default="", help="path to the ONNX model")
    parser.add_argument(
        "--output-format",
        default="",
        choices=("yolox_raw", "yolox_decoded", "yolox_dfl", "xyxy"),
        help="output layout of the ONNX export",
    )
    parser.add_argument("--num-classes", type=int, default=0, help="number of model classes")
    parser.add_argument(
        "--source",
        default="",
        choices=("camera", "video"),
        help="where the frames come from",
    )
    parser.add_argument("--camera-index", type=int, default=-1, help="camera index")
    parser.add_argument("--video", default="", help="path to a video file")
    parser.add_argument(
        "--target-track-id",
        type=int,
        default=0,
        help="select this track ID as the target",
    )
    parser.add_argument(
        "--target-mode",
        default="",
        choices=("none", "largest", "track_id", "identity"),
        help="how the target is selected",
    )
    parser.add_argument(
        "--target-name",
        default="",
        help="the enrolled person to follow (with --target-mode identity)",
    )
    parser.add_argument(
        "--identity",
        dest="identity",
        action="store_true",
        default=None,
        help="enable the identity layer (ArcFace): recognise the target when "
        "it comes back into view",
    )
    parser.add_argument(
        "--no-identity",
        dest="identity",
        action="store_false",
        help="disable the identity layer",
    )
    parser.add_argument(
        "--identity-model",
        default="",
        help="path to the ArcFace ONNX embedder",
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=-1.0,
        help="minimum face similarity to accept a candidate as the target",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="do not open a window (headless)",
    )
    parser.add_argument("--max-frames", type=int, default=-1, help="stop after N frames")
    parser.add_argument("--log-level", default="", help="DEBUG, INFO, WARNING, ERROR")
    return parser


def apply_overrides(config, args: argparse.Namespace) -> None:
    """Apply the command line overrides to the configuration."""
    if args.backend:
        config.detector.backend = args.backend
    if args.model:
        config.detector.model_path = args.model
    if args.output_format:
        config.detector.output_format = args.output_format
    if args.num_classes > 0:
        config.detector.num_classes = args.num_classes
    if args.source:
        config.source.type = args.source
    if args.camera_index >= 0:
        config.source.camera_index = args.camera_index
    if args.video:
        config.source.video_path = args.video
        config.source.type = "video"
    if args.target_track_id > 0:
        config.target.mode = "track_id"
        config.target.track_id = args.target_track_id
    if args.target_mode:
        config.target.mode = args.target_mode
    if args.target_name:
        config.target.name = args.target_name
        if config.target.mode in ("", "none"):
            config.target.mode = "identity"
    if args.identity is not None:
        config.identity.enabled = bool(args.identity)
    if config.target.mode == "identity":
        # The target is an enrolled person: the identity layer has to run.
        config.identity.enabled = True
    if args.identity_model:
        config.identity.embedder_path = args.identity_model
    if args.identity_threshold >= 0:
        config.identity.threshold = float(args.identity_threshold)
    if args.max_frames >= 0:
        config.runtime.max_frames = args.max_frames
    if args.log_level:
        config.logging.level = args.log_level.upper()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.inspect_model:
        from .inspect_model import inspect

        return inspect(args.inspect_model)

    try:
        config = load_config(args.config or None)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    apply_overrides(config, args)
    try:
        config.validate()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    setup_logging(config.logging.level, config.logging.file, config.logging.format)

    if args.check_config:
        print("The configuration is valid:")
        for section, values in config.to_dict().items():
            if section == "project_root":
                continue
            print(f"\n[{section}]")
            for key, value in values.items():
                print(f"  {key} = {value}")
        return 0

    try:
        pipeline = Pipeline(config, display=not args.no_display)
    except DetectorError as exc:
        print(f"Detector error: {exc}", file=sys.stderr)
        return 3
    except SourceError as exc:
        print(f"Source error: {exc}", file=sys.stderr)
        return 4
    except IdentityError as exc:
        print(f"Identity error: {exc}", file=sys.stderr)
        return 5
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        stats = pipeline.run()
    except KeyboardInterrupt:
        LOGGER.info("Interrupted")
        return 130
    finally:
        pipeline.release()

    LOGGER.info("Run finished: %s", stats.summary())
    for event in stats.target_events:
        LOGGER.info("target event: %s", event)
    print(stats.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
