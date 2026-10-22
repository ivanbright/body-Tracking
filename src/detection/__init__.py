"""Detection backends.

Every backend implements :class:`~src.detection.base.FaceDetector` and returns
:class:`~src.detection.base.Detection` objects whose boxes are expressed in
original-frame pixels.

The backend only answers *where are the faces in this frame*. It never decides
which face is the target - that is the job of the target lock.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .base import Detection, DetectorError, DetectorStats, FaceDetector, ScriptedDetector
from .haar import HaarFaceDetector
from .postprocessing import (
    DecodeError,
    DecodeSpec,
    apply_class_filter,
    decode_yolox_output,
    non_maximum_suppression,
)
from .preprocessing import TransformInfo, infer_layout, letterbox, resize_frame, stretch, to_blob

LOGGER = logging.getLogger(__name__)

__all__ = [
    "DecodeError",
    "DecodeSpec",
    "Detection",
    "DetectorError",
    "DetectorStats",
    "FaceDetector",
    "HaarFaceDetector",
    "ScriptedDetector",
    "TransformInfo",
    "apply_class_filter",
    "create_detector",
    "decode_yolox_output",
    "infer_layout",
    "letterbox",
    "non_maximum_suppression",
    "resize_frame",
    "stretch",
    "to_blob",
]


def load_labels(labels_path: str | Path | None) -> tuple[str, ...]:
    """Read a plain text label file (one class name per line)."""
    if not labels_path:
        return ()
    path = Path(labels_path)
    if not path.exists():
        LOGGER.warning("Label file not found: %s", path)
        return ()
    try:
        names = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except OSError as exc:  # pragma: no cover - defensive
        LOGGER.warning("Could not read label file %s: %s", path, exc)
        return ()
    return names


def create_detector(config) -> FaceDetector:
    """Build the detector described by a :class:`~src.config.DetectorConfig`.

    Raises:
        DetectorError: The backend is unknown, or the model file is missing.
    """
    backend = config.backend
    if backend == "yolox_onnx":
        from .yolox_onnx import YOLOXOnnxDetector

        return YOLOXOnnxDetector(
            config.model_path,
            input_size=config.input_size,
            score_threshold=config.score_thresh,
            nms_threshold=config.nms_thresh,
            providers=config.providers,
            input_layout=config.input_layout,
            resize_mode=config.resize_mode,
            input_range=config.input_range,
            color_order=config.color_order,
            output_format=config.output_format,
            activation=config.activation,
            num_classes=config.num_classes,
            strides=config.strides,
            reg_max=config.reg_max,
            labels=load_labels(config.labels_path),
            target_classes=config.target_classes,
            num_threads=config.num_threads,
            warmup=config.warmup,
        )
    if backend == "haar":
        return HaarFaceDetector(
            min_size=config.haar_min_size,
            target_classes=config.target_classes,
            warmup=config.warmup,
        )
    raise DetectorError(
        f"unknown detector backend {backend!r}; expected one of yolox_onnx, haar"
    )
