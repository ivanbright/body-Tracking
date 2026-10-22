"""Haar cascade face detector - explicit fallback backend.

This backend exists for one reason: the repository does **not** contain a
face-trained YOLOX model, and the tracking / reacquisition machinery must be
demonstrable on real camera frames today.

It is *not* a YOLOX detector and it is never presented as one: the overlay
and the log always show the backend name. Accuracy and robustness are clearly
below a trained YOLOX face model, which is why the YOLOX path remains the
primary one.

The cascade file ships with OpenCV, so no extra weights are required.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .base import Detection, DetectorError, DetectorStats

LOGGER = logging.getLogger(__name__)

DEFAULT_CASCADE = "haarcascade_frontalface_default.xml"


class HaarFaceDetector:
    """OpenCV Haar cascade detector for frontal faces."""

    name = "haar"

    def __init__(
        self,
        *,
        cascade_path: str | Path = "",
        min_size: tuple[int, int] = (40, 40),
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        score: float = 1.0,
        target_classes: Sequence[str] = (),
        warmup: bool = True,
    ) -> None:
        self.stats = DetectorStats()
        self.min_size = (int(min_size[0]), int(min_size[1]))
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.score = float(score)
        self.target_classes = tuple(target_classes)

        if cascade_path:
            path = Path(cascade_path).expanduser()
            if not path.exists():
                raise DetectorError(f"Haar cascade not found: {path}")
        else:
            path = Path(cv2.data.haarcascades) / DEFAULT_CASCADE
            if not path.exists():
                raise DetectorError(
                    f"OpenCV does not provide {DEFAULT_CASCADE} (looked in {cv2.data.haarcascades}). "
                    "Install opencv-python or pass an explicit cascade_path."
                )

        self.cascade_path = path
        self.classifier = cv2.CascadeClassifier(str(path))
        if self.classifier.empty():
            raise DetectorError(f"Failed to load Haar cascade: {path}")
        LOGGER.info("Loaded Haar cascade %s", path.name)

        if warmup:
            self.warmup()

    @property
    def input_size(self) -> tuple[int, int]:
        return (0, 0)  # the cascade works on the native frame resolution

    def warmup(self, frame_shape: tuple[int, int] | None = None) -> None:
        height, width = frame_shape if frame_shape else (480, 640)
        dummy = np.zeros((max(1, height), max(1, width), 3), dtype=np.uint8)
        try:
            self.detect(dummy)
        except DetectorError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Warm-up failed: %s", exc)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            raise DetectorError("detect() received an empty frame")

        start = time.perf_counter()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        if not len(faces):
            # The histogram equalisation helps in hard light, but it also
            # destroys the contrast structure of some faces - a face that is
            # clearly visible is then not detected at all. It is therefore only
            # used as a second attempt.
            faces = self.classifier.detectMultiScale(
                cv2.equalizeHist(gray),
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
            )

        detections: list[Detection] = []
        if len(faces):
            height, width = frame_bgr.shape[:2]
            for x, y, w, h in faces:
                detections.append(
                    Detection(
                        bbox=(float(x), float(y), float(x + w), float(y + h)),
                        score=self.score,
                        class_id=0,
                        class_name="face",
                    ).clipped(width, height)
                )
        detections.sort(key=lambda d: -d.area)

        latency_ms = (time.perf_counter() - start) * 1000.0
        self.stats.record(latency_ms, len(detections))
        return detections

    def close(self) -> None:
        classifier = getattr(self, "classifier", None)
        if classifier is not None:
            try:
                del self.classifier
            except Exception:  # pragma: no cover - defensive
                pass
            self.classifier = None  # type: ignore[assignment]

    def __enter__(self) -> "HaarFaceDetector":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
