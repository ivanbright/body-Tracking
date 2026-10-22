"""Detector interface shared by every backend.

The tracking layer only ever sees :class:`Detection` objects, which keeps the
tracker independent of ONNX Runtime, TFLite, OpenCV and of the actual weights
that happen to be installed.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


class DetectorError(RuntimeError):
    """Raised when a detector cannot be created or cannot produce valid output.

    The message is always actionable: a missing model file, an incompatible
    model, a missing runtime and a malformed output all produce a distinct,
    human readable reason.
    """


@dataclass(frozen=True)
class Detection:
    """A single detected object.

    Coordinate contract
    -------------------
    ``bbox`` is ``(x1, y1, x2, y2)`` in **pixels of the original input frame**
    (not of the network input). The origin is the top-left corner of the frame,
    ``x`` grows to the right and ``y`` grows downwards.
    """

    bbox: tuple[float, float, float, float]
    score: float
    class_id: int = 0
    class_name: str = "face"

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = (float(v) for v in self.bbox)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        object.__setattr__(self, "bbox", (x1, y1, x2, y2))
        object.__setattr__(self, "score", float(self.score))

    # ------------------------------------------------------------------ #
    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (
            0.5 * (self.bbox[0] + self.bbox[2]),
            0.5 * (self.bbox[1] + self.bbox[3]),
        )

    @property
    def diagonal(self) -> float:
        return float(np.hypot(self.width, self.height))

    def to_array(self) -> np.ndarray:
        """Return the box as a ``[x1, y1, x2, y2]`` float32 array."""
        return np.asarray(self.bbox, dtype=np.float32)

    def clipped(self, width: int, height: int) -> "Detection":
        """Return a copy clipped to the ``width`` x ``height`` frame."""
        x1 = min(max(self.bbox[0], 0.0), float(width))
        y1 = min(max(self.bbox[1], 0.0), float(height))
        x2 = min(max(self.bbox[2], 0.0), float(width))
        y2 = min(max(self.bbox[3], 0.0), float(height))
        return Detection(
            bbox=(x1, y1, x2, y2),
            score=self.score,
            class_id=self.class_id,
            class_name=self.class_name,
        )


class FaceDetector(abc.ABC):
    """Abstract detector.

    Sub-classes must implement :meth:`detect`. Everything else (warm-up,
    resource release) has a sensible default implementation.
    """

    #: Human readable backend name, shown in the overlay and in the log.
    name: str = "detector"

    @abc.abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Detect objects in ``frame_bgr`` (a BGR uint8 image).

        Returns an empty list when nothing is found. Must raise
        :class:`DetectorError` when the frame cannot be processed.
        """

    @property
    def input_size(self) -> tuple[int, int]:
        """Network input resolution as ``(width, height)``."""
        return (640, 640)

    def warmup(self, frame_shape: tuple[int, int] | None = None) -> None:
        """Optionally run a dummy inference so the first real frame is fast."""

    def close(self) -> None:
        """Release inference resources. Safe to call more than once."""

    # ------------------------------------------------------------------ #
    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<{type(self).__name__} name={self.name!r} input_size={self.input_size}>"


class ScriptedDetector(FaceDetector):
    """Detector that replays a scripted list of detections.

    Used by the test-suite and by the synthetic benchmark so that the tracking
    behaviour can be verified without model weights or a physical camera.
    """

    name = "scripted"

    def __init__(self, frames: Sequence[Sequence[Detection]] | None = None) -> None:
        self._frames: list[list[Detection]] = [list(frame) for frame in (frames or [])]
        self._index = 0
        self.calls = 0
        self.stats = DetectorStats()

    def push(self, detections: Iterable[Detection]) -> None:
        self._frames.append(list(detections))

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        import time

        start = time.perf_counter()
        self.calls += 1
        detections: list[Detection] = []
        if self._frames:
            index = min(self._index, len(self._frames) - 1)
            self._index += 1
            detections = list(self._frames[index])
        self.stats.record((time.perf_counter() - start) * 1000.0, len(detections))
        return detections


@dataclass
class DetectorStats:
    """Lightweight runtime counters for the detector."""

    frames: int = 0
    detections: int = 0
    last_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    history: list[float] = field(default_factory=list)

    def record(self, latency_ms: float, detection_count: int) -> None:
        self.frames += 1
        self.detections += detection_count
        self.last_latency_ms = latency_ms
        self.total_latency_ms += latency_ms
        self.history.append(latency_ms)

    @property
    def average_latency_ms(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)
