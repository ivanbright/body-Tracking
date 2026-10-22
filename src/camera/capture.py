"""Frame sources: a camera or a video file.

The source is responsible for exactly three things: delivering frames,
reporting the time of each frame and releasing its resources. It knows nothing
about detection, tracking or the target lock.
"""

from __future__ import annotations

import abc
import logging
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

#: Frame rates outside this range are treated as broken metadata. A container
#: that reports 1000 fps would otherwise make every time based decision
#: (retention, decay, expiry) happen a hundred times too fast.
MIN_PLAUSIBLE_FPS = 1.0
MAX_PLAUSIBLE_FPS = 240.0


def _plausible_fps(reported: float) -> float | None:
    """Return the reported frame rate when it can be trusted."""
    if reported and MIN_PLAUSIBLE_FPS <= reported <= MAX_PLAUSIBLE_FPS:
        return float(reported)
    return None


class SourceError(RuntimeError):
    """Raised when a frame source cannot be opened or read."""


class FrameSource(abc.ABC):
    """Common interface of the frame sources."""

    def __init__(self, fps: float = 30.0) -> None:
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self._closed = False
        self.frame_count = 0

    # ------------------------------------------------------------------ #
    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` of the frames."""
        raise NotImplementedError

    @abc.abstractmethod
    def read(self) -> tuple[np.ndarray | None, float]:
        """Read the next frame.

        Returns:
            ``(frame, timestamp)``. ``frame`` is ``None`` when the source is
            exhausted.
        """

    def release(self) -> None:
        """Release every resource. Safe to call more than once."""
        self._closed = True

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        while True:
            frame, timestamp = self.read()
            if frame is None:
                return
            yield frame, timestamp


class CameraSource(FrameSource):
    """A live camera."""

    def __init__(
        self,
        index: int = 0,
        *,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
    ) -> None:
        super().__init__(fps=fps)
        self.index = int(index)
        self._capture = cv2.VideoCapture(self.index)
        if not self._capture.isOpened():
            self._capture.release()
            raise SourceError(
                f"Cannot open camera {self.index}. "
                "Check that the device is connected and not in use by another program."
            )
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        reported = self._capture.get(cv2.CAP_PROP_FPS)
        usable = _plausible_fps(reported)
        if usable is not None:
            self.fps = usable
        elif reported:
            LOGGER.warning(
                "Camera %d reports an implausible frame rate of %.1f fps - "
                "using the configured %.1f fps instead",
                self.index,
                reported,
                self.fps,
            )
        LOGGER.info(
            "Opened camera %d (%dx%d @ %.1f fps)",
            self.index,
            self.size[0],
            self.size[1],
            self.fps,
        )

    @property
    def size(self) -> tuple[int, int]:
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width or 0, height or 0)

    def read(self) -> tuple[np.ndarray | None, float]:
        if self._closed:
            return None, time.time()
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None, time.time()
        self.frame_count += 1
        return frame, time.time()

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._capture.release()
        except Exception:  # pragma: no cover - defensive
            pass


class VideoSource(FrameSource):
    """A video file."""

    def __init__(self, path: str | Path, *, fps: float = 30.0, loop: bool = False) -> None:
        super().__init__(fps=fps)
        self.path = Path(path)
        self.loop = bool(loop)
        if not self.path.exists():
            raise SourceError(f"Video file not found: {self.path}")
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self._capture.release()
            raise SourceError(f"Cannot open video file: {self.path}")
        reported = self._capture.get(cv2.CAP_PROP_FPS)
        usable = _plausible_fps(reported)
        if usable is not None:
            self.fps = usable
        elif reported:
            LOGGER.warning(
                "Video %s reports an implausible frame rate of %.1f fps - "
                "using the configured %.1f fps instead",
                self.path.name,
                reported,
                self.fps,
            )
        LOGGER.info("Opened video %s (%.1f fps)", self.path.name, self.fps)

    @property
    def size(self) -> tuple[int, int]:
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width or 0, height or 0)

    def read(self) -> tuple[np.ndarray | None, float]:
        if self._closed:
            return None, time.time()
        ok, frame = self._capture.read()
        if not ok or frame is None:
            if not self.loop:
                return None, time.time()
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
            if not ok or frame is None:
                return None, time.time()
        self.frame_count += 1
        return frame, self.frame_count / self.fps

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._capture.release()
        except Exception:  # pragma: no cover - defensive
            pass


def create_source(config) -> FrameSource:
    """Build the frame source described by a :class:`~src.config.SourceConfig`."""
    if config.type == "camera":
        return CameraSource(
            config.camera_index,
            width=config.width,
            height=config.height,
            fps=config.fps,
        )
    if config.type == "video":
        return VideoSource(config.video_path, fps=config.fps, loop=config.loop)
    raise SourceError(f"unknown source type {config.type!r}")
