"""Frame sources.

A source yields ``(frame, timestamp)`` pairs. Two implementations are
available: a camera and a video file. Both release their resources
deterministically, which matters when the process is interrupted.
"""

from __future__ import annotations

from .capture import CameraSource, FrameSource, SourceError, VideoSource, create_source

__all__ = [
    "CameraSource",
    "FrameSource",
    "SourceError",
    "VideoSource",
    "create_source",
]
