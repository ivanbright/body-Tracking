"""Shared fixtures.

The unit tests never need a camera, a model or a network: the detections are
generated in code. That keeps the tests deterministic and fast, and it makes
clear which part of the system is actually being verified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, DetectorConfig, ReacquisitionConfig, TargetConfig, TrackerConfig  # noqa: E402
from src.detection.base import Detection  # noqa: E402
from src.tracking.target_lock import TargetLock  # noqa: E402
from src.tracking.tracker import Tracker  # noqa: E402

FPS = 30.0


def make_detection(
    x: float = 100.0,
    y: float = 100.0,
    width: float = 40.0,
    height: float = 40.0,
    score: float = 0.9,
    class_name: str = "face",
) -> Detection:
    """A single detection with the given top-left corner and size."""
    return Detection(
        bbox=(x, y, x + width, y + height),
        score=score,
        class_name=class_name,
    )


def moving_detection(
    frame: int,
    x0: float = 100.0,
    y: float = 100.0,
    size: float = 40.0,
    speed: float = 2.0,
    score: float = 0.9,
) -> Detection:
    """A detection that moves linearly with the frame index."""
    return make_detection(x0 + frame * speed, y, size, size, score)


@pytest.fixture
def app_config() -> AppConfig:
    """A configuration that is valid and quick to run."""
    config = AppConfig()
    config.detector = DetectorConfig(backend="haar")
    config.tracker = TrackerConfig()
    config.target = TargetConfig()
    config.reacquisition = ReacquisitionConfig()
    config.target.require_confirmed = False
    return config


@pytest.fixture
def tracker(app_config: AppConfig) -> Tracker:
    return Tracker(
        app_config.tracker,
        app_config.prediction,
        app_config.reacquisition,
        fps=FPS,
    )


@pytest.fixture
def lock(app_config: AppConfig, tracker: Tracker) -> TargetLock:
    return TargetLock(
        app_config.target, tracker=tracker, prediction_config=app_config.prediction
    )


class Harness:
    """Feeds scripted detections into a tracker and a target lock.

    It records the state of the target after every frame, which is what the
    scenarios assert on.
    """

    def __init__(self, tracker: Tracker, lock: TargetLock, fps: float = FPS) -> None:
        self.tracker = tracker
        self.lock = lock
        self.fps = fps
        self.frame = 0
        self.states: list = []
        self.track_ids: list[int | None] = []

    def step(self, detections, timestamp: float | None = None) -> object:
        """Process one frame and return the tracker result."""
        if timestamp is None:
            timestamp = self.frame / self.fps
        result = self.tracker.update(detections, timestamp, self.frame)
        self.lock.update(result.tracks, timestamp, self.frame)
        self.states.append(self.lock.state)
        self.track_ids.append(self.lock.track_id)
        self.frame += 1
        return result

    def run(self, frames) -> list:
        """Process a sequence of detection lists."""
        results = []
        for detections in frames:
            results.append(self.step(detections))
        return results


@pytest.fixture
def harness(tracker: Tracker, lock: TargetLock) -> Harness:
    return Harness(tracker, lock)
