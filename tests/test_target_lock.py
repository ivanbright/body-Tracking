"""The target lock: the behaviour the whole project is built around."""

from __future__ import annotations

import pytest

from src.tracking.target_lock import TargetState
from src.tracking.track_state import TrackState
from tests.conftest import make_detection, moving_detection


def test_nothing_is_selected_by_default(harness) -> None:
    harness.step([make_detection()])
    assert harness.lock.state is TargetState.IDLE
    assert harness.lock.track_id is None


def test_selecting_a_track_by_click(harness) -> None:
    result = harness.step([make_detection()])
    track = result.tracks[0]
    harness.lock.select(track, frame=0, timestamp=0.0)

    assert harness.lock.track_id == track.track_id
    assert harness.lock.state is TargetState.LOCKED
    assert harness.lock.status(result.tracks).visible is True


def test_the_target_stays_locked_while_it_is_visible(harness) -> None:
    result = harness.step([make_detection()])
    harness.lock.select(result.tracks[0], frame=0, timestamp=0.0)
    for frame in range(1, 10):
        harness.step([moving_detection(frame)])
    assert harness.lock.state is TargetState.LOCKED
    assert harness.lock.track_id == 1


def test_the_target_keeps_its_id_while_it_is_lost(harness) -> None:
    for frame in range(5):
        harness.step([moving_detection(frame)])
    harness.lock.select(harness.tracker.track(1), frame=5, timestamp=5 / 30.0)

    for frame in range(5, 15):
        harness.step([])

    assert harness.lock.state is TargetState.LOST
    assert harness.lock.track_id == 1
    status = harness.lock.status(harness.tracker.tracks)
    assert status.visible is False
    assert status.time_since_seen > 0.0


def test_the_target_is_reacquired_with_the_same_id(harness) -> None:
    for frame in range(5):
        harness.step([moving_detection(frame)])
    harness.lock.select(harness.tracker.track(1), frame=5, timestamp=5 / 30.0)

    for frame in range(5, 12):
        harness.step([])

    harness.step([moving_detection(11)])

    assert harness.lock.track_id == 1
    assert harness.lock.state is TargetState.REACQUIRED
    kinds = [event.kind for event in harness.lock.events]
    assert "lost" in kinds
    assert "reacquired" in kinds


def test_the_lock_never_switches_to_another_face(harness) -> None:
    """A stranger appearing must not take over the target."""
    for frame in range(5):
        harness.step([moving_detection(frame)])
    harness.lock.select(harness.tracker.track(1), frame=5, timestamp=5 / 30.0)

    # The target disappears, a different face shows up in the same place.
    for frame in range(5, 12):
        harness.step([make_detection(x=400, y=400, score=0.95)])

    assert harness.lock.track_id == 1
    assert harness.lock.state is TargetState.LOST
    # The stranger has its own, new ID.
    assert 2 in [t.track_id for t in harness.tracker.tracks]


def test_the_target_expires_and_is_not_replaced_silently(harness) -> None:
    result = harness.step([make_detection()])
    harness.lock.select(result.tracks[0], frame=0, timestamp=0.0)

    # Stay missing for longer than target.lost_retention_seconds (10 s).
    for frame in range(1, 400):
        harness.step([], timestamp=frame / 30.0)

    assert harness.lock.state is TargetState.EXPIRED
    assert harness.lock.track_id is None
    kinds = [event.kind for event in harness.lock.events]
    assert "expired" in kinds


def test_auto_reselect_can_be_enabled(app_config, tracker) -> None:
    from src.tracking.target_lock import TargetLock

    app_config.target.auto_reselect = True
    app_config.target.mode = "largest"
    lock = TargetLock(
        app_config.target, tracker=tracker, prediction_config=app_config.prediction
    )

    for frame in range(5):
        result = tracker.update([make_detection()], frame / 30.0, frame)
        lock.update(result.tracks, frame / 30.0, frame)

    assert lock.state is TargetState.LOCKED
    assert lock.track_id == 1


def test_clearing_the_target(harness) -> None:
    harness.step([make_detection()])
    harness.lock.select(harness.tracker.track(1), frame=0, timestamp=0.0)
    harness.lock.clear(frame=1, timestamp=1 / 30.0)

    assert harness.lock.track_id is None
    assert harness.lock.state is TargetState.CLEARED


def test_the_locked_track_survives_longer_than_an_ordinary_track(harness, app_config) -> None:
    """The target is protected: it may stay lost longer than a normal track."""
    for frame in range(5):
        harness.step([moving_detection(frame)])
    harness.lock.select(harness.tracker.track(1), frame=5, timestamp=5 / 30.0)

    # 5 s of absence - longer than tracker.lost_retention_seconds (3 s).
    for frame in range(5, 155):
        harness.step([])

    assert harness.lock.track_id == 1
    assert harness.lock.state is TargetState.LOST
    assert harness.tracker.track(1) is not None


def test_the_prediction_is_reported_for_a_lost_target(harness) -> None:
    for frame in range(5):
        harness.step([moving_detection(frame, speed=3.0)])
    harness.lock.select(harness.tracker.track(1), frame=5, timestamp=5 / 30.0)

    for frame in range(5, 9):
        harness.step([])

    status = harness.lock.status(harness.tracker.tracks)
    assert status.has_prediction is True
    assert status.predicted_bbox != (0.0, 0.0, 0.0, 0.0)
