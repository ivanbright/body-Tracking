"""The tracker: identity stability and the life cycle of a track."""

from __future__ import annotations

import pytest

from src.tracking.track_state import TrackState
from tests.conftest import make_detection, moving_detection


def test_a_face_creates_exactly_one_track(tracker) -> None:
    result = tracker.update([make_detection()], 0.0, 0)
    assert len(result.new_tracks) == 1
    assert result.new_tracks[0].track_id == 1


def test_the_id_is_stable_over_many_frames(tracker) -> None:
    ids = []
    for frame in range(20):
        result = tracker.update([moving_detection(frame)], frame / 30.0, frame)
        ids.append(result.tracks[0].track_id)
    assert set(ids) == {1}


def test_a_second_face_gets_a_new_id(tracker) -> None:
    tracker.update([make_detection(x=100), make_detection(x=300)], 0.0, 0)
    result = tracker.update([make_detection(x=101), make_detection(x=301)], 1 / 30.0, 1)
    assert sorted(t.track_id for t in result.tracks) == [1, 2]


def test_a_track_that_is_not_seen_is_kept_as_lost(tracker) -> None:
    tracker.update([make_detection()], 0.0, 0)
    tracker.update([make_detection()], 1 / 30.0, 1)
    result = tracker.update([], 2 / 30.0, 2)

    assert len(result.lost_tracks) == 1
    assert result.tracks[0].state is TrackState.LOST
    # The ID is kept - this is what makes a later reacquisition possible.
    assert result.tracks[0].track_id == 1
    assert result.tracks[0].bbox == pytest.approx((100.0, 100.0, 140.0, 140.0))


def test_a_lost_track_expires_after_the_retention(tracker) -> None:
    tracker.update([make_detection()], 0.0, 0)
    result = tracker.update([], 60.0, 100)
    assert len(result.expired_tracks) == 1
    assert tracker.tracks == []


def test_a_returning_face_keeps_its_id(tracker) -> None:
    tracker.update([make_detection()], 0.0, 0)
    tracker.update([make_detection()], 1 / 30.0, 1)
    for frame in range(2, 10):
        tracker.update([], frame / 30.0, frame)

    result = tracker.update([make_detection()], 10 / 30.0, 10)

    assert len(result.reacquired_tracks) == 1
    assert result.reacquired_tracks[0].track_id == 1
    assert result.reacquired_tracks[0].state is TrackState.REACQUIRED
    assert result.reacquired_tracks[0].reacquisition_count == 1


def test_ids_are_never_reused(tracker) -> None:
    tracker.update([make_detection()], 0.0, 0)
    tracker.update([], 60.0, 100)  # the track expires
    result = tracker.update([make_detection()], 61.0, 101)
    assert result.new_tracks[0].track_id == 2


def test_a_low_score_detection_keeps_the_track_alive(tracker) -> None:
    """The second association pass is what makes a blurred face survivable."""
    tracker.update([make_detection(score=0.9)], 0.0, 0)
    result = tracker.update([make_detection(score=0.2)], 1 / 30.0, 1)
    assert len(result.new_tracks) == 0
    assert result.tracks[0].track_id == 1
