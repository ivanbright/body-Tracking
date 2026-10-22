"""The reacquisition rules, including the duration aware behaviour."""

from __future__ import annotations

import pytest

from src.tracking.reacquisition import ReacquisitionEngine
from src.tracking.track_state import Track
from tests.conftest import make_detection


def make_track(track_id: int = 1, bbox=(100.0, 100.0, 140.0, 140.0)) -> Track:
    track = Track(track_id=track_id)
    track.initialize(bbox, 0.9, 0, 0.0)
    track.update(bbox, 0.9, 1, 1 / 30.0)
    return track


def test_a_matching_detection_is_accepted(app_config) -> None:
    engine = ReacquisitionEngine(app_config.reacquisition)
    track = make_track()
    candidate = engine.evaluate(track, (102.0, 101.0, 142.0, 141.0), 0.9, timestamp=0.1)
    assert candidate.accepted is True
    assert candidate.is_long_gap is False
    assert candidate.reference == "predicted"


def test_a_far_away_detection_is_rejected(app_config) -> None:
    engine = ReacquisitionEngine(app_config.reacquisition)
    track = make_track()
    candidate = engine.evaluate(track, (400.0, 400.0, 440.0, 440.0), 0.9, timestamp=0.1)
    assert candidate.accepted is False
    assert "too far" in candidate.reason


def test_a_detection_with_the_wrong_size_is_rejected(app_config) -> None:
    engine = ReacquisitionEngine(app_config.reacquisition)
    track = make_track()
    candidate = engine.evaluate(track, (100.0, 100.0, 200.0, 200.0), 0.9, timestamp=0.1)
    assert candidate.accepted is False
    assert "size" in candidate.reason


def test_a_long_gap_uses_the_last_observed_box(app_config) -> None:
    engine = ReacquisitionEngine(app_config.reacquisition)
    track = make_track()
    candidate = engine.evaluate(track, (102.0, 101.0, 142.0, 141.0), 0.9, timestamp=30.0)
    assert candidate.is_long_gap is True
    assert candidate.reference == "last_observed"
    # The stricter rules demand more evidence.
    assert candidate.required_score == pytest.approx(
        app_config.reacquisition.score_threshold
        * app_config.reacquisition.long_gap_score_multiplier
    )


def test_a_long_gap_needs_a_minimum_confidence(app_config) -> None:
    engine = ReacquisitionEngine(app_config.reacquisition)
    track = make_track()
    candidate = engine.evaluate(track, (102.0, 101.0, 142.0, 141.0), 0.1, timestamp=30.0)
    assert candidate.accepted is False
    assert "confidence" in candidate.reason


def test_a_long_gap_tightens_the_distance_gate(app_config) -> None:
    engine = ReacquisitionEngine(app_config.reacquisition)
    track = make_track()
    # Far enough for the ordinary gate, too far for the long-gap gate.
    box = (100.0 + 120.0, 100.0, 140.0 + 120.0, 140.0)
    short_gap = engine.evaluate(track, box, 0.9, timestamp=0.5)
    long_gap = engine.evaluate(track, box, 0.9, timestamp=30.0)
    assert short_gap.is_long_gap is False
    assert long_gap.is_long_gap is True
    assert long_gap.distance > app_config.reacquisition.long_gap_max_distance_ratio


def test_an_ambiguous_match_is_rejected(app_config) -> None:
    """Two lost tracks, one detection: the evidence is ambiguous."""
    engine = ReacquisitionEngine(app_config.reacquisition)
    first = make_track(1, (100.0, 100.0, 140.0, 140.0))
    second = make_track(2, (105.0, 100.0, 145.0, 140.0))

    accepted = engine.select(
        [first, second], [((102.0, 100.0, 142.0, 140.0), 0.9)], timestamp=0.1
    )
    assert accepted == {}


def test_an_unambiguous_match_is_accepted(app_config) -> None:
    engine = ReacquisitionEngine(app_config.reacquisition)
    first = make_track(1, (100.0, 100.0, 140.0, 140.0))
    second = make_track(2, (400.0, 400.0, 440.0, 440.0))

    accepted = engine.select(
        [first, second], [((102.0, 100.0, 142.0, 140.0), 0.9)], timestamp=0.1
    )
    assert list(accepted) == [0]
    assert accepted[0].track_id == 1


def test_appearance_matching_is_disabled_by_default(app_config) -> None:
    engine = ReacquisitionEngine(app_config.reacquisition)
    assert getattr(engine.encoder, "enabled", False) is False


def test_appearance_alone_never_creates_a_match(app_config) -> None:
    """A perfect appearance score must not rescue a geometrically bad match."""
    from src.tracking.reacquisition import AppearanceEncoder

    class PerfectAppearance(AppearanceEncoder):
        enabled = True

        def similarity(self, track, detection):
            return 1.0

    app_config.reacquisition.appearance_enabled = True
    app_config.reacquisition.appearance_weight = 0.5
    engine = ReacquisitionEngine(app_config.reacquisition, appearance_encoder=PerfectAppearance())

    track = make_track()
    # Far away: the geometric gates reject it before the appearance is used.
    candidate = engine.evaluate(track, (500.0, 500.0, 540.0, 540.0), 0.9, timestamp=0.1)
    assert candidate.accepted is False
    assert "too far" in candidate.reason


def test_a_low_appearance_score_rejects_the_candidate(app_config) -> None:
    from src.tracking.reacquisition import AppearanceEncoder

    class BadAppearance(AppearanceEncoder):
        enabled = True

        def similarity(self, track, detection):
            return 0.0

    app_config.reacquisition.appearance_enabled = True
    app_config.reacquisition.appearance_weight = 0.5
    app_config.reacquisition.appearance_threshold = 0.4
    engine = ReacquisitionEngine(app_config.reacquisition, appearance_encoder=BadAppearance())

    track = make_track()
    candidate = engine.evaluate(track, (102.0, 100.0, 142.0, 140.0), 0.9, timestamp=0.1)
    assert candidate.accepted is False
    assert "appearance" in candidate.reason


def test_reacquisition_can_be_switched_off(app_config) -> None:
    app_config.reacquisition.enabled = False
    engine = ReacquisitionEngine(app_config.reacquisition)
    track = make_track()
    assert engine.select([track], [((102.0, 100.0, 142.0, 140.0), 0.9)], timestamp=0.1) == {}
