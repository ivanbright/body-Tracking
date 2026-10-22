"""The identity layer: recognising the target when it comes back.

The tests never load a real model: the embeddings are vectors that are built in
the code, so the rules can be verified exactly.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.config import ConfigError, IdentityConfig, PredictionConfig, ReacquisitionConfig, TrackerConfig, load_config
from src.pipeline import Pipeline
from src.recognition.alignment import (
    ARCFACE_TEMPLATE_5PT,
    align_face_5pt,
    crop_face,
    estimate_norm_5pt,
)
from src.recognition.arcface import ArcFaceEncoder, IdentityError, cosine_similarity
from src.recognition.identity import IdentityMatcher
from src.tracking.reacquisition import IdentityEvidence, ReacquisitionEngine
from src.tracking.track_state import Track, TrackState
from src.tracking.tracker import Tracker
from tests.conftest import make_detection


def make_track(track_id: int = 1, bbox=(100.0, 100.0, 140.0, 140.0)) -> Track:
    track = Track(track_id=track_id)
    track.initialize(bbox, 0.9, 0, 0.0)
    track.update(bbox, 0.9, 1, 1 / 30.0)
    return track


def identity_config(**overrides) -> IdentityConfig:
    config = IdentityConfig(enabled=True)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def unit_vector(seed: int) -> np.ndarray:
    """A deterministic unit vector."""
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(512).astype(np.float32)
    return vector / np.linalg.norm(vector)


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #
def test_the_template_maps_onto_itself() -> None:
    matrix = estimate_norm_5pt(ARCFACE_TEMPLATE_5PT)
    mapped = cv2.transform(
        ARCFACE_TEMPLATE_5PT.reshape(1, -1, 2).astype(np.float32), matrix
    )[0]
    assert np.allclose(mapped, ARCFACE_TEMPLATE_5PT, atol=1e-3)


def test_aligning_produces_a_112x112_crop() -> None:
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    aligned, matrix = align_face_5pt(frame, ARCFACE_TEMPLATE_5PT)
    assert aligned.shape == (112, 112, 3)
    assert matrix.shape == (2, 3)


def test_cropping_keeps_the_content_of_the_box() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[50:100, 60:120] = 255
    crop = crop_face(frame, (60.0, 50.0, 120.0, 100.0), out_size=(112, 112))
    assert crop is not None
    assert crop.shape == (112, 112, 3)
    assert crop.mean() > 200


def test_cropping_outside_the_frame_is_clipped() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = crop_face(frame, (-50.0, -50.0, 50.0, 50.0), out_size=(112, 112))
    assert crop is not None
    assert crop.shape == (112, 112, 3)


# --------------------------------------------------------------------------- #
# the encoder
# --------------------------------------------------------------------------- #
def test_a_missing_embedder_is_reported_clearly(tmp_path) -> None:
    with pytest.raises(IdentityError):
        ArcFaceEncoder(tmp_path / "does_not_exist.onnx")


def test_cosine_similarity_of_two_vectors() -> None:
    a = unit_vector(1)
    assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-5)
    assert cosine_similarity(a, unit_vector(2)) < 0.5


def test_the_matcher_is_inactive_when_it_is_disabled() -> None:
    matcher = IdentityMatcher(IdentityConfig(enabled=False))
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    assert matcher.embed(frame, (10.0, 10.0, 90.0, 90.0)) is None


def test_the_matcher_ignores_tiny_faces() -> None:
    matcher = IdentityMatcher(IdentityConfig(enabled=True, min_face_size=40))
    matcher._encoder = object()  # type: ignore[assignment]
    matcher._aligner = object()  # type: ignore[assignment]
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    assert matcher.embed(frame, (10.0, 10.0, 20.0, 20.0)) is None


# --------------------------------------------------------------------------- #
# the reference embedding of a track
# --------------------------------------------------------------------------- #
def test_the_first_observation_creates_the_reference() -> None:
    track = make_track()
    assert track.embedding is None
    accepted, similarity = track.add_embedding(unit_vector(1))
    assert accepted is True
    assert similarity == pytest.approx(1.0)
    assert track.embedding is not None
    assert track.embedding_count == 1


def test_similar_observations_are_stored_as_poses() -> None:
    track = make_track()
    track.add_embedding(unit_vector(1))
    accepted, similarity = track.add_embedding(unit_vector(1))
    assert accepted is True
    assert similarity == pytest.approx(1.0, abs=1e-5)
    assert track.embedding_count == 2


def test_a_dissimilar_observation_does_not_drift_the_reference() -> None:
    track = make_track()
    track.add_embedding(unit_vector(1))
    reference = track.embedding.copy()
    accepted, similarity = track.add_embedding(unit_vector(2), min_similarity=0.4)
    assert accepted is False
    assert similarity < 0.4
    assert track.embedding_count == 1
    assert np.allclose(track.embedding, reference)


def test_the_reference_stays_normalized() -> None:
    track = make_track()
    track.add_embedding(unit_vector(1))
    track.add_embedding(unit_vector(3))
    assert float(np.linalg.norm(track.embedding)) == pytest.approx(1.0, abs=1e-5)


def test_the_similarity_is_the_maximum_over_all_poses() -> None:
    track = make_track()
    track.add_embedding(unit_vector(1))
    track.add_embedding(unit_vector(2))
    # unit_vector(2) is completely different from unit_vector(1)
    assert track.similarity_to(unit_vector(2)) == pytest.approx(1.0, abs=1e-5)
    assert track.similarity_to(unit_vector(3)) < 0.3


def test_the_pose_set_does_not_grow_without_a_limit() -> None:
    track = make_track()
    for index in range(10):
        track.add_embedding(unit_vector(index % 5 + 1), max_embeddings=4)
    assert track.embedding_count == 4


# --------------------------------------------------------------------------- #
# the reacquisition rules
# --------------------------------------------------------------------------- #
def test_a_different_person_is_rejected_even_when_the_geometry_is_perfect() -> None:
    engine = ReacquisitionEngine(
        ReacquisitionConfig(),
        identity_config=identity_config(),
    )
    track = make_track()
    candidate = engine.evaluate(
        track,
        (102.0, 101.0, 142.0, 141.0),
        0.9,
        timestamp=0.1,
        identity_similarity=0.10,
    )
    assert candidate.accepted is False
    assert "identity mismatch" in candidate.reason
    assert candidate.identity_confirmed is False


def test_the_identity_restores_a_person_that_is_far_away() -> None:
    engine = ReacquisitionEngine(
        ReacquisitionConfig(), identity_config=identity_config()
    )
    track = make_track()
    # Far away: the geometric gates would reject this.
    candidate = engine.evaluate(
        track,
        (400.0, 400.0, 440.0, 440.0),
        0.9,
        timestamp=0.1,
        identity_similarity=0.92,
    )
    assert candidate.identity_confirmed is True
    assert candidate.identity_similarity == pytest.approx(0.92)
    assert candidate.accepted is True


def test_the_same_detection_is_rejected_without_the_identity() -> None:
    engine = ReacquisitionEngine(
        ReacquisitionConfig(), identity_config=identity_config()
    )
    track = make_track()
    candidate = engine.evaluate(
        track,
        (400.0, 400.0, 440.0, 440.0),
        0.9,
        timestamp=0.1,
    )
    assert candidate.accepted is False
    assert "too far" in candidate.reason


def test_an_ambiguous_identity_is_rejected() -> None:
    engine = ReacquisitionEngine(
        ReacquisitionConfig(), identity_config=identity_config()
    )
    track = make_track()
    detections = [
        ((300.0, 300.0, 340.0, 340.0), 0.9),
        ((320.0, 320.0, 360.0, 360.0), 0.9),
    ]
    evidence = IdentityEvidence(track_id=track.track_id, scores=[0.90, 0.89])
    accepted = engine.select([track], detections, timestamp=0.1, identity=evidence)
    assert accepted == {}


def test_a_clear_identity_wins_over_the_runner_up() -> None:
    engine = ReacquisitionEngine(
        ReacquisitionConfig(), identity_config=identity_config()
    )
    track = make_track()
    detections = [
        ((300.0, 300.0, 340.0, 340.0), 0.9),
        ((320.0, 320.0, 360.0, 360.0), 0.9),
    ]
    evidence = IdentityEvidence(track_id=track.track_id, scores=[0.95, 0.40])
    accepted = engine.select([track], detections, timestamp=0.1, identity=evidence)
    assert list(accepted) == [0]


def test_the_identity_is_only_used_for_its_own_track() -> None:
    engine = ReacquisitionEngine(
        ReacquisitionConfig(), identity_config=identity_config()
    )
    target = make_track(track_id=1)
    other = make_track(track_id=2, bbox=(400.0, 400.0, 440.0, 440.0))
    evidence = IdentityEvidence(track_id=1, scores=[0.95])
    candidate = engine.evaluate(
        other,
        (402.0, 401.0, 442.0, 441.0),
        0.9,
        timestamp=0.1,
        identity_similarity=evidence.score_for(other.track_id, 0),
    )
    assert candidate.identity_similarity is None
    assert candidate.identity_confirmed is False


# --------------------------------------------------------------------------- #
# the tracker
# --------------------------------------------------------------------------- #
def test_the_tracker_restores_the_original_id_with_the_identity() -> None:
    tracker = Tracker(
        TrackerConfig(),
        PredictionConfig(),
        ReacquisitionConfig(),
        identity_config=identity_config(),
        fps=30.0,
    )

    # The target is seen, then it disappears.
    tracker.update([make_detection(100.0, 100.0)], 0.0, 0)
    tracker.update([make_detection(102.0, 100.0)], 1 / 30.0, 1)
    tracker.update([], 2 / 30.0, 2)
    lost = tracker.tracks[0]
    assert lost.track_id == 1
    assert lost.is_lost

    # It comes back somewhere else entirely - the geometry alone would reject.
    evidence = IdentityEvidence(track_id=1, scores=[0.93])
    result = tracker.update([make_detection(380.0, 380.0)], 3 / 30.0, 3, identity=evidence)

    assert len(result.reacquired_tracks) == 1
    restored = result.reacquired_tracks[0]
    assert restored.track_id == 1
    assert restored.identity_reacquisition_count == 1
    assert len(result.identity_reacquired_tracks) == 1


def test_the_tracker_rejects_a_different_person() -> None:
    tracker = Tracker(
        TrackerConfig(),
        PredictionConfig(),
        ReacquisitionConfig(),
        identity_config=identity_config(),
        fps=30.0,
    )

    tracker.update([make_detection(100.0, 100.0)], 0.0, 0)
    tracker.update([make_detection(102.0, 100.0)], 1 / 30.0, 1)
    tracker.update([], 2 / 30.0, 2)

    evidence = IdentityEvidence(track_id=1, scores=[0.12])
    result = tracker.update([make_detection(110.0, 100.0)], 3 / 30.0, 3, identity=evidence)

    assert result.reacquired_tracks == []
    assert len(result.new_tracks) == 1
    assert result.new_tracks[0].track_id == 2


# --------------------------------------------------------------------------- #
# the configuration
# --------------------------------------------------------------------------- #
def test_an_invalid_threshold_is_rejected() -> None:
    config = IdentityConfig(enabled=True, threshold=1.5)
    with pytest.raises(ConfigError):
        config.validate()


def test_an_invalid_alignment_is_rejected() -> None:
    config = IdentityConfig(enabled=True, alignment="magic")
    with pytest.raises(ConfigError):
        config.validate()


def test_the_identity_section_is_part_of_the_configuration(app_config) -> None:
    assert isinstance(app_config.identity, IdentityConfig)
    assert app_config.identity.enabled is False


def test_the_identity_target_mode_needs_a_name(app_config) -> None:
    app_config.identity.enabled = True
    app_config.target.mode = "identity"
    app_config.target.name = ""
    with pytest.raises(ConfigError):
        app_config.validate()

    app_config.target.name = "Ivan"
    app_config.validate()


def test_the_identity_target_mode_needs_the_identity_layer(app_config) -> None:
    app_config.identity.enabled = False
    app_config.target.mode = "identity"
    app_config.target.name = "Ivan"
    with pytest.raises(ConfigError):
        app_config.validate()


# --------------------------------------------------------------------------- #
# the target that is selected by its name
# --------------------------------------------------------------------------- #
def make_identity_pipeline() -> Pipeline:
    """A pipeline whose tracks carry their name.

    The identity layer itself is not needed for the selection - the name is
    already on the track - so the models are not loaded here.
    """
    config = load_config(None)
    config.detector.backend = "haar"
    config.target.mode = "identity"
    config.target.name = "Ivan"
    pipeline = Pipeline(config, display=False)
    return pipeline


def test_the_target_is_selected_by_the_enrolled_name() -> None:
    pipeline = make_identity_pipeline()
    try:
        target = make_track(track_id=7)
        target.state = TrackState.TRACKED
        target.identity_name = "Ivan"
        pipeline.tracker._tracks[7] = target

        pipeline._select_by_identity(1.0)

        assert pipeline.lock.track_id == 7
    finally:
        pipeline.release()


def test_a_target_with_another_name_is_not_selected() -> None:
    pipeline = make_identity_pipeline()
    try:
        target = make_track(track_id=7)
        target.state = TrackState.TRACKED
        target.identity_name = "Anna"
        pipeline.tracker._tracks[7] = target

        pipeline._select_by_identity(1.0)

        assert pipeline.lock.track_id is None
    finally:
        pipeline.release()


def test_a_cleared_target_is_not_selected_again() -> None:
    pipeline = make_identity_pipeline()
    try:
        target = make_track(track_id=7)
        target.state = TrackState.TRACKED
        target.identity_name = "Ivan"
        pipeline.tracker._tracks[7] = target

        pipeline.lock.clear(frame=0, timestamp=0.0)
        pipeline._select_by_identity(1.0)

        assert pipeline.lock.track_id is None
    finally:
        pipeline.release()
