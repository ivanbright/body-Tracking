"""The multi-object tracker.

The tracker maintains **one independent track per detected face**. It knows
nothing about the selected target - that is the job of
:mod:`src.tracking.target_lock`. Keeping the two apart is what makes it
possible to say exactly what happens when the selected face disappears: the
tracker keeps every track alive on its own terms, and the lock decides which
one is the target.

The association follows the ByteTrack idea:

1. the detections are split into a *primary* set (high confidence) and a
   *secondary* set (low confidence),
2. the primary set is associated with the active tracks first,
3. the secondary set gets a second chance against the tracks that are still
   unmatched - this is what keeps the ID of a blurred or partially occluded
   face,
4. the tracks that are still unmatched are marked as **lost**. They are not
   deleted: their ID, their last reliable box, their motion state and their
   history are preserved so that the person can be found again,
5. the lost tracks are offered the remaining detections by the reacquisition
   engine, which applies the stricter, duration aware rules,
6. the high confidence detections that nobody claimed become **new** tracks
   with a brand new ID. A new face never inherits the ID of a lost track.

The ID counter is monotonic: an ID is never reused, so a new face can never be
mistaken for the target that was lost earlier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..detection.base import Detection
from .association import association_cost, linear_assignment
from .reacquisition import AppearanceEncoder, IdentityEvidence, ReacquisitionEngine
from .track_state import Track, TrackEvent, TrackState

LOGGER = logging.getLogger(__name__)


@dataclass
class TrackerResult:
    """What the tracker did during one frame."""

    frame: int = 0
    timestamp: float = 0.0
    tracks: list[Track] = field(default_factory=list)
    new_tracks: list[Track] = field(default_factory=list)
    reacquired_tracks: list[Track] = field(default_factory=list)
    #: Tracks that were restored because the identity confirmed them.
    identity_reacquired_tracks: list[Track] = field(default_factory=list)
    lost_tracks: list[Track] = field(default_factory=list)
    expired_tracks: list[Track] = field(default_factory=list)
    events: list[TrackEvent] = field(default_factory=list)

    @property
    def visible_tracks(self) -> list[Track]:
        """Tracks that are currently backed by a detection."""
        return [
            t
            for t in self.tracks
            if t.state in (TrackState.NEW, TrackState.TRACKED, TrackState.REACQUIRED)
        ]

    @property
    def lost_only(self) -> list[Track]:
        """Tracks that are currently retained but not observed."""
        return [t for t in self.tracks if t.state is TrackState.LOST]


class Tracker:
    """ByteTrack-style multi-object tracker with retained lost tracks."""

    def __init__(
        self,
        tracker_config,
        prediction_config,
        reacquisition_config,
        *,
        appearance_encoder: AppearanceEncoder | None = None,
        identity_config=None,
        fps: float = 30.0,
    ) -> None:
        self.config = tracker_config
        self.prediction_config = prediction_config
        self.reacquisition = ReacquisitionEngine(
            reacquisition_config,
            appearance_encoder=appearance_encoder,
            identity_config=identity_config,
        )
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self._protected: dict[int, float] = {}

    # ------------------------------------------------------------------ #
    @property
    def tracks(self) -> list[Track]:
        """All tracks that are still active (including the lost ones)."""
        return list(self._tracks.values())

    def track(self, track_id: int) -> Track | None:
        return self._tracks.get(int(track_id))

    def protect(self, track_id: int, retention_seconds: float) -> None:
        """Keep a track alive at least ``retention_seconds`` after it is lost.

        The target lock uses this so that the selected target survives longer
        than an ordinary lost track. A protected track is not limited by the
        frame based limit any more - only by the time based one.
        """
        self._protected[int(track_id)] = float(retention_seconds)

    def unprotect(self, track_id: int) -> None:
        self._protected.pop(int(track_id), None)

    # ------------------------------------------------------------------ #
    def update(
        self,
        detections: Sequence[Detection],
        timestamp: float,
        frame: int = 0,
        identity: IdentityEvidence | None = None,
    ) -> TrackerResult:
        """Process one frame of detections.

        Args:
            detections: The faces that were detected in this frame.
            timestamp: Timestamp of the frame.
            frame: Index of the frame.
            identity: Optional identity scores of the current frame. They are
                used by the reacquisition engine to recognise the target when
                it comes back into view.
        """
        result = TrackerResult(frame=frame, timestamp=timestamp)

        for track in self._tracks.values():
            track.set_now(timestamp)

        # --- 1. predict the current position of every track ------------ #
        self._predict_all()

        # --- 2. split the detections ----------------------------------- #
        high, low = self._split(detections)

        active = [
            t
            for t in self._tracks.values()
            if t.state in (TrackState.NEW, TrackState.TRACKED, TrackState.REACQUIRED)
        ]

        # --- 3. first association pass: high confidence detections ------ #
        matched_high, _ = self._associate(active, detections, high)

        # --- 4. second association pass: low confidence detections ------- #
        still_unmatched = [
            track for index, track in enumerate(active) if index not in matched_high
        ]
        matched_low, _ = self._associate(still_unmatched, detections, low)

        matched_track_ids = {active[index].track_id for index in matched_high}
        matched_track_ids |= {still_unmatched[index].track_id for index in matched_low}

        # --- 5. apply the matches --------------------------------------- #
        for index, detection_index in matched_high.items():
            self._apply(active[index], detections[detection_index], timestamp, frame, result)
        for index, detection_index in matched_low.items():
            self._apply(still_unmatched[index], detections[detection_index], timestamp, frame, result)

        # --- 6. the tracks nobody claimed become lost -------------------- #
        for track in active:
            if track.track_id in matched_track_ids:
                continue
            self._mark_lost(track, timestamp, frame, result)

        # --- 7. offer the free detections to the lost tracks ------------- #
        claimed = set(matched_high.values()) | set(matched_low.values())
        free = [index for index in range(len(detections)) if index not in claimed]
        lost_tracks = [t for t in self._tracks.values() if t.state is TrackState.LOST]
        consumed: set[int] = set()
        if lost_tracks and free:
            consumed = self._reacquire(
                lost_tracks, detections, free, timestamp, frame, result, identity
            )

        # --- 8. the remaining high confidence detections become new tracks #
        for index in high:
            if index in claimed or index in consumed:
                continue
            self._create_track(detections[index], timestamp, frame, result)

        # --- 9. expire the tracks that have been missing for too long ----- #
        self._expire(timestamp, frame, result)

        result.tracks = self.tracks
        return result

    # ------------------------------------------------------------------ #
    def _predict_all(self) -> None:
        """Advance the motion model of every track."""
        max_distance = float(self.prediction_config.max_prediction_distance)
        for track in self._tracks.values():
            if track.kalman is None:
                continue
            track.predicted_bbox = track.kalman.predict(max_distance=max_distance)

    def _split(self, detections: Sequence[Detection]) -> tuple[list[int], list[int]]:
        """Split the detection indices into the primary and the secondary set."""
        high: list[int] = []
        low: list[int] = []
        for index, detection in enumerate(detections):
            if detection.score >= self.config.track_high_thresh:
                high.append(index)
            elif detection.score >= self.config.track_low_thresh:
                low.append(index)
        return high, low

    def _associate(
        self,
        tracks: Sequence[Track],
        detections: Sequence[Detection],
        detection_indices: Sequence[int],
    ) -> tuple[dict[int, int], list[int]]:
        """Associate tracks with detections.

        Returns:
            ``(matched, unmatched_detections)`` where ``matched`` maps the
            index into ``tracks`` to the index into the detection list.
        """
        if not tracks or not detection_indices:
            return {}, list(detection_indices)

        track_boxes = np.asarray(
            [self._reference_box(track) for track in tracks], dtype=np.float32
        ).reshape(-1, 4)
        detection_boxes = np.asarray(
            [detections[index].bbox for index in detection_indices],
            dtype=np.float32,
        ).reshape(-1, 4)

        velocities = [track.velocity for track in tracks]
        cost = association_cost(
            track_boxes,
            detection_boxes,
            distance_weight=self.config.distance_weight,
            use_motion=self.config.use_motion,
            velocities=velocities,
        )
        matches, _, unmatched_columns = linear_assignment(cost, self.config.match_thresh)

        matched: dict[int, int] = {}
        for track_index, column in matches:
            matched[track_index] = detection_indices[column]
        unmatched = [detection_indices[column] for column in unmatched_columns]
        return matched, unmatched

    def _reference_box(self, track: Track) -> tuple[float, float, float, float]:
        """The box used for association.

        While the prediction is still fresh the predicted box is used, because
        it is a better estimate of where the face is now. Once the prediction
        is stale the last observed box is used instead - a prediction that has
        been running for seconds is a guess, not evidence.
        """
        if self._prediction_is_fresh(track):
            return track.predicted_bbox
        return track.bbox

    def _prediction_is_fresh(self, track: Track) -> bool:
        horizon = float(self.prediction_config.max_prediction_time)
        if horizon <= 0:
            return False
        if track.last_seen_time <= 0:
            return True
        return (Track._now - track.last_seen_time) <= horizon

    def _apply(
        self,
        track: Track,
        detection: Detection,
        timestamp: float,
        frame: int,
        result: TrackerResult,
    ) -> None:
        """Fold a detection into a track."""
        track.update(detection.bbox, detection.score, frame, timestamp)
        # A reacquired track returns to TRACKED on the next successful update.
        track.transition_to(TrackState.TRACKED, frame=frame, timestamp=timestamp)
        if not track.is_confirmed and track.hits >= self.config.min_hits_to_confirm:
            track.is_confirmed = True
            track.log_event("confirmed", frame, timestamp, f"hits={track.hits}")

    def _mark_lost(
        self,
        track: Track,
        timestamp: float,
        frame: int,
        result: TrackerResult,
    ) -> None:
        """Retain a track that was not matched this frame."""
        track.mark_missed()
        if track.state in (TrackState.TRACKED, TrackState.REACQUIRED):
            track.transition_to(TrackState.LOST, frame=frame, timestamp=timestamp)
            track.log_event("lost", frame, timestamp, f"missed={track.time_since_update}")
            result.lost_tracks.append(track)
        elif track.state is TrackState.NEW and not track.is_confirmed:
            # A tentative track that was never confirmed is dropped - by frames
            # and by wall-clock time - instead of being kept as a lost track.
            age_seconds = timestamp - track.created_time
            if (
                track.frames_since_creation > self.config.max_init_frames
                or age_seconds > self.config.lost_retention_seconds
            ):
                track.transition_to(TrackState.EXPIRED, frame=frame, timestamp=timestamp)
                track.log_event("expired", frame, timestamp, "never confirmed")
                result.expired_tracks.append(track)
                self._remove(track)

    def _reacquire(
        self,
        lost_tracks: Sequence[Track],
        detections: Sequence[Detection],
        free: Sequence[int],
        timestamp: float,
        frame: int,
        result: TrackerResult,
        identity: IdentityEvidence | None = None,
    ) -> set[int]:
        """Try to restore lost tracks with the free detections.

        Returns:
            The indices of the detections that were consumed.
        """
        boxes = [(detections[index].bbox, detections[index].score) for index in free]

        # The identity scores are aligned with the *whole* detection list, so
        # they have to be re-indexed to the free detections.
        evidence = None
        if identity is not None:
            scores = [
                identity.scores[index] if index < len(identity.scores) else None
                for index in free
            ]
            evidence = IdentityEvidence(track_id=identity.track_id, scores=scores)

        accepted = self.reacquisition.select(
            lost_tracks, boxes, timestamp=timestamp, identity=evidence
        )
        if not accepted:
            return set()

        consumed: set[int] = set()
        for local_index, candidate in accepted.items():
            detection_index = free[local_index]
            detection = detections[detection_index]
            track = self._tracks.get(candidate.track_id)
            if track is None:
                continue
            gap = max(0.0, timestamp - track.last_seen_time)
            track.update(detection.bbox, detection.score, frame, timestamp)
            track.transition_to(TrackState.REACQUIRED, frame=frame, timestamp=timestamp)
            track.reacquisition_count += 1
            track.last_reacquisition_gap = gap
            if candidate.identity_confirmed:
                track.identity_reacquisition_count += 1
                result.identity_reacquired_tracks.append(track)
            track.log_event(
                "reacquired",
                frame,
                timestamp,
                f"score={candidate.score:.3f}/{candidate.required_score:.3f} "
                f"gap={candidate.time_gap:.2f}s ref={candidate.reference}"
                + (
                    f" id={candidate.identity_similarity:.2f}"
                    if candidate.identity_confirmed
                    else ""
                ),
            )
            result.reacquired_tracks.append(track)
            consumed.add(detection_index)
        return consumed

    def _create_track(
        self,
        detection: Detection,
        timestamp: float,
        frame: int,
        result: TrackerResult,
    ) -> Track | None:
        """Create a brand new track with a fresh ID."""
        if len(self._tracks) >= self.config.max_active_tracks:
            LOGGER.debug(
                "Track limit reached (%d), dropping a detection",
                self.config.max_active_tracks,
            )
            return None

        track_id = self._next_id
        self._next_id += 1
        track = Track(track_id=track_id, class_name=detection.class_name)
        track.initialize(detection.bbox, detection.score, frame, timestamp)
        track.set_now(timestamp)
        track.state = TrackState.NEW
        self._tracks[track_id] = track
        track.log_event("created", frame, timestamp, f"score={detection.score:.2f}")
        result.new_tracks.append(track)
        return track

    def _expire(self, timestamp: float, frame: int, result: TrackerResult) -> None:
        """Expire the tracks that have been missing for too long."""
        lost_tracks = [t for t in self._tracks.values() if t.state is TrackState.LOST]
        if not lost_tracks:
            return

        for track in lost_tracks:
            age_frames = track.time_since_update
            age_seconds = timestamp - track.last_seen_time if track.last_seen_time else 0.0
            protected = self._protected.get(track.track_id)

            if protected is None:
                retention = float(self.config.lost_retention_seconds)
                expired = (
                    age_frames > self.config.max_lost_time
                    or age_seconds > retention
                )
            else:
                # A protected track (the selected target) is only limited by
                # its own, usually longer, retention.
                expired = age_seconds > protected

            if expired:
                track.transition_to(TrackState.EXPIRED, frame=frame, timestamp=timestamp)
                track.log_event(
                    "expired",
                    frame,
                    timestamp,
                    f"missing {age_seconds:.2f}s / {age_frames} frames",
                )
                result.expired_tracks.append(track)
                self._remove(track)

        # Cap the number of retained lost tracks.
        remaining = sorted(
            (t for t in self._tracks.values() if t.state is TrackState.LOST),
            key=lambda t: t.last_seen_time,
        )
        overflow = len(remaining) - self.config.max_lost_tracks
        for track in remaining[: max(0, overflow)]:
            track.transition_to(TrackState.EXPIRED, frame=frame, timestamp=timestamp)
            result.expired_tracks.append(track)
            self._remove(track)

    def _remove(self, track: Track) -> None:
        self._tracks.pop(track.track_id, None)
        self._protected.pop(track.track_id, None)

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Forget every track (the IDs are not reused)."""
        self._tracks.clear()
        self._protected.clear()
