"""Reacquisition: finding a person who was temporarily not detected.

When a tracked face disappears, the track is **kept**. The system keeps
predicting where the person should be and, on every new detection, asks the
question:

    "Is this detection the person I lost, or somebody else?"

Answering that question with a distance threshold alone is how trackers end up
attaching a stranger's face to the target. This module therefore scores the
*evidence* instead of applying a single threshold:

===================  =========================================================
Evidence              Why it matters
===================  =========================================================
Predicted distance    The detection should be close to where the Kalman filter
                      expects the person to be.
Box overlap          A real re-appearance looks like the box we saw before.
Velocity consistency The person should keep moving in the same direction.
Size consistency      A face does not change size drastically in a few frames.
Detection confidence  A doubtful detection is weak evidence.
Time since seen      The longer the person has been missing, the stronger the
                      spatial evidence has to be.
Appearance           Optional, and never allowed to *create* a match on its own.
===================  =========================================================

A match is accepted only when the weighted score passes the threshold **and**
the hard geometric gates pass **and** the evidence is not ambiguous. When in
doubt the track simply stays LOST - keeping a lost track is cheap, attaching
the wrong person is not.

Identity
--------

The geometric evidence has one blind spot: when the person **turns away** and
comes back, the motion prediction is stale and the detection may be far away
from where the target was last seen. The geometry then says "not the same
person" even though it obviously is.

The identity layer closes that gap. While the target is visible, its face is
described by an embedding (ArcFace). While the target is missing, every
candidate face is compared against that reference:

* a similarity **below** the threshold means "somebody else" - the candidate
  is rejected, no matter how good the geometry looks,
* a similarity **at or above** the threshold **confirms** the identity: the
  geometric distance and size gates are relaxed, because the face - not the
  motion - is the evidence,
* two confirmed candidates that are almost equally similar are **ambiguous**
  (two people that look alike) and both are rejected. The system must never
  guess.

The identity is only used for the track that owns the reference embedding, and
only as a *confirmation*: the identity can restore a track, but it never
creates one.

Duration awareness
------------------
A two second occlusion and a two minute absence are not the same problem. The
rules therefore depend on how long the target has been missing:

* **short gap** - the motion prediction is still meaningful. The ordinary
  gates apply, the prediction is used as the spatial reference and the
  evidence decays with the length of the gap (``exp(-dt / time_decay_tau)``).
* **long gap** (``> reacquisition.long_gap_seconds``) - the prediction has
  drifted, so the spatial evidence is measured against the **last observed**
  box, the distance gate is tightened, the required evidence score is raised
  and a minimum detection confidence is enforced. The decay is not applied
  here: the reference no longer moves, so there is nothing to discount, and
  the stricter rules above carry the whole burden.

A face that is fully occluded or out of view simply cannot be observed. The
engine can only preserve the track and try again when the person returns; it
cannot promise that the person is still at the predicted position.

The module knows nothing about cameras or detectors: it works on
:class:`~src.tracking.track_state.Track` objects and plain boxes.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .track_state import Track, TrackState

LOGGER = logging.getLogger(__name__)


class AppearanceEncoder(Protocol):
    """Optional appearance representation.

    An implementation must return a *similarity* in ``[0, 1]`` between two
    observations. The default configuration disables appearance matching
    entirely: an unvalidated similarity score must never be the reason a
    track is restored.
    """

    def similarity(self, track: Track, detection: Sequence[float]) -> float:  # pragma: no cover
        ...


class NullAppearanceEncoder:
    """Appearance encoder that is always disabled."""

    enabled = False

    def similarity(self, track: Track, detection: Sequence[float]) -> float:
        return 0.0


@dataclass
class IdentityEvidence:
    """Identity scores of the current frame.

    The scores belong to **one** track - the one that owns the reference
    embedding, i.e. the target. They are aligned with the detection list that
    is passed to :meth:`ReacquisitionEngine.select`.
    """

    #: The track the scores were computed for.
    track_id: int
    #: Cosine similarity per detection (``None`` when it could not be computed).
    scores: Sequence[float | None] = ()

    def score_for(self, track_id: int, detection_index: int) -> float | None:
        """The identity score of one (track, detection) pair."""
        if track_id != self.track_id:
            return None
        if detection_index < 0 or detection_index >= len(self.scores):
            return None
        return self.scores[detection_index]


@dataclass
class ReacquisitionCandidate:
    """The outcome of evaluating one (lost track, detection) pair."""

    track_id: int
    detection_index: int
    score: float = 0.0
    distance: float = 0.0
    overlap: float = 0.0
    velocity_consistency: float = 0.0
    size_ratio: float = 1.0
    confidence: float = 0.0
    time_gap: float = 0.0
    accepted: bool = False
    reason: str = ""
    #: ``True`` when the target has been missing for longer than
    #: ``reacquisition.long_gap_seconds`` and the stricter rules applied.
    is_long_gap: bool = False
    #: Evidence score that had to be reached for this candidate.
    required_score: float = 0.0
    #: Which box the spatial evidence was measured against.
    reference: str = "predicted"
    #: Factor that was applied to the positional evidence.
    decay: float = 1.0
    #: Cosine similarity of the faces (``None`` when it was not computed).
    identity_similarity: float | None = None
    #: ``True`` when the identity confirmed that this is the same person.
    identity_confirmed: bool = False

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"track {self.track_id} <- detection {self.detection_index} "
            f"score={self.score:.3f}/{self.required_score:.3f} ({self.reason})"
        )


class ReacquisitionEngine:
    """Scores lost tracks against new detections and decides what to restore."""

    def __init__(
        self,
        config,
        *,
        appearance_encoder: AppearanceEncoder | None = None,
        identity_config=None,
    ) -> None:
        self.config = config
        self.encoder = appearance_encoder or NullAppearanceEncoder()
        self.identity_config = identity_config

    @property
    def identity_enabled(self) -> bool:
        """Whether the identity layer may be used."""
        return bool(
            self.identity_config is not None
            and getattr(self.identity_config, "enabled", False)
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _center(box: Sequence[float]) -> np.ndarray:
        x1, y1, x2, y2 = (float(v) for v in box)
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)

    @staticmethod
    def _iou(a: Sequence[float], b: Sequence[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        track: Track,
        detection_box: Sequence[float],
        detection_score: float,
        detection_index: int = 0,
        timestamp: float = 0.0,
        identity_similarity: float | None = None,
    ) -> ReacquisitionCandidate:
        """Score a single (lost track, detection) pair.

        The rules are **duration aware**. While the gap is short the motion
        prediction is still meaningful and the ordinary gates apply. Once the
        target has been missing for longer than ``long_gap_seconds`` the
        prediction has drifted and the spatial evidence is measured against the
        **last observed** box instead, the gate is tightened and the required
        evidence score is raised.

        When the identity layer provides a similarity for this track, the
        identity decides first: a low similarity rejects the candidate
        ("somebody else"), a high similarity confirms it and relaxes the
        geometric gates.
        """
        cfg = self.config
        candidate = ReacquisitionCandidate(
            track_id=track.track_id,
            detection_index=detection_index,
            confidence=float(detection_score),
        )

        # --- how long has the target been missing? ---------------------- #
        time_gap = max(0.0, timestamp - track.last_seen_time) if track.last_seen_time else 0.0
        candidate.time_gap = time_gap
        is_long_gap = time_gap > cfg.long_gap_seconds
        candidate.is_long_gap = is_long_gap

        required_score = cfg.score_threshold * (
            cfg.long_gap_score_multiplier if is_long_gap else 1.0
        )
        required_score = min(1.0, required_score)
        candidate.required_score = required_score

        # --- identity ---------------------------------------------------- #
        # The identity is checked *before* the geometry: it can reject a
        # candidate ("different person") and it can relax the geometric gates
        # ("same person, the prediction was just wrong").
        identity_confirmed = False
        if self.identity_enabled and identity_similarity is not None:
            candidate.identity_similarity = float(identity_similarity)
            threshold = float(self.identity_config.threshold)
            if identity_similarity < threshold:
                candidate.reason = (
                    f"identity mismatch ({identity_similarity:.2f} < {threshold:.2f})"
                )
                return candidate
            identity_confirmed = True
            candidate.identity_confirmed = True
            # The face confirmed the identity, so the evidence requirement is
            # met - but the ambiguity guard in :meth:`select` still applies.
            required_score = min(required_score, float(self.identity_config.confirmed_score))
            candidate.required_score = required_score

        if identity_confirmed:
            distance_gate = float(self.identity_config.max_distance_ratio)
        elif is_long_gap:
            distance_gate = cfg.long_gap_max_distance_ratio
        else:
            distance_gate = cfg.max_distance_ratio

        # The prediction is unreliable after a long gap: measure the spatial
        # evidence against the last reliable observation instead. The same is
        # true when the identity confirms the candidate - the face is the
        # evidence then, and the position of a face that was just walking
        # around is not what the prediction says.
        use_last_observed = is_long_gap or identity_confirmed
        reference_box = track.bbox if use_last_observed else track.predicted_bbox
        candidate.reference = "last_observed" if use_last_observed else "predicted"

        reference_center = self._center(reference_box)
        detection_center = self._center(detection_box)
        scale = max(track.diagonal, 1.0)

        distance = float(np.linalg.norm(detection_center - reference_center))
        normalized_distance = distance / scale
        candidate.distance = normalized_distance

        # --- hard gate 1: geometric distance --------------------------- #
        if normalized_distance > distance_gate:
            candidate.reason = (
                f"too far ({normalized_distance:.2f} > {distance_gate:.2f}, "
                f"{candidate.reference})"
            )
            return candidate

        # --- hard gate 2: size consistency ----------------------------- #
        det_height = max(1e-3, float(detection_box[3]) - float(detection_box[1]))
        det_width = max(1e-3, float(detection_box[2]) - float(detection_box[0]))
        track_height = max(1e-3, track.height)
        track_width = max(1e-3, track.width)
        height_ratio = det_height / track_height
        width_ratio = det_width / track_width
        size_ratio = min(height_ratio, width_ratio)
        candidate.size_ratio = size_ratio
        if identity_confirmed:
            min_size_ratio = float(self.identity_config.min_size_ratio)
            max_size_ratio = float(self.identity_config.max_size_ratio)
        else:
            min_size_ratio = cfg.min_size_ratio
            max_size_ratio = cfg.max_size_ratio
        if size_ratio < min_size_ratio or size_ratio > max_size_ratio:
            candidate.reason = f"size mismatch ({size_ratio:.2f})"
            return candidate

        # --- hard gate 3: confidence (long gaps only) ------------------- #
        if is_long_gap and float(detection_score) < cfg.long_gap_min_confidence:
            candidate.reason = (
                f"confidence {float(detection_score):.2f} < "
                f"{cfg.long_gap_min_confidence:.2f} required after a long gap"
            )
            return candidate

        # --- soft evidence --------------------------------------------- #
        distance_score = 1.0 / (1.0 + normalized_distance)
        overlap = self._iou(reference_box, detection_box)
        candidate.overlap = overlap

        velocity_score = self._velocity_consistency(track, detection_center)
        candidate.velocity_consistency = velocity_score

        size_score = max(0.0, 1.0 - abs(math.log(max(size_ratio, 1e-6))))
        confidence_score = float(np.clip(detection_score, 0.0, 1.0))

        # --- time decay ------------------------------------------------ #
        # The decay expresses how much the *motion prediction* can still be
        # trusted, so it discounts the positional evidence only. The evidence
        # that does not depend on the prediction (size, confidence, velocity)
        # and the overlap - which is measured against a box that does not
        # drift - are not decayed. In the long-gap branch the reference is the
        # last observed box, so there is nothing to discount at all and the
        # stricter long-gap rules carry the whole burden.
        if is_long_gap or cfg.time_decay_tau <= 0:
            decay = 1.0
        else:
            decay = math.exp(-time_gap / cfg.time_decay_tau)
        distance_score *= decay
        candidate.decay = decay

        weighted = (
            cfg.distance_weight * distance_score
            + cfg.overlap_weight * overlap
            + cfg.velocity_weight * velocity_score
            + cfg.size_weight * size_score
            + cfg.confidence_weight * confidence_score
        )
        weight_sum = (
            cfg.distance_weight
            + cfg.overlap_weight
            + cfg.velocity_weight
            + cfg.size_weight
            + cfg.confidence_weight
        )
        weighted = weighted / weight_sum if weight_sum > 0 else 0.0

        # The face confirmed the identity: the geometric score describes the
        # *position*, which is exactly what is unreliable after a disappearance.
        if identity_confirmed:
            weighted = max(weighted, float(self.identity_config.confirmed_score))

        # --- appearance (optional, can only add or subtract) ----------- #
        if getattr(self.encoder, "enabled", False):
            appearance = float(self.encoder.similarity(track, detection_box))
            if appearance < cfg.appearance_threshold:
                candidate.score = weighted
                candidate.reason = f"appearance below threshold ({appearance:.2f})"
                return candidate
            weighted = weighted * (1.0 - cfg.appearance_weight) + cfg.appearance_weight * appearance

        candidate.score = float(weighted)

        if candidate.score < required_score:
            candidate.reason = f"score {candidate.score:.3f} < {required_score:.3f}"
            return candidate

        candidate.accepted = True
        candidate.reason = "accepted"
        return candidate

    # ------------------------------------------------------------------ #
    def _velocity_consistency(self, track: Track, detection_center: np.ndarray) -> float:
        """Cosine similarity between the track velocity and the displacement."""
        vx, vy = track.velocity
        speed = math.hypot(vx, vy)
        if speed < 0.75:
            # The track is essentially stationary: direction carries no
            # information, so it must not be used as evidence either way.
            return 0.5

        last_center = self._center(track.bbox)
        displacement = detection_center - last_center
        displacement_norm = float(np.linalg.norm(displacement))
        if displacement_norm < 1e-3:
            return 1.0
        cosine = (vx * displacement[0] + vy * displacement[1]) / (speed * displacement_norm)
        return float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    def select(
        self,
        tracks: Sequence[Track],
        detections: Sequence[tuple[Sequence[float], float]],
        timestamp: float = 0.0,
        identity: IdentityEvidence | None = None,
    ) -> dict[int, ReacquisitionCandidate]:
        """Decide which detections restore which lost tracks.

        Args:
            tracks: The lost tracks that may be restored.
            detections: ``(box, score)`` pairs of the unmatched detections.
            timestamp: Current time, used for the time decay.
            identity: Optional identity scores of the current frame. They are
                only used for the track that owns the reference embedding.

        Returns:
            Mapping ``detection_index -> ReacquisitionCandidate`` for the
            accepted matches only.
        """
        if not tracks or not detections or not self.config.enabled:
            return {}

        candidates: list[list[ReacquisitionCandidate]] = []
        for detection_index, (box, score) in enumerate(detections):
            row = []
            for track in tracks:
                similarity = (
                    identity.score_for(track.track_id, detection_index)
                    if identity is not None
                    else None
                )
                row.append(
                    self.evaluate(
                        track,
                        box,
                        score,
                        detection_index,
                        timestamp,
                        identity_similarity=similarity,
                    )
                )
            candidates.append(row)

        # Two confirmed candidates that are almost equally similar are
        # ambiguous: two people that look alike. The system must not guess.
        ambiguous_tracks = self._ambiguous_identity_tracks(candidates)

        # Best candidate per detection.
        best_per_detection: list[ReacquisitionCandidate | None] = []
        for row in candidates:
            if not row:
                best_per_detection.append(None)
                continue
            best_per_detection.append(max(row, key=lambda c: (c.score, -c.track_id)))

        # Best detection per track (mutual best requirement).
        best_per_track: dict[int, tuple[int, float]] = {}
        for detection_index, row in enumerate(candidates):
            for candidate in row:
                if candidate.score <= 0:
                    continue
                current = best_per_track.get(candidate.track_id)
                if current is None or candidate.score > current[1]:
                    best_per_track[candidate.track_id] = (detection_index, candidate.score)

        accepted: dict[int, ReacquisitionCandidate] = {}
        for detection_index, best in enumerate(best_per_detection):
            if best is None or not best.accepted:
                continue

            # The identity was ambiguous for this track: refuse to guess.
            if best.track_id in ambiguous_tracks:
                continue

            # Mutual best: the track must prefer this detection as well.
            track_best = best_per_track.get(best.track_id)
            if track_best is None or track_best[0] != detection_index:
                continue

            # Ambiguity guard: a close runner-up means the evidence is not
            # strong enough to force a match.
            row = candidates[detection_index]
            others = [c for c in row if c.track_id != best.track_id and c.score > 0]
            if others:
                runner_up = max(others, key=lambda c: c.score)
                if best.score - runner_up.score < self.config.ambiguity_margin:
                    continue

            accepted[detection_index] = best
        return accepted

    def _ambiguous_identity_tracks(
        self, candidates: Sequence[Sequence[ReacquisitionCandidate]]
    ) -> set[int]:
        """Tracks whose identity was confirmed by more than one candidate.

        When two faces are confirmed as the same person but are almost equally
        similar, the system cannot tell which one is the target. Such a track
        is *not* restored - keeping it lost is the safe answer.
        """
        if not self.identity_enabled:
            return set()

        by_track: dict[int, list[ReacquisitionCandidate]] = {}
        for row in candidates:
            for candidate in row:
                if candidate.identity_confirmed:
                    by_track.setdefault(candidate.track_id, []).append(candidate)

        margin = float(self.identity_config.ambiguity_margin)
        ambiguous: set[int] = set()
        for track_id, items in by_track.items():
            if len(items) < 2:
                continue
            items.sort(key=lambda c: c.identity_similarity or 0.0, reverse=True)
            gap = (items[0].identity_similarity or 0.0) - (
                items[1].identity_similarity or 0.0
            )
            if gap < margin:
                LOGGER.debug(
                    "track %d: identity is ambiguous (%.3f vs %.3f)",
                    track_id,
                    items[0].identity_similarity or 0.0,
                    items[1].identity_similarity or 0.0,
                )
                ambiguous.add(track_id)
        return ambiguous
