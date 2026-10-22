"""Track state, the lifecycle state machine and the track record.

The tracker keeps a single, explicit state machine so that "where is my
target?" always has a precise answer:

``NEW``
    A detection created a candidate track. It has not been confirmed yet.
``TRACKED``
    The track is associated with a detection right now.
``LOST``
    The track was not associated with any detection, but it is **retained**:
    its ID, its last reliable box, its motion state and its history are kept
    so that the person can be found again.
``REACQUIRED``
    A detection was accepted for this lost track. The original ID is
    restored. This state is transient - the next successful update moves the
    track back to ``TRACKED``.
``EXPIRED``
    The track exceeded its retention limit. It is removed from the active
    pool and only minimal diagnostic information is kept.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Deque, Iterable

import numpy as np

from .motion_prediction import KalmanBoxFilter


class TrackState(str, Enum):
    """Lifecycle state of a track."""

    NEW = "new"
    TRACKED = "tracked"
    LOST = "lost"
    REACQUIRED = "reacquired"
    EXPIRED = "expired"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value.upper()


class IllegalTransitionError(RuntimeError):
    """Raised when a track is asked to make a transition that is not allowed."""


#: Allowed state transitions. Anything not listed here is rejected.
ALLOWED_TRANSITIONS: dict[TrackState, frozenset[TrackState]] = {
    TrackState.NEW: frozenset({TrackState.TRACKED, TrackState.LOST, TrackState.EXPIRED}),
    TrackState.TRACKED: frozenset({TrackState.TRACKED, TrackState.LOST, TrackState.EXPIRED}),
    TrackState.LOST: frozenset({TrackState.LOST, TrackState.REACQUIRED, TrackState.EXPIRED}),
    TrackState.REACQUIRED: frozenset(
        {TrackState.TRACKED, TrackState.REACQUIRED, TrackState.LOST, TrackState.EXPIRED}
    ),
    TrackState.EXPIRED: frozenset(),
}

#: States whose track still occupies a slot in the active pool.
ACTIVE_STATES = frozenset(
    {TrackState.NEW, TrackState.TRACKED, TrackState.LOST, TrackState.REACQUIRED}
)


@dataclass
class TrackEvent:
    """A diagnostic event emitted by a track."""

    frame: int
    timestamp: float
    kind: str
    detail: str = ""

    def __str__(self) -> str:
        return f"[frame {self.frame}] {self.kind}{(': ' + self.detail) if self.detail else ''}"


@dataclass
class Track:
    """A tracked face.

    The record keeps everything that is needed to find the person again:
    the last reliable box, the motion state, the time since the last
    observation and a bounded history of past observations.
    """

    track_id: int
    state: TrackState = TrackState.NEW
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    predicted_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    score: float = 0.0
    hits: int = 0
    time_since_update: int = 0
    frames_since_creation: int = 0
    first_frame: int = 0
    last_seen_frame: int = 0
    last_seen_time: float = 0.0
    created_time: float = 0.0
    kalman: KalmanBoxFilter | None = None
    history: Deque[tuple[int, float, tuple[float, float, float, float]]] = field(
        default_factory=lambda: deque(maxlen=64)
    )
    events: list[TrackEvent] = field(default_factory=list)
    reacquisition_count: int = 0
    #: How long the track was missing before it was restored (seconds).
    last_reacquisition_gap: float = 0.0
    #: The embeddings of this face - one per pose that was observed.
    #: A face that turns to the side produces a different embedding, so the
    #: set is what makes the recognition survive a head turn.
    embeddings: list[np.ndarray] = field(default_factory=list, compare=False, repr=False)
    #: The name of the enrolled person this track was identified as.
    identity_name: str = ""
    #: The similarity of the last recognition attempt.
    identity_similarity: float = 0.0
    #: Whether this face was already compared with the gallery.
    identity_checked: bool = False
    #: How often the track was restored because the identity confirmed it.
    identity_reacquisition_count: int = 0
    is_confirmed: bool = False
    class_name: str = "face"

    # ------------------------------------------------------------------ #
    def initialize(self, bbox: tuple[float, float, float, float], score: float, frame: int, timestamp: float) -> None:
        """Set the first observation of a brand new track."""
        self.bbox = tuple(float(v) for v in bbox)
        self.predicted_bbox = self.bbox
        self.score = float(score)
        self.hits = 1
        self.time_since_update = 0
        self.frames_since_creation = 0
        self.first_frame = frame
        self.last_seen_frame = frame
        self.last_seen_time = timestamp
        self.created_time = timestamp
        self.kalman = KalmanBoxFilter(self.bbox)
        self.history.append((frame, timestamp, self.bbox))

    # ------------------------------------------------------------------ #
    def transition_to(self, new_state: TrackState, *, frame: int = 0, timestamp: float = 0.0, detail: str = "") -> None:
        """Move to ``new_state``, validating the transition.

        Raises:
            IllegalTransitionError: The transition is not part of the lifecycle.
        """
        if new_state == self.state:
            return
        allowed = ALLOWED_TRANSITIONS.get(self.state, frozenset())
        if new_state not in allowed:
            raise IllegalTransitionError(
                f"track {self.track_id}: illegal transition {self.state.value} -> {new_state.value}"
            )
        previous = self.state
        self.state = new_state
        self.events.append(TrackEvent(frame=frame, timestamp=timestamp, kind="state", detail=f"{previous.value}->{new_state.value}"))

    # ------------------------------------------------------------------ #
    def mark_missed(self) -> None:
        """Record that no detection was associated with this track."""
        self.time_since_update += 1
        self.frames_since_creation += 1

    def update(self, bbox: tuple[float, float, float, float], score: float, frame: int, timestamp: float) -> None:
        """Fold a new observation into the track."""
        self.bbox = tuple(float(v) for v in bbox)
        self.predicted_bbox = self.bbox
        self.score = float(score)
        self.hits += 1
        self.time_since_update = 0
        self.frames_since_creation += 1
        self.last_seen_frame = frame
        self.last_seen_time = timestamp
        if self.kalman is None:
            self.kalman = KalmanBoxFilter(self.bbox)
        else:
            self.kalman.update(self.bbox)
        self.history.append((frame, timestamp, self.bbox))

    # ------------------------------------------------------------------ #
    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def predicted_center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.predicted_bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def diagonal(self) -> float:
        return float(np.hypot(self.width, self.height))

    @property
    def velocity(self) -> tuple[float, float]:
        if self.kalman is None:
            return (0.0, 0.0)
        return self.kalman.velocity

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_STATES

    @property
    def is_lost(self) -> bool:
        return self.state is TrackState.LOST

    @property
    def time_since_seen(self) -> float:
        """Seconds since the last reliable observation (needs ``last_seen_time``)."""
        return self._now - self.last_seen_time if self.last_seen_time else 0.0

    #: Timestamp of the frame currently being processed. The tracker refreshes
    #: it on every frame; it is a class variable because it is identical for
    #: every track and must not take part in ``__init__``.
    _now: ClassVar[float] = 0.0

    def set_now(self, timestamp: float) -> None:
        """Publish the timestamp of the frame that is being processed.

        The value is written to the class attribute on purpose: the tracker
        reads it as ``Track._now`` while it decides whether the prediction of
        a track is still fresh.
        """
        Track._now = float(timestamp)
        self._now = float(timestamp)

    def log_event(self, kind: str, frame: int, timestamp: float, detail: str = "") -> None:
        self.events.append(TrackEvent(frame=frame, timestamp=timestamp, kind=kind, detail=detail))

    # ------------------------------------------------------------------ #
    @property
    def embedding(self) -> np.ndarray | None:
        """The mean template of the observed poses (L2-normalized).

        This is the value of the working reference system: the average of all
        the samples of a person. It is kept for the target reference, while
        the recognition itself uses :meth:`similarity_to`.
        """
        if not self.embeddings:
            return None
        mean = np.mean(np.stack(self.embeddings, axis=0), axis=0)
        norm = float(np.linalg.norm(mean))
        if norm <= 1e-12:
            return None
        return (mean / norm).astype(np.float32)

    @property
    def embedding_count(self) -> int:
        """How many poses are stored."""
        return len(self.embeddings)

    def similarity_to(self, embedding: np.ndarray | None) -> float:
        """The best similarity against the stored poses.

        The **maximum** over all the poses is used: a face that turns to the
        side is very similar to the sample that was captured from that side,
        even when it is quite different from the average of all the samples.
        """
        if embedding is None or not self.embeddings:
            return 0.0
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if vector.size == 0 or norm <= 1e-12:
            return 0.0
        vector = vector / norm
        return max(float(np.dot(stored, vector)) for stored in self.embeddings)

    def add_embedding(
        self,
        embedding: np.ndarray,
        min_similarity: float = 0.0,
        max_embeddings: int = 8,
    ) -> tuple[bool, float]:
        """Add an observed pose to this face.

        An observation that is too far away from every known pose is rejected:
        it is most likely a different person and must not pollute the track.

        When the set is full, the most redundant sample is dropped - the one
        that is the most similar to all the others - so that the different
        angles survive.

        Returns:
            ``(accepted, similarity)`` - the similarity to the closest known
            pose (1.0 for the very first observation).
        """
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if vector.size == 0 or norm <= 1e-12:
            return False, 0.0
        vector = (vector / norm).astype(np.float32)

        if not self.embeddings:
            self.embeddings.append(vector)
            return True, 1.0

        similarity = max(float(np.dot(stored, vector)) for stored in self.embeddings)
        if similarity < min_similarity:
            return False, similarity

        if max_embeddings > 0 and len(self.embeddings) >= max_embeddings:
            self._drop_redundant_sample()
        self.embeddings.append(vector)
        return True, similarity

    def _drop_redundant_sample(self) -> None:
        """Remove the sample that carries the least new information."""
        if len(self.embeddings) < 2:
            return
        redundancy: list[float] = []
        for index, sample in enumerate(self.embeddings):
            others = [float(np.dot(other, sample)) for position, other in enumerate(self.embeddings)
                      if position != index]
            redundancy.append(sum(others) / len(others))
        del self.embeddings[int(max(range(len(redundancy)), key=lambda i: redundancy[i]))]

    def trail(self, length: int) -> list[tuple[int, float, tuple[float, float, float, float]]]:
        if length <= 0:
            return []
        items = list(self.history)
        return items[-length:]

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<Track id={self.track_id} state={self.state.value} hits={self.hits} "
            f"missed={self.time_since_update} box={tuple(round(v, 1) for v in self.bbox)}>"
        )


def summarize_tracks(tracks: Iterable[Track]) -> dict[str, int]:
    """Count the tracks per lifecycle state."""
    counts = {state.value: 0 for state in TrackState}
    for track in tracks:
        counts[track.state.value] += 1
    return counts
