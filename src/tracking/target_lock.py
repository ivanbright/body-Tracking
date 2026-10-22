"""The target lock: following **one** selected face.

The tracker keeps every face alive. The target lock decides which one of those
tracks the operator cares about and then stays with it.

The distinction matters, because the two components fail in different ways:

* the **tracker** can lose a track when a face disappears - it keeps the track
  alive for a while, but its retention is short and generic,
* the **target lock** must *not* react to a disappearance by picking somebody
  else. It keeps the selected track ID, keeps predicting where the face is
  likely to be, keeps looking at every incoming frame and only gives up when
  the configured retention is exhausted.

What the lock guarantees
------------------------
* The selected track keeps its **ID** while it is lost and after it is found
  again.
* The lock never switches to another face on its own. A new detection becomes a
  new track with a new ID; it never inherits the target's ID.
* The lock reports every state change: selected, visible, lost, reacquired,
  expired, cleared.
* The selection can be changed or cleared explicitly at any time.

What the lock does **not** claim
--------------------------------
Selecting the largest face is a *geometric* heuristic: it does not identify a
person: two people of different sizes are not told apart by it, and the same
person is not recognised after they walk away.

The lock itself never *decides* who somebody is. It only carries the track ID
that the **identity layer** confirmed, and it keeps that ID. The identity
evidence is produced outside the lock (see ``src.recognition`` and
``src.tracking.reacquisition``) and only ever *restores* a track that already
existed - the identity never creates a track and never changes the selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from .track_state import Track, TrackState

LOGGER = logging.getLogger(__name__)


class TargetState(str, Enum):
    """Lifecycle state of the selected target."""

    #: No target is selected.
    IDLE = "idle"
    #: A target is selected and currently visible.
    LOCKED = "locked"
    #: A target is selected but was not observed in the current frame.
    LOST = "lost"
    #: Transient: the target was just found again after being lost.
    REACQUIRED = "reacquired"
    #: The target was missing for longer than the configured retention.
    EXPIRED = "expired"
    #: The selection was cleared explicitly by the operator.
    CLEARED = "cleared"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value.upper()


@dataclass
class TargetEvent:
    """A diagnostic event of the target lock."""

    frame: int
    timestamp: float
    kind: str
    track_id: int | None = None
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        target = f" track={self.track_id}" if self.track_id is not None else ""
        return (
            f"[frame {self.frame}] {self.kind}{target}"
            f"{(': ' + self.detail) if self.detail else ''}"
        )


@dataclass
class TargetStatus:
    """A snapshot of what the lock is doing."""

    state: TargetState = TargetState.IDLE
    track_id: int | None = None
    visible: bool = False
    time_since_seen: float = 0.0
    last_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    predicted_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    has_prediction: bool = False
    reacquisition_count: int = 0

    @property
    def label(self) -> str:
        if self.track_id is None:
            return "NO TARGET"
        return f"TARGET #{self.track_id} [{self.state.value.upper()}]"


class TargetLock:
    """Selects one track and keeps following it while it is temporarily gone."""

    def __init__(self, config, *, tracker=None, prediction_config=None) -> None:
        self.config = config
        self.tracker = tracker
        self.prediction_config = prediction_config
        self._track_id: int | None = None
        self._state = TargetState.IDLE
        self._events: list[TargetEvent] = []
        self._auto_select_done = False

    # ------------------------------------------------------------------ #
    @property
    def state(self) -> TargetState:
        return self._state

    @property
    def track_id(self) -> int | None:
        """The ID of the selected track, or ``None``."""
        return self._track_id

    @property
    def is_selected(self) -> bool:
        return self._track_id is not None

    @property
    def events(self) -> list[TargetEvent]:
        return list(self._events)

    def status(self, tracks: Sequence[Track] | None = None) -> TargetStatus:
        """Describe the current state of the target."""
        track = self.target_track(tracks)
        status = TargetStatus(state=self._state, track_id=self._track_id)
        if track is None:
            return status
        status.visible = track.state in (TrackState.TRACKED, TrackState.REACQUIRED, TrackState.NEW)
        status.time_since_seen = track.time_since_seen
        status.last_bbox = track.bbox
        status.predicted_bbox = track.predicted_bbox
        status.reacquisition_count = track.reacquisition_count
        status.has_prediction = self._prediction_is_fresh(track)
        return status

    def target_track(self, tracks: Sequence[Track] | None = None) -> Track | None:
        """The :class:`Track` of the selected target, if it still exists."""
        if self._track_id is None:
            return None
        if tracks is not None:
            for track in tracks:
                if track.track_id == self._track_id:
                    return track
            return None
        if self.tracker is not None:
            return self.tracker.track(self._track_id)
        return None

    # ------------------------------------------------------------------ #
    def select(self, track: Track, *, frame: int = 0, timestamp: float = 0.0) -> None:
        """Select a track as the target."""
        if self.config.require_confirmed and not track.is_confirmed and track.state is TrackState.NEW:
            LOGGER.debug(
                "Refusing to select the unconfirmed track %d", track.track_id
            )
            return
        previous = self._track_id
        self._track_id = track.track_id
        self._state = TargetState.LOCKED
        self._log(
            "selected",
            frame=frame,
            timestamp=timestamp,
            detail=(
                f"track={track.track_id}"
                + (f" (replaced {previous})" if previous is not None else "")
            ),
        )
        self._protect(track.track_id)

    def select_track_id(self, track_id: int, *, frame: int = 0, timestamp: float = 0.0) -> bool:
        """Select a track by its ID. Returns ``True`` when the track exists."""
        if self.tracker is None:
            return False
        track = self.tracker.track(track_id)
        if track is None:
            self._log(
                "select_failed",
                frame=frame,
                timestamp=timestamp,
                detail=f"track {track_id} does not exist",
            )
            return False
        self.select(track, frame=frame, timestamp=timestamp)
        return True

    def clear(self, *, frame: int = 0, timestamp: float = 0.0) -> None:
        """Clear the selection explicitly."""
        if self._track_id is not None:
            self._log("cleared", frame=frame, timestamp=timestamp, detail=f"track={self._track_id}")
            self._unprotect(self._track_id)
        self._track_id = None
        self._state = TargetState.CLEARED

    # ------------------------------------------------------------------ #
    def update(
        self,
        tracks: Sequence[Track],
        timestamp: float,
        frame: int = 0,
    ) -> TargetState:
        """Update the lock for the current frame."""
        if self._track_id is None:
            self._maybe_auto_select(tracks, timestamp, frame)
            return self._state

        track = None
        for candidate in tracks:
            if candidate.track_id == self._track_id:
                track = candidate
                break

        if track is None:
            # The tracker dropped the track (it expired). The target is gone.
            self._expire(frame=frame, timestamp=timestamp)
            return self._state

        self._protect(track.track_id)

        if track.state in (TrackState.TRACKED, TrackState.REACQUIRED, TrackState.NEW):
            if track.state is TrackState.REACQUIRED:
                self._state = TargetState.REACQUIRED
                self._log(
                    "reacquired",
                    frame=frame,
                    timestamp=timestamp,
                    detail=f"gap={track.last_reacquisition_gap:.2f}s",
                )
            else:
                if self._state is not TargetState.LOCKED:
                    self._state = TargetState.LOCKED
                    self._log("visible", frame=frame, timestamp=timestamp)
            return self._state

        # The target is not visible in this frame.
        if self._state is not TargetState.LOST:
            self._state = TargetState.LOST
            self._log(
                "lost",
                frame=frame,
                timestamp=timestamp,
                detail=f"since={track.time_since_seen:.2f}s",
            )

        if track.time_since_seen > self.config.lost_retention_seconds:
            self._expire(frame=frame, timestamp=timestamp)

        return self._state

    # ------------------------------------------------------------------ #
    def _maybe_auto_select(
        self, tracks: Sequence[Track], timestamp: float, frame: int
    ) -> None:
        """Select a target when the configuration asks for it.

        ``target.mode == "largest"`` picks the largest *visible* face. That is
        a geometric heuristic, not an identity decision: it does not know who
        the face belongs to.

        The automatic selection happens **once**. When the target expires the
        lock stays idle unless ``target.auto_reselect`` is enabled, because
        silently switching to a different face is exactly what must not
        happen.
        """
        if self.config.mode != "largest" or self._auto_select_done:
            return
        if self._state in (TargetState.EXPIRED, TargetState.CLEARED):
            return
        candidates = [
            t
            for t in tracks
            if t.state in (TrackState.TRACKED, TrackState.REACQUIRED)
            and (t.is_confirmed or not self.config.require_confirmed)
        ]
        if not candidates:
            return
        best = max(candidates, key=lambda t: t.bbox[2] * t.bbox[3])
        self._auto_select_done = True
        self.select(best, frame=frame, timestamp=timestamp)

    def _expire(self, *, frame: int, timestamp: float) -> None:
        track_id = self._track_id
        self._log("expired", frame=frame, timestamp=timestamp, detail=f"track={track_id}")
        if track_id is not None:
            self._unprotect(track_id)
        self._track_id = None
        self._state = TargetState.EXPIRED
        if self.config.auto_reselect:
            self._auto_select_done = False

    def _prediction_is_fresh(self, track: Track) -> bool:
        if self.prediction_config is None:
            return False
        horizon = float(self.prediction_config.max_prediction_time)
        if horizon <= 0:
            return False
        return track.time_since_seen <= horizon

    def _protect(self, track_id: int) -> None:
        if self.tracker is not None:
            self.tracker.protect(track_id, self.config.lost_retention_seconds)

    def _unprotect(self, track_id: int) -> None:
        if self.tracker is not None:
            self.tracker.unprotect(track_id)

    def _log(self, kind: str, *, frame: int, timestamp: float, detail: str = "") -> None:
        event = TargetEvent(
            frame=frame,
            timestamp=timestamp,
            kind=kind,
            track_id=self._track_id,
            detail=detail,
        )
        self._events.append(event)
        LOGGER.info("target: %s", event)
