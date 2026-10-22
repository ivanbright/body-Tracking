"""Tracking layer.

The layer has two clearly separated components:

``Tracker``
    keeps one independent track per detected face.
``TargetLock``
    selects one of those tracks and follows it, even while it is temporarily
    invisible.

The separation is what makes the behaviour predictable: the tracker never
decides that a person is "the target" and the lock never creates tracks.
"""

from __future__ import annotations

from .association import (
    association_cost,
    centre_distance_matrix,
    iou_matrix,
    linear_assignment,
)
from .motion_prediction import KalmanBoxFilter, cxah_to_xyxy, xyxy_to_cxah
from .reacquisition import (
    AppearanceEncoder,
    NullAppearanceEncoder,
    ReacquisitionCandidate,
    ReacquisitionEngine,
)
from .target_lock import TargetEvent, TargetLock, TargetState, TargetStatus
from .track_state import (
    ALLOWED_TRANSITIONS,
    ACTIVE_STATES,
    IllegalTransitionError,
    Track,
    TrackEvent,
    TrackState,
    summarize_tracks,
)
from .tracker import Tracker, TrackerResult

__all__ = [
    "ACTIVE_STATES",
    "ALLOWED_TRANSITIONS",
    "AppearanceEncoder",
    "IllegalTransitionError",
    "KalmanBoxFilter",
    "NullAppearanceEncoder",
    "ReacquisitionCandidate",
    "ReacquisitionEngine",
    "TargetEvent",
    "TargetLock",
    "TargetState",
    "TargetStatus",
    "Track",
    "TrackEvent",
    "TrackState",
    "Tracker",
    "TrackerResult",
    "association_cost",
    "centre_distance_matrix",
    "cxah_to_xyxy",
    "iou_matrix",
    "linear_assignment",
    "summarize_tracks",
    "xyxy_to_cxah",
]
