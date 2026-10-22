"""Motion prediction for tracked faces.

The tracker needs to know where a face is *expected* to be while it cannot be
observed: the detector flickers under motion blur, a head turns away, someone
walks behind a column. A constant-velocity Kalman filter gives a short term
estimate of the position, and that estimate is what the association step and
the reacquisition step are gated against.

The filter is the classic 8-state box filter used by SORT / ByteTrack:

``state = [cx, cy, area, height, vx, vy, v_area, v_height]``
``measurement = [cx, cy, area, height]``

Implementation notes
--------------------
* Pure NumPy, no SciPy dependency - the state space is tiny (8x8).
* The filter is *predictive only*: a prediction guides the search, it never
  proves that an object is the one we are looking for.
* ``max_prediction_distance`` caps how far a prediction may travel, so a track
  cannot drift out of the frame while it is lost.

Provenance: the constant-velocity box Kalman filter was introduced by SORT
(Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016) and is used
by ByteTrack (Zhang et al., ECCV 2022). Both are MIT licensed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

BoxLike = Sequence[float]


def xyxy_to_cxah(box: BoxLike) -> np.ndarray:
    """Convert ``(x1, y1, x2, y2)`` into the ``(cx, cy, area, height)`` measurement."""
    x1, y1, x2, y2 = (float(v) for v in box)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    return np.array([x1 + width / 2.0, y1 + height / 2.0, area, height], dtype=np.float32)


def cxah_to_xyxy(state: Sequence[float]) -> tuple[float, float, float, float]:
    """Convert the ``(cx, cy, area, height)`` measurement back into ``xyxy``."""
    cx, cy, area, height = (float(v) for v in state)
    height = max(height, 1e-3)
    width = max(area / height, 1e-3)
    return (
        cx - width / 2.0,
        cy - height / 2.0,
        cx + width / 2.0,
        cy + height / 2.0,
    )


class KalmanBoxFilter:
    """Constant-velocity Kalman filter over a bounding box."""

    def __init__(self, bbox: BoxLike) -> None:
        self._motion_mat = np.eye(8, dtype=np.float32)
        for i in range(4):
            self._motion_mat[i, i + 4] = 1.0

        self._update_mat = np.eye(4, 8, dtype=np.float32)

        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

        measurement = xyxy_to_cxah(bbox)
        self._state = np.zeros(8, dtype=np.float32)
        self._state[:4] = measurement
        self._state[4:] = 0.0

        self._covariance = np.eye(8, dtype=np.float32)
        self._covariance[4:, 4:] *= 1000.0

        self._measurement_covariance = np.eye(4, dtype=np.float32) * self._std_weight_position
        self._motion_covariance = np.eye(8, dtype=np.float32) * self._std_weight_velocity
        self._motion_covariance[4:, 4:] *= self._std_weight_velocity

    # ------------------------------------------------------------------ #
    def predict(self, max_distance: float = 0.0) -> tuple[float, float, float, float]:
        """Advance the filter by one frame and return the predicted box."""
        # Predict.
        self._state = self._motion_mat @ self._state
        self._covariance = self._motion_mat @ self._covariance @ self._motion_mat.T + self._motion_covariance

        box = cxah_to_xyxy(self._state[:4])
        if max_distance and max_distance > 0:
            box = self._clamp_movement(box, max_distance)
        return box

    def update(self, bbox: BoxLike) -> None:
        """Fold an observation into the filter."""
        measurement = xyxy_to_cxah(bbox)
        innovation = measurement - self._update_mat @ self._state
        innovation_covariance = (
            self._update_mat @ self._covariance @ self._update_mat.T + self._measurement_covariance
        )
        kalman_gain = (
            self._covariance @ self._update_mat.T @ np.linalg.inv(innovation_covariance)
        )
        self._state = self._state + kalman_gain @ innovation

        identity = np.eye(8, dtype=np.float32)
        residual = identity - kalman_gain @ self._update_mat
        self._covariance = residual @ self._covariance @ residual.T + kalman_gain @ self._measurement_covariance @ kalman_gain.T

    # ------------------------------------------------------------------ #
    def _clamp_movement(
        self, box: tuple[float, float, float, float], max_distance: float
    ) -> tuple[float, float, float, float]:
        """Limit how far the prediction may move compared to the previous state."""
        previous = cxah_to_xyxy(self._state[:4])
        pcx = (previous[0] + previous[2]) / 2.0
        pcy = (previous[1] + previous[3]) / 2.0
        ncx = (box[0] + box[2]) / 2.0
        ncy = (box[1] + box[3]) / 2.0
        dx = ncx - pcx
        dy = ncy - pcy
        distance = float(np.hypot(dx, dy))
        if distance <= max_distance or distance == 0.0:
            return box
        scale = max_distance / distance
        shifted_cx = pcx + dx * scale
        shifted_cy = pcy + dy * scale
        width = box[2] - box[0]
        height = box[3] - box[1]
        return (
            shifted_cx - width / 2.0,
            shifted_cy - height / 2.0,
            shifted_cx + width / 2.0,
            shifted_cy + height / 2.0,
        )

    # ------------------------------------------------------------------ #
    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @property
    def velocity(self) -> tuple[float, float]:
        """Estimated velocity of the box centre, in pixels per frame."""
        return (float(self._state[4]), float(self._state[5]))

    @property
    def speed(self) -> float:
        vx, vy = self.velocity
        return float(np.hypot(vx, vy))

    @property
    def measured_box(self) -> tuple[float, float, float, float]:
        return cxah_to_xyxy(self._state[:4])

    def reset(self, bbox: BoxLike) -> None:
        """Re-initialise the filter from a new observation."""
        measurement = xyxy_to_cxah(bbox)
        self._state[:4] = measurement
        self._state[4:] = 0.0
        self._covariance = np.eye(8, dtype=np.float32)
        self._covariance[4:, 4:] *= 1000.0


def predicted_center(velocity: tuple[float, float], steps: int) -> tuple[float, float]:
    """Extrapolate a centre position ``steps`` frames ahead."""
    vx, vy = velocity
    return (vx * steps, vy * steps)
