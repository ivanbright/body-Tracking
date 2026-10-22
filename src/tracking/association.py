"""Association between detections and tracks.

The tracker has to answer one question every frame: *which detection belongs
to which track?* The answer is found by building a cost matrix and solving
the linear assignment problem optimally (Hungarian algorithm).

The cost blends two terms:

* **IoU** - how much the boxes overlap. Strong evidence, but it collapses
  when a face moves fast or is partially occluded.
* **normalised centre distance** - how far the centres are, relative to the
  size of the boxes. Weak on its own, but it keeps the identity of a fast
  moving face, which is exactly the situation where IoU alone fails.

``scipy.optimize.linear_sum_assignment`` solves the assignment optimally and
is already available in the environment, so no extra dependency (and no
compilation) is required.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

BoxLike = Sequence[float]


def boxes_to_array(boxes: Sequence[BoxLike]) -> np.ndarray:
    """Stack ``(x1, y1, x2, y2)`` boxes into an ``(N, 4)`` float array."""
    if boxes is None:
        return np.zeros((0, 4), dtype=np.float32)
    array = np.asarray(boxes, dtype=np.float32)
    if array.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return array.reshape(-1, 4)


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of boxes."""
    a = boxes_to_array(boxes_a)
    b = boxes_to_array(boxes_b)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    intersection = wh[..., 0] * wh[..., 1]

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, intersection / np.maximum(union, 1e-9), 0.0)
    return iou.astype(np.float32)


def centre_distance_matrix(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Pairwise centre distance between two sets of boxes.

    With ``normalize=True`` the distance is divided by the mean box diagonal,
    so the result is roughly "how many box widths apart are they" and is
    therefore independent of the resolution of the frame.
    """
    a = boxes_to_array(boxes_a)
    b = boxes_to_array(boxes_b)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    centre_a = np.stack([(a[:, 0] + a[:, 2]) / 2.0, (a[:, 1] + a[:, 3]) / 2.0], axis=1)
    centre_b = np.stack([(b[:, 0] + b[:, 2]) / 2.0, (b[:, 1] + b[:, 3]) / 2.0], axis=1)
    distances = np.linalg.norm(centre_a[:, None, :] - centre_b[None, :, :], axis=2)

    if normalize:
        diag_a = np.hypot(a[:, 2] - a[:, 0], a[:, 3] - a[:, 1])
        diag_b = np.hypot(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
        scale = np.maximum((diag_a[:, None] + diag_b[None, :]) / 2.0, 1.0)
        distances = distances / scale
    return distances.astype(np.float32)


def association_cost(
    track_boxes: np.ndarray,
    detection_boxes: np.ndarray,
    *,
    distance_weight: float = 0.35,
    use_motion: bool = True,
    velocities: Sequence[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Build the association cost matrix.

    Args:
        track_boxes: Boxes of the tracks (predicted when ``use_motion``).
        detection_boxes: Boxes of the detections.
        distance_weight: Weight of the distance term (0 = pure IoU, 1 = pure distance).
        use_motion: Use the motion-aware box (predicted) instead of the last observed box.
        velocities: Optional per-track velocity, used to widen the gate for fast movers.

    Returns:
        Cost matrix of shape ``(num_tracks, num_detections)`` in ``[0, ~1+]``.
        A cost above the match threshold means "not a candidate".
    """
    weight = float(np.clip(distance_weight, 0.0, 1.0))
    iou = iou_matrix(track_boxes, detection_boxes)
    distance = centre_distance_matrix(track_boxes, detection_boxes, normalize=True)

    cost = (1.0 - weight) * (1.0 - iou) + weight * np.clip(distance, 0.0, 2.0)

    if use_motion and velocities:
        # A track that is moving fast is expected to be further away, so the
        # distance term is relaxed for it.
        track_boxes_arr = boxes_to_array(track_boxes)
        for index, velocity in enumerate(velocities):
            if index >= cost.shape[0]:
                break
            speed = float(np.hypot(velocity[0], velocity[1]))
            if speed <= 0.5:
                continue
            diag = max(
                1.0,
                float(
                    np.hypot(
                        track_boxes_arr[index, 2] - track_boxes_arr[index, 0],
                        track_boxes_arr[index, 3] - track_boxes_arr[index, 1],
                    )
                ),
            )
            allowance = min(2.0, speed / diag)
            mask = distance[index] > allowance
            if np.any(mask):
                blended = np.where(mask, cost[index] * (1.0 - 0.5 * weight), cost[index])
                cost[index] = blended
    return cost.astype(np.float32)


def linear_assignment(
    cost_matrix: np.ndarray,
    threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Solve the assignment problem.

    Args:
        cost_matrix: ``(num_tracks, num_detections)`` cost matrix.
        threshold: Maximum cost for a pair to be accepted.

    Returns:
        ``(matches, unmatched_track_indices, unmatched_detection_indices)``.
        Every track and every detection appears in at most one match.
    """
    matrix = np.asarray(cost_matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"cost_matrix must be 2D, got shape {matrix.shape}")
    num_tracks, num_detections = matrix.shape
    if num_tracks == 0 or num_detections == 0:
        return [], list(range(num_tracks)), list(range(num_detections))

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:  # pragma: no cover - scipy is a hard requirement
        raise RuntimeError(
            "scipy is required for the assignment step. Install it with: pip install scipy"
        )

    row_indices, column_indices = linear_sum_assignment(matrix)
    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_columns: set[int] = set()
    for row, column in zip(row_indices, column_indices):
        cost = float(matrix[row, column])
        if cost <= threshold:
            matches.append((int(row), int(column)))
            matched_rows.add(int(row))
            matched_columns.add(int(column))

    unmatched_tracks = [i for i in range(num_tracks) if i not in matched_rows]
    unmatched_detections = [j for j in range(num_detections) if j not in matched_columns]
    return matches, unmatched_tracks, unmatched_detections
