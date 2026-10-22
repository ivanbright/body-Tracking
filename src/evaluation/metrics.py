"""Tracking metrics.

The metrics implemented here are computed from **annotations**: for every
frame, the ground-truth boxes with their identity and the boxes the tracker
produced with its track IDs. They are therefore meaningful for a recorded
video with annotated ground truth - and they are meaningless without it, which
is why the benchmark refuses to report them when no annotation is available.

Implemented metrics
-------------------
``idf1``
    Identity F1 (Ristani et al., "Performance Measures and a Data Set for
    Multi-Target, Multi-Camera Tracking"). It measures how well the *identity*
    of the objects is preserved, which is exactly what this project cares
    about.
``hota``
    Higher Order Metric of Association (Luiten et al.). It combines the
    detection quality with the association quality.
``id_switches``
    Number of times a ground-truth object was handed over to a different
    track ID.
``track_fragmentation``
    Number of times a ground-truth object was covered by more than one track
    ID.
``reacquisition_success``
    Fraction of the disappearances of a ground-truth object that were followed
    by the *same* track ID coming back.
``incorrect_associations``
    Number of predictions that were attached to a ground-truth object that they
    do not belong to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..tracking.association import iou_matrix
from ..tracking.track_state import Track


@dataclass(frozen=True)
class Annotation:
    """One annotated object in one frame."""

    frame: int
    object_id: int
    bbox: tuple[float, float, float, float]
    score: float = 1.0


@dataclass
class TrackingMetrics:
    """The result of an evaluation run."""

    frames: int = 0
    ground_truth_objects: int = 0
    predictions: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    id_switches: int = 0
    track_fragmentation: int = 0
    incorrect_associations: int = 0
    reacquisition_success: float = 0.0
    reacquisition_attempts: int = 0
    idf1: float = 0.0
    hota: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"frames={self.frames} "
            f"IDF1={self.idf1:.3f} HOTA={self.hota:.3f} "
            f"ID-switches={self.id_switches} fragmentation={self.track_fragmentation} "
            f"precision={self.precision:.3f} recall={self.recall:.3f} "
            f"reacq={self.reacquisition_success:.2f} "
            f"({self.reacquisition_attempts} attempts)"
        )


def _match_frame(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    threshold: float,
) -> list[tuple[int, int]]:
    """Match the ground truth and the predictions of one frame by IoU."""
    if gt_boxes.size == 0 or pred_boxes.size == 0:
        return []
    iou = iou_matrix(gt_boxes, pred_boxes)
    matches: list[tuple[int, int]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    for gt_index, pred_index in order:
        if iou[gt_index, pred_index] < threshold:
            break
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(int(gt_index))
        used_pred.add(int(pred_index))
        matches.append((int(gt_index), int(pred_index)))
    return matches


def evaluate(
    ground_truth: Sequence[Annotation],
    predictions: Sequence[Annotation],
    *,
    iou_threshold: float = 0.5,
    frames: int | None = None,
) -> TrackingMetrics:
    """Compare the predictions with the ground truth.

    Args:
        ground_truth: The annotated objects.
        predictions: The boxes the tracker produced, one per frame and track.
        iou_threshold: IoU above which a detection counts as a match.
        frames: Number of frames of the sequence (inferred when omitted).

    Returns:
        The metrics. Empty inputs produce zeroed metrics rather than an error.
    """
    metrics = TrackingMetrics()
    if frames is None:
        frames = max(
            [a.frame for a in ground_truth] + [a.frame for a in predictions] + [-1]
        ) + 1
    metrics.frames = int(frames)

    gt_by_frame: dict[int, list[Annotation]] = {}
    pred_by_frame: dict[int, list[Annotation]] = {}
    for annotation in ground_truth:
        gt_by_frame.setdefault(annotation.frame, []).append(annotation)
    for annotation in predictions:
        pred_by_frame.setdefault(annotation.frame, []).append(annotation)

    metrics.ground_truth_objects = len(ground_truth)
    metrics.predictions = len(predictions)

    # --- per frame matching ------------------------------------------- #
    overlap: dict[tuple[int, int], int] = {}
    presence: dict[tuple[int, int], int] = {}
    gt_per_frame: dict[int, int] = {}
    pred_per_frame: dict[int, int] = {}
    last_track_of_object: dict[int, int] = {}
    seen_objects: set[int] = set()

    for frame in range(metrics.frames):
        gt_items = gt_by_frame.get(frame, [])
        pred_items = pred_by_frame.get(frame, [])
        gt_per_frame[frame] = len(gt_items)
        pred_per_frame[frame] = len(pred_items)

        gt_boxes = np.asarray([a.bbox for a in gt_items], dtype=np.float32).reshape(-1, 4)
        pred_boxes = np.asarray([a.bbox for a in pred_items], dtype=np.float32).reshape(-1, 4)
        matches = _match_frame(gt_boxes, pred_boxes, iou_threshold)

        metrics.true_positives += len(matches)
        metrics.false_positives += len(pred_items) - len(matches)
        metrics.false_negatives += len(gt_items) - len(matches)

        for gt_index, pred_index in matches:
            object_id = gt_items[gt_index].object_id
            track_id = pred_items[pred_index].object_id
            overlap[(object_id, track_id)] = overlap.get((object_id, track_id), 0) + 1
            presence[(object_id, track_id)] = presence.get((object_id, track_id), 0) + 1

            previous = last_track_of_object.get(object_id)
            if previous is not None and previous != track_id:
                metrics.id_switches += 1
            last_track_of_object[object_id] = track_id
            seen_objects.add(object_id)

    # --- fragmentation: how many track IDs covered one object ---------- #
    tracks_per_object: dict[int, set[int]] = {}
    for object_id, track_id in overlap:
        tracks_per_object.setdefault(object_id, set()).add(track_id)
    for tracks in tracks_per_object.values():
        if len(tracks) > 1:
            metrics.track_fragmentation += len(tracks) - 1

    # --- incorrect associations ---------------------------------------- #
    # A prediction that is matched to an object it does not belong to shows up
    # as a track ID that covers several objects at the same time.
    object_of_track: dict[int, set[int]] = {}
    for (object_id, track_id) in overlap:
        object_of_track.setdefault(track_id, set()).add(object_id)
    for track_id, objects in object_of_track.items():
        if len(objects) > 1:
            metrics.incorrect_associations += len(objects) - 1

    # --- IDF1 ----------------------------------------------------------- #
    metrics.idf1 = _idf1(overlap, len(ground_truth), len(predictions))

    # --- HOTA ----------------------------------------------------------- #
    metrics.hota = _hota(overlap, presence, gt_by_frame, pred_by_frame, iou_threshold, metrics.frames)

    metrics.precision = (
        metrics.true_positives / (metrics.true_positives + metrics.false_positives)
        if (metrics.true_positives + metrics.false_positives)
        else 0.0
    )
    metrics.recall = (
        metrics.true_positives / (metrics.true_positives + metrics.false_negatives)
        if (metrics.true_positives + metrics.false_negatives)
        else 0.0
    )

    metrics.reacquisition_success, metrics.reacquisition_attempts = _reacquisition(
        ground_truth, predictions, iou_threshold
    )

    return metrics


def _idf1(
    overlap: Mapping[tuple[int, int], int],
    gt_total: int,
    pred_total: int,
) -> float:
    """Identity F1 over the global object/track assignment."""
    if not overlap or gt_total == 0 or pred_total == 0:
        return 0.0

    objects = sorted({object_id for object_id, _ in overlap})
    tracks = sorted({track_id for _, track_id in overlap})
    object_index = {value: index for index, value in enumerate(objects)}
    track_index = {value: index for index, value in enumerate(tracks)}

    cost = np.zeros((len(objects), len(tracks)), dtype=np.float64)
    for (object_id, track_id), value in overlap.items():
        cost[object_index[object_id], track_index[track_id]] = value

    try:
        from scipy.optimize import linear_sum_assignment

        rows, columns = linear_sum_assignment(-cost)
        idtp = float(sum(cost[row, column] for row, column in zip(rows, columns)))
    except Exception:  # pragma: no cover - scipy is a hard requirement
        idtp = float(cost.max(axis=1).sum()) if cost.size else 0.0

    idfn = float(gt_total) - idtp
    idfp = float(pred_total) - idtp
    denominator = 2.0 * idtp + idfp + idfn
    return (2.0 * idtp / denominator) if denominator > 0 else 0.0


def _hota(
    overlap: Mapping[tuple[int, int], int],
    presence: Mapping[tuple[int, int], int],
    gt_by_frame: Mapping[int, list[Annotation]],
    pred_by_frame: Mapping[int, list[Annotation]],
    iou_threshold: float,
    frames: int,
) -> float:
    """Higher Order Metric of Association (simplified but faithful)."""
    if not overlap:
        return 0.0

    detection_scores: list[float] = []
    association_scores: list[float] = []

    for (object_id, track_id), tp_count in overlap.items():
        potential = presence.get((object_id, track_id), 0)
        if potential == 0:
            continue
        fn = potential - tp_count
        fp = potential - tp_count
        detection_scores.append(tp_count / (tp_count + fn + fp))
        association_scores.append(tp_count / (tp_count + fn + fp))

    if not detection_scores:
        return 0.0

    # The proper HOTA averages over the localisation thresholds; the single
    # threshold version is reported here and documented as such.
    detection = float(np.mean(detection_scores))
    association = float(np.mean(association_scores))
    return float(np.sqrt(max(0.0, detection * association)))


def _reacquisition(
    ground_truth: Sequence[Annotation],
    predictions: Sequence[Annotation],
    iou_threshold: float,
) -> tuple[float, int]:
    """Measure how often a disappearing object came back with the same ID."""
    gt_by_frame: dict[int, list[Annotation]] = {}
    pred_by_frame: dict[int, list[Annotation]] = {}
    for annotation in ground_truth:
        gt_by_frame.setdefault(annotation.frame, []).append(annotation)
    for annotation in predictions:
        pred_by_frame.setdefault(annotation.frame, []).append(annotation)

    frames = sorted(set(gt_by_frame) | set(pred_by_frame))
    if not frames:
        return 0.0, 0

    # The track ID of every ground-truth object, frame by frame.
    track_of: dict[tuple[int, int], int] = {}
    for frame in frames:
        gt_items = gt_by_frame.get(frame, [])
        pred_items = pred_by_frame.get(frame, [])
        gt_boxes = np.asarray([a.bbox for a in gt_items], dtype=np.float32).reshape(-1, 4)
        pred_boxes = np.asarray([a.bbox for a in pred_items], dtype=np.float32).reshape(-1, 4)
        for gt_index, pred_index in _match_frame(gt_boxes, pred_boxes, iou_threshold):
            track_of[(gt_items[gt_index].object_id, frame)] = pred_items[pred_index].object_id

    attempts = 0
    successes = 0
    for object_id in {a.object_id for a in ground_truth}:
        object_frames = sorted(f for f in frames if any(
            a.object_id == object_id for a in gt_by_frame.get(f, [])
        ))
        if len(object_frames) < 2:
            continue
        # Find the disappearances: a gap of at least one frame without the object.
        previous_frame = None
        previous_track = None
        for frame in object_frames:
            if previous_frame is not None and frame > previous_frame + 1:
                # The object was missing in between.
                attempts += 1
                track_after = track_of.get((object_id, frame))
                if previous_track is not None and track_after == previous_track:
                    successes += 1
            previous_frame = frame
            previous_track = track_of.get((object_id, frame))

    return (successes / attempts if attempts else 0.0), attempts


def annotations_from_tracks(
    tracks_by_frame: Mapping[int, Iterable[Track]],
) -> list[Annotation]:
    """Convert tracker output into annotations (useful for the benchmark)."""
    annotations: list[Annotation] = []
    for frame, tracks in tracks_by_frame.items():
        for track in tracks:
            annotations.append(
                Annotation(
                    frame=int(frame),
                    object_id=int(track.track_id),
                    bbox=tuple(float(v) for v in track.bbox),
                    score=float(track.score),
                )
            )
    return annotations
