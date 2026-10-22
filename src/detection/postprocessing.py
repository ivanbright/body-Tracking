"""Decoding of YOLOX model outputs and non-maximum suppression.

Verified export formats
-----------------------
The decoders in this module are **not** guessed from the data. They mirror the
actual Megvii implementation, which was read from the upstream sources:

``yolox/models/yolo_head.py`` (``YOLOXHead.forward``, evaluation branch)::

    output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)

``yolox/models/yolo_head.py`` (``YOLOXHead.decode_outputs``)::

    (outputs[..., 0:2] + grids) * strides
    torch.exp(outputs[..., 2:4]) * strides

``tools/export_onnx.py``::

    model.head.decode_in_inference = args.decode_in_inference

Three consequences follow from those three facts:

1. The exported tensor is ``(1, N, K)`` with the columns
   ``[box(4), obj(1), cls(C)]``.
2. The objectness and the class scores are **already probabilities** - the
   sigmoid happens inside the head. They must not be squashed a second time.
3. ``export_onnx.py`` uses ``action="store_true"`` for ``--decode_in_inference``,
   so the flag is **off by default**: the official export returns the *raw*
   head output and the box has to be decoded with the grid and the strides.

Supported layouts
-----------------
``yolox_raw`` (default, matches the official export script)
    ``(1, N, 5 + C)`` - ``[cx_off, cy_off, log_w, log_h, obj, cls...]`` where
    the box is expressed in stride units and is decoded exactly like
    ``YOLOXHead.decode_outputs`` does.
``yolox_decoded``
    ``(1, N, 5 + C)`` - ``[cx, cy, w, h, obj, cls...]`` already in
    network-input pixels. This is what ``--decode_in_inference`` produces.
``yolox_dfl``
    ``(1, N, 4 * reg_max + 1 + C)`` - distribution-focal-regression logits of
    the older YOLOX releases. The decoding follows the documented Megvii
    convention (softmax over the ``reg_max`` bins, expectation, ``dist2bbox``).
``xyxy``
    ``(1, N, 6)`` - ``[x1, y1, x2, y2, score, class_id]`` for custom exports.

Every layout is validated against the **real** output shape of the model: a
mismatch produces an actionable error instead of silently wrong boxes.

Everything is returned in **original frame pixels** so that the tracker never
has to know about the network input geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .base import Detection
from .preprocessing import TransformInfo

#: Supported output layouts of the YOLOX exports.
OUTPUT_FORMATS = ("yolox_raw", "yolox_decoded", "yolox_dfl", "xyxy")

#: ``none`` keeps the values as they are (the head already applied the
#: sigmoid), ``sigmoid`` squashes raw logits.
ACTIVATIONS = ("none", "sigmoid")

#: Default strides of the YOLOX detection head.
DEFAULT_STRIDES = (8, 16, 32)

#: Class names of the standard 80 class COCO model.
#:
#: They are only used when a model has 80 classes and no label file was
#: configured. A single class face model never reaches this list, so the
#: assumption cannot silently rename the faces of a face model.
COCO_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


class DecodeError(ValueError):
    """Raised when the model output does not match the configured layout."""


@dataclass(frozen=True)
class DecodeSpec:
    """Everything the decoder needs to know about the model."""

    output_format: str = "yolox_raw"
    activation: str = "none"
    num_classes: int = 1
    strides: tuple[int, ...] = DEFAULT_STRIDES
    reg_max: int = 16
    score_threshold: float = 0.3
    input_size: tuple[int, int] = (640, 640)
    labels: tuple[str, ...] = ()

    # ------------------------------------------------------------------ #
    def expected_columns(self) -> int:
        """Number of columns the configured layout implies."""
        if self.output_format in {"yolox_raw", "yolox_decoded"}:
            return 5 + self.num_classes
        if self.output_format == "yolox_dfl":
            return 4 * self.reg_max + 1 + self.num_classes
        if self.output_format == "xyxy":
            return 6
        raise DecodeError(
            f"unsupported output_format {self.output_format!r}; "
            f"expected one of {', '.join(OUTPUT_FORMATS)}"
        )

    def expected_anchors(self) -> int:
        """Number of anchors the configured geometry implies."""
        if self.output_format != "yolox_raw":
            return 0
        input_w, input_h = self.input_size
        total = 0
        for stride in self.strides:
            total += (input_h // stride) * (input_w // stride)
        return total

    def objectness_column(self) -> int:
        """Index of the objectness column.

        It is column 4 for the plain YOLOX layouts and column ``4 * reg_max``
        for the distribution-focal-regression layout, where the first
        ``4 * reg_max`` columns hold the regression distribution.
        """
        if self.output_format == "yolox_dfl":
            return 4 * int(self.reg_max)
        return 4

    def class_columns(self) -> range:
        """Range of the class-score columns."""
        start = self.objectness_column() + 1
        return range(start, start + int(self.num_classes))

    def class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.labels):
            return self.labels[class_id]
        return "face" if self.num_classes == 1 else f"class_{class_id}"


def implied_num_classes(
    output_format: str, columns: int, reg_max: int = 16
) -> int | None:
    """Number of classes a layout of the given width would contain.

    The **layout** is never guessed: the operator declares it. But once the
    layout is declared, the number of classes follows from the real width of
    the output tensor. ``yolox_raw`` and ``yolox_decoded`` are ``5 + C``
    columns wide, ``yolox_dfl`` is ``4 * reg_max + 1 + C``.

    Returns ``None`` when the width cannot belong to the layout at all.
    """
    columns = int(columns)
    if output_format in {"yolox_raw", "yolox_decoded"}:
        count = columns - 5
    elif output_format == "yolox_dfl":
        count = columns - (4 * int(reg_max) + 1)
    else:
        return None
    return count if count >= 1 else None


# --------------------------------------------------------------------------- #
# low level helpers
# --------------------------------------------------------------------------- #
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of ``[x1, y1, x2, y2]`` boxes."""
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)
    a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)

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


def non_maximum_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """Class-aware NMS. Returns the indices of the boxes to keep."""
    if boxes.size == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)

    keep: list[int] = []
    for class_id in np.unique(class_ids):
        idx = np.flatnonzero(class_ids == class_id)
        order = idx[np.argsort(scores[idx])[::-1]]
        while order.size:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break
            rest = order[1:]
            ious = iou_matrix(boxes[current : current + 1], boxes[rest])[0]
            order = rest[ious <= iou_threshold]
    keep.sort(key=lambda i: -float(scores[i]))
    return keep


def anchor_grid(
    input_size: tuple[int, int],
    strides: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the grid offsets of every anchor.

    The exported tensor lists the anchors level by level (stride 8 first, then
    16, then 32) and row-major inside a level, exactly like
    ``torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2)`` does in
    ``YOLOXHead.forward``.

    Returns:
        ``(col, row, stride_per_anchor)`` arrays, one entry per anchor.
    """
    input_w, input_h = int(input_size[0]), int(input_size[1])
    cols: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    per_anchor_stride: list[np.ndarray] = []

    for stride in strides:
        feature_h = input_h // int(stride)
        feature_w = input_w // int(stride)
        if feature_h <= 0 or feature_w <= 0:
            continue
        count = feature_h * feature_w
        row_index, col_index = np.divmod(np.arange(count), feature_w)
        cols.append(col_index.astype(np.float32))
        rows.append(row_index.astype(np.float32))
        per_anchor_stride.append(np.full(count, float(stride), dtype=np.float32))

    if not cols:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    return (
        np.concatenate(cols),
        np.concatenate(rows),
        np.concatenate(per_anchor_stride),
    )


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
def decode_yolox_output(
    output: np.ndarray,
    transform: TransformInfo,
    spec: DecodeSpec,
) -> list[Detection]:
    """Decode a raw YOLOX output tensor into :class:`Detection` objects.

    Args:
        output: The tensor returned by the model, shape ``(1, N, K)`` or ``(N, K)``.
        transform: Inverse mapping produced by the pre-processing step.
        spec: The layout of the model.

    Returns:
        Detections in original-frame pixels, sorted by descending score.

    Raises:
        DecodeError: The output does not match the configured layout.
    """
    data = np.asarray(output, dtype=np.float32)
    if data.ndim == 3:
        if data.shape[0] != 1:
            raise DecodeError(
                f"unexpected output batch size {data.shape[0]}; only batch size 1 is supported"
            )
        data = data[0]
    if data.ndim != 2:
        raise DecodeError(f"unexpected output shape {tuple(np.asarray(output).shape)}")
    if data.shape[0] == 0:
        return []

    expected = spec.expected_columns()
    if data.shape[1] != expected:
        raise DecodeError(
            f"the model produced {data.shape[1]} columns but the configured "
            f"output_format {spec.output_format!r} with num_classes="
            f"{spec.num_classes} expects {expected}. "
            "Check detector.output_format / detector.num_classes against the "
            "model (run `python -m src.inspect_model <model.onnx>`)."
        )

    if spec.output_format == "xyxy":
        return _decode_xyxy(data, transform, spec)
    if spec.output_format == "yolox_decoded":
        return _decode_decoded(data, transform, spec)
    if spec.output_format == "yolox_dfl":
        return _decode_dfl(data, transform, spec)
    return _decode_raw(data, transform, spec)


def _apply_activation(values: np.ndarray, spec: DecodeSpec) -> np.ndarray:
    if spec.activation == "sigmoid":
        return sigmoid(values)
    if spec.activation == "none":
        return values
    raise DecodeError(f"unsupported activation {spec.activation!r}")


def _scores_and_classes(
    rows: np.ndarray, spec: DecodeSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(scores, class_ids, mask)`` for a YOLOX style row block."""
    objectness = _apply_activation(rows[:, spec.objectness_column()], spec)
    class_scores = _apply_activation(rows[:, spec.class_columns()], spec)
    if class_scores.shape[1] == 0:
        raise DecodeError("the model has no class scores")
    class_ids = class_scores.argmax(axis=1)
    scores = objectness * class_scores.max(axis=1)
    return scores.astype(np.float32), class_ids.astype(np.int64), scores >= spec.score_threshold


def _to_detections(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    transform: TransformInfo,
    spec: DecodeSpec,
) -> list[Detection]:
    detections: list[Detection] = []
    for box, score, class_id in zip(boxes_xyxy, scores, class_ids):
        x1, y1, x2, y2 = transform.box_to_original(box)
        detections.append(
            Detection(
                bbox=(x1, y1, x2, y2),
                score=float(score),
                class_id=int(class_id),
                class_name=spec.class_name(int(class_id)),
            )
        )
    return detections


def _decode_raw(
    rows: np.ndarray,
    transform: TransformInfo,
    spec: DecodeSpec,
) -> list[Detection]:
    """Decode the raw head output (the official export default).

    Mirrors ``YOLOXHead.decode_outputs``::

        cx = (cx_off + grid_x) * stride
        cy = (cy_off + grid_y) * stride
        w  = exp(log_w) * stride
        h  = exp(log_h) * stride
    """
    col, row, stride_per_anchor = anchor_grid(spec.input_size, spec.strides)
    if col.shape[0] != rows.shape[0]:
        expected = spec.expected_anchors()
        raise DecodeError(
            f"the model produced {rows.shape[0]} anchors but a "
            f"{spec.input_size[0]}x{spec.input_size[1]} input with strides "
            f"{list(spec.strides)} implies {expected}. "
            "The model was probably exported for a different input size; "
            "adjust detector.input_size / detector.strides."
        )

    scores, class_ids, keep = _scores_and_classes(rows, spec)
    if not np.any(keep):
        return []

    selected = rows[keep]
    selected_scores = scores[keep]
    selected_classes = class_ids[keep]
    col = col[keep]
    row = row[keep]
    stride_per_anchor = stride_per_anchor[keep]

    cx = (selected[:, 0] + col) * stride_per_anchor
    cy = (selected[:, 1] + row) * stride_per_anchor
    width = np.exp(selected[:, 2]) * stride_per_anchor
    height = np.exp(selected[:, 3]) * stride_per_anchor

    boxes = np.stack(
        [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0], axis=1
    )
    return _to_detections(boxes, selected_scores, selected_classes, transform, spec)


def _decode_decoded(
    rows: np.ndarray,
    transform: TransformInfo,
    spec: DecodeSpec,
) -> list[Detection]:
    """Decode an output that already contains ``[cx, cy, w, h]`` in input pixels."""
    scores, class_ids, keep = _scores_and_classes(rows, spec)
    if not np.any(keep):
        return []

    selected = rows[keep]
    cx, cy = selected[:, 0], selected[:, 1]
    width, height = selected[:, 2], selected[:, 3]
    boxes = np.stack(
        [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0], axis=1
    )
    return _to_detections(boxes, scores[keep], class_ids[keep], transform, spec)


def _decode_dfl(
    rows: np.ndarray,
    transform: TransformInfo,
    spec: DecodeSpec,
) -> list[Detection]:
    """Decode a distribution-focal-regression output.

    The regression part of the older YOLOX releases holds ``4 * reg_max``
    logits. They are laid out as ``[bin_group][side]`` with the sides in the
    order ``l, t, r, b``. The distribution is turned into an expectation and
    the resulting ``ltrb`` distances are turned into a box around the anchor
    point of the cell.

    .. note::
       This layout could **not** be verified against real weights: the
       upstream ``main`` branch does not use ``reg_max`` any more and the
       release that does could not be fetched. The implementation follows the
       documented distribution-focal convention (softmax over the bins,
       expectation, anchor +/- distance). If a model turns out to use a
       different convention, the boxes will be wrong - and ``inspect_model``
       plus the visual check will show it immediately.
    """
    reg_max = int(spec.reg_max)
    col, row, stride_per_anchor = anchor_grid(spec.input_size, spec.strides)
    if col.shape[0] != rows.shape[0]:
        raise DecodeError(
            f"the model produced {rows.shape[0]} anchors but a "
            f"{spec.input_size[0]}x{spec.input_size[1]} input with strides "
            f"{list(spec.strides)} implies {spec.expected_anchors()}."
        )

    reg = rows[:, : 4 * reg_max].reshape(-1, reg_max, 4)
    reg = reg - reg.max(axis=1, keepdims=True)
    probabilities = np.exp(reg)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    # The expectation of the distribution: sum over the *bins* of the bin
    # index weighted by its probability. The bin axis is the second one.
    expectation = (probabilities * np.arange(reg_max, dtype=np.float32)[None, :, None]).sum(axis=1)

    scores, class_ids, keep = _scores_and_classes(rows, spec)
    if not np.any(keep):
        return []

    expectation = expectation[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    col = col[keep]
    row = row[keep]
    stride_per_anchor = stride_per_anchor[keep]

    left, top, right, bottom = expectation[:, 0], expectation[:, 1], expectation[:, 2], expectation[:, 3]
    anchor_x = (col + 0.5) * stride_per_anchor
    anchor_y = (row + 0.5) * stride_per_anchor

    x1 = anchor_x - left * stride_per_anchor
    y1 = anchor_y - top * stride_per_anchor
    x2 = anchor_x + right * stride_per_anchor
    y2 = anchor_y + bottom * stride_per_anchor
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    return _to_detections(boxes, scores, class_ids, transform, spec)


def _decode_xyxy(
    rows: np.ndarray,
    transform: TransformInfo,
    spec: DecodeSpec,
) -> list[Detection]:
    """Decode a ``[x1, y1, x2, y2, score, class_id]`` output."""
    scores = rows[:, 4]
    class_ids = rows[:, 5].round().astype(np.int64)
    keep = scores >= spec.score_threshold
    if not np.any(keep):
        return []
    boxes = rows[keep, :4]
    return _to_detections(boxes, scores[keep], class_ids[keep], transform, spec)


def apply_class_filter(
    detections: Sequence[Detection],
    class_names: Sequence[str] | None,
) -> list[Detection]:
    """Keep only the detections whose class name is in ``class_names``.

    An empty or ``None`` filter keeps everything.
    """
    if not class_names:
        return list(detections)
    wanted = {name.strip().lower() for name in class_names if name.strip()}
    if not wanted:
        return list(detections)
    return [d for d in detections if d.class_name.strip().lower() in wanted]
