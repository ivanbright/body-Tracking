"""The decoder: the export layouts and the shape validation."""

from __future__ import annotations

import numpy as np
import pytest

from src.detection.postprocessing import (
    DecodeError,
    DecodeSpec,
    anchor_grid,
    decode_yolox_output,
    non_maximum_suppression,
)
from src.detection.preprocessing import TransformInfo


def identity_transform(width: int = 640, height: int = 640) -> TransformInfo:
    return TransformInfo(
        scale_x=1.0,
        scale_y=1.0,
        pad_x=0.0,
        pad_y=0.0,
        orig_width=width,
        orig_height=height,
        input_width=width,
        input_height=height,
        mode="stretch",
    )


def test_anchor_grid_covers_every_level() -> None:
    col, row, stride = anchor_grid((640, 640), (8, 16, 32))
    assert col.shape[0] == 80 * 80 + 40 * 40 + 20 * 20
    # The first level is stride 8 and starts at the top-left cell.
    assert stride[0] == 8.0
    assert col[0] == 0.0 and row[0] == 0.0
    assert col[1] == 1.0 and row[1] == 0.0


def test_decoded_layout_produces_pixel_boxes() -> None:
    spec = DecodeSpec(output_format="yolox_decoded", num_classes=1, score_threshold=0.1)
    output = np.zeros((1, 1, 6), dtype=np.float32)
    output[0, 0] = [100.0, 200.0, 40.0, 60.0, 0.9, 0.8]

    detections = decode_yolox_output(output, identity_transform(), spec)

    assert len(detections) == 1
    x1, y1, x2, y2 = detections[0].bbox
    assert x1 == pytest.approx(80.0)
    assert y1 == pytest.approx(170.0)
    assert x2 == pytest.approx(120.0)
    assert y2 == pytest.approx(230.0)
    assert detections[0].score == pytest.approx(0.72)


def test_raw_layout_matches_the_upstream_decoding() -> None:
    """``(cx_off + grid) * stride`` and ``exp(w) * stride``, like Megvii does."""
    spec = DecodeSpec(
        output_format="yolox_raw",
        num_classes=1,
        strides=(8,),
        input_size=(64, 64),
        score_threshold=0.1,
    )
    output = np.zeros((1, 64, 6), dtype=np.float32)
    output[0, 0] = [1.0, 2.0, np.log(4.0), np.log(2.0), 0.9, 0.9]

    detections = decode_yolox_output(output, identity_transform(64, 64), spec)

    assert len(detections) == 1
    x1, y1, x2, y2 = detections[0].bbox
    # anchor (0, 0), stride 8 -> centre (8, 16), size 32 x 16
    assert (x1 + x2) / 2 == pytest.approx(8.0)
    assert (y1 + y2) / 2 == pytest.approx(16.0)
    assert (x2 - x1) == pytest.approx(32.0)
    assert (y2 - y1) == pytest.approx(16.0)


def test_raw_layout_rejects_a_wrong_anchor_count() -> None:
    spec = DecodeSpec(
        output_format="yolox_raw",
        num_classes=1,
        strides=(8, 16, 32),
        input_size=(64, 64),
    )
    output = np.zeros((1, 10, 6), dtype=np.float32)
    with pytest.raises(DecodeError) as error:
        decode_yolox_output(output, identity_transform(64, 64), spec)
    assert "anchors" in str(error.value)


def test_wrong_column_count_is_reported_clearly() -> None:
    spec = DecodeSpec(output_format="yolox_raw", num_classes=1)
    output = np.zeros((1, 10, 7), dtype=np.float32)
    with pytest.raises(DecodeError) as error:
        decode_yolox_output(output, identity_transform(), spec)
    message = str(error.value)
    assert "7 columns" in message
    assert "expects 6" in message


def test_xyxy_layout() -> None:
    spec = DecodeSpec(output_format="xyxy", score_threshold=0.2)
    output = np.zeros((1, 2, 6), dtype=np.float32)
    output[0, 0] = [10, 20, 50, 80, 0.9, 0]
    output[0, 1] = [10, 20, 50, 80, 0.1, 0]

    detections = decode_yolox_output(output, identity_transform(), spec)

    assert len(detections) == 1
    assert detections[0].bbox == (10.0, 20.0, 50.0, 80.0)


def test_dfl_layout_decodes_a_distribution() -> None:
    spec = DecodeSpec(
        output_format="yolox_dfl",
        num_classes=1,
        reg_max=4,
        strides=(8,),
        input_size=(64, 64),
        score_threshold=0.1,
    )
    output = np.zeros((1, 64, 4 * 4 + 1 + 1), dtype=np.float32)
    # A uniform distribution over the 4 bins has an expectation of 1.5.
    output[0, 0, :16] = 0.0
    output[0, 0, 16] = 0.9
    output[0, 0, 17] = 0.9

    detections = decode_yolox_output(output, identity_transform(64, 64), spec)

    assert len(detections) == 1
    x1, y1, x2, y2 = detections[0].bbox
    # anchor centre (4, 4) with stride 8, distances 1.5 * 8 = 12
    assert (x1 + x2) / 2 == pytest.approx(4.0)
    assert (x2 - x1) == pytest.approx(24.0)


def test_scores_below_the_threshold_are_dropped() -> None:
    spec = DecodeSpec(output_format="yolox_decoded", num_classes=1, score_threshold=0.5)
    output = np.zeros((1, 1, 6), dtype=np.float32)
    output[0, 0] = [100.0, 200.0, 40.0, 60.0, 0.9, 0.1]
    assert decode_yolox_output(output, identity_transform(), spec) == []


def test_nms_removes_duplicates() -> None:
    boxes = np.asarray(
        [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32
    )
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    classes = np.asarray([0, 0, 0], dtype=np.int64)
    keep = non_maximum_suppression(boxes, scores, classes, 0.5)
    assert keep == [0, 2]
