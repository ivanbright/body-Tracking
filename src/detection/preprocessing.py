"""Image pre-processing for YOLOX-style detectors.

Two things have to be right for a detector to produce boxes in the right
place:

1. the frame has to be resized to the network input resolution, and
2. the resize has to be invertible, so the decoded boxes can be mapped back to
   the original frame.

:class:`TransformInfo` stores that inverse mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import cv2
import numpy as np

#: Padding colour used by the YOLOX reference implementation.
PAD_COLOR = (114, 114, 114)

#: Anything that can be interpreted as an ``(x1, y1, x2, y2)`` box.
BoxLike = Union[Sequence[float], np.ndarray]


@dataclass(frozen=True)
class TransformInfo:
    """Inverse mapping from network-input space back to original-frame space.

    ``x_orig = (x_input - pad_x) / scale_x`` and the same for ``y``.
    """

    scale_x: float
    scale_y: float
    pad_x: float
    pad_y: float
    orig_width: int
    orig_height: int
    input_width: int
    input_height: int
    mode: str = "letterbox"

    def to_original(self, x: float, y: float) -> tuple[float, float]:
        return ((x - self.pad_x) / self.scale_x, (y - self.pad_y) / self.scale_y)

    def box_to_original(self, box: BoxLike) -> tuple[float, float, float, float]:
        """Map an ``(x1, y1, x2, y2)`` box from input space to original space."""
        x1, y1, x2, y2 = (float(v) for v in box)
        ox1, oy1 = self.to_original(x1, y1)
        ox2, oy2 = self.to_original(x2, y2)
        return (ox1, oy1, ox2, oy2)

    def point_to_original(self, x: float, y: float) -> tuple[float, float]:
        return self.to_original(x, y)


def letterbox(
    image: np.ndarray,
    input_size: tuple[int, int],
    *,
    color: tuple[int, int, int] = PAD_COLOR,
    scaleup: bool = True,
) -> tuple[np.ndarray, TransformInfo]:
    """Resize ``image`` into ``input_size`` keeping the aspect ratio.

    The remaining area is padded with ``color`` (YOLOX uses 114). The returned
    :class:`TransformInfo` can undo the operation.
    """
    if image is None or image.size == 0:
        raise ValueError("letterbox: received an empty image")
    input_w, input_h = int(input_size[0]), int(input_size[1])
    if input_w <= 0 or input_h <= 0:
        raise ValueError("letterbox: input size must be positive")

    height, width = image.shape[:2]
    ratio = min(input_w / width, input_h / height)
    if not scaleup:
        ratio = min(ratio, 1.0)

    new_w = max(1, int(round(width * ratio)))
    new_h = max(1, int(round(height * ratio)))

    interpolation = cv2.INTER_LINEAR if ratio > 1 else cv2.INTER_AREA
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    canvas = np.full((input_h, input_w, image.shape[2]), color, dtype=image.dtype)
    pad_x = (input_w - new_w) / 2.0
    pad_y = (input_h - new_h) / 2.0
    top = int(round(pad_y - 0.1))
    left = int(round(pad_x - 0.1))
    canvas[top : top + new_h, left : left + new_w] = resized

    # Effective offsets after rounding.
    used_pad_x = left
    used_pad_y = top

    info = TransformInfo(
        scale_x=new_w / float(width),
        scale_y=new_h / float(height),
        pad_x=used_pad_x,
        pad_y=used_pad_y,
        orig_width=width,
        orig_height=height,
        input_width=input_w,
        input_height=input_h,
        mode="letterbox",
    )
    return canvas, info


def stretch(image: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, TransformInfo]:
    """Resize ``image`` to ``input_size`` ignoring the aspect ratio."""
    if image is None or image.size == 0:
        raise ValueError("stretch: received an empty image")
    input_w, input_h = int(input_size[0]), int(input_size[1])
    height, width = image.shape[:2]
    resized = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    info = TransformInfo(
        scale_x=input_w / float(width),
        scale_y=input_h / float(height),
        pad_x=0.0,
        pad_y=0.0,
        orig_width=width,
        orig_height=height,
        input_width=input_w,
        input_height=input_h,
        mode="stretch",
    )
    return resized, info


def resize_frame(
    image: np.ndarray,
    input_size: tuple[int, int],
    mode: str = "letterbox",
) -> tuple[np.ndarray, TransformInfo]:
    """Resize ``image`` to ``input_size`` using ``mode``."""
    if mode == "letterbox":
        return letterbox(image, input_size)
    if mode == "stretch":
        return stretch(image, input_size)
    raise ValueError(f"resize_frame: unsupported mode {mode!r} (use 'letterbox' or 'stretch')")


def to_blob(
    image: np.ndarray,
    *,
    layout: str = "nchw",
    scale: bool = True,
    order: str = "bgr",
) -> np.ndarray:
    """Convert a BGR uint8 image into a contiguous model input tensor.

    Args:
        image: HWC BGR uint8 image.
        layout: ``"nchw"`` (1, 3, H, W) or ``"nhwc"`` (1, H, W, 3).
        scale: divide by 255 to obtain the [0, 1] range.
        order: ``"bgr"`` keeps the channel order of the source image, ``"rgb"``
            swaps it. The YOLOX reference implementation feeds **RGB** pixels
            (``yolox/models/transforms.py``: ``img.transpose((2, 0, 1))[::-1]``).
    """
    if image.ndim != 3:
        raise ValueError(f"to_blob: expected a 3 channel image, got shape {image.shape}")
    data = image
    if scale:
        data = image.astype(np.float32) / 255.0
    else:
        data = image.astype(np.float32)

    if order == "rgb":
        data = data[:, :, ::-1]
    elif order != "bgr":
        raise ValueError(f"to_blob: unsupported colour order {order!r} (use 'bgr' or 'rgb')")

    if layout == "nchw":
        blob = np.transpose(data, (2, 0, 1))[None, ...]
    elif layout == "nhwc":
        blob = data[None, ...]
    else:
        raise ValueError(f"to_blob: unsupported layout {layout!r} (use 'nchw' or 'nhwc')")
    return np.ascontiguousarray(blob, dtype=np.float32)


def infer_layout(shape: Sequence[int]) -> str:
    """Infer the input layout from a model input shape.

    ``(1, 3, H, W)`` -> ``"nchw"``, ``(1, H, W, 3)`` -> ``"nhwc"``.
    """
    dims = [int(d) for d in shape]
    if len(dims) != 4:
        raise ValueError(f"infer_layout: expected a 4D input shape, got {tuple(dims)}")
    if dims[1] == 3:
        return "nchw"
    if dims[3] == 3:
        return "nhwc"
    raise ValueError(
        f"infer_layout: cannot determine the layout of the input shape {tuple(dims)}; "
        "set detector.input_layout explicitly"
    )
