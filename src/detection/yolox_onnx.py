"""YOLOX detector executed with ONNX Runtime.

This is the primary detector of the project. It expects a YOLOX model that was
exported to ONNX and that was **trained on faces**.

Nothing about the export is assumed silently:

* the input tensor is inspected to learn the layout (``NCHW`` / ``NHWC``),
* the input resolution is taken from the model when it is static and from the
  configuration otherwise,
* the output layout is **declared** in the configuration and then **verified**
  against the real output shape of the model - a mismatch is an error, never a
  silent misinterpretation,
* a missing file, an unloadable model or a malformed output produce a
  :class:`~src.detection.base.DetectorError` with an actionable message.

The detector answers one question only: *where are the faces in this frame?*
It does not know and does not care whether a face is the selected target.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .base import Detection, DetectorError, DetectorStats
from .postprocessing import (
    COCO_LABELS,
    DecodeError,
    DecodeSpec,
    apply_class_filter,
    decode_yolox_output,
    implied_num_classes,
    non_maximum_suppression,
)
from .preprocessing import infer_layout, resize_frame, to_blob

LOGGER = logging.getLogger(__name__)

#: Providers that are worth trying first when they are installed.
_PREFERRED_PROVIDERS = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "OpenVINOExecutionProvider",
    "DmlExecutionProvider",
    "AzureExecutionProvider",
    "CPUExecutionProvider",
)


def select_providers(requested: Sequence[str]) -> tuple[list[str], str | None]:
    """Return ``(providers_to_use, missing)``.

    ``providers_to_use`` always contains ``CPUExecutionProvider`` as the last
    entry so that the session can always be created.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise DetectorError(
            "ONNX Runtime is not installed. Install it with: pip install onnxruntime"
        ) from exc

    available = list(ort.get_available_providers())
    selected: list[str] = []
    missing: list[str] = []
    for provider in requested:
        if provider in available:
            selected.append(provider)
        else:
            missing.append(provider)
    if "CPUExecutionProvider" not in selected:
        selected.append("CPUExecutionProvider")
    return selected, (missing[0] if missing else None)


class YOLOXOnnxDetector:
    """YOLOX ONNX detector."""

    name = "yolox_onnx"

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_size: tuple[int, int] = (640, 640),
        score_threshold: float = 0.3,
        nms_threshold: float = 0.45,
        providers: Sequence[str] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
        input_layout: str = "auto",
        resize_mode: str = "letterbox",
        output_format: str = "yolox_raw",
        activation: str = "none",
        num_classes: int = 1,
        strides: Sequence[int] = (8, 16, 32),
        reg_max: int = 16,
        labels: Sequence[str] = (),
        target_classes: Sequence[str] = (),
        num_threads: int = 0,
        warmup: bool = True,
        input_range: str = "0_255",
        color_order: str = "rgb",
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.input_layout = input_layout
        self.resize_mode = resize_mode
        self.output_format = output_format
        self.activation = activation
        self.num_classes = int(num_classes)
        self.strides = tuple(int(s) for s in strides)
        self.reg_max = int(reg_max)
        self.labels = tuple(labels)
        self.target_classes = tuple(target_classes)
        self.stats = DetectorStats()

        if not str(self.model_path):
            raise DetectorError(
                "No ONNX model configured. Set detector.model_path to a YOLOX "
                "ONNX file (see yolox/models/README.md)."
            )
        if not self.model_path.exists():
            raise DetectorError(
                f"ONNX model not found: {self.model_path}\n"
                "Place a YOLOX face-detection ONNX file at that path, or point "
                "detector.model_path at an existing one. "
                "See yolox/models/README.md for the expected format."
            )
        if not self.model_path.is_file():
            raise DetectorError(f"ONNX model path is not a file: {self.model_path}")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise DetectorError(
                "ONNX Runtime is not installed. Install it with: pip install onnxruntime"
            ) from exc

        options = ort.SessionOptions()
        if num_threads and num_threads > 0:
            options.intra_op_num_threads = int(num_threads)
        options.log_severity_level = 3  # errors only

        chosen, missing = select_providers(providers)
        if missing:
            LOGGER.warning(
                "Execution provider %s is not installed, falling back to CPU. "
                "Available providers: %s",
                missing,
                ", ".join(ort.get_available_providers()),
            )

        try:
            self.session = ort.InferenceSession(
                str(self.model_path), sess_options=options, providers=chosen
            )
        except Exception as exc:
            raise DetectorError(
                f"Failed to load ONNX model {self.model_path}: {exc}\n"
                "The file may be corrupt, truncated, or not an ONNX model."
            ) from exc

        self.providers = self.session.get_providers()
        self.active_provider = self.providers[0] if self.providers else "unknown"
        LOGGER.info(
            "Loaded YOLOX ONNX model %s | providers: %s | active: %s",
            self.model_path.name,
            ", ".join(self.providers),
            self.active_provider,
        )

        self._input = self.session.get_inputs()[0]
        self._outputs = list(self.session.get_outputs())
        self._output_names = [out.name for out in self._outputs]

        layout = input_layout
        if layout == "auto":
            try:
                layout = infer_layout(self._input.shape)
            except ValueError as exc:
                raise DetectorError(
                    f"{exc}. Set detector.input_layout explicitly."
                ) from exc
        elif layout not in {"nchw", "nhwc"}:
            raise DetectorError(f"unsupported input layout {input_layout!r}")
        self.layout = layout

        model_hw = self._static_input_hw()
        if model_hw is not None:
            self._input_size = model_hw
            if model_hw != tuple(input_size):
                LOGGER.info(
                    "Using the model input size %sx%s (configuration said %sx%s)",
                    model_hw[0],
                    model_hw[1],
                    input_size[0],
                    input_size[1],
                )
        else:
            self._input_size = (int(input_size[0]), int(input_size[1]))

        self._input_dtype = self._input.type
        self._scale_input = input_range not in {"0_255", "raw"}
        self._color_order = color_order
        self._input_range = input_range

        self.spec = DecodeSpec(
            output_format=self.output_format,
            activation=self.activation,
            num_classes=self.num_classes,
            strides=self.strides,
            reg_max=self.reg_max,
            score_threshold=self.score_threshold,
            input_size=self._input_size,
            labels=self.labels,
        )
        self._verify_output_layout()

        if warmup:
            self.warmup()

    # ------------------------------------------------------------------ #
    def _static_input_hw(self) -> tuple[int, int] | None:
        shape = list(self._input.shape)
        if len(shape) != 4:
            return None
        if self.layout == "nchw":
            height, width = shape[2], shape[3]
        else:
            height, width = shape[1], shape[2]
        if not isinstance(height, int) or not isinstance(width, int):
            return None
        if height <= 0 or width <= 0:
            return None
        return (int(width), int(height))

    def _verify_output_layout(self) -> None:
        """Check the declared layout against the model's real output shape.

        This is the *verification* step: the decoder is never allowed to guess.

        The **layout** comes from the configuration and is never changed here.
        The **number of classes**, however, is a property of the model: once the
        layout is fixed, the real width of the output tensor determines how
        many classes the model has. A configured value that does not match is
        therefore corrected (with a warning) instead of aborting the run.
        """
        if not self._outputs:
            raise DetectorError("the ONNX model does not expose an output tensor")
        shape = list(self._outputs[0].shape)
        if len(shape) != 3:
            LOGGER.warning(
                "Unexpected output shape %s; the decoder expects (1, N, K).",
                tuple(shape),
            )
            return
        columns = shape[2]
        if not isinstance(columns, int):
            LOGGER.info(
                "The output has a dynamic number of columns (%s); the layout "
                "check is skipped.", columns
            )
            return
        expected = self.spec.expected_columns()
        if int(columns) == expected:
            LOGGER.info(
                "Output layout verified: %s with %d columns matches the "
                "configured output_format %r.",
                tuple(shape),
                int(columns),
                self.output_format,
            )
            return

        implied = implied_num_classes(self.output_format, int(columns), self.reg_max)
        if implied is not None and implied != self.num_classes:
            LOGGER.warning(
                "The model output has %d columns, which means %d classes for "
                "the declared output_format %r, but detector.num_classes was "
                "%d. Using %d classes - the model decides how many classes it "
                "has.",
                int(columns),
                implied,
                self.output_format,
                self.num_classes,
                implied,
            )
            self.num_classes = implied
            self.spec = replace(self.spec, num_classes=implied)
            self._apply_default_labels(implied)
            return

        alternatives = []
        for candidate in ("yolox_raw", "yolox_decoded"):
            if 5 + self.num_classes == int(columns):
                alternatives.append(candidate)
        if 4 * self.reg_max + 1 + self.num_classes == int(columns):
            alternatives.append("yolox_dfl")
        if int(columns) == 6:
            alternatives.append("xyxy")

        hint = (
            "The compatible layouts are: " + ", ".join(alternatives) + "."
            if alternatives
            else "No known layout matches that column count."
        )
        raise DetectorError(
            f"The model output has {int(columns)} columns but the configured "
            f"detector.output_format {self.output_format!r} with "
            f"detector.num_classes={self.num_classes} expects {expected} "
            f"columns. {hint} Fix detector.output_format and/or "
            "detector.num_classes (or use `python -m src.inspect_model "
            f"{self.model_path}` to inspect the model)."
        )

    def _apply_default_labels(self, num_classes: int) -> None:
        """Use the COCO class names when a 80 class model has no label file."""
        if self.labels or num_classes != len(COCO_LABELS):
            return
        self.labels = COCO_LABELS
        self.spec = replace(self.spec, labels=self.labels)
        LOGGER.warning(
            "No label file is configured and the model has %d classes: the "
            "standard COCO class names are used for the overlay.",
            num_classes,
        )

    # ------------------------------------------------------------------ #
    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size

    def describe(self) -> dict[str, Any]:
        """Machine readable description of the loaded model."""
        return {
            "model_path": str(self.model_path),
            "backend": self.name,
            "input_name": self._input.name,
            "input_shape": list(self._input.shape),
            "input_type": self._input.type,
            "layout": self.layout,
            "input_size": list(self._input_size),
            "outputs": [
                {"name": out.name, "shape": list(out.shape), "type": out.type}
                for out in self._outputs
            ],
            "output_format": self.output_format,
            "activation": self.activation,
            "num_classes": self.num_classes,
            "strides": list(self.strides),
            "reg_max": self.reg_max,
            "expected_columns": self.spec.expected_columns(),
            "providers": list(self.providers),
            "active_provider": self.active_provider,
        }

    # ------------------------------------------------------------------ #
    def warmup(self, frame_shape: tuple[int, int] | None = None) -> None:
        """Run one dummy inference so the first real frame is not penalised."""
        height, width = frame_shape if frame_shape else (self._input_size[1], self._input_size[0])
        dummy = np.zeros((max(1, int(height)), max(1, int(width)), 3), dtype=np.uint8)
        try:
            self._infer(dummy)
        except DetectorError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Warm-up inference failed: %s", exc)

    def _infer(self, frame: np.ndarray) -> tuple[np.ndarray, Any]:
        resized, transform = resize_frame(frame, self._input_size, self.resize_mode)
        blob = to_blob(
            resized,
            layout=self.layout,
            scale=self._scale_input,
            order=self._color_order,
        )
        try:
            outputs = self.session.run(self._output_names, {self._input.name: blob})
        except Exception as exc:
            raise DetectorError(f"ONNX inference failed: {exc}") from exc
        if not outputs:
            raise DetectorError("ONNX model returned no output tensors")
        return np.asarray(outputs[0]), transform

    # ------------------------------------------------------------------ #
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            raise DetectorError("detect() received an empty frame")

        start = time.perf_counter()
        output, transform = self._infer(frame_bgr)
        try:
            detections = decode_yolox_output(output, transform, self.spec)
        except DecodeError as exc:
            raise DetectorError(str(exc)) from exc
        detections = self._suppress(detections)
        detections = apply_class_filter(detections, self.target_classes)
        detections.sort(key=lambda d: -d.score)

        latency_ms = (time.perf_counter() - start) * 1000.0
        self.stats.record(latency_ms, len(detections))
        return detections

    def _suppress(self, detections: list[Detection]) -> list[Detection]:
        if not detections:
            return []
        boxes = np.asarray([d.bbox for d in detections], dtype=np.float32)
        scores = np.asarray([d.score for d in detections], dtype=np.float32)
        class_ids = np.asarray([d.class_id for d in detections], dtype=np.int64)
        keep = non_maximum_suppression(boxes, scores, class_ids, self.nms_threshold)
        return [detections[i] for i in keep]

    def close(self) -> None:
        session = getattr(self, "session", None)
        if session is not None:
            try:
                del self.session
            except Exception:  # pragma: no cover - defensive
                pass
            self.session = None  # type: ignore[assignment]

    def __enter__(self) -> "YOLOXOnnxDetector":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
