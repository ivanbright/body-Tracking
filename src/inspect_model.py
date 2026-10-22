"""Inspect an ONNX model.

The decoder of this project is never allowed to guess. This tool tells the
operator what a model *actually* is:

* the input tensor: name, shape, dtype and the resulting layout,
* the output tensors: name, shape, dtype,
* the value range of the output, so that the presence or absence of a sigmoid
  can be seen,
* the output layout that matches the real column count.

Usage::

    python -m src.inspect_model models/yolox_face.onnx
    python -m src.inspect_model models/yolox_face.onnx --image sample.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .detection.postprocessing import DecodeSpec, implied_num_classes


def _describe(spec: DecodeSpec, columns: int) -> str:
    """Return the layouts whose column count matches the model.

    For every layout the number of classes it implies is shown, because the
    class count follows from the real width of the output tensor.
    """
    matches = []
    for candidate in ("yolox_raw", "yolox_decoded"):
        classes = implied_num_classes(candidate, columns, spec.reg_max)
        if classes is not None:
            matches.append(f"{candidate} ({classes} classes)")
    classes = implied_num_classes("yolox_dfl", columns, spec.reg_max)
    if classes is not None:
        matches.append(f"yolox_dfl ({classes} classes)")
    if columns == 6:
        matches.append("xyxy (class id in the last column)")
    return ", ".join(matches) if matches else "none"


def inspect(model_path: str | Path, *, image_path: str | Path | None = None) -> int:
    """Print everything that is known about a model. Returns an exit code."""
    path = Path(model_path)
    if not path.exists():
        print(f"Model not found: {path}", file=sys.stderr)
        return 2

    try:
        import onnxruntime as ort
    except ImportError:
        print(
            "ONNX Runtime is not installed. Install it with: pip install onnxruntime",
            file=sys.stderr,
        )
        return 2

    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        print(f"Cannot load {path}: {exc}", file=sys.stderr)
        return 2

    print(f"model      : {path}")
    print(f"providers  : {', '.join(session.get_providers())}")
    for tensor in session.get_inputs():
        print(
            f"input      : {tensor.name} shape={tensor.shape} type={tensor.type}"
        )
    for tensor in session.get_outputs():
        print(
            f"output     : {tensor.name} shape={tensor.shape} type={tensor.type}"
        )

    # --- run a dummy (or a real) image through the model --------------- #
    try:
        frame = None
        if image_path:
            import cv2

            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"Could not read the image {image_path}", file=sys.stderr)
                return 2

        input_tensor = session.get_inputs()[0]
        shape = list(input_tensor.shape)
        height = shape[2] if len(shape) == 4 and isinstance(shape[2], int) else 640
        width = shape[3] if len(shape) == 4 and isinstance(shape[3], int) else 640
        if frame is None:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (width, height))

        conventions = {
            "0_255 rgb": (np.ascontiguousarray(frame[:, :, ::-1].astype(np.float32)), False),
            "0_255 bgr": (np.ascontiguousarray(frame.astype(np.float32)), False),
            "0_1   rgb": (np.ascontiguousarray(frame[:, :, ::-1].astype(np.float32) / 255.0), True),
            "0_1   bgr": (np.ascontiguousarray(frame.astype(np.float32) / 255.0), True),
        }

        best_label = ""
        best_score = -1.0
        for label, (data, scaled) in conventions.items():
            if len(shape) == 4 and isinstance(shape[1], int) and shape[1] == 3:
                blob = np.transpose(data, (2, 0, 1))[None, ...]
            else:
                blob = data[None, ...]
            out = np.asarray(session.run(None, {input_tensor.name: blob})[0])
            score = 0.0
            if out.ndim == 3 and out.shape[2] >= 5:
                objects = out[0, :, 4]
                classes = out[0, :, 5:]
                score = float((objects * classes.max(axis=1)).max()) if classes.size else 0.0
                score = max(score, float(objects.max()))
            print(f"input {label} -> best score {score:.4f}")
            if score > best_score:
                best_score = score
                best_label = label.strip()

        print(f"\nThe model reacts best to the input convention: {best_label}")

        if len(shape) == 4 and isinstance(shape[1], int) and shape[1] == 3:
            data = frame.astype(np.float32)
            blob = np.transpose(data, (2, 0, 1))[None, ...]
        else:
            blob = frame.astype(np.float32)[None, ...]

        outputs = session.run(None, {input_tensor.name: blob})
        output = np.asarray(outputs[0])
        print(f"run output : shape={output.shape}")
        if output.size:
            print(
                f"value range: min={float(output.min()):.4f} max={float(output.max()):.4f}"
            )
            if output.ndim == 3:
                columns = output.shape[2]
                spec = DecodeSpec()
                print(f"columns    : {columns}")
                print(f"compatible : {_describe(spec, int(columns))}")
                if output.shape[1] and int(columns) >= 5:
                    objects = output[0, :, 4]
                    print(
                        f"column 4   : min={float(objects.min()):.4f} "
                        f"max={float(objects.max()):.4f} "
                        "(objectness; values > 1 mean the export emits raw logits)"
                    )
        print(
            "\nSet detector.output_format to one of the compatible layouts above."
        )
    except Exception as exc:
        print(f"The model could not be executed: {exc}", file=sys.stderr)
        return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.inspect_model",
        description="Inspect a YOLOX ONNX model and report its real input/output.",
    )
    parser.add_argument("model", help="path to the ONNX model")
    parser.add_argument(
        "--image",
        default="",
        help="optional image to run through the model (a black frame by default)",
    )
    args = parser.parse_args(argv)
    return inspect(args.model, image_path=args.image or None)


if __name__ == "__main__":
    raise SystemExit(main())
