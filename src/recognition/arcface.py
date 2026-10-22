"""ArcFace face embeddings (ONNX Runtime).

The tracker can follow a face, but it cannot *recognise* it: a box does not say
**who** it belongs to. This module provides the missing half - a face
embedding, i.e. a vector that describes the identity of a person.

The embedder is the InsightFace / ArcFace ``w600k_r50`` graph:

* input: an **aligned** 112x112 face crop (see :mod:`src.recognition.alignment`),
* output: a 512-D vector, L2-normalized.

The pre-processing is important and was verified empirically: the graph does
**not** normalize the pixels internally, so feeding raw 0..255 values saturates
the network and every face collapses onto the same vector. The correct input is
RGB, normalized with ``(x - 127.5) / 128``.

The implementation follows the working reference pipeline of the parent
project (``src/embed.py``) so that both systems produce the same vectors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort

LOGGER = logging.getLogger(__name__)


class IdentityError(RuntimeError):
    """Raised when the identity layer cannot be used."""


@dataclass
class EmbeddingResult:
    """One face embedding."""

    #: L2-normalized embedding vector.
    embedding: np.ndarray
    #: Length of the raw vector before normalization (a quality indicator).
    norm_before: float
    #: Dimension of the embedding.
    dim: int


def _available_providers(requested: Sequence[str]) -> list[str]:
    """Keep only the providers that ONNX Runtime actually offers."""
    available = set(ort.get_available_providers())
    providers = [p for p in requested if p in available]
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")
    return providers


class ArcFaceEncoder:
    """Turns an aligned face crop into an identity embedding."""

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (112, 112),
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise IdentityError(
                f"ArcFace embedder not found at: {path}\n"
                "Place an ArcFace/InsightFace ONNX model there or point the "
                "configuration at it (identity.embedder_path)."
            )

        self.model_path = path
        self.in_w, self.in_h = int(input_size[0]), int(input_size[1])

        try:
            self.session = ort.InferenceSession(
                str(path), providers=_available_providers(providers)
            )
        except Exception as exc:  # pragma: no cover - depends on the runtime
            raise IdentityError(f"Could not load the ArcFace model {path}: {exc}") from exc

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape

        # Some exports use NHWC, the reference graph uses NCHW.
        self.channels_first = bool(len(input_shape) == 4 and input_shape[1] == 3)

        # The input size is defined by the model when it is static.
        if len(input_shape) == 4:
            height, width = input_shape[2], input_shape[3]
            if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
                self.in_h, self.in_w = height, width

        LOGGER.info(
            "ArcFace embedder loaded: %s (input %s, channels_first=%s)",
            path.name,
            f"{self.in_w}x{self.in_h}",
            self.channels_first,
        )

    # ------------------------------------------------------------------ #
    def preprocess(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Prepare the network input from an aligned BGR crop."""
        if aligned_bgr.shape[0] != self.in_h or aligned_bgr.shape[1] != self.in_w:
            aligned_bgr = cv2.resize(aligned_bgr, (self.in_w, self.in_h))

        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        if self.channels_first:
            blob = rgb.transpose(2, 0, 1)[None, ...]
        else:
            blob = rgb[None, ...]
        return np.ascontiguousarray(blob, dtype=np.float32)

    def embed(self, aligned_bgr: np.ndarray) -> EmbeddingResult:
        """Embed one aligned face crop."""
        blob = self.preprocess(aligned_bgr)
        output = self.session.run([self.output_name], {self.input_name: blob})[0]
        vector = np.asarray(output, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise IdentityError("The ArcFace model returned a zero vector")
        return EmbeddingResult(vector / norm, norm, int(vector.size))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two embeddings."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size != b.size or a.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
