"""The identity layer: "is this the person I am following?"

The tracker answers *where* the face is. The identity layer answers *who* it
is. Both are needed to keep a target across a disappearance: when somebody
turns away and comes back, the motion prediction is stale and the geometric
gates would reject the detection - but the **face** says "same person".

The matcher is deliberately small:

1. :meth:`IdentityMatcher.embed` turns a face box into an embedding,
2. :meth:`IdentityMatcher.similarity` compares two embeddings,
3. the caller decides what a similarity means.

The reference embedding of the target lives on the track itself, so it
survives exactly as long as the target does.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from .alignment import FaceAligner
from .arcface import ArcFaceEncoder, IdentityError, cosine_similarity
from .gallery import Gallery

LOGGER = logging.getLogger(__name__)


class IdentityMatcher:
    """Embeds face crops and compares them."""

    def __init__(self, config, *, encoder=None, aligner=None, gallery=None) -> None:
        self.config = config
        self.enabled = bool(getattr(config, "enabled", False))

        self._encoder = encoder
        self._aligner = aligner
        self._gallery = gallery
        self._landmarks: list[np.ndarray] | None = None
        self._landmarks_frame = -1

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Load the models. Raises :class:`IdentityError` when impossible."""
        if self._encoder is None:
            try:
                self._encoder = ArcFaceEncoder(
                    self.config.embedder_path,
                    providers=getattr(self.config, "providers", ("CPUExecutionProvider",)),
                )
            except IdentityError:
                raise
            except Exception as exc:
                raise IdentityError(
                    f"Could not load the ArcFace embedder "
                    f"({self.config.embedder_path}): {exc}"
                ) from exc
        if self._aligner is None:
            try:
                self._aligner = FaceAligner(
                    mode=self.config.alignment,
                    landmarker_path=self.config.landmarker_path,
                    margin=self.config.crop_margin,
                )
            except Exception as exc:
                raise IdentityError(
                    f"Could not initialise the face aligner: {exc}"
                ) from exc

    @property
    def encoder(self) -> ArcFaceEncoder | None:
        return self._encoder

    @property
    def aligner(self) -> FaceAligner | None:
        return self._aligner

    @property
    def gallery(self) -> Gallery | None:
        """The enrolled people."""
        return self._gallery

    def set_gallery(self, gallery: Gallery | None) -> None:
        self._gallery = gallery

    def load_gallery(self) -> int:
        """Load the gallery from disk. Returns the number of profiles."""
        if self._gallery is None:
            self._gallery = Gallery(self.config.gallery_path)
        return self._gallery.load()

    # ------------------------------------------------------------------ #
    def identify(self, embedding: np.ndarray | None) -> tuple[str | None, float]:
        """The enrolled person that matches the embedding best.

        The similarity is returned even when it is below the threshold - the
        caller decides whether the face is an enrolled person or "Unknown".
        """
        if self._gallery is None or embedding is None:
            return None, 0.0
        return self._gallery.best_match(embedding)

    def similarity_to(self, name: str, embedding: np.ndarray | None) -> float:
        """The similarity of a face with one specific enrolled person."""
        if self._gallery is None or embedding is None:
            return 0.0
        return self._gallery.similarity(name, embedding)

    # ------------------------------------------------------------------ #
    def landmarks(self, frame: np.ndarray, frame_index: int = -1) -> list[np.ndarray]:
        """The landmarks of the current frame (computed once per frame)."""
        if self._aligner is None or not self._aligner.uses_landmarks:
            return []
        if frame_index >= 0 and frame_index == self._landmarks_frame:
            return self._landmarks or []
        self._landmarks = self._aligner._landmarker.landmarks(frame)  # type: ignore[union-attr]
        self._landmarks_frame = frame_index
        return self._landmarks

    def embed(
        self,
        frame: np.ndarray,
        bbox: Sequence[float],
        frame_index: int = -1,
    ) -> np.ndarray | None:
        """Embed the face inside ``bbox``.

        Returns ``None`` when the crop is not usable (too small, no face in
        it, or the model failed). A missing embedding means *unknown* - it
        never means *different person*.
        """
        if not self.enabled or self._encoder is None or self._aligner is None:
            return None

        x1, y1, x2, y2 = (float(v) for v in bbox)
        if min(x2 - x1, y2 - y1) < float(self.config.min_face_size):
            return None

        try:
            aligned = self._aligner.align(
                frame,
                bbox,
                self.landmarks(frame, frame_index),
                allow_crop=self._is_face_box(bbox),
            )
            if aligned is None:
                return None
            return self._encoder.embed(aligned).embedding
        except IdentityError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Could not embed a face crop: %s", exc)
            return None

    def _is_face_box(self, bbox: Sequence[float]) -> bool:
        """Whether the box is a face box or a whole-body box.

        A face box is roughly square; a body box is (much) taller than wide.
        The distinction decides what may be done with the box:

        * a **face** box may be resized directly when the landmarks are
          missing - that is the turned-head fallback,
        * a **body** box may not: resizing a body to 112x112 is not a face.
          For a body box the landmarks are required.
        """
        box_mode = str(getattr(self.config, "box", "auto")).lower()
        if box_mode == "body":
            return False
        if box_mode == "face":
            return True
        x1, y1, x2, y2 = (float(v) for v in bbox)
        return (y2 - y1) <= 1.3 * max(1.0, x2 - x1)

    @staticmethod
    def similarity(reference: np.ndarray | None, candidate: np.ndarray | None) -> float:
        """Cosine similarity of two embeddings (``0.0`` when unknown)."""
        if reference is None or candidate is None:
            return 0.0
        return cosine_similarity(reference, candidate)
