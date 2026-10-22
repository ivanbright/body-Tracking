"""Identity recognition: the part that answers *who* the face belongs to.

The modules here are optional. The tracker works without them; they only add
the ability to recognise the enrolled people and to recognise the selected
target when it comes back into view.

* :mod:`src.recognition.arcface` - the ArcFace ONNX embedder.
* :mod:`src.recognition.alignment` - 5-point alignment of the face crop.
* :mod:`src.recognition.gallery` - the enrolled people (the "face database").
* :mod:`src.recognition.identity` - the matcher that ties everything together.
"""

from __future__ import annotations

from .alignment import FaceAligner, align_face_5pt, crop_face, estimate_norm_5pt
from .arcface import ArcFaceEncoder, EmbeddingResult, IdentityError, cosine_similarity
from .gallery import Gallery, IdentityProfile
from .identity import IdentityMatcher

__all__ = [
    "ArcFaceEncoder",
    "EmbeddingResult",
    "FaceAligner",
    "Gallery",
    "IdentityError",
    "IdentityMatcher",
    "IdentityProfile",
    "align_face_5pt",
    "cosine_similarity",
    "crop_face",
    "estimate_norm_5pt",
]
