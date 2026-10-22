"""Face alignment for the ArcFace embedder.

The ArcFace graph expects a face that is **aligned**: the eyes, the nose and
the mouth corners are at fixed positions of a 112x112 image. A detector box
does not have that property - the face can be tilted, off-centre or cropped
differently - and feeding such a crop to the embedder costs a lot of accuracy.

Two strategies are supported:

``landmarks``
    The five ArcFace keypoints (left eye, right eye, nose, left mouth corner,
    right mouth corner) are estimated with MediaPipe FaceLandmarker and the
    crop is warped onto the standard template. This is what the reference
    implementation does and it is the accurate one.
``crop``
    The detector box is simply resized to 112x112. It needs no extra model and
    works everywhere, but the similarity scores are weaker.

``auto`` (the default) uses the landmarks when the landmark model is available
and falls back to the plain crop otherwise.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

#: The standard ArcFace 5-point template for a 112x112 image.
ARCFACE_TEMPLATE_5PT = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)


def estimate_norm_5pt(
    kps_5x2: np.ndarray, out_size: tuple[int, int] = (112, 112)
) -> np.ndarray:
    """Build the 2x3 affine matrix that maps five keypoints onto the template."""
    kps = np.asarray(kps_5x2, dtype=np.float32).reshape(5, 2)

    out_w, out_h = int(out_size[0]), int(out_size[1])
    template = ARCFACE_TEMPLATE_5PT
    if (out_w, out_h) != (112, 112):
        template = template * np.array([out_w / 112.0, out_h / 112.0], dtype=np.float32)

    matrix, _ = cv2.estimateAffinePartial2D(kps, template, method=cv2.LMEDS)
    if matrix is None:
        matrix = cv2.getAffineTransform(
            np.array([kps[0], kps[1], kps[2]], dtype=np.float32),
            np.array([template[0], template[1], template[2]], dtype=np.float32),
        )
    return matrix.astype(np.float32)


def align_face_5pt(
    frame_bgr: np.ndarray,
    kps_5x2: np.ndarray,
    out_size: tuple[int, int] = (112, 112),
) -> tuple[np.ndarray, np.ndarray]:
    """Warp the face around the five keypoints onto the ArcFace template."""
    matrix = estimate_norm_5pt(kps_5x2, out_size=out_size)
    out_w, out_h = int(out_size[0]), int(out_size[1])
    aligned = cv2.warpAffine(
        frame_bgr,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned, matrix


def crop_face(
    frame_bgr: np.ndarray,
    bbox: Sequence[float],
    out_size: tuple[int, int] = (112, 112),
    margin: float = 0.0,
) -> np.ndarray | None:
    """Cut the box out of the frame and resize it.

    ``margin`` enlarges the box relative to its own size, which keeps a bit of
    chin and forehead in the crop.
    """
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)

    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    x1 -= box_w * margin
    x2 += box_w * margin
    y1 -= box_h * margin
    y2 += box_h * margin

    x1 = int(round(max(0.0, min(x1, width - 1.0))))
    y1 = int(round(max(0.0, min(y1, height - 1.0))))
    x2 = int(round(max(x1 + 1.0, min(x2, width))))
    y2 = int(round(max(y1 + 1.0, min(y2, height))))

    patch = frame_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    return cv2.resize(patch, (int(out_size[0]), int(out_size[1])))


class FaceLandmarker:
    """Five-point landmarks via MediaPipe FaceLandmarker.

    The landmark model is optional: when it is missing the aligner falls back
    to a plain crop.
    """

    #: FaceMesh indices of the five ArcFace keypoints.
    _INDEXES = (33, 263, 1, 61, 291)

    def __init__(self, model_path: str | Path, num_faces: int = 5) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Face landmark model not found: {path}")

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(f"mediapipe is not available: {exc}") from exc

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=int(num_faces),
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        LOGGER.info("Face landmarker loaded: %s", path.name)

    def landmarks(self, frame_bgr: np.ndarray) -> list[np.ndarray]:
        """Return the five-point landmarks of every face in the frame."""
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)

        faces: list[np.ndarray] = []
        for landmarks in result.face_landmarks or []:
            points = np.array(
                [
                    [landmarks[index].x * width, landmarks[index].y * height]
                    for index in self._INDEXES
                ],
                dtype=np.float32,
            )
            # Keep the left/right order stable.
            if points[0, 0] > points[1, 0]:
                points[[0, 1]] = points[[1, 0]]
            if points[3, 0] > points[4, 0]:
                points[[3, 4]] = points[[4, 3]]
            faces.append(points)
        return faces


class FaceAligner:
    """Produces the 112x112 crop that the embedder expects."""

    def __init__(
        self,
        mode: str = "auto",
        landmarker_path: str | Path = "",
        out_size: tuple[int, int] = (112, 112),
        margin: float = 0.15,
    ) -> None:
        #: The mode that was asked for. It decides what happens when the
        #: landmark model does not deliver a face for a given box.
        self.requested_mode = mode
        self.mode = mode
        self.out_size = (int(out_size[0]), int(out_size[1]))
        self.margin = float(margin)
        self._landmarker: FaceLandmarker | None = None

        if mode in ("auto", "landmarks") and landmarker_path:
            try:
                self._landmarker = FaceLandmarker(landmarker_path)
            except Exception as exc:
                if mode == "landmarks":
                    raise
                LOGGER.warning(
                    "Landmark model unusable (%s) - falling back to plain crops", exc
                )
                self._landmarker = None

        self.mode = "landmarks" if self._landmarker is not None else "crop"

    @property
    def uses_landmarks(self) -> bool:
        return self._landmarker is not None

    def align(
        self,
        frame_bgr: np.ndarray,
        bbox: Sequence[float],
        landmarks: Sequence[np.ndarray] | None = None,
        *,
        allow_crop: bool = True,
    ) -> np.ndarray | None:
        """Return the aligned crop of the face inside ``bbox``.

        In the ``auto`` mode the plain crop is used whenever the landmark
        model does not deliver a face for the box. That happens regularly -
        a head that is turned to the side is exactly the case where the
        landmark model gives up - and a missing crop would mean a missing
        embedding, which would mean "Unknown" for a face that is perfectly
        recognisable.

        ``allow_crop=False`` is used for a **whole-body** box: resizing a body
        to 112x112 would not be a face at all. For a body box the landmarks are
        the only usable source, so a missing face means *no embedding*.
        """
        if self._landmarker is not None:
            points = landmarks if landmarks is not None else self._landmarker.landmarks(frame_bgr)
            best = self._best_landmarks(points, bbox)
            if best is not None:
                aligned, _ = align_face_5pt(frame_bgr, best, out_size=self.out_size)
                return aligned
            if self.requested_mode == "landmarks" or not allow_crop:
                return None
        elif not allow_crop:
            return None
        return crop_face(frame_bgr, bbox, out_size=self.out_size, margin=self.margin)

    @staticmethod
    def _best_landmarks(
        landmarks: Sequence[np.ndarray], bbox: Sequence[float]
    ) -> np.ndarray | None:
        """Pick the landmark set that belongs to the given box."""
        if not landmarks:
            return None
        x1, y1, x2, y2 = (float(v) for v in bbox)
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)

        if height > 1.3 * width:
            # A whole-body box: the face sits near the top. Pick the face that
            # is inside the box and closest to its top centre - a neighbour
            # standing next to the person must not win.
            target = np.array([(x1 + x2) / 2.0, y1 + 0.15 * height], dtype=np.float32)
            inside = [
                points
                for points in landmarks
                if x1 <= points.mean(axis=0)[0] <= x2
                and y1 <= points.mean(axis=0)[1] <= y1 + 0.6 * height
            ]
            if not inside:
                return None
            return min(inside, key=lambda points: float(np.linalg.norm(points.mean(axis=0) - target)))

        center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
        scale = max(1.0, float(np.hypot(x2 - x1, y2 - y1)))

        best = None
        best_distance = float("inf")
        for points in landmarks:
            face_center = points.mean(axis=0)
            distance = float(np.linalg.norm(face_center - center))
            if distance < best_distance:
                best_distance = distance
                best = points
        if best is None or best_distance > scale:
            # The landmark does not belong to this box.
            return None
        return best
