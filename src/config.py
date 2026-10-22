"""Configuration for the YOLOX persistent face tracking system.

The whole system is driven by a YAML file (see ``config/default.yaml``).
Values are loaded into frozen-ish dataclasses, validated eagerly at start-up so
that a broken configuration fails fast with an actionable message instead of
producing nonsense frames later.

Every setting can also be overridden from the environment using the
``YOLOX_`` prefix and ``__`` as the section separator, e.g.::

    YOLOX_DETECTOR__BACKEND=yolox_onnx
    YOLOX_TRACKER__MAX_LOST_TIME=90

Environment variables use a *double* underscore because single underscores are
used inside the key names themselves (e.g. ``score_thresh``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

ENV_PREFIX = "YOLOX_"
ENV_SEPARATOR = "__"

DETECTOR_BACKENDS = ("yolox_onnx", "haar")
SOURCE_TYPES = ("camera", "video")
RESIZE_MODES = ("letterbox", "stretch")
INPUT_LAYOUTS = ("auto", "nchw", "nhwc")

#: Value range the network input is fed with.
#:
#: ``0_255``
#:     The YOLOX reference convention: raw pixel values as float32. The YOLOX
#:     head was trained on 0-255 inputs (``yolox/models/transforms.py`` does
#:     not divide by 255), so an official export expects this range.
#: ``0_1``
#:     Divide by 255. Needed for exports that were wrapped with their own
#:     normalisation.
INPUT_RANGES = ("0_255", "0_1")

#: Channel order of the network input.
#:
#: ``rgb``
#:     The YOLOX reference convention (``img.transpose((2, 0, 1))[::-1]``
#:     converts BGR to RGB).
#: ``bgr``
#:     Keep the order of the source image.
COLOR_ORDERS = ("rgb", "bgr")

#: Output layouts of the supported YOLOX exports.
#:
#: The decoder is **never** guessed from the data. The operator states which
#: export produced the model, and the decoder validates the model's real
#: output shape against that statement.
#:
#: ``yolox_raw``
#:     ``(1, N, 5 + C)`` - ``[cx, cy, w, h, obj, cls...]`` where the box is
#:     still expressed in stride units and has to be decoded with the anchor
#:     grid and the strides. This is what Megvii's ``tools/export_onnx.py``
#:     produces **by default**, because ``--decode_in_inference`` is a
#:     ``store_true`` flag and therefore defaults to ``False``
#:     (``yolox/models/yolo_head.py``).
#: ``yolox_decoded``
#:     ``(1, N, 5 + C)`` - the same columns, but the box is already decoded into
#:     network-input pixels. This is what the export produces with
#:     ``--decode_in_inference``.
#: ``yolox_dfl``
#:     ``(1, N, 4 * reg_max + 1 + C)`` - distribution-focal regression logits.
#: ``xyxy``
#:     ``(1, N, 6)`` - ``[x1, y1, x2, y2, score, class_id]``.
OUTPUT_FORMATS = ("yolox_decoded", "yolox_raw", "yolox_dfl", "xyxy")

#: YOLOX applies the sigmoid **inside the head**
#: (``obj_output.sigmoid()``, ``cls_output.sigmoid()``), so the exported
#: objectness and class scores are already probabilities and must not be
#: squashed again. ``sigmoid`` is only for custom exports that emit logits.
ACTIVATIONS = ("none", "sigmoid")

#: How the operator selects the target.
#:
#: ``none``      - nothing is selected.
#: ``largest``   - the largest visible face is selected. This is a *geometric*
#:                heuristic, it does **not** identify a person.
#: ``track_id``  - the track with that ID is selected.
#: ``identity``  - the face that was enrolled under that name is selected.
TARGET_MODES = ("none", "largest", "track_id", "identity")

#: How the face crop is prepared for the ArcFace embedder.
#:
#: ``auto``      - landmarks when the landmark model is available, plain crop
#:                 otherwise.
#: ``landmarks`` - always use the 5-point landmarks (accurate, needs the
#:                 MediaPipe landmark model).
#: ``crop``      - always use the plain detector box (no extra model).
ALIGNMENT_MODES = ("auto", "landmarks", "crop")

#: ``auto``  - a box much taller than wide is treated as a whole body.
#: ``face``  - the detector box is the face.
#: ``body``  - the detector box is the whole person.
BOX_MODES = ("auto", "face", "body")


class ConfigError(ValueError):
    """Raised when the configuration is missing, malformed or fails validation."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ConfigError(f"{field_name}: expected a boolean, got {value!r}")


def _as_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{field_name}: expected an integer, got {value!r}") from exc


def _as_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{field_name}: expected a number, got {value!r}") from exc


def _as_str_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items: Iterable[Any] = [part.strip() for part in value.split(",")]
    elif isinstance(value, Sequence):
        items = value
    else:
        raise ConfigError(f"{field_name}: expected a list of strings, got {value!r}")
    return tuple(str(item).strip() for item in items if str(item).strip())


def _reject_unknown(data: Mapping[str, Any], allowed: Iterable[str], section: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ConfigError(
            f"{section}: unknown configuration key(s): {', '.join(unknown)}. "
            f"Allowed keys: {', '.join(sorted(allowed))}"
        )


# --------------------------------------------------------------------------- #
# configuration sections
# --------------------------------------------------------------------------- #
@dataclass
class DetectorConfig:
    """Which detector backend to use and how to run it."""

    backend: str = "yolox_onnx"
    model_path: str = "models/yolox_face.onnx"
    labels_path: str = ""
    input_size: tuple[int, int] = (640, 640)
    score_thresh: float = 0.3
    nms_thresh: float = 0.45
    target_classes: tuple[str, ...] = ()
    providers: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")
    input_layout: str = "auto"
    resize_mode: str = "letterbox"
    input_range: str = "0_255"
    color_order: str = "rgb"
    output_format: str = "yolox_raw"
    activation: str = "none"
    num_classes: int = 1
    strides: tuple[int, ...] = (8, 16, 32)
    reg_max: int = 16
    haar_min_size: tuple[int, int] = (40, 40)
    warmup: bool = True
    num_threads: int = 0

    def validate(self) -> None:
        if self.backend not in DETECTOR_BACKENDS:
            raise ConfigError(
                f"detector.backend: {self.backend!r} is not supported. "
                f"Choose one of: {', '.join(DETECTOR_BACKENDS)}"
            )
        if len(self.input_size) != 2:
            raise ConfigError("detector.input_size: expected [width, height]")
        width, height = self.input_size
        if width <= 0 or height <= 0:
            raise ConfigError("detector.input_size: width and height must be positive")
        if not 0.0 <= self.score_thresh <= 1.0:
            raise ConfigError("detector.score_thresh: must be within [0.0, 1.0]")
        if not 0.0 <= self.nms_thresh <= 1.0:
            raise ConfigError("detector.nms_thresh: must be within [0.0, 1.0]")
        if self.resize_mode not in RESIZE_MODES:
            raise ConfigError(
                f"detector.resize_mode: {self.resize_mode!r} is not supported. "
                f"Choose one of: {', '.join(RESIZE_MODES)}"
            )
        if self.output_format not in OUTPUT_FORMATS:
            raise ConfigError(
                f"detector.output_format: {self.output_format!r} is not supported. "
                f"Choose one of: {', '.join(OUTPUT_FORMATS)}. "
                "Run `python -m src.inspect_model <model.onnx>` to see the real "
                "output shape of the model."
            )
        if self.activation not in ACTIVATIONS:
            raise ConfigError(
                f"detector.activation: {self.activation!r} is not supported. "
                f"Choose one of: {', '.join(ACTIVATIONS)}"
            )
        if self.input_layout not in INPUT_LAYOUTS:
            raise ConfigError(
                f"detector.input_layout: {self.input_layout!r} is not supported. "
                f"Choose one of: {', '.join(INPUT_LAYOUTS)}"
            )
        if self.input_range not in INPUT_RANGES:
            raise ConfigError(
                f"detector.input_range: {self.input_range!r} is not supported. "
                f"Choose one of: {', '.join(INPUT_RANGES)}"
            )
        if self.color_order not in COLOR_ORDERS:
            raise ConfigError(
                f"detector.color_order: {self.color_order!r} is not supported. "
                f"Choose one of: {', '.join(COLOR_ORDERS)}"
            )
        if self.num_classes < 1:
            raise ConfigError("detector.num_classes: must be >= 1")
        if not self.strides or any(int(s) <= 0 for s in self.strides):
            raise ConfigError("detector.strides: expected positive strides, e.g. [8, 16, 32]")
        if self.reg_max < 1:
            raise ConfigError("detector.reg_max: must be >= 1")
        if len(self.haar_min_size) != 2 or min(self.haar_min_size) <= 0:
            raise ConfigError(
                "detector.haar_min_size: expected [width, height] with positive values"
            )
        if self.num_threads < 0:
            raise ConfigError("detector.num_threads: must be >= 0 (0 = runtime default)")
        if self.backend == "haar" and self.target_classes:
            # The Haar cascade is a single-class face detector; filtering by name
            # would silently drop every detection.
            raise ConfigError(
                "detector.target_classes: the 'haar' backend is a single-class face "
                "detector and cannot filter by class name"
            )


@dataclass
class SourceConfig:
    """Where the frames come from."""

    type: str = "camera"
    camera_index: int = 0
    video_path: str = ""
    width: int = 640
    height: int = 480
    fps: float = 30.0
    loop: bool = False

    def validate(self) -> None:
        if self.type not in SOURCE_TYPES:
            raise ConfigError(
                f"source.type: {self.type!r} is not supported. "
                f"Choose one of: {', '.join(SOURCE_TYPES)}"
            )
        if self.type == "camera" and self.camera_index < 0:
            raise ConfigError("source.camera_index: must be >= 0")
        if self.type == "video" and not self.video_path:
            raise ConfigError("source.video_path: required when source.type == 'video'")
        if self.width <= 0 or self.height <= 0:
            raise ConfigError("source.width / source.height: must be positive")
        if self.fps <= 0:
            raise ConfigError("source.fps: must be positive")


@dataclass
class TrackerConfig:
    """ByteTrack-style association thresholds and track lifecycle limits."""

    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    match_thresh: float = 0.8
    min_hits_to_confirm: int = 3
    max_init_frames: int = 15
    max_lost_time: int = 90
    lost_retention_seconds: float = 3.0
    max_lost_tracks: int = 32
    max_active_tracks: int = 32
    distance_weight: float = 0.35
    use_motion: bool = True

    def validate(self) -> None:
        if not 0.0 <= self.track_low_thresh <= 1.0:
            raise ConfigError("tracker.track_low_thresh: must be within [0.0, 1.0]")
        if not 0.0 <= self.track_high_thresh <= 1.0:
            raise ConfigError("tracker.track_high_thresh: must be within [0.0, 1.0]")
        if self.track_low_thresh > self.track_high_thresh:
            raise ConfigError(
                "tracker.track_low_thresh: must be <= tracker.track_high_thresh"
            )
        if not 0.0 <= self.match_thresh <= 1.0:
            raise ConfigError("tracker.match_thresh: must be within [0.0, 1.0]")
        if self.min_hits_to_confirm < 1:
            raise ConfigError("tracker.min_hits_to_confirm: must be >= 1")
        if self.max_init_frames < 1:
            raise ConfigError("tracker.max_init_frames: must be >= 1")
        if self.max_lost_time < 1:
            raise ConfigError("tracker.max_lost_time: must be >= 1")
        if self.lost_retention_seconds <= 0:
            raise ConfigError("tracker.lost_retention_seconds: must be > 0")
        if self.max_lost_tracks < 1:
            raise ConfigError("tracker.max_lost_tracks: must be >= 1")
        if self.max_active_tracks < 1:
            raise ConfigError("tracker.max_active_tracks: must be >= 1")
        if not 0.0 <= self.distance_weight <= 1.0:
            raise ConfigError("tracker.distance_weight: must be within [0.0, 1.0]")


@dataclass
class ReacquisitionConfig:
    """Gating rules for restoring a lost track.

    The rules are **duration aware**: a two second occlusion and a two minute
    absence are not the same problem. While the gap is short the motion
    prediction is still trustworthy and the ordinary gates apply. Once the gap
    grows beyond ``long_gap_seconds`` the prediction becomes unreliable, so the
    spatial gate is tightened, the required evidence score is raised and a
    minimum detection confidence is enforced.
    """

    enabled: bool = True
    score_threshold: float = 0.55
    max_distance_ratio: float = 2.5
    min_size_ratio: float = 0.5
    max_size_ratio: float = 2.0
    ambiguity_margin: float = 0.08
    distance_weight: float = 0.40
    overlap_weight: float = 0.20
    velocity_weight: float = 0.15
    size_weight: float = 0.10
    confidence_weight: float = 0.15
    time_decay_tau: float = 2.0

    # --- duration aware rules ------------------------------------------ #
    long_gap_seconds: float = 2.0
    long_gap_score_multiplier: float = 1.3
    long_gap_max_distance_ratio: float = 1.5
    long_gap_min_confidence: float = 0.35

    # --- optional appearance matching --------------------------------- #
    appearance_enabled: bool = False
    appearance_weight: float = 0.0
    appearance_threshold: float = 0.0

    def validate(self) -> None:
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ConfigError("reacquisition.score_threshold: must be within [0.0, 1.0]")
        if self.max_distance_ratio <= 0:
            raise ConfigError("reacquisition.max_distance_ratio: must be > 0")
        if not 0.0 < self.min_size_ratio <= 1.0:
            raise ConfigError("reacquisition.min_size_ratio: must be within (0.0, 1.0]")
        if self.max_size_ratio < 1.0:
            raise ConfigError("reacquisition.max_size_ratio: must be >= 1.0")
        if self.min_size_ratio > self.max_size_ratio:
            raise ConfigError(
                "reacquisition.min_size_ratio: must be <= reacquisition.max_size_ratio"
            )
        if self.ambiguity_margin < 0:
            raise ConfigError("reacquisition.ambiguity_margin: must be >= 0")
        weights = {
            "distance_weight": self.distance_weight,
            "overlap_weight": self.overlap_weight,
            "velocity_weight": self.velocity_weight,
            "size_weight": self.size_weight,
            "confidence_weight": self.confidence_weight,
        }
        for name, value in weights.items():
            if value < 0:
                raise ConfigError(f"reacquisition.{name}: must be >= 0")
        if self.time_decay_tau <= 0:
            raise ConfigError("reacquisition.time_decay_tau: must be > 0")
        if self.long_gap_seconds <= 0:
            raise ConfigError("reacquisition.long_gap_seconds: must be > 0")
        if self.long_gap_score_multiplier < 1.0:
            raise ConfigError(
                "reacquisition.long_gap_score_multiplier: must be >= 1.0 - a long "
                "gap may only make the evidence requirement stricter"
            )
        if self.long_gap_max_distance_ratio <= 0:
            raise ConfigError("reacquisition.long_gap_max_distance_ratio: must be > 0")
        if not 0.0 <= self.long_gap_min_confidence <= 1.0:
            raise ConfigError("reacquisition.long_gap_min_confidence: must be within [0.0, 1.0]")
        if self.appearance_enabled and self.appearance_weight <= 0:
            raise ConfigError(
                "reacquisition.appearance_weight: must be > 0 when appearance is enabled"
            )


@dataclass
class IdentityConfig:
    """The optional face identity layer (ArcFace).

    The layer is **off by default**. When it is enabled every visible face is
    matched against the **gallery** (the enrolled people) and gets its name
    when the similarity is high enough. The name is what the overlay shows in
    the green frame, and ``target.mode == "identity"`` selects the target by
    that name.

    The layer also restores lost tracks: while a face is missing every
    candidate is compared against the gallery, and a candidate that is
    recognised gets the **original** track ID back, even when the motion
    prediction no longer points at the right place.

    The rules are deliberately conservative:

    * a similarity **below** ``threshold`` means "this is somebody else" and
      the candidate is rejected,
    * a similarity **at or above** ``threshold`` confirms the identity and
      relaxes the geometric gates (distance, size),
    * two candidates that are both confirmed but too similar to each other are
      **ambiguous** and both are rejected - the system must not guess.
    """

    enabled: bool = False
    embedder_path: str = "models/embedder_arcface.onnx"
    landmarker_path: str = "models/face_landmarker.task"
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    alignment: str = "auto"

    #: What the detector boxes are: ``auto`` (a box much taller than wide is a
    #: whole body), ``face`` (the box is the face) or ``body`` (the box is the
    #: whole person; the face is located with the landmark model inside it).
    box: str = "auto"

    #: The enrolled people (the "gallery" of the working reference system).
    gallery_path: str = "profiles/face_database.json"

    #: How often (in frames) the visible faces are matched against the gallery.
    gallery_interval: int = 10

    #: How many embeddings of one face are kept (the different poses).
    max_embeddings: int = 8

    #: Minimum cosine similarity to accept a candidate as the same person.
    threshold: float = 0.6

    #: Faces smaller than this (in pixels) are not embedded.
    min_face_size: int = 40

    #: Relative enlargement of the crop around the detector box.
    crop_margin: float = 0.15

    #: Observations that are less similar than this to the known poses are
    #: ignored, so the reference cannot drift onto a different person.
    reference_min_similarity: float = 0.4

    #: How many frames pass between two reference updates.
    reference_interval: int = 3

    #: Maximum distance (relative to the track diagonal) that is still accepted
    #: when the identity confirms the candidate. The face is the evidence then:
    #: a person who walked across the room is still that person.
    max_distance_ratio: float = 15.0

    #: Size ratio limits that are still accepted when the identity confirms
    #: the candidate (a face that turned back may be much closer or further).
    min_size_ratio: float = 0.25
    max_size_ratio: float = 4.0

    #: Evidence score of a candidate that was confirmed by the identity.
    confirmed_score: float = 0.95

    #: Two confirmed candidates that are closer than this to each other are
    #: considered ambiguous and are both rejected.
    ambiguity_margin: float = 0.1

    def validate(self) -> None:
        if self.alignment not in ALIGNMENT_MODES:
            raise ConfigError(
                f"identity.alignment: {self.alignment!r} is not supported. "
                f"Choose one of: {', '.join(ALIGNMENT_MODES)}"
            )
        if self.box not in BOX_MODES:
            raise ConfigError(
                f"identity.box: {self.box!r} is not supported. "
                f"Choose one of: {', '.join(BOX_MODES)}"
            )
        if self.gallery_interval < 1:
            raise ConfigError("identity.gallery_interval: must be >= 1")
        if self.max_embeddings < 1:
            raise ConfigError("identity.max_embeddings: must be >= 1")
        if not 0.0 <= self.threshold <= 1.0:
            raise ConfigError("identity.threshold: must be within [0.0, 1.0]")
        if self.min_face_size < 0:
            raise ConfigError("identity.min_face_size: must be >= 0")
        if self.crop_margin < 0:
            raise ConfigError("identity.crop_margin: must be >= 0")
        if not -1.0 <= self.reference_min_similarity <= 1.0:
            raise ConfigError("identity.reference_min_similarity: must be within [-1.0, 1.0]")
        if self.reference_interval < 1:
            raise ConfigError("identity.reference_interval: must be >= 1")
        if self.max_distance_ratio <= 0:
            raise ConfigError("identity.max_distance_ratio: must be > 0")
        if not 0.0 < self.min_size_ratio <= 1.0:
            raise ConfigError("identity.min_size_ratio: must be within (0.0, 1.0]")
        if self.max_size_ratio < 1.0:
            raise ConfigError("identity.max_size_ratio: must be >= 1.0")
        if self.min_size_ratio > self.max_size_ratio:
            raise ConfigError(
                "identity.min_size_ratio: must be <= identity.max_size_ratio"
            )
        if not 0.0 <= self.confirmed_score <= 1.0:
            raise ConfigError("identity.confirmed_score: must be within [0.0, 1.0]")
        if self.ambiguity_margin < 0:
            raise ConfigError("identity.ambiguity_margin: must be >= 0")


@dataclass
class TargetConfig:
    """How the operator selects the target and how long it is retained.

    Selecting a target is a **tracking** operation: the operator picks one of
    the tracks the detector produced. It is not an identity decision - the
    system never claims that a track belongs to a specific person.

    ``mode``
        ``none``, ``largest`` (largest visible face - a geometric heuristic),
        ``track_id`` (a specific track, e.g. chosen with a mouse click) or
        ``identity`` (the face that was enrolled under ``name``).
    ``track_id``
        The track that is selected at start-up when ``mode == "track_id"``.
    ``name``
        The enrolled person that is selected when ``mode == "identity"``.
    ``lost_retention_seconds``
        How long the selected target survives without being observed. When the
        limit is reached the target *expires*: the ID is released and the
        system stops searching for it.
    ``require_confirmed``
        Only tracks that have been confirmed by several observations may be
        selected.
    ``auto_reselect``
        When the target expires, pick a new one automatically. Off by default:
        the system must not silently switch to a different face.
    """

    mode: str = "none"
    track_id: int = 0
    name: str = ""
    lost_retention_seconds: float = 10.0
    require_confirmed: bool = True
    auto_reselect: bool = False

    def validate(self) -> None:
        if self.mode not in TARGET_MODES:
            raise ConfigError(
                f"target.mode: {self.mode!r} is not supported. "
                f"Choose one of: {', '.join(TARGET_MODES)}"
            )
        if self.track_id < 0:
            raise ConfigError("target.track_id: must be >= 0")
        if self.mode == "track_id" and self.track_id <= 0:
            raise ConfigError(
                "target.track_id: must be a valid track ID when target.mode == 'track_id'"
            )
        if self.mode == "identity" and not self.name.strip():
            raise ConfigError(
                "target.name: must be set when target.mode == 'identity' "
                "(the name of an enrolled person)"
            )
        if self.lost_retention_seconds <= 0:
            raise ConfigError("target.lost_retention_seconds: must be > 0")


@dataclass
class PredictionConfig:
    """How long predictions are trusted after the last observation."""

    max_prediction_time: float = 1.0
    max_prediction_distance: float = 0.0  # 0 = unlimited

    def validate(self) -> None:
        if self.max_prediction_time <= 0:
            raise ConfigError("prediction.max_prediction_time: must be > 0")
        if self.max_prediction_distance < 0:
            raise ConfigError("prediction.max_prediction_distance: must be >= 0")


@dataclass
class VisualizationConfig:
    """Overlay behaviour."""

    enabled: bool = True
    show_predictions: bool = True
    show_trail: bool = True
    trail_length: int = 30
    window_name: str = "YOLOX Face Tracking"
    show_dashboard: bool = True

    def validate(self) -> None:
        if self.trail_length < 0:
            raise ConfigError("visualization.trail_length: must be >= 0")


@dataclass
class LoggingConfig:
    """Diagnostic logging."""

    level: str = "INFO"
    file: str = ""
    format: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

    def validate(self) -> None:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid:
            raise ConfigError(
                f"logging.level: {self.level!r} is not valid. Choose from: {', '.join(sorted(valid))}"
            )


@dataclass
class RuntimeConfig:
    """Pipeline execution options."""

    detection_interval: int = 1
    max_frames: int = 0
    record_output: str = ""
    save_debug_frames: bool = False

    def validate(self) -> None:
        if self.detection_interval < 1:
            raise ConfigError("runtime.detection_interval: must be >= 1")
        if self.max_frames < 0:
            raise ConfigError("runtime.max_frames: must be >= 0 (0 = unlimited)")


@dataclass
class AppConfig:
    """Root configuration object."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    reacquisition: ReacquisitionConfig = field(default_factory=ReacquisitionConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    project_root: Path = field(default_factory=Path.cwd)

    # ------------------------------------------------------------------ #
    def validate(self) -> "AppConfig":
        self.detector.validate()
        self.source.validate()
        self.tracker.validate()
        self.target.validate()
        self.reacquisition.validate()
        self.identity.validate()
        self.prediction.validate()
        self.visualization.validate()
        self.logging.validate()
        self.runtime.validate()
        if self.target.mode == "identity" and not self.identity.enabled:
            raise ConfigError(
                "target.mode: 'identity' needs the identity layer. "
                "Set identity.enabled: true (or use --identity)."
            )
        return self

    def resolve_paths(self) -> "AppConfig":
        """Resolve relative paths against :attr:`project_root`.

        Absolute paths are left untouched. Paths that do not exist are *not*
        validated here — the detector raises a clear error when it needs the
        file, which keeps ``--check-config`` useful for dry runs.
        """
        root = Path(self.project_root)

        def resolve(value: str) -> str:
            if not value:
                return value
            candidate = Path(value)
            if candidate.is_absolute():
                return str(candidate)
            return str((root / candidate).resolve())

        self.detector.model_path = resolve(self.detector.model_path)
        self.detector.labels_path = resolve(self.detector.labels_path)
        self.source.video_path = resolve(self.source.video_path)
        self.identity.embedder_path = resolve(self.identity.embedder_path)
        self.identity.landmarker_path = resolve(self.identity.landmarker_path)
        self.identity.gallery_path = resolve(self.identity.gallery_path)
        self.logging.file = resolve(self.logging.file)
        self.runtime.record_output = resolve(self.runtime.record_output)
        return self

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if is_dataclass(value):
                return {f.name: convert(getattr(value, f.name)) for f in fields(value)}
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            return value

        return convert(self)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
_SECTIONS: dict[str, type] = {
    "detector": DetectorConfig,
    "source": SourceConfig,
    "tracker": TrackerConfig,
    "target": TargetConfig,
    "reacquisition": ReacquisitionConfig,
    "identity": IdentityConfig,
    "prediction": PredictionConfig,
    "visualization": VisualizationConfig,
    "logging": LoggingConfig,
    "runtime": RuntimeConfig,
}


def _coerce_section(section_name: str, section_type: type, data: Mapping[str, Any]) -> Any:
    allowed = {f.name for f in fields(section_type)}
    _reject_unknown(data, allowed, section_name)
    kwargs: dict[str, Any] = {}
    for f in fields(section_type):
        if f.name not in data:
            continue
        value = data[f.name]
        if f.name in {"input_size", "haar_min_size"}:
            kwargs[f.name] = (_as_int(value[0], f"{section_name}.{f.name}[0]"),
                             _as_int(value[1], f"{section_name}.{f.name}[1]"))
        elif f.name == "strides":
            kwargs[f.name] = tuple(
                _as_int(item, f"{section_name}.{f.name}[{index}]")
                for index, item in enumerate(value)
            )
        elif f.name in {"target_classes", "providers"}:
            kwargs[f.name] = _as_str_tuple(value, f"{section_name}.{f.name}")
        elif f.type == "bool" or isinstance(getattr(section_type(), f.name), bool):
            kwargs[f.name] = _as_bool(value, f"{section_name}.{f.name}")
        elif isinstance(getattr(section_type(), f.name), int):
            kwargs[f.name] = _as_int(value, f"{section_name}.{f.name}")
        elif isinstance(getattr(section_type(), f.name), float):
            kwargs[f.name] = _as_float(value, f"{section_name}.{f.name}")
        else:
            kwargs[f.name] = str(value)
    return section_type(**kwargs)


def _apply_env_overrides(data: dict[str, Any], environ: Mapping[str, str]) -> None:
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = key[len(ENV_PREFIX):].lower().split(ENV_SEPARATOR)
        if not path or not path[0]:
            continue
        section = path[0]
        if section not in _SECTIONS:
            raise ConfigError(
                f"environment variable {key}: unknown configuration section {section!r}. "
                f"Known sections: {', '.join(sorted(_SECTIONS))}"
            )
        if len(path) == 1:
            raise ConfigError(
                f"environment variable {key}: expected a specific setting, e.g. "
                f"{ENV_PREFIX}{section.upper()}{ENV_SEPARATOR}SETTING"
            )
        target = data.setdefault(section, {})
        if len(path) != 2:
            raise ConfigError(f"environment variable {key}: nested paths deeper than one level are not supported")
        target[path[1]] = raw


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    project_root: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load, merge, override and validate the application configuration.

    Args:
        path: YAML file to load. ``None`` uses ``<project_root>/config/default.yaml``.
        project_root: Root used to resolve relative paths and to find the
            default configuration file. Defaults to the ``yolox`` package parent.
        environ: Environment mapping used for overrides (defaults to ``os.environ``).

    Raises:
        ConfigError: The file is missing, malformed, contains unknown keys or
            fails validation.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    root = Path(project_root).resolve()

    if path is None:
        path = root / "config" / "default.yaml"
    config_path = Path(path).resolve()

    if not config_path.exists():
        raise ConfigError(
            f"configuration file not found: {config_path}. "
            "Pass --config <file> or restore config/default.yaml."
        )

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"configuration file {config_path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"configuration file {config_path} must contain a mapping at the top level")

    _reject_unknown(raw, _SECTIONS, "<root>")

    data: dict[str, Any] = {key: dict(value) for key, value in raw.items() if isinstance(value, dict)}
    _apply_env_overrides(data, environ if environ is not None else os.environ)

    sections: dict[str, Any] = {}
    for name, section_type in _SECTIONS.items():
        section_data = data.get(name, {})
        if not isinstance(section_data, dict):
            raise ConfigError(f"{name}: expected a mapping, got {type(section_data).__name__}")
        sections[name] = _coerce_section(name, section_type, section_data)

    config = AppConfig(project_root=root, **sections)
    return config.resolve_paths().validate()
