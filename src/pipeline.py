"""The tracking pipeline.

The pipeline wires the components together and owns the main loop:

.. code-block:: text

    frame source -> detector -> multi-object tracker -> target lock -> overlay

The lock is updated **after** the tracker, so it always sees the state that the
tracker produced for the current frame.

Nothing in here talks to hardware, to a message broker or to any other
project: the pipeline reads frames, tracks faces and draws the result.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .camera.capture import FrameSource, SourceError, create_source
from .detection.base import Detection, DetectorError, FaceDetector
from .detection import create_detector
from .recognition.arcface import IdentityError
from .recognition.identity import IdentityMatcher
from .tracking.reacquisition import IdentityEvidence
from .tracking.target_lock import TargetLock, TargetState
from .tracking.track_state import Track, TrackState
from .tracking.tracker import Tracker
from .visualization.renderer import DashboardStats, Renderer

LOGGER = logging.getLogger(__name__)

#: Keyboard controls of the live view.
KEY_QUIT = (ord("q"), 27)  # q and ESC
KEY_SELECT_LARGEST = ord("t")
KEY_CLEAR = ord("c")


@dataclass
class PipelineStats:
    """What the pipeline measured during a run."""

    frames: int = 0
    detections: int = 0
    average_fps: float = 0.0
    average_inference_ms: float = 0.0
    average_frame_ms: float = 0.0
    total_time_s: float = 0.0
    target_events: list[str] = field(default_factory=list)
    track_events: list[str] = field(default_factory=list)
    expired_tracks: int = 0
    reacquisitions: int = 0
    identity_reacquisitions: int = 0

    def summary(self) -> str:
        return (
            f"frames={self.frames} "
            f"detections={self.detections} "
            f"fps={self.average_fps:.1f} "
            f"inference={self.average_inference_ms:.1f}ms "
            f"frame={self.average_frame_ms:.1f}ms "
            f"reacquisitions={self.reacquisitions} "
            f"expired={self.expired_tracks}"
        )


class Pipeline:
    """Runs the detection, tracking and target-lock loop."""

    def __init__(
        self,
        config,
        *,
        detector: FaceDetector | None = None,
        source: FrameSource | None = None,
        display: bool = True,
    ) -> None:
        self.config = config
        self.display = display and config.visualization.enabled
        self._owns_detector = detector is None
        self._owns_source = source is None

        self.detector = detector or create_detector(config.detector)
        self.source = source or create_source(config.source)

        self.tracker = Tracker(
            config.tracker,
            config.prediction,
            config.reacquisition,
            identity_config=config.identity,
            fps=config.source.fps,
        )
        self.lock = TargetLock(
            config.target,
            tracker=self.tracker,
            prediction_config=config.prediction,
        )
        self.renderer = Renderer(config.visualization, config.target)

        # The identity layer is optional. When it is on, the selected target is
        # described by a face embedding and can be recognised when it comes
        # back into view. It also names the faces that were enrolled.
        self.identity: IdentityMatcher | None = None
        if config.identity.enabled:
            self.identity = IdentityMatcher(config.identity)
            self.identity.load()
            enrolled = self.identity.load_gallery()
            LOGGER.info("%d enrolled profile(s) available", enrolled)
            if config.target.mode == "identity" and enrolled == 0:
                LOGGER.warning(
                    "target.mode is 'identity' but nobody is enrolled - "
                    "run 'python -m src.enroll --name <name>' first"
                )

        self.frame_index = 0
        self._last_detections: list[Detection] = []
        self._window_name = config.visualization.window_name
        self._writer: cv2.VideoWriter | None = None
        self._running = False
        self._identity_similarity: float | None = None

        if config.target.mode == "track_id" and config.target.track_id > 0:
            # The track may not exist yet - it is selected as soon as it does.
            self._pending_track_id = int(config.target.track_id)
        else:
            self._pending_track_id = 0

    # ------------------------------------------------------------------ #
    def run(self) -> PipelineStats:
        """Process frames until the source is exhausted or the user quits."""
        stats = PipelineStats()
        self._running = True
        start_time = time.perf_counter()
        last_frame_time = start_time
        frame_times: list[float] = []
        inference_times: list[float] = []

        try:
            self._open_recorder()
            while self._running:
                frame, timestamp = self.source.read()
                if frame is None:
                    break

                frame_start = time.perf_counter()
                detections = self._detect(frame)
                identity = self._identity_evidence(frame, detections)
                tracker_result = self.tracker.update(
                    detections, timestamp, self.frame_index, identity=identity
                )
                self._identify_faces(frame, tracker_result.tracks)
                self.lock.update(tracker_result.tracks, timestamp, self.frame_index)
                self._apply_pending_selection(timestamp)
                self._refresh_target_identity(frame, tracker_result.tracks)

                overlay = self._render(frame, tracker_result.tracks, detections)

                frame_times.append((time.perf_counter() - frame_start) * 1000.0)
                if self.detector.stats.history:
                    inference_times.append(self.detector.stats.last_latency_ms)

                stats.frames += 1
                stats.detections += len(detections)
                stats.reacquisitions += len(tracker_result.reacquired_tracks)
                stats.identity_reacquisitions += len(
                    tracker_result.identity_reacquired_tracks
                )
                stats.expired_tracks += len(tracker_result.expired_tracks)
                self.frame_index += 1

                last_frame_time = time.perf_counter()
                if not self._handle_events(overlay):
                    break

                if self.config.runtime.max_frames and self.frame_index >= self.config.runtime.max_frames:
                    break
        finally:
            self._close_recorder()
            self.release()

        stats.total_time_s = time.perf_counter() - start_time
        if frame_times:
            stats.average_frame_ms = sum(frame_times) / len(frame_times)
        if inference_times:
            stats.average_inference_ms = sum(inference_times) / len(inference_times)
        if stats.total_time_s > 0:
            stats.average_fps = stats.frames / stats.total_time_s
        stats.target_events = [str(event) for event in self.lock.events]
        stats.track_events = [
            str(event)
            for track in self.tracker.tracks
            for event in track.events[-3:]
        ]
        return stats

    # ------------------------------------------------------------------ #
    def _apply_pending_selection(self, timestamp: float) -> None:
        """Select the configured track ID as soon as that track exists."""
        if self._pending_track_id:
            if self.lock.select_track_id(
                self._pending_track_id, frame=self.frame_index, timestamp=timestamp
            ):
                self._pending_track_id = 0
            return
        self._select_by_identity(timestamp)

    def _select_by_identity(self, timestamp: float) -> None:
        """Select the face that was enrolled under the configured name.

        The selection follows the **name**, not the track ID: when the person
        comes back after a disappearance the track may have a new ID, but the
        name is the same. The lock is never switched to a different face by
        this - only the face with that name is selected.
        """
        if self.config.target.mode != "identity":
            return
        if self.lock.is_selected:
            return
        if self.lock.state is TargetState.CLEARED:
            # The operator cleared the target on purpose - do not override it.
            return

        name = self.config.target.name.strip()
        if not name:
            return
        for track in self.tracker.tracks:
            if track.identity_name != name:
                continue
            if track.state not in (TrackState.TRACKED, TrackState.REACQUIRED):
                continue
            self.lock.select(track, frame=self.frame_index, timestamp=timestamp)
            LOGGER.info("Target selected by identity: %s (track %d)", name, track.track_id)
            return

    def _detect(self, frame: np.ndarray) -> list[Detection]:
        interval = max(1, int(self.config.runtime.detection_interval))
        if self.frame_index % interval == 0 or not self._last_detections:
            try:
                self._last_detections = self.detector.detect(frame)
            except DetectorError as exc:
                LOGGER.error("Detection failed: %s", exc)
                self._last_detections = []
        return self._last_detections

    # ------------------------------------------------------------------ #
    # identity
    # ------------------------------------------------------------------ #
    def _identify_faces(self, frame: np.ndarray, tracks: Sequence[Track]) -> None:
        """Name the visible faces and collect the poses of every face.

        Every visible face is compared with the enrolled people (the gallery).
        A face that matches an enrolled person gets the name of that person -
        that is the green frame in the overlay.

        The comparison is expensive (one ArcFace embedding per face), so it
        only happens every ``identity.gallery_interval`` frames. The name is
        kept in between, and it survives a face that is turned away as long as
        the similarity stays close to the threshold.
        """
        if self.identity is None or not tracks:
            return
        gallery = self.identity.gallery
        if gallery is None or not len(gallery):
            return

        interval = max(1, int(self.config.identity.gallery_interval))
        if self.frame_index % interval != 0:
            return

        visible = [t for t in tracks if t.state in (TrackState.TRACKED, TrackState.REACQUIRED, TrackState.NEW)]
        if not visible:
            return

        # The target is checked first, it is the one that matters.
        target_id = self.lock.track_id
        visible.sort(
            key=lambda t: (
                t.track_id != target_id,
                -(t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]),
            )
        )

        threshold = float(self.config.identity.threshold)
        for track in visible:
            embedding = self.identity.embed(frame, track.bbox, self.frame_index)
            if embedding is None:
                continue
            self._add_pose(track, embedding)

            name, similarity = self.identity.identify(embedding)
            track.identity_checked = True
            track.identity_similarity = similarity
            if not name:
                track.identity_name = ""
                continue
            if similarity >= threshold:
                track.identity_name = name
            elif (
                track.identity_name == name
                and similarity >= threshold - 0.1
            ):
                # The face is turned away - keep the name it already had.
                pass
            else:
                track.identity_name = ""

    def _add_pose(self, track: Track, embedding: np.ndarray) -> None:
        """Store an observed pose of a face.

        An observation that is not similar to any known pose of the track is
        dropped: it is most likely a different person and must not pollute
        the track.
        """
        track.add_embedding(
            embedding,
            min_similarity=float(self.config.identity.reference_min_similarity),
            max_embeddings=int(self.config.identity.max_embeddings),
        )

    def _identity_evidence(
        self, frame: np.ndarray, detections: Sequence[Detection]
    ) -> IdentityEvidence | None:
        """Compare every candidate face with the reference of the target.

        This is only done while the target is **missing**: when it is visible
        the tracker associates it by position and the embedding is only used
        to keep the reference up to date. That keeps the cost of the identity
        layer low: one embedding per frame while the target is visible, one
        per candidate while it is gone.
        """
        if self.identity is None or not detections:
            return None

        target = self.lock.target_track(self.tracker.tracks)
        if target is None or not target.embeddings:
            return None
        if target.state is not TrackState.LOST:
            return None

        scores: list[float | None] = []
        best: float | None = None
        for detection in detections:
            embedding = self.identity.embed(frame, detection.bbox, self.frame_index)
            if embedding is None:
                scores.append(None)
                continue
            similarity = target.similarity_to(embedding)
            if target.identity_name:
                # The enrolled person is the better reference: the gallery
                # holds the face from every angle that was captured.
                similarity = max(
                    similarity,
                    self.identity.similarity_to(target.identity_name, embedding),
                )
            scores.append(similarity)
            if best is None or similarity > best:
                best = similarity
        self._identity_similarity = best
        return IdentityEvidence(track_id=target.track_id, scores=scores)

    def _refresh_target_identity(self, frame: np.ndarray, tracks: Sequence[Track]) -> None:
        """Keep the reference embedding of the target up to date."""
        if self.identity is None:
            return

        target = self.lock.target_track(tracks)
        if target is None:
            return
        if target.state not in (TrackState.TRACKED, TrackState.REACQUIRED, TrackState.NEW):
            return

        if target.embeddings:
            interval = max(1, int(self.config.identity.reference_interval))
            if self.frame_index % interval != 0:
                return

        embedding = self.identity.embed(frame, target.bbox, self.frame_index)
        if embedding is None:
            return

        self._add_pose(target, embedding)
        self._identity_similarity = 1.0

    def _render(
        self, frame: np.ndarray, tracks: Sequence[Track], detections: Sequence[Detection]
    ) -> np.ndarray:
        visible = [t for t in tracks if t.state in (TrackState.TRACKED, TrackState.REACQUIRED, TrackState.NEW)]
        lost = [t for t in tracks if t.state is TrackState.LOST]
        stats = DashboardStats(
            fps=self._current_fps(),
            inference_ms=self.detector.stats.last_latency_ms,
            visible_tracks=len(visible),
            lost_tracks=len(lost),
            detections=len(detections),
            backend=getattr(self.detector, "name", "detector"),
        )
        if self.identity is not None:
            similarity = self._identity_similarity
            stats.extra["identity"] = (
                f"identity: {similarity:.2f}" if similarity is not None else "identity: -"
            )
        overlay = self.renderer.draw(frame, tracks, self.lock, stats)
        self._write_frame(overlay)
        return overlay

    def _current_fps(self) -> float:
        if self.frame_index <= 1:
            return 0.0
        return self.source.fps

    # ------------------------------------------------------------------ #
    def _handle_events(self, overlay: np.ndarray) -> bool:
        """Process the window events. Returns ``False`` when the user quits."""
        if self.display:
            cv2.imshow(self._window_name, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in KEY_QUIT:
                LOGGER.info("Quit requested")
                return False
            if key == KEY_SELECT_LARGEST:
                self._select_largest()
            elif key == KEY_CLEAR:
                self.lock.clear(frame=self.frame_index, timestamp=time.time())
                LOGGER.info("Target cleared")
            try:
                cv2.setMouseCallback(self._window_name, self._on_mouse)
            except cv2.error:  # pragma: no cover - headless OpenCV builds
                pass
        return True

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """Select the track that was clicked on."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        tracks = self.tracker.tracks
        best: Track | None = None
        best_area = 0.0
        for track in tracks:
            if track.state is TrackState.LOST:
                continue
            x1, y1, x2, y2 = track.bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best = track
                    best_area = area
        if best is not None:
            self.lock.select(best, frame=self.frame_index, timestamp=time.time())
            LOGGER.info("Target selected by click: track %d", best.track_id)
        else:
            self.lock.clear(frame=self.frame_index, timestamp=time.time())
            LOGGER.info("No track under the cursor - target cleared")

    def _select_largest(self) -> None:
        candidates = [
            t
            for t in self.tracker.tracks
            if t.state in (TrackState.TRACKED, TrackState.REACQUIRED)
        ]
        if not candidates:
            LOGGER.info("No visible face to select")
            return
        best = max(candidates, key=lambda t: t.bbox[2] * t.bbox[3])
        self.lock.select(best, frame=self.frame_index, timestamp=time.time())
        LOGGER.info("Target selected: track %d (largest face)", best.track_id)

    # ------------------------------------------------------------------ #
    def _open_recorder(self) -> None:
        path = self.config.runtime.record_output
        if not path:
            return
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        size = self.source.size
        if not size or size[0] <= 0 or size[1] <= 0:
            LOGGER.warning("Cannot record: the source has no known frame size")
            return
        fps = self.source.fps or 30.0
        self._writer = cv2.VideoWriter(
            str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
        )
        if not self._writer.isOpened():
            LOGGER.warning("Cannot open the recording file %s", target)
            self._writer.release()
            self._writer = None
            return
        LOGGER.info("Recording to %s", target)

    def _write_frame(self, overlay: np.ndarray) -> None:
        if self._writer is None:
            return
        try:
            self._writer.write(overlay)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Could not write a frame: %s", exc)

    def _close_recorder(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    # ------------------------------------------------------------------ #
    def release(self) -> None:
        """Release every resource the pipeline owns."""
        if self.display:
            try:
                cv2.destroyWindow(self._window_name)
            except cv2.error:  # pragma: no cover - headless OpenCV builds
                pass
        if self._owns_source:
            self.source.release()
        if self._owns_detector:
            try:
                self.detector.close()
            except Exception:  # pragma: no cover - defensive
                pass

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
