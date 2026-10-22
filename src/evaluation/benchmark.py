"""The benchmark.

The benchmark reports **measured** numbers and nothing else. When no ground
truth is available it says so instead of inventing metrics, and every report
carries the label of its data source:

``synthetic``
    The detections were generated in code. The tracking logic is verified, but
    the numbers say nothing about the quality of a real detector.
``recorded-video``
    A real video file was processed. The runtime numbers are real; the
    tracking metrics are only real when an annotation file is supplied.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import AppConfig
from ..detection.base import Detection
from ..tracking.target_lock import TargetLock
from ..tracking.track_state import TrackState
from ..tracking.tracker import Tracker
from .metrics import Annotation, TrackingMetrics, evaluate

LOGGER = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Everything the benchmark measured."""

    source: str = "synthetic"
    frames: int = 0
    detections: int = 0
    duration_s: float = 0.0
    fps: float = 0.0
    average_inference_ms: float = 0.0
    min_inference_ms: float = 0.0
    max_inference_ms: float = 0.0
    p95_inference_ms: float = 0.0
    average_track_ms: float = 0.0
    tracks_created: int = 0
    tracks_expired: int = 0
    reacquisitions: int = 0
    target_events: list[str] = field(default_factory=list)
    metrics: TrackingMetrics | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = {
            "source": self.source,
            "frames": self.frames,
            "detections": self.detections,
            "duration_s": round(self.duration_s, 3),
            "fps": round(self.fps, 2),
            "inference_ms": {
                "average": round(self.average_inference_ms, 3),
                "min": round(self.min_inference_ms, 3),
                "max": round(self.max_inference_ms, 3),
                "p95": round(self.p95_inference_ms, 3),
            },
            "average_track_ms": round(self.average_track_ms, 3),
            "tracks_created": self.tracks_created,
            "tracks_expired": self.tracks_expired,
            "reacquisitions": self.reacquisitions,
            "target_events": self.target_events,
            "notes": self.notes,
        }
        if self.metrics is not None:
            data["metrics"] = {
                "idf1": round(self.metrics.idf1, 4),
                "hota": round(self.metrics.hota, 4),
                "id_switches": self.metrics.id_switches,
                "track_fragmentation": self.metrics.track_fragmentation,
                "incorrect_associations": self.metrics.incorrect_associations,
                "precision": round(self.metrics.precision, 4),
                "recall": round(self.metrics.recall, 4),
                "reacquisition_success": round(self.metrics.reacquisition_success, 4),
                "reacquisition_attempts": self.metrics.reacquisition_attempts,
            }
        return data

    def summary(self) -> str:
        lines = [
            f"source           : {self.source}",
            f"frames           : {self.frames}",
            f"detections       : {self.detections}",
            f"duration         : {self.duration_s:.2f} s ({self.fps:.1f} fps)",
            f"inference        : avg {self.average_inference_ms:.2f} ms | "
            f"min {self.min_inference_ms:.2f} | max {self.max_inference_ms:.2f} | "
            f"p95 {self.p95_inference_ms:.2f}",
            f"tracking         : avg {self.average_track_ms:.2f} ms/frame",
            f"tracks created   : {self.tracks_created}",
            f"tracks expired   : {self.tracks_expired}",
            f"reacquisitions   : {self.reacquisitions}",
        ]
        if self.metrics is not None:
            lines.append(f"metrics          : {self.metrics.summary()}")
        else:
            lines.append("metrics          : not available (no ground truth was supplied)")
        for note in self.notes:
            lines.append(f"note             : {note}")
        return "\n".join(lines)


def run_synthetic_benchmark(
    config: AppConfig,
    *,
    frames: int = 300,
    seed: int = 7,
) -> BenchmarkResult:
    """Run the tracker over a synthetic sequence and measure it.

    The sequence contains two faces that move, cross and disappear. It is
    generated deterministically, so the result can be compared between runs.
    """
    rng = np.random.default_rng(seed)
    tracker = Tracker(config.tracker, config.prediction, config.reacquisition, fps=30.0)
    lock = TargetLock(config.target, tracker=tracker, prediction_config=config.prediction)

    detections_per_frame: list[list[Detection]] = []
    ground_truth: list[Annotation] = []

    for frame in range(frames):
        timestamp = frame / 30.0
        detections: list[Detection] = []

        # Face A moves to the right and disappears between frame 120 and 150.
        if not (120 <= frame < 150):
            x = 60.0 + frame * 1.2
            y = 120.0 + 8.0 * np.sin(frame / 12.0)
            box = (x, y, x + 48, y + 48)
            detections.append(Detection(bbox=box, score=0.9, class_name="face"))
            ground_truth.append(Annotation(frame=frame, object_id=1, bbox=box))

        # Face B moves to the left and crosses the path of face A.
        x = 320.0 - frame * 0.9
        y = 200.0 + 6.0 * np.cos(frame / 15.0)
        box = (x, y, x + 40, y + 40)
        detections.append(Detection(bbox=box, score=0.8, class_name="face"))
        ground_truth.append(Annotation(frame=frame, object_id=2, bbox=box))

        # A third, short lived face.
        if 200 <= frame < 230:
            box = (400.0, 60.0, 440.0, 100.0)
            detections.append(Detection(bbox=box, score=0.7, class_name="face"))
            ground_truth.append(Annotation(frame=frame, object_id=3, bbox=box))

        detections_per_frame.append(detections)

    inference_times: list[float] = []
    track_times: list[float] = []
    predictions: list[Annotation] = []
    result = BenchmarkResult(source="synthetic")

    start = time.perf_counter()
    for frame, detections in enumerate(detections_per_frame):
        timestamp = frame / 30.0
        track_start = time.perf_counter()
        tracker_result = tracker.update(detections, timestamp, frame)
        lock.update(tracker_result.tracks, timestamp, frame)
        track_times.append((time.perf_counter() - track_start) * 1000.0)

        for track in tracker_result.tracks:
            if track.state in (TrackState.TRACKED, TrackState.REACQUIRED):
                predictions.append(
                    Annotation(
                        frame=frame,
                        object_id=track.track_id,
                        bbox=tuple(float(v) for v in track.bbox),
                        score=float(track.score),
                    )
                )
        result.tracks_created += len(tracker_result.new_tracks)
        result.tracks_expired += len(tracker_result.expired_tracks)
        result.reacquisitions += len(tracker_result.reacquired_tracks)
    duration = time.perf_counter() - start

    result.frames = frames
    result.detections = sum(len(d) for d in detections_per_frame)
    result.duration_s = duration
    result.fps = frames / duration if duration > 0 else 0.0
    result.average_track_ms = float(np.mean(track_times)) if track_times else 0.0
    result.target_events = [str(event) for event in lock.events]
    result.notes.append(
        "The detections were generated in code: the numbers verify the tracking "
        "logic, they do not measure the quality of a detector."
    )

    metrics = evaluate(ground_truth, predictions, frames=frames)
    result.metrics = metrics
    return result


def run_video_benchmark(
    config: AppConfig,
    video_path: str | Path,
    *,
    annotations_path: str | Path | None = None,
    max_frames: int = 0,
) -> BenchmarkResult:
    """Process a real video file and measure the runtime.

    The tracking metrics are only reported when an annotation file is
    supplied; otherwise the result explicitly says that no ground truth was
    available.
    """
    from ..camera.capture import VideoSource
    from ..detection import create_detector

    video_path = Path(video_path)
    result = BenchmarkResult(source="recorded-video")
    result.notes.append(
        "Runtime numbers are measured on a real video. The detection quality "
        "depends on the model that is installed."
    )

    tracker = Tracker(config.tracker, config.prediction, config.reacquisition, fps=30.0)
    lock = TargetLock(config.target, tracker=tracker, prediction_config=config.prediction)

    source = VideoSource(video_path, fps=config.source.fps, loop=False)
    detector = create_detector(config.detector)

    predictions: list[Annotation] = []
    inference_times: list[float] = []
    track_times: list[float] = []
    frame_index = 0
    start = time.perf_counter()

    try:
        while True:
            frame, timestamp = source.read()
            if frame is None:
                break
            if max_frames and frame_index >= max_frames:
                break

            detect_start = time.perf_counter()
            detections = detector.detect(frame)
            inference_times.append((time.perf_counter() - detect_start) * 1000.0)

            track_start = time.perf_counter()
            tracker_result = tracker.update(detections, timestamp, frame_index)
            lock.update(tracker_result.tracks, timestamp, frame_index)
            track_times.append((time.perf_counter() - track_start) * 1000.0)

            for track in tracker_result.tracks:
                if track.state in (TrackState.TRACKED, TrackState.REACQUIRED):
                    predictions.append(
                        Annotation(
                            frame=frame_index,
                            object_id=track.track_id,
                            bbox=tuple(float(v) for v in track.bbox),
                            score=float(track.score),
                        )
                    )
            result.tracks_created += len(tracker_result.new_tracks)
            result.tracks_expired += len(tracker_result.expired_tracks)
            result.reacquisitions += len(tracker_result.reacquired_tracks)
            frame_index += 1
    finally:
        source.release()
        detector.close()

    duration = time.perf_counter() - start
    result.frames = frame_index
    result.duration_s = duration
    result.fps = frame_index / duration if duration > 0 else 0.0
    if inference_times:
        result.average_inference_ms = float(np.mean(inference_times))
        result.min_inference_ms = float(np.min(inference_times))
        result.max_inference_ms = float(np.max(inference_times))
        result.p95_inference_ms = float(np.percentile(inference_times, 95))
    result.average_track_ms = float(np.mean(track_times)) if track_times else 0.0
    result.target_events = [str(event) for event in lock.events]

    if annotations_path:
        ground_truth = load_annotations(annotations_path)
        result.metrics = evaluate(ground_truth, predictions, frames=frame_index)
        result.notes.append(
            "The metrics were computed against the supplied ground-truth annotation."
        )
    else:
        result.notes.append(
            "No ground-truth annotation was supplied, so no IDF1/HOTA could be "
            "computed. The runtime numbers are measured; the tracking quality "
            "is not evaluated."
        )
    return result


def load_annotations(path: str | Path) -> list[Annotation]:
    """Load annotations from a JSON file.

    Expected format::

        {"annotations": [
            {"frame": 0, "object_id": 1, "bbox": [x1, y1, x2, y2]},
            ...
        ]}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data.get("annotations", data) if isinstance(data, dict) else data
    annotations: list[Annotation] = []
    for entry in entries:
        annotations.append(
            Annotation(
                frame=int(entry["frame"]),
                object_id=int(entry["object_id"]),
                bbox=tuple(float(v) for v in entry["bbox"]),
                score=float(entry.get("score", 1.0)),
            )
        )
    return annotations


def write_report(result: BenchmarkResult, path: str | Path) -> None:
    """Write the benchmark result as JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    LOGGER.info("Benchmark report written to %s", target)
