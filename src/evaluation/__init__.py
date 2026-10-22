"""Evaluation of the tracking behaviour.

Two kinds of measurement live here and they are never mixed:

* :mod:`src.evaluation.metrics` computes tracking metrics (IDF1, HOTA, ID
  switches, fragmentation, reacquisition) from annotations.
* :mod:`src.evaluation.benchmark` runs the system and reports what was
  measured, including *where* the data came from.
"""

from __future__ import annotations

from .benchmark import BenchmarkResult, load_annotations, run_synthetic_benchmark, run_video_benchmark, write_report
from .metrics import Annotation, TrackingMetrics, annotations_from_tracks, evaluate

__all__ = [
    "Annotation",
    "BenchmarkResult",
    "TrackingMetrics",
    "annotations_from_tracks",
    "evaluate",
    "load_annotations",
    "run_synthetic_benchmark",
    "run_video_benchmark",
    "write_report",
]
