"""Live visualisation.

The overlay answers the questions an operator actually has:

* where are the faces right now,
* which of them is the selected target,
* what is the state of that target (visible / lost / reacquired / expired),
* where was it last seen and how long ago,
* where is it *predicted* to be while it is missing,
* how fast is the system running.

The visual language is deliberate:

===============  ==========================================================
Element          Style
===============  ==========================================================
Enrolled person  thick bright green box, label ``name 0.87``
Selected target  thick magenta box, label ``TARGET #id name [STATE]``
Observation      thin light box, label ``#id``
Prediction       dashed yellow box, label ``predicted``
Lost track       thin grey box, label ``#id lost``
===============  ==========================================================

A prediction is drawn differently from an observation on purpose: a prediction
is an estimate of where the face *might* be, never a proof that it is still
there. The dashboard states the age of the last observation.

The **green** box is reserved for the faces that were enrolled: it is the
answer to "is this the person I am supposed to keep an eye on?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from ..tracking.target_lock import TargetLock, TargetState
from ..tracking.track_state import Track, TrackState

#: BGR colours.
COLOR_IDENTIFIED = (0, 255, 0)
COLOR_OBSERVED = (200, 200, 200)
COLOR_TARGET = (0, 140, 255)
COLOR_PREDICTION = (0, 220, 255)
COLOR_LOST = (140, 140, 140)
COLOR_TEXT = (255, 255, 255)
COLOR_PANEL = (32, 32, 32)

FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class DashboardStats:
    """The numbers shown in the corner of the frame."""

    fps: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0
    visible_tracks: int = 0
    lost_tracks: int = 0
    detections: int = 0
    backend: str = ""
    extra: dict[str, str] = field(default_factory=dict)


class Renderer:
    """Draws the tracking state on top of the video frame."""

    def __init__(self, visualization_config, target_config=None) -> None:
        self.config = visualization_config
        self.target_config = target_config

    # ------------------------------------------------------------------ #
    def draw(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        lock: TargetLock,
        stats: DashboardStats,
    ) -> np.ndarray:
        """Render one frame. The input frame is not modified."""
        canvas = frame.copy()
        target_id = lock.track_id

        for track in tracks:
            if track.track_id == target_id:
                continue
            self._draw_track(canvas, track, is_target=False)

        target = lock.target_track(tracks)
        if target is not None:
            self._draw_track(canvas, target, is_target=True)
            self._draw_target_details(canvas, target, lock)

        if self.config.show_dashboard:
            self._draw_dashboard(canvas, lock, stats)
        if self.config.show_predictions:
            self._draw_predictions(canvas, tracks, target_id)

        return canvas

    # ------------------------------------------------------------------ #
    def _draw_track(self, canvas: np.ndarray, track: Track, *, is_target: bool) -> None:
        x1, y1, x2, y2 = (int(round(v)) for v in track.bbox)
        visible = track.state in (TrackState.TRACKED, TrackState.REACQUIRED, TrackState.NEW)

        if visible and track.identity_name:
            # An enrolled person: the green frame. It is the answer to "is
            # this the person I am supposed to keep an eye on?" - it does not
            # matter whether that person is also the selected target.
            colour = COLOR_IDENTIFIED
            thickness = 3
        elif is_target:
            colour = COLOR_TARGET
            thickness = 3
        elif visible:
            colour = COLOR_OBSERVED
            thickness = 2
        else:
            colour = COLOR_LOST
            thickness = 1

        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)

        if is_target:
            label = f"TARGET #{track.track_id}"
            if track.identity_name:
                label += f" {track.identity_name}"
            label += f" [{track.state.value.upper()}]"
        elif visible and track.identity_name:
            label = f"{track.identity_name} {track.identity_similarity:.2f}"
        elif visible:
            label = f"#{track.track_id}"
        else:
            label = f"#{track.track_id} lost"

        _put_label(canvas, label, (x1, max(0, y1 - 6)), colour)

    def _draw_target_details(
        self, canvas: np.ndarray, track: Track, lock: TargetLock
    ) -> None:
        """Show where the target was last seen and how long ago that was."""
        x1, y1, _x2, _y2 = (int(round(v)) for v in track.bbox)
        age = track.time_since_seen
        text = f"last seen {age:.1f}s ago"
        if track.state is TrackState.LOST:
            text += " - searching"
        _put_label(canvas, text, (x1, max(0, y1 - 22)), COLOR_TEXT)

    def _draw_predictions(
        self, canvas: np.ndarray, tracks: Sequence[Track], target_id: int | None
    ) -> None:
        """Draw the predicted position of the tracks that are currently lost."""
        for track in tracks:
            if track.state is not TrackState.LOST:
                continue
            if track.time_since_seen > 1.0:
                # The prediction is not trusted any more.
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in track.predicted_bbox)
            _dashed_rectangle(canvas, (x1, y1), (x2, y2), COLOR_PREDICTION, 2)
            prefix = "TARGET predicted" if track.track_id == target_id else "predicted"
            _put_label(canvas, prefix, (x1, max(0, y1 - 6)), COLOR_PREDICTION)

    def _draw_dashboard(
        self, canvas: np.ndarray, lock: TargetLock, stats: DashboardStats
    ) -> None:
        """Draw the status panel."""
        lines = [
            f"backend: {stats.backend}" if stats.backend else "",
            f"fps: {stats.fps:.1f}   inference: {stats.inference_ms:.1f} ms",
            f"tracks: {stats.visible_tracks} visible / {stats.lost_tracks} lost",
            f"target: {lock.state.value.upper()}"
            + (f"  #{lock.track_id}" if lock.track_id is not None else ""),
        ]
        lines.extend(stats.extra.values())
        lines = [line for line in lines if line]

        height = 20 * len(lines) + 16
        width = 300
        overlay = canvas.copy()
        cv2.rectangle(overlay, (8, 8), (8 + width, 8 + height), COLOR_PANEL, -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        for index, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (18, 30 + 20 * index),
                FONT,
                0.5,
                COLOR_TEXT,
                1,
                cv2.LINE_AA,
            )

        _put_label(
            canvas,
            "click: select target | t: largest | c: clear | q/ESC: quit",
            (18, canvas.shape[0] - 14),
            COLOR_TEXT,
        )


def _put_label(
    canvas: np.ndarray, text: str, origin: tuple[int, int], colour: tuple[int, int, int]
) -> None:
    """Draw a text with a dark background so that it stays readable."""
    x, y = origin
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, 0.45, 1)
    y = max(text_h + 2, y)
    cv2.rectangle(
        canvas,
        (x, y - text_h - 4),
        (x + text_w + 4, y + baseline),
        COLOR_PANEL,
        -1,
    )
    cv2.putText(canvas, text, (x + 2, y - 2), FONT, 0.45, colour, 1, cv2.LINE_AA)


def _dashed_rectangle(
    canvas: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    colour: tuple[int, int, int],
    thickness: int = 2,
    dash: int = 8,
) -> None:
    """Draw a dashed rectangle (used for predictions)."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    for x in range(x1, x2, dash * 2):
        cv2.line(canvas, (x, y1), (min(x + dash, x2), y1), colour, thickness)
        cv2.line(canvas, (x, y2), (min(x + dash, x2), y2), colour, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(canvas, (x1, y), (x1, min(y + dash, y2)), colour, thickness)
        cv2.line(canvas, (x2, y), (x2, min(y + dash, y2)), colour, thickness)
