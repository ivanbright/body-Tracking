"""Enrol a person into the face gallery.

This is the "training" step of the system: the faces that the camera is
supposed to keep an eye on are captured, aligned, embedded and stored. Later
the tracker recognises them, draws a green frame around them and - if one of
them is the selected target - keeps their track ID alive.

The workflow follows the working reference implementation of the parent
project:

    python -m src.enroll --name Ivan

    SPACE = capture a sample    s = save    q = quit

**Capture the face from several angles.** A sample that was captured while
the head was turned to the side is what makes the recognition survive the
person turning their head later: the similarity is taken as the maximum over
all the samples of a profile, so a turned face matches the turned sample
instead of the average of all of them.

Non-interactive variants::

    python -m src.enroll --name Ivan --image photo.jpg
    python -m src.enroll --name Ivan --video clip.mp4 --max-samples 30
    python -m src.enroll --list
    python -m src.enroll --remove Ivan
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from .camera.capture import SourceError, create_source
from .config import ConfigError, SourceConfig, load_config
from .detection.haar import HaarFaceDetector
from .recognition.alignment import FaceAligner
from .recognition.arcface import ArcFaceEncoder, IdentityError
from .recognition.gallery import Gallery
from .recognition.identity import IdentityMatcher

LOGGER = logging.getLogger(__name__)

#: Two samples that are more similar than this are considered the same pose.
DIVERSITY_LIMIT = 0.92

#: The similarity that a new sample must reach to be accepted as the same
#: person (a different pose of the same face is usually well below that).
SAME_PERSON_LIMIT = 0.45


class EnrolmentSession:
    """Collects the embeddings of one person."""

    def __init__(self, matcher: IdentityMatcher, detector: HaarFaceDetector) -> None:
        self.matcher = matcher
        self.detector = detector
        self.samples: list[np.ndarray] = []

    # ------------------------------------------------------------------ #
    def add(self, embedding: np.ndarray, diversity_limit: float = DIVERSITY_LIMIT) -> tuple[bool, float]:
        """Add a sample, but only when it shows a new appearance.

        ``diversity_limit`` is the similarity above which a sample is
        considered to be the same appearance as an existing one. The
        interactive capture uses it to ask for a **different angle**; a video
        is sampled with a limit of 1.0 (accept everything) because the
        recording already contains the movement.
        """
        best = 0.0
        for existing in self.samples:
            value = float(np.dot(existing, embedding))
            best = max(best, value)
        if best > diversity_limit:
            return False, best
        self.samples.append(embedding)
        return True, best

    def embedding_for(self, frame: np.ndarray, frame_index: int = -1) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Embed the largest face of the frame."""
        detections = self.detector.detect(frame)
        if not detections:
            return None, None
        box = detections[0].bbox
        aligner = self.matcher.aligner
        encoder = self.matcher.encoder
        if aligner is None or encoder is None:
            return None, None
        landmarks = aligner._landmarker.landmarks(frame) if aligner.uses_landmarks else []
        aligned = aligner.align(frame, box, landmarks)
        if aligned is None:
            return None, None
        return encoder.embed(aligned).embedding, aligned

    def landmarks_for(self, frame: np.ndarray) -> list[np.ndarray]:
        aligner = self.matcher.aligner
        if aligner is None or not aligner.uses_landmarks:
            return []
        return aligner._landmarker.landmarks(frame)


def build_matcher(config) -> IdentityMatcher:
    """Create the matcher and load the models."""
    matcher = IdentityMatcher(config.identity)
    matcher.load()
    return matcher


def enrol_interactively(
    matcher: IdentityMatcher,
    detector: HaarFaceDetector,
    name: str,
    camera_index: int,
    width: int,
    height: int,
) -> list[np.ndarray]:
    """Capture the samples from a live camera (SPACE = capture)."""
    source = create_source(
        SourceConfig(
            type="camera",
            camera_index=camera_index,
            width=width,
            height=height,
            fps=30.0,
        )
    )
    session = EnrolmentSession(matcher, detector)
    window = "ArcFace Enrolment"

    print(f"\n[CAPTURE] Enrolling: {name}")
    print("[CAPTURE] Look at the camera, then turn your head left and right.")
    print("[CAPTURE] SPACE = capture | s = save | q = quit")

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            frame, _ = source.read()
            if frame is None:
                break

            display = frame.copy()
            embedding, aligned = session.embedding_for(frame)

            if aligned is not None:
                for points in session.landmarks_for(frame):
                    for x, y in points.astype(int):
                        cv2.circle(display, (int(x), int(y)), 3, (0, 255, 0), -1)
                cv2.putText(display, "FACE LOCKED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "NO FACE - POSITION THE FACE", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(display, f"Captured: {len(session.samples)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "SPACE=capture | s=save | q=quit",
                        (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and embedding is not None:
                accepted, similarity = session.add(embedding)
                if accepted:
                    print(f"[CAPTURE] sample {len(session.samples)} stored")
                else:
                    print(f"[CAPTURE] skipped - too similar to an existing sample "
                          f"({similarity:.2f}), try another angle")
                continue
            if key == ord("s") and session.samples:
                return session.samples
            if key == ord("q"):
                return session.samples
    finally:
        source.release()
        cv2.destroyAllWindows()


def enrol_from_file(
    matcher: IdentityMatcher,
    detector: HaarFaceDetector,
    path: str,
    max_samples: int,
    from_video: bool,
) -> list[np.ndarray]:
    """Capture the samples from a video file or a single image."""
    session = EnrolmentSession(matcher, detector)
    source = create_source(
        SourceConfig(
            type="video",
            video_path=path,
            fps=30.0,
        )
    )
    try:
        frame_index = 0
        while len(session.samples) < max_samples:
            frame, _ = source.read()
            if frame is None:
                break
            embedding, _ = session.embedding_for(frame, frame_index)
            # Every third frame: the recording is continuous, the samples would
            # be the same picture over and over.
            take = frame_index % 3 == 0
            frame_index += 1
            if embedding is None:
                continue
            if take:
                session.add(embedding, diversity_limit=1.0)
            if not from_video:
                break
    finally:
        source.release()
    return session.samples


def report(gallery: Gallery, name: str) -> None:
    """Print the numbers that matter for the threshold."""
    profile = gallery.get(name)
    if profile is None:
        return
    intra = profile.intra_similarity()
    print(f"\n[OK] {name} enrolled with {len(profile.samples)} samples")
    if intra:
        print(f"     similarity of the samples with the template: "
              f"min {min(intra):.2f}, max {max(intra):.2f}")
    other, other_similarity = gallery.closest_other(name)
    if other:
        print(f"     closest other enrolled person: {other} at {other_similarity:.2f}")
        print(f"     -> a threshold of "
              f"{max(gallery_threshold, other_similarity + 0.1):.2f} separates them")
    else:
        print(f"     -> no other person is enrolled yet, the threshold stays at "
              f"{gallery_threshold:.2f}")
    print("     Run the tracker with --target-mode identity --target-name "
          f"\"{name}\" to follow this person.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.enroll",
        description="Enrol people into the face gallery.",
    )
    parser.add_argument("--config", default="", help="path to the YAML configuration")
    parser.add_argument("--name", default="", help="the name of the person to enrol")
    parser.add_argument("--image", default="", help="enrol from a single image")
    parser.add_argument("--video", default="", help="enrol from a video file")
    parser.add_argument("--camera-index", type=int, default=1, help="camera for the capture")
    parser.add_argument("--max-samples", type=int, default=30, help="samples to capture")
    parser.add_argument("--list", action="store_true", help="list the enrolled people")
    parser.add_argument("--remove", default="", help="remove a person from the gallery")
    args = parser.parse_args(argv)

    global gallery_threshold

    try:
        config = load_config(args.config or None)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    gallery_threshold = config.identity.threshold

    gallery = Gallery(config.identity.gallery_path)
    gallery.load()

    if args.list:
        if not gallery.names:
            print("Nobody is enrolled yet.")
        for name in gallery.names:
            profile = gallery.get(name)
            assert profile is not None
            print(f"  {name}: {len(profile.samples)} samples "
                  f"(created {profile.created}, from {profile.source or 'unknown'})")
        return 0

    if args.remove:
        if gallery.remove(args.remove):
            gallery.save()
            print(f"Removed {args.remove}")
            return 0
        print(f"{args.remove} is not enrolled", file=sys.stderr)
        return 1

    if not args.name:
        parser.error("--name is required")

    try:
        matcher = build_matcher(config)
    except IdentityError as exc:
        print(f"Identity error: {exc}", file=sys.stderr)
        return 5

    detector = HaarFaceDetector(min_size=config.detector.haar_min_size)

    if args.image:
        samples = enrol_from_file(matcher, detector, args.image, 1, from_video=False)
        source = args.image
    elif args.video:
        samples = enrol_from_file(matcher, detector, args.video, args.max_samples, from_video=True)
        source = args.video
    else:
        samples = enrol_interactively(
            matcher,
            detector,
            args.name,
            args.camera_index,
            config.source.width,
            config.source.height,
        )
        source = f"camera:{args.camera_index}"

    if not samples:
        print("No samples were captured - nothing was enrolled.", file=sys.stderr)
        return 1

    gallery.add(args.name, samples, source=source)
    gallery.save()
    report(gallery, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
