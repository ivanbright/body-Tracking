"""The gallery of enrolled people.

This is the equivalent of the ``data/face_database.pkl`` of the working
reference system, with two differences:

* it is stored as **JSON** instead of a pickle - it can be inspected, diffed
  and it cannot execute code when it is loaded,
* a profile keeps **all** the samples that were captured, not only their mean.

Why keep the samples? Because the mean template answers "what does this
person look like on average", while the individual samples answer "what does
this person look like *from this angle*". A face that turns to the side is
much closer to one of the samples than to the average of all of them, so the
similarity is taken as the **maximum** over the template and the samples.

The reference implementation of the parent project uses the mean template and
a cosine threshold of 0.6; that is kept as the default here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .arcface import cosine_similarity

LOGGER = logging.getLogger(__name__)

#: Version of the file format. Bumped when the layout changes.
GALLERY_VERSION = 1


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


@dataclass
class IdentityProfile:
    """One enrolled person."""

    name: str
    #: The individual aligned samples that were captured.
    samples: list[np.ndarray] = field(default_factory=list)
    #: The mean of the samples, L2-normalized (the "template").
    template: np.ndarray | None = None
    created: str = ""
    source: str = ""

    # ------------------------------------------------------------------ #
    @classmethod
    def from_samples(
        cls,
        name: str,
        samples: Iterable[np.ndarray],
        source: str = "",
    ) -> "IdentityProfile":
        """Build a profile from the captured embeddings."""
        vectors = [_normalize(np.asarray(sample, dtype=np.float32)) for sample in samples]
        vectors = [vector for vector in vectors if vector.size]
        if not vectors:
            raise ValueError("an enrolment needs at least one embedding")

        stacked = np.stack(vectors, axis=0)
        template = _normalize(stacked.mean(axis=0))
        return cls(
            name=name,
            samples=vectors,
            template=template,
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source=source,
        )

    # ------------------------------------------------------------------ #
    def similarity(self, embedding: np.ndarray | None) -> float:
        """The best similarity of ``embedding`` against this profile.

        The maximum is taken over the template and over every sample, so a
        face that is turned to the side still matches the sample that was
        captured from that angle.
        """
        if embedding is None or not self.samples:
            return 0.0
        candidate = _normalize(np.asarray(embedding, dtype=np.float32))
        best = cosine_similarity(self.template, candidate) if self.template is not None else 0.0
        for sample in self.samples:
            value = cosine_similarity(sample, candidate)
            if value > best:
                best = value
        return float(best)

    def intra_similarity(self) -> list[float]:
        """Similarity of every sample with the template."""
        if self.template is None:
            return []
        return [round(cosine_similarity(self.template, sample), 4) for sample in self.samples]

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        template = self.template if self.template is not None else np.zeros(0)
        return {
            "created": self.created,
            "source": self.source,
            "template": [round(float(v), 6) for v in template],
            "samples": [[round(float(v), 6) for v in sample] for sample in self.samples],
        }

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "IdentityProfile":
        template = data.get("template") or []
        samples = data.get("samples") or []
        return cls(
            name=name,
            samples=[_normalize(np.asarray(sample, dtype=np.float32)) for sample in samples],
            template=_normalize(np.asarray(template, dtype=np.float32)) if template else None,
            created=str(data.get("created", "")),
            source=str(data.get("source", "")),
        )


class Gallery:
    """The database of enrolled people."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.profiles: dict[str, IdentityProfile] = {}

    # ------------------------------------------------------------------ #
    @property
    def names(self) -> list[str]:
        return sorted(self.profiles)

    def __len__(self) -> int:
        return len(self.profiles)

    def __contains__(self, name: object) -> bool:
        return str(name) in self.profiles

    def get(self, name: str) -> IdentityProfile | None:
        return self.profiles.get(name)

    # ------------------------------------------------------------------ #
    def load(self) -> int:
        """Load the gallery. Returns the number of profiles."""
        self.profiles = {}
        if not self.path.exists():
            LOGGER.info("No gallery found at %s", self.path)
            return 0
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Could not read the gallery %s: %s", self.path, exc)
            return 0

        for name, profile_data in (data.get("profiles") or {}).items():
            try:
                self.profiles[name] = IdentityProfile.from_dict(name, profile_data)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Skipping the broken profile %r: %s", name, exc)
        LOGGER.info("Loaded %d enrolled profiles from %s", len(self.profiles), self.path)
        return len(self.profiles)

    def save(self) -> None:
        """Write the gallery to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": GALLERY_VERSION,
            "profiles": {name: profile.to_dict() for name, profile in self.profiles.items()},
        }
        self.path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        LOGGER.info("Saved %d profiles to %s", len(self.profiles), self.path)

    # ------------------------------------------------------------------ #
    def add(
        self,
        name: str,
        samples: Sequence[np.ndarray],
        source: str = "",
    ) -> IdentityProfile:
        """Enrol (or re-enrol) a person."""
        profile = IdentityProfile.from_samples(name, samples, source=source)
        self.profiles[name] = profile
        return profile

    def remove(self, name: str) -> bool:
        return self.profiles.pop(name, None) is not None

    # ------------------------------------------------------------------ #
    def best_match(self, embedding: np.ndarray | None) -> tuple[str | None, float]:
        """The closest enrolled person and the similarity.

        The returned similarity is always the best one - even when it is below
        the threshold - so the caller can show "Unknown (0.21)".
        """
        if embedding is None or not self.profiles:
            return None, 0.0
        best_name: str | None = None
        best_similarity = -1.0
        for name, profile in self.profiles.items():
            similarity = profile.similarity(embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_name = name
        return best_name, float(max(0.0, best_similarity))

    def similarity(self, name: str, embedding: np.ndarray | None) -> float:
        """The similarity of a face with one specific enrolled person."""
        profile = self.profiles.get(name)
        if profile is None:
            return 0.0
        return profile.similarity(embedding)

    def closest_other(self, name: str) -> tuple[str | None, float]:
        """The most similar *other* person (used to calibrate the threshold).

        The similarity is the true maximum, also when it is negative: "the most
        similar other person is at -0.01" is a useful calibration result, but
        it would be lost if only positive values were considered.
        """
        profile = self.profiles.get(name)
        if profile is None:
            return None, 0.0
        best_name: str | None = None
        best = float("-inf")
        for other_name, other in self.profiles.items():
            if other_name == name:
                continue
            # Every sample of one person against every sample of the other:
            # the worst case for the threshold is the best possible match.
            value = max(
                (other.similarity(sample) for sample in profile.samples),
                default=-1.0,
            )
            if value > best:
                best = value
                best_name = other_name
        if best_name is None:
            return None, 0.0
        return best_name, float(best)
