"""The gallery: the enrolled people and their recognition.

The tests never load a real model - the embeddings are vectors that are built
in the code, so the rules can be verified exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.recognition.gallery import Gallery, IdentityProfile


def unit_vector(seed: int) -> np.ndarray:
    """A deterministic unit vector."""
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(512).astype(np.float32)
    return vector / np.linalg.norm(vector)


def make_gallery(tmp_path) -> Gallery:
    gallery = Gallery(tmp_path / "face_database.json")
    gallery.add("Ivan", [unit_vector(1)], source="test")
    gallery.add("Anna", [unit_vector(2)], source="test")
    return gallery


# --------------------------------------------------------------------------- #
# the profile
# --------------------------------------------------------------------------- #
def test_a_profile_needs_at_least_one_sample() -> None:
    with pytest.raises(ValueError):
        IdentityProfile.from_samples("Nobody", [])


def test_the_template_is_the_normalized_mean() -> None:
    profile = IdentityProfile.from_samples("Ivan", [unit_vector(1), unit_vector(1)])
    assert profile.template is not None
    assert float(np.linalg.norm(profile.template)) == pytest.approx(1.0, abs=1e-5)
    assert profile.similarity(unit_vector(1)) == pytest.approx(1.0, abs=1e-5)


def test_the_similarity_is_the_maximum_over_the_samples() -> None:
    # A face that turned to the side is very different from the average of
    # all the samples, but it matches the sample of that angle.
    frontal = unit_vector(1)
    turned = unit_vector(5)
    profile = IdentityProfile.from_samples("Ivan", [frontal, turned])
    assert profile.similarity(turned) == pytest.approx(1.0, abs=1e-5)


def test_a_different_person_does_not_match() -> None:
    profile = IdentityProfile.from_samples("Ivan", [unit_vector(1)])
    assert profile.similarity(unit_vector(2)) < 0.5


# --------------------------------------------------------------------------- #
# the gallery
# --------------------------------------------------------------------------- #
def test_the_gallery_finds_the_best_match(tmp_path) -> None:
    gallery = make_gallery(tmp_path)
    name, similarity = gallery.best_match(unit_vector(2))
    assert name == "Anna"
    assert similarity == pytest.approx(1.0, abs=1e-5)


def test_the_best_match_is_also_reported_when_it_is_too_low(tmp_path) -> None:
    gallery = make_gallery(tmp_path)
    name, similarity = gallery.best_match(unit_vector(9))
    assert name in ("Ivan", "Anna")
    assert similarity < 0.6


def test_an_empty_gallery_matches_nothing(tmp_path) -> None:
    gallery = Gallery(tmp_path / "face_database.json")
    assert gallery.best_match(unit_vector(1)) == (None, 0.0)


def test_the_similarity_with_a_named_person(tmp_path) -> None:
    gallery = make_gallery(tmp_path)
    assert gallery.similarity("Ivan", unit_vector(1)) == pytest.approx(1.0, abs=1e-5)
    assert gallery.similarity("Ivan", unit_vector(2)) < 0.5
    assert gallery.similarity("Nobody", unit_vector(1)) == 0.0


def test_the_closest_other_person_is_reported(tmp_path) -> None:
    gallery = make_gallery(tmp_path)
    name, similarity = gallery.closest_other("Ivan")
    assert name == "Anna"
    assert similarity < 0.5


def test_the_closest_other_person_is_also_reported_when_negative(tmp_path) -> None:
    """A negative similarity is a calibration result and must not get lost."""
    gallery = Gallery(tmp_path / "face_database.json")
    gallery.add("Ivan", [unit_vector(1)])
    gallery.add("Anna", [-unit_vector(1)])

    name, similarity = gallery.closest_other("Ivan")

    assert name == "Anna"
    assert similarity == pytest.approx(-1.0, abs=1e-5)


def test_the_closest_other_person_uses_every_sample(tmp_path) -> None:
    """The worst case for the threshold is the best possible match."""
    gallery = Gallery(tmp_path / "face_database.json")
    gallery.add("Ivan", [unit_vector(1), unit_vector(9)])
    gallery.add("Anna", [unit_vector(2), unit_vector(9)])

    name, similarity = gallery.closest_other("Ivan")

    assert name == "Anna"
    assert similarity == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def test_the_gallery_survives_a_save_and_load(tmp_path) -> None:
    path = tmp_path / "profiles" / "face_database.json"
    gallery = Gallery(path)
    gallery.add("Ivan", [unit_vector(1), unit_vector(5)], source="camera:1")
    gallery.save()

    loaded = Gallery(path)
    assert loaded.load() == 1
    assert loaded.names == ["Ivan"]
    profile = loaded.get("Ivan")
    assert profile is not None
    assert len(profile.samples) == 2
    assert profile.similarity(unit_vector(5)) == pytest.approx(1.0, abs=1e-5)


def test_a_missing_gallery_is_not_an_error(tmp_path) -> None:
    gallery = Gallery(tmp_path / "does_not_exist.json")
    assert gallery.load() == 0
    assert gallery.names == []


def test_a_broken_gallery_is_ignored(tmp_path) -> None:
    path = tmp_path / "face_database.json"
    path.write_text("{not json", encoding="utf-8")
    gallery = Gallery(path)
    assert gallery.load() == 0


def test_a_person_can_be_removed(tmp_path) -> None:
    gallery = make_gallery(tmp_path)
    assert gallery.remove("Ivan") is True
    assert gallery.remove("Ivan") is False
    assert gallery.names == ["Anna"]
