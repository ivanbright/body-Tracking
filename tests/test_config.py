"""Configuration: loading, validation and overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import (
    AppConfig,
    ConfigError,
    DetectorConfig,
    load_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_default_configuration_is_valid() -> None:
    config = load_config(PROJECT_ROOT / "config" / "default.yaml")
    assert config.detector.backend == "yolox_onnx"
    assert config.detector.output_format == "yolox_raw"
    assert config.detector.activation == "none"
    assert config.target.mode == "none"


def test_default_configuration_has_no_legacy_keys() -> None:
    """The MQTT / hardware requirements must not come back."""
    raw = (PROJECT_ROOT / "config" / "default.yaml").read_text(encoding="utf-8").lower()
    for forbidden in ("mqtt", "tachometer", "servo", "needle", "tflite", "part 2"):
        assert forbidden not in raw


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("detector:\n  not_a_real_key: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert "not_a_real_key" in str(error.value)


def test_environment_override_is_applied(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("detector:\n  backend: haar\n", encoding="utf-8")
    config = load_config(
        path,
        environ={"YOLOX_DETECTOR__SCORE_THRESH": "0.75"},
    )
    assert config.detector.backend == "haar"
    assert config.detector.score_thresh == pytest.approx(0.75)


def test_unknown_environment_section_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("detector:\n  backend: haar\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path, environ={"YOLOX_NOSUCHSECTION__KEY": "1"})


def test_missing_file_produces_an_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as error:
        load_config(tmp_path / "missing.yaml")
    assert "not found" in str(error.value)


def test_invalid_output_format_is_rejected() -> None:
    config = AppConfig()
    config.detector = DetectorConfig(output_format="whatever")
    with pytest.raises(ConfigError) as error:
        config.validate()
    assert "output_format" in str(error.value)
    assert "inspect_model" in str(error.value)


def test_relative_paths_are_resolved_against_the_project_root(tmp_path: Path) -> None:
    config = AppConfig(project_root=tmp_path)
    config.detector = DetectorConfig(model_path="models/model.onnx")
    config.resolve_paths()
    assert config.detector.model_path == str((tmp_path / "models" / "model.onnx").resolve())
