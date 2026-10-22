"""Logging helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def setup_logging(level: str = "INFO", log_file: str = "", fmt: str | None = None) -> logging.Logger:
    """Configure the root logger once and return it."""
    global _CONFIGURED

    root = logging.getLogger()
    if not _CONFIGURED:
        formatter = logging.Formatter(
            fmt
            or "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        )
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        root.addHandler(stream)

        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

        root.setLevel(getattr(logging, level.upper(), logging.INFO))
        _CONFIGURED = True
    else:
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
