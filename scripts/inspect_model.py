#!/usr/bin/env python
"""Inspect an ONNX model.

Usage::

    python scripts/inspect_model.py models/yolox_face.onnx
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inspect_model import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
