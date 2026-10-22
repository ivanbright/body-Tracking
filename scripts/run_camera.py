#!/usr/bin/env python
"""Run the tracker on a camera.

Usage::

    python scripts/run_camera.py --camera 0
    python scripts/run_camera.py --camera 0 --target-mode largest
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.main import main as cli_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    forwarded: list[str] = []
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == "--camera":
            forwarded += ["--source", "camera", "--camera-index", args[index + 1]]
            index += 2
            continue
        forwarded.append(argument)
        index += 1
    return cli_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
