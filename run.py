"""Portable module launcher, including Windows Python safe-path runtimes."""

import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/matplotlib")
)
os.environ.setdefault(
    "BLACK_CACHE_DIR", str(Path(__file__).resolve().parent / ".cache/black")
)
module = sys.argv.pop(1) if len(sys.argv) > 1 else "scrc.eval.replay"
runpy.run_module(module, run_name="__main__")
