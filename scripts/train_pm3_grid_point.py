#!/usr/bin/env python3
"""Compatibility wrapper for `studies/clevr/scripts/train_pm3_grid_point.py`."""

from pathlib import Path
import runpy

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "studies/clevr/scripts/train_pm3_grid_point.py"
runpy.run_path(str(TARGET), run_name="__main__")
