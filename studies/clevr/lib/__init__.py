"""Shared CLEVR utilities and compatibility shims."""

from pathlib import Path

from . import clevr_utils

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _REPO_ROOT / "data" / "clevr" / "scenes"

clevr_utils._REPO_ROOT = _REPO_ROOT
clevr_utils._DATA_DIR = _DATA_DIR
clevr_utils.TRAIN_PATH = str(_DATA_DIR / "CLEVR_train_scenes.json")
clevr_utils.VAL_PATH = str(_DATA_DIR / "CLEVR_val_scenes.json")

__all__ = ["clevr_utils"]
