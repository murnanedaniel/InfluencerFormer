"""DETR model wrapper for the COCO benchmark."""

from __future__ import annotations

import torch
from transformers import AutoImageProcessor, DetrForObjectDetection


def infer_device(requested: str) -> torch.device:
    """Resolve the execution device."""
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def build_model_and_processor(cfg):
    """Instantiate the DETR model and image processor."""
    processor = AutoImageProcessor.from_pretrained(
        cfg.train.model_name,
        use_fast=True,
        cache_dir=cfg.train.cache_dir,
    )
    model = DetrForObjectDetection.from_pretrained(
        cfg.train.model_name,
        cache_dir=cfg.train.cache_dir,
    )
    return model, processor
