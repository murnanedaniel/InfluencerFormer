"""Model helpers for the COCO DETR benchmark."""

from .detr_wrapper import build_model_and_processor, infer_device

__all__ = ['build_model_and_processor', 'infer_device']
