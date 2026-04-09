"""COCO detection dataset utilities for DETR-style training."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoDetection


def ensure_coco_layout(coco_root: str | Path, splits: Iterable[str]) -> None:
    """Validate the expected COCO directory layout."""
    root = Path(coco_root)
    ann_dir = root / 'annotations'
    if not ann_dir.exists():
        raise FileNotFoundError(f'Missing COCO annotations directory: {ann_dir}')
    for split in splits:
        image_dir = root / split
        ann_file = ann_dir / f'instances_{split}.json'
        if not image_dir.exists():
            raise FileNotFoundError(f'Missing COCO image directory: {image_dir}')
        if not ann_file.exists():
            raise FileNotFoundError(f'Missing COCO annotation file: {ann_file}')


def _valid_annotations(annotations: list[dict]) -> list[dict]:
    """Filter out crowd and degenerate boxes."""
    valid = []
    for ann in annotations:
        if ann.get('iscrowd', 0):
            continue
        x, y, w, h = ann['bbox']
        if w <= 1 or h <= 1:
            continue
        valid.append(ann)
    return valid


def _annotations_for_processor(image_id: int, annotations: list[dict]) -> dict:
    """Convert one image's labels to DETR processor format."""
    return {'image_id': image_id, 'annotations': annotations}


def _annotations_for_metrics(annotations: list[dict]) -> dict:
    """Convert COCO annotations to torchmetrics detection format."""
    if not annotations:
        return {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.int64),
        }
    boxes = []
    labels = []
    for ann in annotations:
        x, y, w, h = ann['bbox']
        boxes.append([x, y, x + w, y + h])
        labels.append(int(ann['category_id']))
    return {
        'boxes': torch.tensor(boxes, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.int64),
    }


class COCODetrDataset(Dataset):
    """Thin COCO wrapper that preserves raw annotations for DETR."""

    def __init__(self, coco_root: str | Path, split: str, max_images: int | None = None):
        root = Path(coco_root)
        ann_file = root / 'annotations' / f'instances_{split}.json'
        image_dir = root / split
        self.dataset = CocoDetection(root=str(image_dir), annFile=str(ann_file))
        self.indices = list(range(len(self.dataset)))
        if max_images is not None:
            self.indices = self.indices[:max_images]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, annotations = self.dataset[self.indices[idx]]
        image_id = int(self.dataset.ids[self.indices[idx]])
        annotations = _valid_annotations(annotations)
        return image, annotations, image_id, image.size


def build_collate_fn(image_processor) -> Callable:
    """Build a DETR-compatible collate function."""

    def collate_fn(batch):
        images, annotations, image_ids, image_sizes = zip(*batch)
        processor_annotations = [
            _annotations_for_processor(image_id=image_id, annotations=ann)
            for image_id, ann in zip(image_ids, annotations)
        ]
        encoded = image_processor(images=list(images), annotations=processor_annotations, return_tensors='pt')
        metric_targets = [_annotations_for_metrics(ann) for ann in annotations]
        orig_sizes = torch.tensor([[height, width] for width, height in image_sizes], dtype=torch.int64)
        return {
            'pixel_values': encoded['pixel_values'],
            'pixel_mask': encoded.get('pixel_mask'),
            'labels': encoded['labels'],
            'metric_targets': metric_targets,
            'image_ids': torch.tensor(image_ids, dtype=torch.int64),
            'orig_sizes': orig_sizes,
        }

    return collate_fn


def make_dataloaders(cfg, image_processor):
    """Build train and validation dataloaders."""
    ensure_coco_layout(cfg.data.coco_root, [cfg.data.train_split, cfg.data.val_split])
    train_limit = cfg.data.smoke_train_images if cfg.data.smoke else None
    val_limit = cfg.data.smoke_val_images if cfg.data.smoke else None
    train_dataset = COCODetrDataset(cfg.data.coco_root, cfg.data.train_split, max_images=train_limit)
    val_dataset = COCODetrDataset(cfg.data.coco_root, cfg.data.val_split, max_images=val_limit)
    collate_fn = build_collate_fn(image_processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader
