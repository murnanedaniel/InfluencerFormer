"""Configuration helpers for the COCO DETR benchmark."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class MatcherConfig:
    """Matcher hyperparameters.

    Inputs:
        name: Matcher family.
        classification_cost: Classification cost weight in the assignment matrix.
        bbox_l1_cost: Bounding-box L1 cost weight in the assignment matrix.
        giou_cost: GIoU cost weight in the assignment matrix.
        eos_coefficient: Weight of the no-object class.
        pm3_temperature: PM3 temperature for soft matching.
        pm3_power: PM3 power for soft aggregation.
    Outputs:
        Structured matcher configuration.
    """

    name: str = 'hungarian'
    classification_cost: float = 1.0
    bbox_l1_cost: float = 5.0
    giou_cost: float = 2.0
    eos_coefficient: float = 0.1
    pm3_temperature: float = 0.12
    pm3_power: float = 3.0


@dataclass
class DataConfig:
    """Dataset configuration.

    Inputs:
        coco_root: COCO root directory.
        train_split: Train split name.
        val_split: Validation split name.
        num_workers: DataLoader worker count.
        smoke: Whether to use a reduced subset.
        smoke_train_images: Number of train images for smoke mode.
        smoke_val_images: Number of val images for smoke mode.
    Outputs:
        Structured dataset configuration.
    """

    coco_root: str = 'data/coco'
    train_split: str = 'train2017'
    val_split: str = 'val2017'
    num_workers: int = 4
    smoke: bool = False
    smoke_train_images: int = 128
    smoke_val_images: int = 64


@dataclass
class TrainConfig:
    """Training configuration.

    Inputs:
        model_name: Hugging Face DETR checkpoint name.
        batch_size: Train batch size.
        eval_batch_size: Validation batch size.
        epochs: Number of epochs.
        lr: Learning rate.
        weight_decay: Optimizer weight decay.
        gradient_clip_norm: Gradient clipping norm.
        log_every_steps: Logging interval.
        eval_every_epochs: Validation interval.
        device: Explicit device override.
        cache_dir: Optional model cache directory.
        save_every_epochs: Checkpoint save interval.
    Outputs:
        Structured training configuration.
    """

    model_name: str = 'facebook/detr-resnet-50'
    batch_size: int = 2
    eval_batch_size: int = 2
    epochs: int = 12
    lr: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 0.1
    log_every_steps: int = 10
    eval_every_epochs: int = 1
    device: str = 'auto'
    cache_dir: str | None = None
    save_every_epochs: int = 1


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""

    enabled: bool = False
    project: str = 'coco-detr-benchmark'
    entity: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Inputs:
        experiment_name: Run name.
        output_dir: Directory for run outputs.
        seed: Random seed.
        backbone: Backbone label for reporting.
        notes: Free-form notes.
        data: Dataset configuration.
        train: Training configuration.
        matcher: Matcher configuration.
        wandb: Weights & Biases configuration.
    Outputs:
        Structured experiment configuration.
    """

    experiment_name: str = 'coco_detr_smoke'
    output_dir: str = 'benchmarks/coco_detr/runs'
    seed: int = 42
    backbone: str = 'resnet50'
    notes: str = ''
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def _merge_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries."""
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | None = None, overrides: Dict[str, Any] | None = None) -> ExperimentConfig:
    """Load and merge an experiment config."""
    raw: Dict[str, Any] = asdict(ExperimentConfig())
    if config_path:
        with open(Path(config_path), 'r', encoding='utf-8') as handle:
            raw = _merge_dict(raw, json.load(handle))
    if overrides:
        raw = _merge_dict(raw, overrides)
    return ExperimentConfig(
        experiment_name=raw['experiment_name'],
        output_dir=raw['output_dir'],
        seed=raw['seed'],
        backbone=raw.get('backbone', 'resnet50'),
        notes=raw.get('notes', ''),
        data=DataConfig(**raw['data']),
        train=TrainConfig(**raw['train']),
        matcher=MatcherConfig(**raw['matcher']),
        wandb=WandbConfig(**raw.get('wandb', asdict(WandbConfig()))),
    )


def save_config(cfg: ExperimentConfig, path: Path) -> None:
    """Save a resolved config to disk."""
    path.write_text(json.dumps(asdict(cfg), indent=2) + '\n', encoding='utf-8')
