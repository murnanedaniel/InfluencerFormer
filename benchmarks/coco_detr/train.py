"""Train a vanilla DETR benchmark with Hungarian or PM3 matching."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dataclasses import asdict
from torchmetrics.detection.mean_ap import MeanAveragePrecision

try:
    import wandb
except ImportError:
    wandb = None

_ROOT = Path(__file__).resolve().parent
_REPO = _ROOT.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmarks.coco_detr.config import load_config, save_config
from benchmarks.coco_detr.data import make_dataloaders
from benchmarks.coco_detr.losses import DetrCriterion
from benchmarks.coco_detr.models import build_model_and_processor, infer_device


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=str, required=True, help='Benchmark JSON config.')
    parser.add_argument('--output_dir', type=str, default=None, help='Optional override for output root.')
    parser.add_argument('--smoke', action='store_true', help='Force smoke subset mode.')
    parser.add_argument('--epochs', type=int, default=None, help='Optional epoch override.')
    parser.add_argument('--device', type=str, default=None, help='Optional device override.')
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: dict, device: torch.device) -> dict:
    """Move model inputs and labels to device."""
    batch['pixel_values'] = batch['pixel_values'].to(device)
    if batch['pixel_mask'] is not None:
        batch['pixel_mask'] = batch['pixel_mask'].to(device)
    batch['labels'] = [{key: value.to(device) for key, value in label.items()} for label in batch['labels']]
    return batch


def save_checkpoint(run_dir: Path, model, optimizer, epoch: int, history: list[dict]) -> Path:
    """Save a training checkpoint."""
    checkpoint_path = run_dir / 'checkpoint_last.pt'
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
        },
        checkpoint_path,
    )
    return checkpoint_path


def evaluate(model, criterion, image_processor, dataloader, device: torch.device) -> dict:
    """Evaluate on the validation set."""
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    loss_values = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            metric_targets = [{'boxes': target['boxes'], 'labels': target['labels']} for target in batch['metric_targets']]
            orig_sizes = batch['orig_sizes']
            batch = to_device(batch, device)
            outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
            losses = criterion(outputs, batch['labels'])
            loss_values.append(float(losses['loss'].item()))
            results = image_processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=orig_sizes)
            preds = [
                {
                    'boxes': result['boxes'].cpu(),
                    'scores': result['scores'].cpu(),
                    'labels': result['labels'].cpu(),
                }
                for result in results
            ]
            metric.update(preds, metric_targets)
    summary = metric.compute()
    return {
        'val_loss': float(np.mean(loss_values)) if loss_values else float('nan'),
        'map': float(summary['map']),
        'map_50': float(summary['map_50']),
        'map_75': float(summary['map_75']),
    }


def main() -> None:
    """Run benchmark training."""
    args = parse_args()
    overrides = {}
    if args.output_dir is not None:
        overrides['output_dir'] = args.output_dir
    if args.smoke:
        overrides.setdefault('data', {})['smoke'] = True
    if args.epochs is not None:
        overrides.setdefault('train', {})['epochs'] = args.epochs
    if args.device is not None:
        overrides.setdefault('train', {})['device'] = args.device
    cfg = load_config(args.config, overrides=overrides)
    set_seed(cfg.seed)

    run_root = Path(cfg.output_dir).resolve()
    run_dir = run_root / cfg.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / 'resolved_config.json')
    metrics_path = run_dir / 'metrics.jsonl'
    metrics_path.write_text('', encoding='utf-8')
    eval_metrics_path = run_dir / 'eval_metrics.json'
    if eval_metrics_path.exists():
        eval_metrics_path.unlink()

    # Initialize wandb if enabled
    _wandb_enabled = cfg.wandb.enabled and wandb is not None
    if cfg.wandb.enabled and wandb is None:
        print('WARNING: wandb enabled in config but not installed. Skipping.')
    if _wandb_enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=cfg.experiment_name,
            tags=cfg.wandb.tags,
            config=asdict(cfg),
        )

    device = infer_device(cfg.train.device)
    model, image_processor = build_model_and_processor(cfg)
    model.to(device)
    criterion = DetrCriterion(cfg.matcher)
    train_loader, val_loader = make_dataloaders(cfg, image_processor)

    # Separate backbone and head parameter groups
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name or 'model.backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    backbone_lr = cfg.train.lr * cfg.train.backbone_lr_factor
    param_groups = [
        {'params': head_params, 'lr': cfg.train.lr},
        {'params': backbone_params, 'lr': backbone_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)
    print(f'Optimizer: head_lr={cfg.train.lr}, backbone_lr={backbone_lr} '
          f'({len(head_params)} head, {len(backbone_params)} backbone params)')

    # LR scheduler
    scheduler = None
    if cfg.train.lr_schedule == 'cosine':
        total_steps = len(train_loader) * cfg.train.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    elif cfg.train.lr_schedule == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader) * (cfg.train.epochs // 3), gamma=0.1)

    history = []
    for epoch in range(cfg.train.epochs):
        epoch_start = time.time()
        model.train()
        running = []
        for step, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            optimizer.zero_grad()
            outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
            losses = criterion(outputs, batch['labels'])
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running.append(float(losses['loss'].item()))
            if step % cfg.train.log_every_steps == 0:
                print(
                    f'epoch={epoch} step={step} '
                    f'loss={losses["loss"].item():.4f} '
                    f'ce={losses["loss_ce"].item():.4f} '
                    f'bg={losses["loss_background"].item():.4f}'
                )
                if _wandb_enabled:
                    wandb.log({
                        'train/loss': losses['loss'].item(),
                        'train/loss_ce': losses['loss_ce'].item(),
                        'train/loss_bbox': losses['loss_bbox'].item(),
                        'train/loss_giou': losses['loss_giou'].item(),
                        'train/loss_background': losses['loss_background'].item(),
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                    })

        epoch_metrics = {
            'epoch': epoch,
            'train_loss': float(np.mean(running)) if running else float('nan'),
            'epoch_time_s': time.time() - epoch_start,
        }
        if (epoch + 1) % cfg.train.eval_every_epochs == 0:
            epoch_metrics.update(evaluate(model, criterion, image_processor, val_loader, device))
        history.append(epoch_metrics)
        with open(metrics_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(epoch_metrics) + '\n')
        print(json.dumps(epoch_metrics, indent=2))
        if _wandb_enabled:
            wandb.log({
                'epoch/train_loss': epoch_metrics['train_loss'],
                'epoch/epoch_time_s': epoch_metrics['epoch_time_s'],
                'epoch/val_loss': epoch_metrics.get('val_loss'),
                'epoch/map': epoch_metrics.get('map'),
                'epoch/map_50': epoch_metrics.get('map_50'),
                'epoch/map_75': epoch_metrics.get('map_75'),
                'epoch': epoch,
            })
        if (epoch + 1) % cfg.train.save_every_epochs == 0:
            save_checkpoint(run_dir, model, optimizer, epoch, history)

    save_checkpoint(run_dir, model, optimizer, cfg.train.epochs - 1, history)
    if _wandb_enabled:
        wandb.finish()
    print(f'Run artifacts written to {run_dir}')


if __name__ == '__main__':
    main()
