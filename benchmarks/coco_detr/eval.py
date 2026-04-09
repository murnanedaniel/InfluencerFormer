"""Evaluate a trained COCO DETR benchmark run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
_REPO = _ROOT.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmarks.coco_detr.config import load_config
from benchmarks.coco_detr.data import make_dataloaders
from benchmarks.coco_detr.losses import DetrCriterion
from benchmarks.coco_detr.models import build_model_and_processor, infer_device
from benchmarks.coco_detr.train import evaluate


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=str, required=True, help='Run directory with resolved config and checkpoint.')
    return parser.parse_args()


def main() -> None:
    """Evaluate a saved run."""
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    cfg = load_config(str(run_dir / 'resolved_config.json'))
    device = infer_device(cfg.train.device)
    model, image_processor = build_model_and_processor(cfg)
    checkpoint = torch.load(run_dir / 'checkpoint_last.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    criterion = DetrCriterion(cfg.matcher)
    _, val_loader = make_dataloaders(cfg, image_processor)
    metrics = evaluate(model, criterion, image_processor, val_loader, device)
    out_path = run_dir / 'eval_metrics.json'
    out_path.write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metrics, indent=2))
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
