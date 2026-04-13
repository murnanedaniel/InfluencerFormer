"""Evaluate the pretrained DETR checkpoint on COCO val to reproduce published mAP."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

_ROOT = Path(__file__).resolve().parent
_REPO = _ROOT.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmarks.coco_detr.data import COCODetrDataset, build_collate_fn
from benchmarks.coco_detr.models import infer_device
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor


def main():
    coco_root = Path("data/coco")
    model_name = "facebook/detr-resnet-50"
    max_val = None  # None = full val set

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    max_val = args.max_val

    device = infer_device("auto")
    print(f"Device: {device}")

    print(f"Loading model: {model_name}")
    model = DetrForObjectDetection.from_pretrained(model_name)
    image_processor = DetrImageProcessor.from_pretrained(model_name)
    model.to(device)
    model.eval()

    print(f"Loading COCO val2017 (max_images={max_val})...")
    val_dataset = COCODetrDataset(coco_root, "val2017", max_images=max_val)
    print(f"  {len(val_dataset)} images")
    collate_fn = build_collate_fn(image_processor)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn)

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    t0 = time.time()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device) if batch["pixel_mask"] is not None else None
            orig_sizes = batch["orig_sizes"]
            metric_targets = [
                {"boxes": t["boxes"], "labels": t["labels"]}
                for t in batch["metric_targets"]
            ]

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            results = image_processor.post_process_object_detection(
                outputs, threshold=0.0, target_sizes=orig_sizes
            )
            preds = [
                {
                    "boxes": r["boxes"].cpu(),
                    "scores": r["scores"].cpu(),
                    "labels": r["labels"].cpu(),
                }
                for r in results
            ]
            metric.update(preds, metric_targets)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  batch {i+1}/{len(val_loader)} ({elapsed:.1f}s)")

    summary = metric.compute()
    elapsed = time.time() - t0
    results = {
        "model": model_name,
        "val_images": len(val_dataset),
        "map": float(summary["map"]),
        "map_50": float(summary["map_50"]),
        "map_75": float(summary["map_75"]),
        "map_small": float(summary["map_small"]),
        "map_medium": float(summary["map_medium"]),
        "map_large": float(summary["map_large"]),
        "eval_time_s": elapsed,
    }
    print(json.dumps(results, indent=2))

    out_path = _ROOT / "runs" / "pretrained_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
