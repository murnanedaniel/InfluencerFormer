# COCO DETR Benchmark

This benchmark now implements a runnable vanilla DETR comparison with two
assignment modes on the same class + box cost matrix:

- `hungarian`
- `pm3`

## What Is Implemented

- pretrained `facebook/detr-resnet-50` model loading through `transformers`
- COCO dataset loading through `torchvision` + `pycocotools`
- a shared DETR cost matrix with class, box L1, and GIoU terms
- Hungarian loss on that matrix
- PM3-style soft matching on that same matrix
- COCO download script
- tiny COCO generator for end-to-end smoke validation
- train/eval entrypoints with checkpointing and mAP reporting

## Data Setup

Full download:

- `python benchmarks/coco_detr/data/download_coco.py --coco_root data/coco`

Dry run only:

- `python benchmarks/coco_detr/data/download_coco.py --coco_root data/coco --dry_run`

Expected layout:

- `data/coco/annotations/instances_train2017.json`
- `data/coco/annotations/instances_val2017.json`
- `data/coco/train2017/`
- `data/coco/val2017/`

Tiny local validation dataset:

- `python benchmarks/coco_detr/data/create_tiny_coco.py --output_root benchmarks/coco_detr/data/tiny_coco`

## Running

Hungarian smoke run on real COCO:

- `python benchmarks/coco_detr/train.py --config benchmarks/coco_detr/configs/baseline_hungarian.json --smoke`

PM3 smoke run on real COCO:

- `python benchmarks/coco_detr/train.py --config benchmarks/coco_detr/configs/pm3_softmatch.json --smoke`

Tiny end-to-end validation runs:

- `python benchmarks/coco_detr/train.py --config benchmarks/coco_detr/configs/baseline_hungarian_tiny.json`
- `python benchmarks/coco_detr/train.py --config benchmarks/coco_detr/configs/pm3_softmatch_tiny.json`

Evaluate a run directory:

- `python benchmarks/coco_detr/eval.py --run_dir benchmarks/coco_detr/runs/<experiment_name>`

## Benchmark Rule

Keep the detector fixed and vary only the assignment / aggregation rule.
That is the point of this benchmark.
