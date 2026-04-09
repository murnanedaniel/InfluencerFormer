# Data Notes

Expected default data root:

- `COCO_ROOT/annotations/`
- `COCO_ROOT/train2017/`
- `COCO_ROOT/val2017/`

The first implementation pass should keep COCO data handling isolated to this benchmark area rather than adding COCO-specific logic to the older CLEVR utilities.
