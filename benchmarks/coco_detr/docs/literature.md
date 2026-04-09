# Literature Notes

## Core Baseline

### DETR
- Canonical end-to-end object detection as set prediction.
- Uses Hungarian one-to-one matching over a combined cost:
  classification + box L1 + GIoU.
- This is the benchmark baseline to reproduce before changing the matcher.

## Closest Soft-Matching Reference

### RTP-DETR
- Relevant because it replaces strict one-to-one correspondence with a softer,
  transport-style assignment on COCO.
- Good reference for what a detection-relevant soft matching comparison should
  look like.

## Important Context Papers

### Align-DETR
- Useful for alignment-aware loss design and mixed matching supervision.

### H-DETR
- Important reference for combining one-to-one and one-to-many supervision.

### Co-DETR
- Shows how collaborative / auxiliary assignment strategies can improve DETR
  training.

## Practical Lessons For PM3

- Keep the cost matrix fixed when comparing matchers.
- Match only the assignment mechanism first; do not simultaneously change the
  detector, data pipeline, and cost weights.
- Expect soft matching to change the optimization landscape:
  PM3-like objectives may help gradient flow but can also blur one-to-one
  responsibility assignment if the temperature is too high.
- Smoke tests should check:
  - baseline AP parity
  - duplicate predictions
  - convergence speed
  - stability under reduced training schedules
