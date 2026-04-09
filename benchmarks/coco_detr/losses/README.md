# Loss And Matcher Notes

The key design rule for this benchmark is to keep the underlying DETR cost matrix fixed:

- classification term
- bbox L1 term
- GIoU term

and vary only the assignment / aggregation mechanism.

That makes Hungarian vs PM3-style comparisons interpretable.
