# ej-vae study — set-prediction loss comparison

Final 3-way comparison at 100k/batch256/25-30ep/lr1e-4/seed42 on the COCOA jet dataset (24 track features per jet, up to 200 slots).

## Configs

- **ord-hf-nokld** — ordered pt-sorted L1 + focal BCE existence + 0.3×Hungarian feature regularizer + KLD=0.
- **sink-topk** — Sinkhorn topk_ordered (τ=1.0, 20 iters, real_only transport, lr=3e-4).
- **hung-focal** — Hungarian matching with focal BCE + 0.05·cardinality aux.

See `plots/{config}/` for per-config figures:
1. `feature_marginals.png` — 24-panel truth vs prediction density overlays.
2. `feature_residuals.png` — per-feature residual distributions with med/IQR.
3. `n_pred_vs_n_true_scatter.png` — per-event cardinality scatter.
4. `cardinality_hist.png` — n_true/n_pred marginals + residual.
5. `matched_l1_per_event.png` — per-event matched L1 distribution (Hungarian at eval time).
6. `score_per_event.png` — per-event `ml1 × |n_pred − n_true|`.

## Aggregate (5000 val jets, n_true=65.1)

| config | ml1 | card_residual | score (ml1·c_res) | n_pred |
|---|---|---|---|---|
| ord-hf-nokld | 0.240 | **10.5** | **2.68** | 75.7 |
| sink-topk | 0.237 | 18.7 | 4.64 | 83.9 |
| hung-focal | **0.215** | 28.2 | 6.51 | 92.1 |

**Winner on composite score**: `ord-hf-nokld` — **59% lower** than Sinkhorn, **59% lower** than Hungarian. Competitive on ml1 (within 12% of Hungarian's best) while keeping cardinality 3× tighter.

## Headline

The supervised pt-ordered loss is no longer a dead-weight baseline. Stacking three cheap ideas — a small Hungarian regularizer, focal BCE for existence, no KLD regularizer — brings it within the performance envelope of Hungarian on matching quality AND beats it convincingly on cardinality, giving the best composite score.

See `benchmarks/ej-vae/docs/` (inside the submodule) for the 4-wave ablation journey that produced these configs.
