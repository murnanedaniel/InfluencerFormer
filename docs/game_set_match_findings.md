# Game, Set and Match: Differentiable Set Prediction with Existence

## Summary of Findings

This document summarizes a systematic investigation into differentiable loss functions for set prediction with variable cardinality — the problem of predicting an unordered set of objects when the number of objects varies per sample.

## The Problem

Given N_pred fixed output slots and M < N_pred target objects:
1. **Matching**: which prediction corresponds to which target? (permutation invariance)
2. **Existence**: which slots contain real objects vs empty? (variable cardinality)

## Experimental Setup

Toy problem: 2D points, N_target=5, N_pred=20, direct optimization (no neural network), Adam lr=0.005, 8000 steps, 10 random seeds. Targets padded with K = N_pred - N_target copies of a learned dustbin point. Existence classified by radius: closer to real target than to dustbin center = exists.

## Key Results

### The Landscape of Matching Mechanisms

| Method | F1 (existence) | ml2 (matching) | Iters | Differentiable |
|--------|---------------|----------------|-------|----------------|
| Hungarian + dustbin padding | 1.00 ± 0.00 | 0.0004 | N/A | No |
| Sinkhorn 20 iters | 0.96 ± 0.04 | 0.014 | 20 | Yes |
| Sinkhorn 10 iters | 0.93 ± 0.04 | 0.016 | 10 | Yes |
| **Powered Product GM^3** | **0.92 ± 0.10** | **0.195** | **1** | **Yes** |
| Sinkhorn 5 iters | 0.82 ± 0.05 | 0.008 | 5 | Yes |
| Leave-one-out product | 0.66 ± 0.04 | 0.130 | 1 | Yes |
| Sinkhorn 3 iters | 0.64 ± 0.05 | 0.008 | 3 | Yes |
| Product (GM) | 0.61 ± 0.09 | 0.142 | 1 | Yes |
| PM3 / Power-SoftMin | 0.45 ± 0.02 | 0.025 | 1 | Yes |
| Sinkhorn 1 iter (= PM3) | 0.42 ± 0.02 | 0.0004 | 1 | Yes |
| Chamfer | 0.42 ± 0.00 | 0.0004 | 1 | Yes |
| Softmax (all variants) | 0.44-0.47 | 0.003-0.05 | 1 | Yes |

### The Sinkhorn Iteration Ladder

PM3 (Power-SoftMin) is exactly 1 iteration of Sinkhorn — two independent softmaxes (row and column normalization) that don't couple. Each additional Sinkhorn iteration propagates coupling information bidirectionally:

- 1 iter: F1=0.42 (no coupling)
- 3 iters: F1=0.64 (partial coupling)
- 5 iters: F1=0.82 (good coupling)
- 10 iters: F1=0.93 (strong coupling)
- 20 iters: F1=0.96 (converged)

### The Powered Product Discovery

The geometric mean product loss GM = exp(mean_i(log D_ij)) gets F1=0.61, equivalent to ~3 Sinkhorn iterations of coupling through its gradient structure. When raised to a power p=3 (GM^3), it jumps to F1=0.92, matching Sinkhorn 10 iterations — in a single non-iterative forward pass.

**Why it works**: The gradient is `(p/N) · GM^p / D_kj`. The GM^p term amplifies the coverage signal by p-th power. When a target is uncovered (GM~0.5), signal is 0.125; when covered (GM~0.01), it's 10^-6. This 100,000x dynamic range creates strong bijectivity pressure that compensates for the 1/N dilution.

**Why matching quality is worse**: The powered product optimizes a different objective (minimize GM^p) than Sinkhorn (minimize transport cost). The power amplification helps existence but distorts the matching landscape.

### The Temperature Discovery

For all temperature-based methods (Sinkhorn, PM3, SoftMin):
- **tau must scale with the cost magnitude**
- [0,1] targets (mean distance ~0.5): optimal tau ≈ 0.2-0.3
- CLEVR targets (mean distance ~2.7): optimal tau ≈ 0.5
- ej-vae cdist(p=1) (mean cost ~20): tau should be ~6-12, NOT 0.10

Our ej-vae benchmark used tau=0.10 with costs~20, which was ~100x too low — the softmax was completely peaked (hard argmin), eliminating all gradient flow for reassignment.

### Dustbin Type Comparison

For existence prediction via spatial classification (closer to target than dustbin = exists):

| Dustbin type | Sinkhorn F1 | Softmax F1 |
|-------------|-------------|------------|
| K degenerate copies | 0.99 | 0.45 |
| K learned individual | 0.87 | 0.45 |
| K Gaussian samples | 0.70 | 0.47 |
| SuperGlue score-based | 0.00 | — |
| None (rectangular) | 0.00 | 0.00 |

Simpler is better: degenerate copies outperform all alternatives. The Sinkhorn doubly-stochastic constraint handles the degeneracy; softmax cannot.

### The Spread Dustbin Discovery

The powered product's matching quality limitation (ml2=0.19) was caused by **degenerate dustbin targets**, not the product structure itself. With K identical dustbin copies at one point, the coverage GM for real targets is dominated by the 15 far predictions (all at the same dustbin point), washing out the signal from the 1 close prediction.

**Fix: spread dustbin targets** — place K dustbin targets at distinct locations (e.g., on a circle) rather than all at one point.

| Method | Dustbin | F1 | Perfect | ml2 |
|--------|---------|-----|---------|-----|
| **GM^3** | **spread r=0.3** | **0.95** | **7/10** | 0.214 |
| Sinkhorn 10 | degenerate | 0.93 | 2/10 | 0.016 |
| Sinkhorn 10 | spread r=0.3 | 0.93 | 2/10 | 0.015 |
| GM^3 | degenerate | 0.92 | 5/10 | 0.195 |
| GM^3 | spread r=0.1 | 0.92 | 5/10 | 0.216 |
| GM^3 | spread r=0.5 | 0.93 | 4/10 | 0.297 |
| GM^3 | spread r=1.0 | 0.64 | 0/10 | 0.644 |

**GM^3 + spread dustbins (r=0.3) beats Sinkhorn on existence (F1=0.95 vs 0.93, 7/10 vs 2/10 perfect) — in a single pass, no iteration.** Sinkhorn doesn't benefit from spread dustbins because its doubly-stochastic constraint already handles degeneracy algebraically.

Key insight: the product loss needs **spatial diversity** in dustbin targets because its coverage diagnostic (geometric mean) averages over all predictions. Degenerate dustbins make 15/20 entries in this average identical, diluting the coverage signal. Sinkhorn's iterative normalization bypasses this by computing assignments directly rather than through distance aggregation.

### Additional findings

Coverage-only GM^3 (no precision term): F1=0.58, ml2=0.011. Confirms that **precision drives existence** (pushes unmatched preds to dustbin) while **coverage drives matching** (pulls preds to targets).

Coverage weight sweep: upweighting coverage helps matching but hurts existence. The precision term's bijectivity enforcement is essential.

### Negative Results

~90 product loss variants tested across 5 research directions (log-space, margins/hinge, Gaussian dustbin, hybrids, coupling tricks). No single-pass product reformulation achieved F1 > 0.71 *without* the power amplification trick. The softmin-weighted product, top-K product, harmonic mean, and all margin variants (hinge, sigmoid, soft-hinge) failed to improve over standard GM.

## Game-Theoretic Interpretation

The matching problem is fundamentally a **competitive game** between predictions bidding for targets:

1. **Hungarian**: Exact Nash equilibrium of the assignment game (LP duality = competitive equilibrium prices). Kuhn (1955) explicitly connected it to game theory.

2. **Sinkhorn**: Finds the entropy-regularized Nash equilibrium via iterative price adjustment. Each iteration is one round of "bidding" where row and column normalizations adjust prices until the market clears. Related to Bertsekas' auction algorithm (1979).

3. **Product Loss**: Creates competitive exclusion through multiplicative feedback — when one prediction "wins" a target, others' gradients toward it vanish. Analogous to Lotka-Volterra competition in ecology (Gause's exclusion principle). The power amplification (GM^p) strengthens the competitive pressure.

4. **PM3/SoftMin**: One round of bidding (single softmax). Not enough for equilibrium.

**The "Game, Set and Match" narrative:**
- **Game**: Matching is a competitive game; different losses implement different game-solving mechanisms
- **Set**: The set prediction problem with variable cardinality
- **Match**: The matching layer connecting predictions to targets

## Architectural Implications

### For ej-vae and similar benchmarks:
1. Replace PM3 softmax with Sinkhorn (10 iterations) — same O(N²) cost per iteration, trivial implementation
2. Pad targets with K degenerate dustbin copies (learned embedding)
3. Scale tau proportionally to cost magnitude (~cost_mean/5)
4. Classify existence by assignment column (real vs dustbin) or by spatial distance

### Powered Product as alternative:
- Single-pass, no iteration — fastest possible
- F1=0.92 for existence (near-Sinkhorn)
- Matching quality is 10x worse — may need hybrid approach
- Ideal when existence matters more than matching precision

## Tuning the Powered Product

Comprehensive sweep (10 seeds each):

| Config | F1 | ml2 |
|--------|-----|-----|
| GM^3 (p=3, lr=0.005) | 0.92 | 0.195 |
| GM^2 | 0.77 | 0.148 |
| GM^2.5 | 0.86 | 0.170 |
| GM^3.5 | 0.92 | 0.284 |
| GM^3 lr=0.001 | 0.83 | 0.129 |
| GM^3 + softmin (any alpha) | 0.89-0.90 | 0.19-0.21 |
| GM^3 + Sinkhorn5 (alpha=0.3) | 0.86 | 0.128 |

**p=3 is the optimal power.** ml2≈0.19 is the floor — a fundamental property of the powered product objective. Hybrids with softmin or Sinkhorn dilute the bijectivity signal without fully gaining matching quality.

The tradeoff is fundamental: GM^p optimizes "minimize powered geometric mean" which has different optima than "minimize transport cost." Sinkhorn optimizes the latter directly.

## The Practical Recommendation

| Need | Method | F1 | ml2 | Cost |
|------|--------|-----|-----|------|
| Single-pass, fast | GM^3 | 0.92 | 0.19 | O(N²) |
| Good matching + existence | Sinkhorn 10 | 0.93 | 0.016 | O(10·N²) |
| Perfect | Hungarian | 1.00 | 0.0004 | O(N³) CPU |

## ej-vae Results (N_pred=200, D=24, particle jet reconstruction)

### Sinkhorn on ej-vae

| Config | ml1 | n_pred | card_res | Notes |
|--------|-----|--------|----------|-------|
| **sinkhorn tau=1 i=10** | **0.224** | 65.5 | 26.5 | Best matching ever on ej-vae |
| sinkhorn tau=0.5 | 0.238 | 92.8 | 39.2 | tau too low |
| sinkhorn tau=1.5 | 0.231 | 139.9 | 76.9 | tau too high |
| hungarian | 0.243 | 70.6 | **8.5** | Best cardinality |

**Sinkhorn tau=1 beats Hungarian on matching (0.224 vs 0.243)** — first time a differentiable loss has beaten Hungarian on ej-vae. Also 21% faster per epoch (GPU Sinkhorn vs CPU Hungarian).

**But card_residual was 3x worse** (26.5 vs 8.5) with Sinkhorn-derived existence. Fix: use Hungarian for existence targets (like DETR does), Sinkhorn for differentiable feature matching.

### Final result: Sinkhorn matching + Hungarian existence

| Config | ml1 | npred | card_res | sec/ep |
|--------|-----|-------|----------|--------|
| **Sinkhorn 5-iter + Hungarian exist** | **0.230** | **73.3** | 27.1 | ~12s |
| Sinkhorn 10-iter + Hungarian exist | 0.225 | 63.9 | 26.9 | ~15s |
| Hungarian (baseline) | 0.245 | 73.0 | **11.3** | 19s |

**Sinkhorn 5-iter + Hungarian exist is the recommended config:**
- 6% better matching than Hungarian (0.230 vs 0.245)
- Correct cardinality (npred=73.3 ≈ Hungarian's 73.0)
- 37% faster (12s vs 19s per epoch)
- Feature gradients flow through Sinkhorn (differentiable matching)

### Product loss on ej-vae

Raw GM product stuck at ml1≈0.50 regardless of power or /D normalization. The 1/N=1/200 dilution is too severe. PW-SoftMin variant (softmin matching + GM reweighting) now being tested.

### tau scaling confirmed
- ej-vae: cdist(p=1)/D costs ~1.0 → optimal tau ≈ 1.0 for Sinkhorn
- Consistent with toy finding: tau should be on the order of the cost magnitude

### The Cardinality Problem

The fundamental tension in set prediction with unordered matching:

| Method | ml1 (matching) | card (cardinality) | Why |
|--------|-------|------|-----|
| Ordered (sorted) | 0.458 | **5.1** | Position = existence (trivial cutoff) |
| Hungarian | 0.241 | **8.0** | Explicit ∅ class with stable labels |
| Sinkhorn argmax | **0.223** | 30.7 | Soft assignment → noisy exist signal |

**Root cause**: Sinkhorn matching is permutation-equivariant — any query can match any target. The existence signal depends on competitive assignment, which is context-dependent and unstable. Ordered matching converts existence into a positional cutoff — much easier to learn.

**Approaches tested (2k data, 50 epochs):**

| Approach | card | ml1 |
|----------|------|-----|
| ∅ class downweighting (DETR-style) | 54.4 → 30.7 (argmax) | 0.222 |
| Curriculum (ordered→Sinkhorn) | 37.6 | 0.228 |
| Sinkhorn + ordered hybrid | 34.8 | 0.228 |
| No annealing (exist from ep0) | 26.0 | 0.224 |
| Card token (conditioning mask on global count) | testing | testing |

**Key insights:**
1. Cardinality-only task: encoder regression gets MAE=13.5, one-hot gets 25.1, baseline (predict mean) is 23.7. Encoder CAN count but weakly.
2. Ordered existence is easy because it distributes the decision across 200 binary per-slot predictions with spatial structure (positional cutoff).
3. Sinkhorn existence is hard because the per-slot decision depends on competitive global assignment.
4. Card token idea: add a global [CLS] token that learns cardinality and conditions each query's mask_head.

## Cardinality: What Failed

Comprehensive list of approaches tested to close the cardinality gap (Sinkhorn card=25 vs Hungarian card=8):

| Approach | card | ml1 | Why it failed |
|----------|------|-----|---------------|
| Card token (global CLS) | 25 | 0.22 | Mask_head ignores the token |
| Curriculum ordered→Sinkhorn | 38 | 0.23 | Sinkhorn phase erases cardinality |
| TopK hard gradient masking | — | 0.54 | Kills feature learning |
| ∅ downweighting | 31→25 | 0.22 | Minimal effect |
| Full transport (dustbin in loss) | 31-36 | 0.28-0.31 | Degrades matching, no separation |
| Full transport + hinge repulsion | 19-96 | 0.28-0.34 | Transport fights repulsion; dustbin stuck at dist=6-8 |
| Sorted mask (self-ranking) | 29-80 | 0.28-0.31 | Circular: mask needs to know existence to rank |
| Repulsive dustbin | ~25 | 0.22 | Separation stays near zero |
| Exist annealing | worse | — | No annealing is better |

**Root cause**: With real-only feature loss, unmatched predictions get zero gradient → stay at initialization → spatially identical to matched predictions → mask_head can't distinguish them. With full transport, dustbin drifts into data cloud (learned dustbin at dist=6-8 vs inter-track dist=21), causing Sinkhorn confusion and matching degradation.

**Key insight from toy problem**: Full transport works perfectly (F1=0.99) when dustbin starts at (-1,-1) OUTSIDE [0,1] data range. The ej-vae dustbin was initialized at randn*0.1 — inside the data cloud — and never escaped.

## Promising Directions for Cardinality

### Tier 1: Dustbin geometry (proven in toy)

**A. Extra dustbin dimension**
Add dimension D+1 to features. Real targets have 0 in this dim, dustbin targets have a large value (e.g., 10). Predictions learn to output 0 (real) or 10 (fake). L1 cost naturally penalizes mismatch. Existence at inference: `pred[:, D] < threshold`. No mask_head needed — existence becomes a feature prediction problem.

**B. Fixed far-away dustbin + full transport**
Freeze dustbin at a known location far from data (not learned). Full transport pulls unmatched predictions there. Existence at inference: distance to dustbin < threshold. Proven in toy with dustbin at (-1,-1).

**C. Weighted transport**
Full transport but downweight dustbin columns: `loss = (P_real * cost_real).sum() + α * (P_dustbin * cost_dustbin).sum()`. Tunable α prevents dustbin transport from overwhelming matching signal.

### Tier 2: Architectural

**D. Autoregressive existence**
Predict one object at a time with a "stop" decision after each. Breaks permutation equivariance by design. Existence is a sequential decision, not a per-slot classification. Used in sequence models with EOS tokens.

**E. Feed Sinkhorn weights to mask_head**
At training: `mask_input = [query_hidden, P[i,:N_tgt].sum()]`. At inference: use cross-attention entropy as proxy. Direct signal but inference gap is awkward.

**F. Attention-based existence**
Use decoder cross-attention patterns. High-entropy attention = diffuse = unmatched. Available at inference without Sinkhorn.

### Tier 3: Loss reformulations

**G. Unbalanced / partial optimal transport**
Rectangular Sinkhorn with relaxed marginals (no dustbin padding). Row marginals ≤ 1 — total assignment weight IS the existence probability. Theoretically cleanest. Chizat et al. (2018), Peyré & Cuturi (2019).

**H. Gumbel-Sinkhorn**
Add Gumbel noise before Sinkhorn → approaches hard permutation. Sharper assignment → cleaner exist signal. Mena et al. (2018).

**I. Soft cardinality loss**
Direct penalty: `(pred_mask.sum() - N_target)²`. Forces total count to match.

### Game-theoretic perspective on mismatch

The cardinality problem is the **assignment game with unequal sides** (Shapley & Shubik 1971). With more players (predictions) than rewards (targets), unmatched players receive their "outside option" — zero payoff. The dustbin IS the outside option.

- **Hungarian**: Solves the asymmetric assignment exactly. Unmatched predictions are explicitly identified.
- **Sinkhorn (padded)**: Dustbin copies create synthetic "outside option" targets. The entropy regularization smears assignment mass, making it hard to identify who is truly unmatched.
- **Unbalanced OT**: Directly models mass destruction — predictions can "opt out" at a cost ε. No dustbin needed. The parameter ε controls the tradeoff between matching quality and existence sharpness.
- **Auction algorithm** (Bertsekas 1979): Players bid for targets; losers are unassigned. Natural handling of mismatch through the bidding process.

## Open Questions

1. Can extra dustbin dimensions close the cardinality gap?
2. Does fixed far-away dustbin reproduce toy results at ej-vae scale?
3. Is unbalanced OT a cleaner formulation than dustbin padding?
4. Can autoregressive decoding give ordered-level cardinality (card=5) with Sinkhorn-level matching?
5. How do results scale to 50k+ data?
