# Set Prediction Loss Functions: From Product Loss to PW-SoftMin

## Summary

This document synthesises an analysis of loss functions for set prediction tasks, motivated by the HEP4M model for particle physics. Starting from the Influencer Loss (Murnane, 2024) and its product-of-distances structure, we derived a theoretical "Product Loss" for set prediction, proved it has the correct fixed points, discovered it fails catastrophically in practice due to uniform gradients at initialization, and through systematic experimentation arrived at **PW-SoftMin** — a Product-Weighted SoftMin Chamfer loss that is the Pareto-optimal fast differentiable loss for set prediction.

## The Set Prediction Problem

When a model outputs an **unordered set** of objects (e.g., particles in a jet, detected objects in an image), there is no canonical correspondence between prediction slot *i* and target *j*. For *N* targets, there are *N!* valid assignments.

The question: **how do we compute a training loss without knowing which prediction should match which target?**

## Landscape of Existing Approaches

### 1. Canonical Ordering (e.g., HEP4M)
Sort targets by some variable (pT in physics), assign prediction *i* to sorted target *i*.
- **Pros:** Fast, stable, deterministic gradients
- **Cons:** Arbitrary ordering bias; soft particles predicted poorly

### 2. Hungarian Matching (DETR family)
Compute N×N cost matrix, find optimal one-to-one assignment via Hungarian algorithm O(N³).
- **Pros:** Optimal bijective matching, no ordering bias
- **Cons:** Not differentiable (detached matching), slow convergence (~500 epochs for DETR)

### 3. Sinkhorn Soft Matching
Soft doubly-stochastic matrix via Sinkhorn iterations.
- **Pros:** Fully differentiable, GPU-friendly
- **Cons:** Temperature sensitivity, "average particle" collapse early in training

### 4. Chamfer Distance
For each point, find nearest neighbour in the other set. Sum distances.
- **Pros:** O(NM), differentiable, permutation invariant
- **Cons:** **Mode collapse** — multiple targets can match the same prediction

### 5. Earth Mover's Distance
Minimum-cost bijection. O(N³), not differentiable. Primarily used for evaluation.

### 6. Density-aware Chamfer Distance (DCD, Wu et al. 2021)
Chamfer with `1/n` query-frequency weighting to penalise duplication.
- **Pros:** Direct anti-duplication mechanism
- **Cons:** Hard NN (argmin) is non-differentiable; alpha hyperparameter is scale-dependent

### 7. Set Cross Entropy (Asai, 2018)
Likelihood-based permutation-invariant loss: `-Σ_i logsumexp_j(-H(x_i, y_j))`.
- **Pros:** Principled information-theoretic derivation
- **Cons:** One-directional only (precision, no coverage); never tested at scale

## The Product Loss: Theory

### Derivation from the Influencer Loss

The [Influencer Loss](https://doi.org/10.1051/epjconf/202429509016) (Murnane, 2024) uses dual embeddings with attractive/repulsive terms structured via geometric means. Transposing this from clustering (known assignments) to set prediction (unknown assignments):

```
L_product = Σ_j Π_i D_ij  +  Σ_i Π_j D_ij
             coverage          precision
```

This is Chamfer with **product replacing min**. For N=2, it equals the product of all assignment costs: `(d11+d22)(d12+d21)`.

### Theoretical Properties (correct but insufficient)

1. **Emergent bijectivity**: When target j is covered (D_kj ≈ 0), gradient for all other predictions toward j vanishes: `∂(Π_i D_ij)/∂D_lj = Π_{i≠l} D_ij ≈ 0`
2. **Strong mode collapse penalty**: For N=3, mode collapse gives loss 250 vs Chamfer's 3.3
3. **Self-annealing**: Correct-pair gradient ∝ wrong-assignment cost

### Why It Fails in Practice

The gradient is:
```
∂/∂D_kj [GM_i(D_ij)] = (1/M) × GM / D_kj
```

When all `D ≈ c` (initialization), this simplifies to `1/M` for ALL entries. **Uniform gradient — no matching signal.** The model cannot break symmetry to discover assignments.

This is the same chicken-and-egg as Hungarian matching, but worse: Hungarian at least picks *a* matching (even if unstable), while GM provides *no* directional signal.

All rescue attempts failed:
- **Log-product** (`Π log(1+D)`): log compression doesn't change GM's flatness
- **Huber-product**: same problem
- **Sigmoid-product** (`Π σ(α(D-m))`): partial learning (sigmoid creates binary transition) but product still kills it — best result was 1.048 vs random 1.23
- **Warm-start** (Chamfer → Product): Product actively unlearns Chamfer's matches
- **Annealed exponent** (p: 10→0.5): starts Chamfer-like but stalls as p→GM

## What Works: SoftMin and PW-SoftMin

### SoftMin Chamfer

Replace hard `min` with temperature-scaled `softmin`:
```
softmin_j = Σ_i softmax(-D_ij/τ)_i × D_ij
```

`softmax(-D/τ)` is **exponentially sensitive** to differences in D. Even when D ≈ 1.6, the prediction with D=1.5 vs D=1.7 gets `exp(2/0.1) ≈ e²⁰ ≈ 5×10⁸` times more weight. This breaks symmetry at initialization.

### PW-SoftMin (Product-Weighted SoftMin)

Combines SoftMin matching with detached GM reweighting:
```python
softmin_cov = (softmax(-D/τ, dim=preds) * D).sum(dim=preds)  # per-target loss
gm_weight = GM_over_preds(D).detach()                         # coverage score
loss = mean(gm_weight * softmin_cov)                           # upweight uncovered targets
```

The GM serves as a **diagnostic** (which targets are uncovered?), not a gradient source. SoftMin provides the matching signal; GM steers the gradient budget.

## Experimental Results

### N=10 (300 epochs, 3 seeds)

```
Loss                  Match Dist       Dup Rate       Time
sinkhorn          0.670 ± 0.009    0.1% ± 0.0%     55s
hungarian         0.672 ± 0.005    0.2% ± 0.1%     45s
pw_softmin        0.681 ± 0.005    0.2% ± 0.1%     22s  ← best fast
softmin_0.1       0.693 ± 0.004    0.6% ± 0.1%     20s
chamfer           0.697 ± 0.001    1.0% ± 0.2%     18s
combined          0.739 ± 0.024    4.4% ± 1.0%     24s
dcd_6             0.780 ± 0.003    1.4% ± 0.2%     23s
soft_dcd          0.807 ± 0.029    4.6% ± 0.4%     22s
product_gm        1.234 ± 0.003    4.7% ± 0.3%     16s  ✗
```

### N=20 (500 epochs, 3 seeds)

```
Loss                  Match Dist       Dup Rate       Time
hungarian         0.554 ± 0.001    2.5% ± 0.3%     93s
sinkhorn          0.554 ± 0.003    2.3% ± 0.3%    159s
pw_softmin        0.565 ± 0.002    3.2% ± 0.4%     44s  ← best fast
softmin_0.1       0.580 ± 0.003    5.7% ± 0.1%     42s
chamfer           0.583 ± 0.003    7.0% ± 0.2%     38s
combined          0.606 ± 0.013    6.5% ± 0.9%     49s
dcd_6             0.646 ± 0.001   10.3% ± 0.3%     49s
soft_dcd          0.684 ± 0.027    4.3% ± 0.4%     46s
```

### Key Findings

1. **PW-SoftMin is the Pareto-optimal fast differentiable loss.** At N=20 it closes 53% of the Chamfer→Hungarian gap at Chamfer speed, with duplicate rate 3.2% vs Chamfer's 7.0%.

2. **Duplicate rate scales worse with N for Chamfer** (1.0% → 7.0%) **than PW-SoftMin** (0.2% → 3.2%). The GM reweighting provides real value at larger N.

3. **DCD underperforms Chamfer** in our setting. The hard argmin + non-differentiable matching + alpha scale-dependence make it unsuitable for this task. The query-frequency idea is sound but the implementation is too rigid.

4. **Soft DCD and Combined losses over-correct.** The query-frequency weighting is too aggressive — it disrupts the softmin's natural matching signal. The GM reweighting in PW-SoftMin works because it's gentle.

5. **Hungarian and Sinkhorn are 2-3× slower** but only ~2% better. For tasks where training time matters (large datasets, many epochs), PW-SoftMin is the practical choice.

## Architecture of Losses

All losses in this package operate on a pairwise distance matrix `D: (B, M, N)` and are composable:

```
                    ┌─────────────┐
    D (B,M,N) ──→  │  Aggregation │ ──→ scalar loss
                    └─────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
          min (Chamfer)  softmin    product/GM
              │           │           │
              └───────────┼───────────┘
                          │
                    ┌─────┴──────┐
                    ▼            ▼
              Direct loss   Reweighting
              (gradients)   (detached)
```

The key insight: **use softmin for matching (gradient source) and product/GM for reweighting (detached diagnostic)**. Don't mix them — the product is too flat for gradients but valuable as a coverage signal.

## Connection to the Influencer Loss

| Aspect | Influencer Loss | PW-SoftMin |
|---|---|---|
| Task | Clustering (known groups) | Set prediction (unknown assignment) |
| Matching | Dual embeddings (user/influencer) | SoftMin on distance matrix |
| Bijectivity | Repulsive hinge between tracks | GM coverage reweighting (detached) |
| Coverage | GM attractive within track | GM across predictions per target |
| Differentiable | Yes (embedding space) | Yes (distance matrix) |
| Complexity | O(NM) | O(NM) |

PW-SoftMin inherits the Influencer Loss's **product-of-distances structure** as a coverage diagnostic, while replacing the embedding-space matching with softmin on a distance matrix. The key conceptual bridge: the GM serves as a "how covered is this target?" signal in both cases.

## Implementation

```python
import torch
import torch.nn as nn

class ProductWeightedSoftMinLoss(nn.Module):
    def __init__(self, temperature=0.1, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, D):  # D: (B, M, N)
        # SoftMin matching
        w_cov = torch.softmax(-D / self.temperature, dim=1)
        softmin_cov = (w_cov * D).sum(dim=1)  # (B, N)

        w_prec = torch.softmax(-D / self.temperature, dim=2)
        softmin_prec = (w_prec * D).sum(dim=2)  # (B, M)

        # GM coverage reweighting (detached)
        log_D = torch.log(D + self.eps)
        col_gm = torch.exp(log_D.mean(dim=1)).detach()
        row_gm = torch.exp(log_D.mean(dim=2)).detach()

        col_w = col_gm / (col_gm.mean(-1, keepdim=True) + self.eps)
        row_w = row_gm / (row_gm.mean(-1, keepdim=True) + self.eps)

        coverage = (col_w * softmin_cov).mean(-1)
        precision = (row_w * softmin_prec).mean(-1)
        return (coverage + precision).mean()
```

## References

- **Influencer Loss:** Murnane, D. EPJ Web Conf. 295, 09016 (2024). [doi:10.1051/epjconf/202429509016](https://doi.org/10.1051/epjconf/202429509016)
- **DETR:** Carion et al. ECCV 2020. [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
- **Chamfer Distance:** Fan et al. CVPR 2017. [arXiv:1612.00603](https://arxiv.org/abs/1612.00603)
- **DCD:** Wu et al. NeurIPS 2021. [arXiv:2111.12702](https://arxiv.org/abs/2111.12702)
- **Set Cross Entropy:** Asai, M. 2018. [arXiv:1812.01217](https://arxiv.org/abs/1812.01217)
- **Sinkhorn:** Cuturi, M. NeurIPS 2013. [arXiv:1306.0895](https://arxiv.org/abs/1306.0895)
- **DN-DETR:** Li et al. CVPR 2022. [arXiv:2203.01305](https://arxiv.org/abs/2203.01305)
- **Mask2Former:** Cheng et al. CVPR 2022. [arXiv:2112.01527](https://arxiv.org/abs/2112.01527)
- **PointSWD:** Nguyen et al. ICCV 2021. [arXiv:2102.04014](https://arxiv.org/abs/2102.04014)
- **DSPN:** Zhang & Hare. NeurIPS 2019. [arXiv:1906.06565](https://arxiv.org/abs/1906.06565)
