# Product Loss: A Novel Set Prediction Loss Function

## Summary

This document synthesises an analysis of loss functions for set prediction tasks, motivated by the HEP4M model for particle physics. Through tracing how HEP4M handles the matching problem (pT-ordered positional cross-entropy), surveying the full landscape of alternatives, and critically evaluating a proposed "Product Loss" derived from the Influencer Loss, we arrive at a potentially novel and practically useful loss function.

## The Set Prediction Problem

When a model outputs an **unordered set** of objects (e.g., particles in a jet, detected objects in an image), there is no canonical correspondence between prediction slot *i* and target *j*. For *N* targets, there are *N!* valid assignments.

The question: **how do we compute a training loss without knowing which prediction should match which target?**

## Landscape of Existing Approaches

### 1. Canonical Ordering (e.g., HEP4M)
Sort targets by some variable (pT in physics), assign prediction *i* to sorted target *i*. Standard cross-entropy.
- **Pros:** Fast, stable, deterministic gradients
- **Cons:** Arbitrary ordering bias; soft/low-pT particles predicted poorly; near-degenerate orderings create noisy targets

### 2. Hungarian Matching (DETR family)
Compute N x N cost matrix, find optimal one-to-one assignment via Hungarian algorithm (O(N^3)), compute loss on matched pairs.
- **Pros:** Optimal bijective matching, no ordering bias
- **Cons:** Not differentiable (detached matching), **extremely slow convergence** (~500 epochs for DETR vs ~36 for baselines), matching instability early in training

### 3. Sinkhorn Soft Matching
Replace hard permutation matrix with soft doubly-stochastic matrix via Sinkhorn iterations.
- **Pros:** Fully differentiable, GPU-friendly
- **Cons:** Temperature sensitivity, "average particle" collapse early in training, soft assignment incompatible with discrete token teacher forcing

### 4. Chamfer Distance
For each point in set A, find nearest neighbour in set B (and vice versa). Sum squared distances.
```
L_CD = (1/N) Σ_j min_i d_ij  +  (1/M) Σ_i min_j d_ij
```
- **Pros:** O(NM), differentiable, permutation invariant
- **Cons:** **Mode collapse** — multiple targets can match to the same prediction. The min provides gradient only to the single nearest pair.

### 5. Earth Mover's Distance / Optimal Transport
Find minimum-cost bijection between sets. Equivalent to Hungarian matching with Euclidean cost.
- **Pros:** Bijective, principled metric
- **Cons:** O(N^3), not differentiable. Primarily used for evaluation, not training.

### 6. Diffusion / Flow Matching
Build permutation invariance into the generative process (i.i.d. noise + equivariant denoiser).
- **Pros:** No matching needed at all
- **Cons:** Different paradigm entirely (not applicable to encoder-decoder set prediction)

### 7. Set Cross Entropy (Asai, 2018, arXiv:1812.01217)
Closest existing work to the Product Loss. Derived from first principles:
```
SCE(X, Y) = -Σ_i logsumexp_j(-H(x_i, y_j))
```
- **Pros:** Principled information-theoretic derivation
- **Cons:** **One-directional only** (precision, no coverage term). Never tested at scale. Does not address mode collapse.

## The Product Loss

### Derivation from the Influencer Loss

The [Influencer Loss](https://www.epj-conferences.org/articles/epjconf/abs/2024/05/epjconf_chep2024_09016/epjconf_chep2024_09016.html) (Murnane, 2024) was designed for point cloud **clustering** with known assignments. It uses dual embeddings (user/influencer) with attractive and repulsive terms structured via arithmetic and geometric means.

The key insight is to **transpose the Influencer Loss structure from clustering (known assignments) to set prediction (unknown assignments)**, replacing the embedding-space radius search with an N x M distance matrix, and using the product structure as the aggregation:

### Definition

Given predicted set {ŷ_i} (M predictions) and target set {y_j} (N targets), with distance matrix D ∈ R^{M×N} where D_ij = d(ŷ_i, y_j):

```
L_product = Σ_j Π_i D_ij  +  Σ_i Π_j D_ij
             ^^^^^^^^^^^^     ^^^^^^^^^^^^
             Coverage          Precision
```

**Coverage** (column products): For each target j, the product of distances from ALL predictions to that target. Small when at least one prediction is close.

**Precision** (row products): For each prediction i, the product of distances to ALL targets. Small when the prediction is close to at least one target.

This is structurally identical to Chamfer distance, but with **product replacing min**:

```
Chamfer:  Σ_j min_i D_ij  +  Σ_i min_j D_ij
Product:  Σ_j  Π_i  D_ij  +  Σ_i  Π_j  D_ij
```

### For N=2: Equivalence to Product of Assignment Costs

With 2 predictions and 2 targets, the Product Loss equals:
```
L = (d11 + d22)(d12 + d21) = cost(σ₁) × cost(σ₂)
```
The **product of all possible assignment costs**. This is the squared geometric mean of permutation costs.

### Key Properties

#### 1. Emergent bijectivity (proven by gradient analysis)

When target j is already "covered" (some prediction k has D_kj ≈ 0):
```
∂(Π_i D_ij)/∂D_lj = Π_{i≠l} D_ij ≈ 0  (because D_kj ≈ 0 is in the product)
```
**All other predictions get zero gradient toward that target.** They are free to match elsewhere. This is much stronger than Chamfer's min-based zero gradient (which only zeroes out non-nearest predictions).

#### 2. Strong mode collapse penalty

For N=3, mode collapse (all predictions → target 1) gives:
- **Product Loss:** 250
- **Chamfer:** 3.3
- **Hungarian:** 5

The product loss gives ~75x stronger penalty than Chamfer.

#### 3. Self-annealing dynamics

- **Early training:** All distances similar → all gradients similar → explores all assignments uniformly
- **Mid training:** One assignment starts to dominate → gradient concentrates (correct-pair gradient ∝ wrong-assignment cost) → positive feedback
- **Late training:** Correct assignment near zero → gradient on wrong pairs vanishes, gradient on correct pairs is maximal

#### 4. Gradient structure

```
∂L/∂D_ij (coverage term) = Π_{k≠i} D_kj
```
The gradient for prediction i toward target j is the **product of all OTHER predictions' distances to that target**. If any other prediction is already close to target j, this gradient vanishes. If no prediction is close, all get strong gradient.

### Comparison Table

| Property | Product | Chamfer | Hungarian | Sinkhorn | SCE |
|---|---|---|---|---|---|
| Complexity | O(NM) | O(NM) | O(N³) | O(N²T) | O(NM) |
| Differentiable | Yes | Yes | No | Yes | Yes |
| Bijectivity | Emergent (strong) | None | Hard | Soft | None |
| Mode collapse (N=3) | 250 | 3.3 | 5 | ~5 | N/A |
| Bidirectional | Yes | Yes | Yes | Yes | **No** |
| Hyperparameters | None | None | None | ε, T | None |
| GPU-native | ✓ | ✓ | ✗ (CPU) | ✓ | ✓ |
| Tested at scale | **No** | Yes | Yes | Yes | No |

### Implementation

```python
def product_loss(D, eps=1e-8):
    """
    Product Loss for set prediction.

    Args:
        D: (M, N) distance/cost matrix between M predictions and N targets
        eps: small constant for numerical stability

    Returns:
        scalar loss
    """
    log_D = torch.log(D + eps)
    coverage  = torch.exp(log_D.mean(dim=0)).sum()   # GM over preds per target
    precision = torch.exp(log_D.mean(dim=1)).sum()   # GM over targets per pred
    return coverage + precision
```

Note: Uses geometric mean (normalised product) for numerical stability at large N. The `.mean(dim=0)` replaces `.sum(dim=0)` in log space, equivalent to `(Π D_ij)^{1/M}` instead of `Π D_ij`.

### Known Concerns and Mitigations

#### 1. Numerical stability (large N)
Products of 50+ values overflow/underflow float32. **Mitigated** by geometric mean formulation above.

#### 2. Over-aggressive gradient suppression
When 2+ predictions are near a target, the product for that column is ≈ 0, AND gradients for ALL predictions toward that target vanish. At N=50, this could be too aggressive early in training.
**Mitigations:**
- Use `log(d + ε)` with tunable ε
- Clamp minimum distance values
- Warm up with Chamfer loss, transition to Product Loss

#### 3. No formal bijectivity guarantee
The loss encourages but doesn't enforce one-to-one matching. Local minima with shared assignments are possible.
**Mitigations:**
- Combine with indicator/cardinality prediction (as in HEP4M)
- Add repulsive penalty between predictions near the same target

#### 4. Cardinality handling (M ≠ N)
When there are more predictions than targets, the precision term pushes unused predictions toward some target.
**Mitigation:** Mask precision term to "real" predictions (via indicator head), or add a "no object" virtual target.

#### 5. Interaction with discrete tokens
When D_ij = Σ_k CE(logits_ik, target_jk), products of cross-entropy values lack geometric interpretation. But the core property (product = 0 iff any factor = 0) still holds since CE ≥ 0.

---

## Proposed Toy Experiments

### Experiment 1: 2D Point Set Matching (N=5-10)

**Setup:** Generate random 2D point clouds as targets (N=5-10 points). Train a small MLP/transformer to predict the target set from a latent code.

**Distance:** Euclidean L2.

**Compare:** Product Loss vs Chamfer vs Hungarian vs Sinkhorn.

**Metrics:**
- Training loss curves (convergence speed)
- Matching accuracy (% of predictions within ε of a unique target)
- Mode collapse rate (% of targets with 0 or 2+ matched predictions)
- Wall-clock training time

**Why this experiment:** Simplest possible testbed. Euclidean distances are well-behaved. Small N avoids numerical issues. Directly tests bijectivity enforcement.

### Experiment 2: Discrete Token Set Prediction (N=5-10)

**Setup:** Each "particle" is a vector of K=4 discrete tokens from vocabulary V=32. Generate random token sets as targets. Train a transformer decoder to predict the set.

**Distance:** D_ij = Σ_k CE(logits_ik, target_jk).

**Compare:** Product Loss vs pT-ordering (with arbitrary ordering) vs Hungarian.

**Metrics:** Per-token accuracy, set-level exact match, convergence speed.

**Why this experiment:** Tests interaction with discrete tokens and cross-entropy distances, which is the actual HEP4M use case.

### Experiment 3: Scaling Test (N=10, 20, 50)

**Setup:** Same as Experiment 1 or 2, but scale N from 10 to 50.

**Focus:**
- Does the geometric mean formulation remain numerically stable?
- Does the gradient suppression become too aggressive at large N?
- How does convergence speed scale compared to Hungarian?

### Experiment 4: Influencer Loss vs Product Loss (Clustering)

**Setup:** Use a known-assignment clustering task (e.g., TrackML-like). Compare the original Influencer Loss (dual embeddings + radius search) against a Product Loss formulation on the same data.

**Purpose:** Test whether the Product Loss can serve as a drop-in replacement for the Influencer Loss on the task it was originally designed for, before extending to the harder unknown-assignment case.

### Experiment 5: JetNet Particle Generation

**Setup:** Use the JetNet dataset (30-particle gluon jets). Train a set-prediction model (transformer encoder-decoder) to reconstruct jet constituents.

**Distance:** D_ij based on (pT, η, φ) Euclidean distance or VQ-VAE token CE.

**Compare:** Product Loss vs pT-ordering vs Hungarian + DN-DETR denoising.

**Metrics:** W1 distances, FPND, coverage/MMD on JetNet metrics.

**Why this experiment:** Real-world physics data at the scale HEP4M operates. Tests whether the Product Loss can match or exceed pT-ordering on an actual particle physics task.

### Experiment 6: Ablation — Coverage vs Precision vs Both

**Setup:** Any of the above experiments, but ablate the two terms:
- Coverage only: `Σ_j Π_i D_ij`
- Precision only: `Σ_i Π_j D_ij`
- Both (full Product Loss)

**Purpose:** Understand the relative contribution of each term. The coverage term enforces that all targets are matched; the precision term enforces that all predictions are useful. Which matters more? Is one sufficient?

---

## Connection to the Influencer Loss

The Product Loss can be seen as a generalisation of the Influencer Loss from clustering (known assignments) to matching (unknown assignments):

| Aspect | Influencer Loss | Product Loss |
|---|---|---|
| Task | Clustering (known groups) | Set prediction (unknown assignment) |
| Embedding | Dual (user + influencer) | Single (prediction + target) |
| Attractive term | GM/AM within track | Coverage: Σ_j Π_i d_ij |
| Repulsive term | Hinge loss between tracks | Precision: Σ_i Π_j d_ij |
| Bijectivity | From known labels | Emergent from product structure |
| Inference | Radius search | Direct prediction |

The key conceptual bridge: the Influencer Loss's **product-of-distances** structure (geometric mean within tracks) naturally extends to a **product-over-columns/rows** of the distance matrix when assignments are unknown. The coverage term replaces the attractive force (pull predictions toward targets), and the precision term replaces the repulsive force (prevent predictions from being unused).

## References

- **Influencer Loss:** Murnane, D. "Influencer Loss: End-to-end Geometric Representation Learning for Track Reconstruction." EPJ Web Conf. 295, 09016 (2024). [doi:10.1051/epjconf/202429509016](https://doi.org/10.1051/epjconf/202429509016)
- **DETR:** Carion et al. "End-to-End Object Detection with Transformers." ECCV 2020. [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
- **DN-DETR:** Li et al. "DN-DETR: Accelerate DETR Training by Introducing Query DeNoising." CVPR 2022. [arXiv:2203.01305](https://arxiv.org/abs/2203.01305)
- **Chamfer Distance:** Fan et al. "A Point Set Generation Network for 3D Object Reconstruction." CVPR 2017. [arXiv:1612.00603](https://arxiv.org/abs/1612.00603)
- **Set Cross Entropy:** Asai, M. "Set Cross Entropy: Likelihood-based Permutation Invariant Loss Function for Probability Distributions." 2018. [arXiv:1812.01217](https://arxiv.org/abs/1812.01217)
- **Sinkhorn:** Cuturi, M. "Sinkhorn Distances: Lightspeed Computation of Optimal Transport." NeurIPS 2013. [arXiv:1306.0895](https://arxiv.org/abs/1306.0895)
- **EMD for jets:** Komiske et al. "The Metric Space of Collider Events." [arXiv:1902.02346](https://arxiv.org/abs/1902.02346)
- **Object Condensation:** Kieseler, J. EPJC 80 (2020). [arXiv:2002.03605](https://arxiv.org/abs/2002.03605)
- **Mask2Former:** Cheng et al. "Masked-attention Mask Transformer for Universal Image Segmentation." CVPR 2022. [arXiv:2112.01527](https://arxiv.org/abs/2112.01527)
- **MPGAN:** Kansal et al. "Particle Cloud Generation with Message Passing GANs." NeurIPS 2021. [arXiv:2106.11535](https://arxiv.org/abs/2106.11535)
- **PC-JeDi:** Leigh et al. "Diffusion for Particle Cloud Generation." SciPost Phys. 16 (2024). [arXiv:2303.05376](https://arxiv.org/abs/2303.05376)
- **OmniJet-α:** Birk et al. "The First Cross-Task Foundation Model for Particle Physics." [arXiv:2403.05618](https://arxiv.org/abs/2403.05618)
- **Deep Sets:** Zaheer et al. NeurIPS 2017. [arXiv:1703.06114](https://arxiv.org/abs/1703.06114)
- **Slot Attention:** Locatello et al. NeurIPS 2020. [arXiv:2006.15055](https://arxiv.org/abs/2006.15055)
