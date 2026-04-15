# Set Prediction Matching Loss Taxonomy

A comprehensive taxonomy of matching mechanisms and existence/cardinality handling for set prediction tasks with variable output cardinality.

## The Problem

Given N_pred fixed output slots and M < N_pred target objects, we need to:
1. **Match** predictions to targets (permutation-invariant loss)
2. **Identify existence** — which slots contain real objects vs empty

## Taxonomy

### Axis 1: Matching Mechanism

| Category | Method | 1-to-1 | Differentiable | Complexity | Key Insight |
|----------|--------|--------|---------------|------------|-------------|
| **Hard assignment** | Hungarian | Yes | No (through assignment) | O(N³) CPU | Optimal bipartite matching; gold standard |
| | Auction algorithm | Yes | No | O(N²) | Faster alternative to Hungarian |
| **Optimal Transport** | Sinkhorn (entropic OT) | Yes (soft) | Yes | O(K·N²) GPU | Doubly-stochastic matrix via iterative normalization |
| | Unbalanced OT | Partial | Yes | O(K·N²) GPU | Relaxes marginal constraints for variable cardinality |
| **Soft matching (1-dir)** | Chamfer | No | Yes | O(N²) | Hard min in each direction; mode collapse |
| | SoftMin Chamfer | No | Yes | O(N²) | Softmax replaces hard min; temperature-sensitive |
| **Soft matching (2-dir)** | Power-SoftMin (PM3) | No | Yes | O(N²) | Two independent softmaxes + power amplification |
| | PW-SoftMin | No | Yes | O(N²) | GM-weighted softmin; best fast loss at equal cardinality |
| **Product-based** | Product Loss | Implicit | Yes | O(N²) | GM Chamfer; correct fixed points but uniform gradients |
| **Attention-based** | Slot Attention | Emergent | Yes | O(K·N²) | Iterative softmax competition between slots |
| **Match-free** | DN-DETR denoising | N/A | Yes | O(N) | Pre-matched noisy GT queries bypass assignment |

**Key finding from our experiments:** At equal cardinality (N_pred = N_target), Sinkhorn matches Hungarian quality (ml2=0.009 vs 0.0004). PM3/PW-SoftMin are 4x worse (ml2=0.036-0.042) because their independent softmaxes don't enforce 1-to-1. At variable cardinality, only Sinkhorn and Hungarian can handle existence.

### Axis 2: Existence / Variable Cardinality Handling

| Approach | Mechanism | Works with soft matching? | Differentiable? | Key paper |
|----------|-----------|--------------------------|-----------------|-----------|
| **"No object" class** | Pad targets to N_pred with Ø class; match all | Hungarian only (DETR) | Through loss, not matching | DETR (Carion 2020) |
| **Dustbin (score-based)** | Extra row+col in (N+1)×(M+1) matrix; relaxed marginals | Sinkhorn | Yes | SuperGlue (Sarlin 2020) |
| **Dustbin (spatial, K copies)** | Pad targets with K copies of learned "empty" embedding | Sinkhorn, softmax | Yes | This work |
| **Unbalanced OT** | Relax marginal constraints; mass can be "destroyed" | UOT | Yes | De Plaen (CVPR 2023) |
| **Separate exist head** | Binary classifier per slot; trained with BCE | Any | Yes | Various |
| **Cardinality prediction** | Auxiliary head predicts count; top-k by cost | Any | Partially | DSPN variants |
| **Slot competition** | Attention-based; slots with low attention are "empty" | Slot Attention | Yes | Locatello (2020) |
| **Greedy assignment** | CPU greedy 1-to-1 for exist targets; detached | Any | No (targets detached) | This work (ej-vae) |

**Key finding:** On our toy problem, Sinkhorn + K degenerate dustbin copies (pad targets to N_pred with copies of a learned point) achieves F1=0.99 for existence. No special existence head needed — the Sinkhorn 1-to-1 assignment naturally routes excess predictions to dustbin slots.

### Axis 3: Training Stability

| Issue | Affected methods | Solution | Paper |
|-------|-----------------|----------|-------|
| Slow convergence | DETR (Hungarian) | Denoising queries (DN-DETR) | Li (CVPR 2022) |
| Matching instability | All bipartite | Group queries (Group DETR) | Chen (ICCV 2023) |
| Centroid collapse | Chamfer, SoftMin at high τ | Power amplification (p>1) | This work |
| Existence collapse (n_pred→0) | Dustbin with softmax PM3 | Sinkhorn (enforces 1-to-1) | This work |
| τ sensitivity | All temperature-based | τ ∝ cost scale | This work |

## Key Papers

### Foundational
- **DETR** (Carion et al., ECCV 2020): Hungarian matching for set prediction. [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
- **SuperGlue** (Sarlin et al., CVPR 2020): Sinkhorn + dustbin for partial matching. [arXiv:1911.11763](https://arxiv.org/abs/1911.11763)
- **Sinkhorn Distances** (Cuturi, NeurIPS 2013): Entropic OT. [arXiv:1306.0895](https://arxiv.org/abs/1306.0895)

### OT for Detection
- **Unbalanced OT for Object Detection** (De Plaen et al., CVPR 2023): UOT unifies detection matching. [arXiv:2307.02402](https://arxiv.org/abs/2307.02402)
- **RTP-DETR** (Zareapoor et al., 2025): Sinkhorn fractional matching; +3.8% mAP over Deformable DETR. [arXiv:2503.04107](https://arxiv.org/abs/2503.04107)
- **OTA** (Ge et al., ICCV 2021): Sinkhorn-Knopp for label assignment. [arXiv:2103.14259](https://arxiv.org/abs/2103.14259)

### DETR Improvements
- **DN-DETR** (Li et al., CVPR 2022): Denoising queries for stable matching. [arXiv:2203.01305](https://arxiv.org/abs/2203.01305)
- **DINO** (Zhang et al., ICLR 2023): Improved denoising + contrastive. [arXiv:2203.03605](https://arxiv.org/abs/2203.03605)
- **Group DETR** (Chen et al., ICCV 2023): One-to-many during training. [arXiv:2207.13085](https://arxiv.org/abs/2207.13085)

### Object-Centric
- **Slot Attention** (Locatello et al., NeurIPS 2020): Attention-based object discovery. [arXiv:2006.15055](https://arxiv.org/abs/2006.15055)
- **DSPN** (Zhang et al., NeurIPS 2019): Deep Set Prediction Networks. [arXiv:1906.06565](https://arxiv.org/abs/1906.06565)

### Set Prediction Losses
- **PW-SoftMin / Power-SoftMin** (Murnane, 2024): Product-weighted softmin; Pareto-optimal fast loss. [doi:10.1051/epjconf/202429509016](https://doi.org/10.1051/epjconf/202429509016)
- **DCD** (Wu et al., NeurIPS 2021): Density-aware Chamfer. [arXiv:2111.12702](https://arxiv.org/abs/2111.12702)

## Our Position

Our work connects the product loss theory (Murnane 2024) to optimal transport:

1. **Product Loss ≡ Power-SoftMin** structurally (softmin replaces GM as inner diagnostic)
2. **PM3 softmax can't enforce 1-to-1** — two independent singly-stochastic matrices
3. **Sinkhorn enforces 1-to-1** via doubly-stochastic normalization — solves existence
4. **Sinkhorn + K degenerate dustbin copies** = simplest variable-cardinality OT; F1=0.99 on toy
5. **Power amplification** helps matching quality (ml2: 0.055 → 0.0001) orthogonal to existence

The closest prior work is SuperGlue (Sinkhorn + dustbin, 2020) and Unbalanced OT for Detection (CVPR 2023). Our contribution: (a) connecting the product loss / power-SoftMin theory to OT, (b) demonstrating that simple K-copy dustbin padding outperforms SuperGlue's score-based dustbin, (c) applying to physics (particle jet reconstruction).
