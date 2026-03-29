# Session Summary: From HEP4M Loss Analysis to Power-SoftMin

## The Journey

### Starting Point
Traced HEP4M's loss: pT-sorted positional cross-entropy. Asked whether
ordering, matching, or something else was used. Found it's pure ordering.

### Literature Survey
Comprehensive review of 7 paradigms: ordering, Hungarian, Sinkhorn, Chamfer,
EMD, diffusion/flow, GANs. Covered DETR, DN-DETR, Mask2Former, Slot Attention,
Object Condensation, Set Cross Entropy, and 50+ papers.

### The Influencer Loss Connection
The Influencer Loss (Murnane 2024) uses dual embeddings with GM/AM structure
for clustering. Transposing from clustering (known assignments) to set
prediction (unknown assignments) led to the Product Loss proposal.

### The Product Loss: Beautiful Theory, Practical Failure
`L = Σ_j Π_i D_ij + Σ_i Π_j D_ij` — proven to have correct fixed points,
emergent bijectivity, strong mode collapse penalty. Factors as product of
assignment costs for N=2. Scales as O(NM).

**But it completely fails in practice.** The geometric mean gives uniform
gradients at initialization. No transform of D fixes this (proven: any
symmetric function of equal inputs has equal partial derivatives).

### Rescue Attempts (All Failed)
- Log-product, Huber-product, sigmoid-product: same uniform gradient problem
- Warm-start (Chamfer→Product): Product actively unlearns Chamfer's matches
- Annealed exponent (p: 10→0.5): stalls when approaching GM regime
- Exact permanent (Ryser's formula): theoretically optimal, same practical failure
- DCD: exponential saturation + non-differentiable argmin
- Soft DCD, Combined losses: over-correct, worse than baselines

### What Works: SoftMin Family
SoftMin Chamfer provides exponential sensitivity via softmax(-D/τ). This
breaks symmetry at initialization — the key property products lack.

PW-SoftMin adds detached GM reweighting: within 1.3% of Hungarian at 2× speed.

### Power-SoftMin: The Real Discovery
`L = Σ_j softmin_j(D)^p` — one-line change from standard softmin.
At p=3, matches Hungarian within 0.7% at 2.5× speed on toy data.
Confirmed on MNIST digit reconstruction (0.079 vs Hungarian 0.078).
Loss computation 9.5× faster than Hungarian at N=100.

### Benchmark Pivot
PointSWD (point cloud reconstruction) is the WRONG benchmark — it tests
shape quality, not identity matching. MNIST point sets and CLEVR bounding
boxes are the RIGHT benchmarks. Both set up and scaffolded.

## Key Dead Ends
1. Pure product/GM loss (uniform gradients — fundamental, not fixable)
2. DCD (exponential saturation, non-differentiable)
3. Permanent (exact partition function — same gradient problem as product)
4. Log-Chamfer/Log-ProductSoftMin (amplifies covered targets, not uncovered)
5. Token experiments with flat MLP (task too hard, results inconclusive)
6. PointSWD benchmark (wrong evaluation for identity-matching losses)

## Key Findings
1. Product structure is correct at convergence, useless at initialization
2. Exponential sensitivity (softmax) is necessary for symmetry breaking
3. Power wrapping creates coverage enforcement through the gradient
4. DETR uses bounded probabilities for matching, not raw CE
5. The exact partition function (permanent) is NOT the optimal loss
6. Temperature must be calibrated to the distance scale

## Open Elephants
1. **Never tested on HEP4M itself** — the original motivation
2. **Token/discrete distances unsolved** — CE needs different τ, temperature varies during training
3. **Variable cardinality untested** — interaction with indicator predictions unknown
4. **Power p is a hyperparameter** — p=3 best at N=20, p=2 best at N=100
5. **Unknown if ordering bias removal actually helps physics** — pT ordering may be useful
6. **Convergence speed advantage at scale unknown** — toy experiments only

## Assets Produced
- 20+ loss function implementations in `influencerformer/losses/`
- Lightning training module with per-component timing
- MNIST point set benchmark (N=100, real data)
- CLEVR bounding box scaffold (synthetic data fallback)
- PointSWD integration (ready for GPU)
- 30-slide Beamer deck on set prediction losses
- Comprehensive documentation in `docs/product_loss_analysis.md`

## The Mathematical Connection (see `docs/product_softmin_connection.md`)
Power-SoftMin IS the Product Loss, with:
1. Inner aggregation: GM → SoftMin (for gradient quality)
2. Outer aggregation: Π → power (for numerical stability)
3. The mathematical bridge: power-softmin ≈ constant + p × log(product of softmins)
