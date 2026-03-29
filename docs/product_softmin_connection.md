# The Mathematical Connection Between Product Loss and Power-SoftMin

## The Puzzle

The Product Loss has beautiful theoretical properties (correct fixed points, emergent bijectivity, strong mode collapse penalty) but fails in practice due to uniform gradients at initialization.

Power-SoftMin works brilliantly in practice but was discovered empirically (just "add `.pow(3)`").

Are these the same thing in disguise? Can we find a first-principles derivation that connects them?

## The Answer: Yes

**Power-SoftMin IS the Product Loss**, with two modifications:
1. The inner aggregation (per-column) is replaced: GM → SoftMin
2. The outer aggregation (across columns) is the same: convex amplification

And these two formulations are connected by a precise mathematical identity.

## Derivation

### Step 1: Decompose the Product Loss

The coverage term of the Product Loss is:

```
L_product = Σ_j Π_i D_ij
```

Rewrite each column product in log space:

```
Π_i D_ij = exp(Σ_i log D_ij) = exp(M × mean_i(log D_ij))
```

So the product loss is:

```
L_product = Σ_j exp(M × μ_j)    where μ_j = mean_i(log D_ij) = log(GM_j)
```

This has two components:
- **Inner**: μ_j = mean(log D) — the log-geometric-mean, a **coverage diagnostic** (large when target j is uncovered)
- **Outer**: exp(M × ·) — a **convex amplifier** that converts the additive diagnostic into a multiplicative penalty

### Step 2: Decompose Power-SoftMin

The coverage term of Power-SoftMin is:

```
L_power = Σ_j (softmin_i(D_ij))^p
```

Rewrite in log space:

```
(softmin_j)^p = exp(p × log(softmin_j))
```

So:

```
L_power = Σ_j exp(p × ν_j)    where ν_j = log(softmin_j)
```

Same two components:
- **Inner**: ν_j = log(softmin) — the log-softmin, a **coverage diagnostic** (large when target j is uncovered)
- **Outer**: exp(p × ·) — a **convex amplifier** (same structure, different scale)

### Step 3: The Structural Equivalence

Both losses have the form:

```
L = Σ_j exp(α × diagnostic_j)
```

| | Product Loss | Power-SoftMin |
|---|---|---|
| **Inner diagnostic** | μ_j = mean(log D_ij) | ν_j = log(softmin(D_ij)) |
| **Outer amplifier** | exp(M × ·) | exp(p × ·) |
| **Amplification strength** | M (set size, fixed) | p (hyperparameter, tunable) |

The ONLY structural difference is the inner diagnostic function.

### Step 4: Why the Inner Diagnostic Matters

**Product's diagnostic** μ_j = mean(log D):
- Gradient: ∂μ_j/∂D_kj = 1/(M × D_kj)
- At initialization (all D ≈ c): gradient = 1/(Mc) for ALL k
- **Uniform. No matching signal.**

**SoftMin's diagnostic** ν_j = log(softmin(D)):
- Gradient: ∂ν_j/∂D_kj = softmax(-D/τ)_kj × (1/softmin_j) × (correction terms)
- At initialization: gradient ∝ exp(-D_kj/τ), which varies EXPONENTIALLY across k
- **Non-uniform. Strong matching signal.**

The mean-of-logs is SYMMETRIC in its arguments (permutation-invariant), so it cannot distinguish between entries when they're similar. The softmin uses softmax weights that are EXPONENTIALLY sensitive to differences.

### Step 5: The Taylor Connection

For small perturbations δ around a converged solution where softmin ≈ min:

```
softmin(D)^p = (min D + correction)^p ≈ (min D)^p × (1 + p × correction/min D)
```

And:

```
GM(D) = (Π D_i)^{1/M} = min D × (Π_{i≠k*} D_i/min D)^{1/M}
```

Near convergence where one entry dominates:
```
softmin ≈ min D ≈ GM^{M/(M-1)} × (small corrections)
```

Both diagnostics converge to the same value (the minimum distance) as training progresses. The difference is purely in the EARLY training dynamics.

### Step 6: The Linearization Identity

For power-softmin with small p, using the Taylor expansion exp(px) ≈ 1 + px:

```
Σ_j softmin_j^p = Σ_j exp(p × log softmin_j)
                ≈ Σ_j (1 + p × log softmin_j)
                = N + p × Σ_j log(softmin_j)
                = N + p × log(Π_j softmin_j)
```

**Power-SoftMin ≈ constant + p × log(Product of SoftMins)**

The power parameter p controls how strongly the product-like behavior
(coverage enforcement via the Π over targets) influences the loss.
At p=1, there's no coverage pressure. At p=3, the coverage pressure
is strong enough to match Hungarian's bijectivity.

## Why No Transform of D Can Fix the Product

**Theorem**: For any elementwise transform f: ℝ→ℝ, the gradient of Σ_j Π_i f(D_ij) with respect to D_kj, evaluated at D_ij = c for all i,j, is identical for all k.

**Proof**:
```
∂/∂D_kj [Σ_j Π_i f(D_ij)] = f'(D_kj) × Π_{i≠k} f(D_ij)
```

At D_ij = c for all i,j:
```
= f'(c) × f(c)^{M-1}
```

This is independent of k. QED.

**Corollary**: No choice of f(D) = D², log(D), exp(D), sigmoid(D), Huber(D), or any other elementwise function can make the product loss discriminate between predictions at initialization.

**The escape**: The function must depend on MORE than just D_kj. It must depend on the RELATIVE values of D in the column — i.e., it must be a function of D_kj relative to D_*j. This is exactly what softmax does:

```
softmax(-D/τ)_kj = exp(-D_kj/τ) / Σ_l exp(-D_lj/τ)
```

The denominator creates COMPETITION between predictions. The product has no such competition mechanism.

## The Product-SoftMin Spectrum

We can now understand the full spectrum of losses as variations of:

```
L = Σ_j F(agg_j(D_*j))
```

where `agg` is the inner (per-column) aggregation and `F` is the outer (across-column) amplifier:

| Loss | agg (inner) | F (outer) | Works? |
|---|---|---|---|
| Chamfer | min | identity | ✓ (baseline) |
| SoftMin | softmin | identity | ✓ (smooth Chamfer) |
| Product (GM) | geometric mean | identity | ✗ (uniform grad) |
| PW-SoftMin | softmin | softmin × detached(GM) | ✓ (GM reweighting) |
| Power-SoftMin | softmin | x^p | ✓ (BEST) |
| Product Loss | geometric mean | exp(M×·) | ✗ (uniform grad) |
| "Fixed Product" | **softmin** | **exp(p×·)** | **?? (untested!)** |

The "Fixed Product" would be:

```
L = Σ_j exp(p × log(softmin_j)) = Σ_j softmin_j^p = Power-SoftMin
```

**They're the same thing!** Power-SoftMin IS the "fixed" Product Loss — the one where we replace the broken inner aggregation (GM) with the working one (softmin) while keeping the product-like outer amplification (exponentiation).

## Can We Go Further? The Exponential Amplifier

Power-SoftMin uses F(x) = x^p. But the original product uses F(x) = exp(M×x) (in log space). What if we use F(x) = exp(α × softmin)?

```
L = Σ_j exp(α × softmin_j)
```

This would give gradient:
```
∂L/∂D_kj = α × exp(α × softmin_j) × ∂softmin_j/∂D_kj
```

The exp(α × sm_j) factor EXPONENTIALLY amplifies uncovered targets. At α=1 with sm≈0.5 (uncovered) vs sm≈0.05 (covered):

```
exp(1 × 0.5) / exp(1 × 0.05) = exp(0.45) ≈ 1.57 (mild)
exp(3 × 0.5) / exp(3 × 0.05) = exp(1.35) ≈ 3.86 (moderate)
exp(10 × 0.5) / exp(10 × 0.05) = exp(4.5) ≈ 90 (aggressive)
```

Compare with power-softmin at p=3:
```
0.5^3 / 0.05^3 = 0.125 / 0.000125 = 1000 (very aggressive!)
```

Actually, the power amplifier is STRONGER than the exponential at typical values, because x^p with x<1 shrinks fast. That might explain why p=3 works — it provides very strong coverage enforcement for the softmin values that are in [0, 1].

## Concrete Prediction: Exp-SoftMin

Based on this analysis, we predict that:

```python
class ExpSoftMinLoss(nn.Module):
    """L = Σ_j exp(α × softmin_j) — exponential coverage amplification."""
    def __init__(self, temperature=0.1, alpha=3.0):
        ...
    def forward(self, D):
        w = torch.softmax(-D / self.temperature, dim=1)
        sm = (w * D).sum(dim=1)  # (B, N) per-target softmin
        # ... same for precision ...
        return (torch.exp(self.alpha * sm).mean(-1) + ...).mean()
```

should work comparably to Power-SoftMin, because both apply a convex amplifier to softmin. The exp amplifier is mathematically closer to the original product loss structure (which is exp(M × log-GM)), while the power amplifier is what we discovered empirically.

The key insight: **the amplifier choice (power vs exp) is secondary. What matters is that the inner aggregation provides non-uniform gradients (softmin, not GM).**

## Summary

```
Product Loss = exp(M × mean(log D))     — correct structure, broken diagnostic
Power-SoftMin = exp(p × log(softmin D)) — same structure, working diagnostic
                = softmin^p              — simplifies beautifully

The product's problem was never the product structure.
It was using the geometric mean as the inner diagnostic.
Replace GM with softmin, and the product works.
Power-SoftMin IS that replacement.
```
