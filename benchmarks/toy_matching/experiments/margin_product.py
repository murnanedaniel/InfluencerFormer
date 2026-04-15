"""
Margin-based and hinge-style product loss variants.

Tests product losses with different distance transformations that prevent
vanishing/exploding products at N=20 by mapping distances to bounded or
well-scaled ranges before taking the product.

Setup: N_target=5, N_pred=20, 2D, padded targets with 15 dustbin copies
at learned [-1,-1]. Adam lr=0.005, 8000 steps. 5 seeds each.

Variants:
  1. Hinge product:     prod_i max(0, D_ij - margin)
  2. Sigmoid product:   prod_i sigmoid(alpha * (D_ij - margin))
  3. Soft-hinge product: prod_i softplus(D_ij - margin)
  4. Influencer-style:  attractive + repulsive hinge terms
  5. Binary product:    prod_i (D_ij / (D_ij + eps))
  6. Ratio product:     prod_i (D_ij / D_ref), D_ref = row/col mean
  7. Arctan product:    prod_i (2/pi) * arctan(D_ij / scale)
  8. Tanh product:      prod_i tanh(D_ij / scale)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------------ #
#  Evaluation (shared with toy_benchmark.py)
# ------------------------------------------------------------------ #

def evaluate(preds, targets, exist_pred, n_target):
    """Compute matched L2, n_pred, and F1."""
    with torch.no_grad():
        N = preds.shape[0]
        M = targets.shape[0]
        D = torch.cdist(preds.unsqueeze(0), targets.unsqueeze(0))[0]
        D_np = D.numpy()
        row, col = linear_sum_assignment(D_np)
        # Only count matches to real targets (first n_target)
        real_mask = col < n_target
        real_row = row[real_mask]
        real_col = col[real_mask]
        ml2 = D_np[real_row, real_col].mean() if len(real_row) > 0 else 1.0

        # Ground truth: predictions matched to real targets exist
        true_exist = np.zeros(N)
        true_exist[real_row] = 1.0

        pe = exist_pred if isinstance(exist_pred, np.ndarray) else exist_pred.numpy()
        tp = ((pe == 1) & (true_exist == 1)).sum()
        fp = ((pe == 1) & (true_exist == 0)).sum()
        fn = ((pe == 0) & (true_exist == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    return float(ml2), int(pe.sum()), float(f1)


# ------------------------------------------------------------------ #
#  Loss functions: margin/hinge product variants
# ------------------------------------------------------------------ #

def _gm_coverage_precision(log_vals, n_pred_dim=1, n_tgt_dim=2):
    """Given log-transformed values (B, M, N), compute GM coverage + precision."""
    coverage = torch.exp(log_vals.mean(dim=n_pred_dim)).mean(dim=-1)
    precision = torch.exp(log_vals.mean(dim=n_tgt_dim)).mean(dim=-1)
    return (coverage + precision).mean()


class HingeProductLoss:
    """Variant 1: prod_i max(0, D_ij - margin)"""
    def __init__(self, margin=0.1, eps=1e-8):
        self.margin = margin
        self.eps = eps

    def __call__(self, D):
        # D: (M, N) unbatched
        D = D.unsqueeze(0)  # (1, M, N)
        hinged = F.relu(D - self.margin)
        log_h = torch.log(hinged + self.eps)
        return _gm_coverage_precision(log_h)


class SigmoidProductLoss:
    """Variant 2: prod_i sigmoid(alpha * (D_ij - margin))"""
    def __init__(self, margin=0.3, scale=10.0, eps=1e-8):
        self.margin = margin
        self.scale = scale
        self.eps = eps

    def __call__(self, D):
        D = D.unsqueeze(0)
        sig = torch.sigmoid(self.scale * (D - self.margin))
        log_s = torch.log(sig + self.eps)
        return _gm_coverage_precision(log_s)


class SoftHingeProductLoss:
    """Variant 3: prod_i softplus(D_ij - margin)"""
    def __init__(self, margin=0.1, beta=5.0, eps=1e-8):
        self.margin = margin
        self.beta = beta
        self.eps = eps

    def __call__(self, D):
        D = D.unsqueeze(0)
        sp = F.softplus(D - self.margin, beta=self.beta)
        log_sp = torch.log(sp + self.eps)
        return _gm_coverage_precision(log_sp)


class InfluencerStyleLoss:
    """Variant 4: attractive + repulsive hinge-product terms.

    attractive: for each target j, GM_i(D_ij) — want small (coverage)
    repulsive: for each target j, GM_i(max(0, margin - D_ij)) — want small
               (pushes non-matched preds away from target)

    The balance: attractive pulls the closest pred in, repulsive pushes
    non-closest preds away. The product ensures ALL distances contribute.
    """
    def __init__(self, margin=0.5, alpha=1.0, eps=1e-8):
        self.margin = margin
        self.alpha = alpha
        self.eps = eps

    def __call__(self, D):
        D = D.unsqueeze(0)
        # Attractive: GM of distances (want to minimize)
        log_D = torch.log(D + self.eps)
        attract_cov = torch.exp(log_D.mean(dim=1)).mean(dim=-1)
        attract_prec = torch.exp(log_D.mean(dim=2)).mean(dim=-1)

        # Repulsive: GM of max(0, margin - D) — penalizes close non-matches
        repulsive = F.relu(self.margin - D)
        log_r = torch.log(repulsive + self.eps)
        repulse_cov = torch.exp(log_r.mean(dim=1)).mean(dim=-1)
        repulse_prec = torch.exp(log_r.mean(dim=2)).mean(dim=-1)

        return (attract_cov + attract_prec + self.alpha * (repulse_cov + repulse_prec)).mean()


class BinaryProductLoss:
    """Variant 5: prod_i (D_ij / (D_ij + eps_scale))

    Maps distances to (0, 1) range. Close -> 0, far -> 1.
    Product is small if any D is close to 0 (good coverage).
    """
    def __init__(self, eps_scale=0.1, eps=1e-8):
        self.eps_scale = eps_scale
        self.eps = eps

    def __call__(self, D):
        D = D.unsqueeze(0)
        binary = D / (D + self.eps_scale)
        log_b = torch.log(binary + self.eps)
        return _gm_coverage_precision(log_b)


class RatioProductLoss:
    """Variant 6: prod_i (D_ij / D_ref)

    Normalizes each distance by a reference (row or col mean).
    Prevents scale issues in the product.
    """
    def __init__(self, ref_type="mean", eps=1e-8):
        self.ref_type = ref_type
        self.eps = eps

    def __call__(self, D):
        D = D.unsqueeze(0)
        # Coverage direction: for each target j, normalize by mean over preds
        if self.ref_type == "mean":
            ref_cov = D.mean(dim=1, keepdim=True).clamp(min=self.eps)
            ref_prec = D.mean(dim=2, keepdim=True).clamp(min=self.eps)
        elif self.ref_type == "median":
            ref_cov = D.median(dim=1, keepdim=True).values.clamp(min=self.eps)
            ref_prec = D.median(dim=2, keepdim=True).values.clamp(min=self.eps)

        ratio_cov = D / ref_cov
        ratio_prec = D / ref_prec

        log_rc = torch.log(ratio_cov + self.eps)
        log_rp = torch.log(ratio_prec + self.eps)

        coverage = torch.exp(log_rc.mean(dim=1)).mean(dim=-1)
        precision = torch.exp(log_rp.mean(dim=2)).mean(dim=-1)
        return (coverage + precision).mean()


class ArctanProductLoss:
    """Variant 7: prod_i (2/pi) * arctan(D_ij / scale)

    Maps distances to (0, 1) smoothly. Saturates for large D.
    """
    def __init__(self, scale=0.2, eps=1e-8):
        self.scale = scale
        self.eps = eps

    def __call__(self, D):
        D = D.unsqueeze(0)
        mapped = (2.0 / 3.14159265) * torch.arctan(D / self.scale)
        log_m = torch.log(mapped + self.eps)
        return _gm_coverage_precision(log_m)


class TanhProductLoss:
    """Variant 8: prod_i tanh(D_ij / scale)

    Maps distances to (0, 1). Faster saturation than arctan.
    """
    def __init__(self, scale=0.2, eps=1e-8):
        self.scale = scale
        self.eps = eps

    def __call__(self, D):
        D = D.unsqueeze(0)
        mapped = torch.tanh(D / self.scale)
        log_m = torch.log(mapped + self.eps)
        return _gm_coverage_precision(log_m)


class LogCompressedProductLoss:
    """Variant 9: prod_i log(1 + D_ij / scale)

    Log-compresses distances before taking the product.
    More aggressive compression for large D.
    """
    def __init__(self, scale=0.1, eps=1e-8):
        self.scale = scale
        self.eps = eps

    def __call__(self, D):
        D = D.unsqueeze(0)
        compressed = torch.log1p(D / self.scale)
        log_c = torch.log(compressed + self.eps)
        return _gm_coverage_precision(log_c)


class ClampedProductLoss:
    """Variant 10: prod_i clamp(D_ij, min=eps, max=cap)

    Simply caps distances to prevent explosion, then takes GM.
    """
    def __init__(self, cap=1.0, eps=1e-8):
        self.cap = cap
        self.eps = eps

    def __call__(self, D):
        D = D.unsqueeze(0)
        clamped = D.clamp(min=self.eps, max=self.cap)
        log_c = torch.log(clamped)
        return _gm_coverage_precision(log_c)


# ------------------------------------------------------------------ #
#  Single run
# ------------------------------------------------------------------ #

def run_single(loss_fn, seed_data, seed_init, n_target=5, n_pred=20,
               n_steps=8000, lr=0.005):
    """Run one config with one seed, using dustbin degenerate approach."""
    torch.manual_seed(seed_data)
    targets = torch.rand(n_target, 2)

    n_db = n_pred - n_target

    torch.manual_seed(seed_init)
    preds = torch.rand(n_pred, 2, requires_grad=True)
    db_pt = torch.tensor([-1.0, -1.0], requires_grad=True)

    opt = torch.optim.Adam([preds, db_pt], lr=lr)

    for step in range(n_steps):
        opt.zero_grad()
        padded = torch.cat([targets, db_pt.unsqueeze(0).expand(n_db, -1)], dim=0)
        cost = torch.cdist(preds.unsqueeze(0), padded.unsqueeze(0))[0]
        loss = loss_fn(cost)
        loss.backward()
        opt.step()

    # Evaluate existence via distance to dustbin vs distance to targets
    with torch.no_grad():
        center = db_pt.detach()
        dist_db = torch.norm(preds - center, dim=1)
        dist_tgt = torch.cdist(
            preds.unsqueeze(0), targets.unsqueeze(0)
        )[0].min(dim=1).values
        exist_pred = (dist_tgt < dist_db).float().numpy()

    return evaluate(preds.detach(), targets, exist_pred, n_target)


def run_multi_seed(loss_fn_factory, n_seeds=5, **kwargs):
    """Run loss over multiple seeds, return aggregated stats."""
    ml2s, nps, f1s = [], [], []
    for s in range(n_seeds):
        loss_fn = loss_fn_factory()  # fresh instance per seed
        ml2, np_, f1 = run_single(
            loss_fn, seed_data=s * 7, seed_init=s * 13 + 1, **kwargs
        )
        ml2s.append(ml2)
        nps.append(np_)
        f1s.append(f1)
    return {
        "ml2_mean": float(np.mean(ml2s)), "ml2_std": float(np.std(ml2s)),
        "npred_mean": float(np.mean(nps)), "npred_std": float(np.std(nps)),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "f1_all": [float(x) for x in f1s],
        "ml2_all": [float(x) for x in ml2s],
    }


# ------------------------------------------------------------------ #
#  Build config grid
# ------------------------------------------------------------------ #

def build_configs():
    configs = {}

    # --- Variant 1: Hinge product ---
    for margin in [0.05, 0.1, 0.2, 0.3, 0.5]:
        name = f"hinge_m{margin}"
        configs[name] = lambda m=margin: HingeProductLoss(margin=m)

    # --- Variant 2: Sigmoid product ---
    for margin in [0.1, 0.2, 0.3, 0.5]:
        for scale in [5.0, 10.0, 20.0]:
            name = f"sigmoid_m{margin}_s{scale}"
            configs[name] = lambda m=margin, s=scale: SigmoidProductLoss(margin=m, scale=s)

    # --- Variant 3: Soft-hinge (softplus) product ---
    for margin in [0.05, 0.1, 0.2, 0.3]:
        for beta in [3.0, 5.0, 10.0]:
            name = f"softhinge_m{margin}_b{beta}"
            configs[name] = lambda m=margin, b=beta: SoftHingeProductLoss(margin=m, beta=b)

    # --- Variant 4: Influencer-style ---
    for margin in [0.3, 0.5, 0.7, 1.0]:
        for alpha in [0.5, 1.0, 2.0]:
            name = f"influencer_m{margin}_a{alpha}"
            configs[name] = lambda m=margin, a=alpha: InfluencerStyleLoss(margin=m, alpha=a)

    # --- Variant 5: Binary product ---
    for eps_scale in [0.05, 0.1, 0.2, 0.3, 0.5]:
        name = f"binary_e{eps_scale}"
        configs[name] = lambda e=eps_scale: BinaryProductLoss(eps_scale=e)

    # --- Variant 6: Ratio product ---
    for ref in ["mean", "median"]:
        name = f"ratio_{ref}"
        configs[name] = lambda r=ref: RatioProductLoss(ref_type=r)

    # --- Variant 7: Arctan product ---
    for scale in [0.1, 0.2, 0.3, 0.5]:
        name = f"arctan_s{scale}"
        configs[name] = lambda s=scale: ArctanProductLoss(scale=s)

    # --- Variant 8: Tanh product ---
    for scale in [0.1, 0.2, 0.3, 0.5]:
        name = f"tanh_s{scale}"
        configs[name] = lambda s=scale: TanhProductLoss(scale=s)

    # --- Variant 9: Log-compressed product ---
    for scale in [0.05, 0.1, 0.2, 0.5]:
        name = f"logcomp_s{scale}"
        configs[name] = lambda s=scale: LogCompressedProductLoss(scale=s)

    # --- Variant 10: Clamped product ---
    for cap in [0.5, 1.0, 1.5, 2.0]:
        name = f"clamped_c{cap}"
        configs[name] = lambda c=cap: ClampedProductLoss(cap=c)

    # --- Baseline: raw product (GM) ---
    from influencerformer.losses.product_loss import ProductLoss
    configs["raw_gm"] = lambda: _WrappedModuleLoss(ProductLoss())

    return configs


class _WrappedModuleLoss:
    """Wrap nn.Module losses to match our (M,N) interface."""
    def __init__(self, module):
        self.module = module
    def __call__(self, D):
        return self.module(D.unsqueeze(0))


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-target", type=int, default=5)
    parser.add_argument("--n-pred", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = str(Path(__file__).parent.parent / "results" / "margin_product.json")

    configs = build_configs()
    if args.filter:
        configs = {k: v for k, v in configs.items() if args.filter in k}

    print(f"Running {len(configs)} configs x {args.seeds} seeds "
          f"(N_target={args.n_target}, N_pred={args.n_pred}, "
          f"{args.n_steps} steps, lr={args.lr})")
    print()

    kwargs = dict(
        n_target=args.n_target, n_pred=args.n_pred,
        n_steps=args.n_steps, lr=args.lr,
    )

    results = {}
    t0 = time.time()

    for name, factory in configs.items():
        try:
            stats = run_multi_seed(factory, n_seeds=args.seeds, **kwargs)
            results[name] = stats
            elapsed = time.time() - t0
            print(f"[{elapsed:6.0f}s] {name:40s}  "
                  f"ml2={stats['ml2_mean']:.4f}+-{stats['ml2_std']:.4f}  "
                  f"F1={stats['f1_mean']:.3f}+-{stats['f1_std']:.3f}", flush=True)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[{elapsed:6.0f}s] {name:40s}  FAILED: {e}", flush=True)
            results[name] = {"error": str(e)}

    # Summary table sorted by F1
    print("\n" + "=" * 100)
    print(f"{'Config':40s} {'ml2':>16s} {'F1':>16s} {'n_pred':>10s}")
    print("-" * 100)

    valid = {k: v for k, v in results.items() if "error" not in v}
    for name in sorted(valid, key=lambda k: -valid[k]["f1_mean"]):
        s = valid[name]
        print(f"{name:40s} {s['ml2_mean']:.4f}+-{s['ml2_std']:.4f}  "
              f"{s['f1_mean']:.3f}+-{s['f1_std']:.3f}  "
              f"{s['npred_mean']:5.1f}+-{s['npred_std']:.1f}")
    print("=" * 100)

    # Group by variant family
    print("\n--- Best per family ---")
    families = {}
    for name, stats in valid.items():
        family = name.split("_")[0]
        if family not in families or stats["f1_mean"] > families[family][1]["f1_mean"]:
            families[family] = (name, stats)
    for family in sorted(families, key=lambda f: -families[f][1]["f1_mean"]):
        name, s = families[family]
        print(f"  {family:15s} best={name:35s}  F1={s['f1_mean']:.3f}  ml2={s['ml2_mean']:.4f}")

    # Save
    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump({"results": results, "args": vars(args)}, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
