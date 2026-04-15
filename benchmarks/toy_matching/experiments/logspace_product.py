"""
Log-space and normalized reformulations of Product Loss for large N.

Problem: Product Loss L = Σ_j Π_i D_ij works at N_pred=7 but fails at
N_pred=20 because products of 20 terms vanish or explode.

Setup: N_target=5, N_pred=20, 2D points in [0,1]^2, pad targets with
15 copies of a learned dustbin at [-1,-1], optimize predictions directly
with Adam lr=0.005 for 8000 steps.

Variants tested:
  1. log_mean_product   — geometric mean via exp(mean(log D))
  2. log_sum_product    — sum of logs (no exponentiation)
  3. normalized_product — N-th root: (Π D)^{1/N}
  4. softmax_log_product — softmax-weighted log-product
  5. clamped_product    — clamp distances before product
  6. logsumexp_product  — logsumexp approximation to log-product
  7. baseline: vanilla product (expected to fail)
  8. baseline: sinkhorn with dustbin (expected to succeed)

Usage:
    python experiments/logspace_product.py
    python experiments/logspace_product.py --seeds 3  # quick test
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------------ #
#  Sinkhorn baseline                                                   #
# ------------------------------------------------------------------ #

def sinkhorn(log_alpha, n_iters=30):
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
    return log_alpha.exp()


# ------------------------------------------------------------------ #
#  Loss functions (operate on cost matrix, no batch dim)               #
# ------------------------------------------------------------------ #

def vanilla_product_loss(cost, eps=1e-8):
    """Original product: Σ_j Π_i D_ij. Expected to fail at N=20."""
    log_cost = torch.log(cost + eps)
    l_cov = log_cost.sum(dim=0).exp().mean()   # Π over preds for each target
    l_prec = log_cost.sum(dim=1).exp().mean()  # Π over targets for each pred
    return 0.5 * (l_cov + l_prec)


def log_mean_product_loss(cost, eps=1e-8):
    """Geometric mean: Σ_j exp(mean_i(log D_ij)).
    Equivalent to N-th root of the product = (Π D)^{1/N}.
    Avoids vanishing because mean(log D) stays bounded."""
    log_cost = torch.log(cost + eps)
    l_cov = torch.exp(log_cost.mean(dim=0)).mean()
    l_prec = torch.exp(log_cost.mean(dim=1)).mean()
    return 0.5 * (l_cov + l_prec)


def log_sum_product_loss(cost, eps=1e-8):
    """Sum of logs: Σ_j Σ_i log(D_ij).
    Purely additive — no exponentiation at all.
    Gradient: ∂/∂D_kj = 1/D_kj (inverse distance weighting)."""
    log_cost = torch.log(cost + eps)
    l_cov = log_cost.sum(dim=0).mean()   # Σ_i log D for each target j
    l_prec = log_cost.sum(dim=1).mean()  # Σ_j log D for each pred i
    return 0.5 * (l_cov + l_prec)


def normalized_product_loss(cost, eps=1e-8):
    """N-th root: Σ_j (Π_i D_ij)^{1/N}.
    Mathematically identical to log_mean_product but implemented
    differently to verify numerical equivalence."""
    N_pred = cost.shape[0]
    N_tgt = cost.shape[1]
    log_cost = torch.log(cost + eps)
    l_cov = torch.exp(log_cost.sum(dim=0) / N_pred).mean()
    l_prec = torch.exp(log_cost.sum(dim=1) / N_tgt).mean()
    return 0.5 * (l_cov + l_prec)


def softmax_log_product_loss(cost, eps=1e-8, tau=0.3):
    """Softmax-weighted log-product: Σ_j exp(Σ_i w_ij * log D_ij).
    w_ij = softmax(-D/τ) focuses the product on nearby entries,
    ignoring irrelevant far-away distances."""
    log_cost = torch.log(cost + eps)

    # Coverage: for each target j, weight over preds i
    w_cov = torch.softmax(-cost / tau, dim=0)           # (N_pred, N_tgt)
    weighted_log_cov = (w_cov * log_cost).sum(dim=0)    # (N_tgt,)
    l_cov = torch.exp(weighted_log_cov).mean()

    # Precision: for each pred i, weight over targets j
    w_prec = torch.softmax(-cost / tau, dim=1)           # (N_pred, N_tgt)
    weighted_log_prec = (w_prec * log_cost).sum(dim=1)   # (N_pred,)
    l_prec = torch.exp(weighted_log_prec).mean()

    return 0.5 * (l_cov + l_prec)


def clamped_product_loss(cost, eps=1e-4, clamp_max=2.0):
    """Clamped product: clamp D to [eps, C] before taking product.
    Prevents both vanishing (floor at eps) and explosion (ceiling at C).
    Uses geometric mean (1/N exponent) for additional stability."""
    clamped = torch.clamp(cost, min=eps, max=clamp_max)
    log_clamped = torch.log(clamped)
    l_cov = torch.exp(log_clamped.mean(dim=0)).mean()
    l_prec = torch.exp(log_clamped.mean(dim=1)).mean()
    return 0.5 * (l_cov + l_prec)


def logsumexp_product_loss(cost, eps=1e-8, beta=-5.0):
    """LogSumExp approximation: use LSE as a smooth min/product hybrid.
    LSE(beta * log D) / beta approximates min(log D) for beta -> -inf
    and mean(log D) for beta -> 0.
    We use moderate negative beta to interpolate."""
    log_cost = torch.log(cost + eps)

    # Coverage: for each target j, aggregate over preds i
    l_cov_log = torch.logsumexp(beta * log_cost, dim=0) / beta  # (N_tgt,)
    l_cov = torch.exp(l_cov_log).mean()

    # Precision: for each pred i, aggregate over targets j
    l_prec_log = torch.logsumexp(beta * log_cost, dim=1) / beta  # (N_pred,)
    l_prec = torch.exp(l_prec_log).mean()

    return 0.5 * (l_cov + l_prec)


def shifted_log_product_loss(cost, eps=1e-8, shift=1.0):
    """Shifted log-product: use log(1 + D) instead of log(D).
    This maps D=0 -> 0 instead of -inf, avoiding the negative-log
    problem that causes sum-of-logs to push distances to zero
    indiscriminately. Geometric mean of log(1+D)."""
    log1p_cost = torch.log1p(cost)
    log_log1p = torch.log(log1p_cost + eps)
    l_cov = torch.exp(log_log1p.mean(dim=0)).mean()
    l_prec = torch.exp(log_log1p.mean(dim=1)).mean()
    return 0.5 * (l_cov + l_prec)


def power_softmin_product_loss(cost, tau=0.1, power=3.0):
    """Power-SoftMin (PM3): softmin^p. Known good baseline for comparison.
    Not a product loss, but included to calibrate expectations."""
    w_cov = torch.softmax(-cost / tau, dim=0)
    sm_cov = (w_cov * cost).sum(dim=0)
    w_prec = torch.softmax(-cost / tau, dim=1)
    sm_prec = (w_prec * cost).sum(dim=1)
    return 0.5 * (sm_cov.pow(power).mean() + sm_prec.pow(power).mean())


def sinkhorn_loss(cost, tau=0.3, n_iters=30):
    """Sinkhorn baseline."""
    P = sinkhorn(-cost / tau, n_iters=n_iters)
    sm_per_col = (P * cost).sum(dim=0)
    sm_per_row = (P * cost).sum(dim=1)
    return 0.5 * (sm_per_col.mean() + sm_per_row.mean())


# ------------------------------------------------------------------ #
#  Evaluation                                                          #
# ------------------------------------------------------------------ #

def evaluate(preds, targets, dustbin_center):
    """Compute matched L2 and F1 (existence accuracy).

    Existence rule: prediction is 'real' if it's closer to any target
    than to the dustbin center.
    """
    with torch.no_grad():
        N = preds.shape[0]
        M = targets.shape[0]
        D = torch.cdist(preds.unsqueeze(0), targets.unsqueeze(0))[0]
        D_np = D.numpy()
        row, col = linear_sum_assignment(D_np)
        ml2 = D_np[row, col].mean()

        # Ground truth existence: the M predictions matched by Hungarian
        true_exist = np.zeros(N)
        true_exist[row] = 1.0

        # Predicted existence: closer to a target than to dustbin
        dist_to_dustbin = torch.norm(preds - dustbin_center, dim=1).numpy()
        dist_to_nearest_tgt = D.min(dim=1).values.numpy()
        pred_exist = (dist_to_nearest_tgt < dist_to_dustbin).astype(float)

        tp = ((pred_exist == 1) & (true_exist == 1)).sum()
        fp = ((pred_exist == 1) & (true_exist == 0)).sum()
        fn = ((pred_exist == 0) & (true_exist == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    return float(ml2), float(f1)


# ------------------------------------------------------------------ #
#  Single run                                                          #
# ------------------------------------------------------------------ #

LOSS_REGISTRY = {
    "vanilla_product":        (vanilla_product_loss, {}),
    "log_mean_product":       (log_mean_product_loss, {}),
    "log_sum_product":        (log_sum_product_loss, {}),
    "normalized_product":     (normalized_product_loss, {}),
    "softmax_log_product":    (softmax_log_product_loss, {"tau": 0.3}),
    "clamped_product":        (clamped_product_loss, {"eps": 1e-4, "clamp_max": 2.0}),
    "logsumexp_product":      (logsumexp_product_loss, {"beta": -5.0}),
    "shifted_log_product":    (shifted_log_product_loss, {}),
    "pm3_baseline":           (power_softmin_product_loss, {"tau": 0.1, "power": 3.0}),
    "sinkhorn_baseline":      (sinkhorn_loss, {"tau": 0.3}),
}


def run_one(loss_name, seed, n_target=5, n_pred=20, n_steps=8000, lr=0.005):
    """Run a single (loss, seed) experiment. Returns (ml2, f1)."""
    loss_fn, loss_kwargs = LOSS_REGISTRY[loss_name]
    n_db = n_pred - n_target

    torch.manual_seed(seed * 7)
    targets = torch.rand(n_target, 2)

    torch.manual_seed(seed * 13 + 1)
    preds = torch.rand(n_pred, 2, requires_grad=True)

    # Learned dustbin point, initialized at [-1, -1]
    db_pt = torch.tensor([-1.0, -1.0], requires_grad=True)

    opt = torch.optim.Adam([preds, db_pt], lr=lr)

    for step in range(n_steps):
        opt.zero_grad()

        # Pad targets with n_db copies of dustbin
        padded = torch.cat([targets, db_pt.unsqueeze(0).expand(n_db, -1)], dim=0)
        cost = torch.cdist(preds.unsqueeze(0), padded.unsqueeze(0))[0]  # (N_pred, N_pred)

        loss = loss_fn(cost, **loss_kwargs)
        loss.backward()
        opt.step()

    return evaluate(preds.detach(), targets, db_pt.detach())


# ------------------------------------------------------------------ #
#  Multi-seed runner                                                   #
# ------------------------------------------------------------------ #

def run_multi_seed(loss_name, n_seeds=5, **kwargs):
    ml2s, f1s = [], []
    for s in range(n_seeds):
        ml2, f1 = run_one(loss_name, seed=s, **kwargs)
        ml2s.append(ml2)
        f1s.append(f1)
    return {
        "ml2_mean": float(np.mean(ml2s)),
        "ml2_std": float(np.std(ml2s)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "ml2_all": ml2s,
        "f1_all": f1s,
    }


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Log-space product loss variants benchmark")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-target", type=int, default=5)
    parser.add_argument("--n-pred", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--filter", type=str, default=None,
                        help="Only run losses containing this substring")
    parser.add_argument("--output", type=str,
                        default="results/logspace_product.json")
    args = parser.parse_args()

    losses = list(LOSS_REGISTRY.keys())
    if args.filter:
        losses = [l for l in losses if args.filter in l]

    kwargs = dict(
        n_target=args.n_target, n_pred=args.n_pred,
        n_steps=args.n_steps, lr=args.lr,
    )

    print(f"Log-space Product Loss Benchmark")
    print(f"N_target={args.n_target}, N_pred={args.n_pred}, "
          f"{args.n_steps} steps, lr={args.lr}, {args.seeds} seeds")
    print(f"Losses: {len(losses)}")
    print("=" * 75)

    results = {}
    t0 = time.time()

    for loss_name in losses:
        t1 = time.time()
        stats = run_multi_seed(loss_name, n_seeds=args.seeds, **kwargs)
        elapsed = time.time() - t1
        results[loss_name] = stats
        print(f"  {loss_name:30s}  ml2={stats['ml2_mean']:.4f}±{stats['ml2_std']:.3f}  "
              f"F1={stats['f1_mean']:.2f}±{stats['f1_std']:.2f}  "
              f"({elapsed:.0f}s)")

    # Summary table sorted by F1 then ml2
    print("\n" + "=" * 75)
    print(f"{'Loss':30s} {'ml2':>16s} {'F1':>16s}")
    print("-" * 75)
    ranked = sorted(results, key=lambda k: (-results[k]["f1_mean"],
                                             results[k]["ml2_mean"]))
    for name in ranked:
        s = results[name]
        print(f"{name:30s} {s['ml2_mean']:.4f} ± {s['ml2_std']:.3f}  "
              f"{s['f1_mean']:.2f} ± {s['f1_std']:.2f}")
    print("=" * 75)

    total = time.time() - t0
    print(f"\nTotal time: {total:.0f}s")

    # Save
    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"Saved to {outpath}")


if __name__ == "__main__":
    main()
