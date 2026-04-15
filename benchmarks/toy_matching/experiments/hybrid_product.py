"""
Hybrid Product Loss experiments: single-pass matching formulations
that attempt to capture Sinkhorn's coupling effect without iteration.

Key insight: PM3 = 1 iteration of Sinkhorn (F1=0.44 at N=20).
Product Loss also fails at N=20 (F1=0.45) because neither couples
coverage and precision. Can a single operation achieve coupling?

Formulations tested:
  1. product_normalized  — coverage weighted by precision products
  2. cross_product       — each (i,j) weighted by both directions
  3. ratio               — column product / leave-one-out sum
  4. attention_product   — leave-one-out product as attention weights
  5. permanent_bethe     — Bethe permanent approximation
  6. geometric_coupling  — geometric mean of cov/prec softmax products
  7. product_sinkhorn1   — one Sinkhorn step on product-based log scores
  8. entropic_product    — product loss + entropic coupling penalty

Usage:
    python experiments/hybrid_product.py                  # run all
    python experiments/hybrid_product.py --filter ratio   # run subset
    python experiments/hybrid_product.py --n-steps 2000   # quick test
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
#  Baselines (for comparison)                                         #
# ------------------------------------------------------------------ #

def loss_pm3(pred, target, tau=0.1, **kw):
    """PM3 / Power-SoftMin = 1 iteration of Sinkhorn."""
    cost = torch.cdist(pred, target, p=2)
    cov_w = torch.softmax(-cost / tau, dim=0)
    l_cov = (cov_w * cost).sum(dim=0).mean()
    prec_w = torch.softmax(-cost / tau, dim=1)
    l_prec = (prec_w * cost).sum(dim=1).mean()
    return 0.5 * (l_cov + l_prec)


def loss_sinkhorn(pred, target, tau=0.1, n_iters=10, **kw):
    """Sinkhorn with configurable iterations."""
    cost = torch.cdist(pred, target, p=2)
    log_alpha = -cost / tau
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
    P = log_alpha.exp()
    return (P * cost).sum()


def loss_product(pred, target, eps=1e-8, **kw):
    """Standard product loss (known to fail at N=20)."""
    cost = torch.cdist(pred, target, p=2) + eps
    l_cov = cost.log().sum(dim=0).exp().mean()
    l_prec = cost.log().sum(dim=1).exp().mean()
    return 0.5 * (l_cov + l_prec)


def loss_hungarian(pred, target, **kw):
    """Hungarian (oracle baseline)."""
    cost = torch.cdist(pred, target, p=2)
    cost_np = cost.detach().cpu().numpy()
    row, col = linear_sum_assignment(cost_np)
    return cost[row, col].mean()


# ------------------------------------------------------------------ #
#  Hybrid formulations                                                 #
# ------------------------------------------------------------------ #

def loss_product_normalized(pred, target, eps=1e-8, **kw):
    """
    Formulation 1: Product-normalized product.

    For coverage, weight each prediction's contribution by its normalized
    precision product.  w_i = Prod_j D_ij / Sum_k Prod_j D_kj.
    L_cov = Sum_j Prod_i (D_ij * w_i).
    L_prec is the symmetric version.
    """
    cost = torch.cdist(pred, target, p=2) + eps  # (N, M)

    # Precision products: for each pred i, product over targets
    log_prec = cost.log().sum(dim=1)  # (N,)
    log_w_prec = log_prec - torch.logsumexp(log_prec, dim=0)  # (N,) normalized
    w_prec = log_w_prec.exp().unsqueeze(1)  # (N, 1)

    # Weighted coverage: Prod_i (D_ij * w_i) for each target j
    log_weighted_cost = (cost * w_prec).log().sum(dim=0)  # (M,)
    l_cov = log_weighted_cost.exp().mean()

    # Symmetric: coverage products weight precision
    log_cov = cost.log().sum(dim=0)  # (M,)
    log_w_cov = log_cov - torch.logsumexp(log_cov, dim=0)  # (M,)
    w_cov = log_w_cov.exp().unsqueeze(0)  # (1, M)

    log_weighted_cost2 = (cost * w_cov).log().sum(dim=1)  # (N,)
    l_prec = log_weighted_cost2.exp().mean()

    return 0.5 * (l_cov + l_prec)


def loss_cross_product(pred, target, eps=1e-8, **kw):
    """
    Formulation 2: Cross-product.

    L = Sum_j Sum_i D_ij * Prod_{k!=i} D_kj * Prod_{l!=j} D_il

    Each (i,j) entry is weighted by BOTH the coverage product (how well
    other preds cover target j) AND the precision product (how well pred i
    covers other targets).
    """
    cost = torch.cdist(pred, target, p=2) + eps  # (N, M)
    log_cost = cost.log()

    # Leave-one-out log products
    log_col_total = log_cost.sum(dim=0, keepdim=True)  # (1, M)
    log_loo_col = log_col_total - log_cost  # Prod_{k!=i} D_kj

    log_row_total = log_cost.sum(dim=1, keepdim=True)  # (N, 1)
    log_loo_row = log_row_total - log_cost  # Prod_{l!=j} D_il

    # Cross product weight
    log_weight = log_loo_col + log_loo_row  # (N, M)
    weighted = cost * log_weight.exp()

    return weighted.sum() / (cost.shape[0] * cost.shape[1])


def loss_ratio(pred, target, eps=1e-8, **kw):
    """
    Formulation 3: Ratio formulation.

    L_cov = Sum_j [ Prod_i D_ij / Sum_i Prod_{k!=i} D_kj ]

    Each column's full product divided by the sum of leave-one-out products.
    Creates competition: if pred i is close to target j, removing it makes
    the product large, so the denominator is large and the ratio is small,
    indicating good matching.
    """
    cost = torch.cdist(pred, target, p=2) + eps  # (N, M)
    log_cost = cost.log()

    # Full column product: Prod_i D_ij for each j
    log_full_col = log_cost.sum(dim=0)  # (M,)

    # Leave-one-out: Prod_{k!=i} D_kj = full / D_ij
    log_loo_col = log_full_col.unsqueeze(0) - log_cost  # (N, M)
    loo_col_sum = log_loo_col.exp().sum(dim=0)  # (M,)

    l_cov = (log_full_col.exp() / (loo_col_sum + eps)).mean()

    # Symmetric for rows
    log_full_row = log_cost.sum(dim=1)  # (N,)
    log_loo_row = log_full_row.unsqueeze(1) - log_cost  # (N, M)
    loo_row_sum = log_loo_row.exp().sum(dim=1)  # (N,)
    l_prec = (log_full_row.exp() / (loo_row_sum + eps)).mean()

    return 0.5 * (l_cov + l_prec)


def loss_attention_product(pred, target, eps=1e-8, **kw):
    """
    Formulation 4: Attention-product hybrid.

    w_ij = Prod_{k!=i} D_kj / Sum_l Prod_{k!=l} D_kj
    L = Sum_j Sum_i w_ij * D_ij

    The leave-one-out product tells us: if we remove pred i from covering
    target j, how bad is the remaining coverage? High => pred i is essential.
    We normalize to get attention weights, then weight the cost.

    This is like a product-derived softmin where "attention" naturally
    identifies the best-matched prediction for each target.
    """
    cost = torch.cdist(pred, target, p=2) + eps  # (N, M)
    log_cost = cost.log()

    # Leave-one-out products for columns (coverage direction)
    log_full_col = log_cost.sum(dim=0, keepdim=True)  # (1, M)
    log_loo_col = log_full_col - log_cost  # (N, M)

    # Normalize: softmax over predictions for each target
    # High loo product => removing this pred hurts coverage => important pred
    w_cov = torch.softmax(log_loo_col, dim=0)  # (N, M)
    l_cov = (w_cov * cost).sum(dim=0).mean()

    # Symmetric for rows
    log_full_row = log_cost.sum(dim=1, keepdim=True)  # (N, 1)
    log_loo_row = log_full_row - log_cost  # (N, M)
    w_prec = torch.softmax(log_loo_row, dim=1)  # (N, M)
    l_prec = (w_prec * cost).sum(dim=1).mean()

    return 0.5 * (l_cov + l_prec)


def loss_permanent_bethe(pred, target, tau=0.1, n_iters=5, **kw):
    """
    Formulation 5: Bethe permanent approximation.

    The permanent perm(A) = Sum_sigma Prod_i A_{i,sigma(i)} is exactly
    the matching criterion. The Bethe free energy approximation of the
    permanent is related to Sinkhorn. We approximate it via:

    log perm(A) ~ Sum_ij P_ij * log(A_ij / P_ij)
    where P = Sinkhorn(A).

    But we want a SINGLE pass, so we use 1 Sinkhorn iteration to get P,
    then compute the Bethe free energy as the loss. This injects coupling
    information from the single normalization step into the product-like
    objective.
    """
    cost = torch.cdist(pred, target, p=2)  # (N, M)
    N, M = cost.shape

    # Similarity matrix (like permanent uses)
    log_A = -cost / tau  # (N, M)

    # One Sinkhorn iteration to get approximate doubly-stochastic P
    log_P = log_A.clone()
    log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)
    log_P = log_P - torch.logsumexp(log_P, dim=0, keepdim=True)
    P = log_P.exp()

    # Bethe free energy: -Sum_ij P_ij * log(A_ij) + Sum_ij P_ij * log(P_ij)
    # = Sum_ij P_ij * (log(P_ij) - log(A_ij))
    # = Sum_ij P_ij * (log(P_ij) + cost_ij / tau)
    # Minimizing this pushes toward the permanent-maximizing assignment
    bethe = (P * (log_P - log_A)).sum()

    return bethe


def loss_geometric_coupling(pred, target, tau=0.1, **kw):
    """
    Formulation 6: Geometric coupling.

    Instead of separate softmax per direction (PM3), take the geometric
    mean of cov and prec softmax weights. This couples directions in
    one operation:

    w_ij = sqrt(softmax_col(i) * softmax_row(j))
    w_ij /= sum_ij w_ij  (renormalize)
    L = sum_ij w_ij * cost_ij

    The geometric mean naturally favors entries that are good in BOTH
    directions, creating coupling without iteration.
    """
    cost = torch.cdist(pred, target, p=2)

    log_w_cov = torch.log_softmax(-cost / tau, dim=0)   # (N, M)
    log_w_prec = torch.log_softmax(-cost / tau, dim=1)  # (N, M)

    # Geometric mean in log space
    log_w_geo = 0.5 * (log_w_cov + log_w_prec)  # (N, M)

    # Renormalize
    log_w_geo = log_w_geo - torch.logsumexp(log_w_geo.reshape(-1), dim=0)
    w_geo = log_w_geo.exp()

    return (w_geo * cost).sum()


def loss_product_sinkhorn1(pred, target, tau=0.1, **kw):
    """
    Formulation 7: Product-informed single Sinkhorn step.

    Use product-derived importance to initialize Sinkhorn scores before
    a single normalization pass. The product embeds global structure
    that pure softmax misses.

    log_score_ij = -D_ij / tau + alpha * log(Prod_{k!=i} D_kj) + beta * log(Prod_{l!=j} D_il)

    The leave-one-out products modulate scores: an entry (i,j) gets boosted
    if other preds are far from target j (pred i is important for j) and
    other targets are far from pred i (target j is the best option for i).
    """
    cost = torch.cdist(pred, target, p=2) + 1e-8  # (N, M)
    log_cost = cost.log()

    # Leave-one-out products
    log_full_col = log_cost.sum(dim=0, keepdim=True)
    log_loo_col = log_full_col - log_cost  # (N, M)

    log_full_row = log_cost.sum(dim=1, keepdim=True)
    log_loo_row = log_full_row - log_cost  # (N, M)

    # Product-modulated scores (alpha=beta=0.1 to gently inject coupling)
    alpha = 0.1
    log_scores = -cost / tau + alpha * (log_loo_col + log_loo_row)

    # Single Sinkhorn iteration
    log_scores = log_scores - torch.logsumexp(log_scores, dim=1, keepdim=True)
    log_scores = log_scores - torch.logsumexp(log_scores, dim=0, keepdim=True)
    P = log_scores.exp()

    return (P * cost).sum()


def loss_entropic_product(pred, target, eps=1e-8, lam=0.1, **kw):
    """
    Formulation 8: Product loss + entropic coupling regularizer.

    Standard product loss plus a penalty that encourages the implicit
    assignment to be more peaked (less uniform). The coupling penalty
    is the KL divergence between the row-normalized and col-normalized
    cost matrices, encouraging them to agree.
    """
    cost = torch.cdist(pred, target, p=2) + eps  # (N, M)

    # Standard product loss
    l_cov = cost.log().sum(dim=0).exp().mean()
    l_prec = cost.log().sum(dim=1).exp().mean()
    l_product = 0.5 * (l_cov + l_prec)

    # Coupling penalty: KL between row-softmax and col-softmax
    log_p_row = torch.log_softmax(-cost, dim=1)  # row-normalized (prec)
    log_p_col = torch.log_softmax(-cost, dim=0)  # col-normalized (cov)

    # Symmetrized KL
    p_row = log_p_row.exp()
    p_col = log_p_col.exp()
    kl_rc = (p_row * (log_p_row - log_p_col)).sum()
    kl_cr = (p_col * (log_p_col - log_p_row)).sum()
    coupling_penalty = 0.5 * (kl_rc + kl_cr)

    return l_product + lam * coupling_penalty


def loss_softmax_product(pred, target, tau=0.1, eps=1e-8, **kw):
    """
    Formulation 9: Softmax of products (not product of softmaxes).

    Instead of taking softmax per row/col then averaging, compute the
    row/col product first, then use softmax to create competition.

    For coverage: each target j has a coverage product P_j = Prod_i D_ij.
    Use softmax over targets to focus on the worst-covered ones:
    w_j = softmax(P_j / tau)
    L_cov = Sum_j w_j * P_j

    This focuses the loss on under-covered targets and under-matched preds.
    """
    cost = torch.cdist(pred, target, p=2) + eps
    log_cost = cost.log()

    # Coverage products
    log_prod_col = log_cost.sum(dim=0)  # (M,) = log Prod_i D_ij
    prod_col = log_prod_col.exp()
    w_cov = torch.softmax(log_prod_col / tau, dim=0)
    l_cov = (w_cov * prod_col).sum()

    # Precision products
    log_prod_row = log_cost.sum(dim=1)  # (N,)
    prod_row = log_prod_row.exp()
    w_prec = torch.softmax(log_prod_row / tau, dim=0)
    l_prec = (w_prec * prod_row).sum()

    return 0.5 * (l_cov + l_prec)


def loss_doubly_weighted(pred, target, tau=0.1, **kw):
    """
    Formulation 10: Doubly-weighted softmin.

    Compute softmin weights for both directions simultaneously, then
    combine them multiplicatively:

    w_ij = softmax_col(-D/tau)_ij * softmax_row(-D/tau)_ij
    w_ij /= sum w_ij
    L = sum_ij w_ij * D_ij

    This is equivalent to one Sinkhorn iteration where we multiply
    (not alternate) the two normalizations. Creates coupling because
    both directions contribute to every weight.
    """
    cost = torch.cdist(pred, target, p=2)

    w_cov = torch.softmax(-cost / tau, dim=0)   # (N, M)
    w_prec = torch.softmax(-cost / tau, dim=1)  # (N, M)

    # Multiplicative combination
    w = w_cov * w_prec  # (N, M)
    w = w / (w.sum() + 1e-12)

    return (w * cost).sum()


# ------------------------------------------------------------------ #
#  Evaluation                                                          #
# ------------------------------------------------------------------ #

def evaluate(preds, targets, n_target):
    """Compute matched L2 and F1 using Hungarian ground truth + dustbin."""
    with torch.no_grad():
        N = preds.shape[0]
        M = targets.shape[0]
        D = torch.cdist(preds, targets, p=2)
        D_np = D.numpy()
        row, col = linear_sum_assignment(D_np)
        ml2 = D_np[row, col].mean()

        # Existence: is this pred matched to a real target (close enough)?
        # For degenerate dustbin, use distance to dustbin vs target
        db_center = torch.tensor([-1.0, -1.0])
        dist_db = torch.norm(preds - db_center, dim=1)
        dist_tgt = D.min(dim=1).values
        exist_pred = (dist_tgt < dist_db).float().numpy()

        true_exist = np.zeros(N)
        true_exist[row] = 1.0

        tp = ((exist_pred == 1) & (true_exist == 1)).sum()
        fp = ((exist_pred == 1) & (true_exist == 0)).sum()
        fn = ((exist_pred == 0) & (true_exist == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    return float(ml2), float(f1)


# ------------------------------------------------------------------ #
#  Single run                                                          #
# ------------------------------------------------------------------ #

def run_single(loss_fn, n_target=5, n_pred=20, n_steps=8000, lr=0.005,
               seed_data=42, seed_init=123, loss_kwargs=None):
    """Run one seed. Returns (ml2, f1)."""
    if loss_kwargs is None:
        loss_kwargs = {}

    torch.manual_seed(seed_data)
    targets = torch.rand(n_target, 2)

    # Degenerate dustbin: pad targets with distant point
    n_db = n_pred - n_target
    db_pt = torch.tensor([-1.0, -1.0], requires_grad=True)
    padded = torch.cat([targets, db_pt.unsqueeze(0).expand(n_db, -1)], dim=0)

    torch.manual_seed(seed_init)
    preds = torch.rand(n_pred, 2, requires_grad=True)

    opt = torch.optim.Adam([preds, db_pt], lr=lr)

    for step in range(n_steps):
        opt.zero_grad()
        # Recompute padded targets (db_pt may have changed)
        padded = torch.cat([targets, db_pt.unsqueeze(0).expand(n_db, -1)], dim=0)
        loss = loss_fn(preds, padded, **loss_kwargs)
        loss.backward()
        opt.step()

    return evaluate(preds.detach(), targets, n_target)


def run_multi_seed(loss_fn, n_seeds=5, loss_kwargs=None, **run_kwargs):
    """Run multiple seeds, return stats."""
    ml2s, f1s = [], []
    for s in range(n_seeds):
        ml2, f1 = run_single(
            loss_fn, seed_data=s * 7, seed_init=s * 13 + 1,
            loss_kwargs=loss_kwargs, **run_kwargs
        )
        ml2s.append(ml2)
        f1s.append(f1)
    return {
        "ml2_mean": float(np.mean(ml2s)), "ml2_std": float(np.std(ml2s)),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "f1_perfect": int(np.sum(np.array(f1s) == 1.0)),
        "f1s": [float(x) for x in f1s],
        "ml2s": [float(x) for x in ml2s],
        "n_seeds": n_seeds,
    }


# ------------------------------------------------------------------ #
#  Config grid                                                         #
# ------------------------------------------------------------------ #

def build_configs():
    """Build all configurations to test."""
    configs = {}

    # --- Baselines ---
    configs["baseline_pm3"] = {
        "fn": loss_pm3, "kwargs": {"tau": 0.1},
        "desc": "PM3 (1 Sinkhorn iter)"
    }
    configs["baseline_sinkhorn10"] = {
        "fn": loss_sinkhorn, "kwargs": {"tau": 0.1, "n_iters": 10},
        "desc": "Sinkhorn 10 iters"
    }
    configs["baseline_product"] = {
        "fn": loss_product, "kwargs": {},
        "desc": "Standard product loss"
    }
    configs["baseline_hungarian"] = {
        "fn": loss_hungarian, "kwargs": {},
        "desc": "Hungarian (oracle)"
    }

    # --- Hybrid formulations ---
    configs["hybrid_product_normalized"] = {
        "fn": loss_product_normalized, "kwargs": {},
        "desc": "Product-normalized product"
    }
    configs["hybrid_cross_product"] = {
        "fn": loss_cross_product, "kwargs": {},
        "desc": "Cross-product"
    }
    configs["hybrid_ratio"] = {
        "fn": loss_ratio, "kwargs": {},
        "desc": "Ratio (product / leave-one-out sum)"
    }
    configs["hybrid_attention_product"] = {
        "fn": loss_attention_product, "kwargs": {},
        "desc": "Attention-product hybrid"
    }
    configs["hybrid_permanent_bethe"] = {
        "fn": loss_permanent_bethe, "kwargs": {"tau": 0.1},
        "desc": "Bethe permanent (1 Sinkhorn + free energy)"
    }
    configs["hybrid_geometric_coupling"] = {
        "fn": loss_geometric_coupling, "kwargs": {"tau": 0.1},
        "desc": "Geometric mean of cov/prec softmax"
    }
    configs["hybrid_product_sinkhorn1"] = {
        "fn": loss_product_sinkhorn1, "kwargs": {"tau": 0.1},
        "desc": "Product-informed 1-step Sinkhorn"
    }
    configs["hybrid_entropic_product"] = {
        "fn": loss_entropic_product, "kwargs": {"lam": 0.1},
        "desc": "Product + entropic coupling penalty"
    }
    configs["hybrid_softmax_product"] = {
        "fn": loss_softmax_product, "kwargs": {"tau": 0.1},
        "desc": "Softmax of products"
    }
    configs["hybrid_doubly_weighted"] = {
        "fn": loss_doubly_weighted, "kwargs": {"tau": 0.1},
        "desc": "Doubly-weighted softmin (multiplicative)"
    }

    # --- Tau sweep for promising formulations ---
    for tau in [0.05, 0.2, 0.3]:
        configs[f"hybrid_geometric_coupling_tau{tau}"] = {
            "fn": loss_geometric_coupling, "kwargs": {"tau": tau},
            "desc": f"Geometric coupling tau={tau}"
        }
        configs[f"hybrid_doubly_weighted_tau{tau}"] = {
            "fn": loss_doubly_weighted, "kwargs": {"tau": tau},
            "desc": f"Doubly-weighted tau={tau}"
        }
        configs[f"hybrid_permanent_bethe_tau{tau}"] = {
            "fn": loss_permanent_bethe, "kwargs": {"tau": tau},
            "desc": f"Bethe permanent tau={tau}"
        }

    # --- Lambda sweep for entropic product ---
    for lam in [0.01, 0.5, 1.0]:
        configs[f"hybrid_entropic_product_lam{lam}"] = {
            "fn": loss_entropic_product, "kwargs": {"lam": lam},
            "desc": f"Entropic product lam={lam}"
        }

    return configs


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid product loss experiments"
    )
    parser.add_argument("--n-target", type=int, default=5)
    parser.add_argument("--n-pred", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--filter", type=str, default=None,
                        help="Only run configs containing this substring")
    parser.add_argument("--output", type=str,
                        default="results/hybrid_product.json")
    args = parser.parse_args()

    configs = build_configs()
    if args.filter:
        configs = {k: v for k, v in configs.items() if args.filter in k}

    print(f"Hybrid Product Loss Experiment")
    print(f"N_target={args.n_target}, N_pred={args.n_pred}, "
          f"steps={args.n_steps}, lr={args.lr}, seeds={args.seeds}")
    print(f"Running {len(configs)} configurations")
    print("=" * 85)
    print()

    run_kwargs = dict(
        n_target=args.n_target, n_pred=args.n_pred,
        n_steps=args.n_steps, lr=args.lr,
    )

    results = {}
    t0 = time.time()

    for name, cfg in configs.items():
        t1 = time.time()
        print(f"  {name:45s} ... ", end="", flush=True)
        try:
            stats = run_multi_seed(
                cfg["fn"], n_seeds=args.seeds,
                loss_kwargs=cfg["kwargs"], **run_kwargs
            )
            elapsed = time.time() - t1
            print(f"F1={stats['f1_mean']:.2f}+-{stats['f1_std']:.2f}  "
                  f"ml2={stats['ml2_mean']:.4f}  "
                  f"({stats['f1_perfect']}/{args.seeds} perfect)  "
                  f"[{elapsed:.0f}s]")
            stats["desc"] = cfg["desc"]
            stats["kwargs"] = {k: v for k, v in cfg["kwargs"].items()
                               if not callable(v)}
            results[name] = stats
        except Exception as e:
            print(f"FAILED: {e}")
            results[name] = {"error": str(e), "desc": cfg["desc"]}

    total_time = time.time() - t0

    # Summary table sorted by F1
    print()
    print("=" * 85)
    print(f"{'Config':45s} {'F1':>14s} {'ml2':>12s} {'perf':>6s}")
    print("-" * 85)
    valid = {k: v for k, v in results.items() if "error" not in v}
    for name in sorted(valid, key=lambda k: -valid[k]["f1_mean"]):
        s = valid[name]
        print(f"{name:45s} "
              f"{s['f1_mean']:.2f}+-{s['f1_std']:.2f}  "
              f"{s['ml2_mean']:.4f}+-{s['ml2_std']:.3f}  "
              f"{s['f1_perfect']:2d}/{args.seeds}")
    print("=" * 85)
    print(f"\nTotal time: {total_time:.0f}s")

    # Highlight best non-baseline
    hybrids = {k: v for k, v in valid.items() if k.startswith("hybrid")}
    if hybrids:
        best = max(hybrids, key=lambda k: hybrids[k]["f1_mean"])
        s = hybrids[best]
        print(f"\nBest hybrid: {best}")
        print(f"  F1={s['f1_mean']:.2f}+-{s['f1_std']:.2f}  "
              f"ml2={s['ml2_mean']:.4f}")
        print(f"  Description: {s['desc']}")

    # Save results
    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # Make JSON-serializable
    save_data = {
        "args": vars(args),
        "results": results,
        "total_time_s": total_time,
    }
    with open(outpath, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
