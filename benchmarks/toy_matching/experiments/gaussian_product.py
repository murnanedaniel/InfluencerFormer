"""
Gaussian dustbin experiment for Product Loss.

Tests whether non-degenerate dustbin targets fix the Product Loss at large N.
The hypothesis: with K identical dustbin targets, all dustbin columns in the
cost matrix are identical, so the product structure provides no signal for
which prediction should go to which dustbin copy.  With Gaussian (or spread)
dustbins, each column is different, breaking the degeneracy.

Experiments:
  1. Product + degenerate dustbin (baseline)
  2. Product + Gaussian dustbin (learned mean + variance)
  3. Product + learned individual dustbins (15 separate learnable points)
  4. Product + Gaussian, sweep std (fixed mean, vary std)
  5. Product + Gaussian, sweep init location
  6. Product + fixed-but-spread dustbins (grid / circle)
  + Log-normalized product variants for the best dustbin strategies

Usage:
    python experiments/gaussian_product.py                # run all
    python experiments/gaussian_product.py --filter gauss  # subset
    python experiments/gaussian_product.py --seeds 3       # quick test
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------------ #
#  Evaluation (same as toy_benchmark.py)                               #
# ------------------------------------------------------------------ #

def evaluate(preds, targets, exist_pred):
    """Compute matched L2, n_pred, and F1 against Hungarian ground truth."""
    with torch.no_grad():
        N = preds.shape[0]
        D = torch.cdist(preds.unsqueeze(0), targets.unsqueeze(0))[0]
        D_np = D.numpy()
        row, col = linear_sum_assignment(D_np)
        ml2 = D_np[row, col].mean()

        true_exist = np.zeros(N)
        true_exist[row] = 1.0

        pe = exist_pred if isinstance(exist_pred, np.ndarray) else exist_pred.numpy()
        tp = ((pe == 1) & (true_exist == 1)).sum()
        fp = ((pe == 1) & (true_exist == 0)).sum()
        fn = ((pe == 0) & (true_exist == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    return float(ml2), int(pe.sum()), float(f1)


# ------------------------------------------------------------------ #
#  Product loss variants                                               #
# ------------------------------------------------------------------ #

def product_loss_raw(cost, eps=1e-8):
    """Standard product loss: L = mean_j prod_i D_ij + mean_i prod_j D_ij."""
    c = cost + eps
    l_cov = c.log().sum(dim=0).exp().mean()
    l_prec = c.log().sum(dim=1).exp().mean()
    return 0.5 * (l_cov + l_prec)


def product_loss_lognorm(cost, eps=1e-8):
    """
    Log-normalized product: L = mean_j exp(mean_i log D_ij) + symmetric.
    Replaces sum-of-logs with mean-of-logs so the product doesn't vanish
    with large N.
    """
    c = cost + eps
    l_cov = (c.log().mean(dim=0)).exp().mean()   # geometric mean per column
    l_prec = (c.log().mean(dim=1)).exp().mean()  # geometric mean per row
    return 0.5 * (l_cov + l_prec)


def sinkhorn_loss(cost, tau=0.3, n_iters=30):
    """Sinkhorn matching loss for comparison baseline."""
    log_alpha = -cost / tau
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
    P = log_alpha.exp()
    return (P * cost).sum()


def hungarian_loss(cost):
    """Hungarian matching loss for comparison baseline."""
    cost_np = cost.detach().numpy()
    row, col = linear_sum_assignment(cost_np)
    return cost[row, col].mean()


# ------------------------------------------------------------------ #
#  Single experiment runner                                            #
# ------------------------------------------------------------------ #

def run_single(config, seed_data, seed_init, n_target=5, n_pred=20,
               n_steps=8000, lr=0.005):
    """Run one configuration with one seed. Returns (ml2, n_pred_est, f1)."""
    torch.manual_seed(seed_data)
    targets = torch.rand(n_target, 2)

    torch.manual_seed(seed_init)
    preds = torch.rand(n_pred, 2, requires_grad=True)

    loss_type = config["loss_type"]         # "product", "product_lognorm", "sinkhorn", "hungarian"
    dustbin = config.get("dustbin", "none") # "degenerate", "gaussian", "learned_set", "fixed_circle", "fixed_grid"
    dustbin_mean_init = config.get("dustbin_mean_init", [-1.0, -1.0])
    dustbin_std_init = config.get("dustbin_std_init", 0.1)
    dustbin_std_fixed = config.get("dustbin_std_fixed", None)  # if set, std is not learned
    tau = config.get("tau", 0.3)
    n_db = n_pred - n_target

    params = [preds]

    # --- set up dustbin targets ---
    if dustbin == "degenerate":
        db_pt = torch.tensor(dustbin_mean_init, dtype=torch.float32, requires_grad=True)
        params.append(db_pt)

    elif dustbin == "gaussian":
        db_mean = torch.tensor(dustbin_mean_init, dtype=torch.float32, requires_grad=True)
        if dustbin_std_fixed is not None:
            db_logvar = None  # not learned
        else:
            init_logvar = 2.0 * math.log(dustbin_std_init)
            db_logvar = torch.tensor([init_logvar, init_logvar], requires_grad=True)
            params.append(db_logvar)
        params.append(db_mean)

    elif dustbin == "learned_set":
        torch.manual_seed(seed_init + 9999)
        db_pts = torch.randn(n_db, 2) * 0.3 + torch.tensor(dustbin_mean_init)
        db_pts = db_pts.requires_grad_(True)
        params.append(db_pts)

    elif dustbin == "fixed_circle":
        radius = config.get("circle_radius", 0.5)
        center = torch.tensor(dustbin_mean_init, dtype=torch.float32)
        angles = torch.linspace(0, 2 * math.pi, n_db + 1)[:-1]
        db_pts_fixed = center.unsqueeze(0) + radius * torch.stack([angles.cos(), angles.sin()], dim=1)
        # not learnable

    elif dustbin == "fixed_grid":
        side = int(math.ceil(math.sqrt(n_db)))
        center = torch.tensor(dustbin_mean_init, dtype=torch.float32)
        spacing = config.get("grid_spacing", 0.15)
        xs = torch.linspace(-spacing * (side - 1) / 2, spacing * (side - 1) / 2, side)
        ys = torch.linspace(-spacing * (side - 1) / 2, spacing * (side - 1) / 2, side)
        grid = torch.stack(torch.meshgrid(xs, ys, indexing='xy'), dim=-1).reshape(-1, 2)[:n_db]
        db_pts_fixed = grid + center
        # not learnable

    opt = torch.optim.Adam(params, lr=lr)

    for step in range(n_steps):
        opt.zero_grad()

        # Build padded target matrix
        if dustbin == "degenerate":
            padded = torch.cat([targets, db_pt.unsqueeze(0).expand(n_db, -1)], dim=0)
        elif dustbin == "gaussian":
            if dustbin_std_fixed is not None:
                std = torch.tensor([dustbin_std_fixed, dustbin_std_fixed])
            else:
                std = torch.exp(0.5 * db_logvar)
            db_tgts = db_mean + std * torch.randn(n_db, 2)
            padded = torch.cat([targets, db_tgts], dim=0)
        elif dustbin == "learned_set":
            padded = torch.cat([targets, db_pts], dim=0)
        elif dustbin in ("fixed_circle", "fixed_grid"):
            padded = torch.cat([targets, db_pts_fixed], dim=0)
        else:
            padded = targets

        cost = torch.cdist(preds.unsqueeze(0), padded.unsqueeze(0))[0]

        # Compute loss
        if loss_type == "product":
            loss = product_loss_raw(cost)
        elif loss_type == "product_lognorm":
            loss = product_loss_lognorm(cost)
        elif loss_type == "sinkhorn":
            loss = sinkhorn_loss(cost, tau=tau)
        elif loss_type == "hungarian":
            loss = hungarian_loss(cost)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        loss.backward()
        opt.step()

    # --- Evaluate existence ---
    with torch.no_grad():
        if dustbin == "degenerate":
            center = db_pt.detach()
        elif dustbin == "gaussian":
            center = db_mean.detach()
        elif dustbin == "learned_set":
            center = db_pts.detach().mean(dim=0)
        elif dustbin in ("fixed_circle", "fixed_grid"):
            center = db_pts_fixed.mean(dim=0)
        else:
            center = None

        if center is not None:
            dist_db = torch.norm(preds.detach() - center, dim=1)
            dist_tgt = torch.cdist(
                preds.detach().unsqueeze(0), targets.unsqueeze(0)
            )[0].min(dim=1).values
            exist_pred = (dist_tgt < dist_db).float().numpy()
        else:
            exist_pred = np.zeros(n_pred)

    return evaluate(preds.detach(), targets, exist_pred)


def run_multi_seed(config, n_seeds=5, **kwargs):
    """Run config over multiple seeds, return stats."""
    ml2s, nps, f1s = [], [], []
    for s in range(n_seeds):
        ml2, np_, f1 = run_single(
            config, seed_data=s * 7, seed_init=s * 13 + 1, **kwargs
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
        "f1_perfect": int(np.sum(np.array(f1s) == 1.0)),
        "n_seeds": n_seeds,
    }


# ------------------------------------------------------------------ #
#  Config grid                                                         #
# ------------------------------------------------------------------ #

def build_configs():
    """Build the full grid of experiments."""
    configs = {}

    # ---- Baselines ----
    # 0a. Sinkhorn + degenerate (expected F1~0.99)
    configs["sinkhorn_degenerate"] = {
        "loss_type": "sinkhorn", "dustbin": "degenerate", "tau": 0.3,
    }
    # 0b. Hungarian + degenerate
    configs["hungarian_degenerate"] = {
        "loss_type": "hungarian", "dustbin": "degenerate",
    }

    # ---- Experiment 1: Product + degenerate dustbin (baseline, expect F1~0.45) ----
    configs["product_degenerate"] = {
        "loss_type": "product", "dustbin": "degenerate",
    }

    # ---- Experiment 2: Product + Gaussian dustbin (learned mean+var) ----
    configs["product_gaussian_learned"] = {
        "loss_type": "product", "dustbin": "gaussian",
        "dustbin_mean_init": [-1.0, -1.0], "dustbin_std_init": 0.1,
    }

    # ---- Experiment 3: Product + learned individual dustbins ----
    configs["product_learned_set"] = {
        "loss_type": "product", "dustbin": "learned_set",
        "dustbin_mean_init": [-1.0, -1.0],
    }

    # ---- Experiment 4: Product + Gaussian, sweep std ----
    for std in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
        configs[f"product_gaussian_fixstd{std}"] = {
            "loss_type": "product", "dustbin": "gaussian",
            "dustbin_mean_init": [-1.0, -1.0],
            "dustbin_std_fixed": std,
        }

    # ---- Experiment 5: Product + Gaussian, sweep init location ----
    for label, mean in [("neg1", [-1.0, -1.0]), ("pos2", [2.0, 2.0]), ("half", [0.5, -1.0])]:
        configs[f"product_gaussian_loc_{label}"] = {
            "loss_type": "product", "dustbin": "gaussian",
            "dustbin_mean_init": mean, "dustbin_std_init": 0.1,
        }

    # ---- Experiment 6: Product + fixed-but-spread dustbins ----
    configs["product_fixed_circle"] = {
        "loss_type": "product", "dustbin": "fixed_circle",
        "dustbin_mean_init": [-1.0, -1.0], "circle_radius": 0.5,
    }
    configs["product_fixed_circle_r02"] = {
        "loss_type": "product", "dustbin": "fixed_circle",
        "dustbin_mean_init": [-1.0, -1.0], "circle_radius": 0.2,
    }
    configs["product_fixed_grid"] = {
        "loss_type": "product", "dustbin": "fixed_grid",
        "dustbin_mean_init": [-1.0, -1.0], "grid_spacing": 0.15,
    }
    configs["product_fixed_grid_tight"] = {
        "loss_type": "product", "dustbin": "fixed_grid",
        "dustbin_mean_init": [-1.0, -1.0], "grid_spacing": 0.05,
    }

    # ---- Log-normalized variants of the best dustbin strategies ----
    configs["product_lognorm_degenerate"] = {
        "loss_type": "product_lognorm", "dustbin": "degenerate",
    }
    configs["product_lognorm_gaussian_learned"] = {
        "loss_type": "product_lognorm", "dustbin": "gaussian",
        "dustbin_mean_init": [-1.0, -1.0], "dustbin_std_init": 0.1,
    }
    configs["product_lognorm_learned_set"] = {
        "loss_type": "product_lognorm", "dustbin": "learned_set",
        "dustbin_mean_init": [-1.0, -1.0],
    }
    configs["product_lognorm_fixed_circle"] = {
        "loss_type": "product_lognorm", "dustbin": "fixed_circle",
        "dustbin_mean_init": [-1.0, -1.0], "circle_radius": 0.5,
    }
    configs["product_lognorm_fixed_grid"] = {
        "loss_type": "product_lognorm", "dustbin": "fixed_grid",
        "dustbin_mean_init": [-1.0, -1.0], "grid_spacing": 0.15,
    }

    # ---- Combined: log-norm + Gaussian std sweep ----
    for std in [0.05, 0.1, 0.3, 0.5]:
        configs[f"product_lognorm_gaussian_fixstd{std}"] = {
            "loss_type": "product_lognorm", "dustbin": "gaussian",
            "dustbin_mean_init": [-1.0, -1.0],
            "dustbin_std_fixed": std,
        }

    return configs


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Gaussian dustbin experiment for Product Loss"
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-target", type=int, default=5)
    parser.add_argument("--n-pred", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--filter", type=str, default=None,
                        help="Only run configs containing this substring")
    parser.add_argument("--output", type=str,
                        default="results/gaussian_product.json")
    args = parser.parse_args()

    configs = build_configs()
    if args.filter:
        configs = {k: v for k, v in configs.items() if args.filter in k}

    print(f"Gaussian Product Dustbin Experiment")
    print(f"  {len(configs)} configs x {args.seeds} seeds")
    print(f"  N_target={args.n_target}, N_pred={args.n_pred}, "
          f"{args.n_steps} steps, lr={args.lr}")
    print()

    kwargs = dict(
        n_target=args.n_target, n_pred=args.n_pred,
        n_steps=args.n_steps, lr=args.lr,
    )

    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    log_path = str(outpath.with_suffix(".log"))
    with open(log_path, "w") as f:
        f.write(f"Gaussian Product Dustbin Experiment\n")
        f.write(f"{len(configs)} configs x {args.seeds} seeds\n\n")
    print(f"Progress log: {log_path}")

    results = {}
    t0 = time.time()

    for name, cfg in configs.items():
        stats = run_multi_seed(cfg, n_seeds=args.seeds, **kwargs)
        results[name] = stats
        elapsed = time.time() - t0
        msg = (f"[{elapsed:7.1f}s] {name:45s}  "
               f"ml2={stats['ml2_mean']:.4f}+-{stats['ml2_std']:.3f}  "
               f"F1={stats['f1_mean']:.3f}+-{stats['f1_std']:.3f}  "
               f"({stats['f1_perfect']}/{args.seeds} perfect)")
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    # Summary table sorted by F1
    print()
    print("=" * 100)
    print(f"{'Config':45s} {'ml2':>14s} {'F1':>14s} {'perf':>6s}")
    print("-" * 100)
    for name in sorted(results, key=lambda k: -results[k]["f1_mean"]):
        s = results[name]
        print(f"{name:45s} {s['ml2_mean']:.4f}+-{s['ml2_std']:.3f}  "
              f"{s['f1_mean']:.3f}+-{s['f1_std']:.3f}   "
              f"{s['f1_perfect']:2d}/{args.seeds}")
    print("=" * 100)

    # Group summary by category
    print("\n--- Summary by category ---")
    categories = {
        "Baselines (Sinkhorn/Hungarian)": lambda k: k.startswith(("sinkhorn", "hungarian")),
        "Product + degenerate":           lambda k: k == "product_degenerate",
        "Product + Gaussian (learned)":   lambda k: "gaussian_learned" in k and "lognorm" not in k,
        "Product + Gaussian (fixed std)": lambda k: "gaussian_fixstd" in k and "lognorm" not in k,
        "Product + Gaussian (location)":  lambda k: "gaussian_loc" in k and "lognorm" not in k,
        "Product + learned set":          lambda k: k == "product_learned_set",
        "Product + fixed spread":         lambda k: "fixed_" in k and "lognorm" not in k,
        "Log-norm variants":              lambda k: "lognorm" in k,
    }
    for cat_name, pred in categories.items():
        matching = {k: v for k, v in results.items() if pred(k)}
        if not matching:
            continue
        print(f"\n  {cat_name}:")
        for name in sorted(matching, key=lambda k: -matching[k]["f1_mean"]):
            s = matching[name]
            print(f"    {name:43s} F1={s['f1_mean']:.3f}+-{s['f1_std']:.3f}  "
                  f"ml2={s['ml2_mean']:.4f}")

    # Save
    with open(outpath, "w") as f:
        json.dump({
            "configs": configs,
            "results": results,
            "args": vars(args),
        }, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
