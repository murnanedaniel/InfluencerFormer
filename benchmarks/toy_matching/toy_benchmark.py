"""
Comprehensive toy set matching benchmark.

Tests all combinations of matching mechanisms × existence approaches
on a simple 2D direct optimization problem (no neural network).

Usage:
    python toy_benchmark.py                    # run all, 10 seeds
    python toy_benchmark.py --seeds 3          # quick test
    python toy_benchmark.py --filter sinkhorn  # subset of configs
    python toy_benchmark.py --workers 4        # parallel
"""

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------------ #
#  Sinkhorn utilities                                                  #
# ------------------------------------------------------------------ #

def sinkhorn(log_alpha, n_iters=30):
    """Standard Sinkhorn: uniform marginals, square or rectangular."""
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
    return log_alpha.exp()


def sinkhorn_superglue(scores, n_unmatched_rows, n_unmatched_cols, n_iters=30):
    """SuperGlue-style Sinkhorn with non-uniform marginals for dustbin."""
    N1, M1 = scores.shape
    log_mu = torch.zeros(N1)
    log_mu[-1] = math.log(max(n_unmatched_cols, 1))
    log_nu = torch.zeros(M1)
    log_nu[-1] = math.log(max(n_unmatched_rows, 1))
    log_alpha = scores.clone()
    for _ in range(n_iters):
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True) - log_mu.unsqueeze(1))
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=0, keepdim=True) - log_nu.unsqueeze(0))
    return log_alpha.exp()


# ------------------------------------------------------------------ #
#  Evaluation                                                          #
# ------------------------------------------------------------------ #

def evaluate(preds, targets, exist_pred):
    """Compute matched L2, n_pred, and F1 against Hungarian ground truth."""
    with torch.no_grad():
        N = preds.shape[0]
        M = targets.shape[0]
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
#  Single experiment runner                                            #
# ------------------------------------------------------------------ #

def run_config(config, seed_data, seed_init, n_target=5, n_pred=20,
               n_steps=10000, lr=0.005):
    """Run one configuration with one seed. Returns (ml2, n_pred, f1)."""
    torch.manual_seed(seed_data)
    targets = torch.rand(n_target, 2)

    torch.manual_seed(seed_init)
    preds = torch.rand(n_pred, 2, requires_grad=True)

    matching = config["matching"]
    dustbin = config.get("dustbin", "none")
    tau = config.get("tau", 0.3)
    power = config.get("power", 1.0)
    n_db = n_pred - n_target

    # Set up dustbin parameters
    params = [preds]
    db_pt = None

    if dustbin == "degenerate":
        db_pt = torch.tensor([-1.0, -1.0], requires_grad=True)
        params.append(db_pt)
    elif dustbin == "gaussian":
        db_mean = torch.tensor([-1.0, -1.0], requires_grad=True)
        db_logvar = torch.tensor([0.0, 0.0], requires_grad=True)
        params += [db_mean, db_logvar]
    elif dustbin == "learned_set":
        db_pts = torch.randn(n_db, 2) * 0.3 + torch.tensor([-1.0, -1.0])
        db_pts = db_pts.requires_grad_(True)
        params.append(db_pts)
    elif dustbin == "superglue_score":
        db_score = torch.tensor(1.0, requires_grad=True)
        params.append(db_score)

    opt = torch.optim.Adam(params, lr=lr)

    for step in range(n_steps):
        opt.zero_grad()

        # Build cost matrix
        if dustbin == "degenerate":
            padded = torch.cat([targets, db_pt.unsqueeze(0).expand(n_db, -1)], dim=0)
            cost = torch.cdist(preds.unsqueeze(0), padded.unsqueeze(0))[0]
        elif dustbin == "gaussian":
            std = torch.exp(0.5 * db_logvar)
            db_tgts = db_mean + std * torch.randn(n_db, 2)
            padded = torch.cat([targets, db_tgts], dim=0)
            cost = torch.cdist(preds.unsqueeze(0), padded.unsqueeze(0))[0]
        elif dustbin == "learned_set":
            padded = torch.cat([targets, db_pts], dim=0)
            cost = torch.cdist(preds.unsqueeze(0), padded.unsqueeze(0))[0]
        elif dustbin == "superglue_score":
            cost_real = torch.cdist(preds.unsqueeze(0), targets.unsqueeze(0))[0]
            db_cols = db_score.abs().expand(n_pred, n_db)
            cost = torch.cat([cost_real, db_cols], dim=1)
        else:
            cost = torch.cdist(preds.unsqueeze(0), targets.unsqueeze(0))[0]

        # Compute assignment weights
        if matching == "sinkhorn":
            P = sinkhorn(-cost / tau, n_iters=30)
            sm_per_col = (P * cost).sum(dim=0)
            sm_per_row = (P * cost).sum(dim=1)
        elif matching == "superglue_sinkhorn":
            cost_real = torch.cdist(preds.unsqueeze(0), targets.unsqueeze(0))[0]
            scores = torch.zeros(n_pred + 1, n_target + 1)
            scores[:n_pred, :n_target] = -cost_real / tau
            scores[:n_pred, -1] = db_score if dustbin == "superglue_score" else 0
            scores[-1, :n_target] = db_score if dustbin == "superglue_score" else 0
            P = sinkhorn_superglue(scores, n_db, n_db, n_iters=30)
            loss = (P[:n_pred, :n_target] * cost_real).sum()
            loss.backward()
            opt.step()
            continue
        elif matching == "softmax":
            w_cov = torch.softmax(-cost / tau, dim=0)
            w_prec = torch.softmax(-cost / tau, dim=1)
            sm_per_col = (w_cov * cost).sum(dim=0)
            sm_per_row = (w_prec * cost).sum(dim=1)
        elif matching == "chamfer":
            sm_per_col = cost.min(dim=0).values
            sm_per_row = cost.min(dim=1).values
        elif matching == "hungarian":
            cost_np = cost.detach().numpy()
            row, col = linear_sum_assignment(cost_np)
            loss = cost[row, col].mean()
            loss.backward()
            opt.step()
            continue

        # Apply power and aggregate
        if power != 1.0:
            l_cov = sm_per_col.pow(power).mean()
            l_prec = sm_per_row.pow(power).mean()
        else:
            l_cov = sm_per_col.mean()
            l_prec = sm_per_row.mean()

        loss = 0.5 * (l_cov + l_prec)
        loss.backward()
        opt.step()

    # Evaluate existence
    with torch.no_grad():
        if dustbin in ("degenerate", "gaussian", "learned_set"):
            if dustbin == "degenerate":
                center = db_pt.detach()
            elif dustbin == "gaussian":
                center = db_mean.detach()
            elif dustbin == "learned_set":
                center = db_pts.detach().mean(dim=0)
            dist_db = torch.norm(preds - center, dim=1)
            dist_tgt = torch.cdist(
                preds.unsqueeze(0), targets.unsqueeze(0)
            )[0].min(dim=1).values
            exist_pred = (dist_tgt < dist_db).float().numpy()
        elif dustbin == "superglue_score":
            cost_real = torch.cdist(preds.unsqueeze(0), targets.unsqueeze(0))[0]
            db_cols = db_score.abs().expand(n_pred, n_db)
            cost_full = torch.cat([cost_real, db_cols], dim=1)
            P_final = sinkhorn(-cost_full / tau, n_iters=50)
            real_weight = P_final[:, :n_target].sum(dim=1)
            exist_pred = (real_weight > 0.5).float().numpy()
        else:
            exist_pred = np.zeros(n_pred)

    return evaluate(preds.detach(), targets, exist_pred)


def run_multi_seed(config, n_seeds=10, **kwargs):
    """Run config over multiple seeds, return stats."""
    ml2s, nps, f1s = [], [], []
    for s in range(n_seeds):
        ml2, np_, f1 = run_config(
            config, seed_data=s * 7, seed_init=s * 13 + 1, **kwargs
        )
        ml2s.append(ml2)
        nps.append(np_)
        f1s.append(f1)
    return {
        "ml2_mean": np.mean(ml2s), "ml2_std": np.std(ml2s),
        "npred_mean": np.mean(nps), "npred_std": np.std(nps),
        "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
        "f1_perfect": int(np.sum(np.array(f1s) == 1.0)),
        "n_seeds": n_seeds,
    }


def run_worker(args):
    """Worker for parallel execution. Writes progress to a shared log file."""
    name, config, n_seeds, kwargs, log_path = args
    stats = run_multi_seed(config, n_seeds=n_seeds, **kwargs)
    line = (f"{name:45s}  ml2={stats['ml2_mean']:.4f}  "
            f"F1={stats['f1_mean']:.2f}±{stats['f1_std']:.2f}  "
            f"({stats['f1_perfect']}/{n_seeds} perfect)\n")
    if log_path:
        with open(log_path, "a") as f:
            f.write(line)
            f.flush()
    return name, stats


# ------------------------------------------------------------------ #
#  Config grid                                                         #
# ------------------------------------------------------------------ #

def build_configs():
    """Build the full grid of configurations to test."""
    configs = {}

    # === Matching × Dustbin × Tau ===
    for matching in ["sinkhorn", "softmax"]:
        for dustbin in ["degenerate", "gaussian", "learned_set", "none"]:
            for tau in [0.2, 0.3]:
                name = f"{matching}_{dustbin}_tau{tau}"
                configs[name] = {
                    "matching": matching, "dustbin": dustbin, "tau": tau
                }

    # Sinkhorn + superglue score
    for tau in [0.2, 0.3]:
        configs[f"sinkhorn_superglue_tau{tau}"] = {
            "matching": "sinkhorn", "dustbin": "superglue_score", "tau": tau
        }

    # Chamfer (no dustbin, no tau)
    configs["chamfer_none"] = {"matching": "chamfer", "dustbin": "none"}
    configs["chamfer_degenerate"] = {"matching": "chamfer", "dustbin": "degenerate"}

    # Hungarian
    configs["hungarian_none"] = {"matching": "hungarian", "dustbin": "none"}
    configs["hungarian_degenerate"] = {"matching": "hungarian", "dustbin": "degenerate"}

    # Power sweep for best sinkhorn config
    for p in [1.0, 2.0, 3.0, 5.0]:
        configs[f"sinkhorn_degenerate_tau0.2_p{p}"] = {
            "matching": "sinkhorn", "dustbin": "degenerate", "tau": 0.2, "power": p
        }

    return configs


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-target", type=int, default=5)
    parser.add_argument("--n-pred", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--filter", type=str, default=None,
                        help="Only run configs containing this substring")
    parser.add_argument("--output", type=str, default="results/benchmark.json")
    args = parser.parse_args()

    configs = build_configs()
    if args.filter:
        configs = {k: v for k, v in configs.items() if args.filter in k}

    print(f"Running {len(configs)} configs × {args.seeds} seeds "
          f"(N_target={args.n_target}, N_pred={args.n_pred}, "
          f"{args.n_steps} steps)")
    print()

    kwargs = dict(
        n_target=args.n_target, n_pred=args.n_pred,
        n_steps=args.n_steps, lr=args.lr,
    )

    results = {}
    t0 = time.time()

    # Progress log file (written by workers, flushed immediately)
    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    log_path = str(outpath.with_suffix(".log"))
    with open(log_path, "w") as f:
        f.write(f"Running {len(configs)} configs × {args.seeds} seeds\n")
    print(f"Progress log: {log_path}")

    if args.workers > 1:
        work_items = [
            (name, cfg, args.seeds, kwargs, log_path)
            for name, cfg in configs.items()
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for name, stats in executor.map(run_worker, work_items):
                results[name] = stats
                elapsed = time.time() - t0
                msg = (f"[{elapsed:6.0f}s] {name:45s}  "
                       f"ml2={stats['ml2_mean']:.4f}  "
                       f"F1={stats['f1_mean']:.2f}±{stats['f1_std']:.2f}  "
                       f"({stats['f1_perfect']}/{args.seeds} perfect)")
                print(msg, flush=True)
    else:
        for name, cfg in configs.items():
            stats = run_multi_seed(cfg, n_seeds=args.seeds, **kwargs)
            results[name] = stats
            elapsed = time.time() - t0
            msg = (f"[{elapsed:6.0f}s] {name:45s}  "
                   f"ml2={stats['ml2_mean']:.4f}  "
                   f"F1={stats['f1_mean']:.2f}±{stats['f1_std']:.2f}  "
                   f"({stats['f1_perfect']}/{args.seeds} perfect)")
            print(msg, flush=True)
            with open(log_path, "a") as f:
                f.write(msg + "\n")

    # Summary table sorted by F1
    print("\n" + "=" * 90)
    print(f"{'Config':45s} {'ml2':>12s} {'n_pred':>10s} {'F1':>14s} {'perf':>6s}")
    print("-" * 90)
    for name in sorted(results, key=lambda k: -results[k]["f1_mean"]):
        s = results[name]
        print(f"{name:45s} {s['ml2_mean']:.4f}±{s['ml2_std']:.3f} "
              f"{s['npred_mean']:5.1f}±{s['npred_std']:.1f} "
              f"{s['f1_mean']:.2f}±{s['f1_std']:.2f}  "
              f"{s['f1_perfect']:2d}/{args.seeds}")
    print("=" * 90)

    # Save
    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump({"configs": {k: v for k, v in configs.items()},
                   "results": results,
                   "args": vars(args)}, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
