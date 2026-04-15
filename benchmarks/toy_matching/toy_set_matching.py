"""
Toy set matching: optimize predictions directly via gradient descent.

Compare Chamfer, Hungarian, PM3, and Product Loss on a simple 2D problem
with variable cardinality (N_pred > N_target). No neural network — just
raw optimization to isolate matching behavior from training dynamics.

Usage:
    python toy_set_matching.py              # run all losses, save plots
    python toy_set_matching.py --loss pm3   # run single loss
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------ #
#  Target generation                                                   #
# ------------------------------------------------------------------ #

def make_targets(n_target: int, seed: int = 42) -> torch.Tensor:
    """Generate fixed 2D target points in [0, 1]^2."""
    rng = torch.Generator().manual_seed(seed)
    return torch.rand(n_target, 2, generator=rng)


def make_preds_init(n_pred: int, seed: int = 123) -> torch.Tensor:
    """Initialize predictions randomly in [0, 1]^2."""
    rng = torch.Generator().manual_seed(seed)
    return torch.rand(n_pred, 2, generator=rng)


# ------------------------------------------------------------------ #
#  Loss functions (no batch dimension — single sample)                 #
# ------------------------------------------------------------------ #

def chamfer_loss(pred, target):
    """Chamfer distance: nearest-neighbour in both directions."""
    cost = torch.cdist(pred, target, p=2)  # (N, M)
    nn_pred = cost.min(dim=1).values.mean()   # each pred → nearest target
    nn_tgt = cost.min(dim=0).values.mean()    # each target → nearest pred
    return nn_pred + nn_tgt


def hungarian_loss(pred, target):
    """Hungarian matching: optimal 1-to-1 assignment, gradients through cost only."""
    cost = torch.cdist(pred, target, p=2)  # (N, M)
    cost_np = cost.detach().cpu().numpy()
    row_idx, col_idx = linear_sum_assignment(cost_np)
    matched_cost = cost[row_idx, col_idx]
    return matched_cost.mean()


def pm3_loss(pred, target, tau=0.1, exist_logits=None):
    """
    PM3 (Power-SoftMin) loss with optional existence prediction.

    If exist_logits is provided, adds existence BCE loss using dustbin-derived targets.
    Returns (feature_loss, exist_loss, exist_probs) or just feature_loss.
    """
    N, D = pred.shape
    M = target.shape[0]
    cost = torch.cdist(pred, target, p=2)  # (N, M)

    # Coverage: for each target, softmin over predictions
    cov_w = torch.softmax(-cost / tau, dim=0)  # (N, M)
    l_cov = (cov_w * cost).sum(dim=0).mean()   # mean over targets

    # Precision: for each pred, softmin over targets
    prec_w = torch.softmax(-cost / tau, dim=1)  # (N, M)
    l_prec = (prec_w * cost).sum(dim=1).mean()  # mean over preds

    feature_loss = 0.5 * (l_cov + l_prec)

    if exist_logits is None:
        return feature_loss

    # Existence via dustbin-style: how much does each pred's
    # best-match weight indicate it found a real target?
    # Use the max precision weight as a proxy for "matched-ness"
    # (This is conceptually different from the ej-vae dustbin —
    #  here we test whether soft weights can provide existence signal at all)
    exist_probs = torch.sigmoid(exist_logits)
    return feature_loss, exist_probs


def pm3_dustbin_loss(pred, target, tau=0.1, dustbin_cost=1.0,
                     exist_logits=None, multi_dustbin=False):
    """
    PM3 with explicit dustbin column in precision softmax.

    Returns (feature_loss, exist_target, exist_probs) for analysis.
    """
    N, D = pred.shape
    M = target.shape[0]
    cost = torch.cdist(pred, target, p=2)  # (N, M)

    # Coverage: unchanged
    cov_w = torch.softmax(-cost / tau, dim=0)
    l_cov = (cov_w * cost).sum(dim=0).mean()

    # Precision: add dustbin column
    dustbin_col = torch.full((N, 1), dustbin_cost)
    cost_aug = torch.cat([cost, dustbin_col], dim=1)  # (N, M+1)

    logits = -cost_aug / tau
    if multi_dustbin:
        K = max(N - M, 1)
        logits[:, -1] = logits[:, -1] + math.log(K)

    prec_w = torch.softmax(logits, dim=1)  # (N, M+1)
    l_prec = (prec_w * cost_aug).sum(dim=1).mean()

    feature_loss = 0.5 * (l_cov + l_prec)

    # Existence target from dustbin weight
    dustbin_weight = prec_w[:, -1]
    exist_target = (1.0 - dustbin_weight).detach()

    exist_probs = torch.sigmoid(exist_logits) if exist_logits is not None else None

    return feature_loss, exist_target, exist_probs


def product_loss(pred, target, eps=1e-8):
    """
    Product Loss: L = sum_j prod_i D_ij + sum_i prod_j D_ij.

    Known to fail due to uniform gradients at initialization.
    """
    cost = torch.cdist(pred, target, p=2) + eps  # (N, M)

    # Coverage: for each target, product over preds
    l_cov = cost.log().sum(dim=0).exp().mean()  # sum of products

    # Precision: for each pred, product over targets
    l_prec = cost.log().sum(dim=1).exp().mean()

    return 0.5 * (l_cov + l_prec)


# ------------------------------------------------------------------ #
#  Optimization loop                                                   #
# ------------------------------------------------------------------ #

def optimize(loss_fn, targets, n_pred, n_steps=500, lr=0.01,
             with_existence=False, **loss_kwargs):
    """
    Optimize predictions to match targets using gradient descent.

    Returns dict with trajectories of loss, positions, and existence.
    """
    pred = make_preds_init(n_pred).clone().requires_grad_(True)

    exist_logits = None
    if with_existence:
        exist_logits = torch.zeros(n_pred, requires_grad=True)
        optimizer = torch.optim.Adam([pred, exist_logits], lr=lr)
    else:
        optimizer = torch.optim.Adam([pred], lr=lr)

    history = {
        "loss": [],
        "pred_positions": [],
        "matched_l2": [],
        "exist_probs": [],
        "exist_targets": [],
    }

    for step in range(n_steps):
        optimizer.zero_grad()

        if with_existence and "dustbin" in str(loss_fn.__name__):
            feat_loss, exist_tgt, exist_prob = loss_fn(
                pred, targets, exist_logits=exist_logits, **loss_kwargs
            )
            exist_loss = F.binary_cross_entropy(
                exist_prob, exist_tgt, reduction="mean"
            )
            # Anneal existence loss
            anneal = min(1.0, max(0.0, (step - 100) / 200))
            loss = feat_loss + anneal * exist_loss
            history["exist_probs"].append(exist_prob.detach().numpy().copy())
            history["exist_targets"].append(exist_tgt.numpy().copy())
        else:
            result = loss_fn(pred, targets, **loss_kwargs)
            if isinstance(result, tuple):
                loss = result[0]
            else:
                loss = result

        loss.backward()
        optimizer.step()

        # Record
        history["loss"].append(loss.item())
        history["pred_positions"].append(pred.detach().numpy().copy())

        # Compute matched L2 (Hungarian) for fair comparison
        with torch.no_grad():
            cost = torch.cdist(pred, targets, p=2)
            cost_np = cost.numpy()
            row_idx, col_idx = linear_sum_assignment(cost_np)
            matched_l2 = cost[row_idx, col_idx].mean().item()
            history["matched_l2"].append(matched_l2)

    return history


# ------------------------------------------------------------------ #
#  Visualization                                                       #
# ------------------------------------------------------------------ #

def plot_convergence(results, targets, save_path="toy_convergence.png"):
    """Plot convergence curves and final positions for all losses."""
    fig, axes = plt.subplots(2, len(results), figsize=(5 * len(results), 8))
    if len(results) == 1:
        axes = axes.reshape(2, 1)

    colors = {
        "chamfer": "#4CAF50",
        "hungarian": "#2196F3",
        "pm3": "#FF5722",
        "pm3_dustbin": "#9C27B0",
        "pm3_multi_dustbin": "#E91E63",
        "product": "#795548",
    }

    for i, (name, hist) in enumerate(results.items()):
        color = colors.get(name, "gray")

        # Top: convergence
        ax = axes[0, i]
        ax.plot(hist["matched_l2"], color=color, linewidth=2)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Matched L2")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

        # Bottom: final positions
        ax = axes[1, i]
        tgt = targets.numpy()
        final_pred = hist["pred_positions"][-1]

        ax.scatter(tgt[:, 0], tgt[:, 1], c="black", s=80, marker="x",
                   linewidths=2, label="Targets", zorder=5)
        ax.scatter(final_pred[:, 0], final_pred[:, 1], c=color, s=40,
                   alpha=0.7, label="Predictions", zorder=4)

        # Draw Hungarian matching lines
        cost = np.linalg.norm(final_pred[:, None] - tgt[None, :], axis=2)
        row_idx, col_idx = linear_sum_assignment(cost)
        for r, c_idx in zip(row_idx, col_idx):
            ax.plot([final_pred[r, 0], tgt[c_idx, 0]],
                    [final_pred[r, 1], tgt[c_idx, 1]],
                    color=color, alpha=0.3, linewidth=1)

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.set_title(f"{name} (final L2={hist['matched_l2'][-1]:.4f})")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved convergence plot: {save_path}")


def plot_existence(results, n_target, save_path="toy_existence.png"):
    """Plot existence predictions for dustbin-based losses."""
    dustbin_results = {k: v for k, v in results.items()
                       if v["exist_probs"]}
    if not dustbin_results:
        print("No existence results to plot")
        return

    fig, axes = plt.subplots(1, len(dustbin_results),
                              figsize=(6 * len(dustbin_results), 4))
    if len(dustbin_results) == 1:
        axes = [axes]

    for i, (name, hist) in enumerate(dustbin_results.items()):
        ax = axes[i]
        exist_probs = np.array(hist["exist_probs"])  # (steps, N_pred)
        exist_targets = np.array(hist["exist_targets"])  # (steps, N_pred)

        # Plot mean exist prob and target over time
        ax.plot(exist_probs.mean(axis=1), label="mean P(exist)", linewidth=2)
        ax.plot(exist_targets.mean(axis=1), label="mean target",
                linewidth=2, linestyle="--")
        ax.axhline(n_target / exist_probs.shape[1], color="gray",
                   linestyle=":", label=f"true frac ({n_target}/{exist_probs.shape[1]})")
        ax.set_title(name)
        ax.set_xlabel("Step")
        ax.set_ylabel("Existence probability")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved existence plot: {save_path}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Toy set matching experiment")
    parser.add_argument("--n-target", type=int, default=7)
    parser.add_argument("--n-pred", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--loss", type=str, default=None,
                        help="Run single loss: chamfer/hungarian/pm3/pm3_dustbin/product")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--with-existence", action="store_true",
                        help="Enable existence prediction for dustbin losses")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    targets = make_targets(args.n_target)
    print(f"Targets ({args.n_target} points in 2D):")
    print(targets)
    print(f"Predictions: {args.n_pred} slots (excess: {args.n_pred - args.n_target})")
    print()

    loss_fns = {
        "chamfer": (chamfer_loss, {}),
        "hungarian": (hungarian_loss, {}),
        "pm3": (pm3_loss, {"tau": args.tau}),
        "pm3_dustbin": (pm3_dustbin_loss, {"tau": args.tau, "multi_dustbin": False}),
        "pm3_multi_dustbin": (pm3_dustbin_loss, {"tau": args.tau, "multi_dustbin": True}),
        "product": (product_loss, {}),
    }

    if args.loss:
        loss_fns = {args.loss: loss_fns[args.loss]}

    results = {}
    for name, (fn, kwargs) in loss_fns.items():
        use_exist = args.with_existence and "dustbin" in name
        print(f"Running {name}... ", end="", flush=True)
        hist = optimize(fn, targets, args.n_pred, args.n_steps, args.lr,
                        with_existence=use_exist, **kwargs)
        final_l2 = hist["matched_l2"][-1]
        print(f"final matched L2 = {final_l2:.4f}")
        results[name] = hist

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Loss':<25s} {'Final L2':>10s} {'Min L2':>10s} {'Step@Min':>10s}")
    print("-" * 60)
    for name, hist in results.items():
        ml2 = hist["matched_l2"]
        min_l2 = min(ml2)
        min_step = ml2.index(min_l2)
        print(f"{name:<25s} {ml2[-1]:10.4f} {min_l2:10.4f} {min_step:10d}")
    print("=" * 60)

    # Plots
    plot_convergence(results, targets, save_path=str(outdir / "toy_convergence.png"))
    if args.with_existence:
        plot_existence(results, args.n_target,
                       save_path=str(outdir / "toy_existence.png"))


if __name__ == "__main__":
    main()
