"""Toy 2D point set matching: compare set prediction losses.

Trains a small MLP to predict unordered 2D point sets using different
loss functions, and compares convergence, matching quality, and mode collapse.

Usage:
    python examples/toy_set_matching.py [--n_points 10] [--n_seeds 5]
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from influencerformer.losses.product_loss import (
    AnnealedExponentLoss,
    HuberProductLoss,
    LogDistanceProductLoss,
    ProductLoss,
    ProductWeightedSoftMinLoss,
    SigmoidProductLoss,
    SoftMinChamferLoss,
    WarmStartProductLoss,
)
from influencerformer.losses.set_losses import (
    ChamferLoss,
    HungarianLoss,
    SinkhornLoss,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SetPredictor(nn.Module):
    def __init__(self, n_points: int, dim: int = 2, hidden: int = 128):
        super().__init__()
        self.n_points = n_points
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(n_points * dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_points * dim),
        )

    def forward(self, x):
        return self.net(x).reshape(-1, self.n_points, self.dim)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, test_inputs, test_targets, match_threshold=0.3):
    model.eval()
    with torch.no_grad():
        preds = model(test_inputs)
        D = torch.cdist(preds, test_targets)

    B, M, N = D.shape
    matched_dists = []
    n_correct = 0
    n_total = 0
    n_duplicates = 0
    D_np = D.cpu().numpy()

    for b in range(B):
        row_ind, col_ind = linear_sum_assignment(D_np[b])
        matched = D_np[b][row_ind, col_ind]
        matched_dists.append(matched.mean())
        n_correct += (matched < match_threshold).sum()
        n_total += len(row_ind)
        for j in range(N):
            if (D_np[b, :, j] < match_threshold).sum() > 1:
                n_duplicates += 1

    return {
        "matched_distance": np.mean(matched_dists),
        "match_accuracy": n_correct / max(n_total, 1),
        "duplicate_rate": n_duplicates / max(B * N, 1),
    }


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------
def train_one(
    loss_fn, train_targets, test_targets, n_points, n_epochs,
    noise_scale, lr, batch_size, device, eval_every=5,
):
    model = SetPredictor(n_points=n_points).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(train_targets), batch_size=batch_size, shuffle=True)

    test_inputs = (test_targets + noise_scale * torch.randn_like(test_targets)).flatten(-2).to(device)
    test_tgt = test_targets.to(device)

    history = defaultdict(list)
    t0 = time.time()
    has_set_epoch = hasattr(loss_fn, "set_epoch")

    for epoch in range(n_epochs):
        if has_set_epoch:
            loss_fn.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for (targets_batch,) in loader:
            targets_batch = targets_batch.to(device)
            inputs = (targets_batch + noise_scale * torch.randn_like(targets_batch)).flatten(-2)
            preds = model(inputs)
            D = torch.cdist(preds, targets_batch)
            loss = loss_fn(D)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        history["train_loss"].append(epoch_loss / max(n_batches, 1))

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            metrics = evaluate(model, test_inputs, test_tgt)
            history["matched_distance"].append(metrics["matched_distance"])
            history["match_accuracy"].append(metrics["match_accuracy"])
            history["duplicate_rate"].append(metrics["duplicate_rate"])
            history["eval_epoch"].append(epoch)

    history["total_time"] = time.time() - t0
    return history


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------
def run_multi_seed(loss_name, make_loss_fn, train_targets, test_targets, args, device, n_seeds):
    all_runs = []
    for seed in range(n_seeds):
        torch.manual_seed(args.seed + seed * 1000)
        loss_fn = make_loss_fn()
        hist = train_one(
            loss_fn, train_targets, test_targets, args.n_points,
            args.n_epochs, args.noise_scale, args.lr, args.batch_size, device,
        )
        final = {
            "matched_distance": hist["matched_distance"][-1],
            "match_accuracy": hist["match_accuracy"][-1],
            "duplicate_rate": hist["duplicate_rate"][-1],
            "total_time": hist["total_time"],
        }
        all_runs.append({"history": hist, "final": final})
        print(f"  [{loss_name}] seed {seed}: dist={final['matched_distance']:.4f} "
              f"acc={final['match_accuracy']:.3f} dup={final['duplicate_rate']:.3f} "
              f"({final['total_time']:.1f}s)")

    return all_runs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(all_results, n_points, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f"Set Prediction Loss Comparison (N={n_points}, 5 seeds)", fontsize=14)

    # Color scheme
    n = len(all_results)
    cmap = plt.cm.tab20
    colors = {name: cmap(i / max(n - 1, 1)) for i, name in enumerate(all_results)}

    for ax_idx, (metric, title, yscale) in enumerate([
        ("train_loss", "Training Loss", "log"),
        ("matched_distance", "Hungarian-Matched Distance (lower=better)", "linear"),
        ("match_accuracy", "Match Accuracy (higher=better)", "linear"),
        ("duplicate_rate", "Duplicate Rate (lower=better)", "linear"),
    ]):
        ax = axes[ax_idx // 2, ax_idx % 2]

        for name, runs in all_results.items():
            if metric == "train_loss":
                x_key = None
                all_curves = np.array([r["history"]["train_loss"] for r in runs])
            else:
                x_key = "eval_epoch"
                all_curves = np.array([r["history"][metric] for r in runs])

            mean = all_curves.mean(axis=0)
            std = all_curves.std(axis=0)

            if x_key:
                x = runs[0]["history"][x_key]
            else:
                x = list(range(len(mean)))

            ax.plot(x, mean, color=colors[name], label=name, linewidth=1.5)
            ax.fill_between(x, mean - std, mean + std, color=colors[name], alpha=0.15)

        ax.set_xlabel("Epoch")
        ax.set_title(title)
        if yscale == "log":
            ax.set_yscale("log")
        if "accuracy" in metric:
            ax.set_ylim(-0.05, 1.05)
        if "duplicate" in metric:
            ax.set_ylim(-0.01, max(0.1, ax.get_ylim()[1]))
        ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"set_matching_N{n_points}.png"
    plt.savefig(path, dpi=150)
    print(f"Saved plot to {path}")
    plt.close()


def print_summary_table(all_results):
    print(f"\n{'='*90}")
    print(f"{'Loss':<22} {'Match Dist':>14} {'Match Acc':>14} {'Dup Rate':>14} {'Time (s)':>12}")
    print(f"{'':<22} {'mean±std':>14} {'mean±std':>14} {'mean±std':>14} {'mean':>12}")
    print("-" * 90)
    for name, runs in all_results.items():
        finals = [r["final"] for r in runs]
        md = [f["matched_distance"] for f in finals]
        ma = [f["match_accuracy"] for f in finals]
        dr = [f["duplicate_rate"] for f in finals]
        tt = [f["total_time"] for f in finals]
        print(
            f"{name:<22} "
            f"{np.mean(md):>6.4f}±{np.std(md):.4f} "
            f"{np.mean(ma):>6.3f}±{np.std(ma):.3f}  "
            f"{np.mean(dr):>6.4f}±{np.std(dr):.4f} "
            f"{np.mean(tt):>10.1f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_points", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, N={args.n_points}, epochs={args.n_epochs}, seeds={args.n_seeds}")

    # Fixed data across all methods (generated once)
    torch.manual_seed(args.seed)
    train_targets = torch.randn(args.n_train, args.n_points, 2)
    test_targets = torch.randn(args.n_test, args.n_points, 2)

    # All loss constructors
    loss_configs = {
        # --- Baselines ---
        "chamfer":        lambda: ChamferLoss(),
        "hungarian":      lambda: HungarianLoss(),
        "sinkhorn":       lambda: SinkhornLoss(eps=0.1, n_iters=20),
        # --- Softmin family ---
        "softmin_0.1":    lambda: SoftMinChamferLoss(temperature=0.1),
        "pw_softmin":     lambda: ProductWeightedSoftMinLoss(temperature=0.1),
        # --- Product family ---
        "product_gm":     lambda: ProductLoss(),
        "log_product":    lambda: LogDistanceProductLoss(),
        "sigmoid_prod":   lambda: SigmoidProductLoss(margin=1.0, scale=5.0),
        "huber_product":  lambda: HuberProductLoss(delta=1.0),
        # --- Annealing ---
        "warm_start":     lambda: WarmStartProductLoss(transition_epoch=80, blend_width=40),
        "annealed_p":     lambda: AnnealedExponentLoss(p_start=10.0, p_end=0.5, anneal_epochs=200),
    }

    all_results = {}
    for name, make_fn in loss_configs.items():
        print(f"\n{'='*60}")
        print(f"  {name.upper()}")
        print(f"{'='*60}")
        all_results[name] = run_multi_seed(
            name, make_fn, train_targets, test_targets, args, device, args.n_seeds
        )

    print_summary_table(all_results)

    output_dir = Path(__file__).parent / "results"
    plot_comparison(all_results, args.n_points, output_dir)


if __name__ == "__main__":
    main()
