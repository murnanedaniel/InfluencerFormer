"""Toy 2D point set matching: compare set prediction losses.

Trains a small MLP to predict unordered 2D point sets using different
loss functions (Product, Chamfer, Hungarian, Sinkhorn, Ordered), and
compares convergence speed, matching quality, and mode collapse rate.

Usage:
    python examples/toy_set_matching.py [--n_points 10] [--n_epochs 300]

No external datasets needed — generates random 2D point clouds.
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from influencerformer.losses import (
    ChamferLoss,
    HungarianLoss,
    OrderedLoss,
    ProductLoss,
    ProductWeightedSoftMinLoss,
    SinkhornLoss,
    SoftMinChamferLoss,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SetPredictor(nn.Module):
    """MLP that maps corrupted point set → clean point set.

    Input: (B, N*dim) flattened corrupted points
    Output: (B, N, dim) predicted points
    """

    def __init__(self, n_points: int, dim: int = 2, hidden: int = 128):
        super().__init__()
        self.n_points = n_points
        self.dim = dim
        in_dim = n_points * dim
        out_dim = n_points * dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x).reshape(-1, self.n_points, self.dim)


# ---------------------------------------------------------------------------
# Evaluation (same for all losses — uses Hungarian matching as ground truth)
# ---------------------------------------------------------------------------
def evaluate(model, test_inputs, test_targets, match_threshold=0.3):
    """Evaluate set prediction quality using Hungarian matching.

    Returns:
        dict with matched_distance, match_accuracy, duplicate_rate.
    """
    model.eval()
    with torch.no_grad():
        preds = model(test_inputs)
        D = torch.cdist(preds, test_targets)  # (B, N, N)

    B, M, N = D.shape
    matched_dists = []
    n_correct = 0
    n_total = 0
    n_duplicates = 0

    D_np = D.cpu().numpy()
    preds_np = preds.cpu().numpy()
    targets_np = test_targets.cpu().numpy()

    for b in range(B):
        row_ind, col_ind = linear_sum_assignment(D_np[b])
        matched = D_np[b][row_ind, col_ind]
        matched_dists.append(matched.mean())
        n_correct += (matched < match_threshold).sum()
        n_total += len(row_ind)

        # Duplicate check: for each target, count predictions within threshold
        for j in range(N):
            n_nearby = (D_np[b, :, j] < match_threshold).sum()
            if n_nearby > 1:
                n_duplicates += 1

    return {
        "matched_distance": np.mean(matched_dists),
        "match_accuracy": n_correct / max(n_total, 1),
        "duplicate_rate": n_duplicates / max(B * N, 1),
    }


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------
def train_one_loss(
    loss_name,
    loss_fn,
    train_targets,
    test_targets,
    n_points,
    n_epochs,
    noise_scale,
    lr,
    batch_size,
    device,
    eval_every=5,
):
    """Train a model with one loss function, return metrics history."""
    model = SetPredictor(n_points=n_points, dim=2, hidden=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(train_targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Precompute test inputs
    test_inputs = (
        test_targets + noise_scale * torch.randn_like(test_targets)
    ).flatten(-2).to(device)
    test_tgt = test_targets.to(device)

    history = defaultdict(list)
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for (targets_batch,) in loader:
            targets_batch = targets_batch.to(device)
            noise = noise_scale * torch.randn_like(targets_batch)
            inputs = (targets_batch + noise).flatten(-2)

            preds = model(inputs)
            D = torch.cdist(preds, targets_batch)

            loss = loss_fn(D)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            metrics = evaluate(model, test_inputs, test_tgt)
            history["matched_distance"].append(metrics["matched_distance"])
            history["match_accuracy"].append(metrics["match_accuracy"])
            history["duplicate_rate"].append(metrics["duplicate_rate"])
            history["eval_epoch"].append(epoch)

            if (epoch + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(
                    f"  [{loss_name}] epoch {epoch+1}/{n_epochs} "
                    f"loss={avg_loss:.4f} "
                    f"match_dist={metrics['matched_distance']:.4f} "
                    f"match_acc={metrics['match_accuracy']:.3f} "
                    f"dup_rate={metrics['duplicate_rate']:.3f} "
                    f"({elapsed:.1f}s)"
                )

    total_time = time.time() - t0
    history["total_time"] = total_time
    return history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(all_results, n_points, output_dir):
    """Plot side-by-side comparison of all losses."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Set Prediction Loss Comparison (N={n_points} points)", fontsize=14)

    colors = {
        "product_gm": "#E69F00",
        "pw_softmin": "#D55E00",
        "softmin_0.1": "#F0E442",
        "chamfer": "#56B4E9",
        "hungarian": "#009E73",
        "sinkhorn": "#CC79A7",
        "ordered": "#999999",
    }

    # 1. Training loss
    ax = axes[0, 0]
    for name, hist in all_results.items():
        ax.plot(hist["train_loss"], color=colors[name], label=name, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.set_yscale("log")

    # 2. Mean matched distance
    ax = axes[0, 1]
    for name, hist in all_results.items():
        ax.plot(
            hist["eval_epoch"],
            hist["matched_distance"],
            color=colors[name],
            label=name,
            marker=".",
            markersize=3,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Matched Distance")
    ax.set_title("Hungarian-Matched Distance (lower = better)")
    ax.legend()

    # 3. Match accuracy
    ax = axes[1, 0]
    for name, hist in all_results.items():
        ax.plot(
            hist["eval_epoch"],
            hist["match_accuracy"],
            color=colors[name],
            label=name,
            marker=".",
            markersize=3,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Match Accuracy")
    ax.set_title("Fraction of Points Matched < threshold")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)

    # 4. Duplicate rate
    ax = axes[1, 1]
    for name, hist in all_results.items():
        ax.plot(
            hist["eval_epoch"],
            hist["duplicate_rate"],
            color=colors[name],
            label=name,
            marker=".",
            markersize=3,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Duplicate Rate")
    ax.set_title("Fraction of Targets with 2+ Nearby Preds (lower = better)")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"set_matching_N{n_points}.png"
    plt.savefig(path, dpi=150)
    print(f"Saved plot to {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Toy 2D set matching experiment")
    parser.add_argument("--n_points", type=int, default=10, help="Points per set")
    parser.add_argument("--n_epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--n_train", type=int, default=2000, help="Training samples")
    parser.add_argument("--n_test", type=int, default=200, help="Test samples")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--noise_scale", type=float, default=1.0, help="Input noise")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, N={args.n_points}, epochs={args.n_epochs}")

    # Generate data
    train_targets = torch.randn(args.n_train, args.n_points, 2)
    test_targets = torch.randn(args.n_test, args.n_points, 2)

    # Define losses
    loss_fns = {
        "product_gm": ProductLoss(eps=1e-8),
        "pw_softmin": ProductWeightedSoftMinLoss(temperature=0.1),
        "softmin_0.1": SoftMinChamferLoss(temperature=0.1),
        "chamfer": ChamferLoss(),
        "hungarian": HungarianLoss(),
        "sinkhorn": SinkhornLoss(eps=0.1, n_iters=20),
    }

    # Train each
    all_results = {}
    for name, loss_fn in loss_fns.items():
        print(f"\n{'='*60}")
        print(f"Training with {name.upper()} loss")
        print(f"{'='*60}")

        history = train_one_loss(
            loss_name=name,
            loss_fn=loss_fn,
            train_targets=train_targets,
            test_targets=test_targets,
            n_points=args.n_points,
            n_epochs=args.n_epochs,
            noise_scale=args.noise_scale,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
        )
        all_results[name] = history
        print(f"  Completed in {history['total_time']:.1f}s")

    # Summary table
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"{'Loss':<12} {'Match Dist':>12} {'Match Acc':>12} {'Dup Rate':>12} {'Time (s)':>10}")
    print("-" * 60)
    for name, hist in all_results.items():
        print(
            f"{name:<12} "
            f"{hist['matched_distance'][-1]:>12.4f} "
            f"{hist['match_accuracy'][-1]:>12.3f} "
            f"{hist['duplicate_rate'][-1]:>12.3f} "
            f"{hist['total_time']:>10.1f}"
        )

    # Plot
    output_dir = Path(__file__).parent / "results"
    plot_comparison(all_results, args.n_points, output_dir)


if __name__ == "__main__":
    main()
