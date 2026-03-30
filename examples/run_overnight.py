"""Comprehensive overnight comparison of all set prediction losses.

Runs N=10 (300 epochs) and N=20 (500 epochs), 3 seeds each.
Only includes losses that showed promise or are important baselines.

Usage:
    python examples/run_overnight.py
"""

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

from influencerformer.losses import (
    ChamferLoss,
    CombinedSoftMinLoss,
    DCDLoss,
    HungarianLoss,
    ProductWeightedSoftMinLoss,
    SinkhornLoss,
    SoftDCDLoss,
    SoftMinChamferLoss,
)


class SetPredictor(nn.Module):
    def __init__(self, n_points, dim=2, hidden=128):
        super().__init__()
        self.n_points, self.dim = n_points, dim
        self.net = nn.Sequential(
            nn.Linear(n_points * dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_points * dim),
        )
    def forward(self, x):
        return self.net(x).reshape(-1, self.n_points, self.dim)


def evaluate(model, test_inputs, test_targets, threshold=0.3):
    model.eval()
    with torch.no_grad():
        preds = model(test_inputs)
        D = torch.cdist(preds, test_targets)
    B, M, N = D.shape
    matched_dists, n_correct, n_total, n_dup = [], 0, 0, 0
    D_np = D.cpu().numpy()
    for b in range(B):
        row, col = linear_sum_assignment(D_np[b])
        m = D_np[b][row, col]
        matched_dists.append(m.mean())
        n_correct += (m < threshold).sum()
        n_total += len(row)
        for j in range(N):
            if (D_np[b, :, j] < threshold).sum() > 1:
                n_dup += 1
    return {
        "matched_distance": float(np.mean(matched_dists)),
        "match_accuracy": float(n_correct / max(n_total, 1)),
        "duplicate_rate": float(n_dup / max(B * N, 1)),
    }


def train_one(loss_fn, train_targets, test_targets, n_points, n_epochs,
              noise_scale, lr, batch_size, device, eval_every=10):
    model = SetPredictor(n_points=n_points, hidden=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(train_targets), batch_size=batch_size, shuffle=True)
    test_inp = (test_targets + noise_scale * torch.randn_like(test_targets)).flatten(-2).to(device)
    test_tgt = test_targets.to(device)
    history = defaultdict(list)
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        eloss, nb = 0.0, 0
        for (tb,) in loader:
            tb = tb.to(device)
            inp = (tb + noise_scale * torch.randn_like(tb)).flatten(-2)
            D = torch.cdist(model(inp), tb)
            loss = loss_fn(D)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            eloss += loss.item()
            nb += 1
        history["train_loss"].append(eloss / nb)
        if (epoch + 1) % eval_every == 0 or epoch == 0:
            m = evaluate(model, test_inp, test_tgt)
            history["matched_distance"].append(m["matched_distance"])
            history["match_accuracy"].append(m["match_accuracy"])
            history["duplicate_rate"].append(m["duplicate_rate"])
            history["eval_epoch"].append(epoch)
    history["total_time"] = time.time() - t0
    return history


def run_config(n_points, n_epochs, n_seeds, methods, base_seed=42):
    device = torch.device("cpu")
    torch.manual_seed(base_seed)
    train_targets = torch.randn(2000, n_points, 2)
    test_targets = torch.randn(200, n_points, 2)

    all_results = {}
    for name, make_fn in methods.items():
        print(f"\n  {name}")
        runs = []
        for s in range(n_seeds):
            torch.manual_seed(base_seed + s * 1000)
            hist = train_one(make_fn(), train_targets, test_targets, n_points, n_epochs,
                             1.0, 1e-3, 64, device)
            final = {k: hist[k][-1] for k in ["matched_distance", "match_accuracy", "duplicate_rate"]}
            final["total_time"] = hist["total_time"]
            runs.append({"history": dict(hist), "final": final})
            print(f"    seed {s}: dist={final['matched_distance']:.4f} "
                  f"acc={final['match_accuracy']:.3f} dup={final['duplicate_rate']:.3f} "
                  f"({final['total_time']:.1f}s)")
        all_results[name] = runs
    return all_results


def print_table(all_results, title):
    print(f"\n{'='*85}")
    print(f"  {title}")
    print(f"{'='*85}")
    print(f"{'Loss':<20} {'Match Dist':>14} {'Match Acc':>12} {'Dup Rate':>14} {'Time':>10}")
    print("-" * 75)
    sorted_names = sorted(all_results,
                          key=lambda n: np.mean([r["final"]["matched_distance"] for r in all_results[n]]))
    for name in sorted_names:
        f = [r["final"] for r in all_results[name]]
        md = [x["matched_distance"] for x in f]
        ma = [x["match_accuracy"] for x in f]
        dr = [x["duplicate_rate"] for x in f]
        tt = [x["total_time"] for x in f]
        print(f"{name:<20} {np.mean(md):.4f}±{np.std(md):.4f}  "
              f"{np.mean(ma):.3f}±{np.std(ma):.3f}  "
              f"{np.mean(dr):.4f}±{np.std(dr):.4f}  "
              f"{np.mean(tt):>7.1f}s")
    return sorted_names


def plot_results(all_results, n_points, n_seeds, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f"Set Prediction Loss Comparison (N={n_points}, {n_seeds} seeds)", fontsize=14)
    n = len(all_results)
    cmap = plt.cm.tab10
    colors = {name: cmap(i / max(n - 1, 1)) for i, name in enumerate(all_results)}
    sorted_names = sorted(all_results,
                          key=lambda n: np.mean([r["final"]["matched_distance"] for r in all_results[n]]))

    for ax_idx, (metric, title) in enumerate([
        ("train_loss", "Training Loss (log)"),
        ("matched_distance", "Matched Distance (lower=better)"),
        ("match_accuracy", "Match Accuracy (higher=better)"),
        ("duplicate_rate", "Duplicate Rate (lower=better)"),
    ]):
        ax = axes[ax_idx // 2, ax_idx % 2]
        for name in sorted_names:
            runs = all_results[name]
            if metric == "train_loss":
                curves = np.array([r["history"]["train_loss"] for r in runs])
                x = range(len(curves[0]))
            else:
                curves = np.array([r["history"][metric] for r in runs])
                x = runs[0]["history"]["eval_epoch"]
            m, s = curves.mean(0), curves.std(0)
            ax.plot(x, m, color=colors[name], label=name, linewidth=1.5)
            ax.fill_between(x, m - s, m + s, color=colors[name], alpha=0.12)
        ax.set_xlabel("Epoch"); ax.set_title(title)
        if "loss" in metric.lower(): ax.set_yscale("log")
        if "accuracy" in metric: ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    out = output_dir / f"overnight_N{n_points}.png"
    plt.savefig(out, dpi=150); print(f"Saved: {out}"); plt.close()


def main():
    N_SEEDS = 3
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    methods = {
        # --- Baselines ---
        "chamfer":         lambda: ChamferLoss(),
        "hungarian":       lambda: HungarianLoss(),
        "sinkhorn":        lambda: SinkhornLoss(eps=0.1, n_iters=20),
        # --- SoftMin family ---
        "softmin_0.1":     lambda: SoftMinChamferLoss(temperature=0.1),
        "pw_softmin":      lambda: ProductWeightedSoftMinLoss(temperature=0.1),
        # --- New variants ---
        "soft_dcd":        lambda: SoftDCDLoss(temperature=0.1),
        "combined":        lambda: CombinedSoftMinLoss(temperature=0.1, gm_weight=0.5, freq_weight=0.5),
        # --- DCD baseline (properly scaled alpha) ---
        "dcd_6":           lambda: DCDLoss(alpha=6.0),
    }

    # ---- N=10 ----
    print(f"\n{'#'*60}")
    print(f"  N=10, 300 epochs, {N_SEEDS} seeds")
    print(f"{'#'*60}")
    r10 = run_config(10, 300, N_SEEDS, methods)
    print_table(r10, "N=10, 300 epochs")
    plot_results(r10, 10, N_SEEDS, output_dir)

    # ---- N=20 ----
    print(f"\n{'#'*60}")
    print(f"  N=20, 500 epochs, {N_SEEDS} seeds")
    print(f"{'#'*60}")
    r20 = run_config(20, 500, N_SEEDS, methods)
    print_table(r20, "N=20, 500 epochs")
    plot_results(r20, 20, N_SEEDS, output_dir)

    print("\n\nDone! All results saved.")


if __name__ == "__main__":
    main()
