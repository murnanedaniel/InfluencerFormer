"""DCD comparison at N=10 and N=20, 3 seeds each.

Compares DCD against the top-performing losses from previous experiments.
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

from influencerformer.losses.set_losses import ChamferLoss, HungarianLoss, DCDLoss
from influencerformer.losses.product_loss import SoftMinChamferLoss, ProductWeightedSoftMinLoss


class SetPredictor(nn.Module):
    def __init__(self, n_points, dim=2, hidden=256):
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
    model = SetPredictor(n_points=n_points, hidden=256 if n_points >= 20 else 128).to(device)
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


def run_config(n_points, n_epochs, n_seeds, base_seed=42):
    device = torch.device("cpu")
    torch.manual_seed(base_seed)
    train_targets = torch.randn(2000, n_points, 2)
    test_targets = torch.randn(200, n_points, 2)

    methods = {
        "chamfer": lambda: ChamferLoss(),
        "hungarian": lambda: HungarianLoss(),
        "dcd_3": lambda: DCDLoss(alpha=3.0),
        "dcd_6": lambda: DCDLoss(alpha=6.0),
        "dcd_10": lambda: DCDLoss(alpha=10.0),
        "softmin": lambda: SoftMinChamferLoss(temperature=0.1),
        "pw_softmin": lambda: ProductWeightedSoftMinLoss(temperature=0.1),
    }

    all_results = {}
    for name, make_fn in methods.items():
        print(f"\n  {name.upper()}")
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


def print_table(all_results, n_points, n_epochs, n_seeds):
    print(f"\n{'='*80}")
    print(f"N={n_points}, {n_epochs} epochs, {n_seeds} seeds")
    print(f"{'='*80}")
    print(f"{'Loss':<16} {'Match Dist':>14} {'Match Acc':>14} {'Dup Rate':>14} {'Time':>10}")
    print("-" * 70)
    sorted_names = sorted(all_results,
                          key=lambda n: np.mean([r["final"]["matched_distance"] for r in all_results[n]]))
    for name in sorted_names:
        f = [r["final"] for r in all_results[name]]
        md = [x["matched_distance"] for x in f]
        ma = [x["match_accuracy"] for x in f]
        dr = [x["duplicate_rate"] for x in f]
        tt = [x["total_time"] for x in f]
        print(f"{name:<16} {np.mean(md):.4f}±{np.std(md):.4f} "
              f"{np.mean(ma):.3f}±{np.std(ma):.3f}  "
              f"{np.mean(dr):.4f}±{np.std(dr):.4f} "
              f"{np.mean(tt):>8.1f}s")


def main():
    # N=10
    print(f"\n{'#'*60}")
    print(f"  N=10, 300 epochs, 3 seeds")
    print(f"{'#'*60}")
    r10 = run_config(n_points=10, n_epochs=300, n_seeds=3)
    print_table(r10, 10, 300, 3)

    # N=20
    print(f"\n{'#'*60}")
    print(f"  N=20, 500 epochs, 3 seeds")
    print(f"{'#'*60}")
    r20 = run_config(n_points=20, n_epochs=500, n_seeds=3)
    print_table(r20, 20, 500, 3)


if __name__ == "__main__":
    main()
