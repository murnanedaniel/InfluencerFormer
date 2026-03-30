"""Power-SoftMin and new variants comparison.

Tests the key new idea: raising softmin to a power p > 1 creates
coverage enforcement through the gradient (∂(sm^p)/∂D ∝ p·sm^{p-1})
without needing detached GM reweighting.

Usage:
    python examples/run_power_comparison.py
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
    ClampedHungarianLoss,
    HungarianLoss,
    LogChamferLoss,
    LogProductSoftMinLoss,
    PowerSoftMinLoss,
    ProductWeightedSoftMinLoss,
    SinkhornLoss,
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
              noise_scale=1.0, lr=1e-3, batch_size=64, eval_every=10):
    device = torch.device("cpu")
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
            inp = (tb + noise_scale * torch.randn_like(tb)).flatten(-2)
            D = torch.cdist(model(inp), tb)
            loss = loss_fn(D)
            if torch.isnan(loss) or torch.isinf(loss):
                continue  # skip bad batches
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            eloss += loss.item()
            nb += 1
        history["train_loss"].append(eloss / max(nb, 1))
        if (epoch + 1) % eval_every == 0 or epoch == 0:
            m = evaluate(model, test_inp, test_tgt)
            for k in m: history[k].append(m[k])
            history["eval_epoch"].append(epoch)
    history["total_time"] = time.time() - t0
    return history


def run_seeds(name, make_fn, n_points, n_epochs, n_seeds=3, base_seed=42):
    torch.manual_seed(base_seed)
    train = torch.randn(2000, n_points, 2)
    test = torch.randn(200, n_points, 2)
    runs = []
    for s in range(n_seeds):
        torch.manual_seed(base_seed + s * 1000)
        h = train_one(make_fn(), train, test, n_points, n_epochs)
        f = {k: h[k][-1] for k in ["matched_distance", "match_accuracy", "duplicate_rate"]}
        f["total_time"] = h["total_time"]
        runs.append(f)
        print(f"    {name} seed {s}: dist={f['matched_distance']:.4f} "
              f"dup={f['duplicate_rate']:.3f} ({f['total_time']:.1f}s)")
    return runs


def summarize(name, runs):
    md = [r["matched_distance"] for r in runs]
    dr = [r["duplicate_rate"] for r in runs]
    tt = [r["total_time"] for r in runs]
    return (f"{name:<28} {np.mean(md):.4f}±{np.std(md):.4f}  "
            f"dup={np.mean(dr):.4f}±{np.std(dr):.4f}  {np.mean(tt):.1f}s")


def main():
    methods = {
        # Baselines
        "chamfer":             lambda: ChamferLoss(),
        "hungarian":           lambda: HungarianLoss(),
        "sinkhorn":            lambda: SinkhornLoss(eps=0.1, n_iters=20),
        # Previous best
        "softmin":             lambda: SoftMinChamferLoss(temperature=0.1),
        "pw_softmin":          lambda: ProductWeightedSoftMinLoss(temperature=0.1),
        # New: Power-SoftMin
        "power_sm_1.5":        lambda: PowerSoftMinLoss(temperature=0.1, power=1.5),
        "power_sm_2.0":        lambda: PowerSoftMinLoss(temperature=0.1, power=2.0),
        "power_sm_3.0":        lambda: PowerSoftMinLoss(temperature=0.1, power=3.0),
        # New: Log variants
        "log_prod_softmin":    lambda: LogProductSoftMinLoss(temperature=0.1),
        "log_chamfer":         lambda: LogChamferLoss(),
        # Fix: Clamped Hungarian
        "clamped_hungarian":   lambda: ClampedHungarianLoss(max_cost=20.0),
    }

    # ---- N=10 ----
    print(f"\n{'#'*60}")
    print(f"  N=10, 300 epochs, 3 seeds")
    print(f"{'#'*60}")

    r10 = {}
    for name, fn in methods.items():
        r10[name] = run_seeds(name, fn, n_points=10, n_epochs=300)

    print(f"\n{'N=10 Results':}")
    print("-" * 75)
    for name in sorted(r10, key=lambda n: np.mean([r["matched_distance"] for r in r10[n]])):
        print(f"  {summarize(name, r10[name])}")

    # ---- N=20 ----
    print(f"\n{'#'*60}")
    print(f"  N=20, 500 epochs, 3 seeds")
    print(f"{'#'*60}")

    r20 = {}
    for name, fn in methods.items():
        r20[name] = run_seeds(name, fn, n_points=20, n_epochs=500)

    print(f"\n{'N=20 Results':}")
    print("-" * 75)
    for name in sorted(r20, key=lambda n: np.mean([r["matched_distance"] for r in r20[n]])):
        print(f"  {summarize(name, r20[name])}")

    print("\n\nDone!")


if __name__ == "__main__":
    main()
