"""Ablations on PW-SoftMin: temperature sweep, N scaling, gentle variants.

Usage:
    python examples/run_ablations.py
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
    HungarianLoss,
    ProductWeightedSoftMinLoss,
    SoftMinChamferLoss,
)
from influencerformer.losses.product_loss import SoftDCDLoss, CombinedSoftMinLoss


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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            eloss += loss.item()
            nb += 1
        history["train_loss"].append(eloss / nb)
        if (epoch + 1) % eval_every == 0 or epoch == 0:
            m = evaluate(model, test_inp, test_tgt)
            for k in m:
                history[k].append(m[k])
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
    return f"{name:<28} {np.mean(md):.4f}±{np.std(md):.4f}  dup={np.mean(dr):.4f}±{np.std(dr):.4f}  {np.mean(tt):.1f}s"


def main():
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # =====================================================================
    # ABLATION 1: Temperature sweep for PW-SoftMin at N=10
    # =====================================================================
    print("\n" + "="*60)
    print("  ABLATION 1: Temperature sweep (N=10, 300 epochs)")
    print("="*60)

    temps = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    results_temp = {}
    for tau in temps:
        name = f"pw_softmin_τ={tau}"
        runs = run_seeds(name, lambda t=tau: ProductWeightedSoftMinLoss(temperature=t),
                         n_points=10, n_epochs=300)
        results_temp[name] = runs

    # Add baselines
    for bname, bfn in [("chamfer", ChamferLoss), ("hungarian", HungarianLoss)]:
        runs = run_seeds(bname, bfn, n_points=10, n_epochs=300)
        results_temp[bname] = runs

    print(f"\n{'Temperature Sweep Results (N=10)':}")
    print("-"*70)
    for name in sorted(results_temp, key=lambda n: np.mean([r["matched_distance"] for r in results_temp[n]])):
        print(f"  {summarize(name, results_temp[name])}")

    # =====================================================================
    # ABLATION 2: Gentle claim-count weighting
    # =====================================================================
    print("\n" + "="*60)
    print("  ABLATION 2: Gentle claim-count weighting (N=20, 500 epochs)")
    print("="*60)

    # The soft_dcd over-corrected. Try gentler: mix 90% uniform + 10% inverse-freq
    class GentleSoftDCD(nn.Module):
        def __init__(self, temperature=0.1, strength=0.1, eps=1e-8):
            super().__init__()
            self.temperature, self.strength, self.eps = temperature, strength, eps

        def forward(self, D):
            w_cov = torch.softmax(-D / self.temperature, dim=1)
            sm_cov = (w_cov * D).sum(1)  # (B, N)

            claims = torch.softmax(-D / self.temperature, dim=2).sum(1)  # (B, N)
            inv_freq = (1.0 / (claims + self.eps)).detach()
            inv_norm = inv_freq / (inv_freq.mean(-1, keepdim=True) + self.eps)
            # Blend: (1-s)*uniform + s*inverse_freq
            target_w = (1 - self.strength) + self.strength * inv_norm
            target_w = target_w / (target_w.mean(-1, keepdim=True) + self.eps)

            cov = (target_w * sm_cov).mean(-1)

            w_prec = torch.softmax(-D / self.temperature, dim=2)
            sm_prec = (w_prec * D).sum(2)
            claims_p = torch.softmax(-D / self.temperature, dim=1).sum(2)
            inv_p = (1.0 / (claims_p + self.eps)).detach()
            inv_p_n = inv_p / (inv_p.mean(-1, keepdim=True) + self.eps)
            pred_w = (1 - self.strength) + self.strength * inv_p_n
            pred_w = pred_w / (pred_w.mean(-1, keepdim=True) + self.eps)
            prec = (pred_w * sm_prec).mean(-1)

            return (cov + prec).mean()

    gentle_configs = {
        "chamfer":           lambda: ChamferLoss(),
        "hungarian":         lambda: HungarianLoss(),
        "softmin_0.1":       lambda: SoftMinChamferLoss(temperature=0.1),
        "pw_softmin":        lambda: ProductWeightedSoftMinLoss(temperature=0.1),
        "gentle_dcd_0.1":    lambda: GentleSoftDCD(strength=0.1),
        "gentle_dcd_0.3":    lambda: GentleSoftDCD(strength=0.3),
        "gentle_dcd_0.5":    lambda: GentleSoftDCD(strength=0.5),
    }

    results_gentle = {}
    for name, fn in gentle_configs.items():
        runs = run_seeds(name, fn, n_points=20, n_epochs=500)
        results_gentle[name] = runs

    print(f"\n{'Gentle Claim-Count Results (N=20)':}")
    print("-"*70)
    for name in sorted(results_gentle, key=lambda n: np.mean([r["matched_distance"] for r in results_gentle[n]])):
        print(f"  {summarize(name, results_gentle[name])}")

    # =====================================================================
    # ABLATION 3: N=50 scaling test
    # =====================================================================
    print("\n" + "="*60)
    print("  ABLATION 3: N=50 scaling (500 epochs, 2 seeds)")
    print("="*60)

    scale_configs = {
        "chamfer":     lambda: ChamferLoss(),
        "softmin_0.1": lambda: SoftMinChamferLoss(temperature=0.1),
        "pw_softmin":  lambda: ProductWeightedSoftMinLoss(temperature=0.1),
    }

    results_scale = {}
    for name, fn in scale_configs.items():
        runs = run_seeds(name, fn, n_points=50, n_epochs=500, n_seeds=2)
        results_scale[name] = runs

    print(f"\n{'N=50 Scaling Results':}")
    print("-"*70)
    for name in sorted(results_scale, key=lambda n: np.mean([r["matched_distance"] for r in results_scale[n]])):
        print(f"  {summarize(name, results_scale[name])}")

    print("\n\nAll ablations done!")


if __name__ == "__main__":
    main()
