"""Test the Permanent Loss: partition function over all matchings.

Compares F = -T·log(perm(exp(-D/T))) against best losses at N=10 and N=20.
"""

import sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from influencerformer.losses import ChamferLoss, HungarianLoss, SinkhornLoss, PowerSoftMinLoss, ProductWeightedSoftMinLoss
from influencerformer.losses.permanent_loss import PermanentLoss


class SetPredictor(nn.Module):
    def __init__(self, n_points, dim=2, hidden=128):
        super().__init__()
        self.n_points, self.dim = n_points, dim
        self.net = nn.Sequential(nn.Linear(n_points*dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_points*dim))
    def forward(self, x):
        return self.net(x).reshape(-1, self.n_points, self.dim)


def evaluate(model, test_inputs, test_targets, threshold=0.3):
    model.eval()
    with torch.no_grad():
        preds = model(test_inputs)
        D = torch.cdist(preds, test_targets)
    B, M, N = D.shape
    dists, nc, nt, nd = [], 0, 0, 0
    Dn = D.cpu().numpy()
    for b in range(B):
        r, c = linear_sum_assignment(Dn[b])
        m = Dn[b][r, c]
        dists.append(m.mean())
        nc += (m < threshold).sum()
        nt += len(r)
        for j in range(N):
            if (Dn[b, :, j] < threshold).sum() > 1: nd += 1
    return {"matched_distance": float(np.mean(dists)),
            "match_accuracy": float(nc / max(nt, 1)),
            "duplicate_rate": float(nd / max(B * N, 1))}


def train_one(loss_fn, train_tgt, test_tgt, n_points, n_epochs, eval_every=10):
    model = SetPredictor(n_points=n_points, hidden=128)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(train_tgt), batch_size=64, shuffle=True)
    test_inp = (test_tgt + torch.randn_like(test_tgt)).flatten(-2)
    hist = defaultdict(list)
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        el, nb = 0, 0
        for (tb,) in loader:
            inp = (tb + torch.randn_like(tb)).flatten(-2)
            D = torch.cdist(model(inp), tb)
            loss = loss_fn(D)
            if torch.isnan(loss) or torch.isinf(loss): continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            el += loss.item(); nb += 1
        hist["train_loss"].append(el / max(nb, 1))
        if (epoch + 1) % eval_every == 0 or epoch == 0:
            m = evaluate(model, test_inp, test_tgt)
            for k in m: hist[k].append(m[k])
            hist["eval_epoch"].append(epoch)
    hist["total_time"] = time.time() - t0
    return hist


def run(name, fn, n_points, n_epochs, n_seeds=3, seed=42):
    torch.manual_seed(seed)
    tr = torch.randn(2000, n_points, 2)
    te = torch.randn(200, n_points, 2)
    runs = []
    for s in range(n_seeds):
        torch.manual_seed(seed + s * 1000)
        h = train_one(fn(), tr, te, n_points, n_epochs)
        f = {k: h[k][-1] for k in ["matched_distance", "match_accuracy", "duplicate_rate"]}
        f["total_time"] = h["total_time"]
        runs.append(f)
        print(f"  {name} seed {s}: dist={f['matched_distance']:.4f} dup={f['duplicate_rate']:.3f} ({f['total_time']:.1f}s)")
    md = [r["matched_distance"] for r in runs]
    dr = [r["duplicate_rate"] for r in runs]
    tt = [r["total_time"] for r in runs]
    print(f"  → {name}: {np.mean(md):.4f}±{np.std(md):.4f}  dup={np.mean(dr):.4f}  {np.mean(tt):.1f}s")
    return runs


def main():
    methods = {
        "chamfer":      lambda: ChamferLoss(),
        "hungarian":    lambda: HungarianLoss(),
        "sinkhorn":     lambda: SinkhornLoss(eps=0.1, n_iters=20),
        "pw_softmin":   lambda: ProductWeightedSoftMinLoss(temperature=0.1),
        "power_sm_3":   lambda: PowerSoftMinLoss(temperature=0.1, power=3.0),
        "permanent_0.5": lambda: PermanentLoss(temperature=0.5),
        "permanent_1.0": lambda: PermanentLoss(temperature=1.0),
        "permanent_2.0": lambda: PermanentLoss(temperature=2.0),
    }

    print("="*60)
    print("  N=10, 300 epochs, 3 seeds")
    print("="*60)
    for name, fn in methods.items():
        run(name, fn, 10, 300)

    # Only run N=15 for permanent (N=20 would need 2^20 subsets = slow)
    print("\n" + "="*60)
    print("  N=15, 400 epochs, 2 seeds (permanent feasible)")
    print("="*60)
    methods_15 = {k: v for k, v in methods.items() if k != "sinkhorn"}  # skip slow sinkhorn
    for name, fn in methods_15.items():
        run(name, fn, 15, 400, n_seeds=2)


if __name__ == "__main__":
    main()
