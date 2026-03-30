"""Careful comparison: early/mid/late training dynamics and wall-clock efficiency.

For each loss, tracks:
- Loss curve at every epoch
- Matched distance at every epoch
- Duplicate rate at every epoch
- Wall-clock time at every epoch
- Time to reach threshold matched distance (0.8, 0.75, 0.72)

N=10, 500 epochs, 3 seeds. Reports both "best quality at fixed epochs"
and "best quality at fixed wall time."
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Losses ────────────────────────────────────────────────────────
class ChamferLoss(nn.Module):
    def forward(self, D):
        return D.min(1).values.mean(-1).mean() + D.min(2).values.mean(-1).mean()

class HungarianLoss(nn.Module):
    def forward(self, D):
        B = D.shape[0]
        losses = []
        Dn = D.detach().cpu().numpy()
        for b in range(B):
            r, c = linear_sum_assignment(Dn[b])
            losses.append(D[b, r, c].mean())
        return torch.stack(losses).mean()

class SinkhornLoss(nn.Module):
    def __init__(self, eps=0.1, n_iters=20):
        super().__init__()
        self.eps, self.n_iters = eps, n_iters
    def forward(self, D):
        log_K = -D / self.eps
        for _ in range(self.n_iters):
            log_K = log_K - torch.logsumexp(log_K, dim=2, keepdim=True)
            log_K = log_K - torch.logsumexp(log_K, dim=1, keepdim=True)
        return (torch.exp(log_K) * D).sum(dim=(1, 2)).mean()

class SoftMinLoss(nn.Module):
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau
    def forward(self, D):
        w1 = torch.softmax(-D / self.tau, dim=1)
        w2 = torch.softmax(-D / self.tau, dim=2)
        return ((w1 * D).sum(1).mean(-1) + (w2 * D).sum(2).mean(-1)).mean()

class PowerSoftMinLoss(nn.Module):
    def __init__(self, tau=0.1, p=3.0):
        super().__init__()
        self.tau, self.p = tau, p
    def forward(self, D):
        w1 = torch.softmax(-D / self.tau, dim=1)
        sm1 = (w1 * D).sum(1)
        w2 = torch.softmax(-D / self.tau, dim=2)
        sm2 = (w2 * D).sum(2)
        return (sm1.pow(self.p).mean(-1) + sm2.pow(self.p).mean(-1)).mean()

class ExpSoftMinLoss(nn.Module):
    def __init__(self, tau=0.1, alpha=3.0):
        super().__init__()
        self.tau, self.alpha = tau, alpha
    def forward(self, D):
        w1 = torch.softmax(-D / self.tau, dim=1)
        sm1 = (w1 * D).sum(1)
        w2 = torch.softmax(-D / self.tau, dim=2)
        sm2 = (w2 * D).sum(2)
        return (torch.exp(self.alpha * sm1).mean(-1) + torch.exp(self.alpha * sm2).mean(-1)).mean()

class PWsoftminLoss(nn.Module):
    def __init__(self, tau=0.1, eps=1e-8):
        super().__init__()
        self.tau, self.eps = tau, eps
    def forward(self, D):
        w1 = torch.softmax(-D / self.tau, dim=1)
        sm1 = (w1 * D).sum(1)
        w2 = torch.softmax(-D / self.tau, dim=2)
        sm2 = (w2 * D).sum(2)
        logD = torch.log(D + self.eps)
        cgm = torch.exp(logD.mean(1)).detach()
        rgm = torch.exp(logD.mean(2)).detach()
        cw = cgm / (cgm.mean(-1, keepdim=True) + self.eps)
        rw = rgm / (rgm.mean(-1, keepdim=True) + self.eps)
        return ((cw * sm1).mean(-1) + (rw * sm2).mean(-1)).mean()


# ── Model ─────────────────────────────────────────────────────────
class SetPredictor(nn.Module):
    def __init__(self, n_points, dim=2, hidden=128):
        super().__init__()
        self.n_points, self.dim = n_points, dim
        self.net = nn.Sequential(
            nn.Linear(n_points * dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_points * dim))
    def forward(self, x):
        return self.net(x).reshape(-1, self.n_points, self.dim)


# ── Evaluation ────────────────────────────────────────────────────
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
            if (Dn[b, :, j] < threshold).sum() > 1:
                nd += 1
    return {
        "matched_distance": float(np.mean(dists)),
        "match_accuracy": float(nc / max(nt, 1)),
        "duplicate_rate": float(nd / max(B * N, 1)),
    }


# ── Training ──────────────────────────────────────────────────────
def train_full(loss_fn, train_targets, test_targets, n_points, n_epochs,
               noise_scale=1.0, lr=1e-3, batch_size=64):
    model = SetPredictor(n_points=n_points, hidden=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(train_targets), batch_size=batch_size, shuffle=True)
    test_inp = (test_targets + noise_scale * torch.randn_like(test_targets)).flatten(-2)

    history = {
        "epoch": [], "wall_time": [],
        "train_loss": [], "matched_distance": [], "duplicate_rate": [],
        "loss_time_ms": [], "backward_time_ms": [],
    }
    t_start = time.perf_counter()

    for epoch in range(n_epochs):
        model.train()
        eloss, nb = 0.0, 0
        epoch_loss_time = 0.0
        epoch_bwd_time = 0.0

        for (tb,) in loader:
            inp = (tb + noise_scale * torch.randn_like(tb)).flatten(-2)
            preds = model(inp)
            D = torch.cdist(preds, tb)

            t_l0 = time.perf_counter()
            loss = loss_fn(D)
            t_l1 = time.perf_counter()

            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad()
            t_b0 = time.perf_counter()
            loss.backward()
            t_b1 = time.perf_counter()
            optimizer.step()

            eloss += loss.item()
            nb += 1
            epoch_loss_time += (t_l1 - t_l0)
            epoch_bwd_time += (t_b1 - t_b0)

        wall = time.perf_counter() - t_start
        m = evaluate(model, test_inp, test_targets)

        history["epoch"].append(epoch)
        history["wall_time"].append(wall)
        history["train_loss"].append(eloss / max(nb, 1))
        history["matched_distance"].append(m["matched_distance"])
        history["duplicate_rate"].append(m["duplicate_rate"])
        history["loss_time_ms"].append(1000 * epoch_loss_time / max(nb, 1))
        history["backward_time_ms"].append(1000 * epoch_bwd_time / max(nb, 1))

    return history


# ── Main ──────────────────────────────────────────────────────────
def main():
    N_POINTS = 10
    N_EPOCHS = 500
    N_SEEDS = 3
    BASE_SEED = 42

    torch.manual_seed(BASE_SEED)
    train_targets = torch.randn(2000, N_POINTS, 2)
    test_targets = torch.randn(200, N_POINTS, 2)

    losses = {
        "chamfer":      lambda: ChamferLoss(),
        "softmin":      lambda: SoftMinLoss(tau=0.1),
        "power_sm_2":   lambda: PowerSoftMinLoss(tau=0.1, p=2.0),
        "power_sm_3":   lambda: PowerSoftMinLoss(tau=0.1, p=3.0),
        "exp_sm_3":     lambda: ExpSoftMinLoss(tau=0.1, alpha=3.0),
        "pw_softmin":   lambda: PWsoftminLoss(tau=0.1),
        "hungarian":    lambda: HungarianLoss(),
        "sinkhorn":     lambda: SinkhornLoss(eps=0.1, n_iters=20),
    }

    all_results = {}

    for name, make_fn in losses.items():
        print(f"\n{'='*50}")
        print(f"  {name.upper()}")
        print(f"{'='*50}")
        seeds = []
        for s in range(N_SEEDS):
            torch.manual_seed(BASE_SEED + s * 1000)
            h = train_full(make_fn(), train_targets, test_targets, N_POINTS, N_EPOCHS)
            seeds.append(h)
            print(f"  seed {s}: final_dist={h['matched_distance'][-1]:.4f} "
                  f"dup={h['duplicate_rate'][-1]:.3f} wall={h['wall_time'][-1]:.1f}s "
                  f"loss_ms={np.mean(h['loss_time_ms']):.2f} bwd_ms={np.mean(h['backward_time_ms']):.2f}")
        all_results[name] = seeds

    # ── Analysis ──────────────────────────────────────────────────
    thresholds = [0.80, 0.75, 0.72, 0.70]

    print(f"\n{'='*90}")
    print(f"FINAL QUALITY (epoch {N_EPOCHS}, {N_SEEDS} seeds)")
    print(f"{'='*90}")
    print(f"{'Loss':<16} {'Match Dist':>12} {'Dup Rate':>10} {'Wall(s)':>8} {'Loss ms':>8} {'Bwd ms':>8}")
    print("-" * 66)
    sorted_names = sorted(all_results,
        key=lambda n: np.mean([s["matched_distance"][-1] for s in all_results[n]]))
    for name in sorted_names:
        seeds = all_results[name]
        md = [s["matched_distance"][-1] for s in seeds]
        dr = [s["duplicate_rate"][-1] for s in seeds]
        wt = [s["wall_time"][-1] for s in seeds]
        lt = [np.mean(s["loss_time_ms"]) for s in seeds]
        bt = [np.mean(s["backward_time_ms"]) for s in seeds]
        print(f"{name:<16} {np.mean(md):.4f}±{np.std(md):.4f} "
              f"{np.mean(dr):.4f}    {np.mean(wt):>6.1f}  {np.mean(lt):>6.2f}  {np.mean(bt):>6.2f}")

    print(f"\n{'='*90}")
    print(f"EPOCHS TO REACH THRESHOLD (mean over {N_SEEDS} seeds)")
    print(f"{'='*90}")
    header = f"{'Loss':<16}" + "".join(f"{'<'+str(t):>10}" for t in thresholds)
    print(header)
    print("-" * (16 + 10 * len(thresholds)))
    for name in sorted_names:
        row = f"{name:<16}"
        for thresh in thresholds:
            epochs_to = []
            for s in all_results[name]:
                found = None
                for i, d in enumerate(s["matched_distance"]):
                    if d < thresh:
                        found = i
                        break
                epochs_to.append(found if found is not None else N_EPOCHS)
            mean_ep = np.mean(epochs_to)
            if mean_ep >= N_EPOCHS:
                row += f"{'never':>10}"
            else:
                row += f"{mean_ep:>9.0f}"
        print(row)

    print(f"\n{'='*90}")
    print(f"WALL TIME TO REACH THRESHOLD (seconds, mean over {N_SEEDS} seeds)")
    print(f"{'='*90}")
    print(header)
    print("-" * (16 + 10 * len(thresholds)))
    for name in sorted_names:
        row = f"{name:<16}"
        for thresh in thresholds:
            times_to = []
            for s in all_results[name]:
                found = None
                for i, d in enumerate(s["matched_distance"]):
                    if d < thresh:
                        found = s["wall_time"][i]
                        break
                times_to.append(found if found is not None else s["wall_time"][-1])
            mean_t = np.mean(times_to)
            row += f"{mean_t:>9.1f}s"
        print(row)

    # Early/mid/late snapshot
    snapshots = [10, 50, 100, 200, 500]
    print(f"\n{'='*90}")
    print(f"MATCHED DISTANCE AT EPOCH SNAPSHOTS (mean±std, {N_SEEDS} seeds)")
    print(f"{'='*90}")
    header = f"{'Loss':<16}" + "".join(f"{'ep'+str(e):>12}" for e in snapshots)
    print(header)
    print("-" * (16 + 12 * len(snapshots)))
    for name in sorted_names:
        row = f"{name:<16}"
        for ep in snapshots:
            idx = min(ep - 1, N_EPOCHS - 1)
            vals = [s["matched_distance"][idx] for s in all_results[name]]
            row += f" {np.mean(vals):.4f}±{np.std(vals):.4f}"
        print(row)


if __name__ == "__main__":
    main()
