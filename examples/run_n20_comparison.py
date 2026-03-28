"""N=20 comparison of best-performing losses, sequential for fair timing.

Usage:
    python examples/run_n20_comparison.py
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


class SetPredictor(nn.Module):
    def __init__(self, n_points, dim=2, hidden=256):
        super().__init__()
        self.n_points = n_points
        self.dim = dim
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


# --- Loss functions ---
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

class SoftMinChamfer(nn.Module):
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau
    def forward(self, D):
        w1 = torch.softmax(-D / self.tau, dim=1)
        w2 = torch.softmax(-D / self.tau, dim=2)
        return ((w1 * D).sum(1).mean(-1) + (w2 * D).sum(2).mean(-1)).mean()

class PWsoftmin(nn.Module):
    def __init__(self, tau=0.1, eps=1e-8):
        super().__init__()
        self.tau, self.eps = tau, eps
    def forward(self, D):
        w1 = torch.softmax(-D / self.tau, dim=1)
        sm_cov = (w1 * D).sum(1)
        w2 = torch.softmax(-D / self.tau, dim=2)
        sm_prec = (w2 * D).sum(2)
        logD = torch.log(D + self.eps)
        cgm = torch.exp(logD.mean(1)).detach()
        rgm = torch.exp(logD.mean(2)).detach()
        cw = cgm / (cgm.mean(-1, keepdim=True) + self.eps)
        rw = rgm / (rgm.mean(-1, keepdim=True) + self.eps)
        return ((cw * sm_cov).mean(-1) + (rw * sm_prec).mean(-1)).mean()

class SigmoidProduct(nn.Module):
    def __init__(self, margin=1.5, scale=5.0, eps=1e-8):
        super().__init__()
        self.margin, self.scale, self.eps = margin, scale, eps
    def forward(self, D):
        sig = torch.sigmoid(self.scale * (D - self.margin))
        logs = torch.log(sig + self.eps)
        cov = torch.exp(logs.mean(1)).mean(-1)
        prec = torch.exp(logs.mean(2)).mean(-1)
        return (cov + prec).mean()


def train_one(loss_fn, train_targets, test_targets, n_points, n_epochs,
              noise_scale, lr, batch_size, device, eval_every=10):
    model = SetPredictor(n_points=n_points).to(device)
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


def main():
    N_POINTS = 20
    N_EPOCHS = 500
    N_SEEDS = 5
    BASE_SEED = 42
    device = torch.device("cpu")

    # Fixed data
    torch.manual_seed(BASE_SEED)
    train_targets = torch.randn(2000, N_POINTS, 2)
    test_targets = torch.randn(200, N_POINTS, 2)

    methods = {
        "chamfer": lambda: ChamferLoss(),
        "hungarian": lambda: HungarianLoss(),
        "sinkhorn": lambda: SinkhornLoss(eps=0.1, n_iters=20),
        "softmin": lambda: SoftMinChamfer(tau=0.1),
        "pw_softmin": lambda: PWsoftmin(tau=0.1),
        "sigmoid_prod": lambda: SigmoidProduct(margin=1.5, scale=5.0),
    }

    print(f"N={N_POINTS}, epochs={N_EPOCHS}, seeds={N_SEEDS}")
    all_results = {}

    for name, make_fn in methods.items():
        print(f"\n{'='*50}\n  {name.upper()}\n{'='*50}")
        runs = []
        for s in range(N_SEEDS):
            torch.manual_seed(BASE_SEED + s * 1000)
            loss_fn = make_fn()
            hist = train_one(loss_fn, train_targets, test_targets, N_POINTS, N_EPOCHS,
                             1.0, 1e-3, 64, device)
            final = {k: hist[k][-1] for k in ["matched_distance", "match_accuracy", "duplicate_rate"]}
            final["total_time"] = hist["total_time"]
            runs.append({"history": dict(hist), "final": final})
            print(f"  seed {s}: dist={final['matched_distance']:.4f} "
                  f"acc={final['match_accuracy']:.3f} "
                  f"dup={final['duplicate_rate']:.3f} "
                  f"({final['total_time']:.1f}s)")
        all_results[name] = runs

    # Summary
    print(f"\n{'='*80}")
    print(f"RESULTS: N={N_POINTS}, {N_EPOCHS} epochs, {N_SEEDS} seeds")
    print(f"{'='*80}")
    print(f"{'Loss':<18} {'Match Dist':>14} {'Match Acc':>14} {'Dup Rate':>14} {'Time':>10}")
    print("-" * 72)
    sorted_names = sorted(all_results,
                          key=lambda n: np.mean([r["final"]["matched_distance"] for r in all_results[n]]))
    for name in sorted_names:
        f = [r["final"] for r in all_results[name]]
        md = [x["matched_distance"] for x in f]
        ma = [x["match_accuracy"] for x in f]
        dr = [x["duplicate_rate"] for x in f]
        tt = [x["total_time"] for x in f]
        print(f"{name:<18} {np.mean(md):.4f}±{np.std(md):.4f} "
              f"{np.mean(ma):.3f}±{np.std(ma):.3f}  "
              f"{np.mean(dr):.4f}±{np.std(dr):.4f} "
              f"{np.mean(tt):>8.1f}s")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Set Prediction Loss Comparison (N={N_POINTS}, {N_SEEDS} seeds)", fontsize=14)
    colors = {"chamfer": "#56B4E9", "hungarian": "#009E73", "sinkhorn": "#CC79A7",
              "softmin": "#F0E442", "pw_softmin": "#D55E00", "sigmoid_prod": "#E69F00"}
    for ax_idx, (metric, title) in enumerate([
        ("train_loss", "Training Loss"), ("matched_distance", "Matched Distance"),
        ("match_accuracy", "Match Accuracy"), ("duplicate_rate", "Duplicate Rate"),
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
            ax.plot(x, m, color=colors.get(name, "gray"), label=name, linewidth=1.5)
            ax.fill_between(x, m - s, m + s, color=colors.get(name, "gray"), alpha=0.15)
        ax.set_xlabel("Epoch"); ax.set_title(title); ax.legend(fontsize=8)
        if "loss" in metric.lower(): ax.set_yscale("log")
        if "accuracy" in metric: ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    out = Path(__file__).parent / "results" / f"set_matching_N{N_POINTS}.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150); print(f"\nSaved: {out}"); plt.close()


if __name__ == "__main__":
    main()
