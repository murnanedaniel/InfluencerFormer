"""Comprehensive test: fixed Hungarian, bigger N=50, tokens.

Three experiments:
1. N=50 with hidden=512 (bigger model to separate loss quality from capacity)
2. Token experiment with NormalizedHungarian, PowerSoftMin, baselines
3. N=10/20 with NormalizedHungarian vs regular Hungarian
"""

import sys, time, math
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from influencerformer.losses import (
    ChamferLoss, HungarianLoss, PowerSoftMinLoss,
    ProductWeightedSoftMinLoss, SoftMinChamferLoss,
)
from influencerformer.losses.set_losses import NormalizedHungarianLoss, ClampedHungarianLoss


# ── Models ────────────────────────────────────────────────────────
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


class TokenSetPredictor(nn.Module):
    """Predicts a set of discrete token vectors."""
    def __init__(self, n_points, n_tokens=4, vocab=32, hidden=256):
        super().__init__()
        self.n_points, self.n_tokens, self.vocab = n_points, n_tokens, vocab
        in_dim = n_points * n_tokens  # input: corrupted one-hot → indices
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_points * n_tokens * vocab))
    def forward(self, x):
        return self.net(x).reshape(-1, self.n_points, self.n_tokens, self.vocab)


# ── Evaluation ────────────────────────────────────────────────────
def evaluate_l2(model, test_inputs, test_targets, threshold=0.3):
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
        nc += (m < threshold).sum(); nt += len(r)
        for j in range(N):
            if (Dn[b, :, j] < threshold).sum() > 1: nd += 1
    return {"matched_distance": float(np.mean(dists)),
            "duplicate_rate": float(nd / max(B * N, 1))}


def ce_distance_matrix(logits, targets):
    """(B, M, K, V) logits × (B, N, K) targets → (B, M, N) CE distances."""
    B, M, K, V = logits.shape
    N = targets.shape[1]
    logits_exp = logits.unsqueeze(2).expand(B, M, N, K, V)
    targets_exp = targets.unsqueeze(1).expand(B, M, N, K)
    ce = F.cross_entropy(logits_exp.reshape(-1, V), targets_exp.reshape(-1), reduction='none')
    return ce.reshape(B, M, N, K).sum(-1)


def evaluate_tokens(model, test_inputs, test_targets):
    model.eval()
    with torch.no_grad():
        logits = model(test_inputs)
        D = ce_distance_matrix(logits, test_targets)
    B, M, N = D.shape
    dists = []
    Dn = D.cpu().numpy()
    for b in range(B):
        r, c = linear_sum_assignment(Dn[b])
        dists.append(Dn[b][r, c].mean())
    return {"matched_ce": float(np.mean(dists))}


# ── Training ──────────────────────────────────────────────────────
def train_l2(loss_fn, train_tgt, test_tgt, n_points, n_epochs, hidden=128, eval_every=10):
    model = SetPredictor(n_points, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(train_tgt), batch_size=64, shuffle=True)
    test_inp = (test_tgt + torch.randn_like(test_tgt)).flatten(-2)
    hist = defaultdict(list)
    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        el, nb = 0, 0
        for (tb,) in loader:
            D = torch.cdist(model((tb + torch.randn_like(tb)).flatten(-2)), tb)
            loss = loss_fn(D)
            if not (torch.isfinite(loss)): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step(); el += loss.item(); nb += 1
        hist["train_loss"].append(el / max(nb, 1))
        if (ep + 1) % eval_every == 0 or ep == 0:
            m = evaluate_l2(model, test_inp, test_tgt)
            for k in m: hist[k].append(m[k])
            hist["eval_epoch"].append(ep)
    hist["total_time"] = time.time() - t0
    return hist


def train_tokens(loss_fn, train_tgt, test_tgt, n_points, n_tokens, vocab, n_epochs, eval_every=20):
    model = TokenSetPredictor(n_points, n_tokens, vocab, hidden=256)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(train_tgt), batch_size=64, shuffle=True)
    # Corrupt by replacing random tokens
    test_inp_raw = test_tgt.clone()
    mask = torch.rand_like(test_tgt.float()) < 0.3
    test_inp_raw[mask] = torch.randint(0, vocab, (mask.sum(),))
    test_inp = test_inp_raw.float().flatten(-2)
    hist = defaultdict(list)
    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        el, nb = 0, 0
        for (tb,) in loader:
            tb_corrupt = tb.clone()
            m = torch.rand_like(tb.float()) < 0.3
            tb_corrupt[m] = torch.randint(0, vocab, (m.sum(),))
            logits = model(tb_corrupt.float().flatten(-2))
            D = ce_distance_matrix(logits, tb)
            loss = loss_fn(D)
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step(); el += loss.item(); nb += 1
        hist["train_loss"].append(el / max(nb, 1))
        if (ep + 1) % eval_every == 0 or ep == 0:
            m = evaluate_tokens(model, test_inp, test_tgt)
            for k in m: hist[k].append(m[k])
            hist["eval_epoch"].append(ep)
    hist["total_time"] = time.time() - t0
    return hist


def run_seeds(name, fn, trainer, n_seeds=3, seed=42, **kw):
    runs = []
    for s in range(n_seeds):
        torch.manual_seed(seed + s * 1000)
        h = trainer(fn(), **kw)
        # Get last eval metric (whatever it is)
        metric_keys = [k for k in h if k not in ("train_loss", "eval_epoch", "total_time")]
        f = {k: h[k][-1] for k in metric_keys}
        f["total_time"] = h["total_time"]
        runs.append(f)
        metrics_str = " ".join(f"{k}={v:.4f}" for k, v in f.items() if k != "total_time")
        print(f"  {name} seed {s}: {metrics_str} ({f['total_time']:.1f}s)")
    return runs


def summarize(name, runs, main_metric):
    vals = [r[main_metric] for r in runs]
    tt = [r["total_time"] for r in runs]
    return f"{name:<25} {np.mean(vals):.4f}±{np.std(vals):.4f}  {np.mean(tt):.1f}s"


# ── Main ──────────────────────────────────────────────────────────
def main():
    # ================================================================
    # EXPERIMENT 1: N=50 with bigger model (hidden=512)
    # ================================================================
    print("=" * 60)
    print("  EXP 1: N=50, hidden=512, 500 epochs, 2 seeds")
    print("=" * 60)

    torch.manual_seed(42)
    tr50 = torch.randn(2000, 50, 2)
    te50 = torch.randn(200, 50, 2)

    methods_50 = {
        "chamfer":     lambda: ChamferLoss(),
        "power_sm_3":  lambda: PowerSoftMinLoss(temperature=0.1, power=3.0),
        "pw_softmin":  lambda: ProductWeightedSoftMinLoss(temperature=0.1),
    }
    for name, fn in methods_50.items():
        runs = run_seeds(name, fn, lambda lf, tr=tr50, te=te50: train_l2(lf, tr, te, 50, 500, hidden=512),
                         n_seeds=2)
        dr = [r.get("duplicate_rate", 0) for r in runs]
        md = [r["matched_distance"] for r in runs]
        print(f"  → {name}: dist={np.mean(md):.4f}±{np.std(md):.4f} dup={np.mean(dr):.4f}")

    # ================================================================
    # EXPERIMENT 2: Tokens (N=10, K=4, V=32), 500 epochs
    # ================================================================
    print("\n" + "=" * 60)
    print("  EXP 2: Tokens N=10 K=4 V=32, 500 epochs, 3 seeds")
    print("=" * 60)

    torch.manual_seed(42)
    tr_tok = torch.randint(0, 32, (2000, 10, 4))
    te_tok = torch.randint(0, 32, (200, 10, 4))

    methods_tok = {
        "chamfer":        lambda: ChamferLoss(),
        "hungarian":      lambda: HungarianLoss(),
        "norm_hungarian": lambda: NormalizedHungarianLoss(),
        "clamped_hung":   lambda: ClampedHungarianLoss(max_cost=20.0),
        "power_sm_3":     lambda: PowerSoftMinLoss(temperature=0.5, power=3.0),
        "pw_softmin":     lambda: ProductWeightedSoftMinLoss(temperature=0.5),
        "softmin":        lambda: SoftMinChamferLoss(temperature=0.5),
    }
    for name, fn in methods_tok.items():
        runs = run_seeds(name, fn,
                         lambda lf, tr=tr_tok, te=te_tok: train_tokens(lf, tr, te, 10, 4, 32, 500),
                         n_seeds=3)
        mc = [r["matched_ce"] for r in runs]
        print(f"  → {name}: ce={np.mean(mc):.4f}±{np.std(mc):.4f}")

    # ================================================================
    # EXPERIMENT 3: NormalizedHungarian vs Hungarian at N=10/20 (L2)
    # ================================================================
    print("\n" + "=" * 60)
    print("  EXP 3: Normalized vs Regular Hungarian, N=20, 500 epochs")
    print("=" * 60)

    torch.manual_seed(42)
    tr20 = torch.randn(2000, 20, 2)
    te20 = torch.randn(200, 20, 2)

    methods_h = {
        "hungarian":      lambda: HungarianLoss(),
        "norm_hungarian": lambda: NormalizedHungarianLoss(),
        "power_sm_3":     lambda: PowerSoftMinLoss(temperature=0.1, power=3.0),
    }
    for name, fn in methods_h.items():
        runs = run_seeds(name, fn,
                         lambda lf, tr=tr20, te=te20: train_l2(lf, tr, te, 20, 500),
                         n_seeds=3)
        md = [r["matched_distance"] for r in runs]
        dr = [r.get("duplicate_rate", 0) for r in runs]
        print(f"  → {name}: dist={np.mean(md):.4f}±{np.std(md):.4f} dup={np.mean(dr):.4f}")

    print("\n\nAll experiments done!")


if __name__ == "__main__":
    main()
