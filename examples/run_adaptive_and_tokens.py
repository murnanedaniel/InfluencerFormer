"""Adaptive temperature PW-SoftMin + discrete token experiment.

Ablation 4: Adaptive τ = median(D) * scale_factor
Ablation 5: Discrete token set prediction with CE distances

Usage:
    python examples/run_adaptive_and_tokens.py
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from influencerformer.losses import (
    ChamferLoss,
    HungarianLoss,
    ProductWeightedSoftMinLoss,
    SoftMinChamferLoss,
)


# =========================================================================
# Adaptive temperature variants
# =========================================================================

class AdaptivePWSoftMin(nn.Module):
    """PW-SoftMin with data-dependent temperature.

    τ = scale * median(D), so the sharpness automatically adjusts to the
    distance scale. Early in training (large D), τ is large (soft).
    At convergence (small D for matched pairs), τ is small (sharp).
    """

    def __init__(self, scale: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        tau = self.scale * D.median().detach().clamp(min=0.01)

        w_cov = torch.softmax(-D / tau, dim=1)
        sm_cov = (w_cov * D).sum(1)
        w_prec = torch.softmax(-D / tau, dim=2)
        sm_prec = (w_prec * D).sum(2)

        log_D = torch.log(D + self.eps)
        cgm = torch.exp(log_D.mean(1)).detach()
        rgm = torch.exp(log_D.mean(2)).detach()
        cw = cgm / (cgm.mean(-1, keepdim=True) + self.eps)
        rw = rgm / (rgm.mean(-1, keepdim=True) + self.eps)

        return ((cw * sm_cov).mean(-1) + (rw * sm_prec).mean(-1)).mean()


# =========================================================================
# Token-based experiment
# =========================================================================

class TokenSetPredictor(nn.Module):
    """Predicts a set of K-dimensional discrete tokens.

    Each "particle" has K token slots, each from vocab V.
    Output: logits (B, N, K, V).
    """

    def __init__(self, n_points, n_tokens=4, vocab_size=32, hidden=128):
        super().__init__()
        self.n_points = n_points
        self.n_tokens = n_tokens
        self.vocab_size = vocab_size

        # Encode input tokens to continuous
        self.embed = nn.Embedding(vocab_size, 16)
        in_dim = n_points * n_tokens * 16
        out_dim = n_points * n_tokens * vocab_size

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        # x: (B, N, K) integer tokens
        emb = self.embed(x).flatten(-2).flatten(-2)  # (B, N*K*16)
        logits = self.net(emb)
        return logits.reshape(-1, self.n_points, self.n_tokens, self.vocab_size)


def compute_ce_distance_matrix(logits, targets):
    """Compute pairwise CE distance between predictions and targets.

    logits: (B, M, K, V) — predicted logits
    targets: (B, N, K) — target token indices

    Returns: (B, M, N) where D[b,i,j] = Σ_k CE(logits[b,i,k], targets[b,j,k])
    """
    B, M, K, V = logits.shape
    N = targets.shape[1]

    # Expand for pairwise: (B, M, 1, K, V) vs (B, 1, N, K)
    logits_exp = logits.unsqueeze(2).expand(B, M, N, K, V)
    targets_exp = targets.unsqueeze(1).expand(B, M, N, K)

    # Flatten for CE computation
    logits_flat = logits_exp.reshape(-1, V)
    targets_flat = targets_exp.reshape(-1)

    ce = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    D = ce.reshape(B, M, N, K).sum(dim=-1)  # (B, M, N)
    return D


def generate_token_data(n_samples, n_points, n_tokens=4, vocab_size=32):
    """Generate random discrete token sets."""
    return torch.randint(0, vocab_size, (n_samples, n_points, n_tokens))


def evaluate_tokens(model, test_inputs, test_targets, n_tokens, vocab_size):
    """Evaluate token prediction with Hungarian matching."""
    model.eval()
    with torch.no_grad():
        logits = model(test_inputs)
        D = compute_ce_distance_matrix(logits, test_targets)

    B, M, N = D.shape
    D_np = D.cpu().numpy()
    matched_dists, n_correct_tokens, n_total_tokens = [], 0, 0

    for b in range(B):
        row, col = linear_sum_assignment(D_np[b])
        matched_dists.append(D_np[b][row, col].mean())
        # Token accuracy on matched pairs
        pred_tokens = logits[b, row].argmax(dim=-1)  # (N, K)
        true_tokens = test_targets[b, col]  # (N, K)
        n_correct_tokens += (pred_tokens == true_tokens).sum().item()
        n_total_tokens += true_tokens.numel()

    return {
        "matched_ce": float(np.mean(matched_dists)),
        "token_accuracy": float(n_correct_tokens / max(n_total_tokens, 1)),
    }


# =========================================================================
# Training
# =========================================================================

def train_points(loss_fn, train_tgt, test_tgt, n_points, n_epochs):
    """Train 2D point set prediction (same as before)."""
    from influencerformer.losses.set_losses import ChamferLoss

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_points * 2, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, n_points * 2),
            )
        def forward(self, x):
            return self.net(x).reshape(-1, n_points, 2)

    model = M()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(train_tgt), batch_size=64, shuffle=True)
    test_inp = (test_tgt + torch.randn_like(test_tgt)).flatten(-2)
    history = defaultdict(list)
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        el, nb = 0, 0
        for (tb,) in loader:
            inp = (tb + torch.randn_like(tb)).flatten(-2)
            D = torch.cdist(model(inp), tb)
            loss = loss_fn(D)
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); nb += 1
        history["train_loss"].append(el / nb)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                preds = model(test_inp)
                D = torch.cdist(preds, test_tgt)
            D_np = D.numpy()
            md = []
            for b in range(D.shape[0]):
                r, c = linear_sum_assignment(D_np[b])
                md.append(D_np[b][r, c].mean())
            history["matched_distance"].append(float(np.mean(md)))
            history["eval_epoch"].append(epoch)

    history["total_time"] = time.time() - t0
    return history


def train_tokens(loss_name, loss_fn_maker, train_tgt, test_tgt, n_points, n_epochs,
                 n_tokens=4, vocab_size=32):
    """Train discrete token set prediction."""
    model = TokenSetPredictor(n_points, n_tokens, vocab_size, hidden=128)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Corrupt input by replacing 30% of tokens
    def corrupt(x):
        mask = torch.rand_like(x.float()) < 0.3
        noise = torch.randint(0, vocab_size, x.shape)
        return torch.where(mask, noise, x)

    loader = DataLoader(TensorDataset(train_tgt), batch_size=64, shuffle=True)
    test_inp = corrupt(test_tgt)
    history = defaultdict(list)
    t0 = time.time()

    loss_fn = loss_fn_maker()

    for epoch in range(n_epochs):
        model.train()
        el, nb = 0, 0
        for (tb,) in loader:
            inp = corrupt(tb)
            logits = model(inp)
            D = compute_ce_distance_matrix(logits, tb)
            loss = loss_fn(D)
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); nb += 1
        history["train_loss"].append(el / nb)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            m = evaluate_tokens(model, test_inp, test_tgt, n_tokens, vocab_size)
            history["matched_ce"].append(m["matched_ce"])
            history["token_accuracy"].append(m["token_accuracy"])
            history["eval_epoch"].append(epoch)

    history["total_time"] = time.time() - t0
    return history


# =========================================================================
# Main
# =========================================================================

def main():
    BASE_SEED = 42
    N_SEEDS = 3

    # =================================================================
    # ABLATION 4: Adaptive temperature (N=20, 500 epochs)
    # =================================================================
    print("\n" + "="*60)
    print("  ABLATION 4: Adaptive temperature (N=20)")
    print("="*60)

    torch.manual_seed(BASE_SEED)
    train_pts = torch.randn(2000, 20, 2)
    test_pts = torch.randn(200, 20, 2)

    adaptive_methods = {
        "chamfer":         lambda: ChamferLoss(),
        "hungarian":       lambda: HungarianLoss(),
        "pw_softmin_0.1":  lambda: ProductWeightedSoftMinLoss(temperature=0.1),
        "adaptive_0.05":   lambda: AdaptivePWSoftMin(scale=0.05),
        "adaptive_0.1":    lambda: AdaptivePWSoftMin(scale=0.1),
        "adaptive_0.2":    lambda: AdaptivePWSoftMin(scale=0.2),
    }

    for name, make_fn in adaptive_methods.items():
        print(f"\n  {name}")
        for s in range(N_SEEDS):
            torch.manual_seed(BASE_SEED + s * 1000)
            h = train_points(make_fn(), train_pts, test_pts, 20, 500)
            md = h["matched_distance"][-1]
            print(f"    seed {s}: dist={md:.4f} ({h['total_time']:.1f}s)")

    # =================================================================
    # ABLATION 5: Discrete token experiment (N=10, K=4, V=32)
    # =================================================================
    print("\n" + "="*60)
    print("  ABLATION 5: Discrete tokens (N=10, K=4, V=32, 300 epochs)")
    print("="*60)

    torch.manual_seed(BASE_SEED)
    train_tok = generate_token_data(2000, 10, 4, 32)
    test_tok = generate_token_data(200, 10, 4, 32)

    token_methods = {
        "chamfer":     lambda: ChamferLoss(),
        "hungarian":   lambda: HungarianLoss(),
        "softmin_0.1": lambda: SoftMinChamferLoss(temperature=0.1),
        "pw_softmin":  lambda: ProductWeightedSoftMinLoss(temperature=0.1),
    }

    for name, make_fn in token_methods.items():
        print(f"\n  {name}")
        for s in range(N_SEEDS):
            torch.manual_seed(BASE_SEED + s * 1000)
            h = train_tokens(name, make_fn, train_tok, test_tok, 10, 300)
            mc = h["matched_ce"][-1]
            ta = h["token_accuracy"][-1]
            print(f"    seed {s}: CE={mc:.4f} tok_acc={ta:.3f} ({h['total_time']:.1f}s)")

    print("\n\nAll done!")


if __name__ == "__main__":
    main()
