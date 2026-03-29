"""Longer discrete token experiment (1000 epochs) to test convergence.

Uses lower corruption rate (10% instead of 30%) for easier task.

Usage:
    python examples/run_tokens_long.py
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

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


class TokenSetPredictor(nn.Module):
    def __init__(self, n_points, n_tokens=4, vocab_size=32, hidden=256):
        super().__init__()
        self.n_points, self.n_tokens, self.vocab_size = n_points, n_tokens, vocab_size
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
        emb = self.embed(x).flatten(-2).flatten(-2)
        return self.net(emb).reshape(-1, self.n_points, self.n_tokens, self.vocab_size)


def ce_distance_matrix(logits, targets):
    B, M, K, V = logits.shape
    N = targets.shape[1]
    logits_exp = logits.unsqueeze(2).expand(B, M, N, K, V)
    targets_exp = targets.unsqueeze(1).expand(B, M, N, K)
    ce = F.cross_entropy(logits_exp.reshape(-1, V), targets_exp.reshape(-1), reduction='none')
    return ce.reshape(B, M, N, K).sum(-1)


def evaluate(model, test_inp, test_tgt):
    model.eval()
    with torch.no_grad():
        logits = model(test_inp)
        D = ce_distance_matrix(logits, test_tgt)
    B, M, N = D.shape
    D_np = D.cpu().numpy()
    matched_ce, tok_correct, tok_total, n_dup = [], 0, 0, 0
    for b in range(B):
        r, c = linear_sum_assignment(D_np[b])
        matched_ce.append(D_np[b][r, c].mean())
        pred_tok = logits[b, r].argmax(-1)
        true_tok = test_tgt[b, c]
        tok_correct += (pred_tok == true_tok).sum().item()
        tok_total += true_tok.numel()
        # Dup: check if any target has 2+ preds within CE < 10 (below random CE ~13.9)
        for j in range(N):
            if (D_np[b, :, j] < 10.0).sum() > 1:
                n_dup += 1
    return {
        "matched_ce": float(np.mean(matched_ce)),
        "token_accuracy": float(tok_correct / max(tok_total, 1)),
        "duplicate_rate": float(n_dup / max(B * N, 1)),
    }


def main():
    N_POINTS = 10
    N_TOKENS = 4
    VOCAB = 32
    N_EPOCHS = 1000
    N_SEEDS = 2
    CORRUPTION = 0.1  # 10% token replacement
    BASE_SEED = 42

    torch.manual_seed(BASE_SEED)
    train_tgt = torch.randint(0, VOCAB, (2000, N_POINTS, N_TOKENS))
    test_tgt = torch.randint(0, VOCAB, (200, N_POINTS, N_TOKENS))

    def corrupt(x):
        mask = torch.rand_like(x.float()) < CORRUPTION
        return torch.where(mask, torch.randint(0, VOCAB, x.shape), x)

    test_inp = corrupt(test_tgt)

    methods = {
        "chamfer":     lambda: ChamferLoss(),
        "hungarian":   lambda: HungarianLoss(),
        "softmin":     lambda: SoftMinChamferLoss(temperature=0.5),  # higher τ for CE distances
        "pw_softmin":  lambda: ProductWeightedSoftMinLoss(temperature=0.5),
    }

    print(f"Token experiment: N={N_POINTS}, K={N_TOKENS}, V={VOCAB}, "
          f"epochs={N_EPOCHS}, corruption={CORRUPTION}")

    for name, make_fn in methods.items():
        print(f"\n  {name}")
        for s in range(N_SEEDS):
            torch.manual_seed(BASE_SEED + s * 1000)
            model = TokenSetPredictor(N_POINTS, N_TOKENS, VOCAB, hidden=256)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loader = DataLoader(TensorDataset(train_tgt), batch_size=64, shuffle=True)
            loss_fn = make_fn()
            t0 = time.time()

            for epoch in range(N_EPOCHS):
                model.train()
                for (tb,) in loader:
                    inp = corrupt(tb)
                    logits = model(inp)
                    D = ce_distance_matrix(logits, tb)
                    loss = loss_fn(D)
                    opt.zero_grad(); loss.backward(); opt.step()

                if (epoch + 1) % 200 == 0:
                    m = evaluate(model, test_inp, test_tgt)
                    elapsed = time.time() - t0
                    print(f"    seed {s}, epoch {epoch+1}: CE={m['matched_ce']:.3f} "
                          f"tok_acc={m['token_accuracy']:.3f} "
                          f"dup={m['duplicate_rate']:.3f} ({elapsed:.0f}s)")

            m = evaluate(model, test_inp, test_tgt)
            print(f"    seed {s} FINAL: CE={m['matched_ce']:.3f} "
                  f"tok_acc={m['token_accuracy']:.3f} dup={m['duplicate_rate']:.3f} "
                  f"({time.time()-t0:.0f}s)")

    print("\nDone!")


if __name__ == "__main__":
    main()
