"""Baseline set prediction losses for comparison.

All losses take a (B, M, N) pairwise distance matrix D where
M = predictions, N = targets.
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class ChamferLoss(nn.Module):
    """Standard Chamfer distance: sum of nearest-neighbour distances."""

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        coverage = D.min(dim=1).values.mean(dim=-1)   # (B,)
        precision = D.min(dim=2).values.mean(dim=-1)  # (B,)
        return (coverage + precision).mean()


class NormalizedHungarianLoss(nn.Module):
    """Hungarian matching with DETR-style cost normalization.

    DETR uses -softmax_prob[target] (bounded in [-1, 0]) for matching,
    NOT raw cross-entropy (unbounded). This prevents the positive
    feedback loop where confident-but-wrong predictions create huge
    CE values that lock in bad assignments.

    For general distance matrices, we normalize D to [0, 1] per-sample
    before matching, then compute the loss on the original (unnormalized) D.
    """

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        B = D.shape[0]
        losses = []

        for b in range(B):
            # Normalize cost for matching: bounded [0, 1]
            D_b = D[b]
            D_norm = D_b.detach()
            d_min = D_norm.min()
            d_range = D_norm.max() - d_min + 1e-8
            D_matching = ((D_norm - d_min) / d_range).cpu().numpy()

            row_ind, col_ind = linear_sum_assignment(D_matching)
            # Loss on ORIGINAL distances (for proper gradients)
            matched = D[b, row_ind, col_ind]
            losses.append(matched.mean())

        return torch.stack(losses).mean()


class HungarianLoss(nn.Module):
    """Hungarian matching loss (DETR-style).

    Finds optimal one-to-one assignment via scipy, then computes
    loss on matched pairs. Matching is detached (no gradient through it).
    """

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        B = D.shape[0]
        losses = []
        D_detached = D.detach().cpu().numpy()

        for b in range(B):
            row_ind, col_ind = linear_sum_assignment(D_detached[b])
            matched = D[b, row_ind, col_ind]
            losses.append(matched.mean())

        return torch.stack(losses).mean()


class SinkhornLoss(nn.Module):
    """Sinkhorn-based soft matching loss.

    Computes a soft doubly-stochastic assignment matrix via Sinkhorn
    iterations, then uses it to weight the distance matrix.
    """

    def __init__(self, eps: float = 0.1, n_iters: int = 20):
        super().__init__()
        self.eps = eps
        self.n_iters = n_iters

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        # Sinkhorn iterations on kernel K = exp(-D / eps)
        log_K = -D / self.eps  # (B, M, N)

        for _ in range(self.n_iters):
            log_K = log_K - torch.logsumexp(log_K, dim=2, keepdim=True)
            log_K = log_K - torch.logsumexp(log_K, dim=1, keepdim=True)

        P = torch.exp(log_K)  # (B, M, N) soft assignment
        return (P * D).sum(dim=(1, 2)).mean()


class DCDLoss(nn.Module):
    """Density-aware Chamfer Distance (Wu et al., NeurIPS 2021).

    Two key modifications to standard Chamfer:
    1. Exponential distance transform: exp(-alpha * d) with alpha=1000,
       essentially a binary match indicator (1 if d≈0, 0 otherwise)
    2. Query-frequency weighting: each match is weighted by 1/n where n
       is the number of predictions sharing the same nearest target.
       This is the anti-duplication mechanism — if 3 predictions all
       point to the same target, each gets 1/3 credit.

    DCD(S1,S2) = 0.5 * [ mean_x(1 - exp(-α‖x-ŷ‖)/n_ŷ) + mean_y(1 - exp(-α‖y-x̂‖)/n_x̂) ]

    Reference: arXiv:2111.12702
    """

    def __init__(self, alpha: float = 1000.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        B, M, N = D.shape

        # --- Coverage: for each prediction, find nearest target ---
        nn_pred_idx = D.argmin(dim=2)  # (B, M)
        nn_pred_dist = D.min(dim=2).values  # (B, M)
        exp_pred = torch.exp(-self.alpha * nn_pred_dist)  # (B, M)

        # Query frequency: vectorized scatter_add over batch
        # One-hot encode the NN assignments, sum over predictions
        nn_onehot = torch.zeros(B, M, N, device=D.device)
        nn_onehot.scatter_(2, nn_pred_idx.unsqueeze(2), 1.0)  # (B, M, N)
        n_y = nn_onehot.sum(dim=1).clamp(min=1)  # (B, N) — claims per target
        n_y_per_pred = torch.gather(n_y, 1, nn_pred_idx)  # (B, M)
        coverage = (1.0 - exp_pred / n_y_per_pred).mean(dim=-1)

        # --- Precision: for each target, find nearest prediction ---
        nn_tgt_idx = D.argmin(dim=1)  # (B, N)
        nn_tgt_dist = D.min(dim=1).values  # (B, N)
        exp_tgt = torch.exp(-self.alpha * nn_tgt_dist)

        nn_onehot_t = torch.zeros(B, N, M, device=D.device)
        nn_onehot_t.scatter_(2, nn_tgt_idx.unsqueeze(2), 1.0)
        n_x = nn_onehot_t.sum(dim=1).clamp(min=1)  # (B, M)
        n_x_per_tgt = torch.gather(n_x, 1, nn_tgt_idx)
        precision = (1.0 - exp_tgt / n_x_per_tgt).mean(dim=-1)

        return 0.5 * (coverage + precision).mean()


class ClampedHungarianLoss(nn.Module):
    """Hungarian matching with clamped cost matrix.

    Prevents the positive feedback loop that causes divergence with CE
    distances: wrong-but-confident predictions create huge CE values that
    dominate the cost matrix and lock in bad assignments.

    Clamping bounds the cost matrix, so wrong assignments can't grow
    unboundedly and the matching remains responsive to improvements.
    """

    def __init__(self, max_cost: float = 20.0):
        super().__init__()
        self.max_cost = max_cost

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        B = D.shape[0]
        losses = []
        # Clamp the detached cost matrix for matching
        D_clamped = D.detach().clamp(max=self.max_cost).cpu().numpy()

        for b in range(B):
            row_ind, col_ind = linear_sum_assignment(D_clamped[b])
            # But compute loss on the ORIGINAL (unclamped) D for gradient
            matched = D[b, row_ind, col_ind]
            losses.append(matched.mean())

        return torch.stack(losses).mean()


class OrderedLoss(nn.Module):
    """Baseline: assume prediction i matches target i (diagonal of D)."""

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        B, M, N = D.shape
        n = min(M, N)
        diag = torch.diagonal(D[:, :n, :n], dim1=1, dim2=2)  # (B, n)
        return diag.mean()
