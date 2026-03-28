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

        # --- Coverage direction: for each prediction x, find nearest target ŷ ---
        nn_pred_to_tgt = D.argmin(dim=2)  # (B, M) — which target is nearest for each pred
        nn_pred_dist = D.min(dim=2).values  # (B, M) — distance to nearest target
        exp_pred = torch.exp(-self.alpha * nn_pred_dist)  # (B, M)

        # Query frequency: how many predictions share each target as NN?
        # n_ŷ[b, j] = count of predictions in batch b whose NN is target j
        n_y = torch.zeros(B, N, device=D.device)
        for b in range(B):
            n_y[b].scatter_add_(0, nn_pred_to_tgt[b], torch.ones(M, device=D.device))
        n_y = n_y.clamp(min=1)  # avoid div by zero

        # Weight each prediction's match by 1/n_ŷ
        n_y_per_pred = torch.gather(n_y, 1, nn_pred_to_tgt)  # (B, M)
        coverage = (1.0 - exp_pred / n_y_per_pred).mean(dim=-1)  # (B,)

        # --- Precision direction: for each target y, find nearest prediction x̂ ---
        nn_tgt_to_pred = D.argmin(dim=1)  # (B, N)
        nn_tgt_dist = D.min(dim=1).values  # (B, N)
        exp_tgt = torch.exp(-self.alpha * nn_tgt_dist)

        n_x = torch.zeros(B, M, device=D.device)
        for b in range(B):
            n_x[b].scatter_add_(0, nn_tgt_to_pred[b], torch.ones(N, device=D.device))
        n_x = n_x.clamp(min=1)

        n_x_per_tgt = torch.gather(n_x, 1, nn_tgt_to_pred)
        precision = (1.0 - exp_tgt / n_x_per_tgt).mean(dim=-1)

        return 0.5 * (coverage + precision).mean()


class OrderedLoss(nn.Module):
    """Baseline: assume prediction i matches target i (diagonal of D)."""

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        B, M, N = D.shape
        n = min(M, N)
        diag = torch.diagonal(D[:, :n, :n], dim1=1, dim2=2)  # (B, n)
        return diag.mean()
