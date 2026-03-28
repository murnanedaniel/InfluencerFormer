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


class OrderedLoss(nn.Module):
    """Baseline: assume prediction i matches target i (diagonal of D)."""

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        B, M, N = D.shape
        n = min(M, N)
        diag = torch.diagonal(D[:, :n, :n], dim1=1, dim2=2)  # (B, n)
        return diag.mean()
