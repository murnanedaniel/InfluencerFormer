"""Baseline set prediction losses for comparison.

All losses take a (B, M, N) pairwise distance matrix D where
M = predictions, N = targets.

Optional mask (B, N) where 1=real, 0=padding. When provided:
- Padding entries are excluded from softmax/min via masked_fill(inf)
- Averages are over n_real, not N
"""

import os
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import linear_sum_assignment

# Module-level pool: avoids per-call startup overhead.
# Use half the available CPUs, capped at 32.
_N_WORKERS = min(32, max(1, (os.cpu_count() or 2) // 2))
_POOL = ThreadPoolExecutor(max_workers=_N_WORKERS)


def _batch_hungarian(D_np, mask_tensor, D_orig):
    """Solve a batch of LAP problems in parallel, return stacked mean loss.

    For small batches the overhead exceeds savings; falls back to serial.
    """
    B = D_np.shape[0]

    def solve(b):
        n = int(mask_tensor[b].sum().item()) if mask_tensor is not None else D_np.shape[2]
        row, col = linear_sum_assignment(D_np[b, :n, :n])
        return b, row, col

    if B >= 64:   # parallel only worth it for large batches
        results = list(_POOL.map(solve, range(B)))
    else:
        results = [solve(b) for b in range(B)]

    return torch.stack([D_orig[b, row, col].mean() for b, row, col in results]).mean()


def _masked_mean(values, mask, dim):
    """Mean over real entries only. values: (B, K), mask: (B, K)."""
    return (values * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1)


class ChamferLoss(nn.Module):
    """Standard Chamfer distance: sum of nearest-neighbour distances."""

    def forward(self, D: torch.Tensor, mask=None) -> torch.Tensor:
        if mask is not None:
            col_mask = mask.unsqueeze(1).bool()   # (B, 1, N)
            row_mask = mask.unsqueeze(2).bool()   # (B, N, 1)
            # inf for padding so min ignores them
            coverage = D.masked_fill(~col_mask, float('inf')).min(dim=2).values  # (B, M)
            precision = D.masked_fill(~row_mask, float('inf')).min(dim=1).values  # (B, N)
            coverage = _masked_mean(coverage, mask, dim=1)
            precision = _masked_mean(precision, mask, dim=1)
        else:
            coverage = D.min(dim=2).values.mean(dim=-1)   # (B,)
            precision = D.min(dim=1).values.mean(dim=-1)  # (B,)
        return (coverage + precision).mean()


class NormalizedHungarianLoss(nn.Module):
    """Hungarian matching with DETR-style cost normalization."""

    def forward(self, D: torch.Tensor, mask=None) -> torch.Tensor:
        D_np = D.detach().cpu().numpy()
        B = D_np.shape[0]

        def solve(b):
            n = int(mask[b].sum().item()) if mask is not None else D_np.shape[2]
            D_sub = D_np[b, :n, :n]
            d_min, d_range = D_sub.min(), D_sub.max() - D_sub.min() + 1e-8
            row, col = linear_sum_assignment((D_sub - d_min) / d_range)
            return b, row, col

        results = list(_POOL.map(solve, range(B))) if B >= 64 else [solve(b) for b in range(B)]
        return torch.stack([D[b, row, col].mean() for b, row, col in results]).mean()


class HungarianLoss(nn.Module):
    """Hungarian matching loss (DETR-style)."""

    def forward(self, D: torch.Tensor, mask=None) -> torch.Tensor:
        return _batch_hungarian(D.detach().cpu().numpy(), mask, D)


class SinkhornLoss(nn.Module):
    """Sinkhorn-based soft matching loss."""

    def __init__(self, eps: float = 0.1, n_iters: int = 20):
        super().__init__()
        self.eps = eps
        self.n_iters = n_iters

    def forward(self, D: torch.Tensor, mask=None) -> torch.Tensor:
        if mask is not None:
            # Mask padding with -inf in log-kernel so they get 0 assignment
            pad_mask = ~mask.unsqueeze(1).bool()  # (B, 1, N)
            pad_mask_row = ~mask.unsqueeze(2).bool()  # (B, N, 1)
            pad_2d = pad_mask | pad_mask_row  # (B, N, N)
            log_K = (-D / self.eps).masked_fill(pad_2d, float('-inf'))
        else:
            log_K = -D / self.eps  # (B, M, N)

        for _ in range(self.n_iters):
            log_K = log_K - torch.logsumexp(log_K, dim=2, keepdim=True)
            log_K = log_K - torch.logsumexp(log_K, dim=1, keepdim=True)

        P = torch.exp(log_K)  # (B, M, N) soft assignment

        if mask is not None:
            # Sum only real entries, normalize by n_real^2
            n_real = mask.sum(dim=1).clamp(min=1)  # (B,)
            per_sample = (P * D).sum(dim=(1, 2)) / n_real
            return per_sample.mean()
        return (P * D).sum(dim=(1, 2)).mean()


class DCDLoss(nn.Module):
    """Density-aware Chamfer Distance (Wu et al., NeurIPS 2021)."""

    def __init__(self, alpha: float = 1000.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, D: torch.Tensor, mask=None) -> torch.Tensor:
        if mask is not None:
            # Use submatrix per sample (DCD uses argmin which doesn't vectorize well with masking)
            B = D.shape[0]
            losses = []
            for b in range(B):
                n = int(mask[b].sum().item())
                D_sub = D[b:b+1, :n, :n]
                losses.append(self._dcd_unmasked(D_sub))
            return torch.stack(losses).mean()
        return self._dcd_unmasked(D)

    def _dcd_unmasked(self, D):
        B, M, N = D.shape
        nn_pred_idx = D.argmin(dim=2)
        nn_pred_dist = D.min(dim=2).values
        exp_pred = torch.exp(-self.alpha * nn_pred_dist)

        nn_onehot = torch.zeros(B, M, N, device=D.device)
        nn_onehot.scatter_(2, nn_pred_idx.unsqueeze(2), 1.0)
        n_y = nn_onehot.sum(dim=1).clamp(min=1)
        n_y_per_pred = torch.gather(n_y, 1, nn_pred_idx)
        coverage = (1.0 - exp_pred / n_y_per_pred).mean(dim=-1)

        nn_tgt_idx = D.argmin(dim=1)
        nn_tgt_dist = D.min(dim=1).values
        exp_tgt = torch.exp(-self.alpha * nn_tgt_dist)

        nn_onehot_t = torch.zeros(B, N, M, device=D.device)
        nn_onehot_t.scatter_(2, nn_tgt_idx.unsqueeze(2), 1.0)
        n_x = nn_onehot_t.sum(dim=1).clamp(min=1)
        n_x_per_tgt = torch.gather(n_x, 1, nn_tgt_idx)
        precision = (1.0 - exp_tgt / n_x_per_tgt).mean(dim=-1)

        return 0.5 * (coverage + precision).mean()


class ClampedHungarianLoss(nn.Module):
    """Hungarian matching with clamped cost matrix."""

    def __init__(self, max_cost: float = 20.0):
        super().__init__()
        self.max_cost = max_cost

    def forward(self, D: torch.Tensor, mask=None) -> torch.Tensor:
        return _batch_hungarian(D.detach().clamp(max=self.max_cost).cpu().numpy(), mask, D)


class OrderedLoss(nn.Module):
    """Baseline: assume prediction i matches target i (diagonal of D)."""

    def forward(self, D: torch.Tensor, mask=None) -> torch.Tensor:
        B, M, N = D.shape
        n = min(M, N)
        diag = torch.diagonal(D[:, :n, :n], dim1=1, dim2=2)  # (B, n)
        if mask is not None:
            return _masked_mean(diag, mask[:, :n], dim=1).mean()
        return diag.mean()
