"""Product Loss variants for set prediction.

Explores different aggregations that replace `min` in Chamfer distance
with smoother alternatives that provide better gradient flow and
bijectivity enforcement.

Reference:
    Derived from the Influencer Loss (Murnane, 2024).
"""

import torch
import torch.nn as nn


class ProductWeightedSoftMinLoss(nn.Module):
    """Hybrid: softmin matching weighted by product-based coverage scores.

    For each target, compute:
    1. Softmin distance (how close is the nearest prediction?) — drives matching
    2. GM coverage score (product of all distances) — measures how "uncovered" it is

    The final loss upweights uncovered targets, focusing gradient budget on
    the hardest-to-match targets while using softmin for the actual matching signal.

    coverage  = (1/N) Σ_j  w_j * softmin_i(D_ij)
    precision = (1/M) Σ_i  w_i * softmin_j(D_ij)

    where w_j = stop_grad(GM_i(D_ij)) / mean(GM) normalises the weights.
    """

    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        # Softmin distances
        w_cov = torch.softmax(-D / self.temperature, dim=1)
        softmin_cov = (w_cov * D).sum(dim=1)  # (B, N)

        w_prec = torch.softmax(-D / self.temperature, dim=2)
        softmin_prec = (w_prec * D).sum(dim=2)  # (B, M)

        # GM coverage scores (detached — don't backprop through weights)
        log_D = torch.log(D + self.eps)
        col_gm = torch.exp(log_D.mean(dim=1)).detach()  # (B, N)
        row_gm = torch.exp(log_D.mean(dim=2)).detach()  # (B, M)

        # Normalize weights so they sum to 1 (just reweighting, not changing scale)
        col_w = col_gm / (col_gm.mean(dim=-1, keepdim=True) + self.eps)
        row_w = row_gm / (row_gm.mean(dim=-1, keepdim=True) + self.eps)

        coverage = (col_w * softmin_cov).mean(dim=-1)
        precision = (row_w * softmin_prec).mean(dim=-1)

        return (coverage + precision).mean()


class ProductLoss(nn.Module):
    """Geometric-mean Chamfer: replace min with GM in Chamfer distance.

    coverage  = (1/N) Σ_j GM_i(D_ij)
    precision = (1/M) Σ_i GM_j(D_ij)

    GM_i(D_ij) = exp(mean_i(log D_ij))
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        log_D = torch.log(D + self.eps)
        coverage = torch.exp(log_D.mean(dim=1)).mean(dim=-1)
        precision = torch.exp(log_D.mean(dim=2)).mean(dim=-1)
        return (coverage + precision).mean()


class SoftMinChamferLoss(nn.Module):
    """Soft-min Chamfer: replace hard min with temperature-scaled softmin.

    coverage  = (1/N) Σ_j softmin_i(D_ij)
    precision = (1/M) Σ_i softmin_j(D_ij)

    softmin(x) = Σ x_i * softmax(-x_i / τ)

    At τ→0 this recovers Chamfer (hard min).
    At τ→∞ this approaches the arithmetic mean.
    At moderate τ, gradients flow to all predictions, weighted by proximity.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        # Soft-min over predictions for each target (coverage)
        weights_cov = torch.softmax(-D / self.temperature, dim=1)  # (B, M, N)
        coverage = (weights_cov * D).sum(dim=1).mean(dim=-1)  # (B,)

        # Soft-min over targets for each prediction (precision)
        weights_prec = torch.softmax(-D / self.temperature, dim=2)  # (B, M, N)
        precision = (weights_prec * D).sum(dim=2).mean(dim=-1)  # (B,)

        return (coverage + precision).mean()
