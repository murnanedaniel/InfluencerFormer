"""Custom set prediction losses for PointSWD benchmark.

These wrap our distance-matrix-based losses into the PointSWD interface:
    forward(x, y, **kwargs) -> {"loss": scalar}

where x, y are (B, N_points, 3) point clouds.
"""

import torch
import torch.nn as nn


class _DistMatrixWrapper(nn.Module):
    """Base class: computes pairwise L2 distance matrix, delegates to child."""

    def forward(self, x, y, *args, **kwargs):
        D = torch.cdist(x, y)  # (B, N, N)
        return {"loss": self._loss_from_D(D)}

    def _loss_from_D(self, D):
        raise NotImplementedError


class PureTorchChamfer(_DistMatrixWrapper):
    """Pure-PyTorch Chamfer distance (no CUDA extension needed)."""

    def _loss_from_D(self, D):
        coverage = D.min(dim=1).values.mean(dim=-1)
        precision = D.min(dim=2).values.mean(dim=-1)
        return (coverage + precision).mean()


class PowerSoftMin(_DistMatrixWrapper):
    """Power-SoftMin: softmin^p with coverage amplification.

    The key result from our loss function study: raising softmin
    to power p>1 amplifies uncovered targets through the gradient,
    achieving near-Hungarian quality at Chamfer speed.
    """

    def __init__(self, temperature=0.01, power=3.0, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.power = power

    def _loss_from_D(self, D):
        w_cov = torch.softmax(-D / self.temperature, dim=1)
        sm_cov = (w_cov * D).sum(dim=1)
        w_prec = torch.softmax(-D / self.temperature, dim=2)
        sm_prec = (w_prec * D).sum(dim=2)
        return (sm_cov.pow(self.power).mean(-1) + sm_prec.pow(self.power).mean(-1)).mean()


class SoftMinChamfer(_DistMatrixWrapper):
    """SoftMin Chamfer: smooth approximation to Chamfer distance."""

    def __init__(self, temperature=0.01, **kwargs):
        super().__init__()
        self.temperature = temperature

    def _loss_from_D(self, D):
        w_cov = torch.softmax(-D / self.temperature, dim=1)
        w_prec = torch.softmax(-D / self.temperature, dim=2)
        cov = (w_cov * D).sum(1).mean(-1)
        prec = (w_prec * D).sum(2).mean(-1)
        return (cov + prec).mean()


class PWsoftmin(_DistMatrixWrapper):
    """Product-Weighted SoftMin: GM coverage reweighting."""

    def __init__(self, temperature=0.01, eps=1e-8, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def _loss_from_D(self, D):
        w_cov = torch.softmax(-D / self.temperature, dim=1)
        sm_cov = (w_cov * D).sum(1)
        w_prec = torch.softmax(-D / self.temperature, dim=2)
        sm_prec = (w_prec * D).sum(2)

        log_D = torch.log(D + self.eps)
        cgm = torch.exp(log_D.mean(1)).detach()
        rgm = torch.exp(log_D.mean(2)).detach()
        cw = cgm / (cgm.mean(-1, keepdim=True) + self.eps)
        rw = rgm / (rgm.mean(-1, keepdim=True) + self.eps)

        return ((cw * sm_cov).mean(-1) + (rw * sm_prec).mean(-1)).mean()
