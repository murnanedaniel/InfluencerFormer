"""Product Loss variants for set prediction.

Explores different aggregations that replace `min` in Chamfer distance
with smoother alternatives that provide better gradient flow and
bijectivity enforcement.

Reference:
    Derived from the Influencer Loss (Murnane, 2024).
"""

import torch
import torch.nn as nn


# =============================================================================
# Core product variants
# =============================================================================


class ProductLoss(nn.Module):
    """Geometric-mean Chamfer: replace min with GM in Chamfer distance.

    coverage  = (1/N) Σ_j GM_i(D_ij)
    precision = (1/M) Σ_i GM_j(D_ij)
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        log_D = torch.log(D + self.eps)
        coverage = torch.exp(log_D.mean(dim=1)).mean(dim=-1)
        precision = torch.exp(log_D.mean(dim=2)).mean(dim=-1)
        return (coverage + precision).mean()


class LogDistanceProductLoss(nn.Module):
    """Product loss on log-compressed distances: Π log(1 + D_ij).

    The log compresses the distance range, making the product more
    discriminative when distances are similar. Gradient:
        ∂/∂D_kj = Π_{i≠k} log(1+D_ij) * 1/(1+D_kj)
    The 1/(1+D) factor gives stronger gradient for small D.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        log1p_D = torch.log1p(D)  # log(1 + D), well-defined for D >= 0
        log_log1p = torch.log(log1p_D + self.eps)

        coverage = torch.exp(log_log1p.mean(dim=1)).mean(dim=-1)
        precision = torch.exp(log_log1p.mean(dim=2)).mean(dim=-1)
        return (coverage + precision).mean()


class SigmoidProductLoss(nn.Module):
    """Product loss on sigmoid-squashed distances: Π σ(α(D - m)).

    Maps distances to (0, 1) via sigmoid with scale α and shift m.
    Product of sigmoids is small when any D_kj < m (close match).
    The sigmoid provides a sharp transition around the margin m,
    giving strong gradient near the decision boundary.
    """

    def __init__(self, margin: float = 1.0, scale: float = 5.0, eps: float = 1e-8):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.eps = eps

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        sig_D = torch.sigmoid(self.scale * (D - self.margin))
        log_sig = torch.log(sig_D + self.eps)

        coverage = torch.exp(log_sig.mean(dim=1)).mean(dim=-1)
        precision = torch.exp(log_sig.mean(dim=2)).mean(dim=-1)
        return (coverage + precision).mean()


class HuberProductLoss(nn.Module):
    """Product loss with Huber-like distance transform.

    For D < delta: use D (linear, strong gradient for nearby points)
    For D > delta: use delta + log(D/delta) (compressed, prevents overflow)

    This gives sharp gradients for small distances while keeping the
    product numerically stable for large distances.
    """

    def __init__(self, delta: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.delta = delta
        self.eps = eps

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        small = D <= self.delta
        transformed = torch.where(
            small,
            D,
            self.delta + torch.log(D / self.delta + self.eps),
        )
        log_t = torch.log(transformed + self.eps)

        coverage = torch.exp(log_t.mean(dim=1)).mean(dim=-1)
        precision = torch.exp(log_t.mean(dim=2)).mean(dim=-1)
        return (coverage + precision).mean()


# =============================================================================
# Softmin variants
# =============================================================================


class SoftMinChamferLoss(nn.Module):
    """Soft-min Chamfer: replace hard min with temperature-scaled softmin.

    At τ→0 this recovers Chamfer (hard min).
    At moderate τ, gradients flow to multiple predictions per target.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        weights_cov = torch.softmax(-D / self.temperature, dim=1)
        coverage = (weights_cov * D).sum(dim=1).mean(dim=-1)

        weights_prec = torch.softmax(-D / self.temperature, dim=2)
        precision = (weights_prec * D).sum(dim=2).mean(dim=-1)

        return (coverage + precision).mean()


class ProductWeightedSoftMinLoss(nn.Module):
    """Hybrid: softmin matching weighted by product-based coverage scores.

    Uses GM as a detached reweighting signal to upweight uncovered targets.
    """

    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        w_cov = torch.softmax(-D / self.temperature, dim=1)
        softmin_cov = (w_cov * D).sum(dim=1)

        w_prec = torch.softmax(-D / self.temperature, dim=2)
        softmin_prec = (w_prec * D).sum(dim=2)

        log_D = torch.log(D + self.eps)
        col_gm = torch.exp(log_D.mean(dim=1)).detach()
        row_gm = torch.exp(log_D.mean(dim=2)).detach()

        col_w = col_gm / (col_gm.mean(dim=-1, keepdim=True) + self.eps)
        row_w = row_gm / (row_gm.mean(dim=-1, keepdim=True) + self.eps)

        coverage = (col_w * softmin_cov).mean(dim=-1)
        precision = (row_w * softmin_prec).mean(dim=-1)

        return (coverage + precision).mean()


# =============================================================================
# Annealing / scheduling variants
# =============================================================================


class WarmStartProductLoss(nn.Module):
    """Start with Chamfer, transition to Product (GM) based on coverage.

    Monitors mean coverage GM. When it drops below transition_threshold
    (meaning targets are starting to be covered), switches from Chamfer
    to Product. Uses a smooth sigmoid blend.
    """

    def __init__(self, transition_epoch: int = 50, blend_width: int = 20, eps: float = 1e-8):
        super().__init__()
        self.transition_epoch = transition_epoch
        self.blend_width = blend_width
        self.eps = eps
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        # Chamfer
        chamfer_cov = D.min(dim=1).values.mean(dim=-1)
        chamfer_prec = D.min(dim=2).values.mean(dim=-1)
        chamfer = chamfer_cov + chamfer_prec

        # Product (GM)
        log_D = torch.log(D + self.eps)
        product_cov = torch.exp(log_D.mean(dim=1)).mean(dim=-1)
        product_prec = torch.exp(log_D.mean(dim=2)).mean(dim=-1)
        product = product_cov + product_prec

        # Sigmoid blend: 0 = pure Chamfer, 1 = pure Product
        alpha = torch.sigmoid(
            torch.tensor((self._epoch - self.transition_epoch) / max(self.blend_width, 1),
                         dtype=D.dtype, device=D.device)
        )

        loss = (1 - alpha) * chamfer + alpha * product
        return loss.mean()


class AnnealedExponentLoss(nn.Module):
    """Power-mean annealing: interpolate between min (p→∞) and GM (p→0).

    Uses the generalized mean: M_p(x) = (mean(x^{-p}))^{-1/p}

    At large p: approximates min (Chamfer-like)
    At small p: approximates geometric mean (Product-like)

    Anneal p from p_start (large) to p_end (small) over training.
    """

    def __init__(self, p_start: float = 10.0, p_end: float = 0.5, anneal_epochs: int = 200):
        super().__init__()
        self.p_start = p_start
        self.p_end = p_end
        self.anneal_epochs = anneal_epochs
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _get_p(self):
        t = min(self._epoch / max(self.anneal_epochs, 1), 1.0)
        return self.p_start + t * (self.p_end - self.p_start)

    def _power_mean_neg(self, D, p, dim):
        """Compute generalized mean of order -p (approximates min for large p)."""
        # M_{-p}(x) = (mean(x^{-p}))^{-1/p}
        # In log space: -1/p * log(mean(exp(-p * log(x))))
        # = -1/p * logsumexp(-p * log(x)) + 1/p * log(N)
        log_D = torch.log(D + 1e-8)
        N = D.shape[dim]
        result = (-1.0 / p) * (
            torch.logsumexp(-p * log_D, dim=dim)
            - torch.log(torch.tensor(N, dtype=D.dtype, device=D.device))
        )
        return torch.exp(result)

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        p = self._get_p()
        coverage = self._power_mean_neg(D, p, dim=1).mean(dim=-1)
        precision = self._power_mean_neg(D, p, dim=2).mean(dim=-1)
        return (coverage + precision).mean()
