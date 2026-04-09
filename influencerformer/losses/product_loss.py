"""Product Loss variants for set prediction.

Explores different aggregations that replace `min` in Chamfer distance
with smoother alternatives that provide better gradient flow and
bijectivity enforcement.

Optional mask (B, N) where 1=real, 0=padding. When provided:
- Padding entries are excluded from softmax via masked_fill(inf)
- Averages are over n_real, not N

Reference:
    Derived from the Influencer Loss (Murnane, 2024).
"""

import torch
import torch.nn as nn


def _masked_mean(values, mask, dim):
    """Mean over real entries only. values: (B, K), mask: (B, K)."""
    return (values * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1)


def _prepare_softmax_masks(D, mask):
    """Prepare masked D tensors for softmax in coverage and precision directions.

    Returns (D_sm_over_preds, D_sm_over_targets) where padding entries are
    +inf so softmax(-D/tau) gives them zero weight.

    Coverage: for each target j, softmin over predictions i → softmax dim=1
      → need to mask prediction ROWS (padding preds get inf)
    Precision: for each prediction i, softmin over targets j → softmax dim=2
      → need to mask target COLUMNS (padding targets get inf)
    """
    col_mask = mask.unsqueeze(1).bool()   # (B, 1, N) — target columns
    row_mask = mask.unsqueeze(2).bool()   # (B, N, 1) — prediction rows
    D_sm_preds = D.masked_fill(~row_mask, float('inf'))    # for softmax over dim=1
    D_sm_targets = D.masked_fill(~col_mask, float('inf'))  # for softmax over dim=2
    return D_sm_preds, D_sm_targets


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

    def forward(self, D: torch.Tensor, mask=None) -> torch.Tensor:
        if mask is not None:
            D_sm_preds, D_sm_targets = _prepare_softmax_masks(D, mask)
        else:
            D_sm_preds = D_sm_targets = D

        # Coverage: for each target, softmin over preds (dim=1)
        weights_cov = torch.softmax(-D_sm_preds / self.temperature, dim=1)
        sm_cov = (weights_cov * D).sum(dim=1)  # (B, N)

        # Precision: for each pred, softmin over targets (dim=2)
        weights_prec = torch.softmax(-D_sm_targets / self.temperature, dim=2)
        sm_prec = (weights_prec * D).sum(dim=2)  # (B, M)

        if mask is not None:
            coverage = _masked_mean(sm_cov, mask, dim=1)
            precision = _masked_mean(sm_prec, mask, dim=1)
        else:
            coverage = sm_cov.mean(dim=-1)
            precision = sm_prec.mean(dim=-1)

        return (coverage + precision).mean()


class ProductWeightedSoftMinLoss(nn.Module):
    """Hybrid: softmin matching weighted by product-based coverage scores.

    Uses GM as a detached reweighting signal to upweight uncovered targets.
    """

    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, D: torch.Tensor, mask=None) -> torch.Tensor:
        if mask is not None:
            D_sm_preds, D_sm_targets = _prepare_softmax_masks(D, mask)
        else:
            D_sm_preds = D_sm_targets = D

        w_cov = torch.softmax(-D_sm_preds / self.temperature, dim=1)
        softmin_cov = (w_cov * D).sum(dim=1)

        w_prec = torch.softmax(-D_sm_targets / self.temperature, dim=2)
        softmin_prec = (w_prec * D).sum(dim=2)

        log_D = torch.log(D + self.eps)
        if mask is not None:
            # GM only over real entries
            row_mask = mask.unsqueeze(2)  # (B, N, 1)
            col_mask = mask.unsqueeze(1)  # (B, 1, N)
            n_real = mask.sum(dim=1, keepdim=True).clamp(min=1)
            col_gm = torch.exp((log_D * row_mask).sum(dim=1) / n_real).detach()
            row_gm = torch.exp((log_D * col_mask).sum(dim=2) / n_real).detach()
        else:
            col_gm = torch.exp(log_D.mean(dim=1)).detach()
            row_gm = torch.exp(log_D.mean(dim=2)).detach()

        col_w = col_gm / (col_gm.mean(dim=-1, keepdim=True) + self.eps)
        row_w = row_gm / (row_gm.mean(dim=-1, keepdim=True) + self.eps)

        if mask is not None:
            coverage = _masked_mean(col_w * softmin_cov, mask, dim=1)
            precision = _masked_mean(row_w * softmin_prec, mask, dim=1)
        else:
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


class SoftDCDLoss(nn.Module):
    """Soft DCD: SoftMin matching with soft query-frequency anti-duplication.

    Borrows DCD's key insight (1/n_ŷ query-frequency weighting) but makes
    it differentiable by using soft claim counts from the softmax instead
    of hard argmin counts.

    For each target j, the "soft claim count" is:
        c_j = Σ_i softmax(-D/τ, dim=targets)_ij

    This measures how many predictions are "pointed at" target j. Targets
    with high claim count (duplicated) get downweighted; targets with low
    claim count (uncovered) get upweighted.

    This combines DCD's exact anti-duplication principle with SoftMin's
    differentiability.
    """

    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, D: torch.Tensor, mask=None) -> torch.Tensor:
        if mask is not None:
            D_sm_preds, D_sm_targets = _prepare_softmax_masks(D, mask)
        else:
            D_sm_preds = D_sm_targets = D

        # --- Coverage direction ---
        # Soft assignments: softmax over targets → claim count per target
        pred_claims = torch.softmax(-D_sm_targets / self.temperature, dim=2)  # (B, M, N)
        if mask is not None:
            # Sum only over real predictions (dim=1 = M axis)
            claim_count = (pred_claims * mask.unsqueeze(2)).sum(dim=1)  # (B, N)
        else:
            claim_count = pred_claims.sum(dim=1)

        inv_freq = 1.0 / (claim_count + self.eps)
        if mask is not None:
            inv_freq_mean = _masked_mean(inv_freq, mask, dim=1).unsqueeze(1)
            inv_freq_norm = inv_freq / (inv_freq_mean + self.eps)
        else:
            inv_freq_norm = inv_freq / (inv_freq.mean(dim=-1, keepdim=True) + self.eps)

        w_cov = torch.softmax(-D_sm_preds / self.temperature, dim=1)
        softmin_cov = (w_cov * D).sum(dim=1)  # (B, N)

        if mask is not None:
            coverage = _masked_mean(inv_freq_norm.detach() * softmin_cov, mask, dim=1)
        else:
            coverage = (inv_freq_norm.detach() * softmin_cov).mean(dim=-1)

        # --- Precision direction (symmetric) ---
        tgt_claims = torch.softmax(-D_sm_preds / self.temperature, dim=1)  # (B, M, N)
        if mask is not None:
            # Sum only over real targets (dim=2 = N axis)
            claim_count_prec = (tgt_claims * mask.unsqueeze(1)).sum(dim=2)  # (B, M)
        else:
            claim_count_prec = tgt_claims.sum(dim=2)

        inv_freq_prec = 1.0 / (claim_count_prec + self.eps)
        if mask is not None:
            inv_freq_prec_mean = _masked_mean(inv_freq_prec, mask, dim=1).unsqueeze(1)
            inv_freq_prec_norm = inv_freq_prec / (inv_freq_prec_mean + self.eps)
        else:
            inv_freq_prec_norm = inv_freq_prec / (inv_freq_prec.mean(dim=-1, keepdim=True) + self.eps)

        w_prec = torch.softmax(-D_sm_targets / self.temperature, dim=2)
        softmin_prec = (w_prec * D).sum(dim=2)  # (B, M)

        if mask is not None:
            precision = _masked_mean(inv_freq_prec_norm.detach() * softmin_prec, mask, dim=1)
        else:
            precision = (inv_freq_prec_norm.detach() * softmin_prec).mean(dim=-1)

        return (coverage + precision).mean()


class CombinedSoftMinLoss(nn.Module):
    """SoftMin with both GM coverage weighting AND query-frequency weighting.

    Combines the best of both PW-SoftMin and SoftDCD:
    - GM reweighting: upweight uncovered targets (high GM = far from all preds)
    - Query-frequency: downweight over-claimed targets (high count = duplicated)

    The two signals are complementary:
    - GM measures DISTANCE to nearest prediction (coverage quality)
    - Claim count measures NUMBER of competing predictions (duplication)
    """

    def __init__(self, temperature: float = 0.1, gm_weight: float = 0.5,
                 freq_weight: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.gm_weight = gm_weight
        self.freq_weight = freq_weight
        self.eps = eps

    def forward(self, D: torch.Tensor, mask=None) -> torch.Tensor:
        if mask is not None:
            D_sm_preds, D_sm_targets = _prepare_softmax_masks(D, mask)
        else:
            D_sm_preds = D_sm_targets = D

        log_D = torch.log(D + self.eps)

        # --- Coverage direction ---
        w_cov = torch.softmax(-D_sm_preds / self.temperature, dim=1)
        softmin_cov = (w_cov * D).sum(dim=1)  # (B, N)

        # GM weight: mean log_D over real predictions (dim=1)
        if mask is not None:
            n_real = mask.sum(dim=1, keepdim=True).clamp(min=1)
            gm = torch.exp((log_D * mask.unsqueeze(2)).sum(dim=1) / n_real).detach()
        else:
            gm = torch.exp(log_D.mean(dim=1)).detach()
        if mask is not None:
            gm_w = gm / (_masked_mean(gm, mask, dim=1).unsqueeze(1) + self.eps)
        else:
            gm_w = gm / (gm.mean(-1, keepdim=True) + self.eps)

        # Query-frequency weight (sum over real predictions)
        pred_claims = torch.softmax(-D_sm_targets / self.temperature, dim=2)
        if mask is not None:
            claims = (pred_claims * mask.unsqueeze(2)).sum(dim=1)
        else:
            claims = pred_claims.sum(dim=1)
        inv_freq = (1.0 / (claims + self.eps)).detach()
        if mask is not None:
            freq_w = inv_freq / (_masked_mean(inv_freq, mask, dim=1).unsqueeze(1) + self.eps)
        else:
            freq_w = inv_freq / (inv_freq.mean(-1, keepdim=True) + self.eps)

        combined_w = self.gm_weight * gm_w + self.freq_weight * freq_w
        if mask is not None:
            combined_w = combined_w / (_masked_mean(combined_w, mask, dim=1).unsqueeze(1) + self.eps)
            coverage = _masked_mean(combined_w * softmin_cov, mask, dim=1)
        else:
            combined_w = combined_w / (combined_w.mean(-1, keepdim=True) + self.eps)
            coverage = (combined_w * softmin_cov).mean(dim=-1)

        # --- Precision direction (symmetric) ---
        w_prec = torch.softmax(-D_sm_targets / self.temperature, dim=2)
        softmin_prec = (w_prec * D).sum(dim=2)  # (B, M)

        # GM weight: mean log_D over real targets (dim=2)
        if mask is not None:
            gm_prec = torch.exp((log_D * mask.unsqueeze(1)).sum(dim=2) / n_real).detach()
        else:
            gm_prec = torch.exp(log_D.mean(dim=2)).detach()
        if mask is not None:
            gm_w_prec = gm_prec / (_masked_mean(gm_prec, mask, dim=1).unsqueeze(1) + self.eps)
        else:
            gm_w_prec = gm_prec / (gm_prec.mean(-1, keepdim=True) + self.eps)

        tgt_claims = torch.softmax(-D_sm_preds / self.temperature, dim=1)
        if mask is not None:
            claims_prec = (tgt_claims * mask.unsqueeze(1)).sum(dim=2)
        else:
            claims_prec = tgt_claims.sum(dim=2)
        inv_freq_prec = (1.0 / (claims_prec + self.eps)).detach()
        if mask is not None:
            freq_w_prec = inv_freq_prec / (_masked_mean(inv_freq_prec, mask, dim=1).unsqueeze(1) + self.eps)
        else:
            freq_w_prec = inv_freq_prec / (inv_freq_prec.mean(-1, keepdim=True) + self.eps)

        combined_w_prec = self.gm_weight * gm_w_prec + self.freq_weight * freq_w_prec
        if mask is not None:
            combined_w_prec = combined_w_prec / (_masked_mean(combined_w_prec, mask, dim=1).unsqueeze(1) + self.eps)
            precision = _masked_mean(combined_w_prec * softmin_prec, mask, dim=1)
        else:
            combined_w_prec = combined_w_prec / (combined_w_prec.mean(-1, keepdim=True) + self.eps)
            precision = (combined_w_prec * softmin_prec).mean(dim=-1)

        return (coverage + precision).mean()


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


# =============================================================================
# Log-wrapped variants — the product structure enters through the gradient
# =============================================================================


class PowerSoftMinLoss(nn.Module):
    """Power-SoftMin: Σ_j softmin(D_ij)^p.

    Raises softmin to a power p > 1 to amplify uncovered targets.
    The gradient ∂(sm^p)/∂D = p × sm^{p-1} × ∂sm/∂D is larger for
    targets with higher softmin (uncovered), creating a natural
    coverage enforcement that's stronger than PW-SoftMin's detached
    GM reweighting.

    At p=1: recovers standard SoftMin Chamfer.
    At p=2: squared softmin — gradient ∝ softmin value.
    At p=3: cubic — even stronger coverage pressure.

    The power is applied AFTER softmin, so the matching signal (from
    softmax) is preserved. Only the per-target aggregation changes.
    """

    def __init__(self, temperature: float = 0.1, power: float = 2.0):
        super().__init__()
        self.temperature = temperature
        self.power = power

    def forward(self, D: torch.Tensor, mask=None) -> torch.Tensor:
        if mask is not None:
            D_sm_preds, D_sm_targets = _prepare_softmax_masks(D, mask)
        else:
            D_sm_preds = D_sm_targets = D

        # Coverage: for each target, softmin over preds (dim=1)
        w_cov = torch.softmax(-D_sm_preds / self.temperature, dim=1)
        sm_cov = (w_cov * D).sum(dim=1)  # (B, N)

        # Precision: for each pred, softmin over targets (dim=2)
        w_prec = torch.softmax(-D_sm_targets / self.temperature, dim=2)
        sm_prec = (w_prec * D).sum(dim=2)  # (B, M)

        cov = sm_cov.pow(self.power)
        pre = sm_prec.pow(self.power)

        if mask is not None:
            coverage = _masked_mean(cov, mask, dim=1)
            precision = _masked_mean(pre, mask, dim=1)
        else:
            coverage = cov.mean(dim=-1)
            precision = pre.mean(dim=-1)

        return (coverage + precision).mean()


class LogProductSoftMinLoss(nn.Module):
    """Log-Product-SoftMin: -Σ_j log(softmin_i(D_ij)).

    NOTE: This amplifies COVERED targets (small softmin → large -log).
    Included for completeness but Power-SoftMin is the correct
    formulation for coverage enforcement.
    """

    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        w_cov = torch.softmax(-D / self.temperature, dim=1)
        sm_cov = (w_cov * D).sum(dim=1)  # (B, N)
        w_prec = torch.softmax(-D / self.temperature, dim=2)
        sm_prec = (w_prec * D).sum(dim=2)  # (B, M)

        log_coverage = -torch.log(sm_cov + self.eps).mean(dim=-1)
        log_precision = -torch.log(sm_prec + self.eps).mean(dim=-1)

        return (log_coverage + log_precision).mean()


class LogChamferLoss(nn.Module):
    """Log-Chamfer: -Σ_j log(min_i D_ij). Included for ablation."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        coverage = -torch.log(D.min(dim=1).values + self.eps).mean(dim=-1)
        precision = -torch.log(D.min(dim=2).values + self.eps).mean(dim=-1)
        return (coverage + precision).mean()
