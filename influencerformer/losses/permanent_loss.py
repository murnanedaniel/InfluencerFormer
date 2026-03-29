"""Permanent-based set prediction loss.

The partition function over all matchings:
    Z = perm(exp(-D/T)) = Σ_σ Π_i exp(-D_{i,σ(i)}/T)

The loss is the free energy: F = -T · log(Z)

This is the EXACT soft matching — the negative log-likelihood marginalized
over ALL possible permutations, not just the best one (Hungarian) or an
iterative approximation (Sinkhorn).

Computed via Ryser's formula: O(2^N × N). Feasible for N ≤ 20.

Reference: This is the statistical mechanics of bipartite matchings.
The permanent is the partition function, the temperature controls the
sharpness of the matching, and the gradient gives the exact marginal
matching probabilities (thermal averages).
"""

import torch
import torch.nn as nn


def ryser_permanent_batched(A):
    """Vectorized Ryser's formula for batched square matrices.

    perm(A) = (-1)^n Σ_{S⊆[n], S≠∅} (-1)^{n-|S|} Π_i (Σ_{j∈S} A_ij)

    All 2^n subsets are processed in parallel via bitmask tensors.

    Args:
        A: (B, n, n) batch of square matrices

    Returns:
        (B,) permanents
    """
    B, n, _ = A.shape
    device = A.device
    dtype = A.dtype

    # Generate all 2^n - 1 non-empty subsets as bitmasks
    num_subsets = 2**n - 1
    # Create subset masks: (num_subsets, n) binary tensor
    bits = torch.arange(1, 2**n, device=device)  # (2^n - 1,)
    masks = torch.zeros(num_subsets, n, device=device, dtype=dtype)
    for j in range(n):
        masks[:, j] = ((bits >> j) & 1).to(dtype)

    # Row sums for each subset: A @ masks^T → (B, n, num_subsets)
    # masks: (num_subsets, n) → transpose to (n, num_subsets)
    row_sums = torch.bmm(A, masks.T.unsqueeze(0).expand(B, -1, -1))  # (B, n, num_subsets)

    # Product of row sums for each subset: Π_i row_sums[i,S] → (B, num_subsets)
    log_row_sums = torch.log(row_sums.abs() + 1e-38)
    sign_row_sums = row_sums.sign()

    # Product in log space (for magnitude) and track sign separately
    log_products = log_row_sums.sum(dim=1)  # (B, num_subsets)
    sign_products = sign_row_sums.prod(dim=1)  # (B, num_subsets)

    # Subset sizes
    subset_sizes = masks.sum(dim=1)  # (num_subsets,)

    # Signs: (-1)^|S|
    subset_signs = ((-1.0) ** subset_sizes).unsqueeze(0)  # (1, num_subsets)

    # Weighted sum: Σ_S sign(S) × sign_product(S) × exp(log_product(S))
    # To avoid overflow, use logsumexp-like trick
    combined_signs = subset_signs * sign_products  # (B, num_subsets)

    # Separate positive and negative terms for numerical stability
    pos_mask = combined_signs > 0
    neg_mask = combined_signs < 0

    # Positive terms
    pos_log = log_products.clone()
    pos_log[~pos_mask] = -float('inf')
    pos_sum = torch.logsumexp(pos_log, dim=1)  # (B,)

    # Negative terms
    neg_log = log_products.clone()
    neg_log[~neg_mask] = -float('inf')
    neg_sum = torch.logsumexp(neg_log, dim=1)  # (B,)

    # perm = (-1)^n × (exp(pos_sum) - exp(neg_sum))
    # For stability: if pos_sum > neg_sum, perm = (-1)^n × exp(pos_sum) × (1 - exp(neg_sum - pos_sum))
    overall_sign = (-1.0) ** n
    if overall_sign > 0:
        result = torch.exp(pos_sum) - torch.exp(neg_sum)
    else:
        result = torch.exp(neg_sum) - torch.exp(pos_sum)

    return result


class PermanentLoss(nn.Module):
    """Free energy loss: F = -T · log(perm(exp(-D/T))).

    The exact partition function over all matchings.

    At T→0: recovers Hungarian (best matching)
    At T→∞: uniform averaging
    At T~median(D): balanced soft matching

    Complexity: O(B × 2^N × N). Feasible for N ≤ 20.
    For N > 20, fall back to Sinkhorn (approximate Bethe permanent).
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        B, M, N = D.shape
        n = min(M, N)
        D_sq = D[:, :n, :n]

        A = torch.exp(-D_sq / self.temperature)
        perm = ryser_permanent_batched(A)

        # Free energy = -T * log(Z)
        log_perm = torch.log(perm.clamp(min=1e-38))
        free_energy = -self.temperature * log_perm

        return free_energy.mean()
