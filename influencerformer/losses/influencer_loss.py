"""Mask-matrix Influencer Loss for instance segmentation.

Operates on an (N, M) mask matrix where N = points/pixels and M = queries.
Uses the geometric mean formulation from the original InfluencerLoss
(Murnane 2024) to encourage all members of the same instance to unanimously
assign high mask probability to the same query, with a repulsive term
pushing different instances to claim different queries.

This replaces Hungarian matching entirely: the loss itself discovers the
query-to-instance assignment through gradient descent.

Reference:
    Murnane, D. "Influencer Loss: End-to-end Geometric Representation
    Learning for Track Reconstruction." EPJ Web Conf. 295, 09016 (2024).
    https://github.com/murnanedaniel/InfluencerNet
"""

import torch
import torch.nn as nn


class MaskInfluencerLoss(nn.Module):
    """Influencer Loss on an (N, M) mask matrix.

    Given N points and M learned queries, the model produces an (N, M) matrix
    of mask logits. Each entry mask[i, m] represents how strongly point i is
    associated with query m.

    The loss has three components:

    1. **Attractive** (follower-influencer): For each ground-truth instance k,
       compute the geometric mean of mask probabilities across the instance's
       points for each query. The best query (highest geometric mean) is the
       one all points agree on. Minimize the negative log of that geometric
       mean. The geometric mean is strict: one dissenting point tanks the
       score, requiring unanimous agreement.

    2. **Repulsive** (influencer-influencer): Each instance's "query profile"
       is its vector of per-query geometric means. Push profiles of different
       instances apart with a hinge loss, so different instances claim
       different queries.

    3. **Background suppression**: Push mask probabilities toward zero for
       points labeled as background (instance_label == 0).
    """

    def __init__(
        self,
        attr_weight: float = 1.0,
        rep_weight: float = 1.0,
        bg_weight: float = 1.0,
        rep_margin: float = 1.0,
        temperature: float = 1.0,
        eps: float = 1e-6,
    ):
        """
        Args:
            attr_weight: Weight for the attractive (geometric mean) loss.
            rep_weight: Weight for the repulsive (hinge) loss.
            bg_weight: Weight for background suppression.
            rep_margin: Margin for the repulsive hinge loss. Instance query
                profiles closer than this are penalized.
            temperature: Temperature for the soft-max query selection in the
                attractive loss. Lower = sharper selection.
            eps: Small constant for numerical stability in log.
        """
        super().__init__()
        self.attr_weight = attr_weight
        self.rep_weight = rep_weight
        self.bg_weight = bg_weight
        self.rep_margin = rep_margin
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        mask_logits: torch.Tensor,
        instance_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            mask_logits: (N, M) raw logits from cross-attention between
                point embeddings and learned queries.
            instance_labels: (N,) integer instance IDs. 0 = background.

        Returns:
            Dict with keys: loss, attractive, repulsive, background.
        """
        device = mask_logits.device
        mask_probs = torch.sigmoid(mask_logits)  # (N, M)

        unique_instances = instance_labels.unique()
        unique_instances = unique_instances[unique_instances > 0]

        if len(unique_instances) == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                "loss": zero,
                "attractive": zero.detach(),
                "repulsive": zero.detach(),
                "background": zero.detach(),
            }

        # --- Attractive loss: geometric mean per instance per query ---
        instance_profiles = []  # (K, M) — one profile per instance
        attractive_loss = torch.tensor(0.0, device=device)

        for inst_id in unique_instances:
            inst_mask = instance_labels == inst_id
            inst_probs = mask_probs[inst_mask]  # (N_k, M)

            # Log-geometric-mean per query column:
            #   log_geomean[m] = mean_i log(prob[i, m])
            # This is the log of the geometric mean of probabilities for
            # each query across all points of this instance.
            log_geomean = torch.log(inst_probs + self.eps).mean(dim=0)  # (M,)

            # Attractive loss for this instance: maximize the best query's
            # geometric mean. Use logsumexp as a differentiable soft-max:
            #   -logsumexp(log_geomean / T) * T  ≈  -max_m log_geomean[m]
            # When the model is perfect, the best query has geomean ≈ 1,
            # log_geomean ≈ 0, and this loss ≈ 0.
            attractive_loss_k = (
                -torch.logsumexp(log_geomean / self.temperature, dim=0)
                * self.temperature
            )
            attractive_loss = attractive_loss + attractive_loss_k

            instance_profiles.append(log_geomean.exp())  # (M,)

        attractive_loss = attractive_loss / len(unique_instances)

        # --- Repulsive loss: push instance query profiles apart ---
        if len(instance_profiles) > 1:
            profiles = torch.stack(instance_profiles)  # (K, M)
            # Pairwise L2 distances between profiles
            diff = profiles.unsqueeze(0) - profiles.unsqueeze(1)  # (K, K, M)
            dists = torch.norm(diff, dim=2)  # (K, K)

            K = len(unique_instances)
            off_diag = ~torch.eye(K, dtype=torch.bool, device=device)
            # Hinge: penalize pairs closer than rep_margin
            repulsive_loss = torch.clamp(
                self.rep_margin - dists[off_diag], min=0
            ).mean()
        else:
            repulsive_loss = torch.tensor(0.0, device=device)

        # --- Background suppression ---
        bg_mask = instance_labels == 0
        if bg_mask.any():
            bg_loss = mask_probs[bg_mask].mean()
        else:
            bg_loss = torch.tensor(0.0, device=device)

        total = (
            self.attr_weight * attractive_loss
            + self.rep_weight * repulsive_loss
            + self.bg_weight * bg_loss
        )

        return {
            "loss": total,
            "attractive": attractive_loss.detach(),
            "repulsive": repulsive_loss.detach(),
            "background": bg_loss.detach(),
        }
