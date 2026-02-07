"""InfluencerCriterion: drop-in replacement for Mask2Former's SetCriterion.

Mask2Former's SetCriterion does:
    1. HungarianMatcher to assign queries to GT instances
    2. CE loss on matched class predictions
    3. Dice + BCE on matched mask predictions
    4. Repeat for each auxiliary decoder layer (deep supervision)

InfluencerCriterion replaces ALL of this with MaskInfluencerLoss. No matcher
is needed — the loss itself discovers query-to-instance assignments through
the geometric mean attractive/repulsive formulation.

Usage with Mask2Former's MaskFormer model:
    # In MaskFormer.from_config(), replace:
    #   criterion = SetCriterion(matcher=HungarianMatcher(...), ...)
    # with:
    #   criterion = InfluencerCriterion(...)

    # The forward interface is the same:
    #   losses = criterion(outputs, targets)

Usage standalone (point clouds):
    criterion = InfluencerCriterion()
    outputs = {"pred_masks": mask_logits}      # (N, M)
    targets = [{"instance_labels": labels}]    # (N,)
    losses = criterion(outputs, targets)
"""

import torch
import torch.nn as nn

from ..losses.influencer_loss import MaskInfluencerLoss


class InfluencerCriterion(nn.Module):
    """Criterion wrapping MaskInfluencerLoss for MaskFormer-style models.

    Accepts the same (outputs, targets) interface as Mask2Former's
    SetCriterion, but computes the Influencer Loss instead of
    Hungarian-matched CE + dice + BCE.
    """

    def __init__(
        self,
        attr_weight: float = 1.0,
        rep_weight: float = 1.0,
        bg_weight: float = 1.0,
        rep_margin: float = 1.0,
        temperature: float = 1.0,
        deep_supervision: bool = True,
        aux_weight: float = 1.0,
    ):
        """
        Args:
            attr_weight: Weight for the attractive loss.
            rep_weight: Weight for the repulsive loss.
            bg_weight: Weight for background suppression.
            rep_margin: Margin for repulsive hinge loss.
            temperature: Temperature for soft query selection.
            deep_supervision: Whether to compute losses on auxiliary
                (intermediate decoder layer) outputs.
            aux_weight: Weight multiplier for auxiliary losses.
        """
        super().__init__()
        self.loss_fn = MaskInfluencerLoss(
            attr_weight=attr_weight,
            rep_weight=rep_weight,
            bg_weight=bg_weight,
            rep_margin=rep_margin,
            temperature=temperature,
        )
        self.deep_supervision = deep_supervision
        self.aux_weight = aux_weight

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            outputs: Model output dict. Must contain:
                - "pred_masks": Either (N, M) for a single point cloud, or
                  (B, M, H, W) for batched images (Mask2Former format).
                  For point clouds, can also be a list of (N_i, M) tensors.
                Optionally:
                - "aux_outputs": List of dicts with the same format,
                  one per intermediate decoder layer.

            targets: List of dicts (one per batch element), each containing:
                - "instance_labels": (N_i,) integer instance IDs for point
                  clouds, or (H, W) instance label map for images.
                  0 = background.

        Returns:
            Dict of named losses. All losses that should be backpropagated
            have requires_grad=True.
        """
        losses = {}

        # Main output
        main_losses = self._compute_loss(outputs["pred_masks"], targets)
        losses.update(main_losses)

        # Auxiliary losses (deep supervision)
        if self.deep_supervision and "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                aux_losses = self._compute_loss(aux["pred_masks"], targets)
                for k, v in aux_losses.items():
                    losses[f"{k}_{i}"] = v * self.aux_weight

        # Total
        total = sum(v for v in losses.values() if v.requires_grad)
        losses["total_loss"] = total
        return losses

    def _compute_loss(
        self,
        pred_masks: torch.Tensor | list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Compute loss for one set of mask predictions.

        Handles three input formats:
        1. Point cloud, single:  pred_masks is (N, M)
        2. Point cloud, batched: pred_masks is a list of (N_i, M)
        3. Image, batched:       pred_masks is (B, M, H, W)
        """
        # Single point cloud tensor with one target
        if isinstance(pred_masks, torch.Tensor) and pred_masks.dim() == 2:
            return self.loss_fn(pred_masks, targets[0]["instance_labels"])

        # List of per-sample point cloud tensors
        if isinstance(pred_masks, list):
            batch_losses = [
                self.loss_fn(m, t["instance_labels"])
                for m, t in zip(pred_masks, targets)
            ]
            return self._average_loss_dicts(batch_losses)

        # Batched image tensor (B, M, H, W) → flatten to per-pixel
        if isinstance(pred_masks, torch.Tensor) and pred_masks.dim() == 4:
            B, M, H, W = pred_masks.shape
            batch_losses = []
            for b in range(B):
                # (M, H, W) → (H*W, M)
                masks_b = pred_masks[b].permute(1, 2, 0).reshape(-1, M)
                labels_b = targets[b]["instance_labels"].reshape(-1)
                batch_losses.append(self.loss_fn(masks_b, labels_b))
            return self._average_loss_dicts(batch_losses)

        raise ValueError(
            f"Unsupported pred_masks format: type={type(pred_masks)}, "
            f"shape={getattr(pred_masks, 'shape', 'N/A')}"
        )

    @staticmethod
    def _average_loss_dicts(
        loss_dicts: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Average corresponding entries across a list of loss dicts."""
        keys = loss_dicts[0].keys()
        return {
            k: torch.stack([d[k] for d in loss_dicts]).mean() for k in keys
        }
