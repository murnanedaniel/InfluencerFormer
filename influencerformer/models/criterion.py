"""InfluencerCriterion: drop-in replacement for OneFormer3D's InstanceCriterion.

OneFormer3D's InstanceCriterion (oneformer3d/instance_criterion.py) does:
    1. HungarianMatcher to assign queries to GT instances (scipy linear_sum_assignment)
    2. Cross-entropy on matched class predictions
    3. BCE + Dice on matched mask predictions
    4. Objectness score loss (MSE on IoU)
    5. Repeat for each auxiliary decoder layer (deep supervision)

InfluencerCriterion replaces the matcher + mask losses with MaskInfluencerLoss.
No HungarianMatcher, no bipartite assignment. The loss itself discovers
query-to-instance assignments via geometric mean attractive/repulsive potentials.

Integration:
    In the OneFormer3D config, replace the inst_criterion block:

    # Before (Hungarian matching):
    inst_criterion=dict(
        type='InstanceCriterion',
        matcher=dict(
            type='HungarianMatcher',
            costs=[
                dict(type='QueryClassificationCost', weight=0.5),
                dict(type='MaskBCECost', weight=1.0),
                dict(type='MaskDiceCost', weight=1.0)]),
        loss_weight=[0.5, 1.0, 1.0, 0.5],
        num_classes=num_instance_classes,
        non_object_weight=0.05,
        fix_dice_loss_weight=True,
        iter_matcher=True,
        fix_mean_loss=True)

    # After (Influencer Loss):
    inst_criterion=dict(
        type='InfluencerCriterion',
        loss_weight=[0.5, 1.0],   # [cls_weight, influencer_scale]; [2] ignored if present
        num_classes=num_instance_classes,
        non_object_weight=0.05,
        attr_weight=1.0,
        rep_weight=1.0,
        bg_weight=1.0,
        rep_margin=1.0,
        temperature=1.0,
        iter_matcher=True)
"""

import torch
import torch.nn.functional as F

from ..losses.influencer_loss import MaskInfluencerLoss

try:
    from mmdet3d.registry import MODELS
    HAS_MMDET3D = True
except ImportError:
    HAS_MMDET3D = False


def _build_instance_labels(sp_masks):
    """Convert per-instance binary superpoint masks to per-superpoint instance IDs.

    OneFormer3D stores GT as sp_masks: (n_gts, n_superpoints) binary masks.
    MaskInfluencerLoss expects instance_labels: (n_superpoints,) with integer IDs.

    Args:
        sp_masks: (n_gts, n_superpoints) boolean tensor. Each row is one
            ground-truth instance's mask over superpoints.

    Returns:
        (n_superpoints,) integer tensor. 0 = background (no instance),
        1..n_gts = instance IDs.
    """
    n_superpoints = sp_masks.shape[1]
    instance_labels = sp_masks.new_zeros(n_superpoints, dtype=torch.long)
    for inst_id, mask in enumerate(sp_masks):
        instance_labels[mask.bool()] = inst_id + 1  # 1-indexed, 0 = background
    return instance_labels


class InfluencerCriterion:
    """Drop-in replacement for OneFormer3D's InstanceCriterion.

    Same __call__(pred, insts) interface. Same config structure (minus the
    matcher). Returns {'inst_loss': scalar} just like InstanceCriterion.

    The key difference: no HungarianMatcher. Instead of matching queries to
    GT instances and then computing BCE+Dice on matched pairs, we transpose
    the mask matrix and compute the Influencer Loss directly.

    OneFormer3D's masks are (n_queries, n_superpoints) — query-centric.
    MaskInfluencerLoss expects (N, M) — point-centric, N=superpoints, M=queries.
    So we transpose: masks.T gives (n_superpoints, n_queries).
    """

    def __init__(self, loss_weight, num_classes, non_object_weight,
                 attr_weight=1.0, rep_weight=1.0, bg_weight=1.0,
                 rep_margin=1.0, temperature=1.0, iter_matcher=True):
        self.loss_fn = MaskInfluencerLoss(
            attr_weight=attr_weight,
            rep_weight=rep_weight,
            bg_weight=bg_weight,
            rep_margin=rep_margin,
            temperature=temperature,
        )
        # loss_weight[0] = classification weight
        # loss_weight[1] = outer scale on the total influencer loss (attr+rep+bg
        #                  weights are already applied inside MaskInfluencerLoss)
        # loss_weight[2] = silently ignored if present (backward compat)
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        class_weight = [1] * num_classes + [non_object_weight]
        self.class_weight = class_weight
        self.iter_matcher = iter_matcher

    def _get_influencer_loss(self, pred_masks, insts):
        """Compute MaskInfluencerLoss across a batch.

        Args:
            pred_masks: List of len batch_size, each (n_queries, n_superpoints).
            insts: List of len batch_size, each InstanceData_ with
                sp_masks (n_gts, n_superpoints).

        Returns:
            dict of averaged loss components.
        """
        batch_losses = []
        for mask, inst in zip(pred_masks, insts):
            # Transpose: (n_queries, n_superpoints) → (n_superpoints, n_queries)
            mask_logits = mask.T
            instance_labels = _build_instance_labels(inst.sp_masks)
            batch_losses.append(self.loss_fn(mask_logits, instance_labels))

        # Average across batch
        keys = batch_losses[0].keys()
        return {
            k: torch.stack([d[k] for d in batch_losses]).mean() for k in keys
        }

    def _get_cls_loss(self, cls_preds, pred_masks, insts):
        """Classification loss using soft query-instance assignment.

        Without Hungarian matching, we assign each query to the instance
        whose superpoints it most strongly activates (argmax over mean mask
        logit per GT instance). Unmatched queries get the no-object label.

        Args:
            cls_preds: List of (n_queries, n_classes + 1) per batch element.
            pred_masks: List of (n_queries, n_superpoints) per batch element.
            insts: List of InstanceData_ with labels_3d and sp_masks.

        Returns:
            Scalar classification loss.
        """
        cls_losses = []
        for cls_pred, mask, inst in zip(cls_preds, pred_masks, insts):
            n_queries = cls_pred.shape[0]
            n_classes = cls_pred.shape[1] - 1
            # Default: all queries predict "no object"
            cls_target = cls_pred.new_full(
                (n_queries,), n_classes, dtype=torch.long)

            if len(inst) > 0:
                # (n_queries, n_gts): mean mask logit per GT instance
                sp_masks = inst.sp_masks.float()  # (n_gts, n_superpoints)
                # For each query, compute its average activation on each GT
                # instance's superpoints
                query_inst_affinity = torch.mm(
                    mask, sp_masks.T  # (n_queries, n_superpoints) @ (n_superpoints, n_gts)
                )  # (n_queries, n_gts)
                # Normalize by instance size
                inst_sizes = sp_masks.sum(dim=1).clamp(min=1)  # (n_gts,)
                query_inst_affinity = query_inst_affinity / inst_sizes.unsqueeze(0)

                # Each query claims its best GT instance (if affinity > 0)
                best_affinity, best_gt = query_inst_affinity.max(dim=1)
                matched = best_affinity > 0
                cls_target[matched] = inst.labels_3d[best_gt[matched]]

            cls_losses.append(F.cross_entropy(
                cls_pred, cls_target,
                cls_pred.new_tensor(self.class_weight)))

        return torch.mean(torch.stack(cls_losses))

    def get_layer_loss(self, aux_outputs, insts):
        """Auxiliary (intermediate decoder layer) loss.

        Args:
            aux_outputs: Dict with cls_preds, scores, masks.
            insts: List of InstanceData_.

        Returns:
            Scalar loss.
        """
        cls_loss = self._get_cls_loss(
            aux_outputs['cls_preds'], aux_outputs['masks'], insts)
        influencer_losses = self._get_influencer_loss(
            aux_outputs['masks'], insts)

        loss = (
            self.loss_weight[0] * cls_loss
            + self.loss_weight[1] * influencer_losses['loss']
        )
        return loss

    def __call__(self, pred, insts):
        """Loss main function.

        Matches OneFormer3D's InstanceCriterion.__call__ interface exactly.

        Args:
            pred: Dict with:
                - cls_preds: List of (n_queries, n_classes+1) per batch element
                - scores: List of (n_queries, 1) per batch element
                - masks: List of (n_queries, n_superpoints) per batch element
                - aux_outputs (optional): List of dicts with same keys
            insts: List of InstanceData_ with sp_masks, labels_3d.

        Returns:
            Dict: {'inst_loss': scalar} — same key as InstanceCriterion.
        """
        # Main loss
        cls_loss = self._get_cls_loss(
            pred['cls_preds'], pred['masks'], insts)
        influencer_losses = self._get_influencer_loss(pred['masks'], insts)

        loss = (
            self.loss_weight[0] * cls_loss
            + self.loss_weight[1] * influencer_losses['loss']
        )

        # Auxiliary losses (deep supervision)
        if 'aux_outputs' in pred:
            for aux_outputs in pred['aux_outputs']:
                loss += self.get_layer_loss(aux_outputs, insts)

        return {'inst_loss': loss}


# Register with mmdet3d if available (for config-based instantiation)
if HAS_MMDET3D:
    MODELS.register_module()(InfluencerCriterion)
