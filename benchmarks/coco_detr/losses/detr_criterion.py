"""Hungarian and PM3 loss functions on a fixed DETR cost matrix."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from center format to corners."""
    x_c, y_c, w, h = boxes.unbind(-1)
    return torch.stack((x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h), dim=-1)


@dataclass
class LossParts:
    """Container for loss components."""

    loss: torch.Tensor
    loss_ce: torch.Tensor
    loss_bbox: torch.Tensor
    loss_giou: torch.Tensor
    loss_background: torch.Tensor


class DetrCriterion(nn.Module):
    """Compute Hungarian or PM3 losses on DETR outputs."""

    def __init__(self, matcher_cfg):
        super().__init__()
        self.cfg = matcher_cfg

    def _pairwise_terms(self, logits: torch.Tensor, boxes: torch.Tensor, target: dict):
        target_labels = target['class_labels']
        target_boxes = target['boxes']
        probs = logits.softmax(dim=-1)
        log_probs = logits.log_softmax(dim=-1)
        class_cost = -probs[:, target_labels]
        class_ce = -log_probs[:, target_labels]
        bbox_l1 = torch.cdist(boxes, target_boxes, p=1)
        giou = generalized_box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(target_boxes))
        giou_cost = 1.0 - giou
        match_cost = (
            self.cfg.classification_cost * class_cost
            + self.cfg.bbox_l1_cost * bbox_l1
            + self.cfg.giou_cost * giou_cost
        )
        return match_cost, class_ce, bbox_l1, giou_cost, target_labels, target_boxes

    def _hungarian_loss(self, logits: torch.Tensor, boxes: torch.Tensor, target: dict) -> LossParts:
        match_cost, _, _, _, target_labels, target_boxes = self._pairwise_terms(logits, boxes, target)
        no_object = logits.shape[-1] - 1
        class_targets = torch.full((logits.shape[0],), no_object, dtype=torch.long, device=logits.device)
        class_weights = torch.ones(logits.shape[-1], device=logits.device)
        class_weights[-1] = self.cfg.eos_coefficient
        if target_labels.numel() == 0:
            loss_ce = F.cross_entropy(logits, class_targets, weight=class_weights)
            zero = logits.new_tensor(0.0)
            return LossParts(loss_ce, loss_ce, zero, zero, zero)

        pred_idx, tgt_idx = linear_sum_assignment(match_cost.detach().cpu().numpy())
        pred_idx = torch.as_tensor(pred_idx, device=logits.device, dtype=torch.long)
        tgt_idx = torch.as_tensor(tgt_idx, device=logits.device, dtype=torch.long)
        class_targets[pred_idx] = target_labels[tgt_idx]
        loss_ce = F.cross_entropy(logits, class_targets, weight=class_weights)
        matched_boxes = boxes[pred_idx]
        matched_targets = target_boxes[tgt_idx]
        loss_bbox = F.l1_loss(matched_boxes, matched_targets, reduction='none').sum(dim=-1).mean()
        matched_giou = generalized_box_iou(
            box_cxcywh_to_xyxy(matched_boxes),
            box_cxcywh_to_xyxy(matched_targets),
        )
        loss_giou = (1.0 - torch.diag(matched_giou)).mean()
        total = loss_ce + self.cfg.bbox_l1_cost * loss_bbox + self.cfg.giou_cost * loss_giou
        zero = logits.new_tensor(0.0)
        return LossParts(total, loss_ce, loss_bbox, loss_giou, zero)

    def _pm3_loss(self, logits: torch.Tensor, boxes: torch.Tensor, target: dict) -> LossParts:
        match_cost, class_ce, bbox_l1, giou_cost, target_labels, _ = self._pairwise_terms(logits, boxes, target)
        no_object = logits.shape[-1] - 1
        if target_labels.numel() == 0:
            bg_targets = torch.full((logits.shape[0],), no_object, dtype=torch.long, device=logits.device)
            bg_loss = F.cross_entropy(logits, bg_targets)
            zero = logits.new_tensor(0.0)
            return LossParts(self.cfg.eos_coefficient * bg_loss, zero, zero, zero, bg_loss)

        tau = self.cfg.pm3_temperature
        power = self.cfg.pm3_power
        coverage_weights = torch.softmax(-match_cost / tau, dim=0)
        precision_weights = torch.softmax(-match_cost / tau, dim=1)

        # Apply power per-component, then combine (matches PowerSoftMinLoss pattern)
        sm_ce_cov = (coverage_weights * class_ce).sum(dim=0)
        sm_bbox_cov = (coverage_weights * bbox_l1).sum(dim=0)
        sm_giou_cov = (coverage_weights * giou_cost).sum(dim=0)
        loss_cov = (
            self.cfg.classification_cost * sm_ce_cov.pow(power)
            + self.cfg.bbox_l1_cost * sm_bbox_cov.pow(power)
            + self.cfg.giou_cost * sm_giou_cov.pow(power)
        ).mean()

        sm_ce_prec = (precision_weights * class_ce).sum(dim=1)
        sm_bbox_prec = (precision_weights * bbox_l1).sum(dim=1)
        sm_giou_prec = (precision_weights * giou_cost).sum(dim=1)
        loss_prec = (
            self.cfg.classification_cost * sm_ce_prec.pow(power)
            + self.cfg.bbox_l1_cost * sm_bbox_prec.pow(power)
            + self.cfg.giou_cost * sm_giou_prec.pow(power)
        ).mean()
        matched_strength = precision_weights.max(dim=1).values.detach().clamp(0.0, 1.0)
        bg_targets = torch.full((logits.shape[0],), no_object, dtype=torch.long, device=logits.device)
        bg_ce = F.cross_entropy(logits, bg_targets, reduction='none')
        loss_background = ((1.0 - matched_strength) * bg_ce).mean()
        total = loss_cov + loss_prec + self.cfg.eos_coefficient * loss_background
        zero = loss_cov.new_tensor(0.0)
        return LossParts(total, loss_cov + loss_prec, zero, zero, loss_background)

    def forward(self, outputs, labels: list[dict]) -> dict:
        """Compute losses for a batch."""
        parts = []
        for logits, boxes, target in zip(outputs.logits, outputs.pred_boxes, labels):
            if self.cfg.name == 'hungarian':
                parts.append(self._hungarian_loss(logits, boxes, target))
            elif self.cfg.name == 'pm3':
                parts.append(self._pm3_loss(logits, boxes, target))
            else:
                raise ValueError(f'Unsupported matcher: {self.cfg.name}')
        return {
            'loss': torch.stack([part.loss for part in parts]).mean(),
            'loss_ce': torch.stack([part.loss_ce for part in parts]).mean(),
            'loss_bbox': torch.stack([part.loss_bbox for part in parts]).mean(),
            'loss_giou': torch.stack([part.loss_giou for part in parts]).mean(),
            'loss_background': torch.stack([part.loss_background for part in parts]).mean(),
        }
