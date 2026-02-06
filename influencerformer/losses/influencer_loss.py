"""Influencer Loss for instance segmentation via learned condensation.

A simplified, general-purpose implementation of the Influencer Loss concept.
Points are embedded into a latent space where same-instance points cluster
together around learned "influencer" (representative) points.
"""

import torch
import torch.nn as nn


class InfluencerLoss(nn.Module):
    """Influencer Loss: condensation-based instance segmentation loss.

    For each point, the model predicts:
      - embedding: position in latent clustering space
      - beta: condensation weight in (0, 1) — high beta = likely influencer

    The loss has two components:
      1. Attractive: pulls same-instance points toward their influencer
      2. Repulsive: pushes influencers of different instances apart
    """

    def __init__(self, attr_weight=1.0, rep_weight=1.0, beta_weight=1.0, rep_margin=1.0):
        super().__init__()
        self.attr_weight = attr_weight
        self.rep_weight = rep_weight
        self.beta_weight = beta_weight
        self.rep_margin = rep_margin

    def forward(self, embeddings, betas, instance_labels):
        """
        Args:
            embeddings: (N, D) latent coordinates for each point.
            betas: (N,) condensation weights in (0, 1).
            instance_labels: (N,) integer instance IDs. 0 = noise/background.

        Returns:
            Dict with total loss and component losses.
        """
        device = embeddings.device
        unique_instances = instance_labels.unique()
        # Filter out background (label 0)
        unique_instances = unique_instances[unique_instances > 0]

        if len(unique_instances) == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return {"loss": zero, "attractive": zero, "repulsive": zero, "beta": zero}

        # For each instance, find the influencer (highest beta point)
        influencer_embeddings = []
        influencer_betas = []
        attractive_loss = torch.tensor(0.0, device=device)
        beta_loss = torch.tensor(0.0, device=device)

        for inst_id in unique_instances:
            mask = instance_labels == inst_id
            inst_embeddings = embeddings[mask]  # (M, D)
            inst_betas = betas[mask]  # (M,)

            # Soft influencer selection: weighted average by beta
            # (differentiable alternative to argmax)
            weights = torch.softmax(inst_betas * 10, dim=0)  # temperature-scaled
            influencer_emb = (weights.unsqueeze(1) * inst_embeddings).sum(dim=0)  # (D,)
            influencer_embeddings.append(influencer_emb)

            # Attractive loss: pull all instance points toward influencer
            dists = torch.norm(inst_embeddings - influencer_emb.unsqueeze(0), dim=1)  # (M,)
            q = torch.arctanh(inst_betas) ** 2 + 1  # condensation charge
            attractive_loss = attractive_loss + (q * dists**2).mean()

            # Beta loss: encourage at least one high-beta point per instance
            beta_loss = beta_loss + (1 - inst_betas.max())

            influencer_betas.append(inst_betas.max())

        attractive_loss = attractive_loss / len(unique_instances)
        beta_loss = beta_loss / len(unique_instances)

        # Repulsive loss: push influencers of different instances apart
        if len(influencer_embeddings) > 1:
            inf_embs = torch.stack(influencer_embeddings)  # (K, D)
            # Pairwise distances between influencers
            diff = inf_embs.unsqueeze(0) - inf_embs.unsqueeze(1)  # (K, K, D)
            pair_dists = torch.norm(diff, dim=2)  # (K, K)

            # Hinge loss: penalize pairs closer than margin
            K = len(influencer_embeddings)
            mask = ~torch.eye(K, dtype=torch.bool, device=device)
            repulsive_loss = torch.clamp(self.rep_margin - pair_dists[mask], min=0).mean()
        else:
            repulsive_loss = torch.tensor(0.0, device=device)

        # Background suppression: push background betas toward 0
        bg_mask = instance_labels == 0
        if bg_mask.any():
            beta_loss = beta_loss + betas[bg_mask].mean()

        total = (
            self.attr_weight * attractive_loss
            + self.rep_weight * repulsive_loss
            + self.beta_weight * beta_loss
        )

        return {
            "loss": total,
            "attractive": attractive_loss.detach(),
            "repulsive": repulsive_loss.detach(),
            "beta": beta_loss.detach(),
        }
