"""Toy point cloud instance segmentation on S3DIS.

A minimal example demonstrating the condensation/embedding approach
on the S3DIS point cloud benchmark. Uses a simple MLP backbone that
maps each point's (x, y, z, r, g, b) features to an embedding space
where same-instance points cluster together.

Usage:
    pip install torch torch-geometric
    python examples/toy_pointcloud.py

This will:
1. Auto-download S3DIS via PyTorch Geometric (~2 GB on first run)
2. Train a simple MLP-based embedding model with the Influencer Loss
3. Evaluate instance segmentation quality on the test set
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from influencerformer.losses.influencer_loss import InfluencerLoss


# ---------------------------------------------------------------------------
# Simple MLP backbone for point cloud embedding
# ---------------------------------------------------------------------------
class PointEmbeddingMLP(nn.Module):
    """Maps per-point features to embedding space + condensation weight.

    Input: (N, 6) — [x, y, z, r, g, b]
    Output: embeddings (N, embed_dim), betas (N,)
    """

    def __init__(self, input_dim=6, hidden_dim=64, embed_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.embed_head = nn.Linear(hidden_dim, embed_dim)
        self.beta_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (N, 6) point features [xyz, rgb]
        Returns:
            embeddings: (N, embed_dim)
            betas: (N,)
        """
        h = self.encoder(x)
        embeddings = self.embed_head(h)
        betas = torch.sigmoid(self.beta_head(h)).squeeze(-1)
        return embeddings, betas


# ---------------------------------------------------------------------------
# Greedy clustering for inference
# ---------------------------------------------------------------------------
def cluster_embeddings(embeddings, betas, beta_threshold=0.5, dist_threshold=0.5):
    """Greedily cluster points by proximity to high-beta influencer points.

    Args:
        embeddings: (N, D) latent coordinates.
        betas: (N,) condensation weights.
        beta_threshold: Minimum beta to be considered an influencer candidate.
        dist_threshold: Maximum distance to assign a point to an influencer.

    Returns:
        labels: (N,) predicted instance IDs (0 = unassigned).
    """
    N = embeddings.shape[0]
    labels = torch.zeros(N, dtype=torch.long, device=embeddings.device)

    # Sort points by descending beta
    order = torch.argsort(betas, descending=True)

    current_label = 1
    for idx in order:
        if betas[idx] < beta_threshold:
            break
        if labels[idx] > 0:
            continue  # already assigned

        # This point becomes an influencer
        dists = torch.norm(embeddings - embeddings[idx].unsqueeze(0), dim=1)
        nearby = (dists < dist_threshold) & (labels == 0)
        labels[nearby] = current_label
        current_label += 1

    return labels


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
def compute_instance_metrics(pred_labels, true_labels):
    """Compute simple instance segmentation metrics.

    Returns precision, recall, and F1 based on IoU matching.
    """
    pred_instances = pred_labels.unique()
    pred_instances = pred_instances[pred_instances > 0]
    true_instances = true_labels.unique()
    true_instances = true_instances[true_instances > 0]

    if len(true_instances) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if len(pred_instances) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Compute IoU matrix between predicted and true instances
    iou_matrix = torch.zeros(len(pred_instances), len(true_instances))
    for i, pi in enumerate(pred_instances):
        pred_mask = pred_labels == pi
        for j, ti in enumerate(true_instances):
            true_mask = true_labels == ti
            intersection = (pred_mask & true_mask).sum().float()
            union = (pred_mask | true_mask).sum().float()
            iou_matrix[i, j] = intersection / union if union > 0 else 0

    # Match at IoU > 0.5
    matched_true = set()
    matched_pred = set()
    # Greedy matching by descending IoU
    while True:
        if iou_matrix.numel() == 0:
            break
        max_val = iou_matrix.max()
        if max_val < 0.5:
            break
        max_idx = (iou_matrix == max_val).nonzero(as_tuple=False)[0]
        pi, ti = max_idx[0].item(), max_idx[1].item()
        matched_pred.add(pi)
        matched_true.add(ti)
        iou_matrix[pi, :] = 0
        iou_matrix[:, ti] = 0

    tp = len(matched_pred)
    precision = tp / len(pred_instances) if len(pred_instances) > 0 else 0
    recall = tp / len(true_instances) if len(true_instances) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Data ----
    print("\n=== Loading S3DIS dataset ===")
    try:
        from influencerformer.data.s3dis import S3DISInstanceDataset
    except ImportError:
        from data.s3dis import S3DISInstanceDataset

    train_ds = S3DISInstanceDataset(
        root="./data/s3dis", test_area=5, train=True, max_points=4096
    )
    test_ds = S3DISInstanceDataset(
        root="./data/s3dis", test_area=5, train=False, max_points=4096
    )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} rooms, Test: {len(test_ds)} rooms")

    # ---- Model ----
    model = PointEmbeddingMLP(input_dim=6, hidden_dim=64, embed_dim=8).to(device)
    criterion = InfluencerLoss(attr_weight=1.0, rep_weight=1.0, beta_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---- Training ----
    num_epochs = 5
    print(f"\n=== Training for {num_epochs} epochs ===")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            coords = batch["coords"].squeeze(0).to(device)  # (N, 3)
            colors = batch["colors"].squeeze(0).to(device)  # (N, 3)
            instance_labels = batch["instance_labels"].squeeze(0).to(device)  # (N,)

            # Combine coords and colors as input features
            features = torch.cat([coords, colors], dim=1)  # (N, 6)

            # Normalize features
            features = (features - features.mean(0)) / (features.std(0) + 1e-6)

            embeddings, betas = model(features)
            losses = criterion(embeddings, betas, instance_labels)

            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()

            epoch_loss += losses["loss"].item()
            num_batches += 1

            if num_batches % 50 == 0:
                print(
                    f"  [{num_batches}/{len(train_loader)}] "
                    f"loss={losses['loss']:.4f} "
                    f"attr={losses['attractive']:.4f} "
                    f"rep={losses['repulsive']:.4f} "
                    f"beta={losses['beta']:.4f}"
                )

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{num_epochs}: avg_loss={avg_loss:.4f}")

    # ---- Evaluation ----
    print("\n=== Evaluating on test set ===")
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            coords = batch["coords"].squeeze(0).to(device)
            colors = batch["colors"].squeeze(0).to(device)
            instance_labels = batch["instance_labels"].squeeze(0).to(device)

            features = torch.cat([coords, colors], dim=1)
            features = (features - features.mean(0)) / (features.std(0) + 1e-6)

            embeddings, betas = model(features)
            pred_labels = cluster_embeddings(embeddings, betas)

            metrics = compute_instance_metrics(pred_labels, instance_labels)
            all_metrics.append(metrics)

            if i < 5:
                n_pred = pred_labels.unique().numel() - (1 if 0 in pred_labels else 0)
                n_true = instance_labels.unique().numel() - (1 if 0 in instance_labels else 0)
                print(
                    f"  Room {i}: pred={n_pred} instances, true={n_true} instances, "
                    f"F1={metrics['f1']:.3f}"
                )

    avg_precision = np.mean([m["precision"] for m in all_metrics])
    avg_recall = np.mean([m["recall"] for m in all_metrics])
    avg_f1 = np.mean([m["f1"] for m in all_metrics])
    print(f"\nTest results: precision={avg_precision:.3f}, recall={avg_recall:.3f}, F1={avg_f1:.3f}")
    print("\nNote: This is a toy MLP model. Real performance requires GNN/transformer backbones.")


if __name__ == "__main__":
    main()
