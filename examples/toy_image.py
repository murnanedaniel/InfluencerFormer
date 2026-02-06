"""Toy image instance segmentation on COCO.

A minimal example demonstrating the condensation/embedding approach
on the COCO image benchmark. Uses a simple CNN backbone that maps
each pixel to an embedding space where same-instance pixels cluster
together around learned influencer points.

Usage:
    pip install torch torchvision pycocotools

    # First, download COCO val2017:
    mkdir -p data/coco && cd data/coco
    wget http://images.cocodataset.org/zips/val2017.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip val2017.zip
    unzip annotations_trainval2017.zip
    cd ../..

    python examples/toy_image.py

This will:
1. Load a small subset of COCO val images
2. Train a simple pixel-embedding CNN with the Influencer Loss
3. Evaluate instance segmentation quality
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from influencerformer.losses.influencer_loss import InfluencerLoss


# ---------------------------------------------------------------------------
# Simple CNN backbone for per-pixel embeddings
# ---------------------------------------------------------------------------
class PixelEmbeddingCNN(nn.Module):
    """Maps image pixels to embedding space + condensation weight.

    A lightweight encoder-decoder that produces per-pixel outputs.
    Input: (B, 3, H, W)
    Output: embeddings (B, embed_dim, H, W), betas (B, 1, H, W)
    """

    def __init__(self, embed_dim=8):
        super().__init__()
        # Encoder (downsample 4x)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )

        # Decoder (upsample back)
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 2, stride=2),  # 64 + 64 skip
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )

        # Output heads
        self.embed_head = nn.Conv2d(64, embed_dim, 1)  # 32 + 32 skip
        self.beta_head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) image tensor
        Returns:
            embeddings: (B, embed_dim, H, W)
            betas: (B, H, W)
        """
        # Encoder
        e1 = self.enc1(x)  # (B, 32, H, W)
        e2 = self.enc2(e1)  # (B, 64, H/2, W/2)
        e3 = self.enc3(e2)  # (B, 128, H/4, W/4)

        # Decoder with skip connections
        d2 = self.dec2(e3)  # (B, 64, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)  # (B, 128, H/2, W/2)
        d1 = self.dec1(d2)  # (B, 32, H, W)
        d1 = torch.cat([d1, e1], dim=1)  # (B, 64, H, W)

        embeddings = self.embed_head(d1)  # (B, embed_dim, H, W)
        betas = torch.sigmoid(self.beta_head(d1)).squeeze(1)  # (B, H, W)

        return embeddings, betas


# ---------------------------------------------------------------------------
# Greedy pixel clustering
# ---------------------------------------------------------------------------
def cluster_pixel_embeddings(embeddings, betas, beta_threshold=0.5, dist_threshold=0.8):
    """Cluster pixels by embedding proximity to high-beta influencer pixels.

    Args:
        embeddings: (D, H, W) pixel embeddings.
        betas: (H, W) condensation weights.
        beta_threshold: Minimum beta to consider as influencer.
        dist_threshold: Maximum embedding distance for assignment.

    Returns:
        labels: (H, W) instance IDs (0 = unassigned).
    """
    D, H, W = embeddings.shape
    emb_flat = embeddings.reshape(D, -1).T  # (H*W, D)
    beta_flat = betas.reshape(-1)  # (H*W,)
    labels = torch.zeros(H * W, dtype=torch.long, device=embeddings.device)

    # Sort by descending beta
    order = torch.argsort(beta_flat, descending=True)
    current_label = 1

    for idx in order:
        if beta_flat[idx] < beta_threshold:
            break
        if labels[idx] > 0:
            continue

        dists = torch.norm(emb_flat - emb_flat[idx].unsqueeze(0), dim=1)
        nearby = (dists < dist_threshold) & (labels == 0)
        labels[nearby] = current_label
        current_label += 1

    return labels.reshape(H, W)


# ---------------------------------------------------------------------------
# Create synthetic training data from COCO masks
# ---------------------------------------------------------------------------
def masks_to_instance_map(masks):
    """Convert (K, H, W) binary masks to (H, W) instance label map.

    Overlapping pixels are assigned to the mask with smallest area
    (assumes smaller objects are in front).
    """
    K, H, W = masks.shape
    instance_map = torch.zeros(H, W, dtype=torch.long)

    if K == 0:
        return instance_map

    # Sort by area descending (paint large first, small on top)
    areas = masks.sum(dim=(1, 2))
    order = torch.argsort(areas, descending=True)

    for i, idx in enumerate(order):
        instance_map[masks[idx]] = i + 1

    return instance_map


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Data ----
    print("\n=== Loading COCO dataset ===")
    try:
        from influencerformer.data.coco import COCOInstanceDataset
    except ImportError:
        from data.coco import COCOInstanceDataset

    try:
        full_dataset = COCOInstanceDataset(root="./data/coco", split="val", max_size=256)
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nTo download COCO val2017:")
        print("  mkdir -p data/coco && cd data/coco")
        print("  wget http://images.cocodataset.org/zips/val2017.zip")
        print("  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
        print("  unzip val2017.zip && unzip annotations_trainval2017.zip")
        print("\nOr install fiftyone and use: from influencerformer.data.coco import download_coco")
        return

    # Use a small subset for the toy example
    num_train = min(200, len(full_dataset))
    num_test = min(50, len(full_dataset) - num_train)
    train_ds = Subset(full_dataset, list(range(num_train)))
    test_ds = Subset(full_dataset, list(range(num_train, num_train + num_test)))

    print(f"Using {num_train} train, {num_test} test images (subset of COCO val)")

    # ---- Model ----
    model = PixelEmbeddingCNN(embed_dim=8).to(device)
    criterion = InfluencerLoss(attr_weight=1.0, rep_weight=1.0, beta_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---- Training ----
    num_epochs = 5
    print(f"\n=== Training for {num_epochs} epochs ===")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for i in range(len(train_ds)):
            sample = train_ds[i]
            image = sample["image"].unsqueeze(0).to(device)  # (1, 3, H, W)
            masks = sample["masks"]  # (K, H, W)

            if masks.shape[0] == 0:
                continue

            instance_map = masks_to_instance_map(masks).to(device)  # (H, W)

            # Ensure image size is divisible by 4 (for encoder-decoder)
            _, _, H, W = image.shape
            H_pad = (4 - H % 4) % 4
            W_pad = (4 - W % 4) % 4
            if H_pad > 0 or W_pad > 0:
                image = F.pad(image, (0, W_pad, 0, H_pad))
                instance_map = F.pad(instance_map.unsqueeze(0).unsqueeze(0).float(),
                                     (0, W_pad, 0, H_pad)).squeeze().long()

            embeddings, betas = model(image)  # (1, D, H, W), (1, H, W)

            # Flatten spatial dims for the loss
            B, D, Hp, Wp = embeddings.shape
            emb_flat = embeddings[0].reshape(D, -1).T  # (H*W, D)
            beta_flat = betas[0].reshape(-1)  # (H*W,)
            inst_flat = instance_map.reshape(-1)  # (H*W,)

            # Subsample pixels for efficiency (full image is too many pairs)
            num_pixels = emb_flat.shape[0]
            max_sample = 2048
            if num_pixels > max_sample:
                perm = torch.randperm(num_pixels, device=device)[:max_sample]
                emb_flat = emb_flat[perm]
                beta_flat = beta_flat[perm]
                inst_flat = inst_flat[perm]

            losses = criterion(emb_flat, beta_flat, inst_flat)

            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()

            epoch_loss += losses["loss"].item()
            num_batches += 1

            if num_batches % 50 == 0:
                print(
                    f"  [{num_batches}/{len(train_ds)}] "
                    f"loss={losses['loss']:.4f} "
                    f"attr={losses['attractive']:.4f} "
                    f"rep={losses['repulsive']:.4f}"
                )

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{num_epochs}: avg_loss={avg_loss:.4f}")

    # ---- Evaluation ----
    print("\n=== Evaluating on test set ===")
    model.eval()
    results = []

    with torch.no_grad():
        for i in range(min(10, len(test_ds))):
            sample = test_ds[i]
            image = sample["image"].unsqueeze(0).to(device)
            masks = sample["masks"]

            if masks.shape[0] == 0:
                continue

            instance_map = masks_to_instance_map(masks).to(device)

            _, _, H, W = image.shape
            H_pad = (4 - H % 4) % 4
            W_pad = (4 - W % 4) % 4
            if H_pad > 0 or W_pad > 0:
                image = F.pad(image, (0, W_pad, 0, H_pad))
                instance_map = F.pad(instance_map.unsqueeze(0).unsqueeze(0).float(),
                                     (0, W_pad, 0, H_pad)).squeeze().long()

            embeddings, betas = model(image)
            pred_labels = cluster_pixel_embeddings(embeddings[0], betas[0])

            # Flatten for metric computation
            pred_flat = pred_labels.reshape(-1)
            true_flat = instance_map.reshape(-1)

            n_pred = pred_flat.unique().numel() - (1 if 0 in pred_flat else 0)
            n_true = true_flat.unique().numel() - (1 if 0 in true_flat else 0)

            print(f"  Image {i}: pred={n_pred} instances, true={n_true} instances")
            results.append({"pred": n_pred, "true": n_true})

    print(f"\nEvaluated {len(results)} images.")
    print("Note: This is a toy CNN model. Real performance requires transformer backbones (Mask2Former, etc.).")


if __name__ == "__main__":
    main()
