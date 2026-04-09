"""MNIST Point Set Reconstruction: the right benchmark.

Each MNIST digit image is converted to a set of (x,y) pixel coordinates.
The model must reconstruct this unordered set from a latent representation.

This tests MATCHING quality: each predicted point should correspond to a
specific pixel in the digit. Duplication (predicting the same pixel twice)
means another pixel is missing.

At N=100 points per digit:
- Hungarian: O(100³) = 1M ops per sample
- Power-SoftMin: O(100²) = 10K ops per sample (100× faster)

Usage:
    python examples/run_mnist_sets.py
"""

import gzip
import os
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from pytorch_lightning.loggers import WandbLogger
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from influencerformer.losses import (
    ChamferLoss,
    HungarianLoss,
    PowerSoftMinLoss,
    ProductWeightedSoftMinLoss,
    SoftMinChamferLoss,
)
from influencerformer.training import SetPredictionModule


# ── MNIST download (no torchvision needed) ────────────────────────
def download_mnist(root="./data/mnist"):
    """Download MNIST images and labels without torchvision."""
    os.makedirs(root, exist_ok=True)
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train-images-idx3-ubyte.gz": "train_images",
        "train-labels-idx1-ubyte.gz": "train_labels",
        "t10k-images-idx3-ubyte.gz": "test_images",
        "t10k-labels-idx1-ubyte.gz": "test_labels",
    }
    for fname, _ in files.items():
        fpath = os.path.join(root, fname)
        if not os.path.exists(fpath):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(base_url + fname, fpath)

    def read_images(path):
        with gzip.open(path, "rb") as f:
            magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)

    def read_labels(path):
        with gzip.open(path, "rb") as f:
            magic, n = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    train_images = read_images(os.path.join(root, "train-images-idx3-ubyte.gz"))
    test_images = read_images(os.path.join(root, "t10k-images-idx3-ubyte.gz"))
    return train_images, test_images


# ── Dataset ───────────────────────────────────────────────────────
class MNISTPointSetDataset(Dataset):
    """Convert MNIST images to fixed-size sets of (x,y) pixel coordinates.

    Each digit image → set of N_POINTS coordinates where pixels are active.
    If fewer pixels than N_POINTS: pad with random existing points.
    If more: subsample.
    """

    def __init__(self, images: np.ndarray, n_points=100, threshold=80):
        self.images = images  # (N, 28, 28) uint8
        self.n_points = n_points
        self.threshold = threshold

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (28, 28) uint8

        # Extract (row, col) coordinates of active pixels
        rows, cols = np.where(img > self.threshold)
        coords = torch.tensor(np.stack([rows, cols], axis=1), dtype=torch.float32)

        # Normalize to [0, 1]
        coords = coords / 27.0

        K = coords.shape[0]
        if K == 0:
            coords = torch.rand(self.n_points, 2)
        elif K >= self.n_points:
            idx_perm = torch.randperm(K)[:self.n_points]
            coords = coords[idx_perm]
        else:
            pad_idx = torch.randint(0, K, (self.n_points - K,))
            coords = torch.cat([coords, coords[pad_idx]], dim=0)

        # Add small noise for input (denoising task)
        noisy_coords = coords + 0.02 * torch.randn_like(coords)
        noisy_coords = noisy_coords.clamp(0, 1)

        return noisy_coords.flatten(), coords  # (N*2,), (N, 2)


# ── Model ─────────────────────────────────────────────────────────
class PointSetAutoencoder(nn.Module):
    """MLP autoencoder for point sets.

    Encoder: flattened set → hidden → latent
    Decoder: latent → hidden → flattened set → reshape to (N, 2)
    """

    def __init__(self, n_points=100, dim=2, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.n_points = n_points
        self.dim = dim
        in_dim = n_points * dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
            nn.Sigmoid(),  # output in [0, 1] matching normalized coords
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out.reshape(-1, self.n_points, self.dim)


# ── Main ──────────────────────────────────────────────────────────
def main():
    N_POINTS = 100
    BATCH_SIZE = 256
    MAX_EPOCHS = 100
    NUM_SEEDS = 5

    pl.seed_everything(42)

    # Data
    train_images, test_images = download_mnist()
    # Full MNIST on GPU
    train_ds = MNISTPointSetDataset(train_images, n_points=N_POINTS)
    val_ds = MNISTPointSetDataset(test_images, n_points=N_POINTS)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, N_points: {N_POINTS}")
    sample_input, sample_target = train_ds[0]
    print(f"Input shape: {sample_input.shape}, Target shape: {sample_target.shape}")

    # Temperature calibration: coords in [0,1], typical distances ~0.1-0.3
    # So τ=0.05 is appropriate
    tau = 0.05

    losses = {
        "chamfer": ChamferLoss(),
        "hungarian": HungarianLoss(),
        "power_sm_3": PowerSoftMinLoss(temperature=tau, power=3.0),
        "power_sm_2": PowerSoftMinLoss(temperature=tau, power=2.0),
        "pw_softmin": ProductWeightedSoftMinLoss(temperature=tau),
        "softmin": SoftMinChamferLoss(temperature=tau),
    }

    results = {}

    for loss_name, loss_fn in losses.items():
        print(f"\n{'='*50}")
        print(f"  {loss_name.upper()}")
        print(f"{'='*50}")

        seed_results = []
        for seed in range(NUM_SEEDS):
            pl.seed_everything(42 + seed * 100)

            model = PointSetAutoencoder(n_points=N_POINTS, latent_dim=64, hidden_dim=256)
            module = SetPredictionModule(
                model=model,
                loss_fn=loss_fn,
                lr=1e-3,
                match_threshold=0.05,  # ~1.4 pixel distance on 28x28 grid
            )

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4)

            logger = WandbLogger(
                project="influencerformer-benchmarks",
                group="mnist",
                name=f"mnist_{loss_name}_seed{seed}",
                tags=["mnist", loss_name, f"seed{seed}", f"N{N_POINTS}"],
            )

            trainer = pl.Trainer(
                max_epochs=MAX_EPOCHS,
                accelerator="auto",
                enable_checkpointing=False,
                enable_model_summary=False,
                logger=logger,
                enable_progress_bar=True,
            )

            trainer.fit(module, train_loader, val_loader)

            # Get final metrics
            metrics = trainer.callback_metrics
            final = {
                "val_matched_dist": metrics.get("val_matched_dist", torch.tensor(float("inf"))).item(),
                "val_match_acc": metrics.get("val_match_acc", torch.tensor(0.0)).item(),
                "val_dup_rate": metrics.get("val_dup_rate", torch.tensor(0.0)).item(),
            }
            seed_results.append(final)
            print(f"  [{loss_name}] seed {seed}: "
                  f"dist={final['val_matched_dist']:.4f} "
                  f"acc={final['val_match_acc']:.3f} "
                  f"dup={final['val_dup_rate']:.3f}")
            wandb.finish()

        results[loss_name] = seed_results

    # Summary
    print(f"\n{'='*70}")
    print(f"MNIST Point Set Results (N={N_POINTS}, {MAX_EPOCHS} epochs, {NUM_SEEDS} seeds)")
    print(f"{'='*70}")
    print(f"{'Loss':<18} {'Matched Dist':>14} {'Match Acc':>12} {'Dup Rate':>12}")
    print("-" * 58)

    sorted_names = sorted(results, key=lambda n: np.mean([r["val_matched_dist"] for r in results[n]]))
    for name in sorted_names:
        runs = results[name]
        md = [r["val_matched_dist"] for r in runs]
        ma = [r["val_match_acc"] for r in runs]
        dr = [r["val_dup_rate"] for r in runs]
        print(f"{name:<18} {np.mean(md):.4f}±{np.std(md):.4f}  "
              f"{np.mean(ma):.3f}±{np.std(ma):.3f}  "
              f"{np.mean(dr):.4f}±{np.std(dr):.4f}")


if __name__ == "__main__":
    main()
