"""Run our custom losses on ShapeNet autoencoder benchmark.

Reduced settings for CPU: 512 points (not 2048), batch_size=16, 100 epochs.
This tests whether Power-SoftMin works on real 3D point clouds.
"""

import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.shapenet_core55 import ShapeNetCore55XyzOnlyDataset
from models.pointnet import PointNetAE
from loss.custom_losses import PowerSoftMin, PureTorchChamfer, SoftMinChamfer, PWsoftmin
from loss.sw_variants import SWD

NUM_POINTS = 512
BATCH_SIZE = 16
NUM_EPOCHS = 100
LR = 1e-3
DEVICE = "cpu"
DATA_PATH = "dataset/shapenet_core55/shapenet57448xyzonly.npz"


def evaluate_reconstruction(model, dataloader, device):
    """Evaluate with pure-PyTorch Chamfer distance (no CUDA ext needed)."""
    model.eval()
    total_chamfer = 0.0
    n = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            latent = model.encode(data)
            recon = model.decode(latent)
            D = torch.cdist(data, recon)
            chamfer = D.min(1).values.mean(-1) + D.min(2).values.mean(-1)
            total_chamfer += chamfer.sum().item()
            n += data.shape[0]
    return total_chamfer / max(n, 1)


def train_one(loss_name, loss_fn, num_epochs=NUM_EPOCHS):
    print(f"\n  Training with {loss_name}...")

    dataset = ShapeNetCore55XyzOnlyDataset(DATA_PATH, num_points=NUM_POINTS, phase="train")
    # Use a subset for speed (5000 samples)
    indices = list(range(min(5000, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    val_dataset = ShapeNetCore55XyzOnlyDataset(DATA_PATH, num_points=NUM_POINTS, phase="test")
    val_indices = list(range(min(500, len(val_dataset))))
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = PointNetAE(embedding_size=256, input_channels=3, output_channels=3,
                       num_points=NUM_POINTS, normalize=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_chamfer": []}
    t0 = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        nb = 0
        for data in loader:
            data = data.to(DEVICE)
            latent = model.encode(data)
            recon = model.decode(latent)

            result = loss_fn(data, recon)
            loss = result["loss"]

            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            epoch_loss += loss.item()
            nb += 1

        avg_loss = epoch_loss / max(nb, 1)
        history["train_loss"].append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_cd = evaluate_reconstruction(model, val_loader, DEVICE)
            history["val_chamfer"].append(val_cd)

            elapsed = time.time() - t0
            print(f"    [{loss_name}] epoch {epoch+1}/{num_epochs}: "
                  f"train_loss={avg_loss:.6f} val_chamfer={val_cd:.6f} ({elapsed:.0f}s)")

    total_time = time.time() - t0
    final_cd = history["val_chamfer"][-1] if history["val_chamfer"] else float("inf")
    return {"val_chamfer": final_cd, "total_time": total_time, "history": history}


def main():
    print(f"PointSWD Benchmark: {NUM_POINTS} points, batch={BATCH_SIZE}, {NUM_EPOCHS} epochs")

    # Check model exists
    try:
        model = PointNetAE(embedding_size=256, num_points=NUM_POINTS)
        x = torch.randn(2, NUM_POINTS, 3)
        latent = model.encode(x)
        recon = model.decode(latent)
        print(f"Model OK: input {x.shape} → latent {latent.shape} → recon {recon.shape}")
    except Exception as e:
        print(f"Model error: {e}")
        return

    methods = {
        "pytorch_chamfer": PureTorchChamfer(),
        "power_sm_3": PowerSoftMin(temperature=0.01, power=3.0),
        "power_sm_2": PowerSoftMin(temperature=0.01, power=2.0),
        "softmin": SoftMinChamfer(temperature=0.01),
        "pw_softmin": PWsoftmin(temperature=0.01),
        "swd_100": SWD(num_projs=100, device=DEVICE),
    }

    results = {}
    for name, loss_fn in methods.items():
        result = train_one(name, loss_fn)
        results[name] = result

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS ({NUM_POINTS} points, {NUM_EPOCHS} epochs)")
    print(f"{'='*60}")
    print(f"{'Loss':<20} {'Val Chamfer':>12} {'Time':>10}")
    print("-" * 45)
    for name in sorted(results, key=lambda n: results[n]["val_chamfer"]):
        r = results[name]
        print(f"{name:<20} {r['val_chamfer']:.6f}   {r['total_time']:.0f}s")


if __name__ == "__main__":
    main()
