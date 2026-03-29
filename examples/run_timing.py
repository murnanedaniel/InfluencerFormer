"""Timing scaling test: measure per-step cost of each loss at N=10,20,50,100.

Runs 3 epochs of training on synthetic data at each N, reports per-step
breakdown: model forward, distance matrix, loss forward, backward.
"""

import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from influencerformer.losses import (
    ChamferLoss,
    HungarianLoss,
    PowerSoftMinLoss,
    ProductWeightedSoftMinLoss,
    SoftMinChamferLoss,
)
from influencerformer.training import SetPredictionModule


class SetPredictor(nn.Module):
    def __init__(self, n_points, dim=2, hidden=128):
        super().__init__()
        self.n_points, self.dim = n_points, dim
        self.net = nn.Sequential(
            nn.Linear(n_points * dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_points * dim),
        )

    def forward(self, x):
        return self.net(x).reshape(-1, self.n_points, self.dim)


def run_timing(loss_name, loss_fn, n_points, n_epochs=3, n_samples=1000, batch_size=64):
    pl.seed_everything(42)

    targets = torch.randn(n_samples, n_points, 2)
    inputs = (targets + torch.randn_like(targets)).flatten(-2)
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = SetPredictor(n_points, hidden=128)

    # Calibrate temperature to data scale
    tau = 0.1 * (n_points / 10) ** (-0.5)  # rough scaling
    if hasattr(loss_fn, "temperature"):
        loss_fn.temperature = tau

    module = SetPredictionModule(
        model=model, loss_fn=loss_fn, lr=1e-3, match_threshold=0.3,
    )

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator="cpu",
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, loader)

    return module.get_timing_summary()


def main():
    n_values = [10, 20, 50, 100]

    losses = {
        "chamfer": lambda: ChamferLoss(),
        "softmin": lambda: SoftMinChamferLoss(temperature=0.1),
        "power_sm_3": lambda: PowerSoftMinLoss(temperature=0.1, power=3.0),
        "pw_softmin": lambda: ProductWeightedSoftMinLoss(temperature=0.1),
        "hungarian": lambda: HungarianLoss(),
    }

    print(f"{'Loss':<16} {'N':>5} {'Model':>8} {'Dist':>8} {'Loss':>8} {'Bwd':>8} {'Total':>8} {'Steps':>6}")
    print("-" * 75)

    # Collect for summary table
    all_results = {}

    for loss_name, make_fn in losses.items():
        for n in n_values:
            t = run_timing(loss_name, make_fn(), n, n_epochs=3, n_samples=500, batch_size=32)
            print(f"{loss_name:<16} {n:>5} {t['model_ms']:>7.2f} {t['dist_ms']:>7.2f} "
                  f"{t['loss_ms']:>7.2f} {t['backward_ms']:>7.2f} {t['total_ms']:>7.2f} {t['n_steps']:>5.0f}")
            all_results[(loss_name, n)] = t

    # Summary: speedup vs Hungarian at each N
    print(f"\n{'='*60}")
    print("Speedup vs Hungarian (total step time)")
    print(f"{'='*60}")
    print(f"{'Loss':<16} " + "".join(f"{'N='+str(n):>10}" for n in n_values))
    print("-" * 56)
    for loss_name in losses:
        row = f"{loss_name:<16} "
        for n in n_values:
            hung_t = all_results[("hungarian", n)]["total_ms"] + all_results[("hungarian", n)]["backward_ms"]
            this_t = all_results[(loss_name, n)]["total_ms"] + all_results[(loss_name, n)]["backward_ms"]
            speedup = hung_t / max(this_t, 0.001)
            row += f"{speedup:>9.1f}×"
        print(row)


if __name__ == "__main__":
    main()
