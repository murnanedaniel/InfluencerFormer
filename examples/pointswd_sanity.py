"""Quick sanity check: 5 epochs on full data + overfit on 1 sample.

Verifies all losses work on real ShapeNet data and can overfit.
"""

import sys, time
import torch
from dataset.shapenet_core55 import ShapeNetCore55XyzOnlyDataset
from models.pointnet import PointNetAE
from loss.custom_losses import PowerSoftMin, PureTorchChamfer, SoftMinChamfer, PWsoftmin
from loss.sw_variants import SWD
from torch.utils.data import DataLoader, Subset

NUM_POINTS = 512
DEVICE = "cpu"
DATA_PATH = "dataset/shapenet_core55/shapenet57448xyzonly.npz"


def run_epochs(name, loss_fn, loader, model, optimizer, n_epochs):
    losses = []
    for ep in range(n_epochs):
        model.train()
        el, nb = 0, 0
        for data in loader:
            recon = model.decode(model.encode(data))
            result = loss_fn(data, recon)
            loss = result["loss"]
            if not torch.isfinite(loss): continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            el += loss.item(); nb += 1
        avg = el / max(nb, 1)
        losses.append(avg)
    return losses


def main():
    dataset = ShapeNetCore55XyzOnlyDataset(DATA_PATH, num_points=NUM_POINTS, phase="train")

    methods = {
        "chamfer":      PureTorchChamfer(),
        "power_sm_3":   PowerSoftMin(temperature=0.01, power=3.0),
        "power_sm_2":   PowerSoftMin(temperature=0.01, power=2.0),
        "softmin":      SoftMinChamfer(temperature=0.01),
        "pw_softmin":   PWsoftmin(temperature=0.01),
        "swd_100":      SWD(num_projs=100, device=DEVICE),
    }

    # ── Test 1: 5 epochs on 500 samples ──
    print("=" * 60)
    print("TEST 1: 5 epochs on 500 samples (batch=16)")
    print("=" * 60)

    subset = Subset(dataset, list(range(500)))
    loader = DataLoader(subset, batch_size=16, shuffle=True, drop_last=True)

    for name, loss_fn in methods.items():
        model = PointNetAE(embedding_size=256, num_points=NUM_POINTS)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        t0 = time.time()
        losses = run_epochs(name, loss_fn, loader, model, opt, 5)
        dt = time.time() - t0
        print(f"  {name:>15}: epoch 1={losses[0]:.6f} → epoch 5={losses[-1]:.6f}  "
              f"({'↓ learning' if losses[-1] < losses[0] * 0.95 else '~ flat'})  {dt:.1f}s")

    # ── Test 2: Overfit on 1 sample, 200 iters ──
    print(f"\n{'='*60}")
    print("TEST 2: Overfit 1 sample, 200 iterations")
    print("=" * 60)

    single = torch.from_numpy(dataset[0]).unsqueeze(0).float()  # (1, 512, 3)

    for name, loss_fn in methods.items():
        model = PointNetAE(embedding_size=256, num_points=NUM_POINTS)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        first_loss = None
        for i in range(200):
            recon = model.decode(model.encode(single))
            result = loss_fn(single, recon)
            loss = result["loss"]
            if first_loss is None: first_loss = loss.item()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        final_loss = loss.item()

        # Evaluate with Chamfer (ground truth metric)
        model.eval()
        with torch.no_grad():
            recon = model.decode(model.encode(single))
            D = torch.cdist(single, recon)
            cd = (D.min(1).values.mean() + D.min(2).values.mean()).item()
        print(f"  {name:>15}: loss {first_loss:.6f} → {final_loss:.6f}  eval_CD={cd:.6f}")


if __name__ == "__main__":
    main()
