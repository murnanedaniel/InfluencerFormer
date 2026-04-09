"""CLEVR State Set Prediction: variable-cardinality benchmark.

Predicts a set of object state vectors from CLEVR scenes.
Each object is an 18D vector (DSPN format):
  - 3D coordinates (x, y, z) normalized by /3
  - Material: 2-dim one-hot (rubber, metal)
  - Color: 8-dim one-hot (gray, red, blue, green, brown, purple, cyan, yellow)
  - Shape: 3-dim one-hot (sphere, cube, cylinder)
  - Size: 2-dim one-hot (large, small)

Variable cardinality: 3-10 objects per scene, padded to 10 with masks.
This tests matching with variable set sizes — a harder problem than fixed-N.

Reference: Zhang et al., "Deep Set Prediction Networks", NeurIPS 2019
  (github.com/Cyanogenoid/dspn)

Usage:
    python examples/run_clevr_boxes.py
"""

import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from pytorch_lightning.loggers import WandbLogger
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


# ── CLEVR property encodings (DSPN-compatible) ──────────────────
MATERIALS = ["rubber", "metal"]
COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
SHAPES = ["sphere", "cube", "cylinder"]
SIZES = ["large", "small"]

OBJ_DIM = 3 + len(MATERIALS) + len(COLORS) + len(SHAPES) + len(SIZES)  # 18


def encode_object(obj):
    """Encode a CLEVR object dict into an 18D state vector."""
    vec = []

    # 3D coordinates normalized to ~[-1, 1]
    x, y, z = obj["3d_coords"]
    vec.extend([x / 3.0, y / 3.0, z / 3.0])

    # Material one-hot
    mat = [0.0] * len(MATERIALS)
    if obj.get("material") in MATERIALS:
        mat[MATERIALS.index(obj["material"])] = 1.0
    vec.extend(mat)

    # Color one-hot
    col = [0.0] * len(COLORS)
    if obj.get("color") in COLORS:
        col[COLORS.index(obj["color"])] = 1.0
    vec.extend(col)

    # Shape one-hot
    shp = [0.0] * len(SHAPES)
    if obj.get("shape") in SHAPES:
        shp[SHAPES.index(obj["shape"])] = 1.0
    vec.extend(shp)

    # Size one-hot
    sz = [0.0] * len(SIZES)
    if obj.get("size") in SIZES:
        sz[SIZES.index(obj["size"])] = 1.0
    vec.extend(sz)

    return vec


# ── Dataset ───────────────────────────────────────────────────────
class CLEVRStateDataset(Dataset):
    """CLEVR object state dataset with proper masking.

    Each sample: a padded set of 18D object state vectors + binary mask.
    Returns 3-tuple: (input_flat, targets, mask).
    """

    def __init__(self, scenes_path, max_objects=10, max_samples=None):
        with open(scenes_path) as f:
            raw = json.load(f)

        scenes = raw["scenes"]
        if max_samples:
            scenes = scenes[:max_samples]

        self.states = []
        self.masks = []
        self.max_objects = max_objects

        for scene in scenes:
            objects = scene["objects"]
            n_obj = min(len(objects), max_objects)

            state = torch.zeros(max_objects, OBJ_DIM)
            mask = torch.zeros(max_objects)

            for i in range(n_obj):
                state[i] = torch.tensor(encode_object(objects[i]))
                mask[i] = 1.0

            self.states.append(state)
            self.masks.append(mask)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]    # (max_objects, OBJ_DIM)
        mask = self.masks[idx]      # (max_objects,)

        # Autoencoder mode: input = noisy states, target = clean states
        # Only add noise to real objects (masked)
        noise = 0.05 * torch.randn_like(state) * mask.unsqueeze(1)
        noisy_state = state + noise

        return noisy_state.flatten(), state, mask


# ── Model ─────────────────────────────────────────────────────────
class StateSetAutoencoder(nn.Module):
    """MLP autoencoder for CLEVR state sets (18D per object)."""

    def __init__(self, max_objects=10, obj_dim=OBJ_DIM, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.max_objects = max_objects
        self.obj_dim = obj_dim
        in_dim = max_objects * obj_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
            # No Sigmoid — 3D coords can be negative, one-hots are unconstrained logits
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out.reshape(-1, self.max_objects, self.obj_dim)


# ── Data download ─────────────────────────────────────────────────
def download_clevr_scenes(data_dir="data/clevr"):
    """Download CLEVR v1.0 scenes JSON if not present."""
    scenes_dir = os.path.join(data_dir, "scenes")
    train_path = os.path.join(scenes_dir, "CLEVR_train_scenes.json")
    val_path = os.path.join(scenes_dir, "CLEVR_val_scenes.json")

    if os.path.exists(train_path) and os.path.exists(val_path):
        return train_path, val_path

    # Try to download
    zip_path = os.path.join(data_dir, "CLEVR_v1.0.zip")
    if not os.path.exists(zip_path):
        print("CLEVR scenes not found. Downloading CLEVR v1.0 (~18GB)...")
        print("(Only the scenes JSON files will be extracted)")
        import urllib.request
        os.makedirs(data_dir, exist_ok=True)
        url = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
        urllib.request.urlretrieve(url, zip_path)

    # Extract only scenes/ from the zip
    print("Extracting scenes from CLEVR zip...")
    os.makedirs(scenes_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if "scenes/" in name and name.endswith(".json"):
                # Extract to data/clevr/scenes/
                basename = os.path.basename(name)
                if basename:
                    with zf.open(name) as src, open(os.path.join(scenes_dir, basename), "wb") as dst:
                        dst.write(src.read())
                    print(f"  Extracted {basename}")

    if os.path.exists(train_path) and os.path.exists(val_path):
        return train_path, val_path

    raise FileNotFoundError(
        f"Could not find CLEVR scenes at {scenes_dir}. "
        "Please download CLEVR v1.0 manually and place scenes JSON files there."
    )


def generate_synthetic_scenes(data_dir="data/clevr", n_scenes=20000):
    """Generate synthetic CLEVR-like scenes for testing."""
    os.makedirs(os.path.join(data_dir, "scenes"), exist_ok=True)
    scenes = {"scenes": []}

    for _ in range(n_scenes):
        n_obj = np.random.randint(3, 11)
        objects = []
        for _ in range(n_obj):
            objects.append({
                "3d_coords": [
                    float(np.random.uniform(-3, 3)),
                    float(np.random.uniform(-3, 3)),
                    float(np.random.uniform(0.5, 5)),
                ],
                "material": np.random.choice(MATERIALS),
                "color": np.random.choice(COLORS),
                "shape": np.random.choice(SHAPES),
                "size": np.random.choice(SIZES),
            })
        scenes["scenes"].append({"objects": objects})

    path = os.path.join(data_dir, "scenes", "synthetic_scenes.json")
    with open(path, "w") as f:
        json.dump(scenes, f)
    print(f"Created {n_scenes} synthetic CLEVR scenes at {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────
def main():
    MAX_OBJECTS = 10
    BATCH_SIZE = 256
    MAX_EPOCHS = 50
    NUM_SEEDS = 3
    MAX_TRAIN = 20000  # Cap training size for speed (full CLEVR is 70K)
    MAX_VAL = 5000

    pl.seed_everything(42)

    # Use real CLEVR if available, otherwise synthetic
    clevr_train = "data/clevr/scenes/CLEVR_train_scenes.json"
    clevr_val = "data/clevr/scenes/CLEVR_val_scenes.json"

    if os.path.exists(clevr_train) and os.path.exists(clevr_val):
        print(f"Using real CLEVR scenes: {clevr_train}")
        train_ds = CLEVRStateDataset(clevr_train, max_objects=MAX_OBJECTS, max_samples=MAX_TRAIN)
        val_ds = CLEVRStateDataset(clevr_val, max_objects=MAX_OBJECTS, max_samples=MAX_VAL)
    else:
        print("Real CLEVR not found, generating synthetic scenes...")
        print("(To use real CLEVR: download CLEVR_v1.0.zip and extract scenes/ to data/clevr/scenes/)")
        synth_path = generate_synthetic_scenes(n_scenes=20000)
        full_ds = CLEVRStateDataset(synth_path, max_objects=MAX_OBJECTS)
        n_train = int(0.8 * len(full_ds))
        n_val = len(full_ds) - n_train
        train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, "
          f"Max objects: {MAX_OBJECTS}, Obj dim: {OBJ_DIM}")

    # Sanity check: inspect a sample
    sample = train_ds[0]
    print(f"Sample shapes: input={sample[0].shape}, target={sample[1].shape}, mask={sample[2].shape}")
    n_real = int(sample[2].sum().item())
    print(f"  Real objects: {n_real}, Padding: {MAX_OBJECTS - n_real}")

    # Temperature: 18D states with coords in [-1,1] and one-hots in {0,1}
    # Typical L2 distances ~1.0-3.0, so tau=0.5 is appropriate
    tau = 0.5

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

            model = StateSetAutoencoder(
                max_objects=MAX_OBJECTS, latent_dim=64, hidden_dim=256,
            )
            module = SetPredictionModule(
                model=model, loss_fn=loss_fn, lr=1e-3,
                match_threshold=0.5,  # ~L2 distance of one wrong one-hot
            )

            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
            )
            val_loader = DataLoader(
                val_ds, batch_size=BATCH_SIZE, num_workers=4,
            )

            logger = WandbLogger(
                project="influencerformer-benchmarks",
                group="clevr",
                name=f"clevr_{loss_name}_seed{seed}",
                tags=["clevr", loss_name, f"seed{seed}", f"N{MAX_OBJECTS}"],
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
    print(f"CLEVR State Results (max_objects={MAX_OBJECTS}, {MAX_EPOCHS} epochs, {NUM_SEEDS} seeds)")
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
