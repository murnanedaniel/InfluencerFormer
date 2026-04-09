"""CLEVR State Set Prediction: apples-to-apples comparison with DSPN.

Compares set prediction loss functions using set-equivariant encoders
and the DSPN gradient-descent decoder, on CLEVR object state vectors.

Matches DSPN (Zhang et al., NeurIPS 2019) experimental setup:
  - 18D state vectors (3D coords + one-hot properties)
  - Set-equivariant encoder (Set Transformer or DeepSets)
  - Smooth L1 (Huber) distance metric
  - batch_size=32, lr=3e-4, 100 epochs

Usage:
    python examples/run_clevr_dspn.py [--quick]
"""

import argparse
import json
import os
import sys
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
from influencerformer.networks import (
    DeepSetsEncoder,
    DSPNModel,
    MLPSetDecoder,
    SetAutoencoder,
    SetTransformerEncoder,
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
    x, y, z = obj["3d_coords"]
    vec.extend([x / 3.0, y / 3.0, z / 3.0])
    for cats, key in [(MATERIALS, "material"), (COLORS, "color"),
                      (SHAPES, "shape"), (SIZES, "size")]:
        oh = [0.0] * len(cats)
        if obj.get(key) in cats:
            oh[cats.index(obj[key])] = 1.0
        vec.extend(oh)
    return vec


# ── Dataset (set-format, not flattened) ──────────────────────────
class CLEVRSetDataset(Dataset):
    """CLEVR object state dataset returning set-format inputs.

    Returns 3-tuple: (noisy_state, clean_state, mask)
    where state is (max_objects, obj_dim), NOT flattened.
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
        noise = 0.05 * torch.randn_like(state) * mask.unsqueeze(1)
        noisy_state = state + noise
        return noisy_state, state, mask  # (N, D), (N, D), (N,)


# ── Model factory ────────────────────────────────────────────────
def make_model(decoder_type, encoder_type="set_transformer",
               obj_dim=OBJ_DIM, max_objects=10,
               d_model=512, latent_dim=512, hidden_dim=512,
               dspn_inner_lr=800.0, dspn_inner_steps=10):
    """Create encoder + decoder model."""

    if encoder_type == "set_transformer":
        encoder = SetTransformerEncoder(
            input_dim=obj_dim, d_model=d_model, nhead=8,
            num_layers=2, latent_dim=latent_dim,
            dim_feedforward=d_model * 2,
        )
    elif encoder_type == "deepsets":
        encoder = DeepSetsEncoder(
            input_dim=obj_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
        )
    else:
        raise ValueError(f"Unknown encoder: {encoder_type}")

    if decoder_type == "mlp":
        decoder = MLPSetDecoder(
            latent_dim=latent_dim, hidden_dim=hidden_dim,
            max_objects=max_objects, obj_dim=obj_dim,
        )
        return SetAutoencoder(encoder, decoder)
    elif decoder_type == "dspn":
        return DSPNModel(
            encoder=encoder, obj_dim=obj_dim, max_objects=max_objects,
            inner_lr=dspn_inner_lr, inner_steps=dspn_inner_steps,
        )
    else:
        raise ValueError(f"Unknown decoder: {decoder_type}")


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick test run")
    args = parser.parse_args()

    MAX_OBJECTS = 10
    NUM_SEEDS = 3

    # DSPN-matched hyperparameters
    BATCH_SIZE = 32
    LR = 3e-4
    MAX_EPOCHS = 100
    LATENT_DIM = 512
    HIDDEN_DIM = 512

    if args.quick:
        MAX_EPOCHS = 5
        NUM_SEEDS = 1
        LATENT_DIM = 128
        HIDDEN_DIM = 128

    pl.seed_everything(42)

    # Load real CLEVR data
    clevr_train = "data/clevr/scenes/CLEVR_train_scenes.json"
    clevr_val = "data/clevr/scenes/CLEVR_val_scenes.json"

    if not (os.path.exists(clevr_train) and os.path.exists(clevr_val)):
        print("ERROR: Real CLEVR scenes required for DSPN comparison.")
        print("Place CLEVR_train_scenes.json and CLEVR_val_scenes.json in data/clevr/scenes/")
        sys.exit(1)

    max_train = 5000 if args.quick else 20000
    max_val = 1000 if args.quick else 5000

    train_ds = CLEVRSetDataset(clevr_train, max_objects=MAX_OBJECTS, max_samples=max_train)
    val_ds = CLEVRSetDataset(clevr_val, max_objects=MAX_OBJECTS, max_samples=max_val)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, "
          f"Max objects: {MAX_OBJECTS}, Obj dim: {OBJ_DIM}")
    print(f"Config: batch={BATCH_SIZE}, lr={LR}, epochs={MAX_EPOCHS}, "
          f"latent={LATENT_DIM}, seeds={NUM_SEEDS}")

    # Temperature for softmin-based losses
    # L2 distances are ~1.0-3.0 range, so tau=0.5 is appropriate
    tau = 0.5

    # All experiments use: Set Transformer encoder, L2 distance.
    # Only variable: loss function (+ DSPN decoder as one extra comparison).
    losses = {
        "chamfer": ChamferLoss(),
        "hungarian": HungarianLoss(),
        "power_sm_3": PowerSoftMinLoss(temperature=tau, power=3.0),
        "power_sm_2": PowerSoftMinLoss(temperature=tau, power=2.0),
        "pw_softmin": ProductWeightedSoftMinLoss(temperature=tau),
        "softmin": SoftMinChamferLoss(temperature=tau),
    }

    experiments = []

    # ── Standard MLP decoder: one run per loss function ──
    for loss_name, loss_fn in losses.items():
        experiments.append({
            "name": f"{loss_name}",
            "decoder": "mlp", "loss_name": loss_name, "loss_fn": loss_fn,
            "group": "clevr-dspn",
        })

    # ── DSPN decoder (their method) with Chamfer (their default loss) ──
    experiments.append({
        "name": "dspn_chamfer",
        "decoder": "dspn", "loss_name": "chamfer", "loss_fn": ChamferLoss(),
        "group": "clevr-dspn",
    })

    print(f"\nTotal: {len(experiments)} configs x {NUM_SEEDS} seeds = {len(experiments) * NUM_SEEDS} runs")
    print(f"Configs: {[e['name'] for e in experiments]}")

    results = {}

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"  {exp['name'].upper()} (decoder={exp['decoder']})")
        print(f"{'='*60}")

        seed_results = []
        for seed in range(NUM_SEEDS):
            pl.seed_everything(42 + seed * 100)

            model = make_model(
                decoder_type=exp["decoder"],
                encoder_type="set_transformer",
                obj_dim=OBJ_DIM, max_objects=MAX_OBJECTS,
                latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
                d_model=LATENT_DIM,
            )

            module = SetPredictionModule(
                model=model, loss_fn=exp["loss_fn"],
                lr=LR, distance_fn="l2",
                match_threshold=0.5,
            )

            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
            )
            val_loader = DataLoader(
                val_ds, batch_size=BATCH_SIZE, num_workers=4,
            )

            logger = WandbLogger(
                project="influencerformer-benchmarks",
                group=exp["group"],
                name=f"{exp['name']}_seed{seed}",
                tags=["clevr-dspn", exp["decoder"],
                      exp["loss_name"], f"seed{seed}"],
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
            print(f"  [{exp['name']}] seed {seed}: "
                  f"dist={final['val_matched_dist']:.4f} "
                  f"acc={final['val_match_acc']:.3f} "
                  f"dup={final['val_dup_rate']:.3f}")
            wandb.finish()

        results[exp["name"]] = seed_results

    # ── Summary ──
    print(f"\n{'='*80}")
    print(f"CLEVR Set Prediction: Set Transformer + L2 ({MAX_EPOCHS} epochs, {NUM_SEEDS} seeds)")
    print(f"{'='*80}")
    print(f"{'Loss':<25} {'Matched Dist':>16} {'Match Acc':>12} {'Dup Rate':>12}")
    print("-" * 67)

    sorted_names = sorted(results, key=lambda n: np.mean([r["val_matched_dist"] for r in results[n]]))
    for name in sorted_names:
        runs = results[name]
        md = [r["val_matched_dist"] for r in runs]
        ma = [r["val_match_acc"] for r in runs]
        dr = [r["val_dup_rate"] for r in runs]
        print(f"{name:<25} {np.mean(md):.4f}±{np.std(md):.4f}  "
              f"{np.mean(ma):.3f}±{np.std(ma):.3f}  "
              f"{np.mean(dr):.4f}±{np.std(dr):.4f}")


if __name__ == "__main__":
    main()
