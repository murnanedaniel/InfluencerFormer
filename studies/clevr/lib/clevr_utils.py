"""Shared utilities for CLEVR set prediction ablation notebooks.

Import from any ablation notebook with:
    import sys; sys.path.insert(0, '.')
    import clevr_utils as cu
"""

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Path resolution: works whether notebook cwd is notebooks/ or repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from influencerformer.losses import ChamferLoss, HungarianLoss, PowerSoftMinLoss
from influencerformer.networks import (
    DSPNModel, MLPSetDecoder, SetAutoencoder, SetTransformerEncoder,
)
from influencerformer.networks.set_decoders import SlottedMLPSetDecoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MATERIALS   = ['rubber', 'metal']
COLORS      = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
SHAPES      = ['sphere', 'cube', 'cylinder']
SIZES       = ['large', 'small']
OBJ_DIM     = 3 + len(MATERIALS) + len(COLORS) + len(SHAPES) + len(SIZES)  # 18
MAX_OBJECTS = 10

_DATA_DIR   = _REPO_ROOT / 'data' / 'clevr' / 'scenes'
TRAIN_PATH  = str(_DATA_DIR / 'CLEVR_train_scenes.json')
VAL_PATH    = str(_DATA_DIR / 'CLEVR_val_scenes.json')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def encode_object(obj: dict) -> list:
    """Encode CLEVR object dict → 18D float vector."""
    x, y, z = obj['3d_coords']
    vec = [x / 3.0, y / 3.0, z / 3.0]
    for cats, key in [(MATERIALS, 'material'), (COLORS, 'color'),
                      (SHAPES, 'shape'), (SIZES, 'size')]:
        oh = [0.0] * len(cats)
        if obj.get(key) in cats:
            oh[cats.index(obj[key])] = 1.0
        vec.extend(oh)
    return vec


class CLEVRSetDataset(Dataset):
    """CLEVR set dataset.

    Returns (noisy_state, clean_state, mask) where state is (MAX_OBJECTS, OBJ_DIM).
    """

    def __init__(self, path: str, max_samples: Optional[int] = None):
        with open(path) as f:
            scenes = json.load(f)['scenes']
        if max_samples:
            scenes = scenes[:max_samples]
        self.states, self.masks = [], []
        for scene in scenes:
            objects = scene['objects'][:MAX_OBJECTS]
            state = torch.zeros(MAX_OBJECTS, OBJ_DIM)
            mask  = torch.zeros(MAX_OBJECTS)
            for i, obj in enumerate(objects):
                state[i] = torch.tensor(encode_object(obj))
                mask[i]  = 1.0
            self.states.append(state)
            self.masks.append(mask)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        mask  = self.masks[idx]
        noise = 0.05 * torch.randn_like(state) * mask.unsqueeze(1)
        return state + noise, state, mask


def make_dataloader(
    split: str = 'train',
    max_samples: Optional[int] = None,
    batch_size: int = 64,
    num_workers: int = 4,
) -> DataLoader:
    """Create a CLEVR DataLoader.

    Args:
        split: 'train' or 'val'
        max_samples: cap number of scenes (None = all)
        batch_size: batch size
        num_workers: DataLoader workers
    Returns:
        DataLoader yielding (noisy, clean, mask) batches
    """
    path    = TRAIN_PATH if split == 'train' else VAL_PATH
    shuffle = split == 'train'
    ds = CLEVRSetDataset(path, max_samples=max_samples)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_model(
    d_model: int = 128,
    latent_dim: int = 128,
    hidden_dim: int = 128,
    slotted: bool = False,
) -> SetAutoencoder:
    """Build SetTransformer encoder + MLP (or Slotted) decoder.

    Args:
        d_model: transformer hidden dimension
        latent_dim: encoder output / decoder input dimension
        hidden_dim: MLP hidden dimension
        slotted: if True use SlottedMLPSetDecoder (learnable per-slot embeddings)
    Returns:
        SetAutoencoder on DEVICE
    """
    encoder = SetTransformerEncoder(
        input_dim=OBJ_DIM, d_model=d_model, nhead=4,
        num_layers=2, latent_dim=latent_dim,
        dim_feedforward=d_model * 2, dropout=0.0,
    )
    if slotted:
        decoder = SlottedMLPSetDecoder(
            latent_dim=latent_dim, hidden_dim=hidden_dim,
            max_objects=MAX_OBJECTS, obj_dim=OBJ_DIM,
        )
    else:
        decoder = MLPSetDecoder(
            latent_dim=latent_dim, hidden_dim=hidden_dim,
            max_objects=MAX_OBJECTS, obj_dim=OBJ_DIM,
        )
    return SetAutoencoder(encoder, decoder).to(DEVICE)


def make_dspn_model(
    d_model: int = 128,
    latent_dim: int = 128,
    inner_lr: float = 800.0,
    inner_steps: int = 5,
) -> DSPNModel:
    """Build SetTransformer encoder + DSPN gradient-descent decoder.

    Args:
        d_model: transformer hidden dimension
        latent_dim: encoder output dimension
        inner_lr: DSPN inner-loop learning rate
        inner_steps: DSPN inner-loop steps (reduced from 10 for speed)
    Returns:
        DSPNModel on DEVICE
    """
    encoder = SetTransformerEncoder(
        input_dim=OBJ_DIM, d_model=d_model, nhead=4,
        num_layers=2, latent_dim=latent_dim,
        dim_feedforward=d_model * 2, dropout=0.0,
    )
    return DSPNModel(
        encoder=encoder, obj_dim=OBJ_DIM, max_objects=MAX_OBJECTS,
        inner_lr=inner_lr, inner_steps=inner_steps,
    ).to(DEVICE)


# ---------------------------------------------------------------------------
# Monitoring utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def pred_diversity(preds: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean pairwise L2 distance between predicted slots (real slots only).

    Low value → predictions are clustered (potential collapse).

    Args:
        preds: (B, M, D)
        mask:  (B, N) binary
    Returns:
        scalar float
    """
    divs = []
    for b in range(preds.shape[0]):
        n = int(mask[b].sum().item())
        if n < 2:
            continue
        p = preds[b, :n]
        D = torch.cdist(p.unsqueeze(0), p.unsqueeze(0))[0]
        off = D[~torch.eye(n, dtype=bool, device=D.device)]
        divs.append(off.mean().item())
    return float(np.mean(divs)) if divs else 0.0


@torch.no_grad()
def softmax_entropy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    tau: float,
) -> float:
    """Mean normalised entropy of softmax(-D/tau) over targets.

    1.0 = fully uniform = loss is blind to prediction ordering.

    Args:
        preds:   (B, M, D)
        targets: (B, N, D)
        mask:    (B, N) binary
        tau:     softmin temperature
    Returns:
        scalar float in [0, 1]
    """
    D = torch.cdist(preds, targets)
    entropies = []
    for b in range(D.shape[0]):
        n = int(mask[b].sum().item())
        if n < 2:
            continue
        D_b = D[b, :n, :n]
        w   = torch.softmax(-D_b / tau, dim=0)
        H   = -(w * torch.log(w + 1e-10)).sum(dim=0).mean()
        entropies.append((H / np.log(n)).item())
    return float(np.mean(entropies)) if entropies else 1.0


def grad_diversity(
    model: nn.Module,
    batch: Tuple,
    loss_fn: nn.Module,
) -> float:
    """Gradient diversity: std/mean of per-slot gradient norms.

    High value → slots receive distinct signals (good).
    Low value  → all slots receive the same gradient (centroid trap).
    Works for both MLP and DSPN architectures via output hook.

    Args:
        model:   any model with forward(x, mask=...) → (B, M, D)
        batch:   (inp, tgt, msk) tuple of tensors
        loss_fn: loss function taking (D, mask=msk)
    Returns:
        scalar float (nan on failure)
    """
    inp, tgt, msk = [x.to(DEVICE) for x in batch]
    slot_grads: Dict = {}

    def _hook(grad):
        slot_grads['g'] = grad.detach().cpu()

    try:
        model.train()
        preds = model(inp, mask=msk)
        preds.register_hook(_hook)
        D = torch.cdist(preds, tgt)
        loss_fn(D, mask=msk).backward()
        model.zero_grad()
    except Exception:
        return float('nan')

    g = slot_grads.get('g')
    if g is None:
        return float('nan')
    slot_norms = g.norm(dim=2).mean(dim=0)
    return slot_norms.std().item() / (slot_norms.mean().item() + 1e-10)


@torch.no_grad()
def hungarian_dist(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Mean Hungarian-matched L2 distance over a batch.

    Args:
        preds:   (B, M, D)
        targets: (B, N, D)
        mask:    (B, N) binary
    Returns:
        scalar float
    """
    from influencerformer.losses.set_losses import _POOL, _N_WORKERS
    D_np = torch.cdist(preds, targets).cpu().numpy()
    B = D_np.shape[0]

    def solve(b):
        n = int(mask[b].sum().item())
        D_sub = D_np[b, :n, :n]
        row, col = linear_sum_assignment(D_sub)
        return D_sub[row, col].mean()

    if B >= 64:
        dists = list(_POOL.map(solve, range(B)))
    else:
        dists = [solve(b) for b in range(B)]
    return float(np.mean(dists))


def slot_grad_norms(
    model: nn.Module,
    batch: Tuple,
    loss_fn: nn.Module,
) -> np.ndarray:
    """Per-slot gradient norms at initialisation (shape: (MAX_OBJECTS,)).

    Args:
        model:   fresh model
        batch:   (inp, tgt, msk)
        loss_fn: loss function
    Returns:
        (M,) numpy array of mean grad norms per slot
    """
    inp, tgt, msk = [x.to(DEVICE) for x in batch]
    slot_grads: Dict = {}

    def _hook(grad):
        slot_grads['g'] = grad.detach().cpu()

    model.train()
    preds = model(inp, mask=msk)
    preds.register_hook(_hook)
    D = torch.cdist(preds, tgt)
    loss_fn(D, mask=msk).backward()
    model.zero_grad()
    g = slot_grads['g']
    return g.norm(dim=2).mean(dim=0).numpy()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_and_monitor(
    loss_fn: nn.Module,
    loss_name: str,
    model_fn: Optional[Callable] = None,
    tau_for_entropy: float = 0.12,
    n_epochs: int = 50,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    scheduler_fn: Optional[Callable] = None,
    snapshot_epochs: Optional[Set[int]] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    seed: int = 42,
    early_stopping_patience: int = 0,
) -> Tuple[Dict, List]:
    """Train a model from scratch and record monitoring metrics + prediction snapshots.

    Args:
        loss_fn:                  instantiated loss module
        loss_name:                label for print/logging
        model_fn:                 callable returning a fresh model (default: make_model())
        tau_for_entropy:          temperature for softmax_entropy metric
        n_epochs:                 maximum number of training epochs
        lr:                       Adam learning rate
        weight_decay:             Adam weight decay
        scheduler_fn:             callable(optimizer) → lr_scheduler, or None
        snapshot_epochs:          set of epoch indices to save PCA snapshots
        train_loader:             DataLoader (default: 5k train, batch 64)
        val_loader:               DataLoader (default: 1k val, batch 64)
        seed:                     random seed for reproducibility
        early_stopping_patience:  stop if val_matched_dist doesn't improve for this
                                  many epochs. 0 disables early stopping.

    Returns:
        metrics:   dict of lists keyed by metric name
        snapshots: list of (epoch, preds_np, targets_np, mask_np)
    """
    if snapshot_epochs is None:
        snapshot_epochs = {0, 1, 5, 15, 35, n_epochs - 1}
    if model_fn is None:
        model_fn = make_model
    if train_loader is None:
        train_loader = make_dataloader('train', max_samples=5000)
    if val_loader is None:
        val_loader = make_dataloader('val', max_samples=1000)

    torch.manual_seed(seed)
    model     = model_fn()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = scheduler_fn(optimizer) if scheduler_fn is not None else None

    metrics: Dict[str, List] = {
        'epoch': [], 'train_loss': [], 'val_matched_dist': [],
        'pred_diversity': [], 'softmax_entropy': [], 'grad_diversity': [],
        'lr': [], 'epoch_time_s': [],
    }
    snapshots: List = []

    best_val_dist    = float('inf')
    no_improve_count = 0

    val_batch_fixed = next(iter(val_loader))
    inp_v, tgt_v, msk_v = [x.to(DEVICE) for x in val_batch_fixed]

    for epoch in range(n_epochs):
        t_epoch_start = time.time()

        # Training
        model.train()
        train_losses = []
        for inp, tgt, msk in train_loader:
            inp, tgt, msk = inp.to(DEVICE), tgt.to(DEVICE), msk.to(DEVICE)
            optimizer.zero_grad()
            preds = model(inp, mask=msk)
            D     = torch.cdist(preds, tgt)
            loss  = loss_fn(D, mask=msk)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(np.mean(train_losses))
            else:
                scheduler.step()

        # Validation
        model.eval()
        val_dists, diversities, entropies = [], [], []
        with torch.no_grad():
            for inp, tgt, msk in val_loader:
                inp, tgt, msk = inp.to(DEVICE), tgt.to(DEVICE), msk.to(DEVICE)
                preds = model(inp, mask=msk)
                val_dists.append(hungarian_dist(preds, tgt, msk))
                diversities.append(pred_diversity(preds, msk))
                entropies.append(softmax_entropy(preds, tgt, msk, tau=tau_for_entropy))

        gd = grad_diversity(model, next(iter(train_loader)), loss_fn)
        current_lr = optimizer.param_groups[0]['lr']

        metrics['epoch'].append(epoch)
        metrics['train_loss'].append(float(np.mean(train_losses)))
        metrics['val_matched_dist'].append(float(np.mean(val_dists)))
        metrics['pred_diversity'].append(float(np.mean(diversities)))
        metrics['softmax_entropy'].append(float(np.mean(entropies)))
        metrics['grad_diversity'].append(gd)
        metrics['lr'].append(current_lr)
        metrics['epoch_time_s'].append(time.time() - t_epoch_start)

        if epoch in snapshot_epochs:
            model.eval()
            with torch.no_grad():
                preds_snap = model(inp_v, mask=msk_v).cpu().numpy()
            snapshots.append((epoch, preds_snap, tgt_v.cpu().numpy(), msk_v.cpu().numpy()))

        val_dist = metrics['val_matched_dist'][-1]
        if val_dist < best_val_dist:
            best_val_dist    = val_dist
            no_improve_count = 0
        else:
            no_improve_count += 1

        print(f'  [{loss_name}] epoch {epoch:3d} | '
              f'loss={metrics["train_loss"][-1]:.4f} '
              f'dist={val_dist:.4f} '
              f'div={metrics["pred_diversity"][-1]:.3f} '
              f'H={metrics["softmax_entropy"][-1]:.3f} '
              f'gd={metrics["grad_diversity"][-1]:.3f} '
              f'lr={current_lr:.2e}')

        if early_stopping_patience > 0 and no_improve_count >= early_stopping_patience:
            print(f'  [{loss_name}] early stopping at epoch {epoch} '
                  f'(no improvement for {early_stopping_patience} epochs, '
                  f'best dist={best_val_dist:.4f})')
            break

    return metrics, snapshots


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_DEFAULT_COLORS = [
    'steelblue', 'seagreen', 'tomato', 'darkorange', 'mediumorchid',
    'black', 'crimson', 'teal', 'goldenrod', 'slategray',
]


def plot_monitoring(
    all_results: Dict[str, Tuple[Dict, List]],
    colors: Optional[Dict[str, str]] = None,
    title: str = 'Training monitoring',
    figsize: Tuple = (14, 9),
) -> None:
    """2x2 grid of monitoring metrics over epochs.

    Args:
        all_results: {name: (metrics_dict, snapshots)}
        colors:      optional {name: color} mapping
        title:       figure suptitle
    """
    if colors is None:
        names = list(all_results.keys())
        colors = {n: _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i, n in enumerate(names)}

    monitor_keys = [
        ('val_matched_dist', 'Val matched dist (↓ better)'),
        ('pred_diversity',   'Pred diversity (↑ = spread out)'),
        ('softmax_entropy',  'Softmax entropy (↑ = uniform/blind)'),
        ('grad_diversity',   'Gradient diversity (↑ = varied per slot)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax, (key, ylabel) in zip(axes.flatten(), monitor_keys):
        for name, (m, _) in all_results.items():
            if key in m:
                ax.plot(m['epoch'], m[key], label=name,
                        color=colors.get(name, None), lw=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle(title, fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_pca_snapshots(
    snapshots: List,
    title: str,
    n_scenes: int = 4,
) -> None:
    """PCA projection of predictions vs targets at each snapshot epoch.

    Fits PCA on all real targets from the fixed batch at epoch 0, then
    projects predictions into the same space at each snapshot.

    Args:
        snapshots: list of (epoch, preds_np, targets_np, mask_np)
        title:     figure suptitle
        n_scenes:  number of scenes (rows) to display
    """
    if not snapshots:
        print(f'No snapshots available for {title}')
        return

    n_snaps = len(snapshots)
    fig, axes = plt.subplots(n_scenes, n_snaps,
                              figsize=(n_snaps * 2.8, n_scenes * 2.8))
    if n_snaps == 1:
        axes = axes.reshape(-1, 1)

    _, _, tgt_all, msk_all = snapshots[0]
    real_tgts = np.vstack([
        tgt_all[b, :int(msk_all[b].sum())]
        for b in range(tgt_all.shape[0])
    ])
    pca = PCA(n_components=2)
    pca.fit(real_tgts)

    for col, (epoch, preds_np, tgt_np, msk_np) in enumerate(snapshots):
        for row in range(n_scenes):
            ax  = axes[row, col]
            n   = int(msk_np[row].sum())
            tgt_2d  = pca.transform(tgt_np[row, :n])
            pred_2d = pca.transform(preds_np[row, :n])

            ax.scatter(*tgt_2d.T,  s=80, c='steelblue', zorder=5, alpha=0.9)
            ax.scatter(*pred_2d.T, s=50, c='tomato', zorder=4,
                       marker='x', linewidths=1.5)

            D_sub = np.sqrt(((pred_2d[:, None] - tgt_2d[None, :]) ** 2).sum(-1))
            ri, ci = linear_sum_assignment(D_sub)
            for r, c in zip(ri, ci):
                ax.plot([pred_2d[r, 0], tgt_2d[c, 0]],
                        [pred_2d[r, 1], tgt_2d[c, 1]],
                        c='gray', lw=0.8, alpha=0.5, zorder=3)

            if row == 0:
                ax.set_title(f'epoch {epoch}', fontweight='bold', fontsize=9)
            if col == 0:
                ax.set_ylabel(f'scene {row}', fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(alpha=0.2)

    plt.suptitle(f'{title}\nBlue = targets   Red x = predictions   Lines = Hungarian match',
                 fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.show()


def summary_table(all_results: Dict[str, Tuple[Dict, List]]) -> None:
    """Print a summary table of final metrics for all runs.

    Args:
        all_results: {name: (metrics_dict, snapshots)}
    """
    print(f"{'Name':<28} {'Final dist':>12} {'Pred div':>10} "
          f"{'SM entropy':>12} {'Grad div':>10}")
    print('-' * 76)
    rows = sorted(all_results.items(), key=lambda kv: kv[1][0]['val_matched_dist'][-1])
    for name, (m, _) in rows:
        print(f"{name:<28} "
              f"{m['val_matched_dist'][-1]:>12.4f} "
              f"{m['pred_diversity'][-1]:>10.4f} "
              f"{m['softmax_entropy'][-1]:>12.4f} "
              f"{m['grad_diversity'][-1]:>10.4f}")
