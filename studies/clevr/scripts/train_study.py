"""Run CLEVR matcher and loss studies from the cleaned layout.

This is the canonical study entrypoint for the reorganized CLEVR area under
`studies/clevr/`. It restores the deleted study runner while keeping the run
names and pickle structure used by the existing result packaging.
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_STUDY_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _STUDY_ROOT.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from influencerformer.losses import (
    ChamferLoss,
    CombinedSoftMinLoss,
    HungarianLoss,
    PowerSoftMinLoss,
    SoftDCDLoss,
)
from studies.clevr.lib import clevr_utils as cu

LR_BASE = 3e-4
BATCH_BASE = 64
DEFAULT_TAU = 0.12
DEFAULT_POWER = 3.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Inputs:
        None.
    Outputs:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--study',
        required=True,
        choices=[
            'tau_sweep',
            'power_sweep',
            'data_scale',
            'slot_embeddings',
            'lr_schedule',
            'warmstart',
            'softdcd',
            'model_scale',
            'hungarian_baseline',
            'cost_comparison',
            'combined_recipe',
            'pm3_best_recipe',
        ],
        help='Study to run.',
    )
    parser.add_argument('--n_epochs', type=int, default=200, help='Maximum training epochs.')
    parser.add_argument('--batch_size', type=int, default=512, help='Training batch size.')
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate. Defaults to sqrt-scaled value from batch size.',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument(
        '--patience',
        type=int,
        default=30,
        help='Early stopping patience on validation matched distance. 0 disables it.',
    )
    parser.add_argument(
        '--val_samples',
        type=int,
        default=3000,
        help='Validation sample cap used during the studies.',
    )
    parser.add_argument(
        '--train_samples',
        type=int,
        default=5000,
        help='Default training sample cap for studies that do not override it.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(_STUDY_ROOT / 'results'),
        help='Directory for study pickle outputs and derived figures.',
    )
    return parser.parse_args()


def resolve_output_dir(output_dir: str) -> Path:
    """Resolve an output directory relative to the repo root when needed.

    Inputs:
        output_dir: User-supplied output directory.
    Outputs:
        Absolute output directory path.
    """
    path = Path(output_dir)
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def scaled_learning_rate(batch_size: int, lr: Optional[float] = None) -> float:
    """Return the explicit or sqrt-scaled learning rate.

    Inputs:
        batch_size: Training batch size.
        lr: Optional explicit learning rate.
    Outputs:
        Learning rate to use.
    """
    if lr is not None:
        return lr
    return LR_BASE * math.sqrt(batch_size / BATCH_BASE)


def snap_epochs(n_epochs: int) -> set[int]:
    """Choose a compact set of snapshot epochs.

    Inputs:
        n_epochs: Maximum training epochs.
    Outputs:
        Set of epoch indices to snapshot.
    """
    anchors = {0, 1, 5, 15, 35, n_epochs - 1}
    return {epoch for epoch in anchors if 0 <= epoch < n_epochs}


def make_loaders(
    batch_size: int,
    train_samples: Optional[int],
    val_samples: int,
    num_workers: int = 4,
):
    """Build train and validation loaders for a CLEVR study.

    Inputs:
        batch_size: Training and validation batch size.
        train_samples: Training sample cap, or None for full split.
        val_samples: Validation sample cap.
        num_workers: DataLoader workers.
    Outputs:
        Train loader and validation loader.
    """
    train_loader = cu.make_dataloader(
        'train',
        max_samples=train_samples,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader = cu.make_dataloader(
        'val',
        max_samples=val_samples,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    warmup_epochs: int = 10,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a warmup plus cosine decay scheduler.

    Inputs:
        optimizer: Optimizer instance.
        n_epochs: Total training epochs.
        warmup_epochs: Linear warmup length.
    Outputs:
        Configured LambdaLR scheduler.
    """

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        if n_epochs <= warmup_epochs:
            return 1.0
        progress = (epoch - warmup_epochs) / float(max(1, n_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def scheduler_factory(name: str, n_epochs: int, warmup_epochs: int = 10) -> Callable:
    """Build a scheduler factory by name.

    Inputs:
        name: Scheduler name.
        n_epochs: Total training epochs.
        warmup_epochs: Warmup length for warmup-cosine.
    Outputs:
        Callable that accepts an optimizer and returns a scheduler.
    """
    if name == 'none':
        return lambda optimizer: None
    if name == 'cosine':
        return lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    if name == 'warmup_cosine':
        return lambda optimizer: make_warmup_cosine_scheduler(
            optimizer, n_epochs=n_epochs, warmup_epochs=warmup_epochs
        )
    if name == 'plateau':
        return lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
    raise ValueError(f'Unknown scheduler: {name}')


def make_model_factory(
    d_model: int = 128,
    latent_dim: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    slotted: bool = False,
) -> Callable[[], torch.nn.Module]:
    """Build a closure that returns a fresh CLEVR model.

    Inputs:
        d_model: Encoder width.
        latent_dim: Latent dimension. Defaults to d_model.
        hidden_dim: Decoder hidden dimension. Defaults to d_model.
        slotted: Whether to use the slotted decoder.
    Outputs:
        Zero-argument model factory.
    """
    latent_dim = latent_dim or d_model
    hidden_dim = hidden_dim or d_model
    return lambda: cu.make_model(
        d_model=d_model,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        slotted=slotted,
    )


def run_training(
    run_name: str,
    loss_fn: torch.nn.Module,
    args: argparse.Namespace,
    model_fn: Optional[Callable[[], torch.nn.Module]] = None,
    train_samples: Optional[int] = None,
    scheduler_fn: Optional[Callable] = None,
    tau_for_entropy: float = DEFAULT_TAU,
) -> Tuple[dict, list]:
    """Train one study configuration.

    Inputs:
        run_name: Display name for logs and pickle keys.
        loss_fn: Instantiated loss module.
        args: Parsed study arguments.
        model_fn: Optional model factory.
        train_samples: Optional study-specific train sample cap.
        scheduler_fn: Optional optimizer scheduler factory.
        tau_for_entropy: Temperature used for the entropy metric.
    Outputs:
        Metrics dict and snapshot list from `cu.train_and_monitor`.
    """
    train_loader, val_loader = make_loaders(
        batch_size=args.batch_size,
        train_samples=args.train_samples if train_samples is None else train_samples,
        val_samples=args.val_samples,
    )
    return cu.train_and_monitor(
        loss_fn=loss_fn,
        loss_name=run_name,
        model_fn=model_fn,
        tau_for_entropy=tau_for_entropy,
        n_epochs=args.n_epochs,
        lr=scaled_learning_rate(args.batch_size, args.lr),
        scheduler_fn=scheduler_fn,
        snapshot_epochs=snap_epochs(args.n_epochs),
        train_loader=train_loader,
        val_loader=val_loader,
        seed=args.seed,
        early_stopping_patience=args.patience,
    )


def profile_batch_timing(
    loss_fn: torch.nn.Module,
    model_fn: Callable[[], torch.nn.Module],
    batch_size: int,
    train_samples: int,
    steps: int = 20,
    warmup_steps: int = 5,
) -> Tuple[float, float]:
    """Measure approximate GPU and wall time per batch.

    Inputs:
        loss_fn: Loss module to profile.
        model_fn: Model factory.
        batch_size: Batch size to profile.
        train_samples: Number of samples to draw a profiling batch from.
        steps: Timed steps.
        warmup_steps: Untimed warmup steps.
    Outputs:
        Tuple of GPU and wall milliseconds per batch.
    """
    train_loader, _ = make_loaders(batch_size=batch_size, train_samples=train_samples, val_samples=32)
    batch = next(iter(train_loader))
    inp, tgt, msk = [tensor.to(cu.DEVICE) for tensor in batch]
    model = model_fn()
    optimizer = torch.optim.Adam(model.parameters(), lr=scaled_learning_rate(batch_size))

    def step() -> None:
        optimizer.zero_grad()
        preds = model(inp, mask=msk)
        loss = loss_fn(torch.cdist(preds, tgt), mask=msk)
        loss.backward()
        optimizer.step()

    for _ in range(warmup_steps):
        step()

    use_cuda = cu.DEVICE.type == 'cuda'
    if use_cuda:
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        torch.cuda.synchronize()
    wall_start = time.perf_counter()
    for idx in range(steps):
        if use_cuda:
            start_events[idx].record()
        step()
        if use_cuda:
            end_events[idx].record()
    if use_cuda:
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0 / steps
    if not use_cuda:
        return math.nan, wall_ms
    gpu_ms = float(np.mean([start.elapsed_time(end) for start, end in zip(start_events, end_events)]))
    return gpu_ms, wall_ms


def save_study(study: str, args: argparse.Namespace, all_results: Dict[str, Tuple[dict, list]]) -> Path:
    """Write one study pickle in the historical schema.

    Inputs:
        study: Study name.
        args: Parsed CLI args.
        all_results: Mapping from run name to metrics and snapshots.
    Outputs:
        Output pickle path.
    """
    output_dir = resolve_output_dir(args.output_dir)
    out_path = output_dir / f'{study}.pkl'
    payload = {
        'study': study,
        'args': {
            'study': args.study,
            'n_epochs': args.n_epochs,
            'batch_size': args.batch_size,
            'lr': scaled_learning_rate(args.batch_size, args.lr),
            'output_dir': args.output_dir,
            'seed': args.seed,
            'patience': args.patience,
            'val_samples': args.val_samples,
        },
        'all_results': all_results,
    }
    with open(out_path, 'wb') as handle:
        pickle.dump(payload, handle)
    return out_path


def run_named_runs(
    args: argparse.Namespace,
    specs: Iterable[Tuple[str, torch.nn.Module, dict]],
) -> Dict[str, Tuple[dict, list]]:
    """Execute a list of named run specs.

    Inputs:
        args: Parsed CLI args.
        specs: Iterable of run name, loss function, and extra kwargs.
    Outputs:
        Result mapping suitable for pickling.
    """
    all_results: Dict[str, Tuple[dict, list]] = {}
    for run_name, loss_fn, extra in specs:
        metrics, snapshots = run_training(run_name=run_name, loss_fn=loss_fn, args=args, **extra)
        all_results[run_name] = (metrics, snapshots)
    return all_results


def study_tau_sweep(args: argparse.Namespace) -> Dict[str, Tuple[dict, list]]:
    """Run the PM3 temperature sweep."""
    taus = [0.07, 0.10, 0.12, 0.15, 0.20, 0.30]
    specs = [
        (f'τ={tau}', PowerSoftMinLoss(temperature=tau, power=DEFAULT_POWER), {'tau_for_entropy': tau})
        for tau in taus
    ]
    return run_named_runs(args, specs)


def study_power_sweep(args: argparse.Namespace) -> Dict[str, Tuple[dict, list]]:
    """Run the PM temperature power sweep."""
    powers = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    specs = [
        (f'p={power}', PowerSoftMinLoss(temperature=DEFAULT_TAU, power=power), {})
        for power in powers
    ]
    return run_named_runs(args, specs)


def study_data_scale(args: argparse.Namespace) -> Dict[str, Tuple[dict, list]]:
    """Compare Chamfer and PM3 across dataset scales."""
    scales = [(5000, '5k'), (20000, '20k'), (None, '70k')]
    specs = []
    for samples, label in scales:
        specs.append((f'Chamfer n={label}', ChamferLoss(), {'train_samples': samples}))
        specs.append(
            (
                f'PM3 n={label}',
                PowerSoftMinLoss(temperature=DEFAULT_TAU, power=DEFAULT_POWER),
                {'train_samples': samples},
            )
        )
    return run_named_runs(args, specs)


def study_slot_embeddings(args: argparse.Namespace) -> Dict[str, Tuple[dict, list]]:
    """Compare standard and slotted decoders."""
    specs = [
        ('Chamfer (standard)', ChamferLoss(), {'model_fn': make_model_factory(slotted=False)}),
        ('Chamfer (slotted)', ChamferLoss(), {'model_fn': make_model_factory(slotted=True)}),
        (
            'PM3 (standard)',
            PowerSoftMinLoss(temperature=DEFAULT_TAU, power=DEFAULT_POWER),
            {'model_fn': make_model_factory(slotted=False)},
        ),
        (
            'PM3 (slotted)',
            PowerSoftMinLoss(temperature=DEFAULT_TAU, power=DEFAULT_POWER),
            {'model_fn': make_model_factory(slotted=True)},
        ),
    ]
    return run_named_runs(args, specs)


def study_lr_schedule(args: argparse.Namespace) -> Dict[str, Tuple[dict, list]]:
    """Compare learning-rate schedules for PM3."""
    names = ['none', 'cosine', 'warmup_cosine', 'plateau']
    specs = [
        (
            name,
            PowerSoftMinLoss(temperature=DEFAULT_TAU, power=DEFAULT_POWER),
            {'scheduler_fn': scheduler_factory(name, args.n_epochs)},
        )
        for name in names
    ]
    return run_named_runs(args, specs)


def study_warmstart(args: argparse.Namespace) -> Dict[str, Tuple[dict, list]]:
    """Vary warmup length inside the warmup-cosine schedule."""
    warmups = [0, 3, 10, 20]
    specs = [
        (
            f'warmup={warmup}',
            PowerSoftMinLoss(temperature=DEFAULT_TAU, power=DEFAULT_POWER),
            {'scheduler_fn': scheduler_factory('warmup_cosine', args.n_epochs, warmup_epochs=warmup)},
        )
        for warmup in warmups
    ]
    return run_named_runs(args, specs)


def study_softdcd(args: argparse.Namespace) -> Dict[str, Tuple[dict, list]]:
    """Compare PM3 against other soft matching alternatives."""
    specs = [
        ('Chamfer', ChamferLoss(), {}),
        ('PM3 τ=0.12', PowerSoftMinLoss(temperature=DEFAULT_TAU, power=DEFAULT_POWER), {}),
        ('SoftDCD', SoftDCDLoss(temperature=DEFAULT_TAU), {}),
        ('CombinedSM', CombinedSoftMinLoss(temperature=DEFAULT_TAU), {}),
    ]
    return run_named_runs(args, specs)


def study_model_scale(args: argparse.Namespace) -> Dict[str, Tuple[dict, list]]:
    """Scale the backbone width for PM3."""
    widths = [64, 128, 256, 512]
    specs = [
        (
            f'd_model={width}',
            PowerSoftMinLoss(temperature=DEFAULT_TAU, power=DEFAULT_POWER),
            {'model_fn': make_model_factory(d_model=width, latent_dim=width, hidden_dim=width)},
        )
        for width in widths
    ]
    return run_named_runs(args, specs)


def study_hungarian_baseline(args: argparse.Namespace) -> Dict[str, Tuple[dict, list]]:
    """Compare Chamfer, Hungarian, and default PM3."""
    specs = [
        ('Chamfer', ChamferLoss(), {}),
        ('Hungarian', HungarianLoss(), {}),
        ('PM3 τ=0.12', PowerSoftMinLoss(temperature=DEFAULT_TAU, power=DEFAULT_POWER), {}),
    ]
    return run_named_runs(args, specs)


def study_cost_comparison(args: argparse.Namespace) -> Dict[str, Tuple[dict, list]]:
    """Augment the baseline comparison with timing metrics."""
    specs = [
        ('Chamfer', ChamferLoss(), {}),
        ('PM3 τ=0.12', PowerSoftMinLoss(temperature=DEFAULT_TAU, power=DEFAULT_POWER), {}),
        ('Hungarian', HungarianLoss(), {}),
    ]
    all_results = run_named_runs(args, specs)
    batches_per_epoch = len(make_loaders(args.batch_size, args.train_samples, 32)[0])
    for run_name, loss_fn, extra in specs:
        model_fn = extra.get('model_fn', make_model_factory())
        gpu_ms, wall_ms = profile_batch_timing(
            loss_fn=loss_fn,
            model_fn=model_fn,
            batch_size=args.batch_size,
            train_samples=args.train_samples,
        )
        metrics, snapshots = all_results[run_name]
        metrics['gpu_ms_per_batch'] = gpu_ms
        metrics['wall_ms_per_batch'] = wall_ms
        metrics['batches_per_epoch'] = batches_per_epoch
        all_results[run_name] = (metrics, snapshots)
    return all_results


def study_combined_recipe(args: argparse.Namespace) -> Dict[str, Tuple[dict, list]]:
    """Run the slotted plus warmup-cosine recipe across three matchers."""
    scheduler_fn = scheduler_factory('warmup_cosine', args.n_epochs, warmup_epochs=10)
    model_fn = make_model_factory(slotted=True)
    specs = [
        (
            'Chamfer (slotted + warmup_cosine)',
            ChamferLoss(),
            {'model_fn': model_fn, 'scheduler_fn': scheduler_fn},
        ),
        (
            'Hungarian (slotted + warmup_cosine)',
            HungarianLoss(),
            {'model_fn': model_fn, 'scheduler_fn': scheduler_fn},
        ),
        (
            'PM3 τ=0.15 (slotted + warmup_cosine)',
            PowerSoftMinLoss(temperature=0.15, power=DEFAULT_POWER),
            {'model_fn': model_fn, 'scheduler_fn': scheduler_fn, 'tau_for_entropy': 0.15},
        ),
    ]
    return run_named_runs(args, specs)


def study_pm3_best_recipe(args: argparse.Namespace) -> Dict[str, Tuple[dict, list]]:
    """Run the PM3 best-recipe rerun."""
    scheduler_fn = scheduler_factory('warmup_cosine', args.n_epochs, warmup_epochs=10)
    specs = [
        (
            'PM3 τ=0.15 (slotted + warmup_cosine)',
            PowerSoftMinLoss(temperature=0.15, power=DEFAULT_POWER),
            {
                'model_fn': make_model_factory(slotted=True),
                'scheduler_fn': scheduler_fn,
                'tau_for_entropy': 0.15,
            },
        )
    ]
    return run_named_runs(args, specs)


STUDY_DISPATCH = {
    'tau_sweep': study_tau_sweep,
    'power_sweep': study_power_sweep,
    'data_scale': study_data_scale,
    'slot_embeddings': study_slot_embeddings,
    'lr_schedule': study_lr_schedule,
    'warmstart': study_warmstart,
    'softdcd': study_softdcd,
    'model_scale': study_model_scale,
    'hungarian_baseline': study_hungarian_baseline,
    'cost_comparison': study_cost_comparison,
    'combined_recipe': study_combined_recipe,
    'pm3_best_recipe': study_pm3_best_recipe,
}


def main() -> None:
    """Run one CLEVR study and save its pickle output.

    Inputs:
        CLI arguments only.
    Outputs:
        Writes a pickle under the requested results directory.
    """
    args = parse_args()
    results = STUDY_DISPATCH[args.study](args)
    out_path = save_study(args.study, args, results)
    print(f'Saved {args.study} to {out_path}')


if __name__ == '__main__':
    main()
