"""Run one point in the 2x2x2 PM3 interaction scan."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_STUDY_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _STUDY_ROOT.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from influencerformer.losses import PowerSoftMinLoss
from studies.clevr.scripts.train_study import (
    make_model_factory,
    run_training,
    save_study,
    scheduler_factory,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Inputs:
        None.
    Outputs:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tau', type=float, required=True, choices=[0.12, 0.15], help='PM3 temperature.')
    parser.add_argument('--slotted', type=int, required=True, choices=[0, 1], help='Use slotted decoder.')
    parser.add_argument(
        '--warmup_cosine',
        type=int,
        required=True,
        choices=[0, 1],
        help='Use warmup-cosine scheduler.',
    )
    parser.add_argument('--n_epochs', type=int, default=200, help='Maximum training epochs.')
    parser.add_argument('--batch_size', type=int, default=512, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=None, help='Optional explicit learning rate.')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--val_samples', type=int, default=3000, help='Validation sample cap.')
    parser.add_argument('--train_samples', type=int, default=5000, help='Training sample cap.')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(_STUDY_ROOT / 'results' / 'pm3_grid'),
        help='Directory for the point pickle.',
    )
    return parser.parse_args()


def point_name(args: argparse.Namespace) -> str:
    """Return the canonical filename stem for a grid point.

    Inputs:
        args: Parsed CLI args.
    Outputs:
        Grid point name.
    """
    return f'pm3_tau{args.tau:.2f}_slots{args.slotted}_warmup{args.warmup_cosine}'


def main() -> None:
    """Train a single PM3 grid point and save it.

    Inputs:
        CLI arguments only.
    Outputs:
        Writes a point pickle under the requested grid directory.
    """
    args = parse_args()
    run_name = f'PM3 τ={args.tau:.2f} | slots={args.slotted} | warmup={args.warmup_cosine}'
    scheduler_fn = None
    if args.warmup_cosine:
        scheduler_fn = scheduler_factory('warmup_cosine', args.n_epochs, warmup_epochs=10)
    model_fn = make_model_factory(slotted=bool(args.slotted))
    metrics, snapshots = run_training(
        run_name=run_name,
        loss_fn=PowerSoftMinLoss(temperature=args.tau, power=3.0),
        args=args,
        model_fn=model_fn,
        scheduler_fn=scheduler_fn,
        tau_for_entropy=args.tau,
    )

    output_dir = Path(args.output_dir)
    output_dir = output_dir if output_dir.is_absolute() else (_REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    point_path = output_dir / f'{point_name(args)}.pkl'
    payload = {
        'study': point_name(args),
        'args': {
            'tau': args.tau,
            'slotted': args.slotted,
            'warmup_cosine': args.warmup_cosine,
            'n_epochs': args.n_epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'patience': args.patience,
            'seed': args.seed,
            'val_samples': args.val_samples,
            'output_dir': args.output_dir,
        },
        'all_results': {run_name: (metrics, snapshots)},
    }
    with open(point_path, 'wb') as handle:
        pickle.dump(payload, handle)
    print(f'Saved {point_path}')


if __name__ == '__main__':
    main()
