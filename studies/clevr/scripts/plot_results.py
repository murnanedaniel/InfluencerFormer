"""Plot CLEVR study results from saved pickle files."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_STUDY_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _STUDY_ROOT.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from studies.clevr.lib import clevr_utils as cu

RunDict = Dict[str, Tuple[dict, list]]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Inputs:
        None.
    Outputs:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study_path', type=str, required=True, help='Path to a study pickle.')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Optional output directory. Defaults to the pickle parent directory.',
    )
    return parser.parse_args()


def load_results(path: Path) -> tuple[str, RunDict]:
    """Load a study pickle.

    Inputs:
        path: Pickle path.
    Outputs:
        Study name and run dictionary.
    """
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    if isinstance(data, dict) and 'all_results' in data:
        return data.get('study', path.stem), data['all_results']
    return path.stem, data


def sanitize(name: str) -> str:
    """Make a filename-safe label.

    Inputs:
        name: Raw run name.
    Outputs:
        Sanitized filename fragment.
    """
    safe = name.replace(' ', '_').replace('/', '_')
    safe = safe.replace('(', '_').replace(')', '_').replace('+', '_')
    return safe.replace('__', '_')


def save_monitoring_plot(study: str, all_results: RunDict, output_dir: Path) -> Path:
    """Save the standard monitoring grid.

    Inputs:
        study: Study name.
        all_results: Result mapping.
        output_dir: Output directory.
    Outputs:
        Saved figure path.
    """
    cu.plot_monitoring(all_results, title=f'{study} monitoring')
    out_path = output_dir / f'{study}_monitoring.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    return out_path


def save_snapshot_plots(study: str, all_results: RunDict, output_dir: Path) -> list[Path]:
    """Save PCA snapshot figures for each run.

    Inputs:
        study: Study name.
        all_results: Result mapping.
        output_dir: Output directory.
    Outputs:
        List of saved figure paths.
    """
    outputs = []
    for run_name, (_, snapshots) in all_results.items():
        if not snapshots:
            continue
        cu.plot_pca_snapshots(snapshots, title=f'{study}: {run_name}')
        out_path = output_dir / f'{study}_snapshots_{sanitize(run_name)}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close('all')
        outputs.append(out_path)
    return outputs


def plot_cost_comparison(study: str, all_results: RunDict, output_dir: Path) -> Path:
    """Save GPU and wall time versus distance for the cost study.

    Inputs:
        study: Study name.
        all_results: Result mapping.
        output_dir: Output directory.
    Outputs:
        Saved figure path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for run_name, (metrics, _) in all_results.items():
        dist = np.asarray(metrics['val_matched_dist'])
        gpu_ms = float(metrics.get('gpu_ms_per_batch', np.nan))
        wall_ms = float(metrics.get('wall_ms_per_batch', np.nan))
        epochs = np.arange(len(dist))
        if np.isfinite(gpu_ms):
            axes[0].plot(epochs * gpu_ms / 1000.0, dist, linewidth=2, label=run_name)
        if np.isfinite(wall_ms):
            axes[1].plot(epochs * wall_ms / 1000.0, dist, linewidth=2, label=run_name)
    axes[0].set_title('GPU time vs distance')
    axes[1].set_title('Wall time vs distance')
    for ax in axes:
        ax.set_xlabel('Seconds elapsed')
        ax.set_ylabel('val_matched_dist')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    out_path = output_dir / f'{study}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main() -> None:
    """Render plots for one CLEVR study pickle.

    Inputs:
        CLI arguments only.
    Outputs:
        Writes plots to disk.
    """
    args = parse_args()
    study_path = Path(args.study_path).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else study_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    study, all_results = load_results(study_path)

    outputs = [save_monitoring_plot(study, all_results, output_dir)]
    outputs.extend(save_snapshot_plots(study, all_results, output_dir))
    if study == 'cost_comparison':
        outputs.append(plot_cost_comparison(study, all_results, output_dir))

    for out_path in outputs:
        print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
