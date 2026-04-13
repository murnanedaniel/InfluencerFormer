"""Build summary tables and plots for the COCO DETR benchmark report.

Inputs:
    None. The script reads saved run artifacts under benchmarks/coco_detr/runs.
Outputs:
    PNG plots and CSV/Markdown summaries under benchmarks/coco_detr/report_assets.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / 'runs'
ASSETS_DIR = ROOT / 'report_assets'

# All run specs ordered by scale, then matcher
RUN_SPECS = [
    {
        'key': 'hungarian_smoke',
        'label': 'Hungarian smoke',
        'matcher': 'Hungarian',
        'scale': 'Smoke',
        'train_images': 128,
        'run_dir': RUNS_DIR / 'coco_detr_hungarian_smoke',
    },
    {
        'key': 'pm3_smoke',
        'label': 'PM3 smoke',
        'matcher': 'PM3',
        'scale': 'Smoke',
        'train_images': 128,
        'run_dir': RUNS_DIR / 'coco_detr_pm3_softmatch_smoke',
    },
    {
        'key': 'hungarian_small',
        'label': 'Hungarian small',
        'matcher': 'Hungarian',
        'scale': 'Small',
        'train_images': 256,
        'run_dir': RUNS_DIR / 'coco_detr_hungarian_small',
    },
    {
        'key': 'pm3_small',
        'label': 'PM3 small',
        'matcher': 'PM3',
        'scale': 'Small',
        'train_images': 256,
        'run_dir': RUNS_DIR / 'coco_detr_pm3_softmatch_small',
    },
    {
        'key': 'hungarian_medium',
        'label': 'Hungarian medium',
        'matcher': 'Hungarian',
        'scale': 'Medium',
        'train_images': 512,
        'run_dir': RUNS_DIR / 'coco_detr_hungarian_medium',
    },
    {
        'key': 'pm3_medium',
        'label': 'PM3 medium',
        'matcher': 'PM3',
        'scale': 'Medium',
        'train_images': 512,
        'run_dir': RUNS_DIR / 'coco_detr_pm3_softmatch_medium',
    },
    {
        'key': 'hungarian_large',
        'label': 'Hungarian large',
        'matcher': 'Hungarian',
        'scale': 'Large',
        'train_images': 1024,
        'run_dir': RUNS_DIR / 'coco_detr_hungarian_large',
    },
    {
        'key': 'pm3_large',
        'label': 'PM3 large',
        'matcher': 'PM3',
        'scale': 'Large',
        'train_images': 1024,
        'run_dir': RUNS_DIR / 'coco_detr_pm3_softmatch_large',
    },
]

# Improved recipe specs (20 epochs, backbone LR, cosine)
IMPROVED_SPECS = [
    {
        'key': 'hungarian_medium_20ep',
        'label': 'Hungarian med 20ep',
        'matcher': 'Hungarian',
        'scale': 'Medium',
        'train_images': 512,
        'recipe': 'improved',
        'run_dir': RUNS_DIR / 'coco_detr_hungarian_medium_20ep',
    },
    {
        'key': 'pm3_medium_20ep',
        'label': 'PM3 med 20ep',
        'matcher': 'PM3',
        'scale': 'Medium',
        'train_images': 512,
        'recipe': 'improved',
        'run_dir': RUNS_DIR / 'coco_detr_pm3_medium_20ep',
    },
    {
        'key': 'hungarian_large_20ep',
        'label': 'Hungarian lrg 20ep',
        'matcher': 'Hungarian',
        'scale': 'Large',
        'train_images': 1024,
        'recipe': 'improved',
        'run_dir': RUNS_DIR / 'coco_detr_hungarian_large_20ep',
    },
    {
        'key': 'pm3_large_20ep',
        'label': 'PM3 lrg 20ep',
        'matcher': 'PM3',
        'scale': 'Large',
        'train_images': 1024,
        'recipe': 'improved',
        'run_dir': RUNS_DIR / 'coco_detr_pm3_large_20ep',
    },
    {
        'key': 'pm3_norm_medium_20ep',
        'label': 'PM3-norm med 20ep',
        'matcher': 'PM3-norm',
        'scale': 'Medium',
        'train_images': 512,
        'recipe': 'improved',
        'run_dir': RUNS_DIR / 'coco_detr_pm3_norm_medium_20ep',
    },
    {
        'key': 'pm3_norm_large_20ep',
        'label': 'PM3-norm lrg 20ep',
        'matcher': 'PM3-norm',
        'scale': 'Large',
        'train_images': 1024,
        'recipe': 'improved',
        'run_dir': RUNS_DIR / 'coco_detr_pm3_norm_large_20ep',
    },
]

# Temperature ablation specs
TAU_ABLATION_SPECS = [
    {
        'key': 'pm3_tau005_small',
        'label': 'PM3 tau=0.05',
        'matcher': 'PM3',
        'scale': 'Small',
        'train_images': 256,
        'tau': 0.05,
        'run_dir': RUNS_DIR / 'coco_detr_pm3_tau005_small',
    },
    {
        'key': 'pm3_tau008_small',
        'label': 'PM3 tau=0.08',
        'matcher': 'PM3',
        'scale': 'Small',
        'train_images': 256,
        'tau': 0.08,
        'run_dir': RUNS_DIR / 'coco_detr_pm3_tau008_small',
    },
    {
        'key': 'pm3_small',
        'label': 'PM3 tau=0.12',
        'matcher': 'PM3',
        'scale': 'Small',
        'train_images': 256,
        'tau': 0.12,
        'run_dir': RUNS_DIR / 'coco_detr_pm3_softmatch_small',
    },
    {
        'key': 'pm3_tau020_small',
        'label': 'PM3 tau=0.20',
        'matcher': 'PM3',
        'scale': 'Small',
        'train_images': 256,
        'tau': 0.20,
        'run_dir': RUNS_DIR / 'coco_detr_pm3_tau020_small',
    },
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def load_jsonl(path: Path) -> list[dict]:
    lines = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def collect_run_summary(spec: dict) -> dict | None:
    """Collect key metrics for one run. Returns None if run doesn't exist."""
    if not spec['run_dir'].exists():
        return None
    metrics_path = spec['run_dir'] / 'metrics.jsonl'
    if not metrics_path.exists():
        return None
    metrics = load_jsonl(metrics_path)
    if not metrics:
        return None
    resolved = load_json(spec['run_dir'] / 'resolved_config.json')
    eval_metrics = {}
    eval_path = spec['run_dir'] / 'eval_metrics.json'
    if eval_path.exists():
        eval_metrics = load_json(eval_path)
    final = metrics[-1]
    summary = {
        'key': spec['key'],
        'label': spec['label'],
        'matcher': spec['matcher'],
        'scale': spec['scale'],
        'train_images': resolved['data'].get('smoke_train_images', spec.get('train_images', '?')),
        'val_images': resolved['data'].get('smoke_val_images', '?'),
        'epochs': resolved['train']['epochs'],
        'lr': resolved['train']['lr'],
        'final_train_loss': final['train_loss'],
        'final_val_loss': final.get('val_loss', float('nan')),
        'best_train_loss': min(row['train_loss'] for row in metrics),
        'best_val_loss': min((row['val_loss'] for row in metrics if 'val_loss' in row), default=float('nan')),
        'final_map': final.get('map', float('nan')),
        'eval_val_loss': eval_metrics.get('val_loss', final.get('val_loss', float('nan'))),
        'eval_map': eval_metrics.get('map', final.get('map', float('nan'))),
        'eval_map_50': eval_metrics.get('map_50', final.get('map_50', float('nan'))),
        'eval_map_75': eval_metrics.get('map_75', final.get('map_75', float('nan'))),
        'total_time_s': sum(row.get('epoch_time_s', 0) for row in metrics),
        'avg_epoch_time_s': np.mean([row['epoch_time_s'] for row in metrics if 'epoch_time_s' in row]),
        'metrics': metrics,
    }
    return summary


def write_summary_files(summaries: list[dict]) -> None:
    """Write CSV and Markdown summaries."""
    rows = []
    for s in summaries:
        rows.append({
            'label': s['label'],
            'matcher': s['matcher'],
            'scale': s['scale'],
            'train_images': s['train_images'],
            'val_images': s['val_images'],
            'epochs': s['epochs'],
            'lr': s['lr'],
            'final_train_loss': s['final_train_loss'],
            'final_val_loss': s['final_val_loss'],
            'best_train_loss': s['best_train_loss'],
            'best_val_loss': s['best_val_loss'],
            'eval_val_loss': s['eval_val_loss'],
            'eval_map': s['eval_map'],
            'eval_map_50': s['eval_map_50'],
            'eval_map_75': s['eval_map_75'],
            'total_time_s': s['total_time_s'],
            'avg_epoch_time_s': s['avg_epoch_time_s'],
        })

    csv_path = ASSETS_DIR / 'run_summary.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        '# COCO DETR Benchmark Summary',
        '',
        '| Run | Matcher | Scale | Train imgs | Epochs | Final val loss | Eval mAP | AP50 | AP75 | Time (s) |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        md_lines.append(
            f"| {row['label']} | {row['matcher']} | {row['scale']} | {row['train_images']} | "
            f"{row['epochs']} | {row['final_val_loss']:.4f} | "
            f"{row['eval_map']:.4f} | {row['eval_map_50']:.4f} | {row['eval_map_75']:.4f} | "
            f"{row['total_time_s']:.1f} |"
        )
    (ASSETS_DIR / 'run_summary.md').write_text('\n'.join(md_lines) + '\n', encoding='utf-8')


def plot_training_curves(summaries: list[dict]) -> None:
    """Plot train loss, val loss, and mAP curves grouped by scale."""
    scales = []
    seen = set()
    for s in summaries:
        if s['scale'] not in seen and s['scale'] != 'Smoke':
            scales.append(s['scale'])
            seen.add(s['scale'])

    if not scales:
        return

    fig, axes = plt.subplots(len(scales), 3, figsize=(14, 4.5 * len(scales)),
                             squeeze=False)
    colors = {'Hungarian': '#2196F3', 'PM3': '#FF5722'}

    for row_idx, scale in enumerate(scales):
        scale_runs = [s for s in summaries if s['scale'] == scale]
        for s in scale_runs:
            color = colors.get(s['matcher'], 'gray')
            epochs = [r['epoch'] for r in s['metrics']]
            axes[row_idx, 0].plot(epochs, [r['train_loss'] for r in s['metrics']],
                                  marker='o', color=color, label=s['matcher'], markersize=4)
            if 'val_loss' in s['metrics'][0]:
                axes[row_idx, 1].plot(epochs, [r['val_loss'] for r in s['metrics']],
                                      marker='o', color=color, label=s['matcher'], markersize=4)
            if 'map' in s['metrics'][0]:
                axes[row_idx, 2].plot(epochs, [r['map'] for r in s['metrics']],
                                      marker='o', color=color, label=s['matcher'], markersize=4)

        axes[row_idx, 0].set_title(f'{scale} — Train loss')
        axes[row_idx, 1].set_title(f'{scale} — Val loss')
        axes[row_idx, 2].set_title(f'{scale} — Val mAP')
        for col in range(3):
            axes[row_idx, col].set_xlabel('Epoch')
            axes[row_idx, col].grid(True, alpha=0.3)
            axes[row_idx, col].legend(frameon=False, fontsize=9)
        axes[row_idx, 2].set_ylim(bottom=0.0)

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / 'training_curves.png', dpi=200)
    plt.close(fig)


def plot_scaling_comparison(summaries: list[dict]) -> None:
    """Plot mAP / AP50 / AP75 vs dataset size for Hungarian and PM3."""
    non_smoke = [s for s in summaries if s['scale'] != 'Smoke']
    if not non_smoke:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = {'Hungarian': '#2196F3', 'PM3': '#FF5722'}
    metrics_keys = [('eval_map', 'mAP'), ('eval_map_50', 'AP50'), ('eval_map_75', 'AP75')]

    for ax, (key, title) in zip(axes, metrics_keys):
        for matcher in ['Hungarian', 'PM3']:
            runs = [s for s in non_smoke if s['matcher'] == matcher]
            if not runs:
                continue
            runs.sort(key=lambda r: r['train_images'])
            x = [r['train_images'] for r in runs]
            y = [r[key] for r in runs]
            ax.plot(x, y, marker='o', color=colors[matcher], label=matcher, linewidth=2, markersize=8)
        ax.set_xlabel('Training images')
        ax.set_ylabel(title)
        ax.set_title(f'{title} vs Dataset Size')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / 'scaling_comparison.png', dpi=200)
    plt.close(fig)


def plot_timing_comparison(summaries: list[dict]) -> None:
    """Plot wall-clock training time comparison."""
    non_smoke = [s for s in summaries if s['scale'] != 'Smoke']
    if not non_smoke:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = {'Hungarian': '#2196F3', 'PM3': '#FF5722'}

    for matcher in ['Hungarian', 'PM3']:
        runs = sorted([s for s in non_smoke if s['matcher'] == matcher],
                       key=lambda r: r['train_images'])
        if not runs:
            continue
        x = [r['train_images'] for r in runs]
        axes[0].plot(x, [r['avg_epoch_time_s'] for r in runs],
                     marker='o', color=colors[matcher], label=matcher, linewidth=2, markersize=8)
        axes[1].plot(x, [r['total_time_s'] for r in runs],
                     marker='o', color=colors[matcher], label=matcher, linewidth=2, markersize=8)

    axes[0].set_title('Avg Epoch Time vs Dataset Size')
    axes[0].set_xlabel('Training images')
    axes[0].set_ylabel('Seconds / epoch')
    axes[1].set_title('Total Training Time vs Dataset Size')
    axes[1].set_xlabel('Training images')
    axes[1].set_ylabel('Total seconds')
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / 'timing_comparison.png', dpi=200)
    plt.close(fig)


def plot_eval_bars(summaries: list[dict]) -> None:
    """Plot final mAP and AP50 as grouped bars across all runs."""
    labels = [s['label'] for s in summaries]
    map_values = [s['eval_map'] for s in summaries]
    ap50_values = [s['eval_map_50'] for s in summaries]
    x = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([idx - width / 2 for idx in x], map_values, width=width, label='mAP', color='#2196F3')
    ax.bar([idx + width / 2 for idx in x], ap50_values, width=width, label='AP50', color='#FF5722')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Eval Metrics Across All Runs')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / 'eval_metric_bars.png', dpi=200)
    plt.close(fig)


def plot_recipe_comparison(baseline_summaries: list[dict], improved_summaries: list[dict]) -> None:
    """Compare baseline (6ep, flat LR) vs improved (20ep, backbone LR, cosine) recipes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {'Hungarian': '#2196F3', 'PM3': '#FF5722', 'PM3-norm': '#4CAF50'}

    for ax, (key, title) in zip(axes, [('eval_map', 'mAP'), ('eval_map_50', 'AP50')]):
        for matcher in ['Hungarian', 'PM3']:
            # Baseline (6ep)
            base_runs = sorted([s for s in baseline_summaries
                                if s['matcher'] == matcher and s['scale'] in ('Medium', 'Large')],
                               key=lambda r: r['train_images'])
            if base_runs:
                ax.plot([r['train_images'] for r in base_runs],
                        [r[key] for r in base_runs],
                        'o-', color=colors[matcher], label=f'{matcher} 6ep', markersize=7, alpha=0.5)
        for matcher in ['Hungarian', 'PM3', 'PM3-norm']:
            # Improved (20ep)
            imp_runs = sorted([s for s in improved_summaries if s['matcher'] == matcher],
                              key=lambda r: r['train_images'])
            if imp_runs:
                ax.plot([r['train_images'] for r in imp_runs],
                        [r[key] for r in imp_runs],
                        's--', color=colors.get(matcher, 'gray'), label=f'{matcher} 20ep',
                        markersize=7, linewidth=2)

        ax.set_xlabel('Training images')
        ax.set_ylabel(title)
        ax.set_title(f'{title}: Baseline vs Improved Recipe')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=9)
        ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / 'recipe_comparison.png', dpi=200)
    plt.close(fig)


def plot_tau_ablation(tau_summaries: list[dict]) -> None:
    """Plot mAP vs temperature for PM3 ablation."""
    if len(tau_summaries) < 2:
        return

    tau_summaries.sort(key=lambda s: s.get('tau', 0.12))
    taus = [s.get('tau', 0.12) for s in tau_summaries]
    maps = [s['eval_map'] for s in tau_summaries]
    ap50s = [s['eval_map_50'] for s in tau_summaries]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(taus, maps, marker='o', color='#FF5722', label='mAP', linewidth=2, markersize=8)
    ax.plot(taus, ap50s, marker='s', color='#FF9800', label='AP50', linewidth=2, markersize=8)
    ax.set_xlabel('Temperature (tau)')
    ax.set_ylabel('Score')
    ax.set_title('PM3 Temperature Sensitivity (Small, 256 images)')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / 'tau_ablation.png', dpi=200)
    plt.close(fig)


def main() -> None:
    """Build the report assets."""
    ASSETS_DIR.mkdir(exist_ok=True)
    summaries = []
    for spec in RUN_SPECS:
        summary = collect_run_summary(spec)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        print('No completed runs found.')
        return

    print(f'Found {len(summaries)} completed runs:')
    for s in summaries:
        print(f'  {s["label"]}: mAP={s["eval_map"]:.4f}, time={s["total_time_s"]:.1f}s')

    write_summary_files(summaries)
    plot_training_curves(summaries)
    plot_scaling_comparison(summaries)
    plot_timing_comparison(summaries)
    plot_eval_bars(summaries)

    # Improved recipe runs
    improved_summaries = []
    for spec in IMPROVED_SPECS:
        summary = collect_run_summary(spec)
        if summary is not None:
            improved_summaries.append(summary)
    if improved_summaries:
        print(f'\nImproved recipe runs: {len(improved_summaries)}')
        for s in improved_summaries:
            print(f'  {s["label"]}: mAP={s["eval_map"]:.4f}, time={s["total_time_s"]:.1f}s')
        plot_recipe_comparison(summaries, improved_summaries)

    # Temperature ablation
    tau_summaries = []
    for spec in TAU_ABLATION_SPECS:
        summary = collect_run_summary(spec)
        if summary is not None:
            summary['tau'] = spec.get('tau', 0.12)
            tau_summaries.append(summary)
    if tau_summaries:
        print(f'\nTau ablation runs: {len(tau_summaries)}')
        for s in tau_summaries:
            print(f'  tau={s["tau"]}: mAP={s["eval_map"]:.4f}')
        plot_tau_ablation(tau_summaries)

    print(f'\nAssets written to {ASSETS_DIR}')


if __name__ == '__main__':
    main()
