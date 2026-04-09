"""Package final CLEVR study results into plots, tables, and notes.

This script collects the finished study pickles, emits compact CSV/markdown
summaries, and creates a few final comparison figures used by the slide deck.

Usage:
    /global/homes/d/danieltm/.conda/envs/influencer/bin/python \
        scripts/package_clevr_results.py --results_dir results
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RunDict = Dict[str, Tuple[dict, list]]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Inputs:
        None.
    Outputs:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing CLEVR study pickle outputs.",
    )
    return parser.parse_args()


def load_results(path: Path) -> RunDict:
    """Load a study pickle and return its results dict.

    Inputs:
        path: Pickle path.
    Outputs:
        Mapping from run name to (metrics, snapshots).
    """
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    if isinstance(data, dict) and "all_results" in data:
        return data["all_results"]
    return data


def summarize_run(metrics: dict) -> dict:
    """Summarize one metrics dict into final scalar values.

    Inputs:
        metrics: Per-epoch metrics dictionary.
    Outputs:
        Compact summary dictionary.
    """
    values = metrics["val_matched_dist"]
    best_dist = min(values)
    best_epoch = values.index(best_dist)
    epoch_time_s = metrics.get("epoch_time_s", [])
    wall_to_best_min = (
        float(np.sum(epoch_time_s[: best_epoch + 1])) / 60.0 if epoch_time_s else math.nan
    )
    wall_total_min = float(np.sum(epoch_time_s)) / 60.0 if epoch_time_s else math.nan
    return {
        "epochs_logged": len(values),
        "best_dist": float(best_dist),
        "best_epoch": int(best_epoch),
        "final_dist": float(values[-1]),
        "wall_to_best_min": wall_to_best_min,
        "wall_total_min": wall_total_min,
        "final_train_loss": float(metrics.get("train_loss", [math.nan])[-1]),
        "final_pred_diversity": float(metrics.get("pred_diversity", [math.nan])[-1]),
        "final_softmax_entropy": float(metrics.get("softmax_entropy", [math.nan])[-1]),
        "final_grad_diversity": float(metrics.get("grad_diversity", [math.nan])[-1]),
    }


def iter_summary_rows(results_dir: Path) -> Iterable[dict]:
    """Yield summary rows for the packaged study files.

    Inputs:
        results_dir: Root results directory.
    Outputs:
        Iterator of row dicts suitable for CSV writing.
    """
    study_paths = [
        results_dir / "tau_sweep.pkl",
        results_dir / "power_sweep.pkl",
        results_dir / "data_scale.pkl",
        results_dir / "slot_embeddings.pkl",
        results_dir / "lr_schedule.pkl",
        results_dir / "warmstart.pkl",
        results_dir / "softdcd.pkl",
        results_dir / "model_scale.pkl",
        results_dir / "hungarian_baseline.pkl",
        results_dir / "cost_comparison.pkl",
        results_dir / "combined_recipe.pkl",
        results_dir / "pm3_best_recipe.pkl",
        results_dir / "no_earlystop" / "pm3_best_recipe.pkl",
    ]
    study_paths.extend(sorted((results_dir / "pm3_grid").glob("*.pkl")))

    for path in study_paths:
        results = load_results(path)
        study_name = path.relative_to(results_dir).as_posix()
        for run_name, (metrics, _) in results.items():
            row = {"study": study_name, "run_name": run_name}
            row.update(summarize_run(metrics))
            yield row


def write_summary_csv(results_dir: Path, rows: List[dict]) -> Path:
    """Write the packaged scalar summary as CSV.

    Inputs:
        results_dir: Root results directory.
        rows: Summary rows.
    Outputs:
        Output CSV path.
    """
    out_path = results_dir / "final_results_summary.csv"
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def plot_monitoring_grid(results: RunDict, title: str, out_path: Path) -> None:
    """Plot the core monitoring curves for a study.

    Inputs:
        results: Mapping from run name to (metrics, snapshots).
        title: Figure title.
        out_path: Output PNG path.
    Outputs:
        Saves plot to disk.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    keys = [
        ("val_matched_dist", "Validation matched distance"),
        ("train_loss", "Train loss"),
        ("pred_diversity", "Prediction diversity"),
        ("softmax_entropy", "Softmax entropy"),
    ]
    for run_name, (metrics, _) in results.items():
        epochs = metrics["epoch"]
        for ax, (key, label) in zip(axes, keys):
            ax.plot(epochs, metrics[key], linewidth=2, label=run_name)
            ax.set_title(label)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_grid_file(path: Path) -> Tuple[float, int, int]:
    """Extract tau / slotted / warmup flags from a grid pickle name.

    Inputs:
        path: Grid pickle path.
    Outputs:
        Tuple of (tau, slotted, warmup).
    """
    stem = path.stem
    parts = stem.replace("pm3_tau", "").split("_")
    tau = float(parts[0])
    slotted = int(parts[1].replace("slots", ""))
    warmup = int(parts[2].replace("warmup", ""))
    return tau, slotted, warmup


def build_grid_tables(results_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 2x2x2 tables for the PM3 grid scan.

    Inputs:
        results_dir: Root results directory.
    Outputs:
        Arrays for best distance, final distance, and best epoch.
    """
    best_dist = np.zeros((2, 2, 2), dtype=float)
    final_dist = np.zeros((2, 2, 2), dtype=float)
    best_epoch = np.zeros((2, 2, 2), dtype=float)
    tau_index = {0.12: 0, 0.15: 1}

    for path in sorted((results_dir / "pm3_grid").glob("*.pkl")):
        tau, slotted, warmup = parse_grid_file(path)
        run_name, (metrics, _) = next(iter(load_results(path).items()))
        summary = summarize_run(metrics)
        idx = tau_index[tau]
        best_dist[idx, slotted, warmup] = summary["best_dist"]
        final_dist[idx, slotted, warmup] = summary["final_dist"]
        best_epoch[idx, slotted, warmup] = summary["best_epoch"]
        _ = run_name
    return best_dist, final_dist, best_epoch


def plot_pm3_grid_heatmaps(results_dir: Path) -> List[Path]:
    """Create the PM3 grid heatmaps.

    Inputs:
        results_dir: Root results directory.
    Outputs:
        List of saved figure paths.
    """
    best_dist, final_dist, best_epoch = build_grid_tables(results_dir)
    outputs = []
    plot_specs = [
        ("pm3_grid_best_dist.png", best_dist, "Best val_matched_dist", ".3f"),
        ("pm3_grid_final_dist.png", final_dist, "Final val_matched_dist", ".3f"),
        ("pm3_grid_best_epoch.png", best_epoch, "Epoch of best distance", ".0f"),
    ]
    for filename, array, title, fmt in plot_specs:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
        vmin = float(np.min(array))
        vmax = float(np.max(array))
        for idx, tau in enumerate([0.12, 0.15]):
            ax = axes[idx]
            im = ax.imshow(array[idx], cmap="viridis_r" if "dist" in filename else "magma", vmin=vmin, vmax=vmax)
            ax.set_title(f"{title}\nτ={tau}")
            ax.set_xticks([0, 1], labels=["warmup=0", "warmup=1"])
            ax.set_yticks([0, 1], labels=["slots=0", "slots=1"])
            for row in range(2):
                for col in range(2):
                    ax.text(col, row, format(array[idx, row, col], fmt), ha="center", va="center", color="white", fontsize=11)
        fig.colorbar(im, ax=axes, shrink=0.9)
        out_path = results_dir / filename
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def plot_recipe_comparison(results_dir: Path) -> Path:
    """Plot the key PM3 recipe checkpoints as a horizontal bar chart.

    Inputs:
        results_dir: Root results directory.
    Outputs:
        Output PNG path.
    """
    def best_from(path: Path, run_name: str | None = None) -> float:
        results = load_results(path)
        if run_name is None:
            _, (metrics, _) = next(iter(results.items()))
        else:
            metrics, _ = results[run_name]
        return summarize_run(metrics)["best_dist"]

    labels = [
        "warmup only\n(lr_schedule warmup_cosine)",
        "tau only\n(tau=0.15)",
        "slots only\n(PM3 slotted)",
        "grid best\n(tau=0.12 + warmup)",
        "tau=0.15 + slots + warmup\n(grid rerun)",
        "tau=0.15 + slots + warmup\n(no early stop)",
        "default PM3\n(tau=0.12)",
    ]
    values = [
        best_from(results_dir / "lr_schedule.pkl", "warmup_cosine"),
        best_from(results_dir / "tau_sweep.pkl", "τ=0.15"),
        best_from(results_dir / "slot_embeddings.pkl", "PM3 (slotted)"),
        best_from(results_dir / "pm3_grid" / "pm3_tau0.12_slots0_warmup1.pkl"),
        best_from(results_dir / "pm3_grid" / "pm3_tau0.15_slots1_warmup1.pkl"),
        best_from(results_dir / "no_earlystop" / "pm3_best_recipe.pkl"),
        best_from(results_dir / "hungarian_baseline.pkl", "PM3 τ=0.12"),
    ]
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(np.array(labels)[order], np.array(values)[order], color="steelblue")
    ax.set_xlabel("Best val_matched_dist (lower is better)")
    ax.set_title("PM3 recipe comparison")
    ax.grid(True, axis="x", alpha=0.3)
    out_path = results_dir / "pm3_recipe_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def cost_threshold_rows(results_dir: Path, threshold: float = 0.80) -> List[dict]:
    """Compute time-to-threshold statistics for the cost study.

    Inputs:
        results_dir: Root results directory.
        threshold: Distance threshold to test.
    Outputs:
        List of summary dicts per run.
    """
    rows = []
    for run_name, (metrics, _) in load_results(results_dir / "cost_comparison.pkl").items():
        values = metrics["val_matched_dist"]
        epoch_time_s = metrics.get("epoch_time_s", [])
        hit_epoch = next((i for i, value in enumerate(values) if value <= threshold), None)
        rows.append(
            {
                "run_name": run_name,
                "best_dist": summarize_run(metrics)["best_dist"],
                "epoch_to_best": summarize_run(metrics)["best_epoch"],
                "time_to_best_min": summarize_run(metrics)["wall_to_best_min"],
                "epoch_to_threshold_0p80": hit_epoch,
                "time_to_threshold_0p80_min": (
                    float(np.sum(epoch_time_s[: hit_epoch + 1])) / 60.0 if hit_epoch is not None else math.nan
                ),
                "gpu_ms_per_batch": metrics.get("gpu_ms_per_batch", math.nan),
                "wall_ms_per_batch": metrics.get("wall_ms_per_batch", math.nan),
            }
        )
    return rows


def write_interpretation_md(results_dir: Path, rows: List[dict]) -> Path:
    """Write a concise markdown summary of the final results.

    Inputs:
        results_dir: Root results directory.
        rows: Summary rows from all packaged studies.
    Outputs:
        Output markdown path.
    """
    row_map = {(row["study"], row["run_name"]): row for row in rows}
    cost_rows = cost_threshold_rows(results_dir)
    best_grid = min(
        [row for row in rows if row["study"].startswith("pm3_grid/")],
        key=lambda row: row["best_dist"],
    )

    md = [
        "# CLEVR Final Results Summary",
        "",
        "## Headline checks",
        "",
        f"- Best schedule run: `warmup_cosine` at `{row_map[('lr_schedule.pkl', 'warmup_cosine')]['best_dist']:.4f}`.",
        f"- Best tau-only run: `τ=0.15` at `{row_map[('tau_sweep.pkl', 'τ=0.15')]['best_dist']:.4f}`.",
        f"- Best slot-only PM3 run: `PM3 (slotted)` at `{row_map[('slot_embeddings.pkl', 'PM3 (slotted)')]['best_dist']:.4f}`.",
        f"- Combined recipe winner: `Hungarian (slotted + warmup_cosine)` at `{row_map[('combined_recipe.pkl', 'Hungarian (slotted + warmup_cosine)')]['best_dist']:.4f}`.",
        f"- No-early-stop PM3 rerun: `{row_map[('no_earlystop/pm3_best_recipe.pkl', 'PM3 τ=0.15 (slotted + warmup_cosine)')]['best_dist']:.4f}`.",
        f"- Best 2x2x2 PM3 grid point: `{best_grid['run_name']}` at `{best_grid['best_dist']:.4f}`.",
        "",
        "## Interpretation notes",
        "",
        "- PM3 clearly benefits from either `warmup_cosine` alone or `τ=0.15` alone, but the full `τ=0.15 + slots + warmup` recipe is not additive.",
        "- The PM3 grid points to `τ=0.12 + warmup_cosine + standard decoder` as the strongest clean PM3 recipe in this study (`0.5828`).",
        "- Slot embeddings help PM3 when introduced on their own, but they appear to interact poorly with the stronger schedule / temperature recipe.",
        "- Early stopping was a real confounder for the PM3 best-recipe run: removing it improved best distance from `0.7039` to `0.6607`.",
        "- Even after fixing early stopping, the slotted+warmup+τ=0.15 PM3 recipe still underperforms the best single-factor PM3 variants, so the regression is not just truncation.",
        "- Hungarian improves dramatically under the combined recipe (`0.4314`), which suggests the architectural/schedule changes are not universally bad; the issue is PM3-specific interaction, not a broken training stack.",
        "- The PM3 grid and the no-early-stop rerun show some instability across nominally similar settings, so the combined recipe likely has a narrower optimization basin than the simpler PM3 variants.",
        "",
        "## Cost comparison",
        "",
    ]
    for row in cost_rows:
        threshold_text = "not reached" if math.isnan(row["time_to_threshold_0p80_min"]) else f"{row['time_to_threshold_0p80_min']:.2f} min"
        md.append(
            f"- `{row['run_name']}`: best `{row['best_dist']:.4f}`, "
            f"time to best `{row['time_to_best_min']:.2f} min`, "
            f"time to `0.80` `{threshold_text}`, "
            f"GPU `{row['gpu_ms_per_batch']:.2f} ms/batch`, wall `{row['wall_ms_per_batch']:.2f} ms/batch`."
        )
    md.extend(
        [
            "",
            "## Suggested next steps",
            "",
            "- Use the current CLEVR study as parameter discovery and mechanism analysis, not as the final community benchmark claim.",
            "- Carry forward two PM3 candidates: the strongest clean PM3 point from the grid and the best single-factor schedule/tau wins.",
            "- Use COCO DETR-style detection as the next benchmark for matching/cost relevance to the wider literature.",
        ]
    )

    out_path = results_dir / "final_results_summary.md"
    out_path.write_text("\n".join(md))
    return out_path


def main() -> None:
    """Package all CLEVR result artifacts.

    Inputs:
        CLI arguments only.
    Outputs:
        Writes plots and summary files under the results directory.
    """
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()

    rows = list(iter_summary_rows(results_dir))
    if not rows:
        raise RuntimeError(f"No results found under {results_dir}")

    csv_path = write_summary_csv(results_dir, rows)
    notes_path = write_interpretation_md(results_dir, rows)
    plot_monitoring_grid(
        load_results(results_dir / "combined_recipe.pkl"),
        "Combined recipe monitoring",
        results_dir / "combined_recipe_monitoring.png",
    )
    plot_monitoring_grid(
        load_results(results_dir / "no_earlystop" / "pm3_best_recipe.pkl"),
        "PM3 best recipe without early stopping",
        results_dir / "pm3_best_recipe_no_earlystop_monitoring.png",
    )
    heatmaps = plot_pm3_grid_heatmaps(results_dir)
    recipe_plot = plot_recipe_comparison(results_dir)

    print(f"Wrote {csv_path}")
    print(f"Wrote {notes_path}")
    for path in heatmaps:
        print(f"Wrote {path}")
    print(f"Wrote {recipe_plot}")


if __name__ == "__main__":
    main()
