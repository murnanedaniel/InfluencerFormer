# CLEVR Studies

This directory is the organized home for the CLEVR loss and matching studies.

## Layout

- `lib/`: shared CLEVR study utilities and compatibility imports
- `scripts/`: study entrypoints and launch wrappers
- `slides/`: slide entrypoints and deck wrappers
- `results/`: packaged summary artifacts for the final study pass
- `notebooks/`: study notes and pointers to the legacy notebook analyses

## Current canonical code

The historical CLEVR implementation still lives in the repo root for
compatibility with previous runs:

- `scripts/train_study.py`
- `scripts/plot_results.py`
- `scripts/package_clevr_results.py`
- `scripts/train_pm3_grid_point.py`
- `notebooks/clevr_utils.py`
- `slides/results_deck.tex`
- `results/`

New code should prefer the organized import path:

- `studies.clevr.lib.clevr_utils`

and the organized entrypoint wrappers under `studies/clevr/scripts/`.
