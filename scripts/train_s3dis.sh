#!/usr/bin/env bash
# train_s3dis.sh — wrapper for OneFormer3D training on S3DIS Area 5
#
# Usage:
#   ./scripts/train_s3dis.sh baseline [EXTRA_ARGS...]
#   ./scripts/train_s3dis.sh influencerformer [EXTRA_ARGS...]
#   ./scripts/train_s3dis.sh influencerformer --resume work_dirs/run1/epoch_256.pth
#
# Environment:
#   ONEFORMER3D_ROOT   Path to OneFormer3D repo root (auto-detected if unset)
#
# Requirements:
#   - OneFormer3D installed (pip install -e /path/to/oneformer3d)
#   - influencerformer installed or on PYTHONPATH (handled automatically below)
#   - S3DIS data prepared according to OneFormer3D data-prep scripts
#   - configs/s3dis/  data_root updated to point to your S3DIS superpoint data
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Locate OneFormer3D ────────────────────────────────────────────────────────
if [[ -z "${ONEFORMER3D_ROOT:-}" ]]; then
    ONEFORMER3D_ROOT="$(python -c \
        "import oneformer3d, os; print(os.path.dirname(os.path.dirname(oneformer3d.__file__)))" \
        2>/dev/null || true)"
fi

if [[ -z "${ONEFORMER3D_ROOT:-}" ]]; then
    echo "ERROR: Cannot locate OneFormer3D."
    echo "       Either set ONEFORMER3D_ROOT or install OneFormer3D:"
    echo "           pip install -e /path/to/oneformer3d"
    exit 1
fi

TRAIN_SCRIPT="$ONEFORMER3D_ROOT/tools/train.py"
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "ERROR: tools/train.py not found at $TRAIN_SCRIPT"
    echo "       Check that ONEFORMER3D_ROOT=$ONEFORMER3D_ROOT is correct."
    exit 1
fi

# ── Parse mode ────────────────────────────────────────────────────────────────
MODE="${1:-influencerformer}"
shift || true   # remaining args passed through to train.py

case "$MODE" in
    baseline)
        CONFIG="$REPO_ROOT/configs/s3dis/oneformer3d_1xb4_s3dis-area-5.py"
        ;;
    influencerformer)
        CONFIG="$REPO_ROOT/configs/s3dis/influencerformer_1xb4_s3dis-area-5.py"
        ;;
    *)
        echo "ERROR: Unknown mode '$MODE'."
        echo "       Use 'baseline' or 'influencerformer'."
        echo ""
        echo "Usage: ./scripts/train_s3dis.sh <baseline|influencerformer> [EXTRA_ARGS...]"
        exit 1
        ;;
esac

# ── Ensure influencerformer is importable ─────────────────────────────────────
# Exports REPO_ROOT so 'import influencerformer' works without pip install.
# The configs already contain custom_imports to trigger the MODELS registration.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ── Run ───────────────────────────────────────────────────────────────────────
echo "OneFormer3D root : $ONEFORMER3D_ROOT"
echo "Config           : $CONFIG"
echo "PYTHONPATH       : $PYTHONPATH"
echo ""

python "$TRAIN_SCRIPT" "$CONFIG" "$@"
