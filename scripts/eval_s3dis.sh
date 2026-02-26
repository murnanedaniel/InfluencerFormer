#!/usr/bin/env bash
# eval_s3dis.sh — evaluate a trained checkpoint on S3DIS Area 5
#
# Usage:
#   ./scripts/eval_s3dis.sh baseline work_dirs/baseline/epoch_512.pth
#   ./scripts/eval_s3dis.sh influencerformer work_dirs/influencerformer/epoch_512.pth
#
# Environment:
#   ONEFORMER3D_ROOT   Path to OneFormer3D repo root (auto-detected if unset)
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

TEST_SCRIPT="$ONEFORMER3D_ROOT/tools/test.py"
if [[ ! -f "$TEST_SCRIPT" ]]; then
    echo "ERROR: tools/test.py not found at $TEST_SCRIPT"
    echo "       Check that ONEFORMER3D_ROOT=$ONEFORMER3D_ROOT is correct."
    exit 1
fi

# ── Parse arguments ───────────────────────────────────────────────────────────
if [[ $# -lt 2 ]]; then
    echo "Usage: ./scripts/eval_s3dis.sh <baseline|influencerformer> <checkpoint.pth> [EXTRA_ARGS...]"
    exit 1
fi

MODE="$1"
CHECKPOINT="$2"
shift 2   # remaining args passed through to test.py

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
        exit 1
        ;;
esac

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# ── Ensure influencerformer is importable ─────────────────────────────────────
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ── GPU / distributed setup ───────────────────────────────────────────────────
GPUS="${GPUS:-1}"
DIST_TEST="$ONEFORMER3D_ROOT/tools/dist_test.sh"

# ── Run ───────────────────────────────────────────────────────────────────────
echo "OneFormer3D root : $ONEFORMER3D_ROOT"
echo "Config           : $CONFIG"
echo "Checkpoint       : $CHECKPOINT"
echo "GPUs             : $GPUS"
echo ""

if [[ "$GPUS" -gt 1 ]] && [[ -f "$DIST_TEST" ]]; then
    bash "$DIST_TEST" "$CONFIG" "$CHECKPOINT" "$GPUS" "$@"
else
    python "$TEST_SCRIPT" "$CONFIG" "$CHECKPOINT" "$@"
fi
