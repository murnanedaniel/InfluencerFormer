#!/usr/bin/env bash
set -euo pipefail
REPO="/global/cfs/cdirs/m4958/usr/danieltm/Side_Work/ExaTrkX/InfluencerFormer"
exec "${REPO}/studies/clevr/scripts/launch_4gpu.sh" "$@"
