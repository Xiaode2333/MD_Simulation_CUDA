#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/submit_analysis_triangulation_array.sh" \
    --backend "cpu" \
    --job-name "tri2d_xyz_batches_20260325_type0" \
    "$@" \
    "results/20260323_NPH_batch_xyz_saving_all_type0" \
    "results/20260323_NPH_batch_xyz_saving_piston_all_type0" \
    "results/20260323_NVT_batch_xyz_saving_all_type0"
