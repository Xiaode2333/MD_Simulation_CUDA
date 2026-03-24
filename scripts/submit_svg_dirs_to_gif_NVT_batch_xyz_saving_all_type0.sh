#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/submit_svg_dirs_to_gif_array.sh" \
    "results/20260323_NVT_batch_xyz_saving_all_type0" \
    "${1:-4}" \
    "gif_nvt_all0_20260323"
