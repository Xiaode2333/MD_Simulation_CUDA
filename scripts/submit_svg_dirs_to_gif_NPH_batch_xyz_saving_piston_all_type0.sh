#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/submit_svg_dirs_to_gif_array.sh" \
    "results/20260323_NPH_batch_xyz_saving_piston_all_type0" \
    "${1:-4}" \
    "gif_nph_piston_all0_20260323"
