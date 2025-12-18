#!/usr/bin/env bash

# Lightweight local test runner for series_LiquidGas_ABP.
# Runs the binary under mpirun with 4 MPI ranks in the current shell
# (no sbatch, assumes your environment already has MPI/CUDA/etc. loaded).
#
# Usage:
#   scripts/devel_test_series_LiquidGas_ABP.sh <base_dir> <ori_config> [series_bin] [D overrides...]
#
# Examples:
#   scripts/devel_test_series_LiquidGas_ABP.sh \
#       results/devel_LG_ABP_test \
#       results/20251210_LG_series/config.json
#
#   scripts/devel_test_series_LiquidGas_ABP.sh \
#       results/devel_LG_ABP_test \
#       results/20251210_LG_series/config.json \
#       ./build/run_series_LiquidGas_ABP \
#       DT_init=0.1 Ddt=1e-4

set -euo pipefail

module load StdEnv
module load GCCcore/13.3.0
module load CUDA/12.6.0
module load OpenMPI/5.0.3-GCC-13.3.0-CUDA-12.6.0
module load UCX-CUDA/1.16.0-GCCcore-13.3.0-CUDA-12.6.0
module load NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
module load UCC-CUDA/1.3.0-GCCcore-13.3.0-CUDA-12.6.0
module load miniconda/24.11.3
module load git/2.45.1-GCCcore-13.3.0
module load CMake/3.29.3-GCCcore-13.3.0
module load nlohmann_json/3.11.3-GCCcore-13.3.0

conda activate py3

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <base_dir> <ori_config> [series_bin] [D overrides...]" >&2
    exit 1
fi

BASE_DIR="$1"
ORI_CONFIG="$2"
shift 2

# Optional explicit binary path, otherwise default to build/run_series_LiquidGas_ABP
if [ "$#" -ge 1 ] && [[ "$1" == */run_series_LiquidGas_ABP || "$1" == run_series_LiquidGas_ABP* ]]; then
    SERIES_BIN="$1"
    shift 1
else
    SERIES_BIN="build_slurm_tmp/build_4129821591160b4df265c39e78e3b44ec592a7db/run_series_LiquidGas_ABP"
fi

if [ ! -f "$ORI_CONFIG" ]; then
    echo "[ERROR] Config file '$ORI_CONFIG' not found." >&2
    exit 1
fi

if [ ! -x "$SERIES_BIN" ]; then
    echo "[ERROR] series_LiquidGas_ABP binary '$SERIES_BIN' is not executable." >&2
    echo "        Build it first, e.g.:  cmake --build build --target run_series_LiquidGas_ABP" >&2
    exit 2
fi

mkdir -p -- "$BASE_DIR"

# Translate override arguments into --Dkey=value form, same as batch script.
override_cli=()
for override in "$@"; do
    if [ -z "$override" ]; then
        continue
    fi
    if [[ "$override" == --* ]]; then
        override_cli+=("$override")
    else
        override_cli+=("--${override}")
    fi
done

echo "Running devel series_LiquidGas_ABP with:"
echo "  base dir : $BASE_DIR"
echo "  config   : $ORI_CONFIG"
echo "  binary   : $SERIES_BIN"
echo "  overrides: ${override_cli[*]:-<none>}"
echo "Launching mpirun with 4 ranks (oversubscribed if needed)..."

mpirun --oversubscribe -np 4 "$SERIES_BIN" \
    --base-dir "$BASE_DIR" \
    --ori-config "$ORI_CONFIG" \
    "${override_cli[@]}"
