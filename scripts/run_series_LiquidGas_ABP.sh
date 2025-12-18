#!/bin/bash
#SBATCH --partition=pi_co54
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=./results/series_LiquidGas_ABP_%j.out

set -euo pipefail

LAMBDAS_ARRAY=(0 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4 0.425 0.45 0.475 0.5 0.525 0.55 0.575 0.6 0.625 0.65 0.675 0.7 0.725 0.75 0.775 0.8 0.825 0.85 0.875 0.9 0.925 0.95 0.975 1.0)

MODE="direct"
BASE_DIR=""
BASE_ROOT=""
V0_VALUE=""

if [[ -n "${SLURM_ARRAY_TASK_ID-}" && "$#" -ge 4 ]]; then
    # Array mode: <base_root> <ori_config> <series_bin> <v0>
    MODE="array"
    BASE_ROOT="$1"
    ORI_CONFIG="$2"
    SERIES_BIN="$3"
    V0_VALUE="$4"
    shift 4

    idx="${SLURM_ARRAY_TASK_ID}"
    lambda="${LAMBDAS_ARRAY[$idx]}"
    BASE_DIR="${BASE_ROOT}/v0_${V0_VALUE}/lambda_${lambda}"
else
    # Direct mode: original interface
    if [ "$#" -lt 3 ]; then
        echo "Usage: $0 <base_dir> <ori_config> <series_bin> [DT_init=0.1 Ddt=1e-4 ...]" >&2
        exit 1
    fi

    BASE_DIR="$1"
    ORI_CONFIG="$2"
    SERIES_BIN="$3"
    shift 3
fi

if [ ! -f "$ORI_CONFIG" ]; then
    echo "[ERROR] Config file '$ORI_CONFIG' not found." >&2
    exit 1
fi

module reset
module --force purge

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

module list

export CUDA_HOME="/apps/software/2024a/software/CUDA/12.6.0"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

conda init
conda activate py3

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

if [ ! -x "$SERIES_BIN" ]; then
    echo "[ERROR] $SERIES_BIN was not produced." >&2
    exit 2
fi

mkdir -p -- "$BASE_DIR"

if ! GIT_HASH=$(git rev-parse HEAD 2>/dev/null); then
    GIT_HASH="unknown"
fi
RUN_TS=$(date +"%Y-%m-%d %H:%M")

cat > "${BASE_DIR}/version.json" <<EOF
{
  "git_hash": "${GIT_HASH}",
  "timestamp": "${RUN_TS}"
}
EOF

override_cli=()

if [ "$MODE" = "array" ]; then
    # In array mode, always set v0 and match T_init/T_target to v0; lambda from SLURM_ARRAY_TASK_ID.
    override_cli+=( "--Dv0=${V0_VALUE}" )
    override_cli+=( "--DT_init=${V0_VALUE}" )
    override_cli+=( "--DT_target=${V0_VALUE}" )
    override_cli+=( "--lambda-deform=${lambda}" )
fi

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

echo "Launching series_LiquidGas_ABP with base dir '${BASE_DIR}' and config '${ORI_CONFIG}'."
srun --cpu-bind=none "$SERIES_BIN" --base-dir "$BASE_DIR" --ori-config "$ORI_CONFIG" "${override_cli[@]}"
