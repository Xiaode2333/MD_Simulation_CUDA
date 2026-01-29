#!/bin/bash
#SBATCH --partition=pi_co54
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --array=0-40
#SBATCH --output=./results/20260113_test_area_with_num_tri_types_%A_%a.out
#SBATCH --mail-type=FAIL

set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <T_dir> <ori_config> <series_bin> <T_value> [extra_overrides...]" >&2
    exit 1
fi

BASE_ROOT="$1"  # T-specific root
ORI_CONFIG="$2"
SERIES_BIN="$3"
T_VALUE="$4"
shift 4

if [ ! -f "$ORI_CONFIG" ]; then
    echo "[ERROR] Config file '$ORI_CONFIG' not found." >&2
    exit 1
fi

# rm -rf build

module reset
module --force purge

module load StdEnv
module load GCCcore/13.3.0
module load CUDA/12.6.0
module load OpenMPI/5.0.3-GCC-13.3.0-CUDA-12.6.0
module load UCX-CUDA/1.16.0-GCCcore-13.3.0-CUDA-12.6.0
module load NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
module load UCC-CUDA/1.3.0-GCCcore-13.3.0-CUDA-12.6.0
module load miniconda/24.7.1
module load git/2.45.1-GCCcore-13.3.0
module load CMake/3.31.8-GCCcore-13.3.0
module load nlohmann_json/3.11.3-GCCcore-13.3.0

module list

export CUDA_HOME="/apps/software/2024a/software/CUDA/12.6.0"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Source conda profile directly to ensure 'conda' command exists
source /apps/software/2022b/software/miniconda/24.7.1/etc/profile.d/conda.sh
conda activate py3
export LD_LIBRARY_PATH="/apps/software/2024a/software/CUDA/12.6.0/lib64:/apps/software/2024a/software/CUDA/12.6.0/lib:${LD_LIBRARY_PATH:-}"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

if [ ! -x "$SERIES_BIN" ]; then
    echo "[ERROR] $SERIES_BIN was not produced." >&2
    exit 2
fi

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[ERROR] SLURM_ARRAY_TASK_ID is not set; this script is intended for array jobs." >&2
    exit 3
fi

LAMBDA_INDEX="${SLURM_ARRAY_TASK_ID}"
LAMBDA_VALUE=$(python - <<'PY'
import os
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
if idx < 0 or idx > 40:
    raise SystemExit(f"lambda index {idx} out of range [0,40]")
lam = idx / 40.0
print(f"{lam:.8f}")
PY
)
LAMBDA_DIR_NAME=$(printf "lambda_%0.6f" "$LAMBDA_VALUE")
BASE_DIR="${BASE_ROOT}/${LAMBDA_DIR_NAME}"

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
override_cli+=("--DT_target=${T_VALUE}")
override_cli+=("--lambda-deform=${LAMBDA_VALUE}")
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

echo "Launching series_test_area_with_num_tri_types (T=${T_VALUE}, lambda=${LAMBDA_VALUE}) with base dir '${BASE_DIR}' and config '${ORI_CONFIG}'."
srun --cpu-bind=none "$SERIES_BIN" --base-dir "$BASE_DIR" --ori-config "$ORI_CONFIG" "${override_cli[@]}"
