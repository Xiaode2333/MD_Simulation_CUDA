#!/bin/bash
#SBATCH --partition=pi_co54
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=./results/series_LiquidGas_%j.out

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <base_dir> <ori_config> <series_bin> [DT_init=0.1 Ddt=1e-4 ...]" >&2
    exit 1
fi

BASE_DIR="$1"
ORI_CONFIG="$2"
SERIES_BIN="$3"
shift 3

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

source /apps/software/2022b/software/miniconda/24.11.3/etc/profile.d/conda.sh
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

echo "Launching series_LiquidGas with base dir '${BASE_DIR}' and config '${ORI_CONFIG}'."
srun --cpu-bind=none "$SERIES_BIN" --base-dir "$BASE_DIR" --ori-config "$ORI_CONFIG" "${override_cli[@]}"

