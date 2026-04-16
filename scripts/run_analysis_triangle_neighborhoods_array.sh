#!/bin/bash
#SBATCH --partition=pi_co54
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=24G
#SBATCH --mail-type=FAIL

set -euo pipefail

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[ERROR] SLURM_ARRAY_TASK_ID is not set." >&2
    exit 1
fi

if [ -z "${TEMP_DIR_LIST:-}" ]; then
    echo "[ERROR] TEMP_DIR_LIST is not set." >&2
    exit 2
fi

if [ -z "${ANALYSIS_BIN:-}" ]; then
    echo "[ERROR] ANALYSIS_BIN is not set." >&2
    exit 3
fi

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
ANALYSIS_DEVICE="${ANALYSIS_DEVICE:-gpu}"
ANALYSIS_OVERWRITE="${ANALYSIS_OVERWRITE:-0}"

case "$ANALYSIS_DEVICE" in
    cpu|gpu)
        ;;
    *)
        echo "[ERROR] ANALYSIS_DEVICE must be 'cpu' or 'gpu', got '$ANALYSIS_DEVICE'." >&2
        exit 4
        ;;
esac

if ! type module >/dev/null 2>&1; then
    if [ -r /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    elif [ -r /usr/share/Modules/init/bash ]; then
        source /usr/share/Modules/init/bash
    elif [ -r /usr/share/lmod/lmod/init/bash ]; then
        source /usr/share/lmod/lmod/init/bash
    else
        echo "[ERROR] 'module' command is unavailable and module init scripts were not found." >&2
        exit 5
    fi
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
module load CMake/3.31.8-GCCcore-13.3.0
module load nlohmann_json/3.11.3-GCCcore-13.3.0

module list

auto_conda_sh=""
if command -v conda >/dev/null 2>&1; then
    auto_conda_sh="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
fi
if [ -n "$auto_conda_sh" ] && [ -r "$auto_conda_sh" ]; then
    source "$auto_conda_sh"
elif [ -r /apps/software/2022b/software/miniconda/24.11.3/etc/profile.d/conda.sh ]; then
    source /apps/software/2022b/software/miniconda/24.11.3/etc/profile.d/conda.sh
else
    echo "[ERROR] Could not find conda.sh to activate py3." >&2
    exit 6
fi
set +u
conda activate py3
set -u

export CUDA_HOME="/apps/software/2024a/software/CUDA/12.6.0"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/apps/software/2024a/software/CUDA/12.6.0/lib64:/apps/software/2024a/software/CUDA/12.6.0/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

temp_dir="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TEMP_DIR_LIST")"
if [ -z "$temp_dir" ]; then
    echo "[ERROR] No temperature directory found for array index ${SLURM_ARRAY_TASK_ID}." >&2
    exit 7
fi

xyz_file="${temp_dir}/trajectory_nvt.xyz"
tri2d_file="${xyz_file}.tri2d"
output_file="${temp_dir}/aab_abb_triangle_neighbors.csv"

if [ ! -f "$xyz_file" ]; then
    echo "[ERROR] XYZ file '$xyz_file' does not exist." >&2
    exit 8
fi
if [ ! -f "$tri2d_file" ]; then
    echo "[ERROR] TRI2D file '$tri2d_file' does not exist." >&2
    exit 9
fi
if [ ! -x "$ANALYSIS_BIN" ]; then
    echo "[ERROR] Analysis binary '$ANALYSIS_BIN' is not executable." >&2
    exit 10
fi

analysis_args=(
    --xyz "$xyz_file"
    --tri2d "$tri2d_file"
    --output "$output_file"
    --device "$ANALYSIS_DEVICE"
)
if [ "$ANALYSIS_OVERWRITE" = "1" ]; then
    analysis_args+=(--overwrite)
fi

echo "[INFO] Running mixed-triangle neighborhood analysis for '${temp_dir}'"
cd "$REPO_ROOT"
srun --cpu-bind=none "$ANALYSIS_BIN" "${analysis_args[@]}"
