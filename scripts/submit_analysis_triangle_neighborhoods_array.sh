#!/bin/bash

set -euo pipefail

DEFAULT_RESULT_ROOT="./results/20260324_NVT_batch_xyz_saving"
JOB_NAME="tri_neighbors_20260324_nvt"
DEVICE="gpu"
OVERWRITE=0
MISSING_ONLY=0
RESULT_ROOT="$DEFAULT_RESULT_ROOT"

while [ $# -gt 0 ]; do
    case "$1" in
        --job-name)
            if [ $# -lt 2 ]; then
                echo "[ERROR] --job-name requires a value." >&2
                exit 1
            fi
            JOB_NAME="$2"
            shift 2
            ;;
        --device)
            if [ $# -lt 2 ]; then
                echo "[ERROR] --device requires a value." >&2
                exit 1
            fi
            DEVICE="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE=1
            shift
            ;;
        --missing-only)
            MISSING_ONLY=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--job-name NAME] [--device cpu|gpu] [--overwrite] [--missing-only] [RESULT_ROOT]" >&2
            exit 0
            ;;
        *)
            RESULT_ROOT="$1"
            shift
            ;;
    esac
done

case "$DEVICE" in
    cpu|gpu)
        ;;
    *)
        echo "[ERROR] --device must be 'cpu' or 'gpu', got '$DEVICE'." >&2
        exit 2
        ;;
esac

if [ ! -d "$RESULT_ROOT" ]; then
    echo "[ERROR] Result root '$RESULT_ROOT' does not exist." >&2
    exit 3
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_ROOT="${REPO_ROOT}/results/triangle_neighbor_task_arrays/${JOB_NAME}"

mkdir -p "$LOG_ROOT"
TEMP_DIR_LIST="$(mktemp "${LOG_ROOT}/temp_dirs.XXXXXX.txt")"

if ! type module >/dev/null 2>&1; then
    if [ -r /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    elif [ -r /usr/share/Modules/init/bash ]; then
        source /usr/share/Modules/init/bash
    elif [ -r /usr/share/lmod/lmod/init/bash ]; then
        source /usr/share/lmod/lmod/init/bash
    else
        echo "[ERROR] 'module' command is unavailable and module init scripts were not found." >&2
        exit 4
    fi
fi

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
    exit 5
fi
set +u
conda activate py3
set -u

export CUDA_HOME="/apps/software/2024a/software/CUDA/12.6.0"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/apps/software/2024a/software/CUDA/12.6.0/lib64:/apps/software/2024a/software/CUDA/12.6.0/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export VCPKG_CMAKE="$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake"

PY_EXEC="$(which python)"

if ! GIT_HASH=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null); then
    echo "[ERROR] Failed to get git commit hash from '$REPO_ROOT'." >&2
    exit 6
fi

BUILD_ROOT="${REPO_ROOT}/build_slurm_tmp/build_${GIT_HASH}"
ANALYSIS_BIN="${BUILD_ROOT}/analysis_analysis_triangle_neighborhoods"

mkdir -p "$BUILD_ROOT"

if [ ! -f "${BUILD_ROOT}/CMakeCache.txt" ]; then
    echo "[INFO] Configuring '${BUILD_ROOT}' for commit ${GIT_HASH}."
    cmake -B "$BUILD_ROOT" -S "$REPO_ROOT" \
        -DCMAKE_TOOLCHAIN_FILE="$VCPKG_CMAKE" \
        -DVCPKG_TARGET_TRIPLET=x64-linux \
        -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
        -DCUDAToolkit_ROOT="$CUDA_HOME" \
        -DCMAKE_C_COMPILER=mpicc \
        -DCMAKE_CXX_COMPILER=mpicxx \
        -DPython3_EXECUTABLE="$PY_EXEC" \
        -DOMPI_CUDA_PREFIX="/apps/software/2024a/software/OpenMPI/5.0.3-GCC-13.3.0-CUDA-12.6.0" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
fi

if [ ! -x "$ANALYSIS_BIN" ]; then
    echo "[INFO] Building target 'analysis_analysis_triangle_neighborhoods' in '${BUILD_ROOT}'."
    cmake --build "$BUILD_ROOT" -j --target analysis_analysis_triangle_neighborhoods
else
    echo "[INFO] Reusing existing build '${BUILD_ROOT}'."
fi

FOUND_COUNT=0
SKIPPED_COUNT=0

: > "$TEMP_DIR_LIST"
while IFS= read -r temp_dir; do
    FOUND_COUNT=$((FOUND_COUNT + 1))
    output_file="${temp_dir}/aab_abb_triangle_neighbors.csv"
    if [ "$MISSING_ONLY" = "1" ] && [ -f "$output_file" ]; then
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        continue
    fi
    if [ ! -f "${temp_dir}/trajectory_nvt.xyz" ] || [ ! -f "${temp_dir}/trajectory_nvt.xyz.tri2d" ]; then
        echo "[WARN] Skipping '${temp_dir}' because trajectory_nvt.xyz or .tri2d is missing." >&2
        continue
    fi
    printf '%s\n' "$(realpath "$temp_dir")" >> "$TEMP_DIR_LIST"
done < <(find "$(realpath "$RESULT_ROOT")" -mindepth 1 -maxdepth 1 -type d -name 'T=*' | sort -V)

TASK_COUNT="$(wc -l < "$TEMP_DIR_LIST")"
if [ "$TASK_COUNT" -eq 0 ]; then
    if [ "$MISSING_ONLY" = "1" ]; then
        echo "[INFO] No missing outputs found under '${RESULT_ROOT}'."
        echo "[INFO] Temperature directories checked: ${FOUND_COUNT}"
        exit 0
    fi
    echo "[ERROR] No matching temperature directories with trajectory_nvt.xyz and .tri2d were found." >&2
    exit 7
fi

ARRAY_END=$((TASK_COUNT - 1))

echo "Submitting mixed-triangle neighborhood array job"
echo "  root:      ${RESULT_ROOT}"
echo "  device:    ${DEVICE}"
echo "  job name:  ${JOB_NAME}"
echo "  mode:      $([ "$MISSING_ONLY" = "1" ] && echo "missing-only" || echo "all temperatures")"
echo "  found:     ${FOUND_COUNT}"
if [ "$MISSING_ONLY" = "1" ]; then
    echo "  skipped:   ${SKIPPED_COUNT}"
fi
echo "  tasks:     ${TASK_COUNT}"
echo "  list:      ${TEMP_DIR_LIST}"
echo "  logs:      ${LOG_ROOT}"

sbatch \
    --job-name="$JOB_NAME" \
    --array="0-${ARRAY_END}" \
    --output="${LOG_ROOT}/slurm-%A_%a.out" \
    --error="${LOG_ROOT}/slurm-%A_%a.err" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",TEMP_DIR_LIST="${TEMP_DIR_LIST}",ANALYSIS_BIN="${ANALYSIS_BIN}",ANALYSIS_DEVICE="${DEVICE}",ANALYSIS_OVERWRITE="${OVERWRITE}" \
    "${SCRIPT_DIR}/run_analysis_triangle_neighborhoods_array.sh"
