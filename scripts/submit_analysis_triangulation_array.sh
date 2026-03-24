#!/bin/bash

set -euo pipefail

JOB_NAME="tri2d_array"
OVERWRITE=0
RESULT_ROOTS=()

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
        --overwrite)
            OVERWRITE=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--job-name NAME] [--overwrite] RESULT_ROOT [RESULT_ROOT ...]" >&2
            exit 0
            ;;
        *)
            RESULT_ROOTS+=("$1")
            shift
            ;;
    esac
done

if [ "${#RESULT_ROOTS[@]}" -eq 0 ]; then
    echo "[ERROR] Provide at least one result root containing .xyz trajectories." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_ROOT="${REPO_ROOT}/results/triangulation_task_arrays/${JOB_NAME}"
XYZ_FILE_LIST="${LOG_ROOT}/xyz_files.txt"

mkdir -p "$LOG_ROOT"

if ! type module >/dev/null 2>&1; then
    if [ -r /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    elif [ -r /usr/share/Modules/init/bash ]; then
        source /usr/share/Modules/init/bash
    elif [ -r /usr/share/lmod/lmod/init/bash ]; then
        source /usr/share/lmod/lmod/init/bash
    else
        echo "[ERROR] 'module' command is unavailable and module init scripts were not found." >&2
        exit 3
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
    exit 4
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
    exit 5
fi

BUILD_ROOT="${REPO_ROOT}/build_slurm_tmp/build_${GIT_HASH}"
ANALYSIS_BIN="${BUILD_ROOT}/analysis_analysis_triangulation"

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
    echo "[INFO] Building target 'analysis_analysis_triangulation' in '${BUILD_ROOT}'."
    cmake --build "$BUILD_ROOT" -j --target analysis_analysis_triangulation
else
    echo "[INFO] Reusing existing build '${BUILD_ROOT}'."
fi

: > "$XYZ_FILE_LIST"
for root in "${RESULT_ROOTS[@]}"; do
    if [ ! -d "$root" ]; then
        echo "[ERROR] Result root '$root' does not exist." >&2
        exit 6
    fi
    find "$(realpath "$root")" -type f -name '*.xyz' | sort >> "$XYZ_FILE_LIST"
done

sort -u -o "$XYZ_FILE_LIST" "$XYZ_FILE_LIST"

TASK_COUNT="$(wc -l < "$XYZ_FILE_LIST")"
if [ "$TASK_COUNT" -eq 0 ]; then
    echo "[ERROR] No .xyz trajectories found under the provided result roots." >&2
    exit 7
fi

ARRAY_END=$((TASK_COUNT - 1))

echo "Submitting triangulation array job"
echo "  job name: ${JOB_NAME}"
echo "  tasks:    ${TASK_COUNT}"
echo "  list:     ${XYZ_FILE_LIST}"
echo "  logs:     ${LOG_ROOT}"

sbatch \
    --job-name="$JOB_NAME" \
    --array="0-${ARRAY_END}" \
    --output="${LOG_ROOT}/slurm-%A_%a.out" \
    --error="${LOG_ROOT}/slurm-%A_%a.err" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",XYZ_FILE_LIST="${XYZ_FILE_LIST}",ANALYSIS_BIN="${ANALYSIS_BIN}",TRI_OVERWRITE="${OVERWRITE}" \
    "${SCRIPT_DIR}/run_analysis_triangulation_array.sh"
