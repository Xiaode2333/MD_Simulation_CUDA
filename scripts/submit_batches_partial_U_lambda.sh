#!/bin/bash

set -euo pipefail

BASE_ROOT="results/20251209_partial_U_lambda_series"
ORI_CONFIG="${BASE_ROOT}/config.json"

if [ ! -f "$ORI_CONFIG" ]; then
    echo "[ERROR] ori_config '$ORI_CONFIG' not found. Please place the base config there first." >&2
    exit 1
fi

#
# One-time configure + build for this commit.
# All submitted jobs will reuse the same binary in a
# commit-specific build directory under ./build_slurm_tmp/.
# (Avoid 'module reset' here because on this cluster it
# tries to call 'conda' before it exists, which breaks
# when this script is run with 'set -e'.)
#
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

# module list

export CUDA_HOME="/apps/software/2024a/software/CUDA/12.6.0"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Source conda profile directly to ensure 'conda' command exists
conda init
# source /apps/software/2022b/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate py3
export LD_LIBRARY_PATH="/apps/software/2024a/software/CUDA/12.6.0/lib64:/apps/software/2024a/software/CUDA/12.6.0/lib:${LD_LIBRARY_PATH:-}"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export VCPKG_CMAKE="$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake"
export PYTHONPATH=$(python -c "import site; print(site.getsitepackages()[0])")

PY_EXEC="$(which python)"

if ! GIT_HASH=$(git rev-parse HEAD 2>/dev/null); then
    echo "[ERROR] Failed to get git commit hash. Is this a git repo and is git available?" >&2
    exit 1
fi

BUILD_ROOT="./build_slurm_tmp/build_${GIT_HASH}"
SERIES_BIN="${BUILD_ROOT}/run_series_partial_U_lambda_test"

mkdir -p "$BUILD_ROOT"

if [ ! -f "${BUILD_ROOT}/CMakeCache.txt" ] || [ ! -x "$SERIES_BIN" ]; then
    echo "[INFO] Configuring and building in '${BUILD_ROOT}' for commit ${GIT_HASH}."
    cmake -B "$BUILD_ROOT" -S . \
        -DCMAKE_TOOLCHAIN_FILE="$VCPKG_CMAKE" \
        -DVCPKG_TARGET_TRIPLET=x64-linux \
        -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
        -DCUDAToolkit_ROOT="$CUDA_HOME" \
        -DCMAKE_C_COMPILER=mpicc \
        -DCMAKE_CXX_COMPILER=mpicxx \
        -DPython3_EXECUTABLE="$PY_EXEC" \
        -DOMPI_CUDA_PREFIX="/apps/software/2024a/software/OpenMPI/5.0.3-GCC-13.3.0-CUDA-12.6.0" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    cmake --build "$BUILD_ROOT" -j --target run_series_partial_U_lambda_test
else
    echo "[INFO] Reusing existing build in '${BUILD_ROOT}' for commit ${GIT_HASH}."
fi


# Temperatures from 0.5 to 1.0 (inclusive) in steps of 0.1
for T in 0.5 0.6 0.7 0.8 0.9 1.0; do
    T_DIR="${BASE_ROOT}/T_${T}"
    mkdir -p "$T_DIR"

    # Lambdas: 0.0, 0.025, 0.05, 0.075, 0.1, ..., 1.0
    for lambda in 0.025 0.075 0.125 0.175 0.225 0.275 0.325 0.375 0.425 0.475 0.525 0.575 0.625 0.675 0.725 0.775 0.825 0.875 0.925 0.975; do
        LAMBDA_DIR="${T_DIR}/lambda_${lambda}"
        mkdir -p "$LAMBDA_DIR"

        echo "Submitting T=${T}, lambda=${lambda} into ${LAMBDA_DIR}"
        sbatch --job-name="T${T}_lambda${lambda}" \
            scripts/run_series_partial_U_lambda.sh \
            "$LAMBDA_DIR" \
            "$ORI_CONFIG" \
            "$SERIES_BIN" \
            "DT_target=${T}" \
            "lambda-deform=${lambda}"
    done
done
