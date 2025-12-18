#!/bin/bash

set -euo pipefail

BASE_ROOT="results/20251218_LG_ABP_series2"
ORI_CONFIG="${BASE_ROOT}/config.json"

if [ ! -f "$ORI_CONFIG" ]; then
    echo "[ERROR] ori_config '$ORI_CONFIG' not found. Please place the base config there first." >&2
    exit 1
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
module load CMake/3.29.3-GCCcore-13.3.0
module load nlohmann_json/3.11.3-GCCcore-13.3.0

export CUDA_HOME="/apps/software/2024a/software/CUDA/12.6.0"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

conda init
conda activate py3

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export VCPKG_CMAKE="$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake"
export PYTHONPATH=$(python -c "import site; print(site.getsitepackages()[0])")

PY_EXEC="$(which python)"

if ! GIT_HASH=$(git rev-parse HEAD 2>/dev/null); then
    echo "[ERROR] Failed to get git commit hash. Is this a git repo and is git available?" >&2
    exit 1
fi

BUILD_ROOT="./build_slurm_tmp/build_${GIT_HASH}"
SERIES_BIN="${BUILD_ROOT}/run_series_LiquidGas_ABP"

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
    cmake --build "$BUILD_ROOT" -j 32 --target run_series_LiquidGas_ABP
else
    echo "[INFO] Reusing existing build in '${BUILD_ROOT}' for commit ${GIT_HASH}."
fi

LAMBDAS=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for V0 in 1 2 5 10 20 50 100; do
    V0_DIR="${BASE_ROOT}/v0_${V0}"
    mkdir -p "$V0_DIR"

    N_LAMBDAS=${#LAMBDAS[@]}
    echo "Submitting array for v0=${V0} with ${N_LAMBDAS} lambdas"

    sbatch --array=0-$((N_LAMBDAS-1)) scripts/run_series_LiquidGas_ABP.sh \
        "$BASE_ROOT" \
        "$ORI_CONFIG" \
        "$SERIES_BIN" \
        "$V0"
done
