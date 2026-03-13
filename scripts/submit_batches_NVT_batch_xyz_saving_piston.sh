#!/bin/bash

set -euo pipefail

BASE_ROOT="./results/20260313_NVT_batch_xyz_saving_piston"
ORI_CONFIG="./tests/run_NPH_test_piston/config_large_piston.json"

EXTRA_OVERRIDES=("$@")
FIXED_OVERRIDES=(
    "Dbox_h_global=62.4576"
    "Dbox_w_global=374.7456"
    "Dn_particles_global=16384"
    "Dn_particles_type0=8192"
    "Dbarostat_mass=16384"
)

mkdir -p "$BASE_ROOT"

if [ ! -f "$ORI_CONFIG" ]; then
    echo "[ERROR] ori_config '$ORI_CONFIG' not found." >&2
    exit 1
fi

if ! type module >/dev/null 2>&1; then
    if [ -r /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    elif [ -r /usr/share/Modules/init/bash ]; then
        source /usr/share/Modules/init/bash
    elif [ -r /usr/share/lmod/lmod/init/bash ]; then
        source /usr/share/lmod/lmod/init/bash
    else
        echo "[ERROR] 'module' command is unavailable and module init scripts were not found." >&2
        exit 2
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
    exit 3
fi
set +u
conda activate py3
set -u

export CUDA_HOME="/apps/software/2024a/software/CUDA/12.6.0"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/apps/software/2024a/software/CUDA/12.6.0/lib64:/apps/software/2024a/software/CUDA/12.6.0/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
export VCPKG_CMAKE="$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake"

PY_EXEC="$(which python)"

if ! GIT_HASH=$(git rev-parse HEAD 2>/dev/null); then
    echo "[ERROR] Failed to get git commit hash. Is this a git repo?" >&2
    exit 4
fi

BUILD_ROOT="./build_slurm_tmp/build_${GIT_HASH}"
SERIES_BIN="${BUILD_ROOT}/run_series_NVT_batch_xyz_saving_piston"

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
    cmake --build "$BUILD_ROOT" -j --target run_series_NVT_batch_xyz_saving_piston
else
    echo "[INFO] Reusing existing build in '${BUILD_ROOT}' for commit ${GIT_HASH}."
fi

echo "Submitting NVT xyz-saving temperature array (T=0.5..1.0) into ${BASE_ROOT}"
sbatch --job-name="nvt_piston_xyz_20260313" \
    --array=0-5 \
    scripts/run_series_NVT_batch_xyz_saving_piston.sh \
    "$BASE_ROOT" \
    "$ORI_CONFIG" \
    "$SERIES_BIN" \
    "${FIXED_OVERRIDES[@]}" \
    "${EXTRA_OVERRIDES[@]}"
