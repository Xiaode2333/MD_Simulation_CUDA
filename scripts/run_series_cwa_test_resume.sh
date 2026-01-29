#!/bin/bash
#SBATCH --partition=pi_co54
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=128G
#SBATCH --output=./results/series_cwa_test_%j.out

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <base_dir> <ori_config> DT_init=0.1 Ddt=1e-4 ..." >&2
    exit 1
fi

BASE_DIR="$1"
ORI_CONFIG="$2"
shift 2

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
module load miniconda/24.11.3
module load git/2.45.1-GCCcore-13.3.0
module load CMake/3.31.8-GCCcore-13.3.0
module load nlohmann_json/3.11.3-GCCcore-13.3.0

module list

export CUDA_HOME="/apps/software/2024a/software/CUDA/12.6.0"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Source conda profile directly to ensure 'conda' command exists
source /apps/software/2022b/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate py3
export LD_LIBRARY_PATH="/apps/software/2024a/software/CUDA/12.6.0/lib64:/apps/software/2024a/software/CUDA/12.6.0/lib:${LD_LIBRARY_PATH:-}"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export VCPKG_CMAKE="$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake"
export PYTHONPATH=$(python -c "import site; print(site.getsitepackages()[0])")

PY_EXEC="$(which python)"

cmake -B build -S . \
    -DCMAKE_TOOLCHAIN_FILE="$VCPKG_CMAKE" \
    -DVCPKG_TARGET_TRIPLET=x64-linux \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DCUDAToolkit_ROOT="$CUDA_HOME" \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DPython3_EXECUTABLE="$PY_EXEC" \
    -DOMPI_CUDA_PREFIX="/apps/software/2024a/software/OpenMPI/5.0.3-GCC-13.3.0-CUDA-12.6.0" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j

SERIES_BIN="./build/run_series_cwa_test_resume"
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

echo "Launching series_cwa_test with base dir '${BASE_DIR}' and config '${ORI_CONFIG}'."
srun --cpu-bind=none "$SERIES_BIN" --base-dir "$BASE_DIR" --ori-config "$ORI_CONFIG" "${override_cli[@]}"