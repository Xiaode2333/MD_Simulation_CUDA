#!/bin/bash
#SBATCH --job-name=MD_Sim_test
#SBATCH --partition=pi_co54
#SBATCH --time=01:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=./results/test_%j.out

module reset
module --force purge

# Load an explicit, dependency-compatible stack to avoid module churn warnings.
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

# rm -rf build
# rm -rf build-debug

export CUDA_HOME="/apps/software/2024a/software/CUDA/12.6.0"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Ensure conda is initialised before activating the env.
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda command not found after loading miniconda module." >&2
    exit 1
fi
eval "$(conda shell.bash hook)"
conda activate py3

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

srun --cpu-bind=none ./build/run_test_run_test

# cmake -B build-debug -S . \
#     -DCMAKE_BUILD_TYPE=RelWithDebInfo \
#     -DCMAKE_TOOLCHAIN_FILE="$VCPKG_CMAKE" \
#     -DVCPKG_TARGET_TRIPLET=x64-linux \
#     -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
#     -DCUDAToolkit_ROOT="$CUDA_HOME" \
#     -DCMAKE_C_COMPILER=mpicc \
#     -DCMAKE_CXX_COMPILER=mpicxx \
#     -DPython3_EXECUTABLE="$PY_EXEC" \
#     -DOMPI_CUDA_PREFIX="/apps/software/2024a/software/OpenMPI/5.0.3-NVHPC-25.1-CUDA-12.6.0"
# cmake --build build-debug -j

# if [ ! -x ./build-debug/run_test_run_test ]; then
#     echo "[ERROR] ./build-debug/run_test_run_test was not produced." >&2
#     exit 2
# fi

# srun --cpu-bind=none gdb \
#     -ex "set pagination off" \
#     -ex "catch throw std::runtime_error" \
#     -ex run \
#     -ex bt \
#     -batch \
#     --args ./build-debug/run_test_run_test

# srun --cpu-bind=none gdb --args ./build-debug/run_test_run_test
