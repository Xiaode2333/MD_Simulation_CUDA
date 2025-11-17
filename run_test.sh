#!/bin/bash
#SBATCH --job-name=MD_Sim_test
#SBATCH --partition=pi_co54
#SBATCH --time=01:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G

./set_slurm_env.sh

cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=$VCPKG_CMAKE -DVCPKG_TARGET_TRIPLET=x64-linux -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DPython3_EXECUTABLE="$(poetry run which python)"
cmake --build build -j

srun ./build/run_test_run_test
