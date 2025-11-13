#!/bin/bash
#SBATCH --job-name=mpi_cuda_test
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=00:20:00
#SBATCH --partition=pi_co54

module load CUDA
module load OpenMPI
module load miniconda



# If you use conda MPI, activate it here
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate mpi_env

srun ./mpi_cuda_test