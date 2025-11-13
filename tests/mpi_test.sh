#!/bin/bash
#SBATCH --job-name=mpi_cuda_test
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --partition=pi_co54

module purge
module load CUDA
module load OpenMPI

which nvcc
which mpicxx

rm -f tests/mpi_test
nvcc -ccbin mpicxx -arch=sm_89 tests/mpi_test.cu -o tests/mpi_test

srun tests/mpi_test

