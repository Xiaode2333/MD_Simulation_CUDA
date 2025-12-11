#!/bin/bash
#SBATCH --partition=pi_co54
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH --mem=15G
#SBATCH --output=vscode_slurm.out

module load VSCode
code tunnel
