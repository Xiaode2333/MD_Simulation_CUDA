#!/bin/bash
#SBATCH --partition=pi_co54
#SBATCH -c 1
#SBATCH -t 12:00:00
#SBATCH --mem=15G
#SBATCH --output=vscode_slurm.out

module load VSCode
code tunnel
