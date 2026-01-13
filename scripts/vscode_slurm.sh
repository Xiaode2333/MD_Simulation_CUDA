#!/bin/bash
#SBATCH --partition=day
#SBATCH -c 1
#SBATCH -t 10:00:00
#SBATCH --mem=15G
#SBATCH --output=vscode_slurm.out

module load VSCode
code tunnel
