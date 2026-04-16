#!/bin/bash
#SBATCH --job-name=analysis_20260408
#SBATCH --partition=pi_co54
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --output=./results/analysis_20260408_local_rho/slurm_%j.out
#SBATCH --error=./results/analysis_20260408_local_rho/slurm_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=baichuan.he@yale.edu

set -euo pipefail

if [ -n "${REPO_ROOT:-}" ]; then
    REPO_ROOT="$(cd "${REPO_ROOT}" && pwd -P)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
fi

OUTPUT_NB="${OUTPUT_NB:-results/analysis_20260408_local_rho/analysis_20260408.executed.ipynb}"
N_BINS_OVERRIDE="${N_BINS_OVERRIDE:-}"

mkdir -p "${REPO_ROOT}/results/analysis_20260408_local_rho"

cd "$REPO_ROOT"

if ! type module >/dev/null 2>&1; then
    if [ -r /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    elif [ -r /usr/share/Modules/init/bash ]; then
        source /usr/share/Modules/init/bash
    elif [ -r /usr/share/lmod/lmod/init/bash ]; then
        source /usr/share/lmod/lmod/init/bash
    else
        echo "[ERROR] 'module' command is unavailable and module init scripts were not found." >&2
        exit 1
    fi
fi

module load miniconda/24.11.3

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
    exit 2
fi

set +u
conda activate py3
set -u

if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo "[INFO] Running analysis_20260408 notebook from ${REPO_ROOT}"
echo "[INFO] Python: $(which python3)"
echo "[INFO] Output notebook: ${OUTPUT_NB}"

runner_args=(--output-nb "${OUTPUT_NB}")
if [ -n "${N_BINS_OVERRIDE}" ]; then
    runner_args+=(--n-bins "${N_BINS_OVERRIDE}")
    echo "[INFO] N_BINS override: ${N_BINS_OVERRIDE}"
fi

srun --cpu-bind=none python3 python/run_analysis_20260408.py "${runner_args[@]}"
