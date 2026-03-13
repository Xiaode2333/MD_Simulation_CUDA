#!/bin/bash
#SBATCH --partition=pi_co54
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --array=0-5
#SBATCH --output=./results/20260311_NPH_batch_xyz_saving_piston/slurm_%A_%a.out
#SBATCH --mail-type=FAIL

set -euo pipefail

TEMPS=(0.5 0.6 0.7 0.8 0.9 1.0)

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <base_root> <ori_config> <series_bin> [extra_overrides...]" >&2
    echo "Example extra override: DP_target=0.0" >&2
    exit 1
fi

BASE_ROOT="$1"
ORI_CONFIG="$2"
SERIES_BIN="$3"
shift 3

if [ ! -f "$ORI_CONFIG" ]; then
    echo "[ERROR] Config file '$ORI_CONFIG' not found." >&2
    exit 1
fi

if [ ! -x "$SERIES_BIN" ]; then
    echo "[ERROR] Binary '$SERIES_BIN' was not produced or is not executable." >&2
    exit 2
fi

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[ERROR] SLURM_ARRAY_TASK_ID is not set; this script is intended for array jobs." >&2
    exit 3
fi

TASK_ID="${SLURM_ARRAY_TASK_ID}"
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#TEMPS[@]}" ]; then
    echo "[ERROR] SLURM_ARRAY_TASK_ID=$TASK_ID is out of range [0,$(( ${#TEMPS[@]} - 1 ))]." >&2
    exit 4
fi

T_VALUE="${TEMPS[$TASK_ID]}"
BASE_DIR="${BASE_ROOT}/T=${T_VALUE}"
mkdir -p -- "$BASE_DIR"
CONFIGS_DIR="${BASE_DIR}/configs"
mkdir -p -- "$CONFIGS_DIR"
ORI_CONFIG_NAME="$(basename "$ORI_CONFIG")"
ORI_CONFIG_STEM="${ORI_CONFIG_NAME%.json}"

if ! type module >/dev/null 2>&1; then
    if [ -r /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    elif [ -r /usr/share/Modules/init/bash ]; then
        source /usr/share/Modules/init/bash
    elif [ -r /usr/share/lmod/lmod/init/bash ]; then
        source /usr/share/lmod/lmod/init/bash
    else
        echo "[ERROR] 'module' command is unavailable and module init scripts were not found." >&2
        exit 5
    fi
fi

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
    exit 6
fi
set +u
conda activate py3
set -u

export CUDA_HOME="/apps/software/2024a/software/CUDA/12.6.0"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/apps/software/2024a/software/CUDA/12.6.0/lib64:/apps/software/2024a/software/CUDA/12.6.0/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"

if ! GIT_HASH=$(git rev-parse HEAD 2>/dev/null); then
    GIT_HASH="unknown"
fi
RUN_TS=$(date +"%Y-%m-%d %H:%M")

cat > "${BASE_DIR}/version.json" <<EOF_JSON
{
  "git_hash": "${GIT_HASH}",
  "timestamp": "${RUN_TS}",
  "temperature": ${T_VALUE},
  "slurm_array_task_id": ${TASK_ID}
}
EOF_JSON

cp -f -- "$ORI_CONFIG" "${CONFIGS_DIR}/${ORI_CONFIG_STEM}.input.json"

override_cli=()
override_cli+=("--DT_init=${T_VALUE}")
override_cli+=("--DT_target=${T_VALUE}")
has_barostat_mass_override=0

for override in "$@"; do
    if [ -z "$override" ]; then
        continue
    fi
    case "$override" in
        Dbarostat_mass=*|--Dbarostat_mass=*)
            has_barostat_mass_override=1
            ;;
    esac
    if [[ "$override" == --* ]]; then
        override_cli+=("$override")
    else
        override_cli+=("--${override}")
    fi
done

if [ "$has_barostat_mass_override" -eq 0 ]; then
    override_cli+=("--Dbarostat_mass=16384.0")
fi

echo "Launching run_series_NPH_batch_xyz_saving_piston for T=${T_VALUE} with base dir '${BASE_DIR}'."
srun --cpu-bind=none "$SERIES_BIN" --base-dir "$BASE_DIR" --ori-config "$ORI_CONFIG" "${override_cli[@]}"
