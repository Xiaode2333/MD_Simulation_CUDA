#!/bin/bash
# Submit entropy/free-energy analysis jobs for all temperatures.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
PY_SCRIPT="${REPO_ROOT}/python/analysis_batch_mutual_entropy.py"

BASE_ROOT_DEFAULT="${REPO_ROOT}/results/20260108_test_area_with_num_tri_types"
BASE_ROOT="${BASE_ROOT:-$BASE_ROOT_DEFAULT}"

if [ ! -f "$PY_SCRIPT" ]; then
    echo "[ERROR] Cannot find analysis script at '$PY_SCRIPT'." >&2
    exit 1
fi

if [ ! -d "$BASE_ROOT" ]; then
    echo "[ERROR] Base results directory '$BASE_ROOT' does not exist." >&2
    exit 1
fi

BASE_ROOT="$(cd "$BASE_ROOT" && pwd)"

LOG_DIR="${REPO_ROOT}/logs/analysis_batch_mutual_entropy"
mkdir -p "$LOG_DIR"

temps=()
for T_DIR in "${BASE_ROOT}"/T_*; do
    [ -d "$T_DIR" ] || continue
    t_val="${T_DIR##*/}"
    t_val="${t_val#T_}"
    temps+=("$t_val")
done

if [ "${#temps[@]}" -eq 0 ]; then
    echo "[ERROR] No temperature directories found under '$BASE_ROOT'." >&2
    exit 1
fi

IFS=$'\n' temps=($(printf "%s\n" "${temps[@]}" | sort -V))
unset IFS

for T in "${temps[@]}"; do
    T_BASE="${BASE_ROOT}/T_${T}"
    if [ ! -d "$T_BASE" ]; then
        echo "[WARN] Skipping missing temperature directory '$T_BASE'." >&2
        continue
    fi

    echo "[INFO] Submitting mutual entropy analysis for T=${T} (base: ${T_BASE})"
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=mutual_S_T${T}
#SBATCH --partition=pi_co54
#SBATCH --output=${LOG_DIR}/mutual_entropy_T${T}_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --chdir=${REPO_ROOT}

set -euo pipefail

module load miniconda
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate py3

python "${PY_SCRIPT}" -T "${T}" -b "${T_BASE}"
EOF
done
