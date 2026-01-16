#!/bin/bash
# Submit analysis_test_area_with_tri_number.ipynb execution for all temperatures (one job per T).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
PY_RUNNER="${REPO_ROOT}/python/run_analysis_test_area_with_tri_number.py"
NOTEBOOK_SRC="${REPO_ROOT}/python/analysis_test_area_with_tri_number.ipynb"

BASE_ROOT_DEFAULT="${REPO_ROOT}/results/20260113_test_area_with_num_tri_types"
BASE_ROOT="${BASE_ROOT:-$BASE_ROOT_DEFAULT}"

if [ ! -f "$PY_RUNNER" ]; then
    echo "[ERROR] Cannot find runner at '$PY_RUNNER'." >&2
    exit 1
fi

if [ ! -f "$NOTEBOOK_SRC" ]; then
    echo "[ERROR] Cannot find notebook at '$NOTEBOOK_SRC'." >&2
    exit 1
fi

if [ ! -d "$BASE_ROOT" ]; then
    echo "[ERROR] Base results directory '$BASE_ROOT' does not exist." >&2
    exit 1
fi

BASE_ROOT="$(cd "$BASE_ROOT" && pwd)"

LOG_DIR="${REPO_ROOT}/logs/analysis_test_area_with_tri_number"
mkdir -p "$LOG_DIR"

temps=()
# Collect temperature labels from directories T_<T> (excluding *_analysis) and tar files T_<T>.tar
for path in "${BASE_ROOT}"/T_*; do
    [ -e "$path" ] || continue
    name="${path##*/}"
    # Skip analysis outputs
    if [[ "$name" == *"_analysis" ]]; then
        continue
    fi
    # Strip prefix and optional .tar suffix
    t_val="${name#T_}"
    t_val="${t_val%.tar}"
    temps+=("$t_val")
done

if [ "${#temps[@]}" -eq 0 ]; then
    echo "[ERROR] No temperature directories or tar files found under '$BASE_ROOT'." >&2
    exit 1
fi

IFS=$'\n' temps=($(printf "%s\n" "${temps[@]}" | sort -V | uniq))
unset IFS

for T in "${temps[@]}"; do
    echo "[INFO] Submitting notebook analysis for T=${T}"
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=analysis_nb_T${T}
#SBATCH --partition=pi_co54
#SBATCH --output=${LOG_DIR}/analysis_nb_T${T}_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --chdir=${REPO_ROOT}
#SBATCH --mail-type=FAIL

set -euo pipefail

module load miniconda
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate py3

python "${PY_RUNNER}" -T "${T}" -b "${BASE_ROOT}" -i "${NOTEBOOK_SRC}"
EOF
done
