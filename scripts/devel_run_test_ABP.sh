#!/usr/bin/env bash

# Lightweight local runner for tests/test_ABP/run_test_ABP.
# Loads the same module stack used on the cluster, activates py3, and runs the
# ABP test binary under mpirun (default: 1 rank to match the test config).
# Usage:
#   scripts/devel_run_test_ABP.sh [config_path] [binary] [-- extra mpirun args...]
# Examples:
#   scripts/devel_run_test_ABP.sh
#   scripts/devel_run_test_ABP.sh tests/test_ABP/config.json build/test_ABP_run_test_ABP
#   scripts/devel_run_test_ABP.sh tests/test_ABP/config.json build/test_ABP_run_test_ABP -- --bind-to none

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

module load StdEnv
module load GCCcore/13.3.0
module load CUDA/12.6.0
module load OpenMPI/5.0.3-GCC-13.3.0-CUDA-12.6.0
module load UCX-CUDA/1.16.0-GCCcore-13.3.0-CUDA-12.6.0
module load NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
module load UCC-CUDA/1.3.0-GCCcore-13.3.0-CUDA-12.6.0
module load miniconda/24.11.3
module load git/2.45.1-GCCcore-13.3.0
module load CMake/3.29.3-GCCcore-13.3.0
module load nlohmann_json/3.11.3-GCCcore-13.3.0

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not available after loading miniconda module." >&2
    exit 1
fi
eval "$(conda shell.bash hook)"
conda activate py3

CONFIG_PATH="${1:-tests/test_ABP/config.json}"
BINARY="${2:-build/test_ABP_run_test_ABP}"
shift $(( $# >= 1 ? 1 : 0 ))
shift $(( $# >= 1 ? 1 : 0 ))

# Remaining args (if any) are passed to mpirun.
MPIRUN_EXTRAS=("$@")

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "[ERROR] Config file '${CONFIG_PATH}' not found." >&2
    exit 1
fi

if [ ! -x "${BINARY}" ]; then
    echo "[ERROR] ABP test binary '${BINARY}' is not executable." >&2
    echo "        Build it first, e.g.:  cmake --build build --target test_ABP_run_test_ABP" >&2
    exit 2
fi

MPI_RANKS="${MPI_RANKS:-1}"

echo "Running devel ABP test with:"
echo "  repo root : ${REPO_ROOT}"
echo "  config    : ${CONFIG_PATH}"
echo "  binary    : ${BINARY}"
echo "  mpirun    : -np ${MPI_RANKS} ${MPIRUN_EXTRAS[*]:-<none>}"
echo "Launching mpirun..."

mpirun --oversubscribe -np "${MPI_RANKS}" "${MPIRUN_EXTRAS[@]}" "${BINARY}" "${CONFIG_PATH}"
