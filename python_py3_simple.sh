#!/bin/bash
# Run Python from a configurable Conda environment without hard-coded paths.

set -euo pipefail

environment_name="${MD_CONDA_ENV:-py3}"
if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available; load or install it before running this wrapper" >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$environment_name"
exec python3 "$@"
