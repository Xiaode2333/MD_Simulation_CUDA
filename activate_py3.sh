#!/bin/bash
# Script to activate py3 conda environment for Claude Code
# This script sources the conda environment and can be used as a hook or wrapper

# Load miniconda module (if available)
module load miniconda 2>/dev/null || {
    echo "Warning: Failed to load miniconda module" >&2
}

# Activate conda environment
# Note: conda activate needs to be sourced, so this script should be sourced
# or we can manually set the PATH and CONDA_PREFIX
if [ -n "$CONDA_PREFIX" ]; then
    echo "Already in conda environment: $CONDA_PREFIX" >&2
else
    # Try to activate py3 environment
    source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null || {
        echo "Error: Failed to source conda.sh" >&2
        exit 1
    }
    conda activate py3 2>/dev/null || {
        echo "Error: Failed to activate py3 conda environment" >&2
        exit 1
    }
    echo "Activated py3 conda environment" >&2
fi

# If arguments are provided, execute them
if [ $# -gt 0 ]; then
    exec "$@"
fi