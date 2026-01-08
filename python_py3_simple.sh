#!/bin/bash
# Simple wrapper that sets PATH to py3 conda environment and runs Python
# This works in subprocesses without needing to source conda

# Set environment variables
export CONDA_PREFIX="/home/bh692/.conda/envs/py3"
export CONDA_DEFAULT_ENV="py3"
export PATH="/home/bh692/.conda/envs/py3/bin:$PATH"

# Execute Python with all arguments
exec python3 "$@"