#!/bin/bash
# Wrapper script to run Python commands in the py3 conda environment
# Usage: python_py3_wrapper.sh <python_args>

# Source the activation script without passing through our arguments.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
args=("$@")
set --
source "$SCRIPT_DIR/activate_py3.sh"
set -- "${args[@]}"

# Now execute Python with all arguments
exec python3 "$@"
