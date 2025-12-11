#!/usr/bin/env bash

# Cancel all Slurm jobs for user bh692 except those whose job name contains "vscode".
# Usage (on the cluster login node):
#   bash scripts/scancel_except_vscode.sh

set -euo pipefail

USER_NAME="bh692"

# List jobs (job ID and name), filter out those containing "vscode", then cancel by ID.
squeue -u "${USER_NAME}" -h -o '%A %j' | \
    awk '!/vscode/ {print $1}' | \
    xargs -r scancel

