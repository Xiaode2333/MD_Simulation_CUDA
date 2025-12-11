#!/usr/bin/env bash

# Delete all interfaces/csv folders under the 20251209_partial_U_lambda_series results.
# Run this from the project root:
#   bash scripts/cleanup_partial_U_lambda_interfaces_csv.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

rm -rf results/20251209_partial_U_lambda_series/*/*/interfaces/csv

