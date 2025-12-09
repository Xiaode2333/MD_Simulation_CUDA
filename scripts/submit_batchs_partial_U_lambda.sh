#!/bin/bash

set -euo pipefail

BASE_ROOT="results/20251209_partial_U_lambda_series"
ORI_CONFIG="${BASE_ROOT}/config.json"

if [ ! -f "$ORI_CONFIG" ]; then
    echo "[ERROR] ori_config '$ORI_CONFIG' not found. Please place the base config there first." >&2
    exit 1
fi

# Temperatures from 0.5 to 1.0 (inclusive) in steps of 0.1
for T in 0.5 0.6 0.7 0.8 0.9 1.0; do
    T_DIR="${BASE_ROOT}/T_${T}"
    mkdir -p "$T_DIR"

    # Lambdas: 0.0, 0.1, ..., 1.0 (11 points)
    for lambda in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
        LAMBDA_DIR="${T_DIR}/lambda_${lambda}"
        mkdir -p "$LAMBDA_DIR"

        echo "Submitting T=${T}, lambda=${lambda} into ${LAMBDA_DIR}"
        sbatch --job-name="T${T}_lambda${lambda}" \
            scripts/run_series_partial_U_lambda.sh \
            "$LAMBDA_DIR" \
            "$ORI_CONFIG" \
            "T_target=${T}" \
            "lambda-deform=${lambda}"
    done
done

