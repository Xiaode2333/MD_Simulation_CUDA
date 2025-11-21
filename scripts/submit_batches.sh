for T in 0.5 0.6 0.7 0.8 0.9 1.0; do
    sbatch --job-name=$T scripts/run_series_strain_test.sh \
    "results/20251120_strain_series/T_$T" \
    "results/20251120_strain_series/config.json" \
    DT_init=$T DT_target=$T
done