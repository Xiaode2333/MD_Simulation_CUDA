1. run NPT, NPE, save *.xyz. 16K particles
2. show NPE not working, draw mark in diagram. Show NVT, NPE interface figures.

write python/analysis/analysis_NPH_xyz_batches.ipynb
to read from saved data of results/20260309_NPH_batch_xyz_saving and results/20260311_NPH_batch_xyz_saving_piston.
1. Call already written frame snapshot plotting functions, to plot everly 20 snapshots for both data and both phases.
2. Calculate order parameter and plot.
3. Calculate density of A/B along x direction and plot.
4. 