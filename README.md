# CUDA MD Simulation (Multi-GPU + MPI)

## Overview

This project implements a 2D molecular dynamics (MD) simulator for a binary Lennard-Jones mixture using CUDA and MPI. It supports:

- Domain decomposition along the x-direction
- Halo (ghost) exchange between neighboring MPI ranks
- Single or multi-GPU execution per rank
- Velocity-Verlet integrator with optional Nose–Hoover thermostat (NVT)
- Mixed-precision Lennard-Jones force evaluation (pairwise in `float`, accumulation in `double`)
- Compressed binary I/O for trajectories and simple visualization helpers

The core goal is to run large-scale MD on clusters with multiple GPUs and nodes, and to analyze interfacial properties (e.g. line tension via capillary waves).

---

## Main Features

**Dynamics and Physics**

- Binary Lennard-Jones mixture (types `A` and `B`)
- Separate LJ parameters:
  - `SIGMA_AA`, `SIGMA_BB`, `SIGMA_AB`
  - `EPSILON_AA`, `EPSILON_BB`, `EPSILON_AB`
- Optional Nose–Hoover thermostat for NVT
- Velocity-Verlet integrator:
  - `step_single()` for NVE
  - `step_single_nose_hoover()` for NVT
- Periodic boundary conditions in both x and y

**Parallelization**

- MPI domain decomposition along x
- Halo regions (`halo_left`, `halo_right`) for short-range interactions across rank boundaries
- Device-resident particle arrays:
  - `d_particles` for local particles
  - `d_particles_halo_left`, `d_particles_halo_right` for halos
- GPU kernels for:
  - Force computation (LJ)
  - Time integration
  - Kinetic and potential energy reductions

**I/O and Analysis**

- Compressed binary trajectory output (`gz` files) for all particles
- Simple plotting utility `print_particles` (using `matplotlibcpp`) to generate PNG snapshots
- Infrastructure for interfacial analysis (capillary waves, interface length, etc.)

---

## Examples of Simulation

<p align="center">
  <img src="docs/figs/md_snapshot.svg"
       alt="Snapshot of Simulation"
       style="width:100%; height:auto;">
</p>

<p align="center"><em>Snapshot of Simulation.</em></p>


## Code Structure (Typical Layout)

Your repository may look similar to:

```text
.
├── CMakeLists.txt
├── README.md
├── docs
│   └── figs
├── external
│   └── matplotlib-cpp
├── include
│   ├── md_common.hpp
│   ├── md_config.hpp
│   ├── md_cuda_common.hpp
│   ├── md_env.hpp
│   └── md_particle.hpp
├── scripts
├── src
│   ├── md_common.cpp
│   ├── md_config.cpp
│   ├── md_cuda_common.cu
│   ├── md_env.cu
│   └── md_particle.cpp
├── tests
│   ├── env_plot
│   ├── md_config_load_save
│   ├── mpi_build_worlds
│   ├── plot_basic
│   ├── plot_particles
│   ├── run_test
│   └── save_load_frame
├── vcpkg-configuration.json
└── vcpkg.json
