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
├── build
├── CMakeLists.txt
├── docs
│   └── figs
│       └── md_snapshot.svg
├── include
│   ├── md_common.hpp
│   ├── md_config.hpp
│   ├── md_cuda_common.hpp
│   ├── md_env.hpp
│   └── md_particle.hpp
├── legacy
│   └── md_simulation.cu
├── python
│   ├── particle_csv.py
│   ├── plot_particle_python.py
│   ├── plot_triangulation_python.py
│   └── __pycache__
│       └── particle_csv.cpython-312.pyc
├── README.md
├── scripts
│   ├── run_test.sh
│   ├── set_slurm_env.sh
│   └── vscode_slurm.sh
├── src
│   ├── md_common.cpp
│   ├── md_config.cpp
│   ├── md_cuda_common.cu
│   ├── md_env.cu
│   └── md_particle.cpp
├── tests
│   ├── env_plot
│   │   ├── env_plot.cpp
│   │   ├── init_frame.pdf
│   │   ├── init_frame.png
│   │   └── md_config.json
│   ├── md_config_load_save
│   │   ├── test_config
│   │   ├── test_config.cpp
│   │   └── test_config_output.json
│   ├── mpi_build_worlds
│   │   ├── mpi_test
│   │   ├── mpi_test.cu
│   │   └── mpi_test.sh
│   ├── plot_basic
│   │   ├── basic.png
│   │   ├── plot_basic
│   │   └── plot_basic.cpp
│   ├── plot_particles
│   │   ├── particles.png
│   │   ├── plot_particles
│   │   └── plot_particles.cpp
│   ├── run_test
│   │   ├── config.json
│   │   ├── config_large.json
│   │   ├── config_small.json
│   │   ├── csv
│   │   ├── frames
│   │   └── run_test.cpp
│   └── save_load_frame
│       ├── frames.bin
│       ├── save_load_frame
│       └── save_load_frame.cpp
├── vcpkg-configuration.json
├── vcpkg.json
└── vscode_slurm.out

---

## 🛠️ Installation & Usage (Slurm Environment)

This project relies on **vcpkg** for C++ dependency management and **Conda** for Python-based visualization/analysis tools. Follow these steps to set up the necessary environment components on a Slurm cluster.

### 1. Install vcpkg

Clone the vcpkg repository to your home directory and run the bootstrap script so CMake can pick up the toolchain:

```bash
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh
```

### 2. Set up the Python environment

Create a Conda environment named `py3` containing Python 3.12, NumPy, and Matplotlib (the plotting scripts depend on these packages):

```bash
module load miniconda     # or whichever Conda module your cluster provides
conda create -n py3 python=3.12 numpy matplotlib -y
```

### 3. Run the simulation

Submit the provided Slurm script from the project root; it loads the required modules (CUDA, MPI, Conda), builds the project via CMake/vcpkg, and launches the test driver:

```bash
sbatch scripts/run_test.sh
squeue --me    # optional: monitor job status
```

> **Tip:** You can edit `scripts/run_test.sh` to customize module versions, build options, or runtime arguments for your cluster.
