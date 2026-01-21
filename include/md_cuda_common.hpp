#pragma once

#include "md_particle.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

#define CUDA_CHECK(call)                                                         \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__,      \
                    __LINE__, cudaGetErrorString(err));                         \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

__host__ __device__ inline double pbc_wrap_hd(double x, double L) {
    if (L <= 0.0) {
        return x;
    }
    x = fmod(x, L);
    if (x < 0.0) {
        x += L;
    }
    return x;
}

// Confine particles to a central slab along x for liquid–gas setups.
// The slab has width slab_width (typically p * Lx + 2 * sigma_max),
// is centered at Lx / 2, and uses reflective boundaries in x (no PBC).
// In y we still apply standard PBC wrapping.
__global__ void middle_reflect_LG_kernel(Particle *particles, int n, double Lx,
                                         double Ly, double slab_width);

// device kernel to mark halo particles
__global__ void mark_halo_kernel(const Particle *particles, int n_local,
                                 double x_min, double x_max, double Lx,
                                 double halo_width, int *flags_left,
                                 int *flags_right);

// device kernel to pack selected halo particles using prefix-sum positions
__global__ void pack_halo_kernel(const Particle *particles, int n_local,
                                 const int *flags, const int *pos,
                                 int max_count, Particle *out_buf);

// mark particles in d_local with flags_left, flags_right, flags_keep
__global__ void mark_migration_kernel(const Particle *particles, int n_local,
                                      double x_min, double x_max, int rank_idx,
                                      int rank_size, int left_rank,
                                      int right_rank, int *flags_left,
                                      int *flags_right, int *flags_keep);

__global__ void pack_selected_kernel(const Particle *particles, int n_local,
                                     const int *flags, const int *pos,
                                     int n_selected, Particle *out);

__global__ void li_force_kernel(Particle *particles, Particle *halo_left,
                                Particle *halo_right, int n_local, int n_left,
                                int n_right, double Lx, double Ly,
                                double sigma_AA, double sigma_BB,
                                double sigma_AB, double epsilon_AA,
                                double epsilon_BB, double epsilon_AB,
                                double cutoff, double mass_0, double mass_1);

// Compute per-block partial sums of dU/dlambda on the device.
// For each interacting pair with force F = (Fx, Fy) and separation dr = (dx,
// dy), the pair contribution is -epsilon_lambda * (Fx * dx - Fy * dy), with 0.5
// factor inside the kernel to avoid double counting.
__global__ void cal_partial_U_lambda_kernel(
        const Particle *__restrict__ particles,
        const Particle *__restrict__ halo_left,
        const Particle *__restrict__ halo_right, int n_local, int n_left,
        int n_right, double Lx, double Ly, double sigma_AA, double sigma_BB,
        double sigma_AB, double epsilon_AA, double epsilon_BB, double epsilon_AB,
        double cutoff, double epsilon_lambda, double *__restrict__ partial_sums);

// Verlocity-Verlot half kick
// from r^{n}, v^{n}, a^{n} to r^{n+1}, v^{n+1/2}, a^{n};
__global__ void step_half_vv_kernel(Particle *particles, int n_local, double dt,
                                    double Lx, double Ly);

// 2nd half kick. From r^{n+1}, v^{n+1/2}, a^{n+1} to r^{n+1}, v^{n+1}, a^{n+1};
__global__ void step_2nd_half_vv_kernel(Particle *particles, int n_local,
                                        double dt);

__global__ void step_half_vv_nh_kernel(Particle *particles, int n_local,
                                       double dt, double xi, double Lx,
                                       double Ly);

__global__ void step_2nd_half_vv_nh_kernel(Particle *particles, int n_local,
                                           double dt, double xi);

__global__ void cal_local_K_kernel(const Particle *__restrict__ particles,
                                   int n_local, double mass_A, double mass_B,
                                   double *__restrict__ partial_sums);

__global__ void cal_local_U_kernel(const Particle *__restrict__ particles,
                                   const Particle *__restrict__ halo_left,
                                   const Particle *__restrict__ halo_right,
                                   int n_local, int n_left, int n_right,
                                   double Lx, double Ly, double sigma_AA,
                                   double sigma_BB, double sigma_AB,
                                   double epsilon_AA, double epsilon_BB,
                                   double epsilon_AB, double cutoff,
                                   double *__restrict__ partial_sums);

__global__ void
local_density_profile_kernel(const Particle *__restrict__ particles,
                             int n_local, int n_bins_per_rank, double xmin,
                             double xmax, int *count_A, int *count_B);

// Local Irving–Kirkwood pressure tensor profile along y.
// Domain [0, Ly) is split into n_bins_local bins; this kernel accumulates
// unnormalized rank-local P_xx, P_yy, P_xy per bin, using:
//   kinetic: m_i v_i^α v_i^β at particle y_i
//   virial:  (1/2) r_{ij}^α F_{ij}^β at pair midpoint (minimum image).
// To get physical pressure, sum over ranks and divide by (Lx *
// Ly/n_bins_local).
__global__ void local_pressure_tensor_profile_kernel(
        const Particle *__restrict__ particles,
        const Particle *__restrict__ halo_left,
        const Particle *__restrict__ halo_right, int n_local, int n_left,
        int n_right, double Lx, double Ly, double mass_A, double mass_B,
        double sigma_AA, double sigma_BB, double sigma_AB, double epsilon_AA,
        double epsilon_BB, double epsilon_AB, double cutoff, int n_bins_local,
        double *__restrict__ P_xx, double *__restrict__ P_yy,
        double *__restrict__ P_xy);

// ABP (Active Brownian Particle) overdamped integration kernel
// Implements Euler-Maruyama scheme for overdamped Langevin dynamics with
// self-propulsion
__global__ void
step_ABP_kernel(Particle *particles, int n_local, double dt, double Lx,
                double Ly,
                double mu,           // Mobility coefficient
                double v0,           // Self-propulsion speed
                double sqrt_2Dr_dt,  // √(2·D_r·Δt) for translational noise
                double sqrt_2Dth_dt, // √(2·D_θ·Δt) for rotational noise
                curandState *rng_states);

// Initialize RNG states for ABP simulation (one per particle)
__global__ void init_ABP_rng_kernel(curandState *states, int n,
                                    unsigned long long seed);

// Compute eigenvalues/eigenvectors of a symmetric sparse matrix on device (CSR),
// returning the smallest num_eig eigenpairs in ascending order. The matrix data
// may be overwritten internally (no preservation guarantee).
// Follows the Lanczos workflow used in tests/eigenvalue/test_eigenvalue.cu.
// Arguments:
//   n_rows        : matrix dimension
//   nnz           : number of nonzeros
//   d_row_offsets : device array length n_rows+1 (CSR offsets)
//   d_col_indices : device array length nnz (CSR column indices)
//   d_values      : device array length nnz (CSR values)
//   num_eig       : number of smallest eigenpairs to compute
//   d_eigvals     : device output array length num_eig
//   d_eigvecs     : device output matrix (column-major) size n_rows x num_eig
//   max_iterations, ncv, tolerance, seed : solver parameters (mirroring test)
template <typename IndexT, typename ValueT>
void cal_eigen(int n_rows, int nnz, const IndexT *d_row_offsets,
               const IndexT *d_col_indices, const ValueT *d_values,
               int num_eig, int max_iterations, int ncv, ValueT tolerance,
               uint64_t seed, ValueT *d_eigvals, ValueT *d_eigvecs);

extern template void cal_eigen<int32_t, float>(int, int, const int32_t *,
                                               const int32_t *, const float *,
                                               int, int, int, float, uint64_t,
                                               float *, float *);
extern template void cal_eigen<int32_t, double>(int, int, const int32_t *,
                                                const int32_t *, const double *,
                                                int, int, int, double, uint64_t,
                                                double *, double *);
