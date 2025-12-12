#pragma once

#include "md_particle.hpp"

#include <cuda_runtime.h>


#define CUDA_CHECK(call)                                                \
do {                                                                    \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
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

// Wrap particles into a central slab along x for liquid–gas setups.
// The slab has width slab_width (typically p * Lx + 2 * sigma_max)
// and is centered at Lx / 2.
__global__ void middle_wrap_LG_kernel(Particle* particles,
                                      int n,
                                      double Lx,
                                      double Ly,
                                      double slab_width);

// device kernel to mark halo particles
__global__ void mark_halo_kernel(const Particle* particles,
                                 int n_local,
                                 double x_min,
                                 double x_max,
                                 double Lx,
                                 double halo_width,
                                 int* flags_left,
                                 int* flags_right);

//device kernel to pack selected halo particles using prefix-sum positions
__global__ void pack_halo_kernel(const Particle* particles,
                                 int n_local,
                                 const int* flags,
                                 const int* pos,
                                 int max_count,
                                 Particle* out_buf);                              


//mark particles in d_local with flags_left, flags_right, flags_keep
__global__ void mark_migration_kernel(const Particle* particles,
                                      int n_local,
                                      double x_min,
                                      double x_max,
                                      int    rank_idx,   
                                      int    rank_size, 
                                      int left_rank,
                                      int right_rank,
                                      int*   flags_left,
                                      int*   flags_right,
                                      int*   flags_keep);                    

                                      
__global__ void pack_selected_kernel(const Particle* particles,
                                     int n_local,
                                     const int* flags,
                                     const int* pos,
                                     int n_selected,
                                     Particle* out);
                      

__global__ void li_force_kernel(Particle* particles, Particle* halo_left, Particle* halo_right,
        int n_local, int n_left, int n_right,
        double Lx, double Ly,
        double sigma_AA, double sigma_BB, double sigma_AB, 
        double epsilon_AA, double epsilon_BB, double epsilon_AB,
        double cutoff,
        double mass_0, double mass_1);        

// Compute per-block partial sums of dU/dlambda on the device.
// For each interacting pair with force F = (Fx, Fy) and separation dr = (dx, dy),
// the pair contribution is -epsilon_lambda * (Fx * dx - Fy * dy), with 0.5 factor
// inside the kernel to avoid double counting.
__global__ void cal_partial_U_lambda_kernel(const Particle* __restrict__ particles,
                                            const Particle* __restrict__ halo_left,
                                            const Particle* __restrict__ halo_right,
                                            int n_local, int n_left, int n_right,
                                            double Lx, double Ly,
                                            double sigma_AA, double sigma_BB, double sigma_AB,
                                            double epsilon_AA, double epsilon_BB, double epsilon_AB,
                                            double cutoff,
                                            double epsilon_lambda,
                                            double* __restrict__ partial_sums);

// Verlocity-Verlot half kick
//from r^{n}, v^{n}, a^{n} to r^{n+1}, v^{n+1/2}, a^{n};
__global__ void step_half_vv_kernel(Particle* particles, int n_local, double dt, double Lx, double Ly);

//2nd half kick. From r^{n+1}, v^{n+1/2}, a^{n+1} to r^{n+1}, v^{n+1}, a^{n+1};
__global__ void step_2nd_half_vv_kernel(Particle* particles, int n_local, double dt);

__global__ void step_half_vv_nh_kernel(Particle* particles,
                                       int n_local,
                                       double dt,
                                       double xi,
                                       double Lx,
                                       double Ly);

__global__ void step_2nd_half_vv_nh_kernel(Particle* particles,
                                           int n_local,
                                           double dt,
                                           double xi);

__global__ void cal_local_K_kernel(const Particle* __restrict__ particles,
                                   int n_local,
                                   double mass_A,
                                   double mass_B,
                                   double* __restrict__ partial_sums);



__global__ void cal_local_U_kernel(const Particle* __restrict__ particles,
                                   const Particle* __restrict__ halo_left,
                                   const Particle* __restrict__ halo_right,
                                   int n_local, int n_left, int n_right,
                                   double Lx, double Ly,
                                   double sigma_AA, double sigma_BB, double sigma_AB,
                                   double epsilon_AA, double epsilon_BB, double epsilon_AB,
                                   double cutoff,
                                   double* __restrict__ partial_sums);


__global__ void local_density_profile_kernel(const Particle* __restrict__ particles, int n_local, int n_bins_per_rank,
                                   double xmin, double xmax,
                                   int* count_A, int* count_B);

// Local Irving–Kirkwood pressure tensor profile along y.
// Domain [0, Ly) is split into n_bins_local bins; this kernel accumulates
// unnormalized rank-local P_xx, P_yy, P_xy per bin, using:
//   kinetic: m_i v_i^α v_i^β at particle y_i
//   virial:  (1/2) r_{ij}^α F_{ij}^β at pair midpoint (minimum image).
// To get physical pressure, sum over ranks and divide by (Lx * Ly/n_bins_local).
__global__ void local_pressure_tensor_profile_kernel(
    const Particle* __restrict__ particles,
    const Particle* __restrict__ halo_left,
    const Particle* __restrict__ halo_right,
    int n_local, int n_left, int n_right,
    double Lx, double Ly,
    double mass_A, double mass_B,
    double sigma_AA, double sigma_BB, double sigma_AB,
    double epsilon_AA, double epsilon_BB, double epsilon_AB,
    double cutoff,
    int n_bins_local,
    double* __restrict__ P_xx,
    double* __restrict__ P_yy,
    double* __restrict__ P_xy);

                                
