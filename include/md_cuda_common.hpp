#pragma once
#include <cuda_runtime.h>
#include "include/md_particle.hpp"

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

