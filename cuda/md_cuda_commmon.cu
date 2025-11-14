#include "../include/md_cuda_common.hpp"

__global__ void mark_halo_kernel(const Particle* particles,
                                 int n_local,
                                 double x_min,
                                 double x_max,
                                 double Lx,
                                 double halo_width,
                                 int* flags_left,
                                 int* flags_right)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_local) {
        return;
    }

    const Particle p = particles[idx];
    double x = pbc_wrap_hd(p.pos.x, Lx);

    double dx_left  = x - x_min;
    double dx_right = x_max - x;

    int fl = 0;
    int fr = 0;

    if (dx_left <= halo_width && dx_left >= 0.0) {
        fl = 1;
    }
    if (dx_right <= halo_width && dx_right >= 0.0) {
        fr = 1;
    }

    flags_left[idx]  = fl;
    flags_right[idx] = fr;
}

//device kernel to pack selected halo particles using prefix-sum positions
__global__ void pack_halo_kernel(const Particle* particles,
                                 int n_local,
                                 const int* flags,
                                 const int* pos,
                                 int max_count,
                                 Particle* out_buf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_local) {
        return;
    }

    if (flags[idx]) {
        int out_idx = pos[idx];
        if (out_idx < max_count) {
            out_buf[out_idx] = particles[idx];
        }
    }
}