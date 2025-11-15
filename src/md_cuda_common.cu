#include "md_cuda_common.hpp"
#include "md_particle.hpp"

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

void launch_mark_halo_kernel(const Particle* particles,
                                 int n_local,
                                 double x_min,
                                 double x_max,
                                 double Lx,
                                 double halo_width,
                                 int* flags_left,
                                 int* flags_right,
                                 int threads_per_block){
    int blocks = (n_local + threads_per_block - 1) / threads_per_block;

    mark_halo_kernel<<<blocks, threads_per_block>>>(
            particles,
            n_local,
            x_min,
            x_max,
            Lx,
            halo_width,
            flags_left,
            flags_right
    );
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

void launch_pack_halo_kernel(const Particle* particles,
                                 int n_local,
                                 const int* flags,
                                 const int* pos,
                                 int max_count,
                                 Particle* out_buf,
                                 int threads_per_block
                                ){
    int blocks = (n_local + threads_per_block - 1) / threads_per_block;
    pack_halo_kernel<<<blocks, threads_per_block>>>(
            particles,
            n_local,
            flags,
            pos,
            max_count,
            out_buf
    );
}

__global__ void mark_migration_kernel(const Particle* particles,
                                      int n_local,
                                      double x_min,
                                      double x_max,
                                      int* flags_left,
                                      int* flags_right,
                                      int* flags_keep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_local) {
        return;
    }

    const double x = particles[idx].pos.x;

    // Particles that left to the left or right subdomain
    const int to_left  = (x <  x_min) ? 1 : 0;
    const int to_right = (x >= x_max) ? 1 : 0;
    const int keep     = (!to_left && !to_right) ? 1 : 0;

    flags_left[idx]  = to_left;
    flags_right[idx] = to_right;
    flags_keep[idx]  = keep;
}

__global__ void pack_selected_kernel(const Particle* particles,
                                     int n_local,
                                     const int* flags,
                                     const int* pos,
                                     int n_selected,
                                     Particle* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_local) {
        return;
    }

    if (flags[idx]) {
        const int dst = pos[idx];
        if (dst < n_selected) {
            out[dst] = particles[idx];
        }
    }
}

void launch_mark_migration_kernel(const Particle* d_particles,
                                                int n_local,
                                                double x_min,
                                                double x_max,
                                                int* d_flags_left,
                                                int* d_flags_right,
                                                int* d_flags_keep,
                                                int threads)
{
    const int blocks = (n_local + threads - 1) / threads;
    if (blocks > 0) {
        mark_migration_kernel<<<blocks, threads>>>(
            d_particles,
            n_local,
            x_min,
            x_max,
            d_flags_left,
            d_flags_right,
            d_flags_keep
        );
    }
}      

void launch_pack_selected_kernel(const Particle* d_particles,
                                               int n_local,
                                               const int* d_flags,
                                               const int* d_pos,
                                               int n_selected,
                                               Particle* d_out,
                                               int threads)
{
    const int blocks = (n_local + threads - 1) / threads;
    if (blocks > 0 && n_selected > 0) {
        pack_selected_kernel<<<blocks, threads>>>(
            d_particles,
            n_local,
            d_flags,
            d_pos,
            n_selected,
            d_out
        );
    }
}