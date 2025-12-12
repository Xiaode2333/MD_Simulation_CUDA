#include "md_cuda_common.hpp"
#include "md_particle.hpp"
#include <thrust/fill.h>
#include <thrust/device_vector.h>

__global__ void middle_reflect_LG_kernel(Particle* particles,
                                         int n,
                                         double Lx,
                                         double Ly,
                                         double slab_width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    if (Lx <= 0.0 || slab_width <= 0.0) {
        // Fall back to standard wrapping if parameters are invalid.
        particles[idx].pos.x = pbc_wrap_hd(particles[idx].pos.x, Lx);
        particles[idx].pos.y = pbc_wrap_hd(particles[idx].pos.y, Ly);
        return;
    }

    // Start from current coordinates. Positions in x are already in [0, Lx)
    // from the main integrator; y is wrapped with standard PBC.
    double x = particles[idx].pos.x;
    double y = pbc_wrap_hd(particles[idx].pos.y, Ly);

    // Central slab parameters: given width, centered at Lx / 2.
    const double center     = 0.5 * Lx;
    const double half_width = 0.5 * slab_width;
    const double x_min      = center - half_width;
    const double width      = slab_width;
    const double period     = 2.0 * width;

    // Reflect x into [x_min, x_min + width] with mirrored boundaries.
    double y_rel = x - x_min;
    double y_mod = fmod(y_rel, period);
    if (y_mod < 0.0) {
        y_mod += period;
    }
    if (y_mod > width) {
        y_mod = period - y_mod;
    }
    double x_new = x_min + y_mod;

    particles[idx].pos.x = x_new;
    particles[idx].pos.y = y;
}

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
                                      int*   flags_keep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_local) {
        return;
    }
    const double x = particles[idx].pos.x;

    if (x >= x_min && x < x_max){
        flags_left[idx]  = 0;
        flags_right[idx] = 0;
        flags_keep[idx]  = 1;
    }
    else {
        flags_left[idx]  = abs(x - x_min) < abs(x - x_max);
        flags_right[idx] = !flags_left[idx];
        flags_keep[idx]  = 0;
    }
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
  

__global__ void li_force_kernel(Particle* particles,
                                Particle* halo_left,
                                Particle* halo_right,
                                int n_local, int n_left, int n_right,
                                double Lx, double Ly,
                                double sigma_AA, double sigma_BB, double sigma_AB,
                                double epsilon_AA, double epsilon_BB, double epsilon_AB,
                                double cutoff,
                                double mass_0, double mass_1)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_local) {
        return;
    }

    const int n_total       = n_local + n_left + n_right;
    const int offset_left   = n_local;
    const int offset_right  = n_local + n_left;

    const double2 ri_d = particles[idx].pos;
    const int     type_i = particles[idx].type;

    double2 fi_d;
    fi_d.x = 0.0;
    fi_d.y = 0.0;

    const float Lx_f      = static_cast<float>(Lx);
    const float Ly_f      = static_cast<float>(Ly);
    const float cutoff_f  = static_cast<float>(cutoff);

    // loop over all sources (locals + halos)
    for (int j = 0; j < n_total; ++j) {

        // Skip self for local-local interaction
        if (j == idx) {
            continue;
        }

        double2 rj_d;
        int     type_j;

        if (j < n_local) {
            // source is local particle
            rj_d   = particles[j].pos;
            type_j = particles[j].type;
        }
        else if (j < offset_right) {
            // source is left halo
            const int halo_idx = j - offset_left;  // 0 ... n_left-1
            rj_d   = halo_left[halo_idx].pos;
            type_j = halo_left[halo_idx].type;
        }
        else {
            // source is right halo
            const int halo_idx = j - offset_right; // 0 ... n_right-1
            rj_d   = halo_right[halo_idx].pos;
            type_j = halo_right[halo_idx].type;
        }

        // Mixed precision part: compute pair interaction in float
        float dx_f = static_cast<float>(ri_d.x - rj_d.x);
        float dy_f = static_cast<float>(ri_d.y - rj_d.y);

        // MIC in float
        dx_f = dx_f - Lx_f * roundf(dx_f / Lx_f);
        dy_f = dy_f - Ly_f * roundf(dy_f / Ly_f);

        // Choose LJ parameters in float
        float sigma_f   = 0.0f;
        float epsilon_f = 0.0f;
        if (type_i == 0 && type_j == 0) {
            sigma_f   = static_cast<float>(sigma_AA);
            epsilon_f = static_cast<float>(epsilon_AA);
        }
        else if (type_i == 1 && type_j == 1) {
            sigma_f   = static_cast<float>(sigma_BB);
            epsilon_f = static_cast<float>(epsilon_BB);
        }
        else {
            sigma_f   = static_cast<float>(sigma_AB);
            epsilon_f = static_cast<float>(epsilon_AB);
        }

        const float rc_f    = cutoff_f * sigma_f;
        const float rc_sq_f = rc_f * rc_f;
        const float dr_sq_f = dx_f * dx_f + dy_f * dy_f;


        // Proper logical and, pairwise in float
        if (dr_sq_f > 0.0f && dr_sq_f < rc_sq_f) {
            const float sigma_sq_f = sigma_f * sigma_f;
            const float r2_inv_f   = 1.0f / dr_sq_f;
            const float sr2_f      = sigma_sq_f * r2_inv_f;
            const float sr6_f      = sr2_f * sr2_f * sr2_f;
            const float sr12_f     = sr6_f * sr6_f;

            // if (!isfinite(sr12_f)){
            //     printf("[DEBUG] sr12_f is invalid.\n");
            // }

            const float tmp_f = 24.0f * epsilon_f * (2.0f * sr12_f - sr6_f) * r2_inv_f;
            // if (!isfinite(tmp_f)){
            //     printf("[DEBUG] tmp_f is invalid.\n");
            // }
            // Accumulate in double
            fi_d.x += static_cast<double>(tmp_f * dx_f);
            fi_d.y += static_cast<double>(tmp_f * dy_f);
            // if (!isfinite(fi_d.x) || !isfinite(fi_d.y)){
            //     printf("[DEBUG] fi_d is invalid.\n");
            // }
        }
    }

    const double mass = (type_i == 0 ? mass_0 : mass_1);
    particles[idx].acc.x = fi_d.x / mass;
    particles[idx].acc.y = fi_d.y / mass;
}

__global__ void cal_partial_U_lambda_kernel(const Particle* __restrict__ particles,
                                            const Particle* __restrict__ halo_left,
                                            const Particle* __restrict__ halo_right,
                                            int n_local, int n_left, int n_right,
                                            double Lx, double Ly,
                                            double sigma_AA, double sigma_BB, double sigma_AB,
                                            double epsilon_AA, double epsilon_BB, double epsilon_AB,
                                            double cutoff,
                                            double epsilon_lambda,
                                            double* __restrict__ partial_sums)
{
    extern __shared__ double sdata[];
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double ui = 0.0;

    if (idx < n_local) {
        const int n_total      = n_local + n_left + n_right;
        const int offset_left  = n_local;
        const int offset_right = n_local + n_left;

        const double2 ri_d = particles[idx].pos;
        const int     type_i = particles[idx].type;

        const float Lx_f     = static_cast<float>(Lx);
        const float Ly_f     = static_cast<float>(Ly);
        const float cutoff_f = static_cast<float>(cutoff);

        for (int j = 0; j < n_total; ++j) {
            if (j == idx) {
                continue;
            }

            double2 rj_d;
            int     type_j;

            if (j < n_local) {
                rj_d   = particles[j].pos;
                type_j = particles[j].type;
            }
            else if (j < offset_right) {
                const int halo_idx = j - offset_left;
                rj_d   = halo_left[halo_idx].pos;
                type_j = halo_left[halo_idx].type;
            }
            else {
                const int halo_idx = j - offset_right;
                rj_d   = halo_right[halo_idx].pos;
                type_j = halo_right[halo_idx].type;
            }

            float dx_f = static_cast<float>(ri_d.x - rj_d.x);
            float dy_f = static_cast<float>(ri_d.y - rj_d.y);

            dx_f = dx_f - Lx_f * roundf(dx_f / Lx_f);
            dy_f = dy_f - Ly_f * roundf(dy_f / Ly_f);

            float sigma_f   = 0.0f;
            float epsilon_f = 0.0f;
            if (type_i == 0 && type_j == 0) {
                sigma_f   = static_cast<float>(sigma_AA);
                epsilon_f = static_cast<float>(epsilon_AA);
            }
            else if (type_i == 1 && type_j == 1) {
                sigma_f   = static_cast<float>(sigma_BB);
                epsilon_f = static_cast<float>(epsilon_BB);
            }
            else {
                sigma_f   = static_cast<float>(sigma_AB);
                epsilon_f = static_cast<float>(epsilon_AB);
            }

            const float rc_f    = cutoff_f * sigma_f;
            const float rc_sq_f = rc_f * rc_f;
            const float dr_sq_f = dx_f * dx_f + dy_f * dy_f;

            if (dr_sq_f > 0.0f && dr_sq_f < rc_sq_f) {
                const float sigma_sq_f = sigma_f * sigma_f;
                const float r2_inv_f   = 1.0f / dr_sq_f;
                const float sr2_f      = sigma_sq_f * r2_inv_f;
                const float sr6_f      = sr2_f * sr2_f * sr2_f;
                const float sr12_f     = sr6_f * sr6_f;

                const float tmp_f = 24.0f * epsilon_f * (2.0f * sr12_f - sr6_f) * r2_inv_f;

                const float Fx_f = tmp_f * dx_f;
                const float Fy_f = tmp_f * dy_f;

                const double dx_d = static_cast<double>(dx_f);
                const double dy_d = static_cast<double>(dy_f);
                const double Fx_d = static_cast<double>(Fx_f);
                const double Fy_d = static_cast<double>(Fy_f);

                const double pair_val = -epsilon_lambda * (Fx_d * dx_d - Fy_d * dy_d);

                // 0.5 factor to avoid double counting pairs
                ui += 0.5 * pair_val;
            }
        }
    }

    sdata[tid] = ui;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Verlocity-Verlot half kick
//from r^{n}, v^{n}, a^{n} to r^{n+1}, v^{n+1/2}, a^{n};
__global__ void step_half_vv_kernel(Particle* particles, int n_local, double dt, double Lx, double Ly){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_local) {
        return;
    }
    particles[idx].vel.x += 0.5*dt*particles[idx].acc.x;
    particles[idx].vel.y += 0.5*dt*particles[idx].acc.y;
    particles[idx].pos.x += dt*particles[idx].vel.x;
    particles[idx].pos.y += dt*particles[idx].vel.y;
    particles[idx].pos.x = pbc_wrap_hd(particles[idx].pos.x, Lx);
    particles[idx].pos.y = pbc_wrap_hd(particles[idx].pos.y, Ly);
}

//2nd half kick. From r^{n+1}, v^{n+1/2}, a^{n+1} to r^{n+1}, v^{n+1}, a^{n+1};
__global__ void step_2nd_half_vv_kernel(Particle* particles, int n_local, double dt){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_local) {
        return;
    }
    particles[idx].vel.x += 0.5*dt*particles[idx].acc.x;
    particles[idx].vel.y += 0.5*dt*particles[idx].acc.y;
}

__global__ void step_half_vv_nh_kernel(Particle* particles,
                                       int n_local,
                                       double dt,
                                       double xi,
                                       double Lx,
                                       double Ly)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_local) return;

    double vx = particles[idx].vel.x;
    double vy = particles[idx].vel.y;
    const double ax = particles[idx].acc.x;  // LJ part only (F/m)
    const double ay = particles[idx].acc.y;

    // effective acceleration a = a_LJ - xi * v
    vx += 0.5 * dt * (ax - xi * vx);
    vy += 0.5 * dt * (ay - xi * vy);

    particles[idx].vel.x = vx;
    particles[idx].vel.y = vy;

    particles[idx].pos.x += dt * vx;
    particles[idx].pos.y += dt * vy;

    particles[idx].pos.x = pbc_wrap_hd(particles[idx].pos.x, Lx);
    particles[idx].pos.y = pbc_wrap_hd(particles[idx].pos.y, Ly);
}

__global__ void step_2nd_half_vv_nh_kernel(Particle* particles,
                                           int n_local,
                                           double dt,
                                           double xi)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_local) return;

    double vx = particles[idx].vel.x;
    double vy = particles[idx].vel.y;
    const double ax = particles[idx].acc.x;  // LJ part only
    const double ay = particles[idx].acc.y;

    vx += 0.5 * dt * (ax - xi * vx);
    vy += 0.5 * dt * (ay - xi * vy);

    particles[idx].vel.x = vx;
    particles[idx].vel.y = vy;
}

__global__ void cal_local_K_kernel(const Particle* __restrict__ particles,
                                   int n_local,
                                   double mass_A,
                                   double mass_B,
                                   double* __restrict__ partial_sums)
{
    extern __shared__ double sdata[];
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double k = 0.0;
    if (idx < n_local) {
        const Particle p = particles[idx];
        const double vx = p.vel.x;
        const double vy = p.vel.y;
        // if (std::isinf(vx) || std::isfinite(vy) || std::isnan(vx) || std::isnan(vy)){
        //     printf("[DEBUG] idx %d has invalid velocity. n_local = %d\n", idx, n_local);
        // }
        const double m  = (p.type == 0 ? mass_A : mass_B);
        k = 0.5 * m * (vx * vx + vy * vy);  // 2D kinetic energy
    }

    sdata[tid] = k;
    __syncthreads();

    // block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}


__global__ void cal_local_U_kernel(const Particle* __restrict__ particles,
                                   const Particle* __restrict__ halo_left,
                                   const Particle* __restrict__ halo_right,
                                   int n_local, int n_left, int n_right,
                                   double Lx, double Ly,
                                   double sigma_AA, double sigma_BB, double sigma_AB,
                                   double epsilon_AA, double epsilon_BB, double epsilon_AB,
                                   double cutoff,
                                   double* __restrict__ partial_sums)
{
    extern __shared__ double sdata[];
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double ui = 0.0;

    const double inv_c   = 1.0 / cutoff;            // cutoff is dimensionless
    const double sr2     = inv_c * inv_c;           // (1/c)^2
    const double sr6     = sr2 * sr2 * sr2;         // (1/c)^6
    const double sr12    = sr6 * sr6;               // (1/c)^12
    const double Vc_const = 4.0 * (sr12 - sr6);     // same for all ij

    const double Vc_AA = epsilon_AA * Vc_const;
    const double Vc_BB = epsilon_BB * Vc_const;
    const double Vc_AB = epsilon_AB * Vc_const;

    if (idx < n_local) {
        const int   n_total      = n_local + n_left + n_right;
        const int   offset_left  = n_local;
        const int   offset_right = n_local + n_left;

        const double2 ri_d = particles[idx].pos;
        const int     type_i = particles[idx].type;

        const float Lx_f     = static_cast<float>(Lx);
        const float Ly_f     = static_cast<float>(Ly);
        const float cutoff_f = static_cast<float>(cutoff);

        for (int j = 0; j < n_total; ++j) {

            // skip self for local-local
            if (j == idx) {
                continue;
            }

            double2 rj_d;
            int     type_j;
            double  Vc;

            if (j < n_local) {
                rj_d   = particles[j].pos;
                type_j = particles[j].type;
            }
            else if (j < offset_right) {
                const int halo_idx = j - offset_left;  // 0 .. n_left-1
                rj_d   = halo_left[halo_idx].pos;
                type_j = halo_left[halo_idx].type;
            }
            else {
                const int halo_idx = j - offset_right; // 0 .. n_right-1
                rj_d   = halo_right[halo_idx].pos;
                type_j = halo_right[halo_idx].type;
            }

            float dx_f = static_cast<float>(ri_d.x - rj_d.x);
            float dy_f = static_cast<float>(ri_d.y - rj_d.y);

            // MIC in float
            dx_f = dx_f - Lx_f * roundf(dx_f / Lx_f);
            dy_f = dy_f - Ly_f * roundf(dy_f / Ly_f);

            // LJ parameters in float
            float sigma_f   = 0.0f;
            float epsilon_f = 0.0f;
            if (type_i == 0 && type_j == 0) {
                sigma_f   = static_cast<float>(sigma_AA);
                epsilon_f = static_cast<float>(epsilon_AA);
                Vc = Vc_AA;
            }
            else if (type_i == 1 && type_j == 1) {
                sigma_f   = static_cast<float>(sigma_BB);
                epsilon_f = static_cast<float>(epsilon_BB);
                Vc = Vc_BB;
            }
            else {
                sigma_f   = static_cast<float>(sigma_AB);
                epsilon_f = static_cast<float>(epsilon_AB);
                Vc = Vc_AB;
            }

            const float rc_f    = cutoff_f * sigma_f;
            const float rc_sq_f = rc_f * rc_f;
            const float dr_sq_f = dx_f * dx_f + dy_f * dy_f;

            if (dr_sq_f > 0.0f && dr_sq_f < rc_sq_f) {
                const float sigma_sq_f = sigma_f * sigma_f;
                const float r2_inv_f   = 1.0f / dr_sq_f;
                const float sr2_f      = sigma_sq_f * r2_inv_f;
                const float sr6_f      = sr2_f * sr2_f * sr2_f;
                const float sr12_f     = sr6_f * sr6_f;

                const float u_pair_f = 4.0f * epsilon_f * (sr12_f - sr6_f);

                // 0.5 factor to compensate double counting (i–j and j–i,
                // plus symmetric halos across ranks)
                ui += 0.5 * (static_cast<double>(u_pair_f) - Vc);
            }
        }
    }

    sdata[tid] = ui;
    __syncthreads();

    // block-level reduction to partial_sums[blockIdx.x]
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void local_density_profile_kernel(const Particle* __restrict__ particles, int n_local, int n_bins_per_rank,
                                   double xmin, double xmax,
                                   int* count_A, int* count_B){
    extern __shared__ int s_int_data[];
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int* sA = s_int_data;
    int* sB = s_int_data + n_bins_per_rank;

    for (int i = tid; i < n_bins_per_rank; i += blockDim.x) {
        sA[i] = 0.0;
        sB[i] = 0.0;
    }
    __syncthreads();

    if (idx < n_local) {
        const Particle p = particles[idx];
        const double x  = p.pos.x;
        if (x >= xmin && x <= xmax){
            const double Lx_local = xmax - xmin;
            int bin = static_cast<int>((x - xmin) / Lx_local * n_bins_per_rank);

            // Guard against possible roundoff
            if (bin < 0) {
                bin = 0;
            } else if (bin >= n_bins_per_rank) {
                bin = n_bins_per_rank - 1;
            }

            if (p.type == 0) {
                atomicAdd(&sA[bin], 1);
            } else {
                atomicAdd(&sB[bin], 1);
            }
        }
    }
    __syncthreads();

    for (int i = tid; i < n_bins_per_rank; i += blockDim.x) {
        const double valA = sA[i];
        const double valB = sB[i];
        if (valA != 0) {
            atomicAdd(&count_A[i], valA);
        }
        if (valB != 0) {
            atomicAdd(&count_B[i], valB);
        }
    }
}

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
    double* __restrict__ P_xy)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_local || n_bins_local <= 0) {
        return;
    }

    const int n_total      = n_local + n_left + n_right;
    const int offset_left  = n_local;
    const int offset_right = n_local + n_left;

    const Particle pi = particles[idx];
    const double2 ri  = pi.pos;
    const double vx   = pi.vel.x;
    const double vy   = pi.vel.y;
    const int type_i  = pi.type;

    const double mass_i = (type_i == 0 ? mass_A : mass_B);

    // Kinetic contribution: m v_α v_β assigned to bin of y_i
    double y_i = pbc_wrap_hd(ri.y, Ly);
    double pos_norm = (Ly > 0.0) ? (y_i / Ly) : 0.0;
    int bin = static_cast<int>(pos_norm * static_cast<double>(n_bins_local));
    if (bin < 0) {
        bin = 0;
    } else if (bin >= n_bins_local) {
        bin = n_bins_local - 1;
    }

    const double Pxx_kin = mass_i * vx * vx;
    const double Pyy_kin = mass_i * vy * vy;
    const double Pxy_kin = mass_i * vx * vy;

    atomicAdd(&P_xx[bin], Pxx_kin);
    atomicAdd(&P_yy[bin], Pyy_kin);
    atomicAdd(&P_xy[bin], Pxy_kin);

    // Virial (pair) contribution using LJ force and midpoint IK assignment
    const float Lx_f     = static_cast<float>(Lx);
    const float Ly_f     = static_cast<float>(Ly);
    const float cutoff_f = static_cast<float>(cutoff);

    for (int j = 0; j < n_total; ++j) {
        if (j == idx) {
            continue;
        }

        double2 rj;
        int     type_j;

        if (j < n_local) {
            rj     = particles[j].pos;
            type_j = particles[j].type;
        } else if (j < offset_right) {
            const int halo_idx = j - offset_left;
            rj     = halo_left[halo_idx].pos;
            type_j = halo_left[halo_idx].type;
        } else {
            const int halo_idx = j - offset_right;
            rj     = halo_right[halo_idx].pos;
            type_j = halo_right[halo_idx].type;
        }

        float dx_f = static_cast<float>(ri.x - rj.x);
        float dy_f = static_cast<float>(ri.y - rj.y);

        dx_f = dx_f - Lx_f * roundf(dx_f / Lx_f);
        dy_f = dy_f - Ly_f * roundf(dy_f / Ly_f);

        float sigma_f   = 0.0f;
        float epsilon_f = 0.0f;
        if (type_i == 0 && type_j == 0) {
            sigma_f   = static_cast<float>(sigma_AA);
            epsilon_f = static_cast<float>(epsilon_AA);
        } else if (type_i == 1 && type_j == 1) {
            sigma_f   = static_cast<float>(sigma_BB);
            epsilon_f = static_cast<float>(epsilon_BB);
        } else {
            sigma_f   = static_cast<float>(sigma_AB);
            epsilon_f = static_cast<float>(epsilon_AB);
        }

        const float rc_f    = cutoff_f * sigma_f;
        const float rc_sq_f = rc_f * rc_f;
        const float dr_sq_f = dx_f * dx_f + dy_f * dy_f;

        if (dr_sq_f > 0.0f && dr_sq_f < rc_sq_f) {
            const float sigma_sq_f = sigma_f * sigma_f;
            const float r2_inv_f   = 1.0f / dr_sq_f;
            const float sr2_f      = sigma_sq_f * r2_inv_f;
            const float sr6_f      = sr2_f * sr2_f * sr2_f;
            const float sr12_f     = sr6_f * sr6_f;

            const float tmp_f = 24.0f * epsilon_f * (2.0f * sr12_f - sr6_f) * r2_inv_f;
            const float Fx_f  = tmp_f * dx_f;
            const float Fy_f  = tmp_f * dy_f;

            const double dx = static_cast<double>(dx_f);
            const double dy = static_cast<double>(dy_f);
            const double Fx = static_cast<double>(Fx_f);
            const double Fy = static_cast<double>(Fy_f);

            // Midpoint along minimum-image segment in y
            double y_mid = ri.y - 0.5 * dy;
            y_mid        = pbc_wrap_hd(y_mid, Ly);

            double pos_norm_mid = (Ly > 0.0) ? (y_mid / Ly) : 0.0;
            int bin_pair = static_cast<int>(pos_norm_mid * static_cast<double>(n_bins_local));
            if (bin_pair < 0) {
                bin_pair = 0;
            } else if (bin_pair >= n_bins_local) {
                bin_pair = n_bins_local - 1;
            }

            // 0.5 factor to avoid double counting i–j and j–i, consistent
            // with energy kernels; local–halo pairs are split between ranks.
            const double vir_xx = 0.5 * dx * Fx;
            const double vir_yy = 0.5 * dy * Fy;
            const double vir_xy = 0.5 * dx * Fy;

            atomicAdd(&P_xx[bin_pair], vir_xx);
            atomicAdd(&P_yy[bin_pair], vir_yy);
            atomicAdd(&P_xy[bin_pair], vir_xy);
        }
    }
}
