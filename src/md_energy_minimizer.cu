#include "md_energy_minimizer.hpp"

#include "md_cuda_common.hpp"

#include <algorithm>
#include <cmath>
#include <fmt/core.h>
#include <stdexcept>
#include <utility>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/transform_reduce.h>

EnergyMinimizer::EnergyMinimizer(MDConfigManager config_manager, MPI_Comm comm)
        : cfg_manager(std::move(config_manager)), comm(comm) {
    initialize_comm_metadata();
    allocate_buffers();
    buffers_ready = true;
}

// --- Device kernels --------------------------------------------------------
__global__ void velocity_kick_kernel(Particle *particles, int n, double dt) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        particles[idx].vel.x += dt * particles[idx].acc.x;
        particles[idx].vel.y += dt * particles[idx].acc.y;
    }
}

__global__ void zero_velocity_kernel(Particle *particles, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        particles[idx].vel.x = 0.0;
        particles[idx].vel.y = 0.0;
    }
}

__global__ void fire_velocity_mix_kernel(Particle *particles, int n,
                                         double alpha) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Particle p = particles[idx];
        const double vx = p.vel.x;
        const double vy = p.vel.y;
        const double fx = p.acc.x;
        const double fy = p.acc.y;

        const double vmag = sqrt(vx * vx + vy * vy);
        const double fmag = sqrt(fx * fx + fy * fy);

        if (vmag > 0.0 && fmag > 0.0) {
            const double scale = (vmag / fmag);
            p.vel.x = (1.0 - alpha) * vx + alpha * scale * fx;
            p.vel.y = (1.0 - alpha) * vy + alpha * scale * fy;
            particles[idx] = p;
        }
    }
}

__global__ void drift_kernel(Particle *particles, int n, double dt, double Lx,
                             double Ly) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Particle p = particles[idx];
        p.pos.x = pbc_wrap_hd(p.pos.x + dt * p.vel.x, Lx);
        p.pos.y = pbc_wrap_hd(p.pos.y + dt * p.vel.y, Ly);
        particles[idx] = p;
    }
}

void EnergyMinimizer::initialize_comm_metadata() {
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    cfg_manager.config.rank_idx = world_rank;
    cfg_manager.config.rank_size = world_size;

    const double Lx = cfg_manager.config.box_w_global;
    const double dx = (world_size > 0) ? (Lx / static_cast<double>(world_size))
                                       : 0.0;

    cfg_manager.config.x_min = dx * static_cast<double>(world_rank);
    cfg_manager.config.x_max = dx * static_cast<double>(world_rank + 1);
    cfg_manager.config.left_rank = (world_rank + world_size - 1) % world_size;
    cfg_manager.config.right_rank = (world_rank + 1) % world_size;
}

void EnergyMinimizer::allocate_buffers() {
    const int n_cap = cfg_manager.config.n_cap;
    const int halo_left_cap = cfg_manager.config.halo_left_cap;
    const int halo_right_cap = cfg_manager.config.halo_right_cap;

    if (n_cap <= 0) {
        throw std::runtime_error("EnergyMinimizer: n_cap must be positive.");
    }

    h_particles_local.resize(n_cap);
    h_particles_halo_left.resize(halo_left_cap);
    h_particles_halo_right.resize(halo_right_cap);
    h_send_left.resize(n_cap);
    h_send_right.resize(n_cap);

    d_particles.resize(n_cap);
    d_particles_halo_left.resize(halo_left_cap);
    d_particles_halo_right.resize(halo_right_cap);
    d_send_left.resize(n_cap);
    d_send_right.resize(n_cap);
    d_keep.resize(n_cap);

    flags_left.reserve(n_cap);
    flags_right.reserve(n_cap);
    flags_keep.reserve(n_cap);
    pos_left.reserve(n_cap);
    pos_right.reserve(n_cap);
    pos_keep.reserve(n_cap);
}

MinimizeResult EnergyMinimizer::minimize_frame(const std::vector<Particle> &frame,
                                               double force_tol, int max_steps,
                                               const FireParams &params) {
    if (!buffers_ready) {
        throw std::runtime_error("EnergyMinimizer buffers not initialized.");
    }

    if (max_steps <= 0) {
        throw std::runtime_error("EnergyMinimizer: max_steps must be positive.");
    }

    initialize_from_frame(frame);

    current_dt = params.dt_init;
    current_alpha = params.alpha0;
    positive_streak = 0;

    MinimizeResult result;
    ForceStats stats{};
    bool converged = false;
    const int energy_sample_interval = 1000; // sparse energy sampling

    // Record initial potential energy before any FIRE steps.
    double last_potential = compute_potential();
    result.energy_steps.push_back(0);
    result.energy_trace.push_back(last_potential);

    for (int step = 0; step < max_steps; ++step) {
        compute_forces();

        // First kick velocities with current forces so power is evaluated on
        // updated velocities (avoids P=0 at start which freezes FIRE).
        const int n_local = cfg_manager.config.n_local;
        if (n_local > 0) {
            int threads = cfg_manager.config.THREADS_PER_BLOCK;
            if (threads <= 0) {
                threads = 256;
            }
            const int blocks = (n_local + threads - 1) / threads;
            velocity_kick_kernel<<<blocks, threads>>>(
                    thrust::raw_pointer_cast(d_particles.data()), n_local, current_dt);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Evaluate force metrics after the kick so P reflects current motion.
        stats = reduce_force_metrics();

        if (stats.max_force < force_tol) {
            converged = true;
            result.steps = step;
            last_potential = compute_potential();
            result.potential_energy = last_potential;
            result.energy_steps.push_back(step + 1);
            result.energy_trace.push_back(last_potential);
            break;
        }

        if (stats.power > 0.0) {
            positive_streak += 1;
            if (positive_streak > params.N_min) {
                current_dt = std::min(current_dt * params.f_inc, params.dt_max);
                current_alpha = std::max(0.0, current_alpha * 0.99);
            }
        } else {
            current_dt *= params.f_dec;
            positive_streak = 0;
            current_alpha = params.alpha0;
            if (cfg_manager.config.n_local > 0) {
                int threads = cfg_manager.config.THREADS_PER_BLOCK;
                if (threads <= 0) {
                    threads = 256;
                }
                const int blocks =
                        (cfg_manager.config.n_local + threads - 1) / threads;
                zero_velocity_kernel<<<blocks, threads>>>(
                        thrust::raw_pointer_cast(d_particles.data()),
                        cfg_manager.config.n_local);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }

        fire_velocity_mix(current_alpha);
        drift_positions(current_dt);
        exchange_particles();

        // Potential energy after position update and halo exchange (sparse sampling).
        const bool sample_now = ((step + 1) % energy_sample_interval == 0);
        if (sample_now) {
            last_potential = compute_potential();
            result.energy_steps.push_back(step + 1);
            result.energy_trace.push_back(last_potential);
        }
    }

    result.converged = converged;
    result.steps = converged ? result.steps : max_steps;
    result.max_force = stats.max_force;
    if (!converged) {
        // Ensure final potential energy is recorded (at max_steps or last sample).
        if (result.energy_steps.empty() || result.energy_steps.back() != max_steps) {
            last_potential = compute_potential();
            result.energy_steps.push_back(max_steps);
            result.energy_trace.push_back(last_potential);
        }
        result.potential_energy = last_potential;
    }

    // Collect final frame (local copy; rank 0 gathers full frame)
    const int n_local = cfg_manager.config.n_local;
    if (n_local > 0) {
        h_particles_local.resize(n_local);
        CUDA_CHECK(cudaMemcpy(h_particles_local.data(),
                              thrust::raw_pointer_cast(d_particles.data()),
                              static_cast<size_t>(n_local) * sizeof(Particle),
                              cudaMemcpyDeviceToHost));
    }

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    std::vector<int> counts(world_size, 0);
    MPI_Allgather(&cfg_manager.config.n_local, 1, MPI_INT, counts.data(), 1,
                  MPI_INT, comm);

    std::vector<int> counts_bytes(world_size, 0);
    std::vector<int> displs_bytes(world_size, 0);
    int total_particles = counts[0];
    counts_bytes[0] = counts[0] * static_cast<int>(sizeof(Particle));
    for (int i = 1; i < world_size; ++i) {
        total_particles += counts[i];
        counts_bytes[i] = counts[i] * static_cast<int>(sizeof(Particle));
        displs_bytes[i] = displs_bytes[i - 1] + counts_bytes[i - 1];
    }

    if (world_rank == 0) {
        result.frame.resize(total_particles);
    }

    MPI_Gatherv(h_particles_local.data(),
                cfg_manager.config.n_local * static_cast<int>(sizeof(Particle)),
                MPI_BYTE,
                world_rank == 0 ? result.frame.data() : nullptr,
                counts_bytes.data(), displs_bytes.data(), MPI_BYTE, 0, comm);

    result.potential_energy = compute_potential();
    return result;
}

void EnergyMinimizer::sample_collect(std::vector<Particle> &host_out,
                                     bool gather_root) {
    const int n_local = cfg_manager.config.n_local;
    host_out.clear();

    if (n_local > 0) {
        host_out.resize(n_local);
        CUDA_CHECK(cudaMemcpy(host_out.data(),
                              thrust::raw_pointer_cast(d_particles.data()),
                              static_cast<size_t>(n_local) * sizeof(Particle),
                              cudaMemcpyDeviceToHost));
    }

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    if (!gather_root || world_size == 1) {
        return;
    }

    std::vector<int> counts(world_size, 0);
    MPI_Allgather(&cfg_manager.config.n_local, 1, MPI_INT, counts.data(), 1,
                  MPI_INT, comm);

    std::vector<int> counts_bytes(world_size, 0);
    std::vector<int> displs_bytes(world_size, 0);
    for (int i = 0; i < world_size; ++i) {
        counts_bytes[i] = counts[i] * static_cast<int>(sizeof(Particle));
        if (i > 0) {
            displs_bytes[i] = displs_bytes[i - 1] + counts_bytes[i - 1];
        }
    }

    const int total_particles = std::accumulate(counts.begin(), counts.end(), 0);

    if (world_rank == 0) {
        host_out.resize(total_particles);
    }

    MPI_Gatherv(host_out.data(),
                cfg_manager.config.n_local * static_cast<int>(sizeof(Particle)),
                MPI_BYTE,
                world_rank == 0 ? host_out.data() : nullptr, counts_bytes.data(),
                displs_bytes.data(), MPI_BYTE, 0, comm);
}

void EnergyMinimizer::plot_particles(const MinimizeResult &result,
                                     const std::string &csv_filename,
                                     const std::string &figure_filename) {
    if (cfg_manager.config.rank_idx != 0) {
        return;
    }
    if (figure_filename.empty()) {
        throw std::invalid_argument(
                "plot_particles: figure_filename must not be empty");
    }
    if (csv_filename.empty()) {
        throw std::invalid_argument(
                "plot_particles: csv_filename must not be empty");
    }
    plot_particles_python(result.frame, figure_filename, csv_filename,
                          cfg_manager.config.box_w_global,
                          cfg_manager.config.box_h_global,
                          cfg_manager.config.SIGMA_AA,
                          cfg_manager.config.SIGMA_BB);
}

void EnergyMinimizer::initialize_from_frame(
        const std::vector<Particle> &frame) {
    const int n_cap = cfg_manager.config.n_cap;
    const double Lx = cfg_manager.config.box_w_global;
    const double Ly = cfg_manager.config.box_h_global;

    const std::size_t n_local = frame.size();
    if (static_cast<int>(n_local) > n_cap) {
        throw std::runtime_error(
                "EnergyMinimizer: input frame exceeds buffer capacity.");
    }

    h_input_frame = frame; // preserve original

    h_particles_local.resize(cfg_manager.config.n_cap);
    for (std::size_t i = 0; i < n_local; ++i) {
        Particle p = frame[i];
        p.pos.x = pbc_wrap_hd(p.pos.x, Lx);
        p.pos.y = pbc_wrap_hd(p.pos.y, Ly);
        p.vel.x = 0.0;
        p.vel.y = 0.0;
        p.acc.x = 0.0;
        p.acc.y = 0.0;
        h_particles_local[i] = p;
    }

    cfg_manager.config.n_local = static_cast<int>(n_local);
    cfg_manager.config.n_halo_left = 0;
    cfg_manager.config.n_halo_right = 0;

    if (n_local > 0) {
        CUDA_CHECK(cudaMemcpy(
                thrust::raw_pointer_cast(d_particles.data()),
                h_particles_local.data(),
                n_local * sizeof(Particle),
                cudaMemcpyHostToDevice));
    }
}

void EnergyMinimizer::compute_forces() {
    int threads = cfg_manager.config.THREADS_PER_BLOCK;
    if (threads <= 0) {
        threads = 256;
    }

    const int n_local = cfg_manager.config.n_local;
    const int n_left = cfg_manager.config.n_halo_left;
    const int n_right = cfg_manager.config.n_halo_right;

    if (n_local <= 0) {
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

    const int blocks = (n_local + threads - 1) / threads;
    li_force_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_particles.data()),
            thrust::raw_pointer_cast(d_particles_halo_left.data()),
            thrust::raw_pointer_cast(d_particles_halo_right.data()), n_local, n_left,
            n_right, cfg_manager.config.box_w_global, cfg_manager.config.box_h_global,
            cfg_manager.config.SIGMA_AA, cfg_manager.config.SIGMA_BB,
            cfg_manager.config.SIGMA_AB, cfg_manager.config.EPSILON_AA,
            cfg_manager.config.EPSILON_BB, cfg_manager.config.EPSILON_AB,
            cfg_manager.config.cutoff, cfg_manager.config.MASS_A,
            cfg_manager.config.MASS_B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

namespace {
struct ForceMag {
    __device__ double operator()(const Particle &p) const {
        const double fx = p.acc.x;
        const double fy = p.acc.y;
        return sqrt(fx * fx + fy * fy);
    }
};

struct PowerVal {
    __device__ double operator()(const Particle &p) const {
        return p.vel.x * p.acc.x + p.vel.y * p.acc.y;
    }
};

struct VelSq {
    __device__ double operator()(const Particle &p) const {
        return p.vel.x * p.vel.x + p.vel.y * p.vel.y;
    }
};

struct AccSq {
    __device__ double operator()(const Particle &p) const {
        return p.acc.x * p.acc.x + p.acc.y * p.acc.y;
    }
};
} // namespace

EnergyMinimizer::ForceStats EnergyMinimizer::reduce_force_metrics() {
    ForceStats local{};

    const int n_local = cfg_manager.config.n_local;
    if (n_local > 0) {
        auto begin = d_particles.begin();
        auto end = begin + n_local;

        local.max_force = thrust::transform_reduce(
                thrust::device, begin, end, ForceMag{}, 0.0, thrust::maximum<double>());
        local.power = thrust::transform_reduce(
                thrust::device, begin, end, PowerVal{}, 0.0, thrust::plus<double>());
        const double v_sum_sq = thrust::transform_reduce(
                thrust::device, begin, end, VelSq{}, 0.0, thrust::plus<double>());
        const double f_sum_sq = thrust::transform_reduce(
                thrust::device, begin, end, AccSq{}, 0.0, thrust::plus<double>());
        local.v_norm = v_sum_sq;
        local.f_norm = f_sum_sq;
    }

    // Global reductions
    double max_force_global = local.max_force;
    double sums[3] = {local.power, local.v_norm, local.f_norm};

    MPI_Allreduce(MPI_IN_PLACE, &max_force_global, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, sums, 3, MPI_DOUBLE, MPI_SUM, comm);

    ForceStats out{};
    out.max_force = max_force_global;
    out.power = sums[0];
    out.v_norm = std::sqrt(std::max(0.0, sums[1]));
    out.f_norm = std::sqrt(std::max(0.0, sums[2]));
    return out;
}

void EnergyMinimizer::fire_velocity_mix(double alpha) {
    const int n_local = cfg_manager.config.n_local;
    if (n_local <= 0) {
        return;
    }

    int threads = cfg_manager.config.THREADS_PER_BLOCK;
    if (threads <= 0) {
        threads = 256;
    }
    const int blocks = (n_local + threads - 1) / threads;
    fire_velocity_mix_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_particles.data()), n_local, alpha);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void EnergyMinimizer::drift_positions(double dt) {
    const int n_local = cfg_manager.config.n_local;
    if (n_local <= 0) {
        return;
    }

    int threads = cfg_manager.config.THREADS_PER_BLOCK;
    if (threads <= 0) {
        threads = 256;
    }
    const int blocks = (n_local + threads - 1) / threads;
    drift_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_particles.data()), n_local, dt,
            cfg_manager.config.box_w_global, cfg_manager.config.box_h_global);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void EnergyMinimizer::exchange_particles() {
    const int rank_idx = cfg_manager.config.rank_idx;
    const int rank_size = cfg_manager.config.rank_size;

    int world_size = 0;
    int world_rank = 0;
    MPI_Comm_size(comm, &world_size);
    MPI_Comm_rank(comm, &world_rank);

    if (world_size != rank_size) {
        fmt::print(stderr, "[exchange_particles] world_size={} != rank_size={}.\n",
                   world_size, rank_size);
        MPI_Abort(comm, 1);
    }
    if (world_rank != rank_idx) {
        fmt::print(stderr, "[exchange_particles] world_rank={} != rank_idx={}.\n",
                   world_rank, rank_idx);
        MPI_Abort(comm, 1);
    }

    int left_rank = cfg_manager.config.left_rank;
    int right_rank = cfg_manager.config.right_rank;
    if (left_rank < 0 || left_rank >= rank_size || right_rank < 0 ||
        right_rank >= rank_size) {
        left_rank = (rank_idx + rank_size - 1) % rank_size;
        right_rank = (rank_idx + 1) % rank_size;
    }

    const int n_cap = cfg_manager.config.n_cap;
    int n_local = cfg_manager.config.n_local;
    const double x_min = cfg_manager.config.x_min;
    const double x_max = cfg_manager.config.x_max;

    if (n_local > n_cap) {
        fmt::print(stderr,
                   "[exchange_particles] rank {} n_local={} exceeds n_cap={}.\n",
                   rank_idx, n_local, n_cap);
        MPI_Abort(comm, 1);
    }

    Particle *d_local = thrust::raw_pointer_cast(d_particles.data());

    int send_left_count = 0;
    int send_right_count = 0;
    int keep_count = 0;

    int threads = cfg_manager.config.THREADS_PER_BLOCK;
    if (threads <= 0) {
        threads = 256;
    }
    int blocks = (n_local + threads - 1) / threads;

    if (n_local > 0) {
        flags_left.resize(n_local);
        flags_right.resize(n_local);
        flags_keep.resize(n_local);
        pos_left.resize(n_local);
        pos_right.resize(n_local);
        pos_keep.resize(n_local);

        mark_migration_kernel<<<blocks, threads>>>(
            d_local, n_local, x_min, x_max, world_rank, world_size, left_rank,
            right_rank, thrust::raw_pointer_cast(flags_left.data()),
            thrust::raw_pointer_cast(flags_right.data()),
            thrust::raw_pointer_cast(flags_keep.data()));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        thrust::exclusive_scan(flags_left.begin(), flags_left.end(),
                               pos_left.begin());
        thrust::exclusive_scan(flags_right.begin(), flags_right.end(),
                               pos_right.begin());
        thrust::exclusive_scan(flags_keep.begin(), flags_keep.end(),
                               pos_keep.begin());

        if (n_local > 0) {
            const int last = n_local - 1;
            int last_flag_left = 0, last_pos_left = 0;
            int last_flag_right = 0, last_pos_right = 0;
            int last_flag_keep = 0, last_pos_keep = 0;

            CUDA_CHECK(cudaMemcpy(&last_flag_left,
                                  thrust::raw_pointer_cast(flags_left.data()) + last,
                                  sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_pos_left,
                                  thrust::raw_pointer_cast(pos_left.data()) + last,
                                  sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_flag_right,
                                  thrust::raw_pointer_cast(flags_right.data()) + last,
                                  sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_pos_right,
                                  thrust::raw_pointer_cast(pos_right.data()) + last,
                                  sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_flag_keep,
                                  thrust::raw_pointer_cast(flags_keep.data()) + last,
                                  sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_pos_keep,
                                  thrust::raw_pointer_cast(pos_keep.data()) + last,
                                  sizeof(int), cudaMemcpyDeviceToHost));

            send_left_count = last_pos_left + last_flag_left;
            send_right_count = last_pos_right + last_flag_right;
            keep_count = last_pos_keep + last_flag_keep;
        }

        if (keep_count + send_left_count + send_right_count != n_local) {
            fmt::print(stderr,
                       "[exchange_particles] rank {} mismatch keep={} left={} "
                       "right={} n_local={}.\n",
                       rank_idx, keep_count, send_left_count, send_right_count,
                       n_local);
            MPI_Abort(comm, 1);
        }

        if (send_left_count > 0) {
            pack_selected_kernel<<<blocks, threads>>>(
                d_local, n_local, thrust::raw_pointer_cast(flags_left.data()),
                thrust::raw_pointer_cast(pos_left.data()), send_left_count,
                thrust::raw_pointer_cast(d_send_left.data()));
            CUDA_CHECK(cudaGetLastError());
        }
        if (send_right_count > 0) {
            pack_selected_kernel<<<blocks, threads>>>(
                d_local, n_local, thrust::raw_pointer_cast(flags_right.data()),
                thrust::raw_pointer_cast(pos_right.data()), send_right_count,
                thrust::raw_pointer_cast(d_send_right.data()));
            CUDA_CHECK(cudaGetLastError());
        }
        if (keep_count > 0) {
            pack_selected_kernel<<<blocks, threads>>>(
                d_local, n_local, thrust::raw_pointer_cast(flags_keep.data()),
                thrust::raw_pointer_cast(pos_keep.data()), keep_count,
                thrust::raw_pointer_cast(d_keep.data()));
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        if (keep_count > 0) {
            thrust::copy_n(d_keep.begin(), keep_count, d_particles.begin());
        }
    } else {
        send_left_count = send_right_count = keep_count = 0;
    }

    int recv_left_count = 0;
    int recv_right_count = 0;
    MPI_Status status{};

    MPI_Sendrecv(&send_left_count, 1, MPI_INT, left_rank, 10, &recv_right_count,
                 1, MPI_INT, right_rank, 10, comm, &status);
    MPI_Sendrecv(&send_right_count, 1, MPI_INT, right_rank, 11, &recv_left_count,
                 1, MPI_INT, left_rank, 11, comm, &status);

    const int n_new_local = keep_count + recv_left_count + recv_right_count;
    if (n_new_local > n_cap) {
        fmt::print(stderr,
                   "[exchange_particles] rank {} n_new_local={} exceeds n_cap={} "
                   "(keep={} recvL={} recvR={}).\n",
                   rank_idx, n_new_local, n_cap, keep_count, recv_left_count,
                   recv_right_count);
        MPI_Abort(comm, 1);
    }

    d_local = thrust::raw_pointer_cast(d_particles.data());
    Particle *d_recv_left =
        (recv_left_count > 0) ? (d_local + keep_count) : nullptr;
    Particle *d_recv_right =
        (recv_right_count > 0) ? (d_local + keep_count + recv_left_count)
                               : nullptr;

    if (send_left_count > 0) {
        CUDA_CHECK(cudaMemcpy(
            h_send_left.data(), thrust::raw_pointer_cast(d_send_left.data()),
            static_cast<size_t>(send_left_count) * sizeof(Particle),
            cudaMemcpyDeviceToHost));
    }
    if (send_right_count > 0) {
        CUDA_CHECK(cudaMemcpy(
            h_send_right.data(), thrust::raw_pointer_cast(d_send_right.data()),
            static_cast<size_t>(send_right_count) * sizeof(Particle),
            cudaMemcpyDeviceToHost));
    }

    Particle *h_recv_left = h_particles_local.data() + keep_count;
    Particle *h_recv_right =
        h_particles_local.data() + keep_count + recv_left_count;

    Particle *h_send_left_ptr =
        (send_left_count > 0) ? h_send_left.data() : h_particles_local.data();
    Particle *h_send_right_ptr =
        (send_right_count > 0) ? h_send_right.data() : h_particles_local.data();

    MPI_Sendrecv(h_send_left_ptr,
                 send_left_count * static_cast<int>(sizeof(Particle)), MPI_BYTE,
                 left_rank, 20, h_recv_right,
                 recv_right_count * static_cast<int>(sizeof(Particle)), MPI_BYTE,
                 right_rank, 20, comm, &status);

    MPI_Sendrecv(h_send_right_ptr,
                 send_right_count * static_cast<int>(sizeof(Particle)), MPI_BYTE,
                 right_rank, 21, h_recv_left,
                 recv_left_count * static_cast<int>(sizeof(Particle)), MPI_BYTE,
                 left_rank, 21, comm, &status);

    if (recv_left_count > 0) {
        CUDA_CHECK(cudaMemcpy(d_recv_left, h_recv_left,
                              static_cast<size_t>(recv_left_count) *
                                  sizeof(Particle),
                              cudaMemcpyHostToDevice));
    }
    if (recv_right_count > 0) {
        CUDA_CHECK(cudaMemcpy(d_recv_right, h_recv_right,
                              static_cast<size_t>(recv_right_count) *
                                  sizeof(Particle),
                              cudaMemcpyHostToDevice));
    }

    cfg_manager.config.n_local = n_new_local;

    // ---- Halo rebuild ----
    const int halo_left_cap = cfg_manager.config.halo_left_cap;
    const int halo_right_cap = cfg_manager.config.halo_right_cap;
    const double Lx = cfg_manager.config.box_w_global;
    const double sigma_max =
        std::max(cfg_manager.config.SIGMA_AA,
                 std::max(cfg_manager.config.SIGMA_BB, cfg_manager.config.SIGMA_AB));
    const double halo_width = cfg_manager.config.cutoff * sigma_max * 1.2;

    n_local = cfg_manager.config.n_local;
    blocks = (n_local + threads - 1) / threads;
    send_left_count = 0;
    send_right_count = 0;

    if (n_local > 0) {
        flags_left.resize(n_local);
        flags_right.resize(n_local);
        pos_left.resize(n_local);
        pos_right.resize(n_local);

        mark_halo_kernel<<<blocks, threads>>>(
            d_local, n_local, x_min, x_max, Lx, halo_width,
            thrust::raw_pointer_cast(flags_left.data()),
            thrust::raw_pointer_cast(flags_right.data()));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        thrust::exclusive_scan(flags_left.begin(), flags_left.end(),
                               pos_left.begin());
        thrust::exclusive_scan(flags_right.begin(), flags_right.end(),
                               pos_right.begin());

        if (n_local > 0) {
            const int last = n_local - 1;
            int last_flag_left = 0, last_pos_left = 0;
            int last_flag_right = 0, last_pos_right = 0;

            CUDA_CHECK(cudaMemcpy(&last_flag_left,
                                  thrust::raw_pointer_cast(flags_left.data()) + last,
                                  sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_pos_left,
                                  thrust::raw_pointer_cast(pos_left.data()) + last,
                                  sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_flag_right,
                                  thrust::raw_pointer_cast(flags_right.data()) + last,
                                  sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_pos_right,
                                  thrust::raw_pointer_cast(pos_right.data()) + last,
                                  sizeof(int), cudaMemcpyDeviceToHost));

            send_left_count = last_pos_left + last_flag_left;
            send_right_count = last_pos_right + last_flag_right;
        }
    }

    if (send_left_count > halo_left_cap || send_right_count > halo_right_cap) {
        fmt::print(stderr,
                   "[exchange_particles] rank {} halo send exceeds cap (L {}>{} "
                   "R {}>{}).\n",
                   rank_idx, send_left_count, halo_left_cap, send_right_count,
                   halo_right_cap);
        MPI_Abort(comm, 1);
    }

    if (send_left_count > 0) {
        pack_halo_kernel<<<blocks, threads>>>(
            d_local, n_local, thrust::raw_pointer_cast(flags_left.data()),
            thrust::raw_pointer_cast(pos_left.data()), send_left_count,
            thrust::raw_pointer_cast(d_send_left.data()));
        CUDA_CHECK(cudaGetLastError());
    }
    if (send_right_count > 0) {
        pack_halo_kernel<<<blocks, threads>>>(
            d_local, n_local, thrust::raw_pointer_cast(flags_right.data()),
            thrust::raw_pointer_cast(pos_right.data()), send_right_count,
            thrust::raw_pointer_cast(d_send_right.data()));
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    h_send_left.resize(send_left_count);
    h_send_right.resize(send_right_count);
    if (send_left_count > 0) {
        CUDA_CHECK(cudaMemcpy(
            h_send_left.data(), thrust::raw_pointer_cast(d_send_left.data()),
            static_cast<size_t>(send_left_count) * sizeof(Particle),
            cudaMemcpyDeviceToHost));
    }
    if (send_right_count > 0) {
        CUDA_CHECK(cudaMemcpy(
            h_send_right.data(), thrust::raw_pointer_cast(d_send_right.data()),
            static_cast<size_t>(send_right_count) * sizeof(Particle),
            cudaMemcpyDeviceToHost));
    }

    int recv_left_halo_count = 0;
    int recv_right_halo_count = 0;

    MPI_Sendrecv(&send_left_count, 1, MPI_INT, left_rank, 0, &recv_right_halo_count, 1,
                 MPI_INT, right_rank, 0, comm, &status);
    MPI_Sendrecv(&send_right_count, 1, MPI_INT, right_rank, 1, &recv_left_halo_count, 1,
                 MPI_INT, left_rank, 1, comm, &status);

    if (recv_left_halo_count > halo_left_cap ||
        recv_right_halo_count > halo_right_cap) {
        fmt::print(stderr,
                   "[exchange_particles] rank {} halo recv exceeds cap (L {}>{} "
                   "R {}>{}).\n",
                   rank_idx, recv_left_halo_count, halo_left_cap, recv_right_halo_count,
                   halo_right_cap);
        MPI_Abort(comm, 1);
    }

    Particle *h_recv_left_halo =
        (recv_left_halo_count > 0) ? h_particles_halo_left.data() : nullptr;
    Particle *h_recv_right_halo =
        (recv_right_halo_count > 0) ? h_particles_halo_right.data() : nullptr;

    Particle *h_send_left_ptr_halo =
        (send_left_count > 0) ? h_send_left.data() : nullptr;
    Particle *h_send_right_ptr_halo =
        (send_right_count > 0) ? h_send_right.data() : nullptr;

    MPI_Sendrecv(h_send_left_ptr_halo,
                 send_left_count * static_cast<int>(sizeof(Particle)), MPI_BYTE,
                 left_rank, 2, h_recv_right_halo,
                 recv_right_halo_count * static_cast<int>(sizeof(Particle)), MPI_BYTE,
                 right_rank, 2, comm, &status);

    MPI_Sendrecv(h_send_right_ptr_halo,
                 send_right_count * static_cast<int>(sizeof(Particle)), MPI_BYTE,
                 right_rank, 3, h_recv_left_halo,
                 recv_left_halo_count * static_cast<int>(sizeof(Particle)), MPI_BYTE,
                 left_rank, 3, comm, &status);

    if (recv_left_halo_count > 0) {
        CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_particles_halo_left.data()),
                              h_recv_left_halo,
                              static_cast<size_t>(recv_left_halo_count) *
                                  sizeof(Particle),
                              cudaMemcpyHostToDevice));
    }
    if (recv_right_halo_count > 0) {
        CUDA_CHECK(cudaMemcpy(
            thrust::raw_pointer_cast(d_particles_halo_right.data()),
            h_recv_right_halo,
            static_cast<size_t>(recv_right_halo_count) * sizeof(Particle),
            cudaMemcpyHostToDevice));
    }

    cfg_manager.config.n_halo_left = recv_left_halo_count;
    cfg_manager.config.n_halo_right = recv_right_halo_count;

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

double EnergyMinimizer::compute_potential() {
    const int threads = cfg_manager.config.THREADS_PER_BLOCK;
    const int n_local = cfg_manager.config.n_local;
    const int n_left = cfg_manager.config.n_halo_left;
    const int n_right = cfg_manager.config.n_halo_right;

    if (n_local <= 0) {
        return 0.0;
    }

    const int blocks = (n_local + threads - 1) / threads;
    thrust::device_vector<double> d_partial(blocks);

    cal_local_U_kernel<<<blocks, threads,
                         threads * static_cast<int>(sizeof(double))>>>(
            thrust::raw_pointer_cast(d_particles.data()),
            thrust::raw_pointer_cast(d_particles_halo_left.data()),
            thrust::raw_pointer_cast(d_particles_halo_right.data()), n_local, n_left,
            n_right, cfg_manager.config.box_w_global, cfg_manager.config.box_h_global,
            cfg_manager.config.SIGMA_AA, cfg_manager.config.SIGMA_BB,
            cfg_manager.config.SIGMA_AB, cfg_manager.config.EPSILON_AA,
            cfg_manager.config.EPSILON_BB, cfg_manager.config.EPSILON_AB,
            cfg_manager.config.cutoff, thrust::raw_pointer_cast(d_partial.data()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_partial(blocks);
    CUDA_CHECK(cudaMemcpy(h_partial.data(),
                          thrust::raw_pointer_cast(d_partial.data()),
                          static_cast<size_t>(blocks) * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double U_local = 0.0;
    for (int i = 0; i < blocks; ++i) {
        U_local += h_partial[i];
    }

    double U_global = 0.0;
    MPI_Allreduce(&U_local, &U_global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return U_global;
}
