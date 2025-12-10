#include "md_env.hpp"

// GPU Delaunay triangulation (gDel2D)
#include "external/gDel2D-Oct2015/src/gDel2D/GpuDelaunay.h"
#include "md_particle.hpp"
#include "md_config.hpp"
#include "md_common.hpp"
#include "md_cuda_common.hpp"


#include <iostream>
#include <complex>
#include <numeric>
#include <map>
#include <set>
#include <memory>
#include <cstdlib>
#include <cmath>

MDSimulation::MDSimulation(MDConfigManager config_manager, MPI_Comm comm) {
    this->cfg_manager = config_manager;
    this->comm = comm;
    xi = 0.0;

    fmt::print("Starting broadcasting params.\n");
    std::fflush(stdout); // FORCE FLUSH
    
    broadcast_params();
    Lx0 = cfg_manager.config.box_w_global;
    Ly0 = cfg_manager.config.box_h_global;
    init_equilibrium_tracker();
    
    fmt::print("Params broadcasted.\n");
    std::fflush(stdout);

    allocate_memory();
    if (cfg_manager.config.rank_idx == 0) {
        fmt::print("[Rank] {}/{}. Memory allocated.\n", cfg_manager.config.rank_idx, cfg_manager.config.rank_size);
        std::fflush(stdout);
    }

    init_particles(); // Only updates h_particles on Rank 0

    // Distribute
    if (cfg_manager.config.rank_idx == 0) {
        fmt::print("[Rank] {}/{}. Distributing particles.\n", cfg_manager.config.rank_idx, cfg_manager.config.rank_size);
        std::fflush(stdout);
    }
    
    distribute_particles_h2d(); 
    
    if (cfg_manager.config.rank_idx == 0) {
        fmt::print("[Rank] {}/{}. Update_halo.\n", cfg_manager.config.rank_idx, cfg_manager.config.rank_size);
        std::fflush(stdout);
    }
    update_halo();

    // Forces
    if (cfg_manager.config.rank_idx == 0) {
        fmt::print("[Rank] {}/{}. Updating forces.\n", cfg_manager.config.rank_idx, cfg_manager.config.rank_size);
        std::fflush(stdout);
    }
    
    cal_forces();
    update_halo(); 

    // collect_particles_d2h();

    if (cfg_manager.config.rank_idx == 0) {
        fmt::print("[Rank] {}/{}. MD simulation env initialized.\n", cfg_manager.config.rank_idx, cfg_manager.config.rank_size);
        std::fflush(stdout);
    }
}

MDSimulation::MDSimulation(MDConfigManager config_manager, MPI_Comm comm, const std::string& filename, int step) {
    this->cfg_manager = config_manager;
    this->comm = comm;
    xi = 0.0;

    broadcast_params();
    Lx0 = cfg_manager.config.box_w_global;
    Ly0 = cfg_manager.config.box_h_global;
    init_equilibrium_tracker();
    RankZeroPrint("Params broadcasted.\n");

    allocate_memory();
    RankZeroPrint("[Rank] {}/{}. Memory allocated.\n", cfg_manager.config.rank_idx, cfg_manager.config.rank_size);


    int rank;
    MPI_Comm_rank(comm, &rank);

    std::vector<Particle> loaded_particles;
    bool success = false;

    // 1. Rank 0 reads the file
    if (rank == 0) {
        try {
            FileReader reader(filename);
            if (!reader.corrupted()) {
                std::vector<Particle> buf;
                std::uint64_t frame_index_out = 0;
                
                // Scan for the requested frame index
                while (reader.next_frame(buf, frame_index_out)) {
                    if (static_cast<int>(frame_index_out) == step) {
                        loaded_particles = std::move(buf);
                        success = true;
                        break;
                    }
                }
                if (!success) {
                    fmt::print(stderr, "Error: Frame index {} not found in {}.\n", step, filename);
                }
            } else {
                 fmt::print(stderr, "Error: File {} is corrupted.\n", filename);
            }
        } catch (const std::exception& e) {
            fmt::print(stderr, "Error opening file {}: {}\n", filename, e.what());
            success = false;
        }
    }
    MPI_Bcast(&success, 1, MPI_C_BOOL, 0, comm);
    
    if (!success) {
        MPI_Abort(comm, 1);
    }
    if (!loaded_particles.size() == h_particles.size()){
        RankZeroPrint("[Error] Loaded particle count {} doesn't match n_particles_global {}.\n", loaded_particles.size(), h_particles.size());
        MPI_Abort(comm, 1);
    }
    h_particles = loaded_particles;



    // Distribute
    RankZeroPrint("[Rank] {}/{}. Distributing particles.\n", cfg_manager.config.rank_idx, cfg_manager.config.rank_size);
    distribute_particles_h2d(); 
    
    RankZeroPrint("[Rank] {}/{}. Update_halo.\n", cfg_manager.config.rank_idx, cfg_manager.config.rank_size);
    update_halo();

    // Forces
    RankZeroPrint("[Rank] {}/{}. Updating forces.\n", cfg_manager.config.rank_idx, cfg_manager.config.rank_size);
    cal_forces();
    update_halo(); 

    if (cfg_manager.config.rank_idx == 0) {
        fmt::print("[Rank] {}/{}. MD simulation env initialized.\n", cfg_manager.config.rank_idx, cfg_manager.config.rank_size);
        std::fflush(stdout);
    }
}

MDSimulation::~MDSimulation() = default;

void MDSimulation::init_equilibrium_tracker() {
    t = 0.0;
    record_interval_dt = cfg_manager.config.save_dt_interval;
    if (record_interval_dt <= 0.0) {
        record_interval_dt = cfg_manager.config.dt;
    }
    if (record_interval_dt <= 0.0) {
        record_interval_dt = 1.0;
    }
    next_record_time = record_interval_dt;

    const double samples = kEquilibriumWindowTime / record_interval_dt;
    const int approx_samples = static_cast<int>(std::lround(samples));
    energy_window_sample_count = static_cast<std::size_t>(std::max(1, approx_samples));
    energy_history_capacity = std::max<std::size_t>(
        energy_window_sample_count * 2 * (kBaseRequiredPasses + 1),
        energy_window_sample_count * 2);

    energy_history.clear();
    equilibrium_window_streak = 0;
}

void MDSimulation::append_energy_sample(double U) {
    if (cfg_manager.config.rank_idx != 0) {
        return;
    }
    if (energy_history_capacity == 0) {
        energy_history_capacity = std::max<std::size_t>(2, energy_window_sample_count * 2);
    }
    if (energy_history.size() >= energy_history_capacity) {
        energy_history.pop_front();
    }
    energy_history.push_back(U);
}

MDSimulation::WindowStats MDSimulation::compute_window_stats(std::size_t start_idx) const {
    WindowStats stats{0.0, 0.0};
    if (energy_window_sample_count == 0 ||
        energy_history.size() < start_idx + energy_window_sample_count) {
        return stats;
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < energy_window_sample_count; ++i) {
        sum += energy_history[start_idx + i];
    }
    stats.mean = sum / static_cast<double>(energy_window_sample_count);

    if (energy_window_sample_count == 1) {
        stats.variance = 0.0;
        return stats;
    }

    double var_sum = 0.0;
    for (std::size_t i = 0; i < energy_window_sample_count; ++i) {
        double diff = energy_history[start_idx + i] - stats.mean;
        var_sum += diff * diff;
    }
    stats.variance = var_sum / static_cast<double>(energy_window_sample_count - 1);
    return stats;
}

double MDSimulation::compute_window_relative_change(std::size_t start_idx) const {
    if (energy_window_sample_count == 0 ||
        energy_history.size() < start_idx + energy_window_sample_count) {
        return 0.0;
    }

    const double U0 = energy_history[start_idx];
    const double U1 = energy_history[start_idx + energy_window_sample_count - 1];
    const double eps = 1e-12;
    double denom = U0;
    if (std::abs(denom) < eps) {
        denom = (std::abs(U1) > eps) ? U1 : (U0 >= 0.0 ? 1.0 : -1.0);
    }
    if (std::abs(denom) < eps) {
        return 0.0;
    }
    return (U1 - U0) / denom;
}

double MDSimulation::normal_tail_probability(double z) const {
    return std::erfc(std::abs(z) / std::sqrt(2.0));
}

bool MDSimulation::evaluate_equilibrium(double normalized_sensitivity) {
    if (cfg_manager.config.rank_idx != 0) {
        return false;
    }
    // Two adjacent windows (each 100 units long) must exist before we run the slope + t-test checks.
    const std::size_t needed = energy_window_sample_count * 2;
    if (energy_window_sample_count == 0 || energy_history.size() < needed) {
        return false;
    }

    const std::size_t second_start = energy_history.size() - energy_window_sample_count;
    const std::size_t first_start = second_start - energy_window_sample_count;

    const int n_global = std::max(1, cfg_manager.config.n_particles_global);
    const double base_tol = 1.0 / std::sqrt(static_cast<double>(n_global));
    const double slope_tol = base_tol * normalized_sensitivity;
    const double slope_first = std::abs(compute_window_relative_change(first_start));
    const double slope_second = std::abs(compute_window_relative_change(second_start));
    const bool slope_ok = (slope_first < slope_tol) && (slope_second < slope_tol);

    const WindowStats stats_first = compute_window_stats(first_start);
    const WindowStats stats_second = compute_window_stats(second_start);

    const double sample_count = static_cast<double>(energy_window_sample_count);
    const double se = std::sqrt(
        stats_first.variance / sample_count +
        stats_second.variance / sample_count +
        1e-18);
    double t_stat = 0.0;
    if (se > 0.0) {
        t_stat = std::abs(stats_first.mean - stats_second.mean) / se;
    }

    double alpha = kBasePValue / normalized_sensitivity;
    alpha = std::clamp(alpha, 1e-4, 0.5);
    const bool stat_ok = normal_tail_probability(t_stat) > alpha;

    if (slope_ok && stat_ok) {
        ++equilibrium_window_streak;
    } else {
        equilibrium_window_streak = 0;
    }

    const int required_passes = std::max(1, static_cast<int>(std::round(kBaseRequiredPasses / normalized_sensitivity)));
    return equilibrium_window_streak >= required_passes;
}

void MDSimulation::broadcast_params() {
    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    if (world_rank == 0) {
        if (cfg_manager.config.rank_size != 0 && cfg_manager.config.rank_size != world_size) {
            fmt::print(stderr, "[broadcast_params] Error: config.rank_size={} != world_size={}.\n",
                       cfg_manager.config.rank_size, world_size);
            MPI_Abort(comm, 1);
        }
        cfg_manager.config.rank_size = world_size;

        if (cfg_manager.config.box_w_global <= 0.0) {
            fmt::print(stderr, "[broadcast_params] Error: box_w_global <= 0.\n");
            MPI_Abort(comm, 1);
        }
        
        // Auto-calculate n_cap if missing
        if (cfg_manager.config.n_cap <= 0) {
            const double mean_per_rank = static_cast<double>(cfg_manager.config.n_particles_global) / world_size;
            cfg_manager.config.n_cap = static_cast<int>(mean_per_rank * 2) + 128;
        }

        // Auto-calculate halo cap
        const double sigma_max = std::max({cfg_manager.config.SIGMA_AA, cfg_manager.config.SIGMA_AB, cfg_manager.config.SIGMA_BB});
        int halo_cap = static_cast<int>(sigma_max * cfg_manager.config.cutoff * 5.0 / cfg_manager.config.box_w_global * cfg_manager.config.n_particles_global);
        if (halo_cap < 128) halo_cap = 128;

        if (cfg_manager.config.halo_left_cap <= 0) cfg_manager.config.halo_left_cap = halo_cap;
        if (cfg_manager.config.halo_right_cap <= 0) cfg_manager.config.halo_right_cap = halo_cap;
    }

    std::string serialized_config;
    if (world_rank == 0) {
        serialized_config = cfg_manager.serialize();
    }
    int serialized_size = static_cast<int>(serialized_config.size());
    MPI_Bcast(&serialized_size, 1, MPI_INT, 0, comm);
    if (serialized_size < 0) {
        fmt::print(stderr, "[broadcast_params] Invalid serialized size {}.\n", serialized_size);
        MPI_Abort(comm, 1);
    }
    std::vector<char> buffer(static_cast<std::size_t>(serialized_size));
    if (world_rank == 0 && serialized_size > 0) {
        std::copy(serialized_config.begin(), serialized_config.end(), buffer.begin());
    }
    char dummy = 0;
    char* data_ptr = buffer.empty() ? &dummy : buffer.data();
    MPI_Bcast(data_ptr, serialized_size, MPI_CHAR, 0, comm);

    try {
        cfg_manager.deserialize(std::string(buffer.begin(), buffer.end()));
    } catch (const std::exception& e) {
        fmt::print(stderr, "[broadcast_params] Failed to deserialize config: {}\n", e.what());
        MPI_Abort(comm, 1);
    }

    // Post-broadcast fixups
    cfg_manager.config.rank_idx = world_rank;
    cfg_manager.config.rank_size = world_size;

    const double Lx = cfg_manager.config.box_w_global;
    const double dx = Lx / static_cast<double>(world_size);

    cfg_manager.config.x_min = dx * static_cast<double>(world_rank);
    cfg_manager.config.x_max = dx * static_cast<double>(world_rank + 1);
    cfg_manager.config.left_rank = (world_rank + world_size - 1) % world_size;
    cfg_manager.config.right_rank = (world_rank + 1) % world_size;

    // Optional debug
    // fmt::print("[broadcast_params] Rank {}\n", world_rank);
    // std::fflush(stdout);
}

void MDSimulation::allocate_memory(){
    h_particles.resize(cfg_manager.config.n_particles_global);
    h_particles_local.resize(cfg_manager.config.n_cap);
    h_particles_halo_left.resize(cfg_manager.config.halo_left_cap);
    h_particles_halo_right.resize(cfg_manager.config.halo_right_cap);
    d_particles.resize(cfg_manager.config.n_cap);
    d_particles_halo_left.resize(cfg_manager.config.halo_left_cap);
    d_particles_halo_right.resize(cfg_manager.config.halo_right_cap);
    d_send_left.resize(cfg_manager.config.n_cap);
    d_send_right.resize(cfg_manager.config.n_cap);
    d_keep.resize(cfg_manager.config.n_cap);
    h_send_left.resize(cfg_manager.config.n_cap);
    h_send_right.resize(cfg_manager.config.n_cap);

    flags_left.reserve(cfg_manager.config.n_cap);
    flags_right.reserve(cfg_manager.config.n_cap);
    flags_keep.reserve(cfg_manager.config.n_cap);
    pos_left.reserve(cfg_manager.config.n_cap);
    pos_right.reserve(cfg_manager.config.n_cap);
    pos_keep.reserve(cfg_manager.config.n_cap);
    
}



// static inline double pbc_wrap(double x, double L) {
//     if (L <= 0.0) {
//         return x;
//     }
//     x = std::fmod(x, L);
//     if (x < 0.0) {
//         x += L;
//     }
//     return x;
// }

void MDSimulation::distribute_particles_h2d() {
    const int rank_idx  = cfg_manager.config.rank_idx;
    const int rank_size = cfg_manager.config.rank_size;

    int world_size = 0;
    MPI_Comm_size(comm, &world_size);
    if (world_size != rank_size) {
        fmt::print(stderr,
                   "[distribute_particles_h2d] world_size={} != rank_size={}.\n",
                   world_size, rank_size);
        MPI_Abort(comm, 1);
    }

    const double Lx      = cfg_manager.config.box_w_global;
    const int    N_global = cfg_manager.config.n_particles_global;
    const int    n_cap    = cfg_manager.config.n_cap;

    if (n_cap <= 0) {
        fmt::print(stderr,
                   "[distribute_particles_h2d] n_cap <= 0 on rank {}.\n",
                   rank_idx);
        MPI_Abort(comm, 1);
    }

    // only rank 0 distributes particles to all ranks
    if (rank_idx == 0) {
        if (static_cast<int>(h_particles.size()) < N_global) {
            fmt::print(stderr,
                       "[distribute_particles_h2d] h_particles size={} < n_particles_global={}.\n",
                       h_particles.size(), N_global);
            MPI_Abort(comm, 1);
        }

        std::vector<std::vector<Particle>> buckets(rank_size);
        const double inv_dx = static_cast<double>(rank_size) / Lx;

        for (int i = 0; i < N_global; ++i) { //This function will only be called by rank 0 at initialization, so on CPU is fine
            Particle p = h_particles[i];
            double x   = pbc_wrap_hd(p.pos.x, Lx);

            int r = static_cast<int>(x * inv_dx);
            if (r < 0) {
                r = 0;
            }
            if (r >= rank_size) {
                r = rank_size - 1;
            }
            buckets[r].push_back(p);
        }

        // First send counts
        for (int r = 0; r < rank_size; ++r) {
            const int count = static_cast<int>(buckets[r].size());
            if (count > n_cap) {
                fmt::print(stderr,
                           "[distribute_particles_h2d] bucket {} has {} particles, exceeds n_cap={}.\n",
                           r, count, n_cap);
                MPI_Abort(comm, 1);
            }

            if (r == 0) {
                cfg_manager.config.n_local = count;
            } else {
                MPI_Send(&count, 1, MPI_INT, r, 100, comm);
            }
        }

        // Then send particle data to each rank
        for (int r = 1; r < rank_size; ++r) {
            const int count = static_cast<int>(buckets[r].size());
            if (count > 0) {
                MPI_Send(buckets[r].data(),
                         count * static_cast<int>(sizeof(Particle)),
                         MPI_BYTE,
                         r,
                         101,
                         comm);
            }
        }

        // Copy local bucket to device for rank 0
        const int count0 = cfg_manager.config.n_local;
        if (count0 > 0) {
            if (static_cast<int>(d_particles.size()) < count0) {
                fmt::print(stderr,
                           "[distribute_particles_h2d] rank 0 d_particles size={} < n_local={}.\n",
                           d_particles.size(), count0);
                MPI_Abort(comm, 1);
            }
            thrust::copy_n(buckets[0].begin(), count0, d_particles.begin());
        }

    } else {
        // Other ranks receive count then particles from rank 0
        int local_count = 0;
        MPI_Recv(&local_count, 1, MPI_INT, 0, 100, comm, MPI_STATUS_IGNORE);

        if (local_count > n_cap) {
            fmt::print(stderr,
                       "[distribute_particles_h2d] rank {} received {} particles, exceeds n_cap={}.\n",
                       rank_idx, local_count, n_cap);
            MPI_Abort(comm, 1);
        }

        cfg_manager.config.n_local = local_count;

        // if (local_count > 0) { //directly receive to device memory
        //     if (static_cast<int>(d_particles.size()) < local_count) {
        //         fmt::print(stderr,
        //                    "[distribute_particles_h2d] rank {} d_particles size={} < n_local={}.\n",
        //                    rank_idx, d_particles.size(), local_count);
        //         MPI_Abort(comm, 1);
        //     }

        //     Particle* d_ptr = thrust::raw_pointer_cast(d_particles.data());

        //     // NOTE: requires CUDA-aware MPI: d_ptr is a device pointer.
        //     MPI_Recv(d_ptr,
        //              local_count * static_cast<int>(sizeof(Particle)),
        //              MPI_BYTE,
        //              0,
        //              101,
        //              comm,
        //              MPI_STATUS_IGNORE);
        // }

        // // If cann't use MPI-aware GPU, use this host buffer.
        // h_particles_local.resize(N_global);
        if (local_count > 0) {

            if (static_cast<int>(h_particles_local.size()) < local_count) {
                fmt::print(stderr,
                           "[distribute_particles_h2d] rank {} h_particles_local size={} < local_count={}.\n",
                           rank_idx, h_particles_local.size(), local_count);
                MPI_Abort(comm, 1);
            }

            MPI_Recv(h_particles_local.data(),
                     local_count * static_cast<int>(sizeof(Particle)),
                     MPI_BYTE,
                     0,
                     101,
                     comm,
                     MPI_STATUS_IGNORE);

            if (static_cast<int>(d_particles.size()) < local_count) {
                fmt::print(stderr,
                           "[distribute_particles_h2d] rank {} d_particles size={} < n_local={}.\n",
                           rank_idx, d_particles.size(), local_count);
                MPI_Abort(comm, 1);
            }
            thrust::copy_n(h_particles_local.begin(), local_count, d_particles.begin());
        }
    }

    // Build halos on device and exchange them by MPI (device to device)
    // update_halo();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// collect particles from device to host (rank 0 gathers all)
void MDSimulation::collect_particles_d2h() {
    const int rank_idx  = cfg_manager.config.rank_idx;
    const int rank_size = cfg_manager.config.rank_size;

    int world_size = 0;
    int world_rank = 0;
    MPI_Comm_size(comm, &world_size);
    MPI_Comm_rank(comm, &world_rank);

    if (world_size != rank_size) {
        fmt::print(stderr,
                   "[collect_particles_d2h] world_size={} != rank_size={}.\n",
                   world_size, rank_size);
        MPI_Abort(comm, 1);
    }
    if (world_rank != rank_idx) {
        fmt::print(stderr,
                   "[collect_particles_d2h] world_rank={} != rank_idx={}.\n",
                   world_rank, rank_idx);
        MPI_Abort(comm, 1);
    }

    const int N_global = cfg_manager.config.n_particles_global;
    const int n_local  = cfg_manager.config.n_local;

    // ensure h_particles_local has at least 1 element so .data() is always valid
    {
        const int min_host_local = (n_local > 0) ? n_local : 1;
        if (static_cast<int>(h_particles_local.size()) < min_host_local) {
            h_particles_local.resize(static_cast<std::size_t>(min_host_local));
        }
    }
    // stage local device data into host buffer h_particles_local
    if (n_local > 0) {
        Particle* d_ptr = thrust::raw_pointer_cast(d_particles.data());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            h_particles_local.data(),
            d_ptr,
            static_cast<std::size_t>(n_local) * sizeof(Particle),
            cudaMemcpyDeviceToHost
        ));
    }
    // use Allgather so every rank knows all n_local values
    std::vector<int> counts_particles(rank_size);
    int n_local_int = n_local;

    MPI_Allgather(&n_local_int,
                  1,
                  MPI_INT,
                  counts_particles.data(),
                  1,
                  MPI_INT,
                  comm);
    // build recvcounts/displs in units of bytes on all ranks
    std::vector<int> recvcounts_bytes(rank_size);
    std::vector<int> displs_bytes(rank_size);

    int offset_particles = 0;
    for (int r = 0; r < rank_size; ++r) {
        recvcounts_bytes[r] = counts_particles[r] * static_cast<int>(sizeof(Particle));
        displs_bytes[r]     = offset_particles * static_cast<int>(sizeof(Particle));
        offset_particles   += counts_particles[r];
    }
    const int total_particles = offset_particles;
    if (total_particles != N_global) {
        if (world_rank == 0) {
            fmt::print(stderr,
                       "[collect_particles_d2h] sum of n_local across ranks ({}) "
                       "!= n_particles_global={}.\n",
                       total_particles, N_global);
        }
        MPI_Abort(comm, 1);
    }

    // ensure global host buffer exists and has at least 1 element
    {
        const int min_global = (N_global > 0) ? N_global : 1;
        if (static_cast<int>(h_particles.size()) < min_global) {
            h_particles.resize(static_cast<std::size_t>(min_global));
        }
    }

    // Gather actual particle data from all ranks to rank 0 (host-only buffers)
    const int send_bytes = n_local_int * static_cast<int>(sizeof(Particle));

    // always use a valid send buffer (even if send_bytes == 0)
    void* sendbuf = static_cast<void*>(h_particles_local.data());

    void* recvbuf = nullptr;
    if (world_rank == 0) {
        recvbuf = static_cast<void*>(h_particles.data());
    } else {
        // not used on non-root, but must be non-null
        recvbuf = static_cast<void*>(h_particles_local.data());
    }
    MPI_Gatherv(sendbuf,
                send_bytes,
                MPI_BYTE,
                recvbuf,
                recvcounts_bytes.data(),
                displs_bytes.data(),
                MPI_BYTE,
                0,
                comm);        
}



// suppose d_particles is already update 
void MDSimulation::update_halo() {
    const int rank_idx  = cfg_manager.config.rank_idx;
    const int rank_size = cfg_manager.config.rank_size;

    int world_size = 0;
    int world_rank = 0;
    MPI_Comm_size(comm, &world_size);
    MPI_Comm_rank(comm, &world_rank);
    if (world_size != rank_size) {
        fmt::print(stderr,
                   "[update_halo] world_size={} != rank_size={}.\n",
                   world_size, rank_size);
        MPI_Abort(comm, 1);
    }
    if (world_rank != rank_idx) {
        fmt::print(stderr,
                   "[update_halo] world_rank={} != rank_idx={}.\n",
                   world_rank, rank_idx);
        MPI_Abort(comm, 1);
    }

    // Periodic neighbor ranks along x
    int left_rank  = cfg_manager.config.left_rank;
    int right_rank = cfg_manager.config.right_rank;

    // If not prefilled, compute from rank_idx
    if (left_rank < 0 || left_rank >= rank_size ||
        right_rank < 0 || right_rank >= rank_size) {
        left_rank  = (rank_idx + rank_size - 1) % rank_size;
        right_rank = (rank_idx + 1) % rank_size;
    }

    const int    n_local        = cfg_manager.config.n_local;
    const int    n_cap          = cfg_manager.config.n_cap;
    const int    halo_left_cap  = cfg_manager.config.halo_left_cap;
    const int    halo_right_cap = cfg_manager.config.halo_right_cap;
    const double Lx             = cfg_manager.config.box_w_global;
    const double x_min          = cfg_manager.config.x_min;
    const double x_max          = cfg_manager.config.x_max;

    if (n_local > n_cap) {
        fmt::print(stderr,
                   "[update_halo] rank {} n_local={} exceeds n_cap={}.\n",
                   rank_idx, n_local, n_cap);
        MPI_Abort(comm, 1);
    }

    const double sigma_max = std::max(
        cfg_manager.config.SIGMA_AA,
        std::max(cfg_manager.config.SIGMA_BB, cfg_manager.config.SIGMA_AB)
    );
    const double halo_width = cfg_manager.config.cutoff * sigma_max * 1.2; // 1.2 safety factor

    Particle* d_local = (n_local > 0)
                        ? thrust::raw_pointer_cast(d_particles.data())
                        : nullptr;

    int send_left_count  = 0;
    int send_right_count = 0;

    int threads = cfg_manager.config.THREADS_PER_BLOCK;
    if (threads <= 0) {
        threads = 256;
    }
    int blocks = (n_local + threads - 1) / threads;

    if (n_local > 0){
        // Resize persistent device buffers
        flags_left.resize(n_local);
        flags_right.resize(n_local);
        pos_left.resize(n_local);
        pos_right.resize(n_local);

        mark_halo_kernel<<<blocks, threads>>>(
            d_local,
            n_local,
            x_min,
            x_max,
            Lx,
            halo_width,
            thrust::raw_pointer_cast(flags_left.data()),
            thrust::raw_pointer_cast(flags_right.data())
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Prefix sums to get compact indices
        thrust::exclusive_scan(flags_left.begin(),  flags_left.end(),  pos_left.begin());
        thrust::exclusive_scan(flags_right.begin(), flags_right.end(), pos_right.begin());

        const int last = n_local - 1;

        int last_flag_left_host   = 0;
        int last_pos_left_host    = 0;
        int last_flag_right_host  = 0;
        int last_pos_right_host   = 0;

        CUDA_CHECK(cudaMemcpy(&last_flag_left_host,
                              thrust::raw_pointer_cast(flags_left.data()) + last,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last_pos_left_host,
                              thrust::raw_pointer_cast(pos_left.data()) + last,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last_flag_right_host,
                              thrust::raw_pointer_cast(flags_right.data()) + last,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last_pos_right_host,
                              thrust::raw_pointer_cast(pos_right.data()) + last,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));

        send_left_count  = last_pos_left_host  + last_flag_left_host;
        send_right_count = last_pos_right_host + last_flag_right_host;
    } else {
        send_left_count  = 0;
        send_right_count = 0;
    }

    if (send_left_count > halo_left_cap) {
        fmt::print(stderr,
                   "[update_halo] rank {} send_left_count={} exceeds halo_left_cap={}.\n",
                   rank_idx, send_left_count, halo_left_cap);
        MPI_Abort(comm, 1);
    }
    if (send_right_count > halo_right_cap) {
        fmt::print(stderr,
                   "[update_halo] rank {} send_right_count={} exceeds halo_right_cap={}.\n",
                   rank_idx, send_right_count, halo_right_cap);
        MPI_Abort(comm, 1);
    }

    // Pack halos on device into d_send_left / d_send_right (persistent device buffers)
    if (send_left_count > 0) {
        pack_halo_kernel<<<blocks, threads>>>(
            d_local,
            n_local,
            thrust::raw_pointer_cast(flags_left.data()),
            thrust::raw_pointer_cast(pos_left.data()),
            send_left_count,
            thrust::raw_pointer_cast(d_send_left.data())
        );
        CUDA_CHECK(cudaGetLastError());
    }

    if (send_right_count > 0) {
        pack_halo_kernel<<<blocks, threads>>>(
            d_local,
            n_local,
            thrust::raw_pointer_cast(flags_right.data()),
            thrust::raw_pointer_cast(pos_right.data()),
            send_right_count,
            thrust::raw_pointer_cast(d_send_right.data())
        );
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    //stage send halos to host buffers
    // std::vector<Particle> h_send_left;
    // std::vector<Particle> h_send_right;
    h_send_left.resize(send_left_count);
    h_send_right.resize(send_right_count);

    if (send_left_count > 0) {
        CUDA_CHECK(cudaMemcpy(
            h_send_left.data(),
            thrust::raw_pointer_cast(d_send_left.data()),
            static_cast<size_t>(send_left_count) * sizeof(Particle),
            cudaMemcpyDeviceToHost
        ));
    }

    if (send_right_count > 0) {
        CUDA_CHECK(cudaMemcpy(
            h_send_right.data(),
            thrust::raw_pointer_cast(d_send_right.data()),
            static_cast<size_t>(send_right_count) * sizeof(Particle),
            cudaMemcpyDeviceToHost
        ));
    }

    // h_particles_halo_left.resize(halo_left_cap);
    // h_particles_halo_right.resize(halo_right_cap);

    // Exchange counts with neighbors
    int recv_left_count  = 0;
    int recv_right_count = 0;
    MPI_Status status;

    // Counts: send left, receive from right (right neighbor's left halo)
    MPI_Sendrecv(&send_left_count,  1, MPI_INT,
                 left_rank,  0,
                 &recv_right_count, 1, MPI_INT,
                 right_rank, 0,
                 comm, &status);

    // Counts: send right, receive from left (left neighbor's right halo)
    MPI_Sendrecv(&send_right_count, 1, MPI_INT,
                 right_rank, 1,
                 &recv_left_count, 1, MPI_INT,
                 left_rank,  1,
                 comm, &status);

    if (recv_left_count > halo_left_cap) {
        fmt::print(stderr,
                   "[update_halo] rank {} recv_left_count={} exceeds halo_left_cap={}.\n",
                   rank_idx, recv_left_count, halo_left_cap);
        MPI_Abort(comm, 1);
    }
    if (recv_right_count > halo_right_cap) {
        fmt::print(stderr,
                   "[update_halo] rank {} recv_right_count={} exceeds halo_right_cap={}.\n",
                   rank_idx, recv_right_count, halo_right_cap);
        MPI_Abort(comm, 1);
    }

    // host-side MPI exchange (no device pointers in MPI)
    Particle* h_recv_left  = (recv_left_count  > 0)
                             ? h_particles_halo_left.data()
                             : nullptr;
    Particle* h_recv_right = (recv_right_count > 0)
                             ? h_particles_halo_right.data()
                             : nullptr;

    Particle* h_send_left_ptr  = (send_left_count  > 0)
                                 ? h_send_left.data()
                                 : nullptr;
    Particle* h_send_right_ptr = (send_right_count > 0)
                                 ? h_send_right.data()
                                 : nullptr;

    // Left halo (we send to left, receive from right)
    MPI_Sendrecv(
        h_send_left_ptr,
        send_left_count * static_cast<int>(sizeof(Particle)),
        MPI_BYTE,
        left_rank, 2,
        h_recv_right,
        recv_right_count * static_cast<int>(sizeof(Particle)),
        MPI_BYTE,
        right_rank, 2,
        comm, &status
    );

    // Right halo (we send to right, receive from left)
    MPI_Sendrecv(
        h_send_right_ptr,
        send_right_count * static_cast<int>(sizeof(Particle)),
        MPI_BYTE,
        right_rank, 3,
        h_recv_left,
        recv_left_count * static_cast<int>(sizeof(Particle)),
        MPI_BYTE,
        left_rank, 3,
        comm, &status
    );

    // Copy received halos back to device halo arrays
    if (recv_left_count > 0) {
        CUDA_CHECK(cudaMemcpy(
            thrust::raw_pointer_cast(d_particles_halo_left.data()),
            h_particles_halo_left.data(),
            static_cast<size_t>(recv_left_count) * sizeof(Particle),
            cudaMemcpyHostToDevice
        ));
    }
    if (recv_right_count > 0) {
        CUDA_CHECK(cudaMemcpy(
            thrust::raw_pointer_cast(d_particles_halo_right.data()),
            h_particles_halo_right.data(),
            static_cast<size_t>(recv_right_count) * sizeof(Particle),
            cudaMemcpyHostToDevice
        ));
    }

    cfg_manager.config.n_halo_left  = recv_left_count;
    cfg_manager.config.n_halo_right = recv_right_count;

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void MDSimulation::update_d_particles() { // use only d_particles to update particles through particle exchange between ranks
    const int rank_idx  = cfg_manager.config.rank_idx;
    const int rank_size = cfg_manager.config.rank_size;

    int world_size = 0;
    int world_rank = 0;
    MPI_Comm_size(comm, &world_size);
    MPI_Comm_rank(comm, &world_rank);

    if (world_size != rank_size) {
        fmt::print(stderr,
                   "[update_d_particles] world_size={} != rank_size={}.\n",
                   world_size, rank_size);
        MPI_Abort(comm, 1);
    }
    if (world_rank != rank_idx) {
        fmt::print(stderr,
                   "[update_d_particles] world_rank={} != rank_idx={}.\n",
                   world_rank, rank_idx);
        MPI_Abort(comm, 1);
    }

    // Neighbor ranks (periodic along x)
    int left_rank  = cfg_manager.config.left_rank;
    int right_rank = cfg_manager.config.right_rank;
    if (left_rank < 0 || left_rank >= rank_size ||
        right_rank < 0 || right_rank >= rank_size) {
        left_rank  = (rank_idx + rank_size - 1) % rank_size;
        right_rank = (rank_idx + 1) % rank_size;
    }

    const int    n_cap  = cfg_manager.config.n_cap;
    int          n_local = cfg_manager.config.n_local;
    // const double Lx     = cfg_manager.config.box_w_global;
    const double x_min  = cfg_manager.config.x_min;
    const double x_max  = cfg_manager.config.x_max;

    if (n_local > n_cap) {
        fmt::print(stderr,
                   "[update_d_particles] rank {} n_local={} exceeds n_cap={}.\n",
                   rank_idx, n_local, n_cap);
        MPI_Abort(comm, 1);
    }

    Particle* d_local = thrust::raw_pointer_cast(d_particles.data());

    // Flags and positions (only if we currently have particles)
    int send_left_count  = 0;
    int send_right_count = 0;
    int keep_count       = 0;

    // thrust::device_vector<int> flags_left;
    // thrust::device_vector<int> flags_right;
    // thrust::device_vector<int> flags_keep;
    // thrust::device_vector<int> pos_left;
    // thrust::device_vector<int> pos_right;
    // thrust::device_vector<int> pos_keep;

    // thrust::device_vector<Particle> d_send_left;
    // thrust::device_vector<Particle> d_send_right;
    // thrust::device_vector<Particle> d_keep;

    int threads = cfg_manager.config.THREADS_PER_BLOCK;
    if (threads <= 0) {
        threads = 256;
    }
    const int blocks = (n_local + threads - 1) / threads;

    if (n_local > 0) {
        flags_left.resize(n_local);
        flags_right.resize(n_local);
        flags_keep.resize(n_local);
        pos_left.resize(n_local);
        pos_right.resize(n_local);
        pos_keep.resize(n_local);
        
        mark_migration_kernel<<<blocks, threads>>>(
            d_local,
            n_local,
            x_min,
            x_max,
            world_rank,
            world_size,
            left_rank,
            right_rank,
            thrust::raw_pointer_cast(flags_left.data()),
            thrust::raw_pointer_cast(flags_right.data()),
            thrust::raw_pointer_cast(flags_keep.data())
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        thrust::exclusive_scan(flags_left.begin(),  flags_left.end(),  pos_left.begin());
        thrust::exclusive_scan(flags_right.begin(), flags_right.end(), pos_right.begin());
        thrust::exclusive_scan(flags_keep.begin(),  flags_keep.end(),  pos_keep.begin());

        send_left_count  = 0;
        send_right_count = 0;
        keep_count       = 0;

        // read last entries with cudaMemcpy instead of host indexing
        if (n_local > 0) {
            const int last = n_local - 1;

            int last_flag_left_host   = 0;
            int last_pos_left_host    = 0;
            int last_flag_right_host  = 0;
            int last_pos_right_host   = 0;
            int last_flag_keep_host   = 0;
            int last_pos_keep_host    = 0;

            CUDA_CHECK(cudaMemcpy(&last_flag_left_host,
                                thrust::raw_pointer_cast(flags_left.data()) + last,
                                sizeof(int),
                                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_pos_left_host,
                                thrust::raw_pointer_cast(pos_left.data()) + last,
                                sizeof(int),
                                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_flag_right_host,
                                thrust::raw_pointer_cast(flags_right.data()) + last,
                                sizeof(int),
                                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_pos_right_host,
                                thrust::raw_pointer_cast(pos_right.data()) + last,
                                sizeof(int),
                                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_flag_keep_host,
                                thrust::raw_pointer_cast(flags_keep.data()) + last,
                                sizeof(int),
                                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_pos_keep_host,
                                thrust::raw_pointer_cast(pos_keep.data()) + last,
                                sizeof(int),
                                cudaMemcpyDeviceToHost));

            send_left_count  = last_pos_left_host  + last_flag_left_host;
            send_right_count = last_pos_right_host + last_flag_right_host;
            keep_count       = last_pos_keep_host  + last_flag_keep_host;
        }

        if (keep_count + send_left_count + send_right_count != n_local) {
            fmt::print(stderr,
                       "[update_d_particles] rank {} mismatch: keep={} left={} right={} n_local={}.\n",
                       rank_idx, keep_count, send_left_count, send_right_count, n_local);
            MPI_Abort(comm, 1);
        }

        // fmt::print(stderr,
        //                "[DEBUG] rank {} keep={} left={} right={} n_local={}.\n",
        //                rank_idx, keep_count, send_left_count, send_right_count, n_local);

        // d_send_left.resize(send_left_count);
        // d_send_right.resize(send_right_count);
        // d_keep.resize(keep_count);

        if (send_left_count > 0) {
            pack_selected_kernel<<<blocks, threads>>>(
                d_local,
                n_local,
                thrust::raw_pointer_cast(flags_left.data()),
                thrust::raw_pointer_cast(pos_left.data()),
                send_left_count,
                thrust::raw_pointer_cast(d_send_left.data())
            );
            CUDA_CHECK(cudaGetLastError());
        }

        if (send_right_count > 0) {
            pack_selected_kernel<<<blocks, threads>>>(
                d_local,
                n_local,
                thrust::raw_pointer_cast(flags_right.data()),
                thrust::raw_pointer_cast(pos_right.data()),
                send_right_count,
                thrust::raw_pointer_cast(d_send_right.data())
            );
            CUDA_CHECK(cudaGetLastError());
        }

        if (keep_count > 0) {
            pack_selected_kernel<<<blocks, threads>>>(
                d_local,
                n_local,
                thrust::raw_pointer_cast(flags_keep.data()),
                thrust::raw_pointer_cast(pos_keep.data()),
                keep_count,
                thrust::raw_pointer_cast(d_keep.data())
            );
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        // Compact: overwrite d_particles[0:keep_count) with kept particles
        if (keep_count > 0) {
            thrust::copy_n(d_keep.begin(), keep_count, d_particles.begin());
        }
    } else {
        // No local particles, so nothing to mark; we may still receive from neighbors.
        send_left_count  = 0;
        send_right_count = 0;
        keep_count       = 0;
    }

    // Exchange counts with neighbors
    int recv_left_count  = 0;
    int recv_right_count = 0;
    MPI_Status status;

    // Counts: send left, receive from right (right neighbor's left migrants)
    MPI_Sendrecv(&send_left_count, 1, MPI_INT,
                 left_rank, 10,
                 &recv_right_count, 1, MPI_INT,
                 right_rank, 10,
                 comm, &status);

    // Counts: send right, receive from left (left neighbor's right migrants)
    MPI_Sendrecv(&send_right_count, 1, MPI_INT,
                 right_rank, 11,
                 &recv_left_count, 1, MPI_INT,
                 left_rank, 11,
                 comm, &status);

    const int n_new_local = keep_count + recv_left_count + recv_right_count;
    if (n_new_local > n_cap) {
        fmt::print(stderr,
                   "[update_d_particles] rank {} n_new_local={} exceeds n_cap={} (keep={} recvL={} recvR={}).\n",
                   rank_idx, n_new_local, n_cap, keep_count, recv_left_count, recv_right_count);
        MPI_Abort(comm, 1);
    }

    d_local = thrust::raw_pointer_cast(d_particles.data());

    // Device destinations for received particles
    Particle* d_recv_left  = (recv_left_count  > 0)
                             ? (d_local + keep_count)
                             : nullptr;
    Particle* d_recv_right = (recv_right_count > 0)
                             ? (d_local + keep_count + recv_left_count)
                             : nullptr;

    // Copy send buffers from device to host
    if (send_left_count > 0) {
        CUDA_CHECK(cudaMemcpy(
            h_send_left.data(),
            thrust::raw_pointer_cast(d_send_left.data()),
            static_cast<size_t>(send_left_count) * sizeof(Particle),
            cudaMemcpyDeviceToHost
        ));
    }
    if (send_right_count > 0) {
        CUDA_CHECK(cudaMemcpy(
            h_send_right.data(),
            thrust::raw_pointer_cast(d_send_right.data()),
            static_cast<size_t>(send_right_count) * sizeof(Particle),
            cudaMemcpyDeviceToHost
        ));
    }

    // Host receive locations inside h_particles_local
    Particle* h_recv_left  = h_particles_local.data() + keep_count;
    Particle* h_recv_right = h_particles_local.data() + keep_count + recv_left_count;

    // Host send pointers (any valid pointer is fine when count==0)
    Particle* h_send_left_ptr  = (send_left_count  > 0) ? h_send_left.data()
                                                        : h_particles_local.data();
    Particle* h_send_right_ptr = (send_right_count > 0) ? h_send_right.data()
                                                        : h_particles_local.data();

    // Exchange particle data on host
    MPI_Sendrecv(
        h_send_left_ptr,
        send_left_count * static_cast<int>(sizeof(Particle)),
        MPI_BYTE,
        left_rank, 20,
        h_recv_right,
        recv_right_count * static_cast<int>(sizeof(Particle)),
        MPI_BYTE,
        right_rank, 20,
        comm, &status
    );

    MPI_Sendrecv(
        h_send_right_ptr,
        send_right_count * static_cast<int>(sizeof(Particle)),
        MPI_BYTE,
        right_rank, 21,
        h_recv_left,
        recv_left_count * static_cast<int>(sizeof(Particle)),
        MPI_BYTE,
        left_rank, 21,
        comm, &status
    );

    // Copy received particles back to device
    if (recv_left_count > 0) {
        CUDA_CHECK(cudaMemcpy(
            d_recv_left,
            h_recv_left,
            static_cast<size_t>(recv_left_count) * sizeof(Particle),
            cudaMemcpyHostToDevice
        ));
    }
    if (recv_right_count > 0) {
        CUDA_CHECK(cudaMemcpy(
            d_recv_right,
            h_recv_right,
            static_cast<size_t>(recv_right_count) * sizeof(Particle),
            cudaMemcpyHostToDevice
        ));
    }

    // Update local count
    cfg_manager.config.n_local = n_new_local;
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void MDSimulation::init_particles(){
    const int rank_idx  = cfg_manager.config.rank_idx;
    const int rank_size = cfg_manager.config.rank_size;

    int world_size = 0;
    int world_rank = 0;
    MPI_Comm_size(comm, &world_size);
    MPI_Comm_rank(comm, &world_rank);

    if (world_size != rank_size) {
        fmt::print(stderr,
                   "[update_d_particles] world_size={} != rank_size={}.\n",
                   world_size, rank_size);
        MPI_Abort(comm, 1);
    }
    if (world_rank != rank_idx) {
        fmt::print(stderr,
                   "[update_d_particles] world_rank={} != rank_idx={}.\n",
                   world_rank, rank_idx);
        MPI_Abort(comm, 1);
    }

    if (rank_idx == 0) {
        double devide_p            = cfg_manager.config.devide_p;
        int    n_particles_global  = cfg_manager.config.n_particles_global;
        int    n_particles_type0   = cfg_manager.config.n_particles_type0;
        int    n_particles_type1   = n_particles_global - n_particles_type0;
        double box_w               = cfg_manager.config.box_w_global;
        double box_h               = cfg_manager.config.box_h_global;

        double unshifted_x_devide  = box_w * devide_p;
        double shift_dx            = box_w * (0.5 - devide_p / 2.0);
        double T_init              = cfg_manager.config.T_init;
        double mass_type0          = cfg_manager.config.MASS_A;
        double mass_type1          = cfg_manager.config.MASS_B;
        std::mt19937 rng(12345);
        std::uniform_real_distribution<double> pos_perturb(-1.0e-5, 1.0e-5);

        // initial spacing estimates from area / N
        double spacing_type0 = std::sqrt(unshifted_x_devide * box_h / n_particles_type0);
        double spacing_type1 = std::sqrt((box_w - unshifted_x_devide) * box_h / n_particles_type1);

        // helper to adjust spacing and grid so rows*cols >= N, without "+1" and without wrap
        auto adjust_grid = [](double width, double height,
                            int n_particles,
                            double &spacing,
                            int &n_cols, int &n_rows)
        {
            // initial estimate
            n_rows = static_cast<int>(height / spacing);
            n_cols = static_cast<int>(width  / spacing);
            if (n_rows <= 0) n_rows = 1;
            if (n_cols <= 0) n_cols = 1;

            // shrink spacing until enough sites
            while (n_rows * n_cols < n_particles) {
                spacing *= 0.99;
                n_rows = static_cast<int>(height / spacing);
                n_cols = static_cast<int>(width  / spacing);
                if (n_rows <= 0) n_rows = 1;
                if (n_cols <= 0) n_cols = 1;
            }
        };

        // compute grids for type 0 and type 1 separately
        int n_rows_type0 = 0, n_cols_type0 = 0;
        int n_rows_type1 = 0, n_cols_type1 = 0;

        adjust_grid(unshifted_x_devide,       box_h,
                    n_particles_type0,
                    spacing_type0, n_cols_type0, n_rows_type0);

        adjust_grid(box_w - unshifted_x_devide, box_h,
                    n_particles_type1,
                    spacing_type1, n_cols_type1, n_rows_type1);

        // place type-0 particles, no fmod in y, only wrap x after global shift
        for (int j = 0; j < n_rows_type0; ++j) {
            for (int i = 0; i < n_cols_type0; ++i) {
                int idx = j * n_cols_type0 + i;
                if (idx >= n_particles_type0) break;

                double x_local = (i + 0.5) * spacing_type0;
                double y       = (j + 0.5) * spacing_type0;

                if (x_local >= unshifted_x_devide) continue;
                if (y       >= box_h)              continue;

                double x = std::fmod(x_local + shift_dx + box_w, box_w);

                h_particles[idx].pos.x = x + pos_perturb(rng);
                h_particles[idx].pos.y = y + pos_perturb(rng);
                h_particles[idx].type  = 0;
            }
        }

        //place type-1 particles in [unshifted_x_devide, box_w)
        for (int j = 0; j < n_rows_type1; ++j) {
            for (int i = 0; i < n_cols_type1; ++i) {
                int idx = j * n_cols_type1 + i + n_particles_type0;
                if (idx >= n_particles_global) break;

                double x_local = (i + 0.5) * spacing_type1 + unshifted_x_devide;
                double y       = (j + 0.5) * spacing_type1;

                if (x_local >= box_w) continue;
                if (y       >= box_h) continue;

                double x = std::fmod(x_local + shift_dx + box_w, box_w);

                h_particles[idx].pos.x = x + pos_perturb(rng);
                h_particles[idx].pos.y = y + pos_perturb(rng);
                h_particles[idx].type  = 1;
            }
        }

        // Set initial velocities
        double total_mass = n_particles_type0 * mass_type0 + n_particles_type1 * mass_type1;
        double sum_px = 0.0;
        double sum_py = 0.0;
        for (int i = 0; i < n_particles_global; ++i){
            double mass = (h_particles[i].type == 0) ? mass_type0 : mass_type1;
            double stddev = std::sqrt(T_init / mass);
            std::normal_distribution<double> dist(0.0, stddev);
            h_particles[i].vel.x = dist(rng);
            h_particles[i].vel.y = dist(rng);
            sum_px += mass * h_particles[i].vel.x;
            sum_py += mass * h_particles[i].vel.y;
        }
        // Remove net momentum
        double vcm_x = sum_px / total_mass;
        double vcm_y = sum_py / total_mass;
        for (int i = 0; i < n_particles_global; ++i){
            double mass = (h_particles[i].type == 0) ? mass_type0 : mass_type1;
            h_particles[i].vel.x -= vcm_x;
            h_particles[i].vel.y -= vcm_y;  
        }

        double current_K = 0.0;
        double target_K = (2 * static_cast<double>(n_particles_global) - 2) * 0.5 * T_init;
        for (int i = 0; i < n_particles_global; ++i){
            double mass = (h_particles[i].type == 0) ? mass_type0 : mass_type1;
            current_K += 0.5 * mass * (h_particles[i].vel.x * h_particles[i].vel.x +
                 h_particles[i].vel.y * h_particles[i].vel.y);        
        }
        double rescale_factor = std::sqrt(target_K / current_K);
        for (int i = 0; i < n_particles_global; ++i){
            h_particles[i].vel.x *= rescale_factor;
            h_particles[i].vel.y *= rescale_factor;
        }
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void MDSimulation::plot_particles(const std::string& filename, const std::string& csv_path){
    if (cfg_manager.config.rank_idx != 0) return;
    plot_particles_python(h_particles,
                           filename,
                           csv_path,
                           cfg_manager.config.box_w_global,
                            cfg_manager.config.box_h_global,
                            cfg_manager.config.SIGMA_AA,
                            cfg_manager.config.SIGMA_BB);
}

// update forces and store into d_particles
void MDSimulation::cal_forces(){
    const int threads = cfg_manager.config.THREADS_PER_BLOCK;
    const int n_local = cfg_manager.config.n_local;
    const int n_left = cfg_manager.config.n_halo_left;
    const int n_right = cfg_manager.config.n_halo_right;
    int blocks = (n_local + threads - 1)/threads;
    li_force_kernel<<<blocks, threads>>>(thrust::raw_pointer_cast(d_particles.data()),
                                thrust::raw_pointer_cast(d_particles_halo_left.data()),
                                thrust::raw_pointer_cast(d_particles_halo_right.data()),
                                n_local, n_left, n_right,
                                cfg_manager.config.box_w_global, cfg_manager.config.box_h_global,
                                cfg_manager.config.SIGMA_AA, cfg_manager.config.SIGMA_BB, cfg_manager.config.SIGMA_AB,
                                cfg_manager.config.EPSILON_AA, cfg_manager.config.EPSILON_BB, cfg_manager.config.EPSILON_AB,
                                cfg_manager.config.cutoff,
                                cfg_manager.config.MASS_A, cfg_manager.config.MASS_B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


void MDSimulation::step_single_NVE() {
    const double dt = cfg_manager.config.dt;

    int threads = cfg_manager.config.THREADS_PER_BLOCK;
    if (threads <= 0) {
        threads = 256;
    }

    int n_local = cfg_manager.config.n_local;
    const double Lx = cfg_manager.config.box_w_global;
    const double Ly = cfg_manager.config.box_h_global;

    const int n_cap = cfg_manager.config.n_cap;
    if (n_local > n_cap) {
        fmt::print(stderr,
                   "[step_single_NVE] rank {} n_local={} exceeds n_cap={} before step.\n",
                   cfg_manager.config.rank_idx, n_local, n_cap);
        MPI_Abort(comm, 1);
    }

    // first half-kick only if we have local particles
    if (n_local > 0) {
        int blocks = (n_local + threads - 1) / threads;
        // fmt::print("[DEBUG] step_half_vv_kernel\n");
        step_half_vv_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_particles.data()),
            n_local,
            dt,
            Lx,
            Ly
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // exchange particles and halos between ranks
    update_d_particles();   // may change cfg_manager.config.n_local
    update_halo();          // rebuild halos after migration

    //  after exchange, n_local may have changed; recompute and check
    n_local = cfg_manager.config.n_local;
    if (n_local > n_cap) {
        fmt::print(stderr,
                   "[step_single_NVE] rank {} n_local={} exceeds n_cap={} after exchange.\n",
                   cfg_manager.config.rank_idx, n_local, n_cap);
        MPI_Abort(comm, 1);
    }

    // compute forces with updated locals + halos
    cal_forces();

    // second half-kick only if we have local particles
    if (n_local > 0) {
        int blocks = (n_local + threads - 1) / threads;
        step_2nd_half_vv_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_particles.data()),
            n_local,
            dt
        );
        // We don't care about vel and acc in halos
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    t += dt;
}


void MDSimulation::step_single_nose_hoover() {
    MPI_Barrier(comm);
    const double dt = cfg_manager.config.dt;

    int threads = cfg_manager.config.THREADS_PER_BLOCK;
    if (threads <= 0) {
        threads = 256;
    }

    const double Lx = cfg_manager.config.box_w_global;
    const double Ly = cfg_manager.config.box_h_global;

    int n_local = cfg_manager.config.n_local;
    const int n_cap = cfg_manager.config.n_cap;

    if (n_local > n_cap) {
        fmt::print(stderr,
                   "[step_single_nose_hoover] rank {} n_local={} exceeds n_cap={} before step.\n",
                   cfg_manager.config.rank_idx, n_local, n_cap);
        MPI_Abort(comm, 1);
    }

    // global kinetic energy before first half update of xi
    double K_global = cal_total_K();
    // fmt::print("[DEBUG] K_global = {:.4f}\n", K_global);

    const double g_dof = 2.0 * cfg_manager.config.n_particles_global - 2.0;
    const double T0    = cfg_manager.config.T_target;
    const double Q     = cfg_manager.config.Q;  // thermostat mass

    // first half update of xi
    xi += 0.5 * dt * (2.0 * K_global - g_dof * T0) / Q;

    //first half Nose–Hoover VV step only if we have local particles
    if (n_local > 0) {
        int blocks = (n_local + threads - 1) / threads;

        step_half_vv_nh_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_particles.data()),
            n_local,
            dt,
            xi,
            Lx,
            Ly
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // exchange particles and rebuild halos, may change n_local
    update_d_particles();
    update_halo();

    // recompute n_local after migration and check capacity
    n_local = cfg_manager.config.n_local;
    if (n_local > n_cap) {
        fmt::print(stderr,
                   "[step_single_nose_hoover] rank {} n_local={} exceeds n_cap={} after exchange.\n",
                   cfg_manager.config.rank_idx, n_local, n_cap);
        MPI_Abort(comm, 1);
    }
    // End of ADD

    // compute forces with updated locals + halos
    cal_forces();

    // global kinetic energy after forces (velocities unchanged by forces, but
    // particles may have migrated between ranks)
    K_global = cal_total_K();
    // fmt::print("[DEBUG] K_global = {:.4f}\n", K_global);

    // second half update of xi
    xi += 0.5 * dt * (2.0 * K_global - g_dof * T0) / Q;

    //second half Nose–Hoover VV step only if we have local particles
    if (n_local > 0) {
        int blocks = (n_local + threads - 1) / threads;

        step_2nd_half_vv_nh_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_particles.data()),
            n_local,
            dt,
            xi
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    K_global = cal_total_K();
    // fmt::print("[DEBUG] K_global = {:.4f} After 2nd kick\n", K_global);

    t += dt;
}


double MDSimulation::cal_total_K(){
    double K_local  = compute_kinetic_energy_local();
    double K_global = 0.0;
    MPI_Allreduce(&K_local, &K_global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return K_global;
}

double MDSimulation::compute_kinetic_energy_local(){
    const int threads = cfg_manager.config.THREADS_PER_BLOCK;
    const int n_local = cfg_manager.config.n_local;

    if (n_local <= 0) {
        return 0.0;
    }

    const int blocks = (n_local + threads - 1) / threads;

    // temporary device buffer for per-block partial sums
    thrust::device_vector<double> d_partial(blocks);

    cal_local_K_kernel<<<blocks, threads, threads * static_cast<int>(sizeof(double))>>>(
        thrust::raw_pointer_cast(d_particles.data()),
        n_local,
        cfg_manager.config.MASS_A,
        cfg_manager.config.MASS_B,
        thrust::raw_pointer_cast(d_partial.data())
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy partial sums back and accumulate on host
    std::vector<double> h_partial(blocks);
    CUDA_CHECK(cudaMemcpy(h_partial.data(),
                          thrust::raw_pointer_cast(d_partial.data()),
                          static_cast<size_t>(blocks) * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double K_local = 0.0;
    for (int i = 0; i < blocks; ++i) {
        K_local += h_partial[i];
        // if (std::isnan(h_partial[i]) || std::isinf(h_partial[i])){
        //     fmt::print("[DEBUG] Block {} has invalid h_partial.\n", i);
        // }
        
    }
    return K_local;
}

double MDSimulation::compute_U_energy_local() {
    const int threads = cfg_manager.config.THREADS_PER_BLOCK;
    const int n_local = cfg_manager.config.n_local;
    const int n_left  = cfg_manager.config.n_halo_left;
    const int n_right = cfg_manager.config.n_halo_right;

    if (n_local <= 0) {
        return 0.0;
    }

    const int blocks = (n_local + threads - 1) / threads;

    // per-block partial sums on device
    thrust::device_vector<double> d_partial(blocks);

    cal_local_U_kernel<<<blocks,
                         threads,
                         threads * static_cast<int>(sizeof(double))>>>(
        thrust::raw_pointer_cast(d_particles.data()),
        thrust::raw_pointer_cast(d_particles_halo_left.data()),
        thrust::raw_pointer_cast(d_particles_halo_right.data()),
        n_local,
        n_left,
        n_right,
        cfg_manager.config.box_w_global,
        cfg_manager.config.box_h_global,
        cfg_manager.config.SIGMA_AA,
        cfg_manager.config.SIGMA_BB,
        cfg_manager.config.SIGMA_AB,
        cfg_manager.config.EPSILON_AA,
        cfg_manager.config.EPSILON_BB,
        cfg_manager.config.EPSILON_AB,
        cfg_manager.config.cutoff,
        thrust::raw_pointer_cast(d_partial.data())
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy partial sums to host and accumulate
    std::vector<double> h_partial(blocks);
    CUDA_CHECK(cudaMemcpy(h_partial.data(),
                          thrust::raw_pointer_cast(d_partial.data()),
                          static_cast<size_t>(blocks) * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double U_local = 0.0;
    for (int i = 0; i < blocks; ++i) {
        U_local += h_partial[i];
    }

    return U_local;
}

double MDSimulation::cal_total_U() {
    const double U_local  = compute_U_energy_local();
    double       U_global = 0.0;
    MPI_Allreduce(&U_local, &U_global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return U_global;
}

double MDSimulation::cal_partial_U_lambda(double epsilon_lambda) {
    const int threads = cfg_manager.config.THREADS_PER_BLOCK;
    const int n_local = cfg_manager.config.n_local;
    const int n_left  = cfg_manager.config.n_halo_left;
    const int n_right = cfg_manager.config.n_halo_right;

    if (n_local <= 0) {
        return 0.0;
    }

    const int blocks = (n_local + threads - 1) / threads;

    thrust::device_vector<double> d_partial(blocks);

    cal_partial_U_lambda_kernel<<<blocks,
                                  threads,
                                  threads * static_cast<int>(sizeof(double))>>>(
        thrust::raw_pointer_cast(d_particles.data()),
        thrust::raw_pointer_cast(d_particles_halo_left.data()),
        thrust::raw_pointer_cast(d_particles_halo_right.data()),
        n_local,
        n_left,
        n_right,
        cfg_manager.config.box_w_global,
        cfg_manager.config.box_h_global,
        cfg_manager.config.SIGMA_AA,
        cfg_manager.config.SIGMA_BB,
        cfg_manager.config.SIGMA_AB,
        cfg_manager.config.EPSILON_AA,
        cfg_manager.config.EPSILON_BB,
        cfg_manager.config.EPSILON_AB,
        cfg_manager.config.cutoff,
        epsilon_lambda,
        thrust::raw_pointer_cast(d_partial.data())
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_partial(static_cast<size_t>(blocks));
    CUDA_CHECK(cudaMemcpy(h_partial.data(),
                          thrust::raw_pointer_cast(d_partial.data()),
                          static_cast<size_t>(blocks) * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double local_sum = 0.0;
    for (int i = 0; i < blocks; ++i) {
        local_sum += h_partial[i];
    }

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global_sum;
}

double MDSimulation::deform(double epsilon, double U_old) {
    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    if (world_size != cfg_manager.config.rank_size) {
        fmt::print(stderr,
                   "[deform] world_size={} != rank_size={}.\n",
                   world_size, cfg_manager.config.rank_size);
        MPI_Abort(comm, 1);
    }

    if (world_rank != cfg_manager.config.rank_idx) {
        fmt::print(stderr,
                   "[deform] world_rank={} != rank_idx={}.\n",
                   world_rank, cfg_manager.config.rank_idx);
        MPI_Abort(comm, 1);
    }

    const double stretch = 1.0 + epsilon;
    if (stretch <= 0.0) {
        fmt::print(stderr, "[deform] Invalid stretch factor {} (epsilon = {}).\n", stretch, epsilon);
        MPI_Abort(comm, 1);
    }

    const double Lx_old = cfg_manager.config.box_w_global;
    const double Ly_old = cfg_manager.config.box_h_global;

    // Stretch relative to current box dimensions so sequential calls accumulate deformation.
    const double Lx_new = Lx_old * stretch;
    const double Ly_new = Ly_old / stretch;

    const double dLx = Lx_new - Lx_old;
    const double dLy = Ly_new - Ly_old;

    if (cfg_manager.config.rank_idx == 0) {
        const int N_global = cfg_manager.config.n_particles_global;
        if (static_cast<int>(h_particles.size()) < N_global) {
            fmt::print(stderr,
                       "[deform] h_particles size={} < n_particles_global={}.\n",
                       h_particles.size(), N_global);
            MPI_Abort(comm, 1);
        }
        for (int i = 0; i < N_global; ++i) {
            h_particles[i].pos.x = pbc_wrap_hd(h_particles[i].pos.x + 0.5 * dLx, Lx_new);
            h_particles[i].pos.y = pbc_wrap_hd(h_particles[i].pos.y + 0.5 * dLy, Ly_new);
        }

        cfg_manager.config.box_w_global = Lx_new;
        cfg_manager.config.box_h_global = Ly_new;
    }

    // Refresh derived domain parameters on all ranks.
    broadcast_params();

    // Redistribute particles across ranks and rebuild device state for the new box.
    distribute_particles_h2d();
    update_halo();
    cal_forces();
    update_halo();
    collect_particles_d2h();

    const double U_new = cal_total_U();
    return U_new - U_old;
}

bool MDSimulation::check_eqlibrium(double sensitivity) {
    const double normalized = std::max(sensitivity, 1e-3);
    const double interval = (record_interval_dt > 0.0)
                                ? record_interval_dt
                                : (cfg_manager.config.dt > 0.0 ? cfg_manager.config.dt : 1.0);
    const double epsilon = 1e-12;

    // Collect global U on the record interval boundary so rank 0 can buffer it for the statistical test.
    while (t + epsilon >= next_record_time) {
        double U = cal_total_U();
        append_energy_sample(U);
        next_record_time += interval;
    }

    bool local_result = false;
    if (cfg_manager.config.rank_idx == 0) {
        local_result = evaluate_equilibrium(normalized);
    }
    MPI_Bcast(&local_result, 1, MPI_C_BOOL, 0, comm);
    return local_result;
}

void MDSimulation::sample_collect(){
    collect_particles_d2h();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void _build_interface_coords(
    const std::vector<Particle> &h_particles,
    double Lx,
    double Ly,
    const std::vector<double>& interfaces,
    double width_x,
    double width_y,
    std::vector<double> &coords,
    std::vector<int> &vertex_to_idx
) {
    const int N = static_cast<int>(h_particles.size());
    coords.clear();
    vertex_to_idx.clear();

    if (N <= 0 || interfaces.empty()) {
        return;
    }

    coords.reserve(static_cast<size_t>(N * 4)); 
    vertex_to_idx.reserve(static_cast<size_t>(N * 2));

    const double half_wx = width_x * 0.5;

    // Define the padded box limits
    const double x_min_limit = -width_x;
    const double x_max_limit = Lx + width_x;
    const double y_min_limit = -width_y;
    const double y_max_limit = Ly + width_y;

    // Precompute shifts
    const double shift_x[3] = { -Lx, 0.0, Lx };
    const double shift_y[3] = { -Ly, 0.0, Ly };

    for (int i = 0; i < N; ++i) {
        const double x0 = h_particles[i].pos.x;
        const double y0 = h_particles[i].pos.y;

        // 1. Filtering: Is this particle relevant (close to an interface)?
        bool keep = false;
        for (double int_x : interfaces) {
            double dx = x0 - int_x;
            // Minimum image convention for distance check
            while (dx > Lx * 0.5)  dx -= Lx;
            while (dx < -Lx * 0.5) dx += Lx;

            // Use half_wx for filtering "core" particles
            if (std::abs(dx) <= half_wx) {
                keep = true;
                break; 
            }
        }

        if (!keep) continue;

        // 2. Generation: Add all periodic images that fall within the padded box
        for (int ix = 0; ix < 3; ++ix) {
            double nx = x0 + shift_x[ix];
            
            // Check X bounds
            if (nx < x_min_limit || nx > x_max_limit) continue;

            for (int iy = 0; iy < 3; ++iy) {
                double ny = y0 + shift_y[iy];

                // Check Y bounds
                if (ny >= y_min_limit && ny <= y_max_limit) {
                    coords.push_back(nx);
                    coords.push_back(ny);
                    vertex_to_idx.push_back(i);
                }
            }
        }
    }
}

std::optional<TriangulationResult> MDSimulation::triangulation_plot(
    bool is_plot,
    const std::string& filename,
    const std::string& csv_path)
{
    if (cfg_manager.config.rank_idx != 0) {
        return std::nullopt;
    }

    const auto& cfg = cfg_manager.config;
    const double Lx = cfg.box_w_global;
    const double Ly = cfg.box_h_global;

    double sigma_max = std::max({cfg.SIGMA_AA, cfg.SIGMA_BB, cfg.SIGMA_AB});
    const double w   = 3.0 * cfg.cutoff * sigma_max;

    // Build a PBC-expanded point set in a band of width w around the box.
    coords.clear();
    vertex_to_idx.clear();

    const int n_particles = static_cast<int>(h_particles.size());

    // Fast path: fill base positions in a single pass without repeated push_back.
    coords.resize(static_cast<std::size_t>(2 * n_particles));
    vertex_to_idx.resize(static_cast<std::size_t>(n_particles));

    for (int i = 0; i < n_particles; ++i) {
        coords[2 * i]     = h_particles[i].pos.x;
        coords[2 * i + 1] = h_particles[i].pos.y;
        vertex_to_idx[i]  = i;
    }

    auto add_vertex = [&](int particle_idx, double x, double y) {
        coords.push_back(x);
        coords.push_back(y);
        vertex_to_idx.push_back(particle_idx);
    };

    // Add only PBC images near the boundaries; base points are already populated.
    for (int i = 0; i < n_particles; ++i) {
        const double x0 = h_particles[i].pos.x;
        const double y0 = h_particles[i].pos.y;

        // Periodic images near boundaries in X
        if (x0 < w)         add_vertex(i, x0 + Lx, y0);
        if (x0 > Lx - w)    add_vertex(i, x0 - Lx, y0);

        // Periodic images near boundaries in Y
        if (y0 < w)         add_vertex(i, x0, y0 + Ly);
        if (y0 > Ly - w)    add_vertex(i, x0, y0 - Ly);

        // Corner combinations
        if (x0 < w && y0 < w)                 add_vertex(i, x0 + Lx, y0 + Ly);
        if (x0 < w && y0 > Ly - w)            add_vertex(i, x0 + Lx, y0 - Ly);
        if (x0 > Lx - w && y0 < w)            add_vertex(i, x0 - Lx, y0 + Ly);
        if (x0 > Lx - w && y0 > Ly - w)       add_vertex(i, x0 - Lx, y0 - Ly);
    }

    if (coords.size() < 6) {
        if (is_plot) {
            RankZeroPrint("[Warning] Not enough points for triangulation ({} points). Skipping.\n", coords.size()/2);
        }
        return std::nullopt;
    }

    try {
        // Use GPU-based Delaunay triangulation (gDel2D).
        GDel2DInput  input;
        GDel2DOutput output;

        const std::size_t n_points = coords.size() / 2;
        input.pointVec.clear();
        input.pointVec.reserve(n_points);
        input.constraintVec.clear();

        for (std::size_t i = 0; i < n_points; ++i) {
            Point2 p;
            p._p[0] = static_cast<RealType>(coords[2 * i]);
            p._p[1] = static_cast<RealType>(coords[2 * i + 1]);
            input.pointVec.push_back(p);
        }

        // Keep original point order; profiling off.
        input.insAll    = false;
        input.noSort    = true;
        input.noReorder = true;
        input.profLevel = ProfNone;

        GpuDel gpuDel;
        gpuDel.compute(input, &output);

        const int infIdx = static_cast<int>(input.pointVec.size());
        std::vector<std::array<double, 6>> triangles_xy;
        triangles_xy.reserve(output.triVec.size());

        std::vector<std::array<int, 3>> triangles_idx;
        triangles_idx.reserve(output.triVec.size());

        for (const Tri& t : output.triVec) {
            if (t._v[0] == infIdx || t._v[1] == infIdx || t._v[2] == infIdx) {
                continue; // skip triangles incident to infinity point
            }

            const Point2& p0 = input.pointVec[t._v[0]];
            const Point2& p1 = input.pointVec[t._v[1]];
            const Point2& p2 = input.pointVec[t._v[2]];
            const double x0 = static_cast<double>(p0._p[0]);
            const double y0 = static_cast<double>(p0._p[1]);
            const double x1 = static_cast<double>(p1._p[0]);
            const double y1 = static_cast<double>(p1._p[1]);
            const double x2 = static_cast<double>(p2._p[0]);
            const double y2 = static_cast<double>(p2._p[1]);

            const auto in_base_box = [&](double x, double y) {
                return (x >= 0.0 && x < Lx && y >= 0.0 && y < Ly);
            };

            // Drop triangles whose all three vertices lie outside the base box.
            if (!(in_base_box(x0, y0) || in_base_box(x1, y1) || in_base_box(x2, y2))) {
                continue;
            }

            triangles_xy.push_back({x0, y0, x1, y1, x2, y2});
            triangles_idx.push_back({t._v[0], t._v[1], t._v[2]});
        }

        if (is_plot) {
            plot_triangulation_python_from_triangles(
                h_particles, triangles_xy, filename, csv_path,
                Lx, Ly, cfg.SIGMA_AA, cfg.SIGMA_BB
            );
        }

        TriangulationResult result;
        result.coords         = coords;
        result.vertex_to_idx  = vertex_to_idx;
        result.triangles      = std::move(triangles_idx);

        return result;
    } catch (const std::exception& e) {
        RankZeroPrint("[Warning] GPU triangulation failed. Skipping plot. Reason: {}\n", e.what());
        return std::nullopt;
    }
}

ABPairNetworks MDSimulation::get_AB_pair_network(const TriangulationResult& tri) const
{
    ABPairNetworks result;

    if (cfg_manager.config.rank_idx != 0) {
        return result;
    }

    const std::size_t vert_count = tri.vertex_to_idx.size();
    if (vert_count == 0 || tri.triangles.empty()) {
        return result;
    }

    // Helper to map midpoint coordinates to a unique node ID (with quantization).
    const double scale = 1e8;
    std::map<std::pair<long long, long long>, int> node_map;

    std::vector<ABPairNetworks::Node> nodes;
    std::vector<ABPairNetworks::Edge> edges;

    auto get_node_id = [&](double x, double y) {
        long long ix = static_cast<long long>(std::llround(x * scale));
        long long iy = static_cast<long long>(std::llround(y * scale));
        auto key = std::make_pair(ix, iy);
        auto it = node_map.find(key);
        if (it != node_map.end()) {
            return it->second;
        }
        int id = static_cast<int>(nodes.size());
        nodes.push_back(ABPairNetworks::Node{x, y});
        node_map.emplace(key, id);
        return id;
    };

    auto is_A = [](int t) { return t == 0; };
    auto is_B = [](int t) { return t == 1; };

    // Build segments from mixed A/B triangles.
    for (const auto& tri_idx : tri.triangles) {
        const int v0 = tri_idx[0];
        const int v1 = tri_idx[1];
        const int v2 = tri_idx[2];

        if (v0 < 0 || v1 < 0 || v2 < 0 ||
            v0 >= static_cast<int>(vert_count) ||
            v1 >= static_cast<int>(vert_count) ||
            v2 >= static_cast<int>(vert_count)) {
            continue;
        }

        const int p0 = tri.vertex_to_idx[static_cast<std::size_t>(v0)];
        const int p1 = tri.vertex_to_idx[static_cast<std::size_t>(v1)];
        const int p2 = tri.vertex_to_idx[static_cast<std::size_t>(v2)];

        if (p0 < 0 || p1 < 0 || p2 < 0 ||
            p0 >= static_cast<int>(h_particles.size()) ||
            p1 >= static_cast<int>(h_particles.size()) ||
            p2 >= static_cast<int>(h_particles.size())) {
            continue;
        }

        const int t0 = h_particles[p0].type;
        const int t1 = h_particles[p1].type;
        const int t2 = h_particles[p2].type;

        const bool has_A = is_A(t0) || is_A(t1) || is_A(t2);
        const bool has_B = is_B(t0) || is_B(t1) || is_B(t2);

        // Skip pure-A or pure-B triangles.
        if (!(has_A && has_B)) {
            continue;
        }

        std::vector<int> mid_nodes;
        mid_nodes.reserve(2);

        auto process_edge = [&](int va, int vb, int ta, int tb) {
            const bool a_is_A = is_A(ta);
            const bool b_is_B = is_B(tb);
            const bool a_is_B = is_B(ta);
            const bool b_is_A = is_A(tb);

            if (!((a_is_A && b_is_B) || (a_is_B && b_is_A))) {
                return;
            }

            const double x_a = tri.coords[2 * static_cast<std::size_t>(va)];
            const double y_a = tri.coords[2 * static_cast<std::size_t>(va) + 1];
            const double x_b = tri.coords[2 * static_cast<std::size_t>(vb)];
            const double y_b = tri.coords[2 * static_cast<std::size_t>(vb) + 1];

            const double x_mid = 0.5 * (x_a + x_b);
            const double y_mid = 0.5 * (y_a + y_b);

            mid_nodes.push_back(get_node_id(x_mid, y_mid));
        };

        process_edge(v0, v1, t0, t1);
        process_edge(v1, v2, t1, t2);
        process_edge(v2, v0, t2, t0);

        if (mid_nodes.size() == 2) {
            edges.push_back(ABPairNetworks::Edge{mid_nodes[0], mid_nodes[1]});
        }
    }

    if (nodes.empty() || edges.empty()) {
        return result;
    }

    // Build adjacency for connected components.
    std::vector<std::vector<int>> adj(nodes.size());
    adj.assign(nodes.size(), {});
    for (const auto& e : edges) {
        if (e.node0 < 0 || e.node1 < 0 ||
            e.node0 >= static_cast<int>(nodes.size()) ||
            e.node1 >= static_cast<int>(nodes.size())) {
            continue;
        }
        adj[e.node0].push_back(e.node1);
        adj[e.node1].push_back(e.node0);
    }

    std::vector<int> comp_id(nodes.size(), -1);
    int comp_count = 0;

    for (int start = 0; start < static_cast<int>(nodes.size()); ++start) {
        if (comp_id[start] != -1) {
            continue;
        }

        // DFS stack
        std::vector<int> stack;
        stack.push_back(start);
        comp_id[start] = comp_count;

        while (!stack.empty()) {
            int u = stack.back();
            stack.pop_back();

            for (int v : adj[u]) {
                if (comp_id[v] == -1) {
                    comp_id[v] = comp_count;
                    stack.push_back(v);
                }
            }
        }

        ++comp_count;
    }

    if (comp_count == 0) {
        return result;
    }

    result.networks_nodes.resize(comp_count);
    result.networks_edges.resize(comp_count);

    // Map from global node index to local index within its component.
    std::vector<int> local_index(nodes.size(), -1);

    for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
        int cid = comp_id[i];
        if (cid < 0) continue;
        int local = static_cast<int>(result.networks_nodes[cid].size());
        result.networks_nodes[cid].push_back(nodes[i]);
        local_index[i] = local;
    }

    for (const auto& e : edges) {
        int cid = comp_id[e.node0];
        if (cid < 0 || cid != comp_id[e.node1]) {
            continue;
        }
        int u = local_index[e.node0];
        int v = local_index[e.node1];
        if (u < 0 || v < 0) continue;
        result.networks_edges[cid].push_back(ABPairNetworks::Edge{u, v});
    }

    return result;
}

void MDSimulation::plot_interfaces(const std::string& filename, const std::string& csv_path, const std::vector<double>& rho, bool is_LG)
{
    if (cfg_manager.config.rank_idx != 0) return;

    RankZeroPrint("[DEBUG] Generating smooth interface (Grid Method)...\n");

    // Resolution: Use ~1.0 sigma resolution for fine enough detail to see waves,
    // but coarse enough that smoothing works.
    int n_grid_y = static_cast<int>(cfg_manager.config.box_h_global / 2.0); 
    if (n_grid_y < 10) n_grid_y = 10;

    // Smoothing sigma: 2.0 grid cells (approx 2.0 sigma) to kill molecular noise
    std::vector<std::vector<double>> interfaces = is_LG
        ? get_smooth_interface_LG(n_grid_y, 2.0)
        : get_smooth_interface(n_grid_y, 2.0);
    
    plot_interfaces_python(
        h_particles, interfaces, filename, csv_path,
        cfg_manager.config.box_w_global, cfg_manager.config.box_h_global,
        cfg_manager.config.SIGMA_AA, cfg_manager.config.SIGMA_BB
    );
    
    RankZeroPrint("[DEBUG] plot_interfaces finished.\n");
}

namespace {
void plot_cwa_python(const std::string& csv_path, const std::string& figure_path) {
    if (csv_path.empty() || figure_path.empty()) {
        throw std::invalid_argument("plot_cwa_python requires valid paths");
    }
    const std::string command =
        "python ./python/plot_cwa_python.py --csv_path \"" + csv_path +
        "\" --output \"" + figure_path + "\"";
    const int status = std::system(command.c_str());
    if (status != 0) {
        fmt::print(stderr, "[CWA] plot_cwa_python.py failed with status {}\n", status);
    }
}
} // namespace

void MDSimulation::do_CWA_instant(int q_min, int q_max, const std::string& csv_path, const std::string& plot_path, bool is_plot, int step, bool is_LG) {
    if (cfg_manager.config.rank_idx != 0) {
        return;
    }

    if (q_min < 1 || q_max < q_min) {
        fmt::print(stderr, "[CWA] Invalid q range [{} - {}].\n", q_min, q_max);
        return;
    }

    if (csv_path.empty()) {
        fmt::print(stderr, "[CWA] csv_path must not be empty.\n");
        return;
    }

    const double Ly = cfg_manager.config.box_h_global;
    if (Ly <= 0.0) {
        fmt::print(stderr, "[CWA] Invalid box height {}.\n", Ly);
        return;
    }

    int n_grid_y = static_cast<int>(cfg_manager.config.box_h_global / 2.0);
    if (n_grid_y < 32) n_grid_y = 32;
    const double smoothing_sigma = 2.0;

    std::vector<std::vector<double>> interface_paths = is_LG
        ? compute_interface_paths_LG(n_grid_y, smoothing_sigma)
        : compute_interface_paths(n_grid_y, smoothing_sigma);
    if (interface_paths.empty()) {
        RankZeroPrint("[CWA] No interfaces detected for analysis.\n");
        return;
    }

    const double dy = Ly / n_grid_y;
    const double pi = 3.14159265358979323846;
    const int nyquist = n_grid_y / 2;
    const int max_mode = std::min(q_max, nyquist);
    if (max_mode < q_min) {
        fmt::print(stderr, "[CWA] q_max={} is below q_min={} for current grid.\n", q_max, q_min);
        return;
    }

    std::vector<double> mode_accum(static_cast<size_t>(max_mode - q_min + 1), 0.0);
    int n_interfaces_used = 0;

    for (const auto& path : interface_paths) {
        if (path.size() < static_cast<size_t>(n_grid_y)) {
            continue;
        }

        std::vector<double> centered(path.begin(), path.begin() + n_grid_y);
        double mean_x = std::accumulate(centered.begin(), centered.end(), 0.0) / centered.size();
        for (double& value : centered) {
            value -= mean_x;
        }

        bool has_variation = std::any_of(centered.begin(), centered.end(), [](double v) {
            return std::abs(v) > 1e-12;
        });
        if (!has_variation) {
            continue;
        }

        for (int mode = q_min; mode <= max_mode; ++mode) {
            std::complex<double> accum{0.0, 0.0};
            double q_value = 2.0 * pi * static_cast<double>(mode) / Ly;
            for (int j = 0; j < n_grid_y; ++j) {
                double y_pos = (static_cast<double>(j) + 0.5) * dy;
                double phase = -q_value * y_pos;
                accum += std::complex<double>(std::cos(phase), std::sin(phase)) * centered[j];
            }
            double norm = 1.0 / static_cast<double>(n_grid_y);
            double magnitude_sq = std::norm(accum) * (norm * norm);
            mode_accum[static_cast<size_t>(mode - q_min)] += magnitude_sq;
        }
        ++n_interfaces_used;
    }

    if (n_interfaces_used == 0) {
        RankZeroPrint("[CWA] No valid interfaces for FFT.\n");
        return;
    }

    std::vector<int> modes_used;
    std::vector<double> q_sq;
    std::vector<double> spectrum;
    std::vector<double> c_q; // k_B T / (Ly * S(q)) so ideal theory is gamma * q^2
    modes_used.reserve(static_cast<size_t>(max_mode - q_min + 1));
    q_sq.reserve(modes_used.capacity());
    spectrum.reserve(modes_used.capacity());
    c_q.reserve(modes_used.capacity());

    const double temperature = cfg_manager.config.T_target;
    for (int mode = q_min; mode <= max_mode; ++mode) {
        double q_value = 2.0 * pi * static_cast<double>(mode) / Ly;
        double q2 = q_value * q_value;
        double avg_mag = mode_accum[static_cast<size_t>(mode - q_min)] / static_cast<double>(n_interfaces_used);
        if (!std::isfinite(avg_mag) || avg_mag <= 0.0) {
            continue;
        }
        double c_val = temperature / (Ly * avg_mag);
        if (!std::isfinite(c_val)) {
            continue;
        }
        modes_used.push_back(mode);
        q_sq.push_back(q2);
        spectrum.push_back(avg_mag);
        c_q.push_back(c_val);
    }

    if (q_sq.empty()) {
        RankZeroPrint("[CWA] No usable q-modes for regression.\n");
        return;
    }

    double mean_x = std::accumulate(q_sq.begin(), q_sq.end(), 0.0) / q_sq.size();
    double mean_y = std::accumulate(c_q.begin(), c_q.end(), 0.0) / c_q.size();
    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t i = 0; i < q_sq.size(); ++i) {
        double dx = q_sq[i] - mean_x;
        double dy_val = c_q[i] - mean_y;
        numerator += dx * dy_val;
        denominator += dx * dx;
    }

    if (denominator <= 1e-16) {
        RankZeroPrint("[CWA] Unable to determine slope for gamma (degenerate q-range).\n");
        return;
    }

    const double gamma = numerator / denominator;
    const double C0 = mean_y - gamma * mean_x; // intercept to absorb short-wavelength bias

    if (gamma <= 0.0) {
        RankZeroPrint("[CWA] Non-positive gamma from regression; skipping output.\n");
        return;
    }

    fmt::memory_buffer buffer;
    fmt::format_to(std::back_inserter(buffer), "step, {}, gamma, {}, C0, {}, Ly, {}", step, gamma, C0, Ly);
    for (size_t idx = 0; idx < modes_used.size(); ++idx) {
        const int mode = modes_used[idx];
        fmt::format_to(std::back_inserter(buffer), ", q_sq_{}, {}", mode, q_sq[idx]);
        fmt::format_to(std::back_inserter(buffer), ", Cq_{}, {}", mode, c_q[idx]);
        fmt::format_to(std::back_inserter(buffer), ", S_q_{}, {}", mode, spectrum[idx]);
        fmt::format_to(std::back_inserter(buffer), ", hq_sq_{}, {}", mode, spectrum[idx]);
    }
    fmt::format_to(std::back_inserter(buffer), "\n");
    std::string line(buffer.data(), buffer.size());
    write_to_file(csv_path, "{}", line);

    if (is_plot) {
        std::string figure_path = plot_path.empty() ? csv_path : plot_path;
        if (figure_path == csv_path) {
            auto pos = figure_path.rfind('.');
            if (pos == std::string::npos) {
                figure_path += ".svg";
            } else {
                figure_path.replace(pos, std::string::npos, ".svg");
            }
        }
        try {
            plot_cwa_python(csv_path, figure_path);
        } catch (const std::exception& e) {
            fmt::print(stderr, "[CWA] Plotting failed: {}\n", e.what());
        }
    }
}

std::vector<std::vector<double>> MDSimulation::compute_interface_paths(int n_grid_y, double smoothing_sigma) {
    std::vector<std::vector<double>> result;
    if (cfg_manager.config.rank_idx != 0) return result;

    const double Lx = cfg_manager.config.box_w_global;
    const double Ly = cfg_manager.config.box_h_global;
    
    // Calculate aspect-ratio corrected X grid size
    int n_grid_x = static_cast<int>(n_grid_y * (Lx / Ly));
    if (n_grid_x < 10) n_grid_x = 10;

    std::vector<double> grid(n_grid_x * n_grid_y, 0.0);
    double dx = Lx / n_grid_x;
    double dy = Ly / n_grid_y;

    // 1. Binning (Density Field)
    for (const auto& p : h_particles) {
        int ix = static_cast<int>(p.pos.x / dx);
        int iy = static_cast<int>(p.pos.y / dy);
        // PBC clamp
        ix = (ix % n_grid_x + n_grid_x) % n_grid_x;
        iy = (iy % n_grid_y + n_grid_y) % n_grid_y;
        
        // Order Parameter: Type 0 (+1) vs Type 1 (-1)
        grid[iy * n_grid_x + ix] += (p.type == 0 ? 1.0 : -1.0);
    }

    // 2. Gaussian Smoothing (X-axis)
    // Smoothing primarily along X removes density jumps perpendicular to interface
    // (Smoothing along Y removes waves - use carefully!)
    int passes = static_cast<int>(std::ceil(smoothing_sigma));
    std::vector<double> temp_grid = grid;
    
    for (int p = 0; p < passes; ++p) {
        std::vector<double> next_grid = temp_grid;
        for (int y = 0; y < n_grid_y; ++y) {
            for (int x = 0; x < n_grid_x; ++x) {
                int xm = (x - 1 + n_grid_x) % n_grid_x;
                int xp = (x + 1) % n_grid_x;
                // 3-point blur along X
                next_grid[y*n_grid_x + x] = 0.25*temp_grid[y*n_grid_x + xm] + 
                                            0.50*temp_grid[y*n_grid_x + x] + 
                                            0.25*temp_grid[y*n_grid_x + xp];
            }
        }
        temp_grid = next_grid;
    }

    // 3. Identify Anchors (Main Interfaces) using 1D Projection
    // This forces us to find exactly 2 interfaces and ignore bubbles.
    std::vector<double> profile_1d(n_grid_x, 0.0);
    for(int x=0; x<n_grid_x; ++x) {
        for(int y=0; y<n_grid_y; ++y) profile_1d[x] += temp_grid[y*n_grid_x + x];
    }
    
    std::vector<double> anchors;
    for(int x=0; x<n_grid_x; ++x) {
        int x_next = (x+1)%n_grid_x;
        // Zero crossing in 1D profile
        if(profile_1d[x] * profile_1d[x_next] <= 0.0) {
             anchors.push_back((x + 0.5) * dx);
        }
    }
    
    // Enforce exactly 2 interfaces if possible (filter spurious ones)
    if(anchors.size() > 2) {
        // Heuristic: Find the pair with largest separation
        // Or just take first 2 if clean. 
        anchors.resize(2); 
    } else if (anchors.empty()) {
        return result; 
    }

    // 4. Cluster Crossings to Anchors & Build Paths
    struct Point { double x, y; };
    result.resize(anchors.size());

    for(int k=0; k<(int)anchors.size(); ++k) {
        double ref_loc = anchors[k];
        
        // Storage for the averaged path: x_mean per y-bin
        std::vector<double> x_means(n_grid_y, 0.0);
        std::vector<int> counts(n_grid_y, 0);

        // Scan every row for zero crossings near this anchor
        for (int y = 0; y < n_grid_y; ++y) {
            for (int x = 0; x < n_grid_x; ++x) {
                int idx = y * n_grid_x + x;
                int idx_next = y * n_grid_x + (x + 1) % n_grid_x;
                double v1 = temp_grid[idx];
                double v2 = temp_grid[idx_next];

                if (v1 * v2 <= 0.0 && v1 != v2) {
                    // Sub-grid position
                    double frac = std::abs(v1) / (std::abs(v1) + std::abs(v2));
                    double cx = (x + frac) * dx;
                    
                    // Check distance to current anchor (PBC aware)
                    double dist = cx - ref_loc;
                    while(dist > Lx/2.0) dist -= Lx;
                    while(dist < -Lx/2.0) dist += Lx;
                    
                    // Only accept if it belongs to THIS anchor (closest one)
                    // and is within reasonable range (e.g. 1/4 box width)
                    if (std::abs(dist) < Lx/4.0) {
                        // We accumulate the RELATIVE distance (dx) to avoid PBC averaging bugs
                        // (e.g. averaging x=0.1 and x=L-0.1 should give x=0.0, not L/2)
                        x_means[y] += dist; // Sum of relative offsets
                        counts[y]++;
                    }
                }
            }
        }

        // Post-Process Path: Averaging & Interpolation
        std::vector<double> final_path_x(n_grid_y, 0.0);
        std::vector<bool> valid(n_grid_y, false);

        for(int y=0; y<n_grid_y; ++y) {
            if(counts[y] > 0) {
                double avg_dx = x_means[y] / counts[y];
                double abs_x = ref_loc + avg_dx;
                // Wrap back to domain [0, Lx]
                while(abs_x < 0) abs_x += Lx;
                while(abs_x >= Lx) abs_x -= Lx;
                
                final_path_x[y] = abs_x;
                valid[y] = true;
            }
        }

        // Interpolate gaps (Linear or Hold)
        // 1. Find any valid point to start
        int start_idx = -1;
        for(int i=0; i<n_grid_y; ++i) if(valid[i]) { start_idx=i; break; }

        if(start_idx == -1) {
            // No points found for this anchor? Fallback to straight line
            std::fill(final_path_x.begin(), final_path_x.end(), ref_loc);
        } else {
            // Forward fill + Wrap
            // We iterate 2*N times to ensure we cover gaps crossing the periodic boundary
            double last_val = final_path_x[start_idx];
            
            // First pass: fill from start_idx to end
            for(int i=0; i<n_grid_y; ++i) {
                int idx = (start_idx + i) % n_grid_y;
                if(valid[idx]) last_val = final_path_x[idx];
                else final_path_x[idx] = last_val; // Simple hold interpolation
            }
            // Second pass: fill from 0 to start_idx (if any gaps remained at start)
             for(int i=0; i<n_grid_y; ++i) {
                int idx = (start_idx + i) % n_grid_y;
                if(valid[idx]) last_val = final_path_x[idx];
                else final_path_x[idx] = last_val;
            }
        }

        result[k] = std::move(final_path_x);
    }

    return result;
}

std::vector<std::vector<double>> MDSimulation::compute_interface_paths_LG(int n_grid_y, double smoothing_sigma) {
    std::vector<std::vector<double>> result;
    if (cfg_manager.config.rank_idx != 0) return result;

    const double Lx = cfg_manager.config.box_w_global;
    const double Ly = cfg_manager.config.box_h_global;
    
    int n_grid_x = static_cast<int>(n_grid_y * (Lx / Ly));
    if (n_grid_x < 10) n_grid_x = 10;

    std::vector<double> grid(n_grid_x * n_grid_y, 0.0);
    double dx = Lx / n_grid_x;
    double dy = Ly / n_grid_y;

    // 1. Binning (scalar density field)
    for (const auto& p : h_particles) {
        int ix = static_cast<int>(p.pos.x / dx);
        int iy = static_cast<int>(p.pos.y / dy);
        ix = (ix % n_grid_x + n_grid_x) % n_grid_x;
        iy = (iy % n_grid_y + n_grid_y) % n_grid_y;
        grid[iy * n_grid_x + ix] += 1.0;
    }

    // 2. Gaussian smoothing along X (as in compute_interface_paths)
    int passes = static_cast<int>(std::ceil(smoothing_sigma));
    std::vector<double> temp_grid = grid;
    
    for (int p = 0; p < passes; ++p) {
        std::vector<double> next_grid = temp_grid;
        for (int y = 0; y < n_grid_y; ++y) {
            for (int x = 0; x < n_grid_x; ++x) {
                int xm = (x - 1 + n_grid_x) % n_grid_x;
                int xp = (x + 1) % n_grid_x;
                next_grid[y*n_grid_x + x] = 0.25*temp_grid[y*n_grid_x + xm] + 
                                            0.50*temp_grid[y*n_grid_x + x] + 
                                            0.25*temp_grid[y*n_grid_x + xp];
            }
        }
        temp_grid = next_grid;
    }

    // 3. Build order parameter phi = density - mean_density
    double sum_rho = 0.0;
    for (double v : temp_grid) sum_rho += v;
    const double mean_rho = sum_rho / static_cast<double>(n_grid_x * n_grid_y);

    std::vector<double> phi(temp_grid.size());
    for (std::size_t idx = 0; idx < temp_grid.size(); ++idx) {
        phi[idx] = temp_grid[idx] - mean_rho;
    }

    // 4. Identify anchors from 1D projection of phi
    std::vector<double> profile_1d(n_grid_x, 0.0);
    for (int x = 0; x < n_grid_x; ++x) {
        for (int y = 0; y < n_grid_y; ++y) {
            profile_1d[x] += phi[y * n_grid_x + x];
        }
    }
    
    std::vector<double> anchors;
    for (int x = 0; x < n_grid_x; ++x) {
        int x_next = (x + 1) % n_grid_x;
        if (profile_1d[x] * profile_1d[x_next] <= 0.0) {
            anchors.push_back((x + 0.5) * dx);
        }
    }

    if (anchors.size() > 2) {
        anchors.resize(2);
    } else if (anchors.empty()) {
        return result;
    }

    // 5. Build paths per anchor using zero-crossings of phi
    result.resize(anchors.size());

    for (int k = 0; k < static_cast<int>(anchors.size()); ++k) {
        double ref_loc = anchors[k];

        std::vector<double> x_means(n_grid_y, 0.0);
        std::vector<int> counts(n_grid_y, 0);

        for (int y = 0; y < n_grid_y; ++y) {
            for (int x = 0; x < n_grid_x; ++x) {
                int idx = y * n_grid_x + x;
                int idx_next = y * n_grid_x + (x + 1) % n_grid_x;
                double v1 = phi[idx];
                double v2 = phi[idx_next];

                if (v1 * v2 <= 0.0 && v1 != v2) {
                    double frac = std::abs(v1) / (std::abs(v1) + std::abs(v2));
                    double cx = (x + frac) * dx;

                    double dist = cx - ref_loc;
                    while (dist > Lx / 2.0)  dist -= Lx;
                    while (dist < -Lx / 2.0) dist += Lx;

                    if (std::abs(dist) < Lx / 4.0) {
                        x_means[y] += dist;
                        counts[y]  += 1;
                    }
                }
            }
        }

        std::vector<double> final_path_x(n_grid_y, 0.0);
        std::vector<bool> valid(n_grid_y, false);

        for (int y = 0; y < n_grid_y; ++y) {
            if (counts[y] > 0) {
                double avg_dx = x_means[y] / counts[y];
                double abs_x  = ref_loc + avg_dx;
                while (abs_x < 0.0)  abs_x += Lx;
                while (abs_x >= Lx)  abs_x -= Lx;
                final_path_x[y] = abs_x;
                valid[y] = true;
            }
        }

        int start_idx = -1;
        for (int i = 0; i < n_grid_y; ++i) {
            if (valid[i]) { start_idx = i; break; }
        }

        if (start_idx == -1) {
            std::fill(final_path_x.begin(), final_path_x.end(), ref_loc);
        } else {
            double last_val = final_path_x[start_idx];
            for (int i = 0; i < n_grid_y; ++i) {
                int idx = (start_idx + i) % n_grid_y;
                if (valid[idx]) last_val = final_path_x[idx];
                else            final_path_x[idx] = last_val;
            }
            for (int i = 0; i < n_grid_y; ++i) {
                int idx = (start_idx + i) % n_grid_y;
                if (valid[idx]) last_val = final_path_x[idx];
                else            final_path_x[idx] = last_val;
            }
        }

        result[k] = std::move(final_path_x);
    }

    return result;
}

std::vector<std::vector<double>> MDSimulation::get_smooth_interface(int n_grid_y, double smoothing_sigma) {
    std::vector<std::vector<double>> paths = compute_interface_paths(n_grid_y, smoothing_sigma);
    std::vector<std::vector<double>> result;
    if (paths.empty()) {
        return result;
    }

    const double Lx = cfg_manager.config.box_w_global;
    const double Ly = cfg_manager.config.box_h_global;
    double dy = Ly / n_grid_y;

    result.resize(paths.size());
    for (size_t k = 0; k < paths.size(); ++k) {
        const auto& path = paths[k];
        if (path.empty()) continue;

        std::vector<double> segs;
        segs.reserve(path.size() * 4);
        for (int y = 0; y < n_grid_y; ++y) {
            double x1 = path[y];
            double y1 = (y + 0.5) * dy;

            int next_y = (y + 1) % n_grid_y;
            double x2 = path[next_y];
            double y2 = (y + 1.5) * dy;

            double dist_x = std::abs(x2 - x1);
            if (dist_x < Lx * 0.5) {
                segs.push_back(x1);
                segs.push_back(y1);
                segs.push_back(x2);
                segs.push_back(y2);
            }
        }
        result[k] = std::move(segs);
    }

    return result;
}

std::vector<std::vector<double>> MDSimulation::get_smooth_interface_LG(int n_grid_y, double smoothing_sigma) {
    std::vector<std::vector<double>> paths = compute_interface_paths_LG(n_grid_y, smoothing_sigma);
    std::vector<std::vector<double>> result;
    if (paths.empty()) {
        return result;
    }

    const double Lx = cfg_manager.config.box_w_global;
    const double Ly = cfg_manager.config.box_h_global;
    double dy = Ly / n_grid_y;

    result.resize(paths.size());
    for (size_t k = 0; k < paths.size(); ++k) {
        const auto& path = paths[k];
        if (path.empty()) continue;

        std::vector<double> segs;
        segs.reserve(path.size() * 4);
        for (int y = 0; y < n_grid_y; ++y) {
            double x1 = path[y];
            double y1 = (y + 0.5) * dy;

            int next_y = (y + 1) % n_grid_y;
            double x2 = path[next_y];
            double y2 = (y + 1.5) * dy;

            double dist_x = std::abs(x2 - x1);
            if (dist_x < Lx * 0.5) {
                segs.push_back(x1);
                segs.push_back(y1);
                segs.push_back(x2);
                segs.push_back(y2);
            }
        }
        result[k] = std::move(segs);
    }

    return result;
}

std::vector<double> MDSimulation::get_pressure_profile(int n_bins_local)
{
    std::vector<double> empty_result;
    if (n_bins_local <= 0) {
        return empty_result;
    }

    const int n_local = cfg_manager.config.n_local;
    const int n_left  = cfg_manager.config.n_halo_left;
    const int n_right = cfg_manager.config.n_halo_right;
    int threads       = cfg_manager.config.THREADS_PER_BLOCK;
    if (threads <= 0) {
        threads = 256;
    }

    const int blocks = (n_local + threads - 1) / threads;

    thrust::device_vector<double> d_P_xx(n_bins_local, 0.0);
    thrust::device_vector<double> d_P_yy(n_bins_local, 0.0);
    thrust::device_vector<double> d_P_xy(n_bins_local, 0.0);

    if (blocks > 0) {
        local_pressure_tensor_profile_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_particles.data()),
            thrust::raw_pointer_cast(d_particles_halo_left.data()),
            thrust::raw_pointer_cast(d_particles_halo_right.data()),
            n_local,
            n_left,
            n_right,
            cfg_manager.config.box_w_global,
            cfg_manager.config.box_h_global,
            cfg_manager.config.MASS_A,
            cfg_manager.config.MASS_B,
            cfg_manager.config.SIGMA_AA,
            cfg_manager.config.SIGMA_BB,
            cfg_manager.config.SIGMA_AB,
            cfg_manager.config.EPSILON_AA,
            cfg_manager.config.EPSILON_BB,
            cfg_manager.config.EPSILON_AB,
            cfg_manager.config.cutoff,
            n_bins_local,
            thrust::raw_pointer_cast(d_P_xx.data()),
            thrust::raw_pointer_cast(d_P_yy.data()),
            thrust::raw_pointer_cast(d_P_xy.data())
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<double> local_Pxx(static_cast<std::size_t>(n_bins_local), 0.0);
    std::vector<double> local_Pyy(static_cast<std::size_t>(n_bins_local), 0.0);
    std::vector<double> local_Pxy(static_cast<std::size_t>(n_bins_local), 0.0);

    if (n_bins_local > 0) {
        thrust::copy(d_P_xx.begin(), d_P_xx.end(), local_Pxx.begin());
        thrust::copy(d_P_yy.begin(), d_P_yy.end(), local_Pyy.begin());
        thrust::copy(d_P_xy.begin(), d_P_xy.end(), local_Pxy.begin());
    }

    int world_rank = 0;
    MPI_Comm_rank(comm, &world_rank);

    std::vector<double> global_Pxx;
    std::vector<double> global_Pyy;
    std::vector<double> global_Pxy;

    if (world_rank == 0) {
        global_Pxx.assign(static_cast<std::size_t>(n_bins_local), 0.0);
        global_Pyy.assign(static_cast<std::size_t>(n_bins_local), 0.0);
        global_Pxy.assign(static_cast<std::size_t>(n_bins_local), 0.0);
    }

    MPI_Reduce(local_Pxx.data(),
               world_rank == 0 ? global_Pxx.data() : nullptr,
               n_bins_local,
               MPI_DOUBLE,
               MPI_SUM,
               0,
               comm);
    MPI_Reduce(local_Pyy.data(),
               world_rank == 0 ? global_Pyy.data() : nullptr,
               n_bins_local,
               MPI_DOUBLE,
               MPI_SUM,
               0,
               comm);
    MPI_Reduce(local_Pxy.data(),
               world_rank == 0 ? global_Pxy.data() : nullptr,
               n_bins_local,
               MPI_DOUBLE,
               MPI_SUM,
               0,
               comm);

    if (world_rank != 0) {
        return empty_result;
    }

    std::vector<double> result(static_cast<std::size_t>(n_bins_local) * 3u, 0.0);

    const double Lx = cfg_manager.config.box_w_global;
    const double Ly = cfg_manager.config.box_h_global;
    const double dy = Ly / static_cast<double>(n_bins_local);
    const double volume = Lx * dy;

    for (int k = 0; k < n_bins_local; ++k) {
        double Pxx = global_Pxx[static_cast<std::size_t>(k)];
        double Pyy = global_Pyy[static_cast<std::size_t>(k)];
        double Pxy = global_Pxy[static_cast<std::size_t>(k)];

        if (volume > 0.0) {
            Pxx /= volume;
            Pyy /= volume;
            Pxy /= volume;
        }

        result[static_cast<std::size_t>(k)]                          = Pxx;
        result[static_cast<std::size_t>(k + n_bins_local)]           = Pyy;
        result[static_cast<std::size_t>(k + 2 * n_bins_local)]       = Pxy;
    }

    return result;
}

std::vector<int> MDSimulation::get_N_profile(int n_bins_per_rank)
{
    std::vector<int> count_A(n_bins_per_rank, 0);
    std::vector<int> count_B(n_bins_per_rank, 0);

    const int n_local = cfg_manager.config.n_local;
    const int threads = cfg_manager.config.THREADS_PER_BLOCK;
    const double xmin = cfg_manager.config.x_min;
    const double xmax = cfg_manager.config.x_max;

    int blocks = (n_local + threads - 1) / threads;
    size_t shared_mem_size = static_cast<size_t>(2 * n_bins_per_rank * sizeof(int));

    thrust::device_vector<int> d_count_A(n_bins_per_rank, 0);
    thrust::device_vector<int> d_count_B(n_bins_per_rank, 0);

    if (blocks > 0) {
        local_density_profile_kernel<<<blocks, threads, shared_mem_size>>>(
            thrust::raw_pointer_cast(d_particles.data()),
            n_local,
            n_bins_per_rank,
            xmin,
            xmax,
            thrust::raw_pointer_cast(d_count_A.data()),
            thrust::raw_pointer_cast(d_count_B.data())
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (n_bins_per_rank > 0) {
        thrust::copy(d_count_A.begin(), d_count_A.end(), count_A.begin());
        thrust::copy(d_count_B.begin(), d_count_B.end(), count_B.begin());
    }

    int world_size = 1;
    int world_rank = 0;
    MPI_Comm_size(comm, &world_size);
    MPI_Comm_rank(comm, &world_rank);

    const int local_bins = n_bins_per_rank;
    const int total_bins = local_bins * world_size;

    std::vector<int> all_A;
    std::vector<int> all_B;

    if (world_rank == 0) {
        all_A.resize(total_bins);
        all_B.resize(total_bins);
    }

    MPI_Gather(
        count_A.data(), local_bins, MPI_INT,
        world_rank == 0 ? all_A.data() : nullptr, local_bins, MPI_INT,
        0, comm
    );
    MPI_Gather(
        count_B.data(), local_bins, MPI_INT,
        world_rank == 0 ? all_B.data() : nullptr, local_bins, MPI_INT,
        0, comm
    );

    std::vector<int> result;
    if (world_rank == 0) {
        result.resize(static_cast<std::size_t>(total_bins) * 2u);
        for (int i = 0; i < total_bins; ++i) {
            result[i] = all_A[i];
            result[i + total_bins] = all_B[i];
        }
    } else {
        result.resize(static_cast<std::size_t>(total_bins) * 2u);
    }

    if (total_bins > 0) {
        MPI_Bcast(result.data(),
                  total_bins * 2,
                  MPI_INT,
                  0,
                  comm);
    }

    return result;
}

struct GraphEdge {
    size_t u;
    size_t v;
    double length;
};

std::vector<std::vector<double>> MDSimulation::locate_interface(const delaunator::Delaunator& d) {

    std::vector<std::vector<double>> result;

    if (cfg_manager.config.rank_idx != 0 || d.triangles.empty() || vertex_to_idx.empty()) {
        return result;
    }

    const double Ly = cfg_manager.config.box_h_global;
    size_t num_triangles = d.triangles.size() / 3;

    std::vector<GraphEdge> interface_edges;
    std::set<std::pair<size_t, size_t>> processed_edges;

    for (size_t i = 0; i < num_triangles; ++i) {
        for (int j = 0; j < 3; ++j) {
            size_t v1 = d.triangles[3 * i + j];
            size_t v2 = d.triangles[3 * i + (j + 1) % 3];

            if (v1 > v2) std::swap(v1, v2);

            if (processed_edges.count({v1, v2})) continue;
            processed_edges.insert({v1, v2});

            int idx1 = vertex_to_idx[v1];
            int idx2 = vertex_to_idx[v2];

            if (h_particles[idx1].type != h_particles[idx2].type) {
                double x1 = d.coords[2 * v1];
                double y1 = d.coords[2 * v1 + 1];
                double x2 = d.coords[2 * v2];
                double y2 = d.coords[2 * v2 + 1];

                double len = std::sqrt(std::pow(x1-x2, 2) + std::pow(y1-y2, 2));
                interface_edges.push_back({v1, v2, len});
            }
        }
    }

    if (interface_edges.empty()) return result;
    size_t max_v = d.coords.size() / 2;
    std::vector<int> parent(max_v);
    std::iota(parent.begin(), parent.end(), 0);

    auto find_set = [&](int i, auto&& find_ref) -> int {
        if (parent[i] == i) return i;
        return parent[i] = find_ref(parent[i], find_ref);
    };
    auto union_sets = [&](int i, int j) {
        int root_i = find_set(i, find_set);
        int root_j = find_set(j, find_set);
        if (root_i != root_j) parent[root_i] = root_j;
    };

    for (const auto& edge : interface_edges) union_sets(edge.u, edge.v);

    std::map<int, std::vector<GraphEdge>> components;
    std::map<int, double> component_lengths;

    for (const auto& edge : interface_edges) {
        int root = find_set(edge.u, find_set);
        components[root].push_back(edge);
        component_lengths[root] += edge.length;
    }

    for (auto& comp : components) {
        if (component_lengths[comp.first] > Ly) {
            std::vector<double> segment_coords;
            segment_coords.reserve(comp.second.size() * 4);
            for (const auto& edge : comp.second) {
                segment_coords.push_back(d.coords[2 * edge.u]);
                segment_coords.push_back(d.coords[2 * edge.u + 1]);
                segment_coords.push_back(d.coords[2 * edge.v]);
                segment_coords.push_back(d.coords[2 * edge.v + 1]);
            }
            result.push_back(std::move(segment_coords));
        }
    }
    return result;

}


std::vector<double> MDSimulation::get_density_profile(int n_bins_per_rank) {
    std::vector<int> n_profile = get_N_profile(n_bins_per_rank);
    size_t total_bins = n_profile.size() / 2;
    std::vector<double> result(total_bins);

    for (size_t i = 0; i < total_bins; ++i) {
        double n_a = static_cast<double>(n_profile[i]);
        double n_b = static_cast<double>(n_profile[i + total_bins]);
        double sum = n_a + n_b;

        if (sum == 0.0) {
            result[i] = 0.0;
        } else {
            result[i] = (n_a - n_b) / sum;
        }
    }

    return result;
}


void MDSimulation::save_env(const std::string& filename, const int step) {
    // Only Rank 0 performs I/O
    if (cfg_manager.config.rank_idx == 0) {
        // Initialize writer if not already open
        if (!particle_writer) {
            // Try to open in append mode first (assuming trajectory accumulation)
            // Your FileWriter throws if append=true and file doesn't exist.
            try {
                particle_writer = std::make_unique<FileWriter>(filename, true);
            } catch (const std::exception&) {
                // File likely doesn't exist, create new (append=false)
                particle_writer = std::make_unique<FileWriter>(filename, false);
            }
        }
        // h_particles is assumed to hold all particles on Rank 0 (gathered via sample_collect)
        particle_writer->write_frame(h_particles.data(), h_particles.size(), step);
    }
}
