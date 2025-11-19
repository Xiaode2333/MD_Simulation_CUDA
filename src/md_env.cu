#include "md_env.hpp"
#include "md_particle.hpp"
#include "md_config.hpp"
#include "md_common.hpp"
#include "md_cuda_common.hpp"


#include <iostream>
#include <numeric>
#include <map>
#include <set>

MDSimulation::MDSimulation(MDConfigManager config_manager, MPI_Comm comm) {
    this->cfg_manager = config_manager;
    this->comm = comm;
    xi = 0.0;

    fmt::print("Starting broadcasting params.\n");
    std::fflush(stdout); // FORCE FLUSH
    
    broadcast_params();
    
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

    // collect_particles_d2h(); // Disabled in your snippet

    if (cfg_manager.config.rank_idx == 0) {
        fmt::print("[Rank] {}/{}. MD simulation env initialized.\n", cfg_manager.config.rank_idx, cfg_manager.config.rank_size);
        std::fflush(stdout);
    }
}

MDSimulation::~MDSimulation() = default;

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

    // WARNING: MDConfig contains std::string. MPI_Bcast on standard layout structs with strings 
    // sends POINTERS, causing segfaults on receivers. 
    // Ensure MDConfig uses char arrays (e.g. char run_name[64]) OR serialize this properly.
    MPI_Bcast(&cfg_manager.config, static_cast<int>(sizeof(MDConfig)), MPI_BYTE, 0, comm);

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
    fmt::print("[broadcast_params] Rank {}\n", world_rank);
    std::fflush(stdout);
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

std::optional<delaunator::Delaunator> MDSimulation::triangulation_plot(bool is_plot, const std::string& filename, const std::string& csv_path, const std::vector<double>& rho)
{
    if (cfg_manager.config.rank_idx != 0) {
        return std::nullopt;
    }

    const auto& cfg = cfg_manager.config;
    const double Lx = cfg.box_w_global;
    const double Ly = cfg.box_h_global;

    double sigma_max = std::max({cfg.SIGMA_AA, cfg.SIGMA_BB, cfg.SIGMA_AB});
    
    // Define w = 10 * cutoff * sigma_max
    double w = 10.0 * cfg.cutoff * sigma_max;
    
    // Use w for both dimensions as requested
    double width_x = w;
    double width_y = w;

    std::vector<double> interfaces;
    if (!rho.empty()) {
        const int N_bins = static_cast<int>(rho.size());
        if (N_bins > 0) {
            const double dx = Lx / static_cast<double>(N_bins);
            for (int i = 0; i < N_bins; ++i) {
                int idx_left  = (i - 1 + N_bins) % N_bins;
                int idx_right = (i + 1) % N_bins;

                if (rho[idx_left] * rho[idx_right] <= 0.0) {
                    double x_loc = (i + 0.5) * dx; 
                    interfaces.push_back(x_loc);
                }
            }
        }
    }

    _build_interface_coords(
        h_particles, Lx, Ly, interfaces, width_x, width_y,
        coords, vertex_to_idx 
    );

    if (coords.size() < 6) {
        if (is_plot) {
            RankZeroPrint("[Warning] Not enough points for triangulation ({} points). Skipping.\n", coords.size()/2);
        }
        return std::nullopt;
    }

    try {
        delaunator::Delaunator d(coords);

        if (is_plot) {
            plot_triangulation_python(
                h_particles, d, filename, csv_path,
                Lx, Ly, cfg.SIGMA_AA, cfg.SIGMA_BB
            );
        }
        return d;
    } catch (const std::exception& e) {
        RankZeroPrint("[Warning] Triangulation failed (likely collinear points). Skipping plot. Reason: {}\n", e.what());
        return std::nullopt;
    }
}

void MDSimulation::plot_interfaces(const std::string& filename, const std::string& csv_path, const std::vector<double>& rho)
{
    if (cfg_manager.config.rank_idx != 0) return;

    RankZeroPrint("[DEBUG] Generating smooth interface (Grid Method)...\n");

    // Resolution: Use ~1.0 sigma resolution for fine enough detail to see waves,
    // but coarse enough that smoothing works.
    int n_grid_y = static_cast<int>(cfg_manager.config.box_h_global / 2.5); 
    if (n_grid_y < 10) n_grid_y = 10;

    // Smoothing sigma: 2.0 grid cells (approx 2.0 sigma) to kill molecular noise
    std::vector<std::vector<double>> interfaces = get_smooth_interface(n_grid_y, 2.5);
    
    plot_interfaces_python(
        h_particles, interfaces, filename, csv_path,
        cfg_manager.config.box_w_global, cfg_manager.config.box_h_global,
        cfg_manager.config.SIGMA_AA, cfg_manager.config.SIGMA_BB
    );
    
    RankZeroPrint("[DEBUG] plot_interfaces finished.\n");
}

std::vector<std::vector<double>> MDSimulation::get_smooth_interface(int n_grid_y, double smoothing_sigma) {
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

        // Generate Segments
        std::vector<double> segs;
        segs.reserve(n_grid_y * 4);
        
        for(int y=0; y<n_grid_y; ++y) {
            double x1 = final_path_x[y];
            double y1 = (y + 0.5) * dy;
            
            int next_y = (y + 1) % n_grid_y;
            double x2 = final_path_x[next_y];
            double y2 = (y + 1.5) * dy;

            // Horizontal Line Fix:
            // Do not draw segment if X jumps across boundary (wraps)
            double dist_x = std::abs(x2 - x1);
            if (dist_x < Lx * 0.5) {
                 segs.push_back(x1); segs.push_back(y1);
                 segs.push_back(x2); segs.push_back(y2);
            }
        }
        result[k] = std::move(segs);
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
