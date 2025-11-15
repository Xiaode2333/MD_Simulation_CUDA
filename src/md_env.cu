#include "md_env.hpp"
#include "md_particle.hpp"
#include "md_config.hpp"
#include "md_common.hpp"
#include "md_cuda_common.hpp"

MDSimulation::MDSimulation(MDConfigManager config_manager, MPI_Comm comm){
    this->cfg_manager = config_manager;
    this->comm = comm;
    fmt::print("Starting broadcasting params.\n");
    broadcast_params();
    fmt::print("Params broadcasted.\n");
    allocate_memory();
    fmt::print("Memory allocated.\n");
    init_particles();
}

MDSimulation::~MDSimulation() = default;

void MDSimulation::broadcast_params() {
    MPI_Bcast(&cfg_manager, static_cast<int>(sizeof(MDConfigManager)), MPI_BYTE, 0, comm);
}

void MDSimulation::allocate_memory(){
    h_particles.resize(cfg_manager.config.n_particles_global);
    d_particles.resize(cfg_manager.config.n_cap);
    d_particles_halo_left.resize(cfg_manager.config.halo_left_cap);
    d_particles_halo_right.resize(cfg_manager.config.halo_right_cap);
    d_send_left.resize(cfg_manager.config.halo_left_cap);
    d_send_right.resize(cfg_manager.config.halo_right_cap);
    d_keep.resize(cfg_manager.config.n_cap);

    flags_left.reserve(cfg_manager.config.n_cap);
    flags_right.reserve(cfg_manager.config.n_cap);
    flags_keep.reserve(cfg_manager.config.n_cap);
    pos_left.reserve(cfg_manager.config.n_cap);
    pos_right.reserve(cfg_manager.config.n_cap);
    pos_keep.reserve(cfg_manager.config.n_cap);
}



static inline double pbc_wrap(double x, double L) {
    if (L <= 0.0) {
        return x;
    }
    x = std::fmod(x, L);
    if (x < 0.0) {
        x += L;
    }
    return x;
}


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

        if (local_count > 0) { //directly receive to device memory
            if (static_cast<int>(d_particles.size()) < local_count) {
                fmt::print(stderr,
                           "[distribute_particles_h2d] rank {} d_particles size={} < n_local={}.\n",
                           rank_idx, d_particles.size(), local_count);
                MPI_Abort(comm, 1);
            }

            Particle* d_ptr = thrust::raw_pointer_cast(d_particles.data());

            // NOTE: requires CUDA-aware MPI: d_ptr is a device pointer.
            MPI_Recv(d_ptr,
                     local_count * static_cast<int>(sizeof(Particle)),
                     MPI_BYTE,
                     0,
                     101,
                     comm,
                     MPI_STATUS_IGNORE);
        }

        // // If cann't use MPI-aware GPU, use this host buffer.
        // h_particles.resize(N_global);
        // if (local_count > 0) {
            
        //     MPI_Recv(h_particles.data(),
        //              local_count * static_cast<int>(sizeof(Particle)),
        //              MPI_BYTE,
        //              0,
        //              101,
        //              comm,
        //              MPI_STATUS_IGNORE);

        //     if (static_cast<int>(d_particles.size()) < local_count) {
        //         fmt::print(stderr,
        //                    "[distribute_particles_h2d] rank {} d_particles size={} < n_local={}.\n",
        //                    rank_idx, d_particles.size(), local_count);
        //         MPI_Abort(comm, 1);
        //     }
        //     thrust::copy_n(h_particles.begin(), local_count, d_particles.begin());
        // }
    }

    // Build halos on device and exchange them by MPI (device to device)
    // update_halo();
}

// collect particles from device to host (rank 0 gathers all)
void MDSimulation::collect_particles_d2h() { // collect particles from device to host, only to host of rank 0
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

    // // Copy local device data to a temporary host buffer
    // std::vector<Particle> h_local;
    // if (n_local > 0) {
    //     h_local.resize(static_cast<std::size_t>(n_local));

    //     Particle* d_ptr = thrust::raw_pointer_cast(d_particles.data());
    //     CUDA_CHECK(cudaDeviceSynchronize());
    //     CUDA_CHECK(cudaMemcpy(h_local.data(),
    //                           d_ptr,
    //                           static_cast<std::size_t>(n_local) * sizeof(Particle),
    //                           cudaMemcpyDeviceToHost));
    // }

    // Gather per-rank particle counts (in units of "Particle") on rank 0
    std::vector<int> counts;
    int n_local_int = n_local;

    if (world_rank == 0) {
        counts.resize(rank_size);
    }

    MPI_Gather(&n_local_int,
               1,
               MPI_INT,
               world_rank == 0 ? counts.data() : nullptr,
               1,
               MPI_INT,
               0,
               comm);

    // On rank 0, prepare recv counts / displacements (in bytes) and check total
    std::vector<int> recvcounts_bytes;
    std::vector<int> displs_bytes;

    if (world_rank == 0) {
        recvcounts_bytes.resize(rank_size);
        displs_bytes.resize(rank_size);

        int offset_particles = 0;
        for (int r = 0; r < rank_size; ++r) {
            recvcounts_bytes[r] = counts[r] * static_cast<int>(sizeof(Particle));
            displs_bytes[r]     = offset_particles * static_cast<int>(sizeof(Particle));
            offset_particles   += counts[r];
        }

        const int total_particles = offset_particles;
        if (total_particles != N_global * static_cast<int>(sizeof(Particle))) {
            fmt::print(stderr,
                       "[collect_particles_d2h] sum of n_local across ranks ({} Particles) "
                       "!= n_particles_global={}.\n",
                       total_particles / static_cast<int>(sizeof(Particle)),
                       N_global);
            MPI_Abort(comm, 1);
        }

        if (static_cast<int>(h_particles.size()) < N_global) {
            h_particles.resize(static_cast<std::size_t>(N_global));
        }
    }

    // Gather actual particle data from all ranks to rank 0
    const int send_bytes = n_local_int * static_cast<int>(sizeof(Particle));
    void* sendbuf = (n_local_int > 0) ? thrust::raw_pointer_cast(d_particles.data()) : nullptr;

    void* recvbuf = nullptr;
    int*  recvcounts_ptr = nullptr;
    int*  displs_ptr     = nullptr;

    if (world_rank == 0) {
        recvbuf        = static_cast<void*>(h_particles.data());
        recvcounts_ptr = recvcounts_bytes.data();
        displs_ptr     = displs_bytes.data();
    }

    MPI_Gatherv(sendbuf,
                send_bytes,
                MPI_BYTE,
                recvbuf,
                recvcounts_ptr,
                displs_ptr,
                MPI_BYTE,
                0,
                comm);
}


// suppose d_particles is already updated
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

    const int    n_local      = cfg_manager.config.n_local;
    const int    n_cap        = cfg_manager.config.n_cap;
    const int    halo_left_cap  = cfg_manager.config.halo_left_cap;
    const int    halo_right_cap = cfg_manager.config.halo_right_cap;
    const double Lx           = cfg_manager.config.box_w_global;
    const double x_min        = cfg_manager.config.x_min;
    const double x_max        = cfg_manager.config.x_max;

    if (n_local > n_cap) {
        fmt::print(stderr,
                   "[update_halo] rank {} n_local={} exceeds n_cap={}.\n",
                   rank_idx, n_local, n_cap);
        MPI_Abort(comm, 1);
    }

    if (n_local == 0) {
        // Nothing to do
        return;
    }

    const double sigma_max = std::max(
        cfg_manager.config.SIGMA_AA,
        std::max(cfg_manager.config.SIGMA_BB, cfg_manager.config.SIGMA_AB)
    );
    const double halo_width = cfg_manager.config.cutoff * sigma_max * 1.1; // 1.1 safety factor

    Particle* d_local = thrust::raw_pointer_cast(d_particles.data());

    // thrust::device_vector<int> flags_left(n_local);
    // thrust::device_vector<int> flags_right(n_local);
    // thrust::device_vector<int> pos_left(n_local);
    // thrust::device_vector<int> pos_right(n_local);
    flags_left.resize(n_local);
    flags_right.resize(n_local);
    pos_left.resize(n_local);
    pos_right.resize(n_local);

    int threads = cfg_manager.config.THREADS_PER_BLOCK;
    if (threads <= 0) {
        threads = 256;
    }
    launch_mark_halo_kernel(d_local,
                            n_local,
                            x_min,
                            x_max,
                            Lx,
                            halo_width,
                            thrust::raw_pointer_cast(flags_left.data()),
                            thrust::raw_pointer_cast(flags_right.data()),
                            threads);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Prefix sums to get compact indices
    thrust::exclusive_scan(flags_left.begin(), flags_left.end(),
                           pos_left.begin());
    thrust::exclusive_scan(flags_right.begin(), flags_right.end(),
                           pos_right.begin());

    int send_left_count  = 0;
    int send_right_count = 0;

    // Compute counts from last flag and last position (device to host for two ints)
    if (n_local > 0) {
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

    // Temporary send buffers on device
    // thrust::device_vector<Particle> d_send_left(send_left_count);
    // thrust::device_vector<Particle> d_send_right(send_right_count);

    if (send_left_count > 0) {
        launch_pack_halo_kernel(
            d_local,
            n_local,
            thrust::raw_pointer_cast(flags_left.data()),
            thrust::raw_pointer_cast(pos_left.data()),
            send_left_count,
            thrust::raw_pointer_cast(d_send_left.data()),
            threads
        );
        CUDA_CHECK(cudaGetLastError());
    }

    if (send_right_count > 0) {
        launch_pack_halo_kernel(
            d_local,
            n_local,
            thrust::raw_pointer_cast(flags_right.data()),
            thrust::raw_pointer_cast(pos_right.data()),
            send_right_count,
            thrust::raw_pointer_cast(d_send_right.data()),
            threads
        );
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Exchange counts with neighbors
    int recv_left_count  = 0;
    int recv_right_count = 0;
    MPI_Status status;

    // Counts: send left, receive from right (right neighbor's left halo)
    MPI_Sendrecv(&send_left_count, 1, MPI_INT,
                 left_rank, 0,
                 &recv_right_count, 1, MPI_INT,
                 right_rank, 0,
                 comm, &status);

    // Counts: send right, receive from left (left neighbor's right halo)
    MPI_Sendrecv(&send_right_count, 1, MPI_INT,
                 right_rank, 1,
                 &recv_left_count, 1, MPI_INT,
                 left_rank, 1,
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

    Particle* d_recv_left  = (recv_left_count  > 0)
                             ? thrust::raw_pointer_cast(d_particles_halo_left.data())
                             : nullptr;
    Particle* d_recv_right = (recv_right_count > 0)
                             ? thrust::raw_pointer_cast(d_particles_halo_right.data())
                             : nullptr;

    Particle* d_send_left_ptr  = (send_left_count  > 0)
                                 ? thrust::raw_pointer_cast(d_send_left.data())
                                 : nullptr;
    Particle* d_send_right_ptr = (send_right_count > 0)
                                 ? thrust::raw_pointer_cast(d_send_right.data())
                                 : nullptr;

    // Device-to-device particle exchange (CUDA-aware MPI)
    MPI_Sendrecv(
        d_send_left_ptr,
        send_left_count * static_cast<int>(sizeof(Particle)),
        MPI_BYTE,
        left_rank, 2,
        d_recv_right,
        recv_right_count * static_cast<int>(sizeof(Particle)),
        MPI_BYTE,
        right_rank, 2,
        comm, &status
    );

    MPI_Sendrecv(
        d_send_right_ptr,
        send_right_count * static_cast<int>(sizeof(Particle)),
        MPI_BYTE,
        right_rank, 3,
        d_recv_left,
        recv_left_count * static_cast<int>(sizeof(Particle)),
        MPI_BYTE,
        left_rank, 3,
        comm, &status
    );

    cfg_manager.config.n_halo_left  = recv_left_count;
    cfg_manager.config.n_halo_right = recv_right_count;
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
    const double Lx     = cfg_manager.config.box_w_global;
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

    if (n_local > 0) {
        flags_left.resize(n_local);
        flags_right.resize(n_local);
        flags_keep.resize(n_local);
        pos_left.resize(n_local);
        pos_right.resize(n_local);
        pos_keep.resize(n_local);

        launch_mark_migration_kernel(
            d_local,
            n_local,
            x_min,
            x_max,
            thrust::raw_pointer_cast(flags_left.data()),
            thrust::raw_pointer_cast(flags_right.data()),
            thrust::raw_pointer_cast(flags_keep.data()),
            threads
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        thrust::exclusive_scan(flags_left.begin(),  flags_left.end(),  pos_left.begin());
        thrust::exclusive_scan(flags_right.begin(), flags_right.end(), pos_right.begin());
        thrust::exclusive_scan(flags_keep.begin(),  flags_keep.end(),  pos_keep.begin());

        send_left_count  = 0;
        send_right_count = 0;
        keep_count       = 0;

        // ADD: read last entries with cudaMemcpy instead of host indexing
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

        // d_send_left.resize(send_left_count);
        // d_send_right.resize(send_right_count);
        // d_keep.resize(keep_count);

        if (send_left_count > 0) {
            launch_pack_selected_kernel(
                d_local,
                n_local,
                thrust::raw_pointer_cast(flags_left.data()),
                thrust::raw_pointer_cast(pos_left.data()),
                send_left_count,
                thrust::raw_pointer_cast(d_send_left.data()),
                threads
            );
            CUDA_CHECK(cudaGetLastError());
        }

        if (send_right_count > 0) {
            launch_pack_selected_kernel(
                d_local,
                n_local,
                thrust::raw_pointer_cast(flags_right.data()),
                thrust::raw_pointer_cast(pos_right.data()),
                send_right_count,
                thrust::raw_pointer_cast(d_send_right.data()),
                threads
            );
            CUDA_CHECK(cudaGetLastError());
        }

        if (keep_count > 0) {
            launch_pack_selected_kernel(
                d_local,
                n_local,
                thrust::raw_pointer_cast(flags_keep.data()),
                thrust::raw_pointer_cast(pos_keep.data()),
                keep_count,
                thrust::raw_pointer_cast(d_keep.data()),
                threads
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

    // Device pointers for send/recv (CUDA-aware MPI)
    d_local = thrust::raw_pointer_cast(d_particles.data());

    Particle* d_recv_left  = (recv_left_count  > 0)
                             ? (d_local + keep_count)
                             : nullptr;
    Particle* d_recv_right = (recv_right_count > 0)
                             ? (d_local + keep_count + recv_left_count)
                             : nullptr;

    Particle* d_send_left_ptr  = (send_left_count  > 0)
                                 ? thrust::raw_pointer_cast(d_send_left.data())
                                 : nullptr;
    Particle* d_send_right_ptr = (send_right_count > 0)
                                 ? thrust::raw_pointer_cast(d_send_right.data())
                                 : nullptr;

    // Exchange particle data (device-to-device)
    MPI_Sendrecv(
        d_send_left_ptr,
        send_left_count * static_cast<int>(sizeof(Particle)),
        MPI_BYTE,
        left_rank, 20,
        d_recv_right,
        recv_right_count * static_cast<int>(sizeof(Particle)),
        MPI_BYTE,
        right_rank, 20,
        comm, &status
    );

    MPI_Sendrecv(
        d_send_right_ptr,
        send_right_count * static_cast<int>(sizeof(Particle)),
        MPI_BYTE,
        right_rank, 21,
        d_recv_left,
        recv_left_count * static_cast<int>(sizeof(Particle)),
        MPI_BYTE,
        left_rank, 21,
        comm, &status
    );

    // Update local count
    cfg_manager.config.n_local = n_new_local;
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
        double devide_p = cfg_manager.config.devide_p;
        int n_particles_global = cfg_manager.config.n_particles_global;
        int n_particles_type0 = cfg_manager.config.n_particles_type0;
        int n_particles_type1 = n_particles_global - n_particles_type0;
        double box_w = cfg_manager.config.box_w_global;
        double box_h = cfg_manager.config.box_h_global;

        double unshifted_x_devide = box_w * devide_p;
        double shift_dx = box_w * (0.5 - devide_p / 2.0);
        double T_init = cfg_manager.config.T_init;
        double mass_type0 = cfg_manager.config.MASS_A;
        double mass_type1 = cfg_manager.config.MASS_B;
        std::mt19937 rng(12345);

        double spacing_type0 = std::sqrt(unshifted_x_devide*box_h/n_particles_type0);
        double spacing_type1 = std::sqrt((box_w - unshifted_x_devide)*box_h/n_particles_type1);
        
        int n_rows_type0 = static_cast<int>(box_h / spacing_type0);
        int n_cols_type0 = static_cast<int>(unshifted_x_devide / spacing_type0);
        int n_rows_type1 = static_cast<int>(box_h / spacing_type1);
        int n_cols_type1 = static_cast<int>((box_w - unshifted_x_devide) / spacing_type1);

        while (n_rows_type0*n_cols_type0 < n_particles_type0){
            spacing_type0 *= 0.99;
            n_rows_type0 = static_cast<int>(box_h / spacing_type0);
            n_cols_type0 = static_cast<int>(unshifted_x_devide / spacing_type0);
        }
        while (n_rows_type0*n_cols_type0 < n_particles_type0){
            spacing_type1 *= 0.99;
            n_rows_type1 = static_cast<int>(box_h / spacing_type1);
            n_cols_type1 = static_cast<int>((box_w - unshifted_x_devide) / spacing_type1);
        }

        for (int j = 0; j < n_rows_type0; ++j){
            for (int i = 0; i < n_cols_type0; ++i){
                int idx = j*n_cols_type0 + i;
                if (idx >= n_particles_type0) break;
                double x = (i + 0.5)*spacing_type0;
                double y = (j + 0.5)*spacing_type0;
                h_particles[idx].pos.x = fmod(x + shift_dx, box_w);
                h_particles[idx].pos.y = y;
                h_particles[idx].type = 0;
            }
        }
        for (int j = 0; j < n_rows_type1; ++j){
            for (int i = 0; i < n_cols_type1; ++i){
                int idx = j*n_cols_type1 + i + n_particles_type0;
                if (idx >= n_particles_global) break;
                double x = (i + 0.5)*spacing_type1 + unshifted_x_devide;
                double y = (j + 0.5)*spacing_type1;
                h_particles[idx].pos.x = fmod(x + shift_dx, box_w);
                h_particles[idx].pos.y = y;
                h_particles[idx].type = 1;
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

    // Finish this later, cal forces and broadcast to host.
    // Then distribute and update halos.
}

void MDSimulation::plot_particles(const std::string& filename){
    print_particles(h_particles,
                            filename,
                            cfg_manager.config.box_w_global,
                            cfg_manager.config.box_h_global,
                            cfg_manager.config.SIGMA_AA,
                            cfg_manager.config.SIGMA_BB);
}