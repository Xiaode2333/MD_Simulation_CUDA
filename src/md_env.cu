#include "md_env.hpp"
#include "md_particle.hpp"
#include "md_config.hpp"
#include "md_common.hpp"
#include "md_cuda_common.hpp"

MDSimulation::MDSimulation(MDConfigManager config_manager, MPI_Comm comm){
    this->cfg_manager = config_manager;
    this->comm = comm;
    xi = 0.0;

    // fmt::print("Starting broadcasting params.\n");
    broadcast_params();
    // fmt::print("Params broadcasted.\n");

    allocate_memory();
    fmt::print("[Rank] {}/{}. Memory allocated.\n", cfg_manager.config.rank_idx, cfg_manager.config.rank_size);

    init_particles();//only in rank 0 host h_particles

    //Distribute from host of rank 0 to other ranks' devices
    fmt::print("[Rank] {}/{}. Distributing partilces.\n", cfg_manager.config.rank_idx, cfg_manager.config.rank_size);
    distribute_particles_h2d(); 
    fmt::print("[Rank] {}/{}. Update_halo.\n", cfg_manager.config.rank_idx, cfg_manager.config.rank_size);
    update_halo();

    //Update forces for all ranks, and then collect to host on rank 0
    fmt::print("[Rank] {}/{}. Updating forces.\n", cfg_manager.config.rank_idx, cfg_manager.config.rank_size);
    cal_forces();
    update_halo();//copy forces to halos
    fmt::print("[Rank] {}/{}. Collecting particles.\n", cfg_manager.config.rank_idx, cfg_manager.config.rank_size);
    collect_particles_d2h();

    fmt::print("[Rank] {}/{}. MD simulation env initialized.\n", cfg_manager.config.rank_idx, cfg_manager.config.rank_size);
}

MDSimulation::~MDSimulation() = default;

void MDSimulation::broadcast_params() {
    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    // rank 0 sanity and global-derived defaults
    if (world_rank == 0) {
        // Check consistency of rank_size from config (if provided)
        if (cfg_manager.config.rank_size != 0 &&
            cfg_manager.config.rank_size != world_size) {
            fmt::print(stderr,
                       "[broadcast_params] config.rank_size={} != world_size={}.\n",
                       cfg_manager.config.rank_size, world_size);
            MPI_Abort(comm, 1);
        }
        cfg_manager.config.rank_size = world_size;

        // Basic sanity for global box and particle count
        if (cfg_manager.config.box_w_global <= 0.0) {
            fmt::print(stderr,
                       "[broadcast_params] box_w_global must be > 0, got {}.\n",
                       cfg_manager.config.box_w_global);
            MPI_Abort(comm, 1);
        }
        if (cfg_manager.config.n_particles_global <= 0) {
            fmt::print(stderr,
                       "[broadcast_params] n_particles_global must be > 0, got {}.\n",
                       cfg_manager.config.n_particles_global);
            MPI_Abort(comm, 1);
        }

        // If capacity not specified, choose a reasonable default per rank
        if (cfg_manager.config.n_cap <= 0) {
            const double mean_per_rank =
                static_cast<double>(cfg_manager.config.n_particles_global) /
                static_cast<double>(world_size);
            const int n_cap_default =
                static_cast<int>(mean_per_rank * 1.5) + 128;
            cfg_manager.config.n_cap = n_cap_default;
        }

        // If halo capacities not specified, set to a fraction of n_cap with a floor
        if (cfg_manager.config.halo_left_cap <= 0) {
            int cap = cfg_manager.config.n_cap / 4;
            if (cap < 128) {
                cap = 128;
            }
            cfg_manager.config.halo_left_cap = cap;
        }
        if (cfg_manager.config.halo_right_cap <= 0) {
            int cap = cfg_manager.config.n_cap / 4;
            if (cap < 128) {
                cap = 128;
            }
            cfg_manager.config.halo_right_cap = cap;
        }

        // x_min, x_max, left_rank, right_rank are per-rank values
        // they will be recomputed consistently on all ranks below,
        // so the file values (if any) are ignored.
    }

    // broadcast the whole config struct from rank 0
    MPI_Bcast(&cfg_manager.config,
              static_cast<int>(sizeof(MDConfig)),
              MPI_BYTE,
              0,
              comm);

    // after broadcast, fix all rank-dependent fields locally
    cfg_manager.config.rank_idx  = world_rank;
    cfg_manager.config.rank_size = world_size;

    const double Lx = cfg_manager.config.box_w_global;
    const double dx = Lx / static_cast<double>(world_size);

    cfg_manager.config.x_min = dx * static_cast<double>(world_rank);
    cfg_manager.config.x_max = dx * static_cast<double>(world_rank + 1);

    cfg_manager.config.left_rank  =
        (world_rank + world_size - 1) % world_size;
    cfg_manager.config.right_rank =
        (world_rank + 1) % world_size;

    // Optional: debug print
    fmt::print("[broadcast_params] [Rank] {}/{}: x_min={:4.2f}, x_max={:4.2f}, left_rank={:1d}, right_rank={:1d}, n_cap={:4d}, haloL={:4d}, haloR={:4d}\n",
               world_rank,
               world_size,
               cfg_manager.config.x_min,
               cfg_manager.config.x_max,
               cfg_manager.config.left_rank,
               cfg_manager.config.right_rank,
               cfg_manager.config.n_cap,
               cfg_manager.config.halo_left_cap,
               cfg_manager.config.halo_right_cap);
}

void MDSimulation::allocate_memory(){
    h_particles.resize(cfg_manager.config.n_particles_global);
    h_particles_local.resize(cfg_manager.config.n_cap);
    h_particles_halo_left.resize(cfg_manager.config.halo_left_cap);
    h_particles_halo_right.resize(cfg_manager.config.halo_right_cap);
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
    h_send_left.reserve(cfg_manager.config.n_cap);
    h_send_right.reserve(cfg_manager.config.n_cap);
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

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void MDSimulation::plot_particles(const std::string& filename){
    if (cfg_manager.config.rank_idx == 0){
        print_particles(h_particles,
                            filename,
                            cfg_manager.config.box_w_global,
                            cfg_manager.config.box_h_global,
                            cfg_manager.config.SIGMA_AA,
                            cfg_manager.config.SIGMA_BB);
    }
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

    // fmt::print("[DEBUG] update_d_particles\n");
    // exchange particles and halos between ranks
    update_d_particles();   // may change cfg_manager.config.n_local
    // fmt::print("[DEBUG] update_halo\n");
    update_halo();          // rebuild halos after migration

    //  after exchange, n_local may have changed; recompute and check
    n_local = cfg_manager.config.n_local;
    if (n_local > n_cap) {
        fmt::print(stderr,
                   "[step_single_NVE] rank {} n_local={} exceeds n_cap={} after exchange.\n",
                   cfg_manager.config.rank_idx, n_local, n_cap);
        MPI_Abort(comm, 1);
    }

    // fmt::print("[DEBUG] cal_forces\n");
    // compute forces with updated locals + halos
    cal_forces();

    // second half-kick only if we have local particles
    if (n_local > 0) {
        int blocks = (n_local + threads - 1) / threads;
        // fmt::print("[DEBUG] step_2nd_half_vv_kernel\n");
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
    const double dt      = cfg_manager.config.dt;
    const int    threads = cfg_manager.config.THREADS_PER_BLOCK;
    const double Lx      = cfg_manager.config.box_w_global;
    const double Ly      = cfg_manager.config.box_h_global;

    int n_local = cfg_manager.config.n_local;
    int blocks  = (n_local + threads - 1) / threads;

    double K_global = cal_total_K();

    const double g_dof = 2*cfg_manager.config.n_particles_global - 2;
    const double T0    = cfg_manager.config.T_target;
    const double Q     = cfg_manager.config.Q;       // thermostat mass

    xi += 0.5 * dt * (2.0 * K_global - g_dof * T0) / Q;

    step_half_vv_nh_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_particles.data()),
        n_local, dt, xi, Lx, Ly
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    update_d_particles();
    update_halo();

    cal_forces();

    n_local = cfg_manager.config.n_local;//n_local may have changed after migration
    blocks  = (n_local + threads - 1) / threads;

    K_global = cal_total_K();

    xi += 0.5 * dt * (2.0 * K_global - g_dof * T0) / Q;

    step_2nd_half_vv_nh_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_particles.data()),
        n_local, dt, xi
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
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