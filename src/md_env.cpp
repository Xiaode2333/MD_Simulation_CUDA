#include "../include/md_env.hpp"

MDSimulation::MDSimulation(class MDConfigManager config_manager, MPI_Comm comm){
    this->cfg_manager = config_manager;
    this->comm = comm;
    broadcast_params();
    allocate_memory();
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
}

void MDSimulation::init_particles(){
    double p = cfg_manager.config.devide_p;
    double T_init = cfg_manager.config.T_init;
    double MASS_A = cfg_manager.config.MASS_A;
    double MASS_B = cfg_manager.config.MASS_B;

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
    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    const int rank_size = cfg_manager.config.rank_size;
    const int rank_idx  = cfg_manager.config.rank_idx;

    if (world_size != rank_size) {
        fmt::print(stderr,
                   "[distribute_particles_h2d] world_size={} != rank_size={} in config.\n",
                   world_size, rank_size);
        MPI_Abort(comm, 1);
    }

    const double Lx = cfg_manager.config.box_w_global;
    const int    N_global = cfg_manager.config.n_particles_global;

    if (cfg_manager.config.n_cap <= 0) {
        fmt::print(stderr,
                   "[distribute_particles_h2d] n_cap <= 0 on rank {}.\n",
                   world_rank);
        MPI_Abort(comm, 1);
    }

    // Rank 0: partition global h_particles into rank_size buckets by x and send
    if (world_rank == 0) {
        if (static_cast<int>(h_particles.size()) < N_global) {
            fmt::print(stderr,
                       "[distribute_particles_h2d] h_particles size={} < n_particles_global={}.\n",
                       h_particles.size(), N_global);
            MPI_Abort(comm, 1);
        }

        std::vector<std::vector<Particle>> buckets(rank_size);
        buckets.assign(rank_size, std::vector<Particle>());

        const double inv_dx = static_cast<double>(rank_size) / Lx;

        for (int i = 0; i < N_global; ++i) {
            Particle p = h_particles[i];
            double x   = p.pos.x;
            x = pbc_wrap(x, Lx);

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
            if (count > cfg_manager.config.n_cap) {
                fmt::print(stderr,
                           "[distribute_particles_h2d] bucket {} has {} particles, exceeds n_cap={}.\n",
                           r, count, cfg_manager.config.n_cap);
                MPI_Abort(comm, 1);
            }

            if (r == 0) {
                cfg_manager.config.n_local = count;
            } else {
                MPI_Send(&count, 1, MPI_INT, r, 100, comm);
            }
        }

        // Then send actual particle data to other ranks
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
                           "[distribute_particles_h2d] rank0 d_particles size={} < n_local={}.\n",
                           d_particles.size(), count0);
                MPI_Abort(comm, 1);
            }
            thrust::copy_n(buckets[0].begin(), count0, d_particles.begin());
        }

    } else {
        // Other ranks: receive count then particles from rank 0
        int local_count = 0;
        MPI_Recv(&local_count, 1, MPI_INT, 0, 100, comm, MPI_STATUS_IGNORE);

        if (local_count > cfg_manager.config.n_cap) {
            fmt::print(stderr,
                       "[distribute_particles_h2d] rank {} received {} particles, exceeds n_cap={}.\n",
                       world_rank, local_count, cfg_manager.config.n_cap);
            MPI_Abort(comm, 1);
        }

        cfg_manager.config.n_local = local_count;

        h_particles.resize(local_count);
        if (local_count > 0) {
            MPI_Recv(h_particles.data(),
                     local_count * static_cast<int>(sizeof(Particle)),
                     MPI_BYTE,
                     0,
                     101,
                     comm,
                     MPI_STATUS_IGNORE);

            if (static_cast<int>(d_particles.size()) < local_count) {
                fmt::print(stderr,
                           "[distribute_particles_h2d] rank {} d_particles size={} < n_local={}.\n",
                           world_rank, d_particles.size(), local_count);
                MPI_Abort(comm, 1);
            }
            thrust::copy_n(h_particles.begin(), local_count, d_particles.begin());
        }
    }

    // After local particles are distributed to device, build and exchange halos
    update_halo();
}


