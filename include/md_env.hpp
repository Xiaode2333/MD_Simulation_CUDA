#pragma once

#include "include/md_particle.hpp"
#include "include/md_config.hpp"
#include "include/md_common.hpp"
#include "include/md_cuda_common.hpp"

#include <string>
#include <mpi.h>
#include <vector>
#include <fmt/core.h>
#include <thrust/device_vector.h>
#include <algorithm>
#include <cmath>



class MDSimulation {
    public:
        MDSimulation(class MDConfigManager config_manager, MPI_Comm comm);

        ~MDSimulation();

        double cal_total_energy();
    private:
        MDConfigManager cfg_manager;
        MPI_Comm comm;

        std::vector<Particle> h_particles;
        thrust::device_vector<Particle> d_particles;
        thrust::device_vector<Particle> d_particles_halo_left;
        thrust::device_vector<Particle> d_particles_halo_right;

        void broadcast_params();
        void allocate_memory();
        void init_particles();
        void exchange_halo();
        double cal_local_energy();
        void distribute_particles_h2d();
        void collect_particles_d2h();
        void update_halo();
};