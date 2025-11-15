#pragma once

#include "md_particle.hpp"
#include "md_config.hpp"
#include "md_common.hpp"
#include "md_cuda_common.hpp"

#include <string>
#include <mpi.h>
#include <vector>
#include <fmt/core.h>
#include <thrust/device_vector.h>
#include <algorithm>
#include <cmath>
#include <random>




class MDSimulation {
    public:
        MDSimulation(class MDConfigManager config_manager, MPI_Comm comm);

        ~MDSimulation();

        double cal_total_energy();
        void plot_particles(const std::string& filename);//Assume particles are prepared in h_particles on rank 0
        void init_particles();

    private:
        MDConfigManager cfg_manager;
        MPI_Comm comm;

        std::vector<Particle> h_particles;
        thrust::device_vector<Particle> d_particles;
        thrust::device_vector<Particle> d_particles_halo_left;
        thrust::device_vector<Particle> d_particles_halo_right;

        thrust::device_vector<int> flags_left;
        thrust::device_vector<int> flags_right;
        thrust::device_vector<int> flags_keep;
        thrust::device_vector<int> pos_left;
        thrust::device_vector<int> pos_right;
        thrust::device_vector<int> pos_keep;
        thrust::device_vector<Particle> d_send_left;
        thrust::device_vector<Particle> d_send_right;
        thrust::device_vector<Particle> d_keep;

        void broadcast_params();
        void allocate_memory();
        double cal_local_energy();
        void distribute_particles_h2d();
        void collect_particles_d2h(); // collect particles from device to host, only to host of rank 0
        void update_halo();// suppose d_particles is already updated
        void update_d_particles(); // use only d_particles to update particles through particle exchange between ranks
        
};