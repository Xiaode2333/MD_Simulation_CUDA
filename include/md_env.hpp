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
        void init_particles();// Only update h_particles on rank 0
        double cal_total_K();
        double cal_total_U();

    private:
        MDConfigManager cfg_manager;
        MPI_Comm comm;

        std::vector<Particle> h_particles; //typically all data is on device. When sampling first transfer them to host
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

        double xi; //xi for nose hoover

        void broadcast_params();
        void allocate_memory();
        double cal_local_energy();
        double compute_kinetic_energy_local();
        void distribute_particles_h2d();
        void collect_particles_d2h(); // collect particles from device to host, only to host of rank 0
        void update_halo();// suppose d_particles is already updated
        void update_d_particles(); // use only d_particles to update particles through particle exchange between ranks
        void cal_forces();// update force and store into d_particles
        void step_single_NVE();// step single timestep, including subdomain exchanges and halo update, but not collect to host. Assume acc updated, and update acc again after finish.
        void step_single_nose_hoover();
        double compute_U_energy_local();
};