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
#include <delaunator-header-only.hpp>
#include <optional>



class MDSimulation {
    public:
        MDSimulation(class MDConfigManager config_manager, MPI_Comm comm);
        ~MDSimulation();

        void save_env();
        static MDSimulation load_env();
        
        void plot_particles(const std::string& filename, const std::string& csv_path);
        
        double cal_total_K();
        double cal_total_U();
        
        void step_single_NVE();
        void step_single_nose_hoover();
        
        void sample_collect();

        std::optional<delaunator::Delaunator> triangulation_plot(bool is_plot, const std::string& filename, const std::string& csv_path, const std::vector<double>& rho);
        
        std::vector<std::vector<double>> locate_interface(const delaunator::Delaunator& d);

        void plot_interfaces(const std::string& filename, const std::string& csv_path, const std::vector<double>& rho);
        
        std::vector<int> get_N_profile(int n_bins_per_rank); 
        std::vector<double> get_density_profile(int n_bins_per_rank);

        template <typename... Args>
        void RankZeroPrint(fmt::format_string<Args...> format_str, Args&&... args) {
            if (cfg_manager.config.rank_idx == 0) {
                fmt::print(format_str, std::forward<Args>(args)...);
                std::fflush(stdout);
            }
        }
        
    private:
        MDConfigManager cfg_manager;
        MPI_Comm comm;

        std::vector<double> coords; //For triangulation
        std::vector<int> vertex_to_idx;

        std::vector<Particle> h_particles; //typically all data is on device. When sampling first transfer them to host
        std::vector<Particle> h_particles_local;
        std::vector<Particle> h_particles_halo_left;
        std::vector<Particle> h_particles_halo_right;
        std::vector<Particle> h_send_left;
        std::vector<Particle> h_send_right;
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
        void init_particles();// Only update h_particles on rank 0
        double compute_kinetic_energy_local();
        void distribute_particles_h2d();
        void collect_particles_d2h(); // collect particles from device to host, only to host of rank 0
        void update_halo();// suppose d_particles is already updated
        void update_d_particles(); // use only d_particles to update particles through particle exchange between ranks
        void cal_forces();// update force and store into d_particles
        double compute_U_energy_local();
};
