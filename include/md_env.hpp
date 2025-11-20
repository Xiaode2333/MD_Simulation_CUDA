#pragma once

#include "md_particle.hpp"
#include "md_config.hpp"
#include "md_common.hpp"
#include "md_cuda_common.hpp"

#include <string>
#include <fstream>
#include <mpi.h>
#include <vector>
#include <deque>
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
        MDSimulation(MDConfigManager config_manager, MPI_Comm comm, const std::string& filename, int step); // constructor from saved file
        ~MDSimulation();

        double t = 0.0; // evolution time

        void save_env(const std::string& filename, const int step);
        
        void plot_particles(const std::string& filename, const std::string& csv_path);
        
        double cal_total_K();
        double cal_total_U();
        double deform(double epsilon, double U_old);
        double get_Lx() const { return cfg_manager.config.box_w_global; }
        double get_Ly() const { return cfg_manager.config.box_h_global; }
        
        void step_single_NVE();
        void step_single_nose_hoover();

        bool check_eqlibrium(double sensitivity);
        
        void sample_collect();

        std::optional<delaunator::Delaunator> triangulation_plot(bool is_plot, const std::string& filename, const std::string& csv_path, const std::vector<double>& rho);
        
        std::vector<std::vector<double>> locate_interface(const delaunator::Delaunator& d);
        // Returns interface polylines as {x0, y0, x1, y1, ...} for each interface, empty if rank != 0 or none found
        std::vector<std::vector<double>> get_smooth_interface(int n_grid_y, double smoothing_sigma);
        void do_CWA_instant(int q_min, int q_max, const std::string& csv_path, const std::string& plot_path, bool is_plot, int step);

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

        template <typename... Args>
        bool write_to_file(const std::string& filename, fmt::format_string<Args...> format_str, Args&&... args) {
            if (cfg_manager.config.rank_idx != 0) {
                return true;
            }
            std::ofstream out(filename, std::ios::out | std::ios::app);
            if (!out) {
                fmt::print(stderr, "[Error] Failed to open {} for writing.\n", filename);
                return false;
            }
            out << fmt::format(format_str, std::forward<Args>(args)...);
            out.flush();
            return true;
        }

        
        
    private:
        MDConfigManager cfg_manager;
        MPI_Comm comm;

        std::unique_ptr<FileWriter> particle_writer;
        std::unique_ptr<FileReader> particle_reader;

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
        double Lx0;
        double Ly0;

        double record_interval_dt;
        double next_record_time;
        std::deque<double> energy_history;
        std::size_t energy_window_sample_count;
        std::size_t energy_history_capacity;
        int equilibrium_window_streak;

        static constexpr double kEquilibriumWindowTime = 100.0;
        static constexpr int kBaseRequiredPasses = 3;
        static constexpr double kBasePValue = 0.2;

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
        std::vector<std::vector<double>> compute_interface_paths(int n_grid_y, double smoothing_sigma);

        void init_equilibrium_tracker();
        void append_energy_sample(double U);
        bool evaluate_equilibrium(double normalized_sensitivity);
        double compute_window_relative_change(std::size_t start_idx) const;
        struct WindowStats {
            double mean;
            double variance;
        };
        WindowStats compute_window_stats(std::size_t start_idx) const;
        double normal_tail_probability(double z) const;
};
