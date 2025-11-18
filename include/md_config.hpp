// md_config.hpp
#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <fstream>
#include <stdexcept>
#include <fmt/core.h>

// Use a type alias for convenience
using json = nlohmann::json;

struct MDConfig {
    // Global params for all ranks
    int n_particles_global;
    int n_particles_type0;
    double box_w_global; 
    double box_h_global; 
    double T_init; 
    double T_target;
    double SIGMA_AA = 1.0; 
    double SIGMA_BB = 1.0; 
    double SIGMA_AB = 1.0;
    double EPSILON_AA = 1.0; 
    double EPSILON_BB = 1.0; 
    double EPSILON_AB = 0.25;
    double MASS_A = 1.0;
    double MASS_B = 1.0;
    double devide_p = 0.5;
    double dt = 1e-3;
    double Q = 100.0;
    double save_dt_interval = 0.1;
    double cutoff = 2.5;
    std::string run_name = "test";
    std::string load_name = ""; // if needed to load from file
    int THREADS_PER_BLOCK = 256;
    int rank_size = 8;

    // This rank's prarams
    int rank_idx = 0;
    int n_local;
    int n_halo_left;
    int n_halo_right;
    int n_cap; //buffer capacity
    int halo_left_cap;
    int halo_right_cap;
    int left_rank;
    int right_rank;
    double x_min;
    double x_max;
    
};

class MDConfigManager {
    public:
    
        MDConfigManager() = default;
        
        MDConfigManager(MDConfig config);

        ~MDConfigManager();

        struct MDConfig config;

        static MDConfigManager config_from_json(const std::string& filepath);

        void config_to_json(const std::string& filepath);

        void print_config();
    private:

};