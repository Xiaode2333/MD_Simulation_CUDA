// md_config.hpp
#pragma once

#include <string>
#include <nlohmann/json.hpp>

// Use a type alias for convenience
using json = nlohmann::json;

struct MDConfig {
    int n_particles;
    int n_particles_type0;
    double box_w; 
    double box_h; 
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
    std::string run_name = "test";
    std::string load_name = ""; // if needed to load from file

    int mpi_world_size = 8;
    int THREADS_PER_BLOCK = 256;
};

class MDConfigManager {
    public:
    
        MDConfigManager(struct MDConfig config);

        ~MDConfigManager();

        struct MDConfig config;

        static MDConfigManager config_from_json(const std::string& filepath);

        void config_to_json(const std::string& filepath);

        void print_config();
    private:

};