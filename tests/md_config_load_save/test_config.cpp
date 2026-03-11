// g++ ./tests/md_config_load_save/test_config.cpp \
    ./src/md_config.cpp \
    -o ./tests/md_config_load_save/test_config \
    -I${CONDA_PREFIX}/include -L${CONDA_PREFIX}/lib -Wl,-rpath,${CONDA_PREFIX}/lib \
    -lfmt
#include "md_config.hpp"
#include "mpi.h"

int main(){

    MPI_Init(nullptr, nullptr);
    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n_particles_global = 2048;
    int n_particles_type0 = 1024;
    double box_w_global = std::sqrt(n_particles_global/0.6*6); 
    double box_h_global = std::sqrt(n_particles_global/0.6/6); 
    double T_init = 0.5; 
    double T_target = 1.0;
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
    int rank_size = world_size;

    double mu = 1.0;
    double v0 = 0.0;
    double D_r = 0.0;
    double D_theta = 0.0;

    // This rank's prarams
    int rank_idx = 0;
    int n_local = 0;
    int n_halo_left = 0;
    int n_halo_right = 0;
    int n_cap = static_cast<int>(n_particles_global/world_size*2); //buffer capacity
    int halo_left_cap = static_cast<int>(n_particles_global/world_size/2);
    int halo_right_cap = static_cast<int>(n_particles_global/world_size/2);
    int left_rank = (world_rank + world_size - 1) % world_size;
    int right_rank = (world_rank + 1) % world_size;
    double x_min = box_w_global/world_size*world_rank;
    double x_max = box_w_global/world_size*(world_rank + 1);

    MDConfig cfg{};
    cfg.n_particles_global = n_particles_global;
    cfg.n_particles_type0 = n_particles_type0;
    cfg.box_w_global = box_w_global;
    cfg.box_h_global = box_h_global;
    cfg.T_init = T_init;
    cfg.T_target = T_target;
    cfg.SIGMA_AA = SIGMA_AA;
    cfg.SIGMA_BB = SIGMA_BB;
    cfg.SIGMA_AB = SIGMA_AB;
    cfg.EPSILON_AA = EPSILON_AA;
    cfg.EPSILON_BB = EPSILON_BB;
    cfg.EPSILON_AB = EPSILON_AB;
    cfg.MASS_A = MASS_A;
    cfg.MASS_B = MASS_B;
    cfg.devide_p = devide_p;
    cfg.dt = dt;
    cfg.Q = Q;
    cfg.save_dt_interval = save_dt_interval;
    cfg.cutoff = cutoff;
    cfg.run_name = run_name;
    cfg.load_name = load_name;
    cfg.THREADS_PER_BLOCK = THREADS_PER_BLOCK;
    cfg.rank_size = rank_size;
    cfg.mu = mu;
    cfg.v0 = v0;
    cfg.D_r = D_r;
    cfg.D_theta = D_theta;
    cfg.rank_idx = rank_idx;
    cfg.n_local = n_local;
    cfg.n_halo_left = n_halo_left;
    cfg.n_halo_right = n_halo_right;
    cfg.n_cap = n_cap;
    cfg.halo_left_cap = halo_left_cap;
    cfg.halo_right_cap = halo_right_cap;
    cfg.left_rank = left_rank;
    cfg.right_rank = right_rank;
    cfg.x_min = x_min;
    cfg.x_max = x_max;
    MDConfigManager cfg_mng(cfg);
    cfg_mng.config_to_json("./tests/md_config_load_save/test_config_output.json");
    MDConfigManager loaded_cfg_mng = MDConfigManager::config_from_json("./tests/md_config_load_save/test_config_output.json"); 

    loaded_cfg_mng.print_config();

    return 0;
}
