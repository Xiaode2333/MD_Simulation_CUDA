#include "md_env.hpp"
#include <fmt/ranges.h>
#include <filesystem>
#include <cstdio>

int main(){
    const std::string cfg_path = "results/20251120_CWA_series/T_0.5_Q_100/config.json";
    const std::string saved_env_file = "results/20251120_CWA_series/T_0.5_Q_100/saved_env/saved_env.bin";
    const int save_step = 1999000;

    MDConfigManager cfg_mgr;
    cfg_mgr = cfg_mgr.config_from_json(cfg_path);

    MPI_Init(nullptr, nullptr);
    int rank_size = 0;
    int rank_idx = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_idx);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_size);

    if (cfg_mgr.config.rank_size != rank_size){
        fmt::print("[Error] rank size = {} doesn't match config.\n", rank_size);
        return -1;
    }

    cfg_mgr.config.rank_size = rank_size;
    cfg_mgr.config.rank_idx = rank_idx;

    {
        MDSimulation resumed_sim(cfg_mgr, MPI_COMM_WORLD, saved_env_file, save_step);

        for (int step; step < 300000; step++){
            resumed_sim.step_single_nose_hoover();//Refresh U buffer
            if (step % 1000 == 0){
                for (double sensitivity = 1.0; sensitivity <= 5.0; sensitivity += 0.1){
                    resumed_sim.sample_collect();
                    bool is_eq = resumed_sim.check_eqlibrium(sensitivity);
                    RankZeroPrint(rank_idx, "Sensitivity: {}, is_eq: {}.\n", sensitivity, is_eq);
                }
            }
        }
        
    }

    return 0;
}
