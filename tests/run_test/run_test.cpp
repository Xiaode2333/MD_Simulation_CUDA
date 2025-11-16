#include "md_env.hpp"

int main(){
    // The following configurations will be auto calculated in init
    // "rank_idx": 0,
    // "right_rank": 0,
    // "left_rank": 0,
    // "x_max": 0,
    // "x_min": 0.0,
    // "n_cap": 0,
    // "halo_left_cap": 0,
    // "halo_right_cap": 0

    const std::string cfg_path = "./tests/run_test/config.json";
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
    

    MDSimulation sim(cfg_mgr, MPI_COMM_WORLD);

    const int n_steps = 1000000;
    const int n_record_interval = static_cast<int>(cfg_mgr.config.save_dt_interval/cfg_mgr.config.dt);
    std::string frame_dir = "./tests/run_test/frames/";
    
    std::error_code ec;
    if (rank_idx == 0) {
        if (!std::filesystem::exists(frame_dir, ec)) {
            std::filesystem::create_directories(frame_dir, ec);
            if (ec) {
                fmt::print(stderr,
                           "Failed to create dir {}. Error: {}\n",
                           frame_dir, ec.message());
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    // Make sure all ranks wait until rank 0 finishes creating the directory
    MPI_Barrier(MPI_COMM_WORLD);
    

    for (int step = 0; step < n_steps; step++){
        // sim.step_single_NVE();
        sim.step_single_nose_hoover();

        if (step % n_record_interval == 0) {
            std::string frame_path = frame_dir + fmt::format("frame_step_{}.svg", step);
            // before sampling or plot collect all particles to h_particles on rank 0
            sim.sample_collect();
            if (cfg_mgr.config.rank_idx == 0){
                fmt::print("[Step] {}. Plotting frame.\n", step);
                sim.plot_particles(frame_path);
                fmt::print("[Step] {}. Frame saved at {}.\n", step, frame_path);
            }
        }

        // if (cfg_mgr.config.rank_idx == 0){
        //     fmt::print("[Step] {}.\n", step);
        // }
        

    }
    return 0;
}