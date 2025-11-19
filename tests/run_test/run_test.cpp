#include "md_env.hpp"
#include <fmt/ranges.h>
#include <filesystem>
#include <cstdio>

template <typename... Args>
void RankZeroPrint(int rank_idx, fmt::format_string<Args...> format_str, Args&&... args) {
    if (rank_idx == 0) {
        fmt::print(format_str, std::forward<Args>(args)...);
        std::fflush(stdout);
    }
}

bool create_folder(const std::string& path, int rank_idx) {
    if (rank_idx != 0) {
        return true;
    }

    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        std::filesystem::create_directories(path, ec);
        if (ec) {
            fmt::print(stderr, "Failed to create dir {}. Error: {}\n", path, ec.message());
            return false;
        }
    }
    return true;
}

int main(){
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
    int n_grid_y  = static_cast<int>(cfg_mgr.config.box_h_global);

    std::string frame_dir = "./tests/run_test/frames/";
    std::string interface_dir = "./tests/run_test/interfaces/";
    std::string csv_dir = "./tests/run_test/csv/";

    if (!create_folder(frame_dir, rank_idx)) MPI_Abort(MPI_COMM_WORLD, 1);
    if (!create_folder(interface_dir, rank_idx)) MPI_Abort(MPI_COMM_WORLD, 1);
    if (!create_folder(csv_dir, rank_idx)) MPI_Abort(MPI_COMM_WORLD, 1);

    RankZeroPrint(rank_idx, "[RANK] {} waiting for MPI_Barrier.\n", rank_idx);
    MPI_Barrier(MPI_COMM_WORLD);
    RankZeroPrint(rank_idx, "[RANK] {} complete for MPI_Barrier.\n", rank_idx);
    
    for (int step = 0; step < n_steps; step++){
        sim.step_single_nose_hoover();

        if (step % n_record_interval == 0) {
            std::string frame_path = frame_dir + fmt::format("frame_step_{}.svg", step);
            std::string csv_path = csv_dir + fmt::format("frame_step_{}.csv", step);

            std::string frame_triangulation_path = frame_dir + fmt::format("triangulation_frame_step_{}.svg", step);
            std::string csv_path_triangulation = csv_dir + fmt::format("triangulation_frame_step_{}.csv", step);

            std::string frame_interface_path = interface_dir + fmt::format("interface_step_{}.svg", step);
            std::string csv_path_interface = csv_dir + fmt::format("interface_step_{}.csv", step);

            sim.sample_collect();
            
            int n_bins_per_rank = 16;
            std::vector<double> density_profile = sim.get_density_profile(n_bins_per_rank);
            RankZeroPrint(rank_idx, "[Step] {}. Density Profile (rho): {}\n", step, density_profile);

            RankZeroPrint(rank_idx, "[Step] {}. plot_particles.\n", step);
            sim.plot_particles(frame_path, csv_path);
            
            RankZeroPrint(rank_idx, "[Step] {}. triangulation_plot.\n", step);
            sim.triangulation_plot(true, frame_triangulation_path, csv_path_triangulation, density_profile);
            
            RankZeroPrint(rank_idx, "[Step] {}. plot_interfaces.\n", step);
            sim.plot_interfaces(frame_interface_path, csv_path_interface, density_profile);

            RankZeroPrint(rank_idx, "[Step] {}. Frames saved.\n", step);
        }

        if (step % 100 == 0) {
            RankZeroPrint(rank_idx, "[Step] {}.\n", step);
        }
    }
    return 0;
}