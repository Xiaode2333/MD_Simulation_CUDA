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
    

    int n_record_interval = static_cast<int>(cfg_mgr.config.save_dt_interval/cfg_mgr.config.dt);
    if (n_record_interval <= 0) {
        n_record_interval = 1;
    }
    const int n_steps = 2000000;
    const int last_save_step = ((n_steps - 1) / n_record_interval) * n_record_interval;
    
    int n_grid_y  = static_cast<int>(cfg_mgr.config.box_h_global);

    std::string frame_dir = "./tests/run_test/frames/";
    std::string interface_dir = "./tests/run_test/interfaces/";
    std::string csv_dir = "./tests/run_test/csv/";
    std::string sample_csv_dir = "./tests/run_test/sample_csv/";
    std::string saved_env_dir = "./tests/run_test/saved_env/";
    std::string saved_env_file = saved_env_dir + "saved_env.bin";
    std::string saved_cfg_path = saved_env_dir + "config.json";

    if (!create_folder(frame_dir, rank_idx)) MPI_Abort(MPI_COMM_WORLD, 1);
    if (!create_folder(interface_dir, rank_idx)) MPI_Abort(MPI_COMM_WORLD, 1);
    if (!create_folder(csv_dir, rank_idx)) MPI_Abort(MPI_COMM_WORLD, 1);
    if (!create_folder(sample_csv_dir, rank_idx)) MPI_Abort(MPI_COMM_WORLD, 1);
    if (!create_folder(saved_env_dir, rank_idx)) MPI_Abort(MPI_COMM_WORLD, 1);

    if (rank_idx == 0) {
        try {
            std::filesystem::copy_file(cfg_path, saved_cfg_path, std::filesystem::copy_options::overwrite_existing);
        } catch (const std::filesystem::filesystem_error& e) {
            fmt::print(stderr, "[Error] Failed to copy {} to {}: {}\n", cfg_path, saved_cfg_path, e.what());
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
    }

    std::string U_tot_csv_path = sample_csv_dir + "U_tot_log.csv";

    {
        MDSimulation sim(cfg_mgr, MPI_COMM_WORLD);

        RankZeroPrint(rank_idx, "[RANK] {} waiting for MPI_Barrier.\n", rank_idx);
        MPI_Barrier(MPI_COMM_WORLD);
        RankZeroPrint(rank_idx, "[RANK] {} complete for MPI_Barrier.\n", rank_idx);
        
        for (int step = 0; step < n_steps; step++){
            sim.step_single_nose_hoover();

            if (step % n_record_interval == 0) {
                sim.sample_collect();

                bool is_equilibrated = sim.check_eqlibrium(1.0);
                if (is_equilibrated){
                    RankZeroPrint(rank_idx, "Equilibrated. Exiting.\n");
                    break;
                }
                
                double U_tot = sim.cal_total_U();

                sim.RankZeroPrint("U_tot = {:.4e}\n", U_tot);
                sim.write_to_file(U_tot_csv_path, "U_tot, {}, step, {}\n", U_tot, step);

                sim.save_env(saved_env_file, step);

                // std::string frame_path = frame_dir + fmt::format("frame_step_{}.svg", step);
                // std::string csv_path = csv_dir + fmt::format("frame_step_{}.csv", step);

                std::string frame_triangulation_path = frame_dir + fmt::format("triangulation_frame_step_{}.svg", step);
                std::string csv_path_triangulation = csv_dir + fmt::format("triangulation_frame_step_{}.csv", step);

                // std::string frame_interface_path = interface_dir + fmt::format("interface_step_{}.svg", step);
                // std::string csv_path_interface = csv_dir + fmt::format("interface_step_{}.csv", step);

                int n_bins_per_rank = 32;
                std::vector<double> density_profile = sim.get_density_profile(n_bins_per_rank);
                // RankZeroPrint(rank_idx, "[Step] {}. Density Profile (rho): {}\n", step, density_profile);

                // RankZeroPrint(rank_idx, "[Step] {}. plot_particles.\n", step);
                // sim.plot_particles(frame_path, csv_path);
                
                RankZeroPrint(rank_idx, "[Step] {}. triangulation_plot.\n", step);
                sim.triangulation_plot(true, frame_triangulation_path, csv_path_triangulation, density_profile);
                
                // RankZeroPrint(rank_idx, "[Step] {}. plot_interfaces.\n", step);
                // sim.plot_interfaces(frame_interface_path, csv_path_interface, density_profile);

                // RankZeroPrint(rank_idx, "[Step] {}. Frames saved.\n", step);
            }

            if (step % 100 == 0) {
                RankZeroPrint(rank_idx, "[Step] {}.\n", step);
            }
        }
    }

    // MPI_Barrier(MPI_COMM_WORLD);

    // RankZeroPrint(rank_idx, "[RANK] {} starting restored simulation via saved state.\n", rank_idx);

    // const int save_step = last_save_step;

    // MDConfigManager resumed_cfg = MDConfigManager::config_from_json(saved_cfg_path);
    // resumed_cfg.config.rank_size = rank_size;
    // resumed_cfg.config.rank_idx = rank_idx;
    // {
    //     MDSimulation resumed_sim(resumed_cfg, MPI_COMM_WORLD, saved_env_file, save_step);
    //     const int resume_steps = n_record_interval;
    //     for (int offset = 1; offset <= resume_steps; ++offset) {
    //         int resumed_step = save_step + offset;
    //         resumed_sim.step_single_nose_hoover();

    //         if (resumed_step % n_record_interval == 0) {
    //             resumed_sim.sample_collect();
    //             double resumed_U_tot = resumed_sim.cal_total_U();

    //             resumed_sim.RankZeroPrint("U_tot = {:.4e}\n", resumed_U_tot);
    //             resumed_sim.write_to_file(U_tot_csv_path, "U_tot, {}, step, {}\n", resumed_U_tot, resumed_step);
    //         }

    //         if (resumed_step % 100 == 0) {
    //             RankZeroPrint(rank_idx, "[Step] {} (restored).\n", resumed_step);
    //         }
    //     }
    // }

    return 0;
}
