#include "md_env.hpp"

#include <cstdio>
#include <algorithm>
#include <filesystem>
#include <fmt/ranges.h>
#include <vector>

int main() {
    const std::filesystem::path cfg_path = "./tests/test_ABP/config.json";
    MDConfigManager cfg_mgr = MDConfigManager::config_from_json(cfg_path);

    MPI_Init(nullptr, nullptr);
    int rank_size = 0;
    int rank_idx = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_idx);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_size);

    if (cfg_mgr.config.rank_size != rank_size) {
        fmt::print("[Error] rank size = {} doesn't match config.\n", rank_size);
        MPI_Finalize();
        return -1;
    }

    cfg_mgr.config.rank_size = rank_size;
    cfg_mgr.config.rank_idx = rank_idx;

    int n_record_interval =
            static_cast<int>(cfg_mgr.config.save_dt_interval / cfg_mgr.config.dt);
    if (n_record_interval <= 0) {
        n_record_interval = 1;
    }

    // Keep runtime modest for a test: ~2 * save_dt_interval worth of steps.
    const int n_steps = 100000;
    const int n_density_bins =
            std::max(10, static_cast<int>(cfg_mgr.config.box_w_global / 2.0));

    std::string frame_dir = "./tests/test_ABP/frames/";
    std::string interface_dir = "./tests/test_ABP/interfaces/";
    std::string csv_dir = "./tests/test_ABP/csv/";
    std::string sample_csv_dir = "./tests/test_ABP/sample_csv/";
    std::string saved_env_dir = "./tests/test_ABP/saved_env/";
    std::string saved_env_file = saved_env_dir + "saved_env.bin";
    std::string saved_cfg_path = saved_env_dir + "config.json";

    if (!create_folder(frame_dir, rank_idx))
        MPI_Abort(MPI_COMM_WORLD, 1);
    if (!create_folder(interface_dir, rank_idx))
        MPI_Abort(MPI_COMM_WORLD, 1);
    if (!create_folder(csv_dir, rank_idx))
        MPI_Abort(MPI_COMM_WORLD, 1);
    if (!create_folder(sample_csv_dir, rank_idx))
        MPI_Abort(MPI_COMM_WORLD, 1);
    if (!create_folder(saved_env_dir, rank_idx))
        MPI_Abort(MPI_COMM_WORLD, 1);

    if (rank_idx == 0) {
        try {
            std::filesystem::copy_file(
                    cfg_path, saved_cfg_path,
                    std::filesystem::copy_options::overwrite_existing);
        } catch (const std::filesystem::filesystem_error &e) {
            fmt::print(stderr, "[Error] Failed to copy {} to {}: {}\n",
                       cfg_path.string(), saved_cfg_path, e.what());
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
    }

    std::string U_tot_csv_path = sample_csv_dir + "U_tot_log.csv";

    {
        MDSimulation sim(cfg_mgr, MPI_COMM_WORLD);

        RankZeroPrint(rank_idx, "[RANK] {} waiting for MPI_Barrier.\n", rank_idx);
        MPI_Barrier(MPI_COMM_WORLD);
        RankZeroPrint(rank_idx, "[RANK] {} complete for MPI_Barrier.\n", rank_idx);

        for (int step = 0; step < n_steps; step++) {
            // ABP overdamped dynamics with self-propulsion
            sim.step_single_ABP();

            if (step % n_record_interval == 0) {
                sim.sample_collect();

                double U_tot = sim.cal_total_U();

                sim.RankZeroPrint("[ABP] t = {:.4e}, U_tot = {:.4e}\n", sim.t, U_tot);
                sim.write_to_file(U_tot_csv_path,
                                  "U_tot, {}, step, {}, t, {:.6f}\n", U_tot, step, sim.t);

                sim.save_env(saved_env_file, step);

                const std::string triangulation_svg =
                        frame_dir + fmt::format("triangulation_frame_step_{}.svg", step);
                const std::string triangulation_csv =
                        csv_dir + fmt::format("triangulation_frame_step_{}.csv", step);

                const std::string interface_svg =
                        interface_dir + fmt::format("interface_step_{}.svg", step);
                const std::string interface_csv =
                        csv_dir + fmt::format("interface_step_{}.csv", step);

                const std::string density_csv =
                        sample_csv_dir + fmt::format("density_profile_step_{}.csv", step);

                // Triangulation + interface plots need host particles
                RankZeroPrint(rank_idx, "[Step] {}. triangulation_plot.\n", step);
                sim.triangulation_plot(true, triangulation_svg, triangulation_csv);

                const std::vector<double> density_profile =
                        sim.get_density_profile(n_density_bins);
                write_density_profile_csv(density_csv, density_profile, rank_idx,
                                          "ABP");

                RankZeroPrint(rank_idx, "[Step] {}. plot_interfaces.\n", step);
                sim.plot_interfaces(interface_svg, interface_csv, density_profile);

                RankZeroPrint(rank_idx, "[Step] {}. Snapshots saved.\n", step);
            }

            if (step % std::max(1, n_record_interval / 5) == 0) {
                RankZeroPrint(rank_idx, "[Step] {}.\n", step);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
