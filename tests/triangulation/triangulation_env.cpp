// Lightweight end-to-end test of MDSimulation::triangulation_plot.
// Loads configuration from tests/triangulation/config.json and requests
// a triangulation plot (CSV + SVG) over the whole box.

#include "md_env.hpp"

#include <mpi.h>
#include <filesystem>
#include <string>
#include <fmt/core.h>

int main() {
    const std::string cfg_path = "./tests/triangulation/config.json";
    MDConfigManager cfg_mgr;
    cfg_mgr = cfg_mgr.config_from_json(cfg_path);

    MPI_Init(nullptr, nullptr);

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (cfg_mgr.config.rank_size != world_size) {
        fmt::print("[Error] rank size = {} doesn't match config ({}).\n",
                   world_size, cfg_mgr.config.rank_size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    cfg_mgr.config.rank_size = world_size;
    cfg_mgr.config.rank_idx  = world_rank;

    MDSimulation sim(cfg_mgr, MPI_COMM_WORLD);

    // Output locations
    const std::string frame_dir = "./tests/triangulation/frames/";
    const std::string csv_dir   = "./tests/triangulation/csv/";

    if (!create_folder(frame_dir, cfg_mgr.config.rank_idx)) MPI_Abort(MPI_COMM_WORLD, 1);
    if (!create_folder(csv_dir, cfg_mgr.config.rank_idx))   MPI_Abort(MPI_COMM_WORLD, 1);

    std::string frame_triangulation_path = frame_dir + "triangulation_env.svg";
    std::string csv_path_triangulation   = csv_dir + "triangulation_env.csv";

    // Run a short Nose-Hoover simulation and periodically generate
    // triangulation plots, similar in spirit to tests/run_test/run_test.cpp
    int n_record_interval = static_cast<int>(cfg_mgr.config.save_dt_interval / cfg_mgr.config.dt);
    if (n_record_interval <= 0) {
        n_record_interval = 1;
    }
    const int n_steps = 5000;

    for (int step = 0; step < n_steps; ++step) {
        sim.step_single_nose_hoover();

        if (step % n_record_interval == 0) {
            sim.triangulation_plot(
                true,
                frame_triangulation_path,
                csv_path_triangulation);
        }

        if (step % 500 == 0 && cfg_mgr.config.rank_idx == 0) {
            fmt::print("[triangulation_env] Step {}.\n", step);
        }
    }

    // Only rank 0 checks for output; other ranks exit quietly.
    if (cfg_mgr.config.rank_idx == 0) {
        namespace fs = std::filesystem;
        if (!fs::exists(csv_path_triangulation)) {
            fmt::print(stderr,
                       "[triangulation_env] Expected CSV not found at {}\n",
                       csv_path_triangulation);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }

    MPI_Finalize();
    return 0;
}
