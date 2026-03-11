#include "md_env.hpp"

#include <cmath>
#include <filesystem>
#include <fmt/core.h>

int main() {
    const std::filesystem::path cfg_path = "./tests/nph_piston/config_piston.json";
    MDConfigManager cfg_mgr = MDConfigManager::config_from_json(cfg_path);

    MPI_Init(nullptr, nullptr);
    int rank_size = 0;
    int rank_idx = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_idx);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_size);

    if (cfg_mgr.config.rank_size != rank_size) {
        if (rank_idx == 0) {
            fmt::print(stderr,
                       "[nph_smoke_piston] rank size = {} doesn't match config.\n",
                       rank_size);
        }
        MPI_Finalize();
        return 1;
    }

    cfg_mgr.config.rank_size = rank_size;
    cfg_mgr.config.rank_idx = rank_idx;

    MDSimulation sim(cfg_mgr, MPI_COMM_WORLD);

    const int auto_steps = 100;
    const int production_steps = 200;
    double target_pressure = 0.0;

    for (int step = 0; step < auto_steps; ++step) {
        sim.step_single_nose_hoover();
        const double p_inst = sim.cal_instant_pressure();
        target_pressure += p_inst;
        if (!std::isfinite(p_inst)) {
            if (rank_idx == 0) {
                fmt::print(
                        stderr,
                        "[nph_smoke_piston] non-finite pressure during AUTO_TARGET at step {}\n",
                        step);
            }
            MPI_Finalize();
            return 2;
        }
    }
    target_pressure /= static_cast<double>(auto_steps);

    for (int step = 0; step < production_steps; ++step) {
        sim.step_single_NPH_piston(target_pressure);
        const double p_inst = sim.get_last_instant_pressure();
        if (!std::isfinite(p_inst) || !std::isfinite(sim.get_Lx()) ||
            !std::isfinite(sim.get_Ly()) || sim.get_Lx() <= 0.0 ||
            sim.get_Ly() <= 0.0 ||
            !std::isfinite(sim.get_nph_piston_velocity()) ||
            !std::isfinite(sim.get_nph_piston_force())) {
            if (rank_idx == 0) {
                fmt::print(
                        stderr,
                        "[nph_smoke_piston] invalid state in NPH piston at step {}: "
                        "Lx={}, Ly={}, P={}, dLy_dt={}, F_piston={}\n",
                        step, sim.get_Lx(), sim.get_Ly(), p_inst,
                        sim.get_nph_piston_velocity(), sim.get_nph_piston_force());
            }
            MPI_Finalize();
            return 3;
        }
    }

    if (rank_idx == 0) {
        fmt::print(
                "[nph_smoke_piston] completed. Lx={}, Ly={}, P_last={}, dLy_dt={}, F_piston={}\n",
                sim.get_Lx(), sim.get_Ly(), sim.get_last_instant_pressure(),
                sim.get_nph_piston_velocity(), sim.get_nph_piston_force());
    }

    MPI_Finalize();
    return 0;
}
