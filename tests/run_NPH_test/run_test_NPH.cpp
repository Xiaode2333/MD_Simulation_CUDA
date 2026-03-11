#include "md_env.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
enum class Phase {
    AUTO_TARGET = 0,
    NPH_WARMUP = 1,
    NPH_PRODUCTION = 2,
};

const char *phase_name(Phase p) {
    switch (p) {
    case Phase::AUTO_TARGET:
        return "AUTO_TARGET";
    case Phase::NPH_WARMUP:
        return "NPH_WARMUP";
    case Phase::NPH_PRODUCTION:
        return "NPH_PRODUCTION";
    default:
        return "UNKNOWN";
    }
}

struct RollingStats {
    double mean = 0.0;
    double stddev = 0.0;
};

RollingStats compute_stats(const std::deque<double> &values) {
    RollingStats out{};
    if (values.empty()) {
        return out;
    }

    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    out.mean = sum / static_cast<double>(values.size());

    if (values.size() <= 1) {
        out.stddev = 0.0;
        return out;
    }

    double var_sum = 0.0;
    for (double value : values) {
        const double diff = value - out.mean;
        var_sum += diff * diff;
    }
    out.stddev = std::sqrt(var_sum / static_cast<double>(values.size() - 1));
    return out;
}

} // namespace

int main(int argc, char **argv) {
    const std::string cfg_path =
            (argc > 1) ? argv[1] : "./tests/run_NPH_test/config.json";

    MDConfigManager cfg_mgr = MDConfigManager::config_from_json(cfg_path);

    MPI_Init(nullptr, nullptr);
    int rank_size = 0;
    int rank_idx = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_idx);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_size);

    if (cfg_mgr.config.rank_size != rank_size) {
        if (rank_idx == 0) {
            fmt::print(stderr,
                       "[run_test_NPH] rank size = {} doesn't match config.\n",
                       rank_size);
        }
        MPI_Finalize();
        return 1;
    }

    cfg_mgr.config.rank_size = rank_size;
    cfg_mgr.config.rank_idx = rank_idx;
    const int n_record_interval =
            std::max(1, static_cast<int>(cfg_mgr.config.save_dt_interval /
                                         cfg_mgr.config.dt));
    const int eval_interval =
            std::max(1, cfg_mgr.config.pressure_eval_interval_steps);
    const int auto_target_steps =
            std::max(1, cfg_mgr.config.pressure_target_auto_steps);
    const int window_samples =
            std::max(2, cfg_mgr.config.pressure_window_samples);
    const int stable_required =
            std::max(1, cfg_mgr.config.pressure_stable_required_windows);
    const int warmup_min_steps =
            std::max(0, cfg_mgr.config.nph_warmup_min_steps);
    const int warmup_max_steps = std::max(
            warmup_min_steps + 1, cfg_mgr.config.nph_warmup_max_steps);
    const int nph_steps_target = std::max(1, cfg_mgr.config.nph_steps);
    const int snapshot_interval = std::max(1, 20 * n_record_interval);
    const int progress_log_interval_steps = 1000;

    const fs::path base_dir = "./tests/run_NPH_test";
    const fs::path sample_csv_dir = base_dir / "sample_csv";
    const fs::path frame_dir = base_dir / "frames";
    const fs::path frame_csv_dir = frame_dir / "csv";
    const fs::path saved_env_dir = base_dir / "saved_env";
    const fs::path saved_env_file = saved_env_dir / "saved_env.bin";
    const fs::path saved_cfg_path = saved_env_dir / "config.json";
    const fs::path pressure_csv = sample_csv_dir / "pressure_log.csv";
    const fs::path barostat_csv = sample_csv_dir / "barostat_state_log.csv";
    const fs::path thermo_csv = sample_csv_dir / "thermo_log.csv";

    if (!create_folder(base_dir, rank_idx))
        MPI_Abort(MPI_COMM_WORLD, 3);
    if (!create_folder(sample_csv_dir, rank_idx))
        MPI_Abort(MPI_COMM_WORLD, 3);
    if (!create_folder(frame_dir, rank_idx))
        MPI_Abort(MPI_COMM_WORLD, 3);
    if (!create_folder(frame_csv_dir, rank_idx))
        MPI_Abort(MPI_COMM_WORLD, 3);
    if (!create_folder(saved_env_dir, rank_idx))
        MPI_Abort(MPI_COMM_WORLD, 3);

    const std::string tag = "run_test_NPH";

    if (rank_idx == 0) {
        std::ofstream clear_pressure(pressure_csv, std::ios::out | std::ios::trunc);
        std::ofstream clear_barostat(barostat_csv,
                                     std::ios::out | std::ios::trunc);
        std::ofstream clear_thermo(thermo_csv, std::ios::out | std::ios::trunc);

        if (!clear_pressure || !clear_barostat || !clear_thermo) {
            fmt::print(stderr,
                       "[run_test_NPH] Failed to initialize output CSV files under {}\n",
                       sample_csv_dir.string());
            MPI_Abort(MPI_COMM_WORLD, 4);
        }

        append_csv(pressure_csv, rank_idx, tag,
                   "step,t,phase,P_inst,P_target,rolling_mean,rolling_std,rel_err,Lx,Ly,area,dA_dt\n");
        append_csv(barostat_csv, rank_idx, tag,
                   "step,t,phase,Lx,Ly,area,dA_dt,a_barostat,P_inst,P_target\n");
        append_csv(thermo_csv, rank_idx, tag,
                   "step,t,phase,U_tot,K_tot,P_inst,P_target,Lx,Ly,area\n");

        cfg_mgr.config_to_json(saved_cfg_path.string());
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MDSimulation sim(cfg_mgr, MPI_COMM_WORLD);

    Phase phase = Phase::AUTO_TARGET;
    if (cfg_mgr.config.pressure_target_mode != "auto_initial") {
        phase = Phase::NPH_WARMUP;
    }

    double P_target = cfg_mgr.config.P_target;
    int auto_steps_done = 0;
    int warmup_steps_done = 0;
    int nph_steps_done = 0;
    int stable_windows = 0;
    bool finished = false;

    std::deque<double> auto_samples;
    std::deque<double> pressure_window;

    const int hard_step_limit =
            auto_target_steps + warmup_max_steps + nph_steps_target + 50000;

    for (int step = 0; step < hard_step_limit && !finished; ++step) {
        const Phase phase_used = phase;

        double P_inst = 0.0;
        if (phase_used == Phase::AUTO_TARGET) {
            sim.step_single_nose_hoover();
            P_inst = sim.cal_instant_pressure();
        } else {
            sim.step_single_NPH(P_target);
            P_inst = sim.get_last_instant_pressure();
        }

        const bool do_record = (step % n_record_interval == 0);
        const bool do_progress_log = (step % progress_log_interval_steps == 0);
        const bool need_thermo = do_record || do_progress_log;

        double U_tot = 0.0;
        double K_tot = 0.0;
        if (do_record) {
            sim.sample_collect();
        }
        if (need_thermo) {
            U_tot = sim.cal_total_U();
            K_tot = sim.cal_total_K();
        }

        if (rank_idx == 0) {
            if (phase_used == Phase::AUTO_TARGET) {
                ++auto_steps_done;
            } else if (phase_used == Phase::NPH_WARMUP) {
                ++warmup_steps_done;
            } else {
                ++nph_steps_done;
            }

            if (step % eval_interval == 0) {
                if (phase_used == Phase::AUTO_TARGET) {
                    auto_samples.push_back(P_inst);
                } else {
                    pressure_window.push_back(P_inst);
                    while (static_cast<int>(pressure_window.size()) > window_samples) {
                        pressure_window.pop_front();
                    }
                }
            }

            if (phase_used == Phase::AUTO_TARGET &&
                auto_steps_done >= auto_target_steps && !auto_samples.empty()) {
                const double sum =
                        std::accumulate(auto_samples.begin(), auto_samples.end(), 0.0);
                P_target = sum / static_cast<double>(auto_samples.size());
                phase = Phase::NPH_WARMUP;
                pressure_window.clear();
                stable_windows = 0;
                fmt::print("[run_test_NPH] AUTO_TARGET complete at step {}. P_target={}\n",
                           step, P_target);
            }

            if (phase_used == Phase::NPH_WARMUP) {
                if (warmup_steps_done >= warmup_min_steps &&
                    static_cast<int>(pressure_window.size()) >= window_samples) {
                    const RollingStats stats = compute_stats(pressure_window);
                    const double denom = std::max(1e-12, std::abs(P_target));
                    const double rel_err = std::abs(stats.mean - P_target) / denom;
                    const double rel_std = stats.stddev / denom;

                    if (rel_err <= cfg_mgr.config.pressure_tolerance_rel &&
                        rel_std <= cfg_mgr.config.pressure_tolerance_rel) {
                        ++stable_windows;
                    } else {
                        stable_windows = 0;
                    }

                    if (stable_windows >= stable_required) {
                        phase = Phase::NPH_PRODUCTION;
                        fmt::print("[run_test_NPH] NPH warmup complete at step {}\n",
                                   step);
                    }
                }

                if (phase == Phase::NPH_WARMUP &&
                    warmup_steps_done >= warmup_max_steps) {
                    phase = Phase::NPH_PRODUCTION;
                    fmt::print("[run_test_NPH] Forced warmup->production switch at step {}\n",
                               step);
                }
            }

            const double area = sim.get_Lx() * sim.get_Ly();
            const double dA_dt =
                    (phase_used == Phase::AUTO_TARGET) ? 0.0 : sim.get_nph_area_rate();
            const double a_barostat =
                    (phase_used == Phase::AUTO_TARGET)
                            ? 0.0
                            : (P_inst - P_target) /
                                      std::max(cfg_mgr.config.barostat_mass, 1e-12);

            if (step % eval_interval == 0) {
                RollingStats stats{};
                if (!pressure_window.empty()) {
                    stats = compute_stats(pressure_window);
                }
                const double denom = std::max(1e-12, std::abs(P_target));
                const double rel_err = std::abs(stats.mean - P_target) / denom;

                append_csv(pressure_csv, rank_idx, tag,
                           "{},{:.6f},{},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e}\n",
                           step, sim.t, phase_name(phase_used), P_inst, P_target,
                           stats.mean, stats.stddev, rel_err, sim.get_Lx(),
                           sim.get_Ly(), area, dA_dt);

                append_csv(barostat_csv, rank_idx, tag,
                           "{},{:.6f},{},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e}\n",
                           step, sim.t, phase_name(phase_used), sim.get_Lx(),
                           sim.get_Ly(), area, dA_dt, a_barostat, P_inst, P_target);
            }

            if (do_progress_log) {
                fmt::print(
                        "[run_test_NPH] step={} t={:.6f} phase={} Lx={:.10e} Ly={:.10e} "
                        "dA_dt={:.10e} P_inst={:.10e} P_target={:.10e} U_tot={:.10e} K_tot={:.10e}\n",
                        step, sim.t, phase_name(phase_used), sim.get_Lx(), sim.get_Ly(),
                        dA_dt, P_inst, P_target, U_tot, K_tot);
            }

            if (do_record) {
                append_csv(thermo_csv, rank_idx, tag,
                           "{},{:.6f},{},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e}\n",
                           step, sim.t, phase_name(phase_used), U_tot, K_tot, P_inst,
                           P_target, sim.get_Lx(), sim.get_Ly(), area);

                if (step % snapshot_interval == 0) {
                    sim.save_env(saved_env_file.string(), step);
                    std::vector<Particle> frame_particles;
                    sim.get_host_particles(frame_particles);

                    const std::string step_str = std::to_string(step);
                    const fs::path frame_svg =
                            frame_dir / ("frame_nph_step_" + step_str + ".svg");
                    const fs::path frame_csv =
                            frame_csv_dir / ("frame_nph_step_" + step_str + ".csv");

                    try {
                        plot_particles_python(frame_particles, frame_svg.string(),
                                              frame_csv.string(), sim.get_Lx(),
                                              sim.get_Ly(), cfg_mgr.config.SIGMA_AA,
                                              cfg_mgr.config.SIGMA_BB);
                    } catch (const std::exception &e) {
                        fmt::print(stderr,
                                   "[run_test_NPH] Failed to plot frame at step {}: {}\n",
                                   step, e.what());
                        MPI_Abort(MPI_COMM_WORLD, 5);
                    }
                }
            }

            if (phase_used == Phase::NPH_PRODUCTION &&
                nph_steps_done >= nph_steps_target) {
                finished = true;
            }
        }

        int phase_int = static_cast<int>(phase);
        MPI_Bcast(&phase_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        phase = static_cast<Phase>(phase_int);
        MPI_Bcast(&P_target, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&finished, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    }

    if (rank_idx == 0) {
        fmt::print("[run_test_NPH] Finished. phase={}, Lx={}, Ly={}, P_last={}\n",
                   phase_name(phase), sim.get_Lx(), sim.get_Ly(),
                   sim.get_last_instant_pressure());
    }

    MPI_Finalize();
    return 0;
}
