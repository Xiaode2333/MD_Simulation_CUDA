#include "md_common.hpp"
#include "md_env.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
struct ProgramOptions {
    std::string base_dir;
    std::string ori_config;
    std::vector<MDConfigOverride> overrides;
};

void print_usage(const char *prog_name) {
    fmt::print(
            "Usage: {} --base-dir <output_dir> --ori-config <config.json> "
            "[--D<Param>=<value> ...]\n",
            prog_name);
}

std::string consume_value(const std::string &arg, int argc, char **argv,
                          int &idx) {
    const auto equal_pos = arg.find('=');
    if (equal_pos != std::string::npos) {
        return arg.substr(equal_pos + 1);
    }

    if (idx + 1 >= argc) {
        throw std::runtime_error(fmt::format("Missing value for '{}'", arg));
    }

    ++idx;
    return argv[idx];
}

ProgramOptions parse_args(int argc, char **argv) {
    ProgramOptions opts;

    for (int idx = 1; idx < argc; ++idx) {
        std::string arg = argv[idx];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg.rfind("--base-dir", 0) == 0) {
            opts.base_dir = consume_value(arg, argc, argv, idx);
        } else if (arg.rfind("--ori-config", 0) == 0) {
            opts.ori_config = consume_value(arg, argc, argv, idx);
        } else {
            bool is_override = false;
            if (arg.rfind("--D", 0) == 0 || arg.rfind("D", 0) == 0) {
                is_override = true;
            } else if (arg.size() > 2 && arg.rfind("--", 0) == 0) {
                throw std::runtime_error(fmt::format("Unknown option '{}'", arg));
            }

            if (!is_override) {
                throw std::runtime_error(fmt::format("Unknown argument '{}'", arg));
            }

            std::string key;
            std::string value;
            if (!MDConfigManager::parse_override_argument(arg, key, value)) {
                throw std::runtime_error(
                        fmt::format("Failed to parse override '{}'", arg));
            }
            opts.overrides.push_back(MDConfigOverride{key, value});
        }
    }

    if (opts.base_dir.empty()) {
        throw std::runtime_error("--base-dir must be provided");
    }
    if (opts.ori_config.empty()) {
        throw std::runtime_error("--ori-config must be provided");
    }

    return opts;
}

enum class Phase {
    NVT_100 = 0,
    NVT_500 = 1,
};

const char *phase_name(const Phase phase) {
    switch (phase) {
    case Phase::NVT_100:
        return "NVT_100";
    case Phase::NVT_500:
        return "NVT_500";
    default:
        return "UNKNOWN";
    }
}

const char *type_to_xyz_label(const int type) {
    if (type == 0) {
        return "A";
    }
    if (type == 1) {
        return "B";
    }
    return "X";
}

bool append_xyz_snapshot(const fs::path &xyz_path,
                         const std::vector<Particle> &particles, Phase phase,
                         int global_step, int phase_step, double global_time,
                         double phase_time, double Lx, double Ly) {
    std::ofstream out(xyz_path, std::ios::out | std::ios::app);
    if (!out) {
        return false;
    }

    out.setf(std::ios::fixed, std::ios::floatfield);
    out << std::setprecision(10);

    out << particles.size() << "\n";
    out << "phase=" << phase_name(phase) << " global_step=" << global_step
        << " phase_step=" << phase_step << " global_time=" << global_time
        << " phase_time=" << phase_time << " Lx=" << Lx << " Ly=" << Ly
        << "\n";

    for (const auto &p : particles) {
        out << type_to_xyz_label(p.type) << " " << p.pos.x << " " << p.pos.y
            << " 0.0\n";
    }

    out.flush();
    return static_cast<bool>(out);
}
} // namespace

int main(int argc, char **argv) {
    constexpr double kPhase100Dt = 1.0e-3;
    constexpr double kPhase500Dt = 1.0e-4;
    constexpr int kPhase100Steps = 100000;
    constexpr int kPhase500Steps = 5000000;
    constexpr double kSnapshotDt = 1.0e-1;
    constexpr int kPhase100Plots = 20;
    constexpr int kPhase500Plots = 20;

    ProgramOptions options;
    try {
        options = parse_args(argc, argv);
    } catch (const std::exception &ex) {
        fmt::print(stderr, "[run_series_NVT_batch_xyz_saving_piston] {}\n",
                   ex.what());
        return 1;
    }

    MDConfigManager cfg_base =
            MDConfigManager::config_from_json(options.ori_config);
    if (!options.overrides.empty()) {
        cfg_base.apply_overrides(options.overrides);
    }

    MPI_Init(nullptr, nullptr);

    int rank_size = 0;
    int rank_idx = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_idx);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_size);

    if (cfg_base.config.rank_size != rank_size) {
        fmt::print(
                stderr,
                "[run_series_NVT_batch_xyz_saving_piston] rank size = {} doesn't match config.\n",
                rank_size);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    cfg_base.config.rank_size = rank_size;
    cfg_base.config.rank_idx = rank_idx;

    MDConfigManager cfg_phase100 = cfg_base;
    cfg_phase100.config.dt = kPhase100Dt;
    cfg_phase100.config.save_dt_interval = kSnapshotDt;
    cfg_phase100.config.rank_size = rank_size;
    cfg_phase100.config.rank_idx = rank_idx;

    MDConfigManager cfg_phase500 = cfg_base;
    cfg_phase500.config.dt = kPhase500Dt;
    cfg_phase500.config.save_dt_interval = kSnapshotDt;
    cfg_phase500.config.rank_size = rank_size;
    cfg_phase500.config.rank_idx = rank_idx;

    const int phase100_xyz_interval = std::max(
            1, static_cast<int>(std::llround(kSnapshotDt / kPhase100Dt)));
    const int phase500_xyz_interval = std::max(
            1, static_cast<int>(std::llround(kSnapshotDt / kPhase500Dt)));
    const int phase100_plot_interval =
            std::max(1, kPhase100Steps / kPhase100Plots);
    const int phase500_plot_interval =
            std::max(1, kPhase500Steps / kPhase500Plots);

    const fs::path base_dir = fs::path(options.base_dir);
    const fs::path frame_dir = base_dir / "frames";
    const fs::path frame_phase100_dir = frame_dir / "NVT_100";
    const fs::path frame_phase500_dir = frame_dir / "NVT_500";
    const fs::path frame_csv_dir = frame_dir / "csv";
    const fs::path phase100_plot_csv = frame_csv_dir / "nvt_100_plot_input.csv";
    const fs::path phase500_plot_csv = frame_csv_dir / "nvt_500_plot_input.csv";
    const fs::path xyz_file = base_dir / "trajectory_nvt.xyz";
    const fs::path restart_file = base_dir / "nvt_100_restart.bin";
    const fs::path cfg_phase100_path = base_dir / "config_nvt_100.json";
    const fs::path cfg_phase500_path = base_dir / "config_nvt_500.json";

    for (const auto &dir :
         {base_dir, frame_dir, frame_phase100_dir, frame_phase500_dir,
          frame_csv_dir}) {
        if (!create_folder(dir, rank_idx)) {
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
    }

    if (rank_idx == 0) {
        std::error_code rm_ec;
        fs::remove(restart_file, rm_ec);

        std::ofstream clear_xyz(xyz_file, std::ios::out | std::ios::trunc);
        std::ofstream clear_phase100_plot_csv(
                phase100_plot_csv, std::ios::out | std::ios::trunc);
        std::ofstream clear_phase500_plot_csv(
                phase500_plot_csv, std::ios::out | std::ios::trunc);
        if (!clear_xyz || !clear_phase100_plot_csv || !clear_phase500_plot_csv) {
            fmt::print(
                    stderr,
                    "[run_series_NVT_batch_xyz_saving_piston] Failed to initialize outputs in {}.\n",
                    base_dir.string());
            MPI_Abort(MPI_COMM_WORLD, 4);
        }

        cfg_phase100.config_to_json(cfg_phase100_path.string());
        cfg_phase500.config_to_json(cfg_phase500_path.string());
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int global_step = 0;
    double global_time = 0.0;
    int phase100_saved_xyz = 0;
    int phase500_saved_xyz = 0;
    int phase100_saved_plots = 0;
    int phase500_saved_plots = 0;

    {
        MDSimulation sim(cfg_phase100, MPI_COMM_WORLD);

        for (int phase_step = 1; phase_step <= kPhase100Steps; ++phase_step) {
            sim.step_single_nose_hoover();

            ++global_step;
            global_time += kPhase100Dt;

            const bool do_xyz = (phase_step % phase100_xyz_interval == 0);
            const bool do_plot = (phase_step % phase100_plot_interval == 0);
            if (!do_xyz && !do_plot) {
                continue;
            }

            sim.sample_collect();

            int output_ok = 1;
            if (rank_idx == 0) {
                std::vector<Particle> frame_particles;
                sim.get_host_particles(frame_particles);

                if (do_xyz) {
                    const double phase_time =
                            static_cast<double>(phase_step) * kPhase100Dt;
                    output_ok = append_xyz_snapshot(
                                        xyz_file, frame_particles,
                                        Phase::NVT_100, global_step, phase_step,
                                        global_time, phase_time, sim.get_Lx(),
                                        sim.get_Ly())
                                        ? 1
                                        : 0;
                    if (output_ok) {
                        ++phase100_saved_xyz;
                    }
                }

                if (output_ok && do_plot) {
                    ++phase100_saved_plots;
                    const fs::path frame_svg =
                            frame_phase100_dir /
                            fmt::format("snapshot_{:02d}_step_{}.svg",
                                        phase100_saved_plots, global_step);
                    try {
                        plot_particles_python(
                                frame_particles, frame_svg.string(),
                                phase100_plot_csv.string(), sim.get_Lx(),
                                sim.get_Ly(), cfg_base.config.SIGMA_AA,
                                cfg_base.config.SIGMA_BB);
                    } catch (const std::exception &e) {
                        fmt::print(
                                stderr,
                                "[run_series_NVT_batch_xyz_saving_piston] Failed to plot NVT_100 frame at step {}: {}\n",
                                global_step, e.what());
                        output_ok = 0;
                    }
                }
            }

            MPI_Bcast(&output_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (!output_ok) {
                MPI_Abort(MPI_COMM_WORLD, 5);
            }
        }

        sim.save_env(restart_file.string(), global_step);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    const int phase100_restart_frame = global_step;

    {
        MDSimulation sim(cfg_phase500, MPI_COMM_WORLD, restart_file.string(),
                         phase100_restart_frame);

        for (int phase_step = 1; phase_step <= kPhase500Steps; ++phase_step) {
            sim.step_single_nose_hoover();

            ++global_step;
            global_time += kPhase500Dt;

            const bool do_xyz = (phase_step % phase500_xyz_interval == 0);
            const bool do_plot = (phase_step % phase500_plot_interval == 0);
            if (!do_xyz && !do_plot) {
                continue;
            }

            sim.sample_collect();

            int output_ok = 1;
            if (rank_idx == 0) {
                std::vector<Particle> frame_particles;
                sim.get_host_particles(frame_particles);

                if (do_xyz) {
                    const double phase_time =
                            static_cast<double>(phase_step) * kPhase500Dt;
                    output_ok = append_xyz_snapshot(
                                        xyz_file, frame_particles,
                                        Phase::NVT_500, global_step, phase_step,
                                        global_time, phase_time, sim.get_Lx(),
                                        sim.get_Ly())
                                        ? 1
                                        : 0;
                    if (output_ok) {
                        ++phase500_saved_xyz;
                    }
                }

                if (output_ok && do_plot) {
                    ++phase500_saved_plots;
                    const fs::path frame_svg =
                            frame_phase500_dir /
                            fmt::format("snapshot_{:02d}_step_{}.svg",
                                        phase500_saved_plots, global_step);
                    try {
                        plot_particles_python(
                                frame_particles, frame_svg.string(),
                                phase500_plot_csv.string(), sim.get_Lx(),
                                sim.get_Ly(), cfg_base.config.SIGMA_AA,
                                cfg_base.config.SIGMA_BB);
                    } catch (const std::exception &e) {
                        fmt::print(
                                stderr,
                                "[run_series_NVT_batch_xyz_saving_piston] Failed to plot NVT_500 frame at step {}: {}\n",
                                global_step, e.what());
                        output_ok = 0;
                    }
                }
            }

            MPI_Bcast(&output_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (!output_ok) {
                MPI_Abort(MPI_COMM_WORLD, 6);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank_idx == 0) {
        std::error_code rm_ec;
        fs::remove(restart_file, rm_ec);

        fmt::print(
                "[run_series_NVT_batch_xyz_saving_piston] Done. "
                "NVT_100 xyz={}, NVT_500 xyz={}, NVT_100 plots={} (target={}), "
                "NVT_500 plots={} (target={}), final_step={}, final_time={:.6f}, "
                "snapshot_dt={:.6f}\n",
                phase100_saved_xyz, phase500_saved_xyz, phase100_saved_plots,
                kPhase100Plots, phase500_saved_plots, kPhase500Plots,
                global_step, global_time, kSnapshotDt);
    }

    MPI_Finalize();
    return 0;
}
