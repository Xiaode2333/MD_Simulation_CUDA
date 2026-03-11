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
    NVT = 0,
    NPH_PISTON = 1,
};

const char *phase_name(const Phase phase) {
    switch (phase) {
    case Phase::NVT:
        return "NVT";
    case Phase::NPH_PISTON:
        return "NPH_PISTON";
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
    constexpr double kNvtDt = 1.0e-3;
    constexpr double kNphDt = 1.0e-4;
    constexpr int kNvtSteps = 100000;
    constexpr int kNphSteps = 5000000;
    constexpr double kSnapshotDt = 1.0e-1;

    ProgramOptions options;
    try {
        options = parse_args(argc, argv);
    } catch (const std::exception &ex) {
        fmt::print(stderr, "[run_series_NPH_batch_xyz_saving_piston] {}\n",
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
                "[run_series_NPH_batch_xyz_saving_piston] rank size = {} doesn't match config.\n",
                rank_size);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    cfg_base.config.rank_size = rank_size;
    cfg_base.config.rank_idx = rank_idx;

    MDConfigManager cfg_nvt = cfg_base;
    cfg_nvt.config.dt = kNvtDt;
    cfg_nvt.config.rank_size = rank_size;
    cfg_nvt.config.rank_idx = rank_idx;

    MDConfigManager cfg_nph = cfg_base;
    cfg_nph.config.dt = kNphDt;
    cfg_nph.config.rank_size = rank_size;
    cfg_nph.config.rank_idx = rank_idx;

    const int nvt_snapshot_interval = std::max(
            1, static_cast<int>(std::llround(kSnapshotDt / kNvtDt)));
    const int nph_snapshot_interval = std::max(
            1, static_cast<int>(std::llround(kSnapshotDt / kNphDt)));

    const fs::path base_dir = fs::path(options.base_dir);
    const fs::path xyz_file = base_dir / "trajectory_piston.xyz";
    const fs::path restart_file = base_dir / "nvt_restart_piston.bin";

    if (!create_folder(base_dir, rank_idx)) {
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    if (rank_idx == 0) {
        std::error_code rm_ec;
        fs::remove(restart_file, rm_ec);

        std::ofstream clear_xyz(xyz_file, std::ios::out | std::ios::trunc);
        if (!clear_xyz) {
            fmt::print(
                    stderr,
                    "[run_series_NPH_batch_xyz_saving_piston] Failed to initialize {}.\n",
                    xyz_file.string());
            MPI_Abort(MPI_COMM_WORLD, 4);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int global_step = 0;
    double global_time = 0.0;
    int nvt_saved_snapshots = 0;
    int nph_saved_snapshots = 0;

    {
        MDSimulation sim(cfg_nvt, MPI_COMM_WORLD);

        for (int phase_step = 1; phase_step <= kNvtSteps; ++phase_step) {
            sim.step_single_nose_hoover();

            ++global_step;
            global_time += kNvtDt;

            const bool do_snapshot = (phase_step % nvt_snapshot_interval == 0);
            if (!do_snapshot) {
                continue;
            }

            sim.sample_collect();

            int snapshot_ok = 1;
            if (rank_idx == 0) {
                std::vector<Particle> frame_particles;
                sim.get_host_particles(frame_particles);
                const double phase_time = static_cast<double>(phase_step) * kNvtDt;
                snapshot_ok = append_xyz_snapshot(
                                      xyz_file, frame_particles, Phase::NVT,
                                      global_step, phase_step, global_time,
                                      phase_time, sim.get_Lx(), sim.get_Ly())
                                      ? 1
                                      : 0;
            }

            MPI_Bcast(&snapshot_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (!snapshot_ok) {
                MPI_Abort(MPI_COMM_WORLD, 5);
            }

            ++nvt_saved_snapshots;
        }

        sim.save_env(restart_file.string(), global_step);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    const int nvt_restart_frame = global_step;

    {
        MDSimulation sim(cfg_nph, MPI_COMM_WORLD, restart_file.string(),
                         nvt_restart_frame);

        for (int phase_step = 1; phase_step <= kNphSteps; ++phase_step) {
            sim.step_single_NPH_piston(cfg_nph.config.P_target);

            ++global_step;
            global_time += kNphDt;

            const bool do_snapshot = (phase_step % nph_snapshot_interval == 0);
            if (!do_snapshot) {
                continue;
            }

            sim.sample_collect();

            int snapshot_ok = 1;
            if (rank_idx == 0) {
                std::vector<Particle> frame_particles;
                sim.get_host_particles(frame_particles);
                const double phase_time = static_cast<double>(phase_step) * kNphDt;
                snapshot_ok = append_xyz_snapshot(
                                      xyz_file, frame_particles, Phase::NPH_PISTON,
                                      global_step, phase_step, global_time,
                                      phase_time, sim.get_Lx(), sim.get_Ly())
                                      ? 1
                                      : 0;
            }

            MPI_Bcast(&snapshot_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (!snapshot_ok) {
                MPI_Abort(MPI_COMM_WORLD, 6);
            }

            ++nph_saved_snapshots;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank_idx == 0) {
        std::error_code rm_ec;
        fs::remove(restart_file, rm_ec);

        fmt::print(
                "[run_series_NPH_batch_xyz_saving_piston] Done. NVT snapshots={}, "
                "NPH_PISTON snapshots={}, final_step={}, final_time={:.6f}, "
                "snapshot_dt={:.6f}\n",
                nvt_saved_snapshots, nph_saved_snapshots, global_step,
                global_time, kSnapshotDt);
    }

    MPI_Finalize();
    return 0;
}
