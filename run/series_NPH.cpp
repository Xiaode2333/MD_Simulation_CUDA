#include "md_common.hpp"
#include "md_env.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
    NPH = 1,
};

const char *phase_name(const Phase phase) {
    switch (phase) {
    case Phase::NVT:
        return "NVT";
    case Phase::NPH:
        return "NPH";
    default:
        return "UNKNOWN";
    }
}

struct TriTypeStats {
    std::array<int, 4> counts{0, 0, 0, 0};
    std::array<double, 4> areas{0.0, 0.0, 0.0, 0.0};
};

TriTypeStats compute_tri_type_stats(const TriangulationResult &tri,
                                    const std::vector<Particle> &particles) {
    TriTypeStats out{};

    const std::size_t vert_count = tri.vertex_to_idx.size();
    if (vert_count == 0 || tri.triangles.empty()) {
        return out;
    }
    if (tri.coords.size() < 2 * vert_count) {
        return out;
    }

    auto vertex_type = [&](int v_idx) -> int {
        if (v_idx < 0 || v_idx >= static_cast<int>(vert_count)) {
            return -1;
        }
        const int p_idx = tri.vertex_to_idx[static_cast<std::size_t>(v_idx)];
        if (p_idx < 0 || p_idx >= static_cast<int>(particles.size())) {
            return -1;
        }
        return particles[static_cast<std::size_t>(p_idx)].type;
    };

    for (const auto &tri_idx : tri.triangles) {
        const int v0 = tri_idx[0];
        const int v1 = tri_idx[1];
        const int v2 = tri_idx[2];

        const int t0 = vertex_type(v0);
        const int t1 = vertex_type(v1);
        const int t2 = vertex_type(v2);

        if (!((t0 == 0 || t0 == 1) && (t1 == 0 || t1 == 1) &&
              (t2 == 0 || t2 == 1))) {
            continue;
        }

        const int a_count = (t0 == 0) + (t1 == 0) + (t2 == 0);
        int type_idx = -1;
        switch (a_count) {
        case 3:
            type_idx = 0; // AAA
            break;
        case 2:
            type_idx = 1; // AAB
            break;
        case 1:
            type_idx = 2; // ABB
            break;
        case 0:
            type_idx = 3; // BBB
            break;
        default:
            break;
        }
        if (type_idx < 0) {
            continue;
        }

        if (v0 < 0 || v1 < 0 || v2 < 0 ||
            static_cast<std::size_t>(v0) >= vert_count ||
            static_cast<std::size_t>(v1) >= vert_count ||
            static_cast<std::size_t>(v2) >= vert_count) {
            continue;
        }

        const double x0 = tri.coords[2 * static_cast<std::size_t>(v0) + 0];
        const double y0 = tri.coords[2 * static_cast<std::size_t>(v0) + 1];
        const double x1 = tri.coords[2 * static_cast<std::size_t>(v1) + 0];
        const double y1 = tri.coords[2 * static_cast<std::size_t>(v1) + 1];
        const double x2 = tri.coords[2 * static_cast<std::size_t>(v2) + 0];
        const double y2 = tri.coords[2 * static_cast<std::size_t>(v2) + 1];

        const double twice_area =
                (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        const double area = 0.5 * std::abs(twice_area);

        ++out.counts[static_cast<std::size_t>(type_idx)];
        out.areas[static_cast<std::size_t>(type_idx)] += area;
    }

    return out;
}

struct RecordContext {
    int global_step = 0;
    int phase_step = 0;
    double time = 0.0;
    double dt = 0.0;
    Phase phase = Phase::NVT;
};

bool append_observables(const fs::path &observables_csv,
                        const fs::path &tri_csv, MDSimulation &sim,
                        const RecordContext &ctx, int rank_idx,
                        const std::string &tag, double barostat_mass,
                        bool compute_pressure_from_scratch) {
    sim.sample_collect();

    const double U_tot = sim.cal_total_U();
    const double K_tot = sim.cal_total_K();
    const double pressure = compute_pressure_from_scratch
                                    ? sim.cal_instant_pressure()
                                    : sim.get_last_instant_pressure();

    const double Lx = sim.get_Lx();
    const double Ly = sim.get_Ly();
    const double volume = Lx * Ly;
    const double L_tot = sim.get_interface_total_length();

    const double area_rate = (ctx.phase == Phase::NPH) ? sim.get_nph_area_rate() : 0.0;
    const double barostat_kinetic = 0.5 * barostat_mass * area_rate * area_rate;

    double ab_length = 0.0;
    TriTypeStats tri_stats{};

    const auto tri_result = sim.triangulation_plot(false, "", "");
    if (tri_result.has_value()) {
        std::vector<Particle> host_particles;
        sim.get_host_particles(host_particles);

        tri_stats = compute_tri_type_stats(*tri_result, host_particles);

        const ABPairNetworks ab_networks =
                sim.get_AB_pair_network(*tri_result, false);
        ab_length = sim.get_AB_pair_length(ab_networks);
    }

    const bool ok_obs = append_csv(
            observables_csv, rank_idx, tag,
            "{},{},{:.8f},{},{:.8e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e}\n",
            ctx.global_step, ctx.phase_step, ctx.time, phase_name(ctx.phase),
            ctx.dt, U_tot, K_tot, pressure, barostat_kinetic, Lx, Ly, volume,
            L_tot, ab_length);

    const bool ok_tri = append_csv(
            tri_csv, rank_idx, tag,
            "{},{},{:.8f},{},{},{},{},{},{:.10e},{:.10e},{:.10e},{:.10e}\n",
            ctx.global_step, ctx.phase_step, ctx.time, phase_name(ctx.phase),
            tri_stats.counts[0], tri_stats.counts[1], tri_stats.counts[2],
            tri_stats.counts[3], tri_stats.areas[0], tri_stats.areas[1],
            tri_stats.areas[2], tri_stats.areas[3]);

    return ok_obs && ok_tri;
}

} // namespace

int main(int argc, char **argv) {
    constexpr double kNvtDt = 1.0e-3;
    constexpr double kNphDt = 1.0e-4;
    constexpr int kNvtSteps = 100000;
    constexpr int kNphSteps = 5000000;
    constexpr int kNvtSnapshots = 20;
    constexpr int kNphSnapshots = 20;
    constexpr double kRecordDt = 1.0;

    ProgramOptions options;
    try {
        options = parse_args(argc, argv);
    } catch (const std::exception &ex) {
        fmt::print(stderr, "[run_series_NPH] {}\n", ex.what());
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
        fmt::print(stderr,
                   "[run_series_NPH] rank size = {} doesn't match config.\n",
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

    const int nvt_record_interval =
            std::max(1, static_cast<int>(std::llround(kRecordDt / kNvtDt)));
    const int nph_record_interval =
            std::max(1, static_cast<int>(std::llround(kRecordDt / kNphDt)));
    const int nvt_snapshot_interval = std::max(1, kNvtSteps / kNvtSnapshots);
    const int nph_snapshot_interval = std::max(1, kNphSteps / kNphSnapshots);

    const fs::path base_dir = fs::path(options.base_dir);
    const fs::path sample_dir = base_dir / "sample_csv";
    const fs::path saved_env_dir = base_dir / "saved_env";
    const fs::path saved_env_file = saved_env_dir / "saved_env.bin";
    const fs::path frame_dir = base_dir / "frames";
    const fs::path frame_nvt_dir = frame_dir / "NVT";
    const fs::path frame_nph_dir = frame_dir / "NPH";
    const fs::path frame_csv_dir = frame_dir / "csv";
    const fs::path nvt_plot_csv = frame_csv_dir / "nvt_plot_input.csv";
    const fs::path nph_plot_csv = frame_csv_dir / "nph_plot_input.csv";
    const fs::path cfg_nvt_path = base_dir / "config_nvt.json";
    const fs::path cfg_nph_path = base_dir / "config_nph.json";

    const fs::path observables_csv = sample_dir / "observables.csv";
    const fs::path tri_csv = sample_dir / "triangulation_stats.csv";

    for (const auto &dir :
         {base_dir, sample_dir, saved_env_dir, frame_dir, frame_nvt_dir,
          frame_nph_dir, frame_csv_dir}) {
        if (!create_folder(dir, rank_idx)) {
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
    }

    const std::string tag = "run_series_NPH";

    if (rank_idx == 0) {
        std::error_code rm_ec;
        fs::remove(saved_env_file, rm_ec);

        std::ofstream clear_observables(observables_csv,
                                        std::ios::out | std::ios::trunc);
        std::ofstream clear_tri(tri_csv, std::ios::out | std::ios::trunc);
        std::ofstream clear_nvt_plot_csv(nvt_plot_csv,
                                         std::ios::out | std::ios::trunc);
        std::ofstream clear_nph_plot_csv(nph_plot_csv,
                                         std::ios::out | std::ios::trunc);

        if (!clear_observables || !clear_tri || !clear_nvt_plot_csv ||
            !clear_nph_plot_csv) {
            fmt::print(stderr,
                       "[run_series_NPH] Failed to initialize outputs in {}.\n",
                       sample_dir.string());
            MPI_Abort(MPI_COMM_WORLD, 4);
        }

        append_csv(observables_csv, rank_idx, tag,
                   "global_step,phase_step,time,phase,dt,U_tot,K_tot,P,box_motion_K,Lx,Ly,volume,L_tot,AB_pair_length\n");
        append_csv(tri_csv, rank_idx, tag,
                   "global_step,phase_step,time,phase,AAA_count,AAB_count,ABB_count,BBB_count,AAA_area,AAB_area,ABB_area,BBB_area\n");

        cfg_nvt.config_to_json(cfg_nvt_path.string());
        cfg_nph.config_to_json(cfg_nph_path.string());
        cfg_base.print_config();
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

            const bool do_record = (phase_step % nvt_record_interval == 0);
            const bool do_snapshot = (phase_step % nvt_snapshot_interval == 0);

            if (do_record) {
                const RecordContext ctx{global_step, phase_step, global_time,
                                        kNvtDt, Phase::NVT};
                if (!append_observables(observables_csv, tri_csv, sim, ctx,
                                        rank_idx, tag,
                                        std::max(cfg_base.config.barostat_mass,
                                                 1.0e-12),
                                        true)) {
                    MPI_Abort(MPI_COMM_WORLD, 5);
                }

                if (do_snapshot) {
                    sim.save_env(saved_env_file.string(), global_step);
                    if (rank_idx == 0) {
                        ++nvt_saved_snapshots;
                        std::vector<Particle> frame_particles;
                        sim.get_host_particles(frame_particles);
                        const fs::path frame_svg =
                                frame_nvt_dir /
                                fmt::format("snapshot_{:02d}_step_{}.svg",
                                            nvt_saved_snapshots, global_step);
                        try {
                            plot_particles_python(
                                    frame_particles, frame_svg.string(),
                                    nvt_plot_csv.string(), sim.get_Lx(),
                                    sim.get_Ly(), cfg_base.config.SIGMA_AA,
                                    cfg_base.config.SIGMA_BB);
                        } catch (const std::exception &e) {
                            fmt::print(
                                    stderr,
                                    "[run_series_NPH] Failed to plot NVT frame at step {}: {}\n",
                                    global_step, e.what());
                            MPI_Abort(MPI_COMM_WORLD, 7);
                        }
                    }
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    const int nvt_restart_frame = global_step;

    {
        MDSimulation sim(cfg_nph, MPI_COMM_WORLD, saved_env_file.string(),
                         nvt_restart_frame);

        for (int phase_step = 1; phase_step <= kNphSteps; ++phase_step) {
            sim.step_single_NPH(cfg_nph.config.P_target);

            ++global_step;
            global_time += kNphDt;

            const bool do_record = (phase_step % nph_record_interval == 0);
            const bool do_snapshot = (phase_step % nph_snapshot_interval == 0);

            if (do_record) {
                const RecordContext ctx{global_step, phase_step, global_time,
                                        kNphDt, Phase::NPH};
                if (!append_observables(observables_csv, tri_csv, sim, ctx,
                                        rank_idx, tag,
                                        std::max(cfg_base.config.barostat_mass,
                                                 1.0e-12),
                                        false)) {
                    MPI_Abort(MPI_COMM_WORLD, 6);
                }

                if (do_snapshot) {
                    sim.save_env(saved_env_file.string(), global_step);
                    if (rank_idx == 0) {
                        ++nph_saved_snapshots;
                        std::vector<Particle> frame_particles;
                        sim.get_host_particles(frame_particles);
                        const fs::path frame_svg =
                                frame_nph_dir /
                                fmt::format("snapshot_{:02d}_step_{}.svg",
                                            nph_saved_snapshots, global_step);
                        try {
                            plot_particles_python(
                                    frame_particles, frame_svg.string(),
                                    nph_plot_csv.string(), sim.get_Lx(),
                                    sim.get_Ly(), cfg_base.config.SIGMA_AA,
                                    cfg_base.config.SIGMA_BB);
                        } catch (const std::exception &e) {
                            fmt::print(
                                    stderr,
                                    "[run_series_NPH] Failed to plot NPH frame at step {}: {}\n",
                                    global_step, e.what());
                            MPI_Abort(MPI_COMM_WORLD, 8);
                        }
                    }
                }
            }
        }
    }

    if (rank_idx == 0) {
        fmt::print(
                "[run_series_NPH] Done. NVT snapshots={} (target={}), "
                "NPH snapshots={} (target={}), final_step={}, final_time={:.6f}\n",
                nvt_saved_snapshots, kNvtSnapshots, nph_saved_snapshots,
                kNphSnapshots, global_step, global_time);
    }

    MPI_Finalize();
    return 0;
}
