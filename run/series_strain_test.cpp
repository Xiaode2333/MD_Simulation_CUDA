#include "md_env.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
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

void print_usage(const char* prog_name) {
    fmt::print(
        "Usage: {} --base-dir <output_dir> --ori-config <config.json> [--D<Param>=<value> ...]\n",
        prog_name
    );
}

std::string consume_value(const std::string& arg, int argc, char** argv, int& idx) {
    auto equal_pos = arg.find('=');
    if (equal_pos != std::string::npos) {
        return arg.substr(equal_pos + 1);
    }

    if (idx + 1 >= argc) {
        throw std::runtime_error(fmt::format("Missing value for '{}'", arg));
    }

    ++idx;
    return argv[idx];
}

ProgramOptions parse_args(int argc, char** argv) {
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
            if (arg.rfind("--D", 0) == 0 || arg.rfind('D', 0) == 0) {
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
                throw std::runtime_error(fmt::format("Failed to parse override '{}'", arg));
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
} // namespace

int main(int argc, char** argv) {
    ProgramOptions options;
    try {
        options = parse_args(argc, argv);
    } catch (const std::exception& ex) {
        fmt::print(stderr, "[series_strain_test] {}\n", ex.what());
        return 1;
    }

    MDConfigManager cfg_mgr = MDConfigManager::config_from_json(options.ori_config);
    if (!options.overrides.empty()) {
        cfg_mgr.apply_overrides(options.overrides);
    }

    MPI_Init(nullptr, nullptr);
    int rank_size = 0;
    int rank_idx = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_idx);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_size);

    if (cfg_mgr.config.rank_size != rank_size) {
        fmt::print(stderr, "[series_strain_test] rank size = {} doesn't match config.\n", rank_size);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    cfg_mgr.config.rank_size = rank_size;
    cfg_mgr.config.rank_idx = rank_idx;

    int n_record_interval = static_cast<int>(cfg_mgr.config.save_dt_interval / cfg_mgr.config.dt);
    if (n_record_interval <= 0) {
        n_record_interval = 1;
    }

    const int n_steps = 10000000;
    const int n_steps_strain_eq = 50000;
    const int n_steps_sample = 100000;
    const int n_step_pre_eq = 1000000;
    const int q_min = 3;
    const int q_max = 10;
    const int n_bins_per_rank = 32;
    const double epsilon_target = 1e-2;
    const double sigma_max = std::max({cfg_mgr.config.SIGMA_AA, cfg_mgr.config.SIGMA_AB, cfg_mgr.config.SIGMA_BB});
    const double strain_displacement_ratio = 1e-3; // Lx * epsilon = 1e-3 * sigma_max

    fs::path base_dir = fs::path(options.base_dir);
    fs::path cwa_plot_dir = base_dir / "cwa_plot";
    fs::path cwa_plot_csv_dir = cwa_plot_dir / "csv";
    fs::path sample_dir = base_dir / "sample_csv";
    fs::path cwa_sample_csv = sample_dir / "cwa_instant.csv";
    fs::path U_K_tot_csv_path = sample_dir / "U_K_tot_log.csv";
    const std::string cwa_tag = "series_strain_test";
    fs::path strain_log_csv_path = sample_dir / "strain_log.csv";
    fs::path interface_dir = base_dir / "interfaces";
    fs::path interface_csv_dir = interface_dir / "csv";
    fs::path density_dir = base_dir / "density_profile";
    fs::path saved_env_dir = base_dir / "saved_env";
    fs::path saved_env_file = saved_env_dir / "saved_env.bin";
    fs::path saved_cfg_path = base_dir / "config.json";

    std::vector<fs::path> dirs_to_create = {
        base_dir,
        cwa_plot_dir,
        cwa_plot_csv_dir,
        sample_dir,
        interface_dir,
        interface_csv_dir,
        density_dir,
        saved_env_dir
    };

    for (const auto& dir : dirs_to_create) {
        if (!create_folder(dir, rank_idx)) {
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
    }

    if (rank_idx == 0) {
        std::ofstream clear_samples(cwa_sample_csv, std::ios::out | std::ios::trunc);
        std::ofstream clear_uk(U_K_tot_csv_path, std::ios::out | std::ios::trunc);
        std::ofstream clear_strain(strain_log_csv_path, std::ios::out | std::ios::trunc);
        if (!clear_samples || !clear_uk || !clear_strain) {
            fmt::print(stderr, "[series_strain_test] Failed to initialize CSV outputs in {}.\n", sample_dir.string());
            MPI_Abort(MPI_COMM_WORLD, 4);
        }
        cfg_mgr.config_to_json(saved_cfg_path.string());
        cfg_mgr.print_config();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    {
        MDSimulation sim(cfg_mgr, MPI_COMM_WORLD);

        RankZeroPrint(rank_idx, "[series_strain_test] Waiting for MPI_Barrier before stepping.\n");
        MPI_Barrier(MPI_COMM_WORLD);
        RankZeroPrint(rank_idx, "[series_strain_test] Starting simulation loop.\n");

        enum class StrainPhase { Deforming, Equilibrating, Sampling };
        StrainPhase strain_phase = StrainPhase::Deforming;
        double accumulated_epsilon = 0.0;
        int strain_phase_step = 0;

        for (int step = 0; step < n_steps; ++step) {
            sim.step_single_nose_hoover();

            bool is_deforming = false;
            if (step > n_step_pre_eq) {
                switch (strain_phase) {
                    case StrainPhase::Deforming: {
                        const double Lx_current = sim.get_Lx();
                        const double epsilon_step = strain_displacement_ratio * sigma_max / Lx_current;
                        const double remaining = epsilon_target - accumulated_epsilon;
                        const double epsilon = std::min(epsilon_step, remaining);

                        if (epsilon > 0.0) {
                            sim.sample_collect();
                            const double U_old = sim.cal_total_U();
                            const double dU = sim.deform(epsilon, U_old);
                            const double Lx_new = sim.get_Lx();
                            const double Ly_new = sim.get_Ly();
                            accumulated_epsilon += epsilon;
                            is_deforming = true;
                            sim.write_to_file(strain_log_csv_path.string(),
                                              "step, {}, epsilon, {}, Lx, {}, Ly, {}, dU, {}\n",
                                              step, epsilon, Lx_new, Ly_new, dU);

                            if (accumulated_epsilon >= epsilon_target) {
                                strain_phase = StrainPhase::Equilibrating;
                                strain_phase_step = 0;
                            }
                        }
                        break;
                    }
                    case StrainPhase::Equilibrating:
                        ++strain_phase_step;
                        if (strain_phase_step >= n_steps_strain_eq) {
                            strain_phase = StrainPhase::Sampling;
                            strain_phase_step = 0;
                        }
                        break;
                    case StrainPhase::Sampling:
                        ++strain_phase_step;
                        if (strain_phase_step >= n_steps_sample) {
                            strain_phase = StrainPhase::Deforming;
                            strain_phase_step = 0;
                            accumulated_epsilon = 0.0;
                        }
                        break;
                }
            }

            if (step % n_record_interval == 0) {
                sim.sample_collect();

                sim.save_env(saved_env_file.string(), step);

                const double U_tot = sim.cal_total_U();
                const double K_tot = sim.cal_total_K();
                const double Lx = sim.get_Lx();
                const double Ly = sim.get_Ly();

                double L_interface_tot = 0.0;
                int n_grid_y = static_cast<int>(Ly / 2.0);
                if (n_grid_y < 10) n_grid_y = 10;
                const auto interfaces = sim.get_smooth_interface(n_grid_y, 2.0);
                for (const auto& segs : interfaces) {
                    for (std::size_t idx = 0; idx + 3 < segs.size(); idx += 4) {
                        double dx = segs[idx + 2] - segs[idx];
                        if (dx > 0.5 * Lx) dx -= Lx;
                        if (dx < -0.5 * Lx) dx += Lx;
                        double dy = segs[idx + 3] - segs[idx + 1];
                        L_interface_tot += std::sqrt(dx * dx + dy * dy);
                    }
                }

                sim.write_to_file(U_K_tot_csv_path.string(),
                                  "Lx, {}, Ly, {}, L_interface_tot, {}, U_tot, {}, K_tot, {}, step, {}, is_deforming, {}\n",
                                  Lx, Ly, L_interface_tot, U_tot, K_tot, step, (is_deforming ? 1 : 0));

                if (step % (100*n_record_interval) == 0){
                    const auto density_profile = sim.get_density_profile(n_bins_per_rank);
                    fs::path density_step_csv = density_dir / fmt::format("density_step_{}.csv", step);
                    write_density_profile_csv(density_step_csv, density_profile, rank_idx, cwa_tag);

                    fs::path interface_plot_path = interface_dir / fmt::format("interface_step_{}.svg", step);
                    fs::path interface_csv_path = interface_csv_dir / fmt::format("interface_step_{}.csv", step);
                    sim.plot_interfaces(interface_plot_path.string(), interface_csv_path.string(), density_profile);
                }
                

                if (step > n_step_pre_eq && strain_phase == StrainPhase::Sampling) {
                    fs::path cwa_step_csv = cwa_plot_csv_dir / fmt::format("cwa_instant_{}.csv", step);
                    fs::path cwa_step_plot = cwa_plot_dir / fmt::format("cwa_instant_{}.svg", step);
                    sim.do_CWA_instant(q_min, q_max, cwa_step_csv.string(), cwa_step_plot.string(), true, step);
                    if (fs::exists(cwa_step_csv)) {
                        append_latest_line(cwa_step_csv, cwa_sample_csv, rank_idx, cwa_tag);
                    }
                } 
            }

            if (step % 100 == 0) {
                const char* phase_str = "unknown";
                switch (strain_phase) {
                    case StrainPhase::Deforming: phase_str = "deforming"; break;
                    case StrainPhase::Equilibrating: phase_str = "equilibrating"; break;
                    case StrainPhase::Sampling: phase_str = "sampling"; break;
                }
                RankZeroPrint(rank_idx, "[series_strain_test] Step {}, phase: {}.\n", step, phase_str);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
