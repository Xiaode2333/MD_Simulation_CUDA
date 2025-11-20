#include "md_env.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
template <typename... Args>
void RankZeroPrint(int rank_idx, fmt::format_string<Args...> format_str, Args&&... args) {
    if (rank_idx == 0) {
        fmt::print(format_str, std::forward<Args>(args)...);
        std::fflush(stdout);
    }
}

bool create_folder(const fs::path& path, int rank_idx) {
    if (rank_idx != 0) {
        return true;
    }

    std::error_code ec;
    if (!fs::exists(path, ec)) {
        fs::create_directories(path, ec);
        if (ec) {
            fmt::print(stderr, "[series_cwa_test] Failed to create dir {}. Error: {}\n", path.string(), ec.message());
            return false;
        }
    }
    return true;
}

bool append_latest_line(const fs::path& src, const fs::path& dst, int rank_idx) {
    if (rank_idx != 0) return true;

    std::ifstream in(src);
    if (!in) {
        fmt::print(stderr, "[series_cwa_test] Failed to open {} for reading.\n", src.string());
        return false;
    }

    std::string line;
    std::string last_non_empty;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            last_non_empty = line;
        }
    }

    if (last_non_empty.empty()) {
        fmt::print(stderr, "[series_cwa_test] No data found in {}.\n", src.string());
        return false;
    }

    std::ofstream out(dst, std::ios::out | std::ios::app);
    if (!out) {
        fmt::print(stderr, "[series_cwa_test] Failed to open {} for appending.\n", dst.string());
        return false;
    }

    out << last_non_empty << '\n';
    return true;
}

void write_density_profile_csv(const fs::path& filepath, const std::vector<double>& density, int rank_idx) {
    if (rank_idx != 0) {
        return;
    }

    std::ofstream out(filepath, std::ios::out | std::ios::trunc);
    if (!out) {
        fmt::print(stderr, "[series_cwa_test] Failed to open {} for writing density profile.\n", filepath.string());
        return;
    }

    out << "bin,rho\n";
    for (std::size_t idx = 0; idx < density.size(); ++idx) {
        out << idx << ',' << density[idx] << '\n';
    }
}

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
        fmt::print(stderr, "[series_cwa_test] {}\n", ex.what());
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
        fmt::print(stderr, "[series_cwa_test] rank size = {} doesn't match config.\n", rank_size);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    cfg_mgr.config.rank_size = rank_size;
    cfg_mgr.config.rank_idx = rank_idx;

    int n_record_interval = static_cast<int>(cfg_mgr.config.save_dt_interval / cfg_mgr.config.dt);
    if (n_record_interval <= 0) {
        n_record_interval = 1;
    }

    const int n_steps = 1000000;
    const int q_min = 3;
    const int q_max = 10;
    const int n_bins_per_rank = 16;
    const int n_last_env_step = n_steps - n_record_interval;

    fs::path base_dir = fs::path(options.base_dir);
    fs::path cwa_plot_dir = base_dir / "cwa_plot_resumed";
    fs::path cwa_plot_csv_dir = cwa_plot_dir / "csv";
    fs::path sample_dir = base_dir / "sample_csv_resumed";
    fs::path cwa_sample_csv = sample_dir / "cwa_instant.csv";
    fs::path U_K_tot_csv_path = sample_dir / "U_K_tot_log.csv";
    fs::path interface_dir = base_dir / "interfaces_resumed";
    fs::path interface_csv_dir = interface_dir / "csv";
    fs::path density_dir = base_dir / "density_profile_resumed";
    fs::path load_env_dir = base_dir / "saved_env";
    fs::path load_env_file = load_env_dir / "saved_env.bin";
    fs::path saved_env_dir = base_dir / "saved_env_resumed";
    fs::path saved_env_file = saved_env_dir / "saved_env.bin";
    fs::path saved_cfg_path = base_dir / "config_resumed.json";

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
        if (!clear_samples || !clear_uk) {
            fmt::print(stderr, "[series_cwa_test] Failed to initialize CSV outputs in {}.\n", sample_dir.string());
            MPI_Abort(MPI_COMM_WORLD, 4);
        }
        cfg_mgr.config_to_json(saved_cfg_path.string());
        cfg_mgr.print_config();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    {
        MDSimulation sim(cfg_mgr, MPI_COMM_WORLD, load_env_file, n_last_env_step);

        RankZeroPrint(rank_idx, "[series_cwa_test] Waiting for MPI_Barrier before stepping.\n");
        MPI_Barrier(MPI_COMM_WORLD);
        RankZeroPrint(rank_idx, "[series_cwa_test] Starting simulation loop.\n");

        for (int step = 0; step < n_steps; ++step) {
            sim.step_single_nose_hoover();

            if (step % n_record_interval == 0) {
                sim.sample_collect();

                sim.save_env(saved_env_file.string(), step);

                const double U_tot = sim.cal_total_U();
                const double K_tot = sim.cal_total_K();
                sim.write_to_file(U_K_tot_csv_path.string(), "U_tot, {}, K_tot, {}, step, {}\n", U_tot, K_tot, step);

                const auto density_profile = sim.get_density_profile(n_bins_per_rank);
                fs::path density_step_csv = density_dir / fmt::format("density_step_{}.csv", step);
                write_density_profile_csv(density_step_csv, density_profile, rank_idx);

                fs::path interface_plot_path = interface_dir / fmt::format("interface_step_{}.svg", step);
                fs::path interface_csv_path = interface_csv_dir / fmt::format("interface_step_{}.csv", step);
                sim.plot_interfaces(interface_plot_path.string(), interface_csv_path.string(), density_profile);

                fs::path cwa_step_csv = cwa_plot_csv_dir / fmt::format("cwa_instant_{}.csv", step);
                fs::path cwa_step_plot = cwa_plot_dir / fmt::format("cwa_instant_{}.svg", step);
                sim.do_CWA_instant(q_min, q_max, cwa_step_csv.string(), cwa_step_plot.string(), true, step);
                append_latest_line(cwa_step_csv, cwa_sample_csv, rank_idx);
            }

            if (step % 100 == 0) {
                RankZeroPrint(rank_idx, "[series_cwa_test] Step {}.\n", step);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
