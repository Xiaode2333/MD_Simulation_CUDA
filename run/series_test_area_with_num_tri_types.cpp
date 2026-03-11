#include "md_env.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {
struct ProgramOptions {
    std::string base_dir;
    std::string ori_config;
    std::vector<MDConfigOverride> overrides;
    double epsilon_deform = 0.15;
    double lambda_deform = 1.0;
};

void print_usage(const char* prog_name) {
    fmt::print(
        "Usage: {} --base-dir <output_dir> --ori-config <config.json> "
        "[--epsilon-deform <val>] [--lambda-deform <val>] [--D<Param>=<value> ...]\n",
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
        } else if (arg.rfind("--epsilon-deform", 0) == 0) {
            const std::string value = consume_value(arg, argc, argv, idx);
            opts.epsilon_deform = std::stod(value);
        } else if (arg.rfind("--lambda-deform", 0) == 0) {
            const std::string value = consume_value(arg, argc, argv, idx);
            opts.lambda_deform = std::stod(value);
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

bool write_ab_network_snapshot(const fs::path& path,
                               const ABPairNetworks& networks) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        return false;
    }

    std::size_t total_nodes = 0;
    std::size_t total_edges = 0;
    for (const auto& nodes : networks.networks_nodes) {
        total_nodes += nodes.size();
    }
    for (const auto& edges : networks.networks_edges) {
        total_edges += edges.size();
    }

    out << "summary,n_networks," << networks.networks_nodes.size()
        << ",n_nodes_total," << total_nodes
        << ",n_edges_total," << total_edges << '\n';

    for (std::size_t net_idx = 0; net_idx < networks.networks_nodes.size(); ++net_idx) {
        const auto& nodes = networks.networks_nodes[net_idx];
        for (std::size_t node_idx = 0; node_idx < nodes.size(); ++node_idx) {
            const auto& node = nodes[node_idx];
            out << "node,net_idx," << net_idx
                << ",node_idx," << node_idx
                << ",x," << node.x
                << ",y," << node.y << '\n';
        }
    }

    for (std::size_t net_idx = 0; net_idx < networks.networks_edges.size(); ++net_idx) {
        const auto& edges = networks.networks_edges[net_idx];
        const auto& nodes = net_idx < networks.networks_nodes.size()
                                ? networks.networks_nodes[net_idx]
                                : std::vector<ABPairNetworks::Node>{};
        for (std::size_t edge_idx = 0; edge_idx < edges.size(); ++edge_idx) {
            const auto& edge = edges[edge_idx];
            if (edge.node0 < 0 || edge.node1 < 0 ||
                edge.node0 >= static_cast<int>(nodes.size()) ||
                edge.node1 >= static_cast<int>(nodes.size())) {
                continue;
            }
            const auto& node0 = nodes[edge.node0];
            const auto& node1 = nodes[edge.node1];
            out << "edge,net_idx," << net_idx
                << ",edge_idx," << edge_idx
                << ",node0," << edge.node0
                << ",node1," << edge.node1
                << ",x0," << node0.x
                << ",y0," << node0.y
                << ",x1," << node1.x
                << ",y1," << node1.y << '\n';
        }
    }

    return true;
}
} // namespace

int main(int argc, char** argv) {
    constexpr std::string_view prog_name = "series_test_area_with_num_tri_types";
    ProgramOptions options;
    try {
        options = parse_args(argc, argv);
    } catch (const std::exception& ex) {
        fmt::print(stderr, "[{}] {}\n", prog_name, ex.what());
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
        fmt::print(stderr, "[{}] rank size = {} doesn't match config.\n", prog_name, rank_size);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    cfg_mgr.config.rank_size = rank_size;
    cfg_mgr.config.rank_idx = rank_idx;

    // Apply deformation to box dimensions using epsilon_deform and lambda_deform.
    const double Lx_from_cfg = cfg_mgr.config.box_w_global;
    const double Ly_from_cfg = cfg_mgr.config.box_h_global;
    const double exponent = options.epsilon_deform * options.lambda_deform;
    const double Lx_deformed = Lx_from_cfg * std::exp(exponent);
    const double Ly_deformed = Ly_from_cfg * std::exp(-exponent);
    cfg_mgr.config.box_w_global = Lx_deformed;
    cfg_mgr.config.box_h_global = Ly_deformed;

    int n_record_interval = static_cast<int>(cfg_mgr.config.save_dt_interval / cfg_mgr.config.dt);
    if (n_record_interval <= 0) {
        n_record_interval = 1;
    }

    const int n_steps = 500'000;
    const int q_min = 3;
    const int q_max = 10;
    const int n_bins_local = 16;

    fs::path base_dir = fs::path(options.base_dir);
    fs::path cwa_plot_dir = base_dir / "cwa_plot";
    fs::path cwa_plot_csv_dir = cwa_plot_dir / "csv";
    fs::path sample_dir = base_dir / "sample_csv";
    fs::path cwa_sample_csv = sample_dir / "cwa_instant.csv";
    fs::path U_K_tot_csv_path = sample_dir / "U_K_tot_log.csv";
    fs::path density_profile_csv_path = sample_dir / "density_profile_log.csv";
    fs::path pressure_profile_csv_path = sample_dir / "pressure_profile.csv";
    fs::path tri_types_csv_path = sample_dir / "triangulation_types_log.csv";
    fs::path ab_length_csv_path = sample_dir / "ab_pair_length_log.csv";
    const std::string tag = std::string(prog_name);
    fs::path interface_dir = base_dir / "interfaces";
    fs::path interface_csv_dir = interface_dir / "csv";
    fs::path triangulation_dir = base_dir / "triangulation";
    fs::path triangulation_csv_dir = triangulation_dir / "csv";
    fs::path ab_network_dir = base_dir / "ab_networks";
    fs::path ab_network_csv_dir = ab_network_dir / "csv";
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
        triangulation_dir,
        triangulation_csv_dir,
        ab_network_dir,
        ab_network_csv_dir,
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
        std::ofstream clear_density(density_profile_csv_path, std::ios::out | std::ios::trunc);
        std::ofstream clear_pressure(pressure_profile_csv_path, std::ios::out | std::ios::trunc);
        std::ofstream clear_tri_types(tri_types_csv_path, std::ios::out | std::ios::trunc);
        std::ofstream clear_ab_length(ab_length_csv_path, std::ios::out | std::ios::trunc);
        if (!clear_samples || !clear_uk || !clear_density || !clear_pressure ||
            !clear_tri_types || !clear_ab_length) {
            fmt::print(stderr, "[{}] Failed to initialize CSV outputs in {}.\n",
                       prog_name,
                       sample_dir.string());
            MPI_Abort(MPI_COMM_WORLD, 4);
        }
        append_csv(pressure_profile_csv_path,
                   rank_idx,
                   tag,
                   "step,n_pressure_bins,Lx,Ly,values...\n");
        append_csv(tri_types_csv_path,
                   rank_idx,
                   tag,
                   "step,AAA,AAB,ABB,BBB\n");
        append_csv(ab_length_csv_path,
                   rank_idx,
                   tag,
                   "step,AB_pair_length\n");
        cfg_mgr.config_to_json(saved_cfg_path.string());
        cfg_mgr.print_config();
        fmt::print("[{}] epsilon_deform = {}, lambda_deform = {}\n",
                   prog_name,
                   options.epsilon_deform, options.lambda_deform);
        fmt::print("[{}] Lx_from_cfg = {}, Ly_from_cfg = {}, Lx = {}, Ly = {}\n",
                   prog_name,
                   Lx_from_cfg, Ly_from_cfg, Lx_deformed, Ly_deformed);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    {
        MDSimulation sim(cfg_mgr, MPI_COMM_WORLD);

        RankZeroPrint(rank_idx,
                      "[{}] Waiting for MPI_Barrier before stepping.\n", prog_name);
        MPI_Barrier(MPI_COMM_WORLD);
        RankZeroPrint(rank_idx,
                      "[{}] Starting simulation loop.\n", prog_name);

        for (int step = 0; step < n_steps; ++step) {
            sim.step_single_nose_hoover();

            if (step % n_record_interval == 0) {
                sim.sample_collect();

                

                const double U_tot = sim.cal_total_U();
                const double K_tot = sim.cal_total_K();
                const double partial_U_lambda = sim.cal_partial_U_lambda(options.epsilon_deform);
                const double Lx = sim.get_Lx();
                const double Ly = sim.get_Ly();
                const double L_tot = sim.get_interface_total_length();

                append_csv( U_K_tot_csv_path,
                            rank_idx,
                            tag,
                            "U_tot, {}, K_tot, {}, partial_U_lambda, {}, epsilon_deform, {}, lambda_deform, {}, Lx, {}, Ly, {}, L_tot, {}, step, {}\n",
                            U_tot,
                            K_tot,
                            partial_U_lambda,
                            options.epsilon_deform,
                            options.lambda_deform,
                            Lx,
                            Ly,
                            L_tot,
                            step);

                const auto density_profile = sim.get_density_profile(n_bins_local);
                fs::path density_step_csv = density_dir / fmt::format("density_step_{}.csv", step);
                write_density_profile_csv(density_step_csv, density_profile, rank_idx, tag);

                const auto pressure_profile = sim.get_pressure_profile(n_bins_local);

                if (rank_idx == 0 && !pressure_profile.empty()) {
                    const std::size_t n_pressure_bins = pressure_profile.size() / 3u;

                    fmt::memory_buffer pbuf;
                    fmt::format_to(std::back_inserter(pbuf), "{},{},{},{}", step, n_pressure_bins, Lx, Ly);
                    for (double value : pressure_profile) {
                        fmt::format_to(std::back_inserter(pbuf), ",{}", value);
                    }
                    fmt::format_to(std::back_inserter(pbuf), "\n");

                    append_csv(pressure_profile_csv_path,
                               rank_idx,
                               tag,
                               "{}",
                               std::string(pbuf.data(), pbuf.size()));
                }
                

                // Also append density profile into a single CSV: step followed by all bin values.
                if (rank_idx == 0) {
                    fmt::memory_buffer buf;
                    fmt::format_to(std::back_inserter(buf), "{},{}", step, density_profile.front());
                    for (std::size_t i = 1; i < density_profile.size(); ++i) {
                        fmt::format_to(std::back_inserter(buf), ",{}", density_profile[i]);
                    }
                    fmt::format_to(std::back_inserter(buf), "\n");
                    append_csv(density_profile_csv_path,
                               rank_idx,
                               tag,
                               "{}",
                               std::string(buf.data(), buf.size()));

                }
                
                bool large_op = (step % (n_steps/20) == 0);
                fs::path tri_plot_path = triangulation_dir / fmt::format("triangulation_step_{}.svg", step);
                fs::path tri_csv_path = triangulation_csv_dir / fmt::format("triangulation_step_{}.csv", step);
                fs::path ab_plot_path = ab_network_dir / fmt::format("ab_network_step_{}.svg", step);
                fs::path ab_plot_csv_path = ab_network_csv_dir / fmt::format("ab_network_step_{}.csv", step);
                fs::path ab_snapshot_path = ab_network_csv_dir / fmt::format("ab_network_snapshot_step_{}.csv", step);

                std::optional<TriangulationResult> tri_result;
                ABPairNetworks ab_networks;
                bool ab_network_built = false;

                if (large_op) {
                    // Large storage operations, save less often
                    sim.save_env(saved_env_file.string(), step);
                    fs::path interface_plot_path = interface_dir / fmt::format("interface_step_{}.svg", step);
                    fs::path interface_csv_path = interface_csv_dir / fmt::format("interface_step_{}.csv", step);
                    sim.plot_interfaces(interface_plot_path.string(), interface_csv_path.string(), density_profile);
                    tri_result = sim.triangulation_plot(true, tri_plot_path.string(), tri_csv_path.string());
                    if (tri_result) {
                        ab_networks = sim.get_AB_pair_network(*tri_result,
                                                              true,
                                                              ab_plot_path.string(),
                                                              ab_plot_csv_path.string());
                        ab_network_built = true;
                    }
                }

                if (step > 100'000) {
                    if (!tri_result) {
                        tri_result = sim.triangulation_plot(false, tri_plot_path.string(), tri_csv_path.string());
                    }
                    if (tri_result && !ab_network_built) {
                        ab_networks = sim.get_AB_pair_network(*tri_result,
                                                              false,
                                                              ab_plot_path.string(),
                                                              ab_plot_csv_path.string());
                        ab_network_built = true;
                    }

                    if (tri_result) {
                        auto tri_counts = sim.get_tri_types_num(*tri_result);
                        if (tri_counts.size() >= 4) {
                            append_csv(tri_types_csv_path,
                                       rank_idx,
                                       tag,
                                       "step, {}, AAA, {}, AAB, {}, ABB, {}, BBB, {}\n",
                                       step,
                                       tri_counts[0],
                                       tri_counts[1],
                                       tri_counts[2],
                                       tri_counts[3]);
                        }
                    }

                    if (ab_network_built) {
                        double ab_length = sim.get_AB_pair_length(ab_networks);
                        append_csv(ab_length_csv_path,
                                   rank_idx,
                                   tag,
                                   "step, {}, AB_pair_length, {}\n",
                                   step,
                                   ab_length);
                        if (rank_idx == 0) {
                            if (!write_ab_network_snapshot(ab_snapshot_path, ab_networks)) {
                                fmt::print(stderr,
                                           "[{}] Failed to write AB network snapshot at step {}.\n",
                                           prog_name,
                                           step);
                            }
                        }
                    }

                    fs::path cwa_step_csv = cwa_plot_csv_dir / fmt::format("cwa_instant_{}.csv", step);
                    fs::path cwa_step_plot = cwa_plot_dir / fmt::format("cwa_instant_{}.svg", step);
                    sim.do_CWA_instant(q_min, q_max, cwa_step_csv.string(), cwa_step_plot.string(), large_op, step);//only plot when large_op is true
                    append_latest_line(cwa_step_csv, cwa_sample_csv, rank_idx, tag);
                }
            }

            if (step % 1000 == 0) {
                RankZeroPrint(rank_idx, "[{}] Step {}.\n", prog_name, step);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
