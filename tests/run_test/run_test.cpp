#include "md_env.hpp"
#include "md_energy_minimizer.hpp"
#include "md_cuda_common.hpp" // for Hessian computation helpers
#include <algorithm>
#include <fmt/ranges.h>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <cmath>
#include <unordered_set>
#include <cstdint>

// Minimum-image helper for host computations
static inline double mic(double d, double L) {
    if (L <= 0.0) return d;
    return d - L * std::round(d / L);
}

// Dense matrix utilities (row-major)
static void matmul(const std::vector<double> &A, int m, int k,
                   const std::vector<double> &B, int n,
                   std::vector<double> &C) {
    C.assign(static_cast<size_t>(m) * n, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int p = 0; p < k; ++p) {
            const double a = A[static_cast<size_t>(i) * k + p];
            if (a == 0.0) continue;
            const double *brow = &B[static_cast<size_t>(p) * n];
            double *crow = &C[static_cast<size_t>(i) * n];
            for (int j = 0; j < n; ++j) {
                crow[j] += a * brow[j];
            }
        }
    }
}

static bool invert_in_place(std::vector<double> &A, int n) {
    std::vector<double> I(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) I[static_cast<size_t>(i) * n + i] = 1.0;

    for (int col = 0; col < n; ++col) {
        // pivot
        int pivot = col;
        double max_abs = std::fabs(A[static_cast<size_t>(col) * n + col]);
        for (int r = col + 1; r < n; ++r) {
            double v = std::fabs(A[static_cast<size_t>(r) * n + col]);
            if (v > max_abs) {
                max_abs = v;
                pivot = r;
            }
        }
        if (max_abs < 1e-14) return false;
        if (pivot != col) {
            for (int c = 0; c < n; ++c) {
                std::swap(A[static_cast<size_t>(col) * n + c],
                          A[static_cast<size_t>(pivot) * n + c]);
                std::swap(I[static_cast<size_t>(col) * n + c],
                          I[static_cast<size_t>(pivot) * n + c]);
            }
        }

        const double diag = A[static_cast<size_t>(col) * n + col];
        const double inv_diag = 1.0 / diag;
        for (int c = 0; c < n; ++c) {
            A[static_cast<size_t>(col) * n + c] *= inv_diag;
            I[static_cast<size_t>(col) * n + c] *= inv_diag;
        }

        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            const double factor = A[static_cast<size_t>(r) * n + col];
            if (factor == 0.0) continue;
            for (int c = 0; c < n; ++c) {
                A[static_cast<size_t>(r) * n + c] -= factor * A[static_cast<size_t>(col) * n + c];
                I[static_cast<size_t>(r) * n + c] -= factor * I[static_cast<size_t>(col) * n + c];
            }
        }
    }
    A.swap(I);
    return true;
}

int main(){
    const std::string cfg_path = "./tests/run_test/config.json";
    MDConfigManager cfg_mgr;
    cfg_mgr = cfg_mgr.config_from_json(cfg_path);

    MPI_Init(nullptr, nullptr);
    int rank_size = 0;
    int rank_idx = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_idx);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_size);

    // Optional lightweight smoke test for EnergyMinimizer
    if (std::getenv("RUN_MINIMIZER_SMOKE") != nullptr) {
        if (rank_size != 1) {
            if (rank_idx == 0) {
                fmt::print("[MinimizerSmoke] Skipping: run with a single MPI rank.\n");
            }
            MPI_Finalize();
            return 0;
        }

        MDConfig mini_cfg{};
        mini_cfg.rank_idx = 0;
        mini_cfg.rank_size = 1;
        mini_cfg.box_w_global = 10.0;
        mini_cfg.box_h_global = 10.0;
        mini_cfg.dt = 1e-3;
        mini_cfg.cutoff = 2.5;
        mini_cfg.SIGMA_AA = 1.0;
        mini_cfg.SIGMA_BB = 1.0;
        mini_cfg.SIGMA_AB = 1.0;
        mini_cfg.EPSILON_AA = 1.0;
        mini_cfg.EPSILON_BB = 1.0;
        mini_cfg.EPSILON_AB = 1.0;
        mini_cfg.MASS_A = 1.0;
        mini_cfg.MASS_B = 1.0;
        mini_cfg.THREADS_PER_BLOCK = 128;
        mini_cfg.n_particles_global = 2;
        mini_cfg.n_local = 2;
        mini_cfg.n_cap = 16;
        mini_cfg.halo_left_cap = 8;
        mini_cfg.halo_right_cap = 8;
        mini_cfg.x_min = 0.0;
        mini_cfg.x_max = mini_cfg.box_w_global;

        MDConfigManager mini_mgr(mini_cfg);
        EnergyMinimizer minimizer(mini_mgr, MPI_COMM_WORLD);

        std::vector<Particle> frame(2);
        frame[0].pos = {1.0, 1.0};
        frame[1].pos = {3.0, 1.0};
        frame[0].type = 0;
        frame[1].type = 0;
        frame[0].vel = frame[1].vel = make_double2(0.0, 0.0);
        frame[0].acc = frame[1].acc = make_double2(0.0, 0.0);

        FireParams params = FireParams::from_config(mini_cfg);
        auto result = minimizer.minimize_frame(frame, 1e-3, 200, params);

        if (rank_idx == 0) {
            fmt::print("[MinimizerSmoke] steps={} max_force={:.3e} converged={}\n",
                       result.steps, result.max_force, result.converged);
        }

        MPI_Finalize();
        return 0;
    }

    if (cfg_mgr.config.rank_size != rank_size){
        fmt::print("[Error] rank size = {} doesn't match config.\n", rank_size);
        return -1;
    }

    cfg_mgr.config.rank_size = rank_size;
    cfg_mgr.config.rank_idx = rank_idx;
    

    int n_record_interval = static_cast<int>(cfg_mgr.config.save_dt_interval/cfg_mgr.config.dt);
    if (n_record_interval <= 0) {
        n_record_interval = 1;
    }
    const int n_steps = 2000000;
    const int last_save_step = ((n_steps - 1) / n_record_interval) * n_record_interval;
    
    int n_grid_y  = static_cast<int>(cfg_mgr.config.box_h_global);

    std::string frame_dir = "./tests/run_test/frames/";
    std::string interface_dir = "./tests/run_test/interfaces/";
    std::string csv_dir = "./tests/run_test/csv/";
    std::string sample_csv_dir = "./tests/run_test/sample_csv/";
    std::string is_dir = "./tests/run_test/IS/";
    std::string is_csv_dir = is_dir + "csv/";
    std::string saved_env_dir = "./tests/run_test/saved_env/";
    std::string saved_env_file = saved_env_dir + "saved_env.bin";
    std::string saved_cfg_path = saved_env_dir + "config.json";

    if (!create_folder(frame_dir, rank_idx)) MPI_Abort(MPI_COMM_WORLD, 1);
    if (!create_folder(interface_dir, rank_idx)) MPI_Abort(MPI_COMM_WORLD, 1);
    if (!create_folder(csv_dir, rank_idx)) MPI_Abort(MPI_COMM_WORLD, 1);
    if (!create_folder(sample_csv_dir, rank_idx)) MPI_Abort(MPI_COMM_WORLD, 1);
    if (!create_folder(is_dir, rank_idx)) MPI_Abort(MPI_COMM_WORLD, 1);
    if (!create_folder(is_csv_dir, rank_idx)) MPI_Abort(MPI_COMM_WORLD, 1);
    if (!create_folder(saved_env_dir, rank_idx)) MPI_Abort(MPI_COMM_WORLD, 1);

    if (rank_idx == 0) {
        try {
            std::filesystem::copy_file(cfg_path, saved_cfg_path, std::filesystem::copy_options::overwrite_existing);
        } catch (const std::filesystem::filesystem_error& e) {
            fmt::print(stderr, "[Error] Failed to copy {} to {}: {}\n", cfg_path, saved_cfg_path, e.what());
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
    }

    std::string U_tot_csv_path = sample_csv_dir + "U_tot_log.csv";

    {
        MDSimulation sim(cfg_mgr, MPI_COMM_WORLD);

        RankZeroPrint(rank_idx, "[RANK] {} waiting for MPI_Barrier.\n", rank_idx);
        MPI_Barrier(MPI_COMM_WORLD);
        RankZeroPrint(rank_idx, "[RANK] {} complete for MPI_Barrier.\n", rank_idx);
        
        for (int step = 0; step < n_steps; step++){
            sim.step_single_nose_hoover();

            if (step % n_record_interval == 0) {
                sim.sample_collect();

                bool is_equilibrated = sim.check_eqlibrium(1.0);
                if (is_equilibrated){
                    RankZeroPrint(rank_idx, "Equilibrated. Exiting.\n");
                    break;
                }
                
                double U_tot = sim.cal_total_U();

                sim.RankZeroPrint("U_tot = {:.4e}\n", U_tot);
                sim.write_to_file(U_tot_csv_path, "U_tot, {}, step, {}\n", U_tot, step);

                sim.save_env(saved_env_file, step);

                // Inherent-structure (IS) check via FIRE minimization on rank 0
                if (rank_idx == 0) {
                    std::vector<Particle> is_frame;
                    sim.get_host_particles(is_frame);
                    if (!is_frame.empty()) {
                        MDConfig mini_cfg = cfg_mgr.config;
                        mini_cfg.rank_size = 1;
                        mini_cfg.rank_idx = 0;
                        mini_cfg.left_rank = 0;
                        mini_cfg.right_rank = 0;
                        mini_cfg.x_min = 0.0;
                        mini_cfg.x_max = mini_cfg.box_w_global;

                        const int frame_count = static_cast<int>(is_frame.size());
                        if (mini_cfg.n_cap <= 0 || mini_cfg.n_cap < frame_count) {
                            const int n_cap_mean = frame_count * 2 + 128;
                            mini_cfg.n_cap = std::max(n_cap_mean, frame_count);
                        }
                        const int halo_cap_default = std::max(mini_cfg.n_cap, 256);
                        if (mini_cfg.halo_left_cap <= 0 || mini_cfg.halo_left_cap < halo_cap_default) {
                            mini_cfg.halo_left_cap = halo_cap_default;
                        }
                        if (mini_cfg.halo_right_cap <= 0 || mini_cfg.halo_right_cap < halo_cap_default) {
                            mini_cfg.halo_right_cap = halo_cap_default;
                        }
                        mini_cfg.n_local = frame_count;
                        mini_cfg.n_halo_left = 0;
                        mini_cfg.n_halo_right = 0;

                        MDConfigManager mini_mgr(mini_cfg);
                        EnergyMinimizer minimizer(mini_mgr, MPI_COMM_SELF);

                        const double force_tol = 1e-3;
                        const int max_min_steps = 50000;
                        FireParams fire_params = FireParams::from_config(mini_cfg);

                        auto is_result = minimizer.minimize_frame(
                                is_frame, force_tol, max_min_steps, fire_params);

                        std::string is_csv_path =
                                is_csv_dir + fmt::format("is_frame_step_{}.csv", step);
                        std::string is_fig_path =
                                is_dir + fmt::format("is_frame_step_{}.svg", step);
                        minimizer.plot_particles(is_result, is_csv_path, is_fig_path);

                        // Write energy trace vs FIRE steps
                        const std::string is_energy_csv =
                                is_csv_dir + fmt::format("is_energy_step_{}.csv", step);
                        {
                            std::ofstream energy_out(is_energy_csv);
                            energy_out << "fire_step,energy\n";
                            const std::size_t n_samples = std::min(is_result.energy_steps.size(),
                                                                   is_result.energy_trace.size());
                            for (std::size_t k = 0; k < n_samples; ++k) {
                                energy_out << is_result.energy_steps[k] << ","
                                           << is_result.energy_trace[k] << "\n";
                            }
                            energy_out.flush();
                        }
                        // Plot energy curve using the Python helper (best-effort)
                        const std::string is_energy_fig =
                                is_dir + fmt::format("is_energy_step_{}.svg", step);
                        {
                            const std::string cmd =
                                    "~/.conda/envs/py3/bin/python ./python/plot_energy_trace.py "
                                    "--csv_path \"" + is_energy_csv + "\" "
                                    "--figure_path \"" + is_energy_fig + "\"";
                            int status = std::system(cmd.c_str());
                            if (status != 0) {
                                sim.RankZeroPrint("[Minimizer] energy plot failed (status {}).\n",
                                                  status);
                            }
                        }

                        sim.RankZeroPrint(
                                "[Minimizer] step {}: FIRE steps={} max|F|={:.3e} converged={} U={:.6e}\n",
                                step, is_result.steps, is_result.max_force,
                                is_result.converged, is_result.potential_energy);

                        // --- Hessian analysis on inherent structure (rank 0 only) ---
                        if (!is_result.frame.empty()) {
                            const double Lx = cfg_mgr.config.box_w_global;
                            const double Ly = cfg_mgr.config.box_h_global;
                            const double cutoff = cfg_mgr.config.cutoff;
                            const double sigma_AA = cfg_mgr.config.SIGMA_AA;
                            const double sigma_BB = cfg_mgr.config.SIGMA_BB;
                            const double sigma_AB = cfg_mgr.config.SIGMA_AB;
                            const double epsilon_AA = cfg_mgr.config.EPSILON_AA;
                            const double epsilon_BB = cfg_mgr.config.EPSILON_BB;
                            const double epsilon_AB = cfg_mgr.config.EPSILON_AB;

                            // Identify subsystem S: A-type particles participating in any A–B pair within cutoff*sigma_AB
                            std::unordered_set<int> subS_set;
                            const int n_particles = static_cast<int>(is_result.frame.size());
                            const double rc_ab = cutoff * sigma_AB;
                            const double rc_ab_sq = rc_ab * rc_ab;
                            for (int i = 0; i < n_particles; ++i) {
                                if (is_result.frame[i].type != 0) continue; // only A
                                for (int j = 0; j < n_particles; ++j) {
                                    if (is_result.frame[j].type != 1) continue; // B partner
                                    double dx = mic(is_result.frame[i].pos.x - is_result.frame[j].pos.x, Lx);
                                    double dy = mic(is_result.frame[i].pos.y - is_result.frame[j].pos.y, Ly);
                                    double dr2 = dx * dx + dy * dy;
                                    if (dr2 < rc_ab_sq) {
                                        subS_set.insert(i);
                                        break;
                                    }
                                }
                            }

                            std::vector<int> subS(subS_set.begin(), subS_set.end());
                            std::sort(subS.begin(), subS.end());
                            std::vector<int> subE;
                            subE.reserve(n_particles - subS.size());
                            for (int i = 0; i < n_particles; ++i) {
                                if (!subS_set.count(i)) subE.push_back(i);
                            }

                            const int dimS = static_cast<int>(subS.size()) * 2;
                            const int dimE = static_cast<int>(subE.size()) * 2;

                            if (dimS > 0) {
                                std::vector<double> H_SS, H_SE, H_ES, H_EE;
                                compute_hessian_LJ_blocks_host(
                                        is_result.frame, subS, subE, Lx, Ly, sigma_AA, sigma_BB,
                                        sigma_AB, epsilon_AA, epsilon_BB, epsilon_AB, cutoff, H_SS,
                                        H_SE, H_ES, H_EE);

                                std::vector<double> H_eff;
                                if (dimE == 0) {
                                    H_eff = H_SS; // no environment
                                } else {
                                    std::vector<double> H_EE_inv = H_EE;
                                    bool invert_ok = invert_in_place(H_EE_inv, dimE);
                                    if (!invert_ok) {
                                        sim.RankZeroPrint("[Hessian] H_EE inversion failed; skipping eigen solve.\n");
                                    } else {
                                        std::vector<double> tmp_SxE; // H_SE * H_EE_inv
                                        matmul(H_SE, dimS, dimE, H_EE_inv, dimE, tmp_SxE);
                                        std::vector<double> tmp_SxS;
                                        matmul(tmp_SxE, dimS, dimE, H_ES, dimS, tmp_SxS);

                                        H_eff.resize(H_SS.size(), 0.0);
                                        for (std::size_t idx = 0; idx < H_SS.size(); ++idx) {
                                            H_eff[idx] = H_SS[idx] - tmp_SxS[idx];
                                        }
                                    }
                                }

                                if (!H_eff.empty()) {
                                    const int dim = dimS;
                                    const int k = std::min(20, dim);
                                    if (dim > 0 && k > 0) {
                                        const int max_iter = 500;
                                        const int ncv = std::max(k * 2 + 1, 32);
                                        const double tol = 1e-8;
                                        const uint64_t seed = 12345ULL;

                                        std::vector<double> h_eigs;
                                        compute_smallest_eigs_dense_host(
                                                H_eff, dim, k, max_iter, ncv, tol, seed, h_eigs);

                                        const std::string eig_csv =
                                                is_csv_dir +
                                                fmt::format("is_hessian_eigs_step_{}.csv", step);
                                        std::ofstream eig_out(eig_csv);
                                        eig_out << "idx,eigenvalue\n";
                                        for (int i = 0; i < std::min(k, static_cast<int>(h_eigs.size())); ++i) {
                                            eig_out << i << "," << h_eigs[i] << "\n";
                                        }
                                        eig_out.flush();
                                        sim.RankZeroPrint("[Hessian] saved {} eigenvalues to {}\n",
                                                          h_eigs.size(), eig_csv);
                                    }
                                }
                            } else {
                                sim.RankZeroPrint("[Hessian] Subsystem S empty; skipping Hessian.\n");
                            }
                        }
                    }
                }

                // std::string frame_path = frame_dir + fmt::format("frame_step_{}.svg", step);
                // std::string csv_path = csv_dir + fmt::format("frame_step_{}.csv", step);

                std::string frame_triangulation_path = frame_dir + fmt::format("triangulation_frame_step_{}.svg", step);
                std::string csv_path_triangulation   = csv_dir + fmt::format("triangulation_frame_step_{}.csv", step);

                std::string frame_ab_network_path = frame_dir + fmt::format("ab_network_frame_step_{}.svg", step);
                std::string csv_path_ab_network   = csv_dir + fmt::format("ab_network_frame_step_{}.csv", step);

                // std::string frame_interface_path = interface_dir + fmt::format("interface_step_{}.svg", step);
                // std::string csv_path_interface = csv_dir + fmt::format("interface_step_{}.csv", step);

                // RankZeroPrint(rank_idx, "[Step] {}. plot_particles.\n", step);
                // sim.plot_particles(frame_path, csv_path);
                
                RankZeroPrint(rank_idx, "[Step] {}. triangulation_plot.\n", step);
                auto tri_result = sim.triangulation_plot(true, frame_triangulation_path, csv_path_triangulation);
                if (tri_result) {
                    auto tri_counts = sim.get_tri_types_num(*tri_result);
                    sim.RankZeroPrint("Triangle types (AAA, AAB, ABB, BBB): {}\n",
                                      tri_counts);
                    RankZeroPrint(rank_idx, "[Step] {}. plotting AB networks.\n", step);
                    auto ab_networks = sim.get_AB_pair_network(*tri_result, true, frame_ab_network_path,
                                            csv_path_ab_network);
                    double ab_length = sim.get_AB_pair_length(ab_networks);
                    sim.RankZeroPrint("Total AB pair length = {:.6f}\n", ab_length);
                }
                
                // RankZeroPrint(rank_idx, "[Step] {}. plot_interfaces.\n", step);
                // sim.plot_interfaces(frame_interface_path, csv_path_interface, density_profile);

                // RankZeroPrint(rank_idx, "[Step] {}. Frames saved.\n", step);
            }

            if (step % 100 == 0) {
                RankZeroPrint(rank_idx, "[Step] {}.\n", step);
            }
        }
    }

    // MPI_Barrier(MPI_COMM_WORLD);

    // RankZeroPrint(rank_idx, "[RANK] {} starting restored simulation via saved state.\n", rank_idx);

    // const int save_step = last_save_step;

    // MDConfigManager resumed_cfg = MDConfigManager::config_from_json(saved_cfg_path);
    // resumed_cfg.config.rank_size = rank_size;
    // resumed_cfg.config.rank_idx = rank_idx;
    // {
    //     MDSimulation resumed_sim(resumed_cfg, MPI_COMM_WORLD, saved_env_file, save_step);
    //     const int resume_steps = n_record_interval;
    //     for (int offset = 1; offset <= resume_steps; ++offset) {
    //         int resumed_step = save_step + offset;
    //         resumed_sim.step_single_nose_hoover();

    //         if (resumed_step % n_record_interval == 0) {
    //             resumed_sim.sample_collect();
    //             double resumed_U_tot = resumed_sim.cal_total_U();

    //             resumed_sim.RankZeroPrint("U_tot = {:.4e}\n", resumed_U_tot);
    //             resumed_sim.write_to_file(U_tot_csv_path, "U_tot, {}, step, {}\n", resumed_U_tot, resumed_step);
    //         }

    //         if (resumed_step % 100 == 0) {
    //             RankZeroPrint(rank_idx, "[Step] {} (restored).\n", resumed_step);
    //         }
    //     }
    // }

    return 0;
}
