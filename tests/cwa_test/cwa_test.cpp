#include "md_env.hpp"
#include <fmt/ranges.h>
#include <filesystem>
#include <fstream>
#include <cstdio>

namespace {
template <typename... Args>
void RankZeroPrint(int rank_idx, fmt::format_string<Args...> format_str, Args&&... args) {
    if (rank_idx == 0) {
        fmt::print(format_str, std::forward<Args>(args)...);
        std::fflush(stdout);
    }
}

bool create_folder(const std::string& path, int rank_idx) {
    if (rank_idx != 0) {
        return true;
    }

    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        std::filesystem::create_directories(path, ec);
        if (ec) {
            fmt::print(stderr, "Failed to create dir {}. Error: {}\n", path, ec.message());
            return false;
        }
    }
    return true;
}

bool append_latest_line(const std::string& src, const std::string& dst, int rank_idx) {
    if (rank_idx != 0) return true;

    std::ifstream in(src);
    if (!in) {
        fmt::print(stderr, "[CWA Test] Failed to open {} for reading.\n", src);
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
        fmt::print(stderr, "[CWA Test] No data found in {}.\n", src);
        return false;
    }

    std::ofstream out(dst, std::ios::out | std::ios::app);
    if (!out) {
        fmt::print(stderr, "[CWA Test] Failed to open {} for appending.\n", dst);
        return false;
    }

    out << last_non_empty << '\n';
    return true;
}
} // namespace

int main(){
    const std::string cfg_path = "./tests/run_test/saved_env_T_0.9/config.json";
    MDConfigManager cfg_mgr;
    cfg_mgr = cfg_mgr.config_from_json(cfg_path);
    cfg_mgr.config.save_dt_interval = 1.0;

    MPI_Init(nullptr, nullptr);
    int rank_size = 0;
    int rank_idx = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_idx);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_size);

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
    const int n_steps = 500000;
    const int q_min = 3;
    const int q_max = 20;

    std::string cwa_plot_dir = "./tests/cwa_test/cwa_plot/";
    std::string cwa_plot_csv_dir = cwa_plot_dir + "csv/";
    std::string cwa_sample_dir = "./tests/cwa_test/sample_csv/";
    std::string cwa_sample_csv = cwa_sample_dir + "cwa_instant.csv";
    std::string U_tot_csv_path = cwa_sample_dir + "U_tot_log.csv";

    if (!create_folder(cwa_plot_dir, rank_idx)) MPI_Abort(MPI_COMM_WORLD, 1);
    if (!create_folder(cwa_plot_csv_dir, rank_idx)) MPI_Abort(MPI_COMM_WORLD, 1);
    if (!create_folder(cwa_sample_dir, rank_idx)) MPI_Abort(MPI_COMM_WORLD, 1);

    if (rank_idx == 0) {
        std::ofstream clear_cwa(cwa_sample_csv, std::ios::out | std::ios::trunc);
        std::ofstream clear_utot(U_tot_csv_path, std::ios::out | std::ios::trunc);
        (void)clear_cwa;
        (void)clear_utot;
    }

    const std::string saved_env_path = "./tests/run_test/saved_env_T_0.9/saved_env.bin";
    const int saved_step = 450000;

    {
        MDSimulation sim(cfg_mgr, MPI_COMM_WORLD, saved_env_path, saved_step);

        RankZeroPrint(rank_idx, "[RANK] {} waiting for MPI_Barrier.\n", rank_idx);
        MPI_Barrier(MPI_COMM_WORLD);
        RankZeroPrint(rank_idx, "[RANK] {} complete for MPI_Barrier.\n", rank_idx);
        
        for (int step = 0; step < n_steps; step++){
            sim.step_single_nose_hoover();

            if (step % n_record_interval == 0) {
                sim.sample_collect();
                double U_tot = sim.cal_total_U();

                sim.RankZeroPrint("U_tot = {:.4e}\n", U_tot);
                sim.write_to_file(U_tot_csv_path, "U_tot, {}, step, {}\n", U_tot, step);

                std::string cwa_step_csv = cwa_plot_csv_dir + fmt::format("cwa_instant_{}.csv", step);
                std::string cwa_step_plot = cwa_plot_dir + fmt::format("cwa_instant_{}.svg", step);
                sim.do_CWA_instant(q_min, q_max, cwa_step_csv, cwa_step_plot, true, step);
                if (rank_idx == 0) {
                    std::error_code ec;
                    const auto file_sz = std::filesystem::file_size(cwa_step_csv, ec);
                    if (!ec && file_sz > 0) {
                        append_latest_line(cwa_step_csv, cwa_sample_csv, rank_idx);
                    } else {
                        fmt::print("[CWA Test] Skipping append for step {} (no data).\n", step);
                    }
                }
            }

            if (step % 100 == 0) {
                RankZeroPrint(rank_idx, "[Step] {}.\n", step);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
