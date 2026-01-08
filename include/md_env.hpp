#pragma once

#include "md_common.hpp"
#include "md_config.hpp"
#include "md_cuda_common.hpp"
#include "md_particle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <delaunator-header-only.hpp>
#include <deque>
#include <fmt/core.h>
#include <fstream>
#include <mpi.h>
#include <optional>
#include <random>
#include <string>
#include <thrust/device_vector.h>
#include <vector>

// Triangulation data on rank 0: expanded PBC vertices + triangle connectivity.
struct TriangulationResult {
    // Flattened (x,y) coordinates used for triangulation (may include PBC
    // images).
    std::vector<double> coords;
    // For each vertex in coords, index of the corresponding original particle in
    // h_particles.
    std::vector<int> vertex_to_idx;
    // Triangle connectivity as indices into coords / vertex_to_idx.
    std::vector<std::array<int, 3>> triangles;
};

// AB pair “network” built from mixed A/B triangles: midpoints as nodes,
// segments as edges.
struct ABPairNetworks {
    struct Node {
        double x;
        double y;
    };
    struct Edge {
        int node0;
        int node1;
    };

    // Each entry is one connected network of midpoints and edges.
    std::vector<std::vector<Node>> networks_nodes;
    std::vector<std::vector<Edge>> networks_edges;
};

class MDSimulation {
public:
    MDSimulation(class MDConfigManager config_manager, MPI_Comm comm);
    MDSimulation(MDConfigManager config_manager, MPI_Comm comm,
                             const std::string &filename,
                             int step); // constructor from saved file
    ~MDSimulation();

    double t = 0.0; // evolution time

    void save_env(const std::string &filename, const int step);

    void plot_particles(const std::string &filename, const std::string &csv_path);

    double cal_total_K();
    double cal_total_U();
    double cal_partial_U_lambda(double epsilon_lambda);
    double deform(double epsilon, double U_old);
    double get_Lx() const { return cfg_manager.config.box_w_global; }
    double get_Ly() const { return cfg_manager.config.box_h_global; }
    // Confine all particles to a central slab along x for LG systems.
    // The slab occupies a fraction devide_p of the box width (plus a small
    // buffer) and is centered at Lx / 2. Reflective boundaries are used in x,
    // while y retains standard PBC. Operates on device particles via a CUDA
    // kernel.
    void middle_reflect_LG();

    void step_single_NVE();
    void step_single_nose_hoover(bool do_middle_wrap = false);
    void step_single_ABP(bool do_middle_wrap = false); // ABP overdamped dynamics with self-propulsion

    bool check_eqlibrium(double sensitivity);

    void sample_collect();

    std::optional<TriangulationResult>
    triangulation_plot(bool is_plot, const std::string &filename,
                                         const std::string &csv_path);

    // Rank 0 only: build A–B mid-segment networks from a triangulation (mixed A/B
    // triangles → midpoint graphs). When is_plot is true, also emit a CSV + image
    // for the current frame and network.
    ABPairNetworks get_AB_pair_network(const TriangulationResult &tri,
                                                                           bool is_plot = false,
                                                                           const std::string &filename = "",
                                                                           const std::string &csv_path = "") const;

    std::vector<std::vector<double>>
    locate_interface(const delaunator::Delaunator &d);
    // Returns interface polylines as {x0, y0, x1, y1, ...} for each interface,
    // empty if rank != 0 or none found
    void do_CWA_instant(int q_min, int q_max, const std::string &csv_path,
                                            const std::string &plot_path, bool is_plot, int step,
                                            bool is_LG = false);

    void plot_interfaces(const std::string &filename, const std::string &csv_path,
                                             const std::vector<double> &rho, bool is_LG = false);
    // Rank 0 only: compute total interface length L_tot = L1 + L2
    // using the same grid-based interface locator as plot_interfaces.
    // Returns 0.0 on non-root ranks or if no interface is detected.
    double get_interface_total_length(bool is_LG = false);

    // On rank 0, returns 3 * n_bins_local entries:
    //   result[k]                  = P_xx(y_k)
    //   result[k +   n_bins_local] = P_yy(y_k)
    //   result[k + 2*n_bins_local] = P_xy(y_k),
    // already normalized by Lx * Δy; other ranks return an empty vector.
    std::vector<double> get_pressure_profile(int n_bins_local);
    std::vector<int>
    get_N_profile(int n_bins_per_rank); // number of particles per bin
    std::vector<double> get_density_profile(int n_bins_per_rank);

    template <typename... Args>
    void RankZeroPrint(fmt::format_string<Args...> format_str, Args &&...args) {
        if (cfg_manager.config.rank_idx == 0) {
            fmt::print(format_str, std::forward<Args>(args)...);
            std::fflush(stdout);
        }
    }

    template <typename... Args>
    bool write_to_file(const std::string &filename,
                                         fmt::format_string<Args...> format_str, Args &&...args) {
        if (cfg_manager.config.rank_idx != 0) {
            return true;
        }
        std::ofstream out(filename, std::ios::out | std::ios::app);
        if (!out) {
            fmt::print(stderr, "[Error] Failed to open {} for writing.\n", filename);
            return false;
        }
        out << fmt::format(format_str, std::forward<Args>(args)...);
        out.flush();
        return true;
    }

private:
    MDConfigManager cfg_manager;
    MPI_Comm comm;

    std::unique_ptr<FileWriter> particle_writer;
    std::unique_ptr<FileReader> particle_reader;

    std::vector<double> coords; // For triangulation
    std::vector<int> vertex_to_idx;

    std::vector<Particle> h_particles; // typically all data is on device. When
                                                                         // sampling first transfer them to host
    std::vector<Particle> h_particles_local;
    std::vector<Particle> h_particles_halo_left;
    std::vector<Particle> h_particles_halo_right;
    std::vector<Particle> h_send_left;
    std::vector<Particle> h_send_right;
    thrust::device_vector<Particle> d_particles;
    thrust::device_vector<Particle> d_particles_halo_left;
    thrust::device_vector<Particle> d_particles_halo_right;

    thrust::device_vector<int> flags_left;
    thrust::device_vector<int> flags_right;
    thrust::device_vector<int> flags_keep;
    thrust::device_vector<int> pos_left;
    thrust::device_vector<int> pos_right;
    thrust::device_vector<int> pos_keep;
    thrust::device_vector<Particle> d_send_left;
    thrust::device_vector<Particle> d_send_right;
    thrust::device_vector<Particle> d_keep;

    // ABP: cuRAND RNG states (one per particle)
    curandState *d_rng_states = nullptr;
    bool abp_rng_initialized = false;

    double xi; // xi for nose hoover
    double Lx0;
    double Ly0;

    double record_interval_dt;
    double next_record_time;
    std::deque<double> energy_history;
    std::size_t energy_window_sample_count;
    std::size_t energy_history_capacity;
    int equilibrium_window_streak;

    static constexpr double kEquilibriumWindowTime = 100.0;
    static constexpr int kBaseRequiredPasses = 3;
    static constexpr double kBasePValue = 0.2;

    void broadcast_params();
    void allocate_memory();
    void init_particles(); // Only update h_particles on rank 0
    double compute_kinetic_energy_local();
    void distribute_particles_h2d();
    void collect_particles_d2h(); // collect particles from device to host, only
                                                                // to host of rank 0
    void update_halo();           // suppose d_particles is already updated
    void update_d_particles(); // use only d_particles to update particles through
                                                         // particle exchange between ranks
    void cal_forces();         // update force and store into d_particles
    double compute_U_energy_local();
    std::vector<std::vector<double>>
    compute_interface_paths(int n_grid_y, double smoothing_sigma);
    std::vector<std::vector<double>>
    compute_interface_paths_LG(int n_grid_y, double smoothing_sigma);
    std::vector<std::vector<double>> get_smooth_interface(int n_grid_y,
                                                                                                                double smoothing_sigma);
    std::vector<std::vector<double>>
    get_smooth_interface_LG(int n_grid_y, double smoothing_sigma);

    void init_equilibrium_tracker();
    void append_energy_sample(double U);
    bool evaluate_equilibrium(double normalized_sensitivity);
    double compute_window_relative_change(std::size_t start_idx) const;
    struct WindowStats {
        double mean;
        double variance;
    };
    WindowStats compute_window_stats(std::size_t start_idx) const;
    double normal_tail_probability(double z) const;
};
