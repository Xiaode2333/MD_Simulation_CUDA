#pragma once

#include "md_config.hpp"
#include "md_particle.hpp"

#include <mpi.h>
#include <thrust/device_vector.h>
#include <string>
#include <vector>

struct FireParams {
    double dt_init = 0.0;
    double dt_max = 0.0;
    double alpha0 = 0.1;
    double f_inc = 1.1;
    double f_dec = 0.5;
    int N_min = 5;

    static FireParams from_config(const MDConfig &cfg) {
        FireParams params;
        params.dt_init = cfg.dt;
        params.dt_max = 5.0 * cfg.dt;
        return params;
    }
};

struct MinimizeResult {
    std::vector<Particle> frame;
    std::vector<int> energy_steps;     // FIRE step indices where energy sampled
    std::vector<double> energy_trace;  // potential energy samples aligned with energy_steps
    int steps = 0;
    bool converged = false;
    double max_force = 0.0;
    double potential_energy = 0.0;
};

class EnergyMinimizer {
public:
    EnergyMinimizer(MDConfigManager config_manager, MPI_Comm comm);

    MinimizeResult
    minimize_frame(const std::vector<Particle> &frame, double force_tol,
                   int max_steps, const FireParams &params);

    void sample_collect(std::vector<Particle> &host_out,
                        bool gather_root = true);
    void plot_particles(const MinimizeResult &result,
                        const std::string &csv_filename,
                        const std::string &figure_filename);

private:
    struct ForceStats {
        double max_force = 0.0;
        double power = 0.0;
        double v_norm = 0.0;
        double f_norm = 0.0;
    };

    void initialize_comm_metadata();
    void allocate_buffers();
    void initialize_from_frame(const std::vector<Particle> &frame);
    void compute_forces();
    ForceStats reduce_force_metrics();
    void fire_velocity_mix(double alpha);
    void drift_positions(double dt);
    void exchange_particles();
    double compute_potential();

    MDConfigManager cfg_manager;
    MPI_Comm comm;

    std::vector<Particle> h_input_frame; // preserved copy of caller frame
    std::vector<Particle> h_particles_local;
    std::vector<Particle> h_particles_halo_left;
    std::vector<Particle> h_particles_halo_right;
    std::vector<Particle> h_send_left;
    std::vector<Particle> h_send_right;

    thrust::device_vector<Particle> d_particles;
    thrust::device_vector<Particle> d_particles_halo_left;
    thrust::device_vector<Particle> d_particles_halo_right;
    thrust::device_vector<Particle> d_send_left;
    thrust::device_vector<Particle> d_send_right;
    thrust::device_vector<Particle> d_keep;

    thrust::device_vector<int> flags_left;
    thrust::device_vector<int> flags_right;
    thrust::device_vector<int> flags_keep;
    thrust::device_vector<int> pos_left;
    thrust::device_vector<int> pos_right;
    thrust::device_vector<int> pos_keep;

    double current_dt = 0.0;
    double current_alpha = 0.0;
    int positive_streak = 0;
    bool buffers_ready = false;
};
