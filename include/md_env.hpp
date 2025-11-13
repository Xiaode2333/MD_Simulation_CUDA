#pragma once

#include "include/md_particle.hpp"
#include "include/md_config.hpp"

#include <string>
#include <vector>




class MDSimulation {
    public:
        MDSimulation(class MDConfigManager config_manager);

        ~MDSimulation();
        
        void run(const double run_time);

        void equilibrate(const double equil_time);

        void equilibrate_T_ramp(const double ramp_time, const double extra_time);

        double capillary_wave(int mode_min, int mode_max, const std::string& plot_path);

        double statistical_capillary_wave(int mode_min,
                                    int mode_max,
                                    const std::string& plot_path,
                                    int frame_min,
                                    int frame_max,
                                    const std::string& csv_path,
                                    const std::string& data_save_path);
        
        void save_to_file_binary_gz(const std::string& filepath, int step, bool append=false);

        void save(int step);

        bool load(int frame_count, const std::string& filepath);

        void plot(const std::string& output_filename);

        double measure_interface_length(bool plot, const std::string& plot_filename);

        double measure_interface_length_impl(bool plot, const std::string& plot_filename);

    private:
        std::string output_dir_base;
        std::string output_filepath;
        std::vector<int> counts;
        std::vector<int> offsets;
        int N_PARTICLES_TOTAL;
        int N_PARTICLES_TYPE0;
        double MASS_TYPE0 = 1.0;
        double MASS_TYPE1 = 1.0;
        double BOX_WIDTH, BOX_HEIGHT;
        const double TEMP_INIT;
        // double zeta = 0.0;                
        // const double Q = 100.0;       
        NHState nh_state; // {zeta, Q, target_kinetic_E, dt}
        const double TARGET_TEMP;       
        int DEGREES_OF_FREEDOM = 2 * N_PARTICLES_TOTAL - 2;
        double target_kinetic_energy;
        double ramp_kinetic_energy;
        const double SAVE_DT_INTERVAL = 0.1;
        const double DT_INITIAL = 0.001;
        const double DT_MIN = 1e-9;
        const double DT_MAX = 1e-3;
        const int N_STEPS = 10000000;
        const int OUTPUT_FREQ = 1000;
        const int THREADS_PER_BLOCK = 256;
        const double REL_TOL = 1e-6;
        const double ABS_TOL = 1e-9;
        const double SAFETY_FACTOR = 0.99;
        const double SIGMA_AA;
        const double SIGMA_BB;
        const double SIGMA_AB;
        const double EPSILON_AA;
        const double EPSILON_BB;
        const double EPSILON_AB;

        int n_gpus, particles_per_gpu;
        double dt;
        double sim_time = 0.0;

        std::vector<Particle*> d_particles;           // current y_n (pos, vel, acc)
        std::vector<Particle*> d_particles_stage_in;  // global scratch for broadcasting
        std::vector<Particle*> d_particles_stage_out; // mid state (r_{n+1}, v_half) or scratch
        std::vector<Particle*> d_particles_y_final;   // 1 full step result
        std::vector<Particle*> d_particles_y_hat_final; // 2 half-steps result
        std::vector<double*>    d_error_sq;
        std::vector<cudaStream_t> streams;
        std::vector<Particle> h_particles;
        std::vector<double*> d_block_sums;
        std::vector<double*> d_U_sums;
        std::vector<double*> d_K_sums;

        void allocateMemory();

        void initParticles(const double p);
        
        void cleanup();

        double calculateTotalKineticEnergy();

        void load_config();

        void broadcastAndCalculateForces(
            Particle** d_locals,        // per-GPU: local slice used as "i"
            Particle** d_globals,       // per-GPU: full system buffer used as "j"
            Particle** d_acc_out,       // per-GPU: write accelerations for local slice
            int* counts,                // per-GPU local counts
            int* starts,                // per-GPU local start indices in global order
            int  n_gpus,
            double BOX_WIDTH, double BOX_HEIGHT,
            double mass0, double mass1,
            double SIGMA_AA, double EPSILON_AA,
            double SIGMA_BB, double EPSILON_BB,
            double SIGMA_AB, double EPSILON_AB,
            const std::string& ensemble, NHPhase phase, NHState& nh,
            std::vector<double*>& d_U_sums, std::vector<double*>& d_K_sums,
            std::vector<cudaStream_t>& streams
        );

        void performTimestep(const std::string& ensemble, const double t_stop = -1.0, bool* trigger_stop = nullptr);


}