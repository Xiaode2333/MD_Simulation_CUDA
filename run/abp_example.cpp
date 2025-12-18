// Example: Run Active Brownian Particle (ABP) simulation
// Demonstrates overdamped dynamics with self-propulsion

#include "md_env.hpp"
#include <mpi.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  try {
    // Load configuration from JSON file
    std::string config_file = (argc > 1) ? argv[1] : "config_abp.json";
    MDConfigManager cfg_manager =
        MDConfigManager::config_from_json(config_file);

    // Verify ABP mode is enabled
    if (cfg_manager.config.simulation_mode != "abp") {
      if (world_rank == 0) {
        fmt::print(
            stderr,
            "[ERROR] Configuration must have simulation_mode = \"abp\"\n");
      }
      MPI_Finalize();
      return 1;
    }

    // Create simulation
    MDSimulation sim(cfg_manager, MPI_COMM_WORLD);

    // Simulation parameters
    const double dt = cfg_manager.config.dt;
    const double total_time = 100.0; // Total simulation time
    const int n_steps = static_cast<int>(total_time / dt);
    const int save_interval =
        static_cast<int>(0.1 / dt); // Save every 0.1 time units

    if (world_rank == 0) {
      fmt::print("=== Active Brownian Particle Simulation ===\n");
      fmt::print("Parameters:\n");
      fmt::print("  Mobility (μ):     {}\n", cfg_manager.config.mu);
      fmt::print("  Self-propulsion (v0): {}\n", cfg_manager.config.v0);
      fmt::print("  Translational diffusion (D_r): {}\n",
                 cfg_manager.config.D_r);
      fmt::print("  Rotational diffusion (D_θ): {}\n",
                 cfg_manager.config.D_theta);
      fmt::print("  Timestep (dt):    {}\n", dt);
      fmt::print("  Total steps:      {}\n", n_steps);
      fmt::print("==========================================\n\n");
    }

    // Main simulation loop
    for (int step = 0; step < n_steps; ++step) {
      // Integrate one timestep using ABP dynamics
      sim.step_single_ABP();
      // Alternative: use automatic dispatcher
      // sim.step_single();

      // Save snapshot periodically
      if (step % save_interval == 0) {
        if (world_rank == 0) {
          fmt::print("Step {}/{} (t = {:.2f})\n", step, n_steps, sim.t);
        }

        std::string snapshot_file =
            fmt::format("output/abp_frame_{:06d}.bin", step / save_interval);
        sim.save_env(snapshot_file, step);
      }
    }

    if (world_rank == 0) {
      fmt::print("\nSimulation complete!\n");
      fmt::print("Final time: {:.2f}\n", sim.t);
    }

  } catch (const std::exception &e) {
    if (world_rank == 0) {
      fmt::print(stderr, "[ERROR] {}\n", e.what());
    }
    MPI_Finalize();
    return 1;
  }

  MPI_Finalize();
  return 0;
}
