// md_config.cpp
#include "../include/md_config.hpp"


MDConfigManager::MDConfigManager(struct MDConfig config) : config(config) {

}

MDConfigManager::~MDConfigManager() {}

void MDConfigManager::print_config() {
    // Using fmt::print for direct output to stdout
    fmt::print(
        "MDConfig:\n"
        "n_particles: {}\n"
        "n_particles_type0: {}\n"
        "box_w: {}\n"
        "box_h: {}\n"
        "T_init: {}\n"
        "T_target: {}\n"
        "SIGMA_AA: {}\n"
        "SIGMA_BB: {}\n"
        "SIGMA_AB: {}\n"
        "EPSILON_AA: {}\n"
        "EPSILON_BB: {}\n"
        "EPSILON_AB: {}\n"
        "MASS_A: {}\n"
        "MASS_B: {}\n"
        "devide_p: {}\n"
        "dt: {}\n"
        "Q: {}\n"
        "save_dt_interval: {}\n"
        "run_name: {}\n"
        "load_name: {}\n"
        "mpi_world_size: {}\n"
        "THREADS_PER_BLOCK: {}\n",
        config.n_particles,
        config.n_particles_type0,
        config.box_w,
        config.box_h,
        config.T_init,
        config.T_target,
        config.SIGMA_AA,
        config.SIGMA_BB,
        config.SIGMA_AB,
        config.EPSILON_AA,
        config.EPSILON_BB,
        config.EPSILON_AB,
        config.MASS_A,
        config.MASS_B,
        config.devide_p,
        config.dt,
        config.Q,
        config.save_dt_interval,
        config.run_name,
        config.load_name,
        config.mpi_world_size,
        config.THREADS_PER_BLOCK,
    );
}

// load MDConfig from a JSON file
MDConfigManager MDConfigManager::config_from_json(const std::string& filepath)
{
    std::ifstream in(filepath);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open config file: " + filepath);
    }

    json j;
    in >> j;

    // Start from struct defaults (value-initialized, so in-class defaults are kept)
    MDConfig cfg{};

    cfg.n_particles        = j.value("n_particles",        cfg.n_particles);
    cfg.n_particles_type0  = j.value("n_particles_type0",  cfg.n_particles_type0);
    cfg.box_w              = j.value("box_w",              cfg.box_w);
    cfg.box_h              = j.value("box_h",              cfg.box_h);
    cfg.T_init             = j.value("T_init",             cfg.T_init);
    cfg.T_target           = j.value("T_target",           cfg.T_target);

    cfg.SIGMA_AA           = j.value("SIGMA_AA",           cfg.SIGMA_AA);
    cfg.SIGMA_BB           = j.value("SIGMA_BB",           cfg.SIGMA_BB);
    cfg.SIGMA_AB           = j.value("SIGMA_AB",           cfg.SIGMA_AB);

    cfg.EPSILON_AA         = j.value("EPSILON_AA",         cfg.EPSILON_AA);
    cfg.EPSILON_BB         = j.value("EPSILON_BB",         cfg.EPSILON_BB);
    cfg.EPSILON_AB         = j.value("EPSILON_AB",         cfg.EPSILON_AB);

    cfg.MASS_A             = j.value("MASS_A",             cfg.MASS_A);
    cfg.MASS_B             = j.value("MASS_B",             cfg.MASS_B);

    cfg.devide_p           = j.value("devide_p",           cfg.devide_p);
    cfg.dt                 = j.value("dt",                 cfg.dt);
    cfg.Q                  = j.value("Q",                  cfg.Q);
    cfg.save_dt_interval   = j.value("save_dt_interval",   cfg.save_dt_interval);

    cfg.run_name           = j.value("run_name",           cfg.run_name);
    cfg.load_name          = j.value("load_name",          cfg.load_name);

    cfg.mpi_world_size     = j.value("mpi_world_size",     cfg.mpi_world_size);
    cfg.THREADS_PER_BLOCK  = j.value("THREADS_PER_BLOCK",  cfg.THREADS_PER_BLOCK);

    return MDConfigManager(cfg);
}

//save MDConfig to a JSON file
void MDConfigManager::config_to_json(const std::string& filepath)
{
    json j;

    j["n_particles"]        = config.n_particles;
    j["n_particles_type0"]  = config.n_particles_type0;
    j["box_w"]              = config.box_w;
    j["box_h"]              = config.box_h;
    j["T_init"]             = config.T_init;
    j["T_target"]           = config.T_target;

    j["SIGMA_AA"]           = config.SIGMA_AA;
    j["SIGMA_BB"]           = config.SIGMA_BB;
    j["SIGMA_AB"]           = config.SIGMA_AB;

    j["EPSILON_AA"]         = config.EPSILON_AA;
    j["EPSILON_BB"]         = config.EPSILON_BB;
    j["EPSILON_AB"]         = config.EPSILON_AB;

    j["MASS_A"]             = config.MASS_A;
    j["MASS_B"]             = config.MASS_B;

    j["devide_p"]           = config.devide_p;
    j["dt"]                 = config.dt;
    j["Q"]                  = config.Q;
    j["save_dt_interval"]   = config.save_dt_interval;

    j["run_name"]           = config.run_name;
    j["load_name"]          = config.load_name;

    j["mpi_world_size"]     = config.mpi_world_size;
    j["THREADS_PER_BLOCK"]  = config.THREADS_PER_BLOCK;

    std::ofstream out(filepath);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open config file for writing: " + filepath);
    }

    out << j.dump(4) << std::endl;
}