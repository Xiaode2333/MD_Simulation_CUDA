// md_config.cpp
#include "md_config.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <type_traits>

namespace {
template <typename T> bool parse_scalar(const std::string &text, T &out) {
    std::istringstream iss(text);
    iss >> out;
    if (iss.fail()) {
        return false;
    }
    iss >> std::ws;
    return iss.eof();
}

template <>
bool parse_scalar<std::string>(const std::string &text, std::string &out) {
    out = text;
    return true;
}

template <> bool parse_scalar<bool>(const std::string &text, bool &out) {
    std::string lowered = text;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lowered == "1" || lowered == "true" || lowered == "yes" ||
        lowered == "on") {
        out = true;
        return true;
    }
    if (lowered == "0" || lowered == "false" || lowered == "no" ||
        lowered == "off") {
        out = false;
        return true;
    }
    return false;
}

void load_from_json_object(const json &j, MDConfig &cfg) {
    // Start from struct defaults (value-initialized, so in-class defaults are
    // kept)
    MDConfig defaults{};

    // Global params
    cfg.n_particles_global =
            j.value("n_particles_global", defaults.n_particles_global);
    cfg.n_particles_type0 =
            j.value("n_particles_type0", defaults.n_particles_type0);
    cfg.box_w_global = j.value("box_w_global", defaults.box_w_global);
    cfg.box_h_global = j.value("box_h_global", defaults.box_h_global);
    cfg.T_init = j.value("T_init", defaults.T_init);
    cfg.T_target = j.value("T_target", defaults.T_target);

    cfg.SIGMA_AA = j.value("SIGMA_AA", defaults.SIGMA_AA);
    cfg.SIGMA_BB = j.value("SIGMA_BB", defaults.SIGMA_BB);
    cfg.SIGMA_AB = j.value("SIGMA_AB", defaults.SIGMA_AB);

    cfg.EPSILON_AA = j.value("EPSILON_AA", defaults.EPSILON_AA);
    cfg.EPSILON_BB = j.value("EPSILON_BB", defaults.EPSILON_BB);
    cfg.EPSILON_AB = j.value("EPSILON_AB", defaults.EPSILON_AB);

    cfg.MASS_A = j.value("MASS_A", defaults.MASS_A);
    cfg.MASS_B = j.value("MASS_B", defaults.MASS_B);

    cfg.devide_p = j.value("devide_p", defaults.devide_p);
    cfg.dt = j.value("dt", defaults.dt);
    cfg.Q = j.value("Q", defaults.Q);
    cfg.save_dt_interval = j.value("save_dt_interval", defaults.save_dt_interval);
    cfg.cutoff = j.value("cutoff", defaults.cutoff);

    cfg.run_name = j.value("run_name", defaults.run_name);
    cfg.load_name = j.value("load_name", defaults.load_name);

    cfg.THREADS_PER_BLOCK =
            j.value("THREADS_PER_BLOCK", defaults.THREADS_PER_BLOCK);
    cfg.rank_size = j.value("rank_size", defaults.rank_size);

    // ABP parameters
    cfg.mu = j.value("mu", defaults.mu);
    cfg.v0 = j.value("v0", defaults.v0);
    cfg.D_r = j.value("D_r", defaults.D_r);
    cfg.D_theta = j.value("D_theta", defaults.D_theta);

    // Periodic-box NPH barostat controls
    cfg.barostat_mass = j.value("barostat_mass", defaults.barostat_mass);
    cfg.k_piston = j.value("k_piston", defaults.k_piston);
    cfg.barostat_area_rate_init =
            j.value("barostat_area_rate_init", defaults.barostat_area_rate_init);
    cfg.barostat_height_min_ratio = j.value(
            "barostat_height_min_ratio", defaults.barostat_height_min_ratio);
    cfg.barostat_height_max_ratio = j.value(
            "barostat_height_max_ratio", defaults.barostat_height_max_ratio);
    cfg.pressure_target_mode =
            j.value("pressure_target_mode", defaults.pressure_target_mode);
    cfg.P_target = j.value("P_target", defaults.P_target);
    cfg.pressure_target_auto_steps =
            j.value("pressure_target_auto_steps", defaults.pressure_target_auto_steps);
    cfg.pressure_eval_interval_steps =
            j.value("pressure_eval_interval_steps",
                    defaults.pressure_eval_interval_steps);
    cfg.pressure_window_samples =
            j.value("pressure_window_samples", defaults.pressure_window_samples);
    cfg.pressure_tolerance_rel =
            j.value("pressure_tolerance_rel", defaults.pressure_tolerance_rel);
    cfg.pressure_stable_required_windows = j.value(
            "pressure_stable_required_windows",
            defaults.pressure_stable_required_windows);
    cfg.nph_warmup_min_steps =
            j.value("nph_warmup_min_steps", defaults.nph_warmup_min_steps);
    cfg.nph_warmup_max_steps =
            j.value("nph_warmup_max_steps", defaults.nph_warmup_max_steps);
    cfg.nph_steps = j.value("nph_steps", defaults.nph_steps);

    // This rank params (optional in JSON, so they default to struct values)
    cfg.rank_idx = j.value("rank_idx", defaults.rank_idx);
    cfg.n_local = j.value("n_local", defaults.n_local);
    cfg.n_halo_left = j.value("n_halo_left", defaults.n_halo_left);
    cfg.n_halo_right = j.value("n_halo_right", defaults.n_halo_right);
    cfg.n_cap = j.value("n_cap", defaults.n_cap);
    cfg.halo_left_cap = j.value("halo_left_cap", defaults.halo_left_cap);
    cfg.halo_right_cap = j.value("halo_right_cap", defaults.halo_right_cap);
    cfg.left_rank = j.value("left_rank", defaults.left_rank);
    cfg.right_rank = j.value("right_rank", defaults.right_rank);
    cfg.x_min = j.value("x_min", defaults.x_min);
    cfg.x_max = j.value("x_max", defaults.x_max);
}

void store_to_json_object(json &j, const MDConfig &cfg) {
    // Global params
    j["n_particles_global"] = cfg.n_particles_global;
    j["n_particles_type0"] = cfg.n_particles_type0;
    j["box_w_global"] = cfg.box_w_global;
    j["box_h_global"] = cfg.box_h_global;
    j["T_init"] = cfg.T_init;
    j["T_target"] = cfg.T_target;

    j["SIGMA_AA"] = cfg.SIGMA_AA;
    j["SIGMA_BB"] = cfg.SIGMA_BB;
    j["SIGMA_AB"] = cfg.SIGMA_AB;

    j["EPSILON_AA"] = cfg.EPSILON_AA;
    j["EPSILON_BB"] = cfg.EPSILON_BB;
    j["EPSILON_AB"] = cfg.EPSILON_AB;

    j["MASS_A"] = cfg.MASS_A;
    j["MASS_B"] = cfg.MASS_B;

    j["devide_p"] = cfg.devide_p;
    j["dt"] = cfg.dt;
    j["Q"] = cfg.Q;
    j["save_dt_interval"] = cfg.save_dt_interval;
    j["cutoff"] = cfg.cutoff;

    j["run_name"] = cfg.run_name;
    j["load_name"] = cfg.load_name;

    j["THREADS_PER_BLOCK"] = cfg.THREADS_PER_BLOCK;
    j["rank_size"] = cfg.rank_size;

    // ABP parameters
    j["mu"] = cfg.mu;
    j["v0"] = cfg.v0;
    j["D_r"] = cfg.D_r;
    j["D_theta"] = cfg.D_theta;

    // Periodic-box NPH barostat controls
    j["barostat_mass"] = cfg.barostat_mass;
    j["k_piston"] = cfg.k_piston;
    j["barostat_area_rate_init"] = cfg.barostat_area_rate_init;
    j["barostat_height_min_ratio"] = cfg.barostat_height_min_ratio;
    j["barostat_height_max_ratio"] = cfg.barostat_height_max_ratio;
    j["pressure_target_mode"] = cfg.pressure_target_mode;
    j["P_target"] = cfg.P_target;
    j["pressure_target_auto_steps"] = cfg.pressure_target_auto_steps;
    j["pressure_eval_interval_steps"] = cfg.pressure_eval_interval_steps;
    j["pressure_window_samples"] = cfg.pressure_window_samples;
    j["pressure_tolerance_rel"] = cfg.pressure_tolerance_rel;
    j["pressure_stable_required_windows"] = cfg.pressure_stable_required_windows;
    j["nph_warmup_min_steps"] = cfg.nph_warmup_min_steps;
    j["nph_warmup_max_steps"] = cfg.nph_warmup_max_steps;
    j["nph_steps"] = cfg.nph_steps;

    // This rank params
    j["rank_idx"] = cfg.rank_idx;
    j["n_local"] = cfg.n_local;
    j["n_halo_left"] = cfg.n_halo_left;
    j["n_halo_right"] = cfg.n_halo_right;
    j["n_cap"] = cfg.n_cap;
    j["halo_left_cap"] = cfg.halo_left_cap;
    j["halo_right_cap"] = cfg.halo_right_cap;
    j["left_rank"] = cfg.left_rank;
    j["right_rank"] = cfg.right_rank;
    j["x_min"] = cfg.x_min;
    j["x_max"] = cfg.x_max;
}
} // namespace

MDConfigManager::MDConfigManager(MDConfig config) { this->config = config; }

MDConfigManager::~MDConfigManager() = default;

void MDConfigManager::print_config() {
    fmt::print("MDConfig:\n"
                         "Global parameters:\n"
                         "  n_particles_global: {}\n"
                         "  n_particles_type0: {}\n"
                         "  box_w_global: {}\n"
                         "  box_h_global: {}\n"
                         "  T_init: {}\n"
                         "  T_target: {}\n"
                         "  SIGMA_AA: {}\n"
                         "  SIGMA_BB: {}\n"
                         "  SIGMA_AB: {}\n"
                         "  EPSILON_AA: {}\n"
                         "  EPSILON_BB: {}\n"
                         "  EPSILON_AB: {}\n"
                         "  MASS_A: {}\n"
                         "  MASS_B: {}\n"
                         "  devide_p: {}\n"
                         "  dt: {}\n"
                         "  Q: {}\n"
                         "  save_dt_interval: {}\n"
                         "  cutoff: {}\n"
                         "  run_name: {}\n"
                         "  load_name: {}\n"
                         "  THREADS_PER_BLOCK: {}\n"
                         "  rank_size: {}\n"
                         "ABP parameters:\n"
                         "  mu: {}\n"
                         "  v0: {}\n"
                         "  D_r: {}\n"
                         "  D_theta: {}\n"
                         "Rank local parameters:\n"
                         "  rank_idx: {}\n"
                         "  n_local: {}\n"
                         "  n_halo_left: {}\n"
                         "  n_halo_right: {}\n"
                         "  n_cap: {}\n"
                         "  halo_left_cap: {}\n"
                         "  halo_right_cap: {}\n"
                         "  left_rank: {}\n"
                         "  right_rank: {}\n"
                         "  x_min: {}\n"
                         "  x_max: {}\n",
                         config.n_particles_global, config.n_particles_type0,
                         config.box_w_global, config.box_h_global, config.T_init,
                         config.T_target, config.SIGMA_AA, config.SIGMA_BB, config.SIGMA_AB,
                         config.EPSILON_AA, config.EPSILON_BB, config.EPSILON_AB,
                         config.MASS_A, config.MASS_B, config.devide_p, config.dt, config.Q,
                         config.save_dt_interval, config.cutoff, config.run_name,
                         config.load_name, config.THREADS_PER_BLOCK, config.rank_size,
                         config.mu, config.v0, config.D_r, config.D_theta, config.rank_idx,
                         config.n_local, config.n_halo_left, config.n_halo_right,
                         config.n_cap, config.halo_left_cap, config.halo_right_cap,
                         config.left_rank, config.right_rank, config.x_min, config.x_max);

    fmt::print("NPH barostat controls:\n"
               "  barostat_mass: {}\n"
               "  k_piston: {}\n"
               "  barostat_area_rate_init: {}\n"
               "  barostat_height_min_ratio: {}\n"
               "  barostat_height_max_ratio: {}\n"
               "  pressure_target_mode: {}\n"
               "  P_target: {}\n"
               "  pressure_target_auto_steps: {}\n"
               "  pressure_eval_interval_steps: {}\n"
               "  pressure_window_samples: {}\n"
               "  pressure_tolerance_rel: {}\n"
               "  pressure_stable_required_windows: {}\n"
               "  nph_warmup_min_steps: {}\n"
               "  nph_warmup_max_steps: {}\n"
               "  nph_steps: {}\n",
               config.barostat_mass, config.k_piston,
               config.barostat_area_rate_init,
               config.barostat_height_min_ratio, config.barostat_height_max_ratio,
               config.pressure_target_mode, config.P_target,
               config.pressure_target_auto_steps, config.pressure_eval_interval_steps,
               config.pressure_window_samples, config.pressure_tolerance_rel,
               config.pressure_stable_required_windows,
               config.nph_warmup_min_steps, config.nph_warmup_max_steps,
               config.nph_steps);
}

// load MDConfig from a JSON file
MDConfigManager MDConfigManager::config_from_json(const std::string &filepath) {
    std::ifstream in(filepath);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open config file: " + filepath);
    }

    json j;
    in >> j;

    MDConfig cfg{};
    load_from_json_object(j, cfg);

    return MDConfigManager(cfg);
}

// save MDConfig to a JSON file
void MDConfigManager::config_to_json(const std::string &filepath) {
    json j;
    store_to_json_object(j, config);

    std::ofstream out(filepath);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open config file for writing: " +
                                                         filepath);
    }

    out << j.dump(4) << std::endl;
}

std::string MDConfigManager::serialize() const {
    json j;
    store_to_json_object(j, config);
    return j.dump();
}

void MDConfigManager::deserialize(const std::string &data) {
    auto j = json::parse(data);
    load_from_json_object(j, config);
}

bool MDConfigManager::apply_override(const std::string &key,
                                                                         const std::string &value) {
    auto assign_numeric = [&](auto &field) -> bool {
        using FieldType = std::decay_t<decltype(field)>;
        FieldType parsed{};
        if (!parse_scalar<FieldType>(value, parsed)) {
            return false;
        }
        field = parsed;
        return true;
    };

    auto assign_string = [&](std::string &field) {
        field = value;
        return true;
    };

    if (key == "n_particles_global")
        return assign_numeric(config.n_particles_global);
    if (key == "n_particles_type0")
        return assign_numeric(config.n_particles_type0);
    if (key == "box_w_global")
        return assign_numeric(config.box_w_global);
    if (key == "box_h_global")
        return assign_numeric(config.box_h_global);
    if (key == "T_init")
        return assign_numeric(config.T_init);
    if (key == "T_target")
        return assign_numeric(config.T_target);
    if (key == "SIGMA_AA")
        return assign_numeric(config.SIGMA_AA);
    if (key == "SIGMA_BB")
        return assign_numeric(config.SIGMA_BB);
    if (key == "SIGMA_AB")
        return assign_numeric(config.SIGMA_AB);
    if (key == "EPSILON_AA")
        return assign_numeric(config.EPSILON_AA);
    if (key == "EPSILON_BB")
        return assign_numeric(config.EPSILON_BB);
    if (key == "EPSILON_AB")
        return assign_numeric(config.EPSILON_AB);
    if (key == "MASS_A")
        return assign_numeric(config.MASS_A);
    if (key == "MASS_B")
        return assign_numeric(config.MASS_B);
    if (key == "devide_p")
        return assign_numeric(config.devide_p);
    if (key == "dt")
        return assign_numeric(config.dt);
    if (key == "Q")
        return assign_numeric(config.Q);
    if (key == "save_dt_interval")
        return assign_numeric(config.save_dt_interval);
    if (key == "cutoff")
        return assign_numeric(config.cutoff);
    if (key == "run_name")
        return assign_string(config.run_name);
    if (key == "load_name")
        return assign_string(config.load_name);
    if (key == "THREADS_PER_BLOCK")
        return assign_numeric(config.THREADS_PER_BLOCK);
    if (key == "rank_size")
        return assign_numeric(config.rank_size);
    if (key == "rank_idx")
        return assign_numeric(config.rank_idx);
    if (key == "n_local")
        return assign_numeric(config.n_local);
    if (key == "n_halo_left")
        return assign_numeric(config.n_halo_left);
    if (key == "n_halo_right")
        return assign_numeric(config.n_halo_right);
    if (key == "n_cap")
        return assign_numeric(config.n_cap);
    if (key == "halo_left_cap")
        return assign_numeric(config.halo_left_cap);
    if (key == "halo_right_cap")
        return assign_numeric(config.halo_right_cap);
    if (key == "left_rank")
        return assign_numeric(config.left_rank);
    if (key == "right_rank")
        return assign_numeric(config.right_rank);
    if (key == "x_min")
        return assign_numeric(config.x_min);
    if (key == "x_max")
        return assign_numeric(config.x_max);

    // ABP parameters
    if (key == "mu")
        return assign_numeric(config.mu);
    if (key == "v0")
        return assign_numeric(config.v0);
    if (key == "D_r")
        return assign_numeric(config.D_r);
    if (key == "D_theta")
        return assign_numeric(config.D_theta);

    // Periodic-box NPH barostat controls
    if (key == "barostat_mass")
        return assign_numeric(config.barostat_mass);
    if (key == "k_piston")
        return assign_numeric(config.k_piston);
    if (key == "barostat_area_rate_init")
        return assign_numeric(config.barostat_area_rate_init);
    if (key == "barostat_height_min_ratio")
        return assign_numeric(config.barostat_height_min_ratio);
    if (key == "barostat_height_max_ratio")
        return assign_numeric(config.barostat_height_max_ratio);
    if (key == "pressure_target_mode")
        return assign_string(config.pressure_target_mode);
    if (key == "P_target")
        return assign_numeric(config.P_target);
    if (key == "pressure_target_auto_steps")
        return assign_numeric(config.pressure_target_auto_steps);
    if (key == "pressure_eval_interval_steps")
        return assign_numeric(config.pressure_eval_interval_steps);
    if (key == "pressure_window_samples")
        return assign_numeric(config.pressure_window_samples);
    if (key == "pressure_tolerance_rel")
        return assign_numeric(config.pressure_tolerance_rel);
    if (key == "pressure_stable_required_windows")
        return assign_numeric(config.pressure_stable_required_windows);
    if (key == "nph_warmup_min_steps")
        return assign_numeric(config.nph_warmup_min_steps);
    if (key == "nph_warmup_max_steps")
        return assign_numeric(config.nph_warmup_max_steps);
    if (key == "nph_steps")
        return assign_numeric(config.nph_steps);

    return false;
}

void MDConfigManager::apply_overrides(
        const std::vector<MDConfigOverride> &overrides) {
    for (const auto &override_item : overrides) {
        if (!apply_override(override_item.key, override_item.value)) {
            throw std::runtime_error(fmt::format(
                    "Invalid override '{}={}'", override_item.key, override_item.value));
        }
    }
}

bool MDConfigManager::parse_override_argument(const std::string &argument,
                                                                                            std::string &out_key,
                                                                                            std::string &out_value) {
    if (argument.empty()) {
        return false;
    }

    std::string token = argument;
    if (token.rfind("--", 0) == 0) {
        token.erase(0, 2);
    }

    if (token.rfind('D', 0) == 0) {
        token.erase(0, 1);
    }

    auto equal_pos = token.find('=');
    if (equal_pos == std::string::npos) {
        return false;
    }

    out_key = token.substr(0, equal_pos);
    out_value = token.substr(equal_pos + 1);

    return !out_key.empty();
}
