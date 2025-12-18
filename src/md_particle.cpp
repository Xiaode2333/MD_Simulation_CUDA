#include "md_particle.hpp"

#include <array>
#include <delaunator-header-only.hpp>

namespace {
struct Extents {
        double min_x = 0.0;
        double max_x = 0.0;
        double min_y = 0.0;
        double max_y = 0.0;
};

Extents compute_extents(const std::vector<Particle>& particles) {
        Extents e;
        if (particles.empty()) {
                return e;
        }

        e.min_x = e.max_x = particles.front().pos.x;
        e.min_y = e.max_y = particles.front().pos.y;

        for (const auto& p : particles) {
                if (p.pos.x < e.min_x) e.min_x = p.pos.x;
                if (p.pos.x > e.max_x) e.max_x = p.pos.x;
                if (p.pos.y < e.min_y) e.min_y = p.pos.y;
                if (p.pos.y > e.max_y) e.max_y = p.pos.y;
        }

        return e;
}
} // namespace

void print_particles_csv(const std::vector<Particle>& particles,
                                         const std::string& filename,
                                         double box_w,
                                         double box_h,
                                         double sigma_aa,
                                         double sigma_bb)
{
        const Extents ext = compute_extents(particles);

        double resolved_box_w = box_w;
        double resolved_box_h = box_h;
        const bool draw_box_input = (box_w > 0.0 && box_h > 0.0);

        if (resolved_box_w <= 0.0) {
                resolved_box_w = particles.empty() ? 1.0 : (ext.max_x - ext.min_x);
                if (resolved_box_w <= 0.0) {
                        resolved_box_w = 1.0;
                }
        }

        if (resolved_box_h <= 0.0) {
                resolved_box_h = particles.empty() ? 1.0 : (ext.max_y - ext.min_y);
                if (resolved_box_h <= 0.0) {
                        resolved_box_h = 1.0;
                }
        }

        std::ofstream out(filename, std::ios::trunc);
        if (!out) {
                throw std::runtime_error("Failed to open particle CSV for writing: " + filename);
        }

        out.setf(std::ios::fixed, std::ios::floatfield);
        out << std::setprecision(12);

        out << "box_w," << resolved_box_w << ','
                << "box_h," << resolved_box_h << ','
                << "sigma_aa," << sigma_aa << ','
                << "sigma_bb," << sigma_bb << ','
                << "draw_box," << (draw_box_input ? 1 : 0) << ','
                << "n_particles," << particles.size() << '\n';

        for (const auto& p : particles) {
                out << "x," << p.pos.x << ','
                        << "y," << p.pos.y << ','
                        << "type," << p.type << '\n';
        }

        if (!out) {
                throw std::runtime_error("Failed to write particle CSV: " + filename);
        }
}

namespace {

void print_triangulation_csv_impl(const std::vector<Particle>& particles,
                                                                    const std::vector<std::array<double, 6>>& triangles,
                                                                    const std::string& csv_name,
                                                                    double box_w,
                                                                    double box_h,
                                                                    double sigma_aa,
                                                                    double sigma_bb)
{
        if (csv_name.empty()) {
                throw std::invalid_argument("print_triangulation_csv: csv_name must not be empty");
        }

        const Extents ext = compute_extents(particles);

        double resolved_box_w = box_w;
        double resolved_box_h = box_h;
        const bool draw_box_input = (box_w > 0.0 && box_h > 0.0);

        if (resolved_box_w <= 0.0) {
                resolved_box_w = particles.empty() ? 1.0 : (ext.max_x - ext.min_x);
                if (resolved_box_w <= 0.0) {
                        resolved_box_w = 1.0;
                }
        }

        if (resolved_box_h <= 0.0) {
                resolved_box_h = particles.empty() ? 1.0 : (ext.max_y - ext.min_y);
                if (resolved_box_h <= 0.0) {
                        resolved_box_h = 1.0;
                }
        }

        std::ofstream out(csv_name, std::ios::trunc);
        if (!out) {
                throw std::runtime_error("Failed to open triangulation CSV for writing: " + csv_name);
        }

        out.setf(std::ios::fixed, std::ios::floatfield);
        out << std::setprecision(12);

        out << "box_w," << resolved_box_w << ','
                << "box_h," << resolved_box_h << ','
                << "sigma_aa," << sigma_aa << ','
                << "sigma_bb," << sigma_bb << ','
                << "draw_box," << (draw_box_input ? 1 : 0) << ','
                << "n_particles," << particles.size() << ','
                << "n_triangles," << triangles.size() << '\n';

        for (const auto& p : particles) {
                out << "x," << p.pos.x << ','
                        << "y," << p.pos.y << ','
                        << "type," << p.type << '\n';
        }

        for (const auto& tri : triangles) {
                out << "x0," << tri[0] << ','
                        << "y0," << tri[1] << ','
                        << "x1," << tri[2] << ','
                        << "y1," << tri[3] << ','
                        << "x2," << tri[4] << ','
                        << "y2," << tri[5] << '\n';
        }

        if (!out) {
                throw std::runtime_error("Failed to write triangulation CSV: " + csv_name);
        }
}

} // namespace

void print_triangulation_csv(const std::vector<Particle>& particles,
                                                         const delaunator::Delaunator& triangulation,
                                                         const std::string& csv_name,
                                                         double box_w,
                                                         double box_h,
                                                         double sigma_aa,
                                                         double sigma_bb)
{
        const auto in_base_box = [&particles, box_w, box_h](double x, double y) {
                double resolved_box_w = box_w;
                double resolved_box_h = box_h;

                if (resolved_box_w <= 0.0 || resolved_box_h <= 0.0) {
                        const Extents ext = compute_extents(particles);
                        if (resolved_box_w <= 0.0) {
                                resolved_box_w = particles.empty() ? 1.0 : (ext.max_x - ext.min_x);
                                if (resolved_box_w <= 0.0) resolved_box_w = 1.0;
                        }
                        if (resolved_box_h <= 0.0) {
                                resolved_box_h = particles.empty() ? 1.0 : (ext.max_y - ext.min_y);
                                if (resolved_box_h <= 0.0) resolved_box_h = 1.0;
                        }
                }

                return (x >= 0.0 && x < resolved_box_w && y >= 0.0 && y < resolved_box_h);
        };

        std::vector<std::array<double, 6>> triangles;
        triangles.reserve(triangulation.triangles.size() / 3);

        for (std::size_t t = 0; t + 2 < triangulation.triangles.size(); t += 3) {
                const std::size_t i0 = triangulation.triangles[t];
                const std::size_t i1 = triangulation.triangles[t + 1];
                const std::size_t i2 = triangulation.triangles[t + 2];

                const double x0 = triangulation.coords[2 * i0];
                const double y0 = triangulation.coords[2 * i0 + 1];
                const double x1 = triangulation.coords[2 * i1];
                const double y1 = triangulation.coords[2 * i1 + 1];
                const double x2 = triangulation.coords[2 * i2];
                const double y2 = triangulation.coords[2 * i2 + 1];

                const bool inside0 = in_base_box(x0, y0);
                const bool inside1 = in_base_box(x1, y1);
                const bool inside2 = in_base_box(x2, y2);

                if (!(inside0 || inside1 || inside2)) {
                        continue;
                }

                triangles.push_back({x0, y0, x1, y1, x2, y2});
        }

        print_triangulation_csv_impl(
                particles, triangles, csv_name, box_w, box_h, sigma_aa, sigma_bb);
}

void print_triangulation_csv_from_triangles(
        const std::vector<Particle>& particles,
        const std::vector<std::array<double, 6>>& triangles,
        const std::string& csv_name,
        double box_w,
        double box_h,
        double sigma_aa,
        double sigma_bb)
{
        print_triangulation_csv_impl(
                particles, triangles, csv_name, box_w, box_h, sigma_aa, sigma_bb);
}

void plot_particles_python(const std::vector<Particle>& particles,
                                                     const std::string& filename,
                                                     const std::string& csv_path,
                                                     const double box_w,
                                                     const double box_h,
                                                     const double sigma_aa,
                                                     const double sigma_bb)
{
        if (filename.empty()) {
                throw std::invalid_argument("plot_particles_python: filename must not be empty");
        }

        // const std::string csv_path = filename + ".csv";
        print_particles_csv(particles, csv_path, box_w, box_h, sigma_aa, sigma_bb);

        const std::string command =
                "~/.conda/envs/py3/bin/python ./python/plot_particle_python.py --filename \"" + filename +
                "\" --csv_path \"" + csv_path + "\" --strict-box-limits";

        const int status = std::system(command.c_str());
        if (status != 0) {
                throw std::runtime_error(
                        "plot_particle_python.py failed with status " + std::to_string(status));
        }
}

void plot_triangulation_python(const std::vector<Particle>& particles,
                                                             const delaunator::Delaunator& triangulation,
                                                             const std::string& filename,
                                                             const std::string& csv_path,
                                                             const double box_w,
                                                             const double box_h,
                                                             const double sigma_aa,
                                                             const double sigma_bb)
{
        if (filename.empty()) {
                throw std::invalid_argument("plot_triangulation_python: filename must not be empty");
        }
        if (csv_path.empty()) {
                throw std::invalid_argument("plot_triangulation_python: csv_path must not be empty");
        }

        print_triangulation_csv(
                particles,
                triangulation,
                csv_path,
                box_w,
                box_h,
                sigma_aa,
                sigma_bb);

        const std::string command =
                "~/.conda/envs/py3/bin/python ./python/plot_triangulation_python.py --csv_name \"" + csv_path +
                "\" --output_name \"" + filename + "\" --strict-box-limits";

        const int status = std::system(command.c_str());
        if (status != 0) {
                throw std::runtime_error(
                        "plot_triangulation_python.py failed with status " + std::to_string(status));
        }
}

void plot_triangulation_python_from_triangles(
        const std::vector<Particle>& particles,
        const std::vector<std::array<double, 6>>& triangles,
        const std::string& filename,
        const std::string& csv_path,
        const double box_w,
        const double box_h,
        const double sigma_aa,
        const double sigma_bb)
{
        if (filename.empty()) {
                throw std::invalid_argument("plot_triangulation_python_from_triangles: filename must not be empty");
        }
        if (csv_path.empty()) {
                throw std::invalid_argument("plot_triangulation_python_from_triangles: csv_path must not be empty");
        }

        print_triangulation_csv_from_triangles(
                particles,
                triangles,
                csv_path,
                box_w,
                box_h,
                sigma_aa,
                sigma_bb);

        const std::string command =
                "~/.conda/envs/py3/bin/python ./python/plot_triangulation_python.py --csv_name \"" + csv_path +
                "\" --output_name \"" + filename + "\" --strict-box-limits";

        const int status = std::system(command.c_str());
        if (status != 0) {
                throw std::runtime_error(
                        "plot_triangulation_python.py failed with status " + std::to_string(status));
        }
}


void print_interfaces_csv(const std::vector<Particle>& particles,
                                                    const std::vector<std::vector<double>>& interfaces,
                                                    const std::string& filename,
                                                    double box_w,
                                                    double box_h,
                                                    double sigma_aa,
                                                    double sigma_bb)
{
        // Compute extents if box dimensions are not provided
        double min_x = particles.empty() ? 0.0 : particles[0].pos.x;
        double max_x = min_x;
        double min_y = particles.empty() ? 0.0 : particles[0].pos.y;
        double max_y = min_y;

        if (box_w <= 0.0 || box_h <= 0.0) {
                for (const auto& p : particles) {
                        if (p.pos.x < min_x) min_x = p.pos.x;
                        if (p.pos.x > max_x) max_x = p.pos.x;
                        if (p.pos.y < min_y) min_y = p.pos.y;
                        if (p.pos.y > max_y) max_y = p.pos.y;
                }
        }

        double resolved_box_w = (box_w > 0.0) ? box_w : (max_x - min_x);
        double resolved_box_h = (box_h > 0.0) ? box_h : (max_y - min_y);
        if (resolved_box_w <= 0.0) resolved_box_w = 1.0;
        if (resolved_box_h <= 0.0) resolved_box_h = 1.0;
        const bool draw_box = (box_w > 0.0 && box_h > 0.0);

        std::ofstream out(filename, std::ios::trunc);
        if (!out) {
                throw std::runtime_error("Failed to open interfaces CSV for writing: " + filename);
        }

        out.setf(std::ios::fixed, std::ios::floatfield);
        out << std::setprecision(12);

        // Header line
        out << "box_w," << resolved_box_w << ','
                << "box_h," << resolved_box_h << ','
                << "sigma_aa," << sigma_aa << ','
                << "sigma_bb," << sigma_bb << ','
                << "draw_box," << (draw_box ? 1 : 0) << ','
                << "n_particles," << particles.size() << ','
                << "n_interfaces," << interfaces.size();

        // Add length of each interface (number of segments) to header
        for (size_t i = 0; i < interfaces.size(); ++i) {
                // interfaces[i] has [x1, y1, x2, y2, ...], so segments = size / 4
                out << ",len_interface_" << i << "," << (interfaces[i].size() / 4);
        }
        out << '\n';

        // Particle Data
        for (const auto& p : particles) {
                out << "x," << p.pos.x << ','
                        << "y," << p.pos.y << ','
                        << "type," << p.type << '\n';
        }

        // Interface Data
        for (size_t i = 0; i < interfaces.size(); ++i) {
                const auto& iface = interfaces[i];
                for (size_t j = 0; j + 3 < iface.size(); j += 4) {
                        out << "iface_idx," << i << ','
                                << "x1," << iface[j] << ','
                                << "y1," << iface[j+1] << ','
                                << "x2," << iface[j+2] << ','
                                << "y2," << iface[j+3] << '\n';
                }
        }

        if (!out) {
                throw std::runtime_error("Failed to write interfaces CSV: " + filename);
        }
}


void plot_interfaces_python(const std::vector<Particle>& particles,
                                                        const std::vector<std::vector<double>>& interfaces,
                                                        const std::string& filename,
                                                        const std::string& csv_path,
                                                        const double box_w,
                                                        const double box_h,
                                                        const double sigma_aa,
                                                        const double sigma_bb)
{
        if (filename.empty()) {
                throw std::invalid_argument("plot_interfaces_python: filename must not be empty");
        }
        if (csv_path.empty()) {
                throw std::invalid_argument("plot_interfaces_python: csv_path must not be empty");
        }

        print_interfaces_csv(particles, interfaces, csv_path, box_w, box_h, sigma_aa, sigma_bb);

        const std::string command =
                "~/.conda/envs/py3/bin/python ./python/plot_interface_python.py --filename \"" + filename +
                "\" --csv_path \"" + csv_path + "\" --strict-box-limits";

        const int status = std::system(command.c_str());
        if (status != 0) {
                throw std::runtime_error(
                        "plot_interface_python.py failed with status " + std::to_string(status));
        }
}
