// Basic smoke test for triangulation CSV / plotting helpers.
// Constructs a tiny particle set and a single triangle, writes a triangulation
// CSV, and invokes the Python plotting script.

#include <vector>
#include <array>
#include <string>
#include <filesystem>
#include <system_error>

#include <fmt/core.h>

#include "md_particle.hpp"

int main() {
    const double box_w = 10.0;
    const double box_h = 10.0;

    // Simple three-particle configuration forming one triangle.
    std::vector<Particle> particles;
    particles.reserve(3);

    {
        Particle p{};
        p.pos.x = 1.0;
        p.pos.y = 1.0;
        p.vel.x = p.vel.y = 0.0;
        p.acc.x = p.acc.y = 0.0;
        p.type = 0;
        particles.push_back(p);
    }
    {
        Particle p{};
        p.pos.x = 4.0;
        p.pos.y = 1.0;
        p.vel.x = p.vel.y = 0.0;
        p.acc.x = p.acc.y = 0.0;
        p.type = 0;
        particles.push_back(p);
    }
    {
        Particle p{};
        p.pos.x = 2.5;
        p.pos.y = 4.0;
        p.vel.x = p.vel.y = 0.0;
        p.acc.x = p.acc.y = 0.0;
        p.type = 1;
        particles.push_back(p);
    }

    // Single triangle using the three particle positions.
    std::vector<std::array<double, 6>> triangles;
    triangles.push_back({1.0, 1.0, 4.0, 1.0, 2.5, 4.0});

    const std::string tmp_dir = "./tmp";
    const std::string csv_path = tmp_dir + "/triangulation.csv";
    const std::string out_png = "tests/triangulation/triangulation.png";

    std::error_code ec;
    std::filesystem::create_directories(tmp_dir, ec);
    if (ec) {
        fmt::print(stderr, "Failed to create {}: {}\n", tmp_dir, ec.message());
        return 1;
    }

    const double sigma_aa = 1.0;
    const double sigma_bb = 1.0;

    // Write triangulation CSV from explicit triangles.
    print_triangulation_csv_from_triangles(
        particles,
        triangles,
        csv_path,
        box_w,
        box_h,
        sigma_aa,
        sigma_bb);

    if (!std::filesystem::exists(csv_path)) {
        fmt::print(stderr, "Triangulation CSV was not created at {}\n", csv_path);
        return 1;
    }

    // Also test the Python plotting wrapper.
    try {
        plot_triangulation_python_from_triangles(
            particles,
            triangles,
            out_png,
            csv_path,
            box_w,
            box_h,
            sigma_aa,
            sigma_bb);
    } catch (const std::exception& e) {
        fmt::print(stderr, "plot_triangulation_python_from_triangles failed: {}\n", e.what());
        return 1;
    }

    fmt::print("Triangulation test completed. CSV: {}, PNG: {}\n", csv_path, out_png);
    return 0;
}

