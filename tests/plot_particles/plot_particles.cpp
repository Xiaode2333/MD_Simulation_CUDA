// tests/plot_particles/plot_particles.cpp

//g++ tests/plot_particles/plot_particles.cpp src/md_particle.cpp -I. \
 -o tests/plot_particles/plot_particles -Wno-register \
 -I"$CONDA_PREFIX/include" $(python-config --includes)  \
 $(python-config --ldflags)  -I"$(python -c 'import numpy; print(numpy.get_include())')" -fPIC -lfmt
#include <vector>
#include <random>
#include <fmt/core.h>
#include <string>

#include "../../include/md_particle.hpp"   // contains Particle + print_particles

int main() {
    const double box_w = 200;
    const double box_h = 20;
    const int    N     = 2000;

    std::vector<Particle> particles;
    particles.reserve(N);

    // RNG for uniform positions in [0, box_w] x [0, box_h]
    std::mt19937_64 rng(123456);  // fixed seed for reproducibility
    std::uniform_real_distribution<double> dist_x(0.0, box_w);
    std::uniform_real_distribution<double> dist_y(0.0, box_h);

    for (int i = 0; i < N; ++i) {
        Particle p{};
        p.pos.x = dist_x(rng);
        p.pos.y = dist_y(rng);

        p.vel.x = 0.0;
        p.vel.y = 0.0;
        p.acc.x = 0.0;
        p.acc.y = 0.0;

        // Half A (type 0), half B (type 1)
        p.type = (i < N / 2) ? 0 : 1;

        particles.push_back(p);
    }

    const std::string out_png = "tests/plot_particles/particles.png";

    // Example sigma values; adjust if you want different size ratio
    double sigma_aa = 1.0;
    double sigma_bb = 1.0;

    print_particles(particles, out_png, box_w, box_h, sigma_aa, sigma_bb);

    fmt::print("Saved {} particles plot to: {}\n", N, out_png);
    return 0;
}
