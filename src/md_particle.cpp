#include "../include/md_particle.hpp"

void print_particle(const Particle& p, const std::string& filename,
                           double box_w = NULL, double box_h = NULL) {
    namespace plt = matplotlibcpp;

    // Single point
    std::vector<double> xs(1);
    std::vector<double> ys(1);
    xs[0] = p.pos.x;
    ys[0] = p.pos.y;

    plt::figure();
    plt::scatter(xs, ys);

    // Optional box size for axes
    if (box_w > 0.0) {
        plt::xlim(0.0, box_w);
    }
    if (box_h > 0.0) {
        plt::ylim(0.0, box_h);
    }

    plt::xlabel("x");
    plt::ylabel("y");
    plt::title("Particle position");

    plt::save(filename);
    plt::close();
}