#include "md_particle.hpp"


void print_particles(const std::vector<Particle>& particles,
                     const std::string& filename,
                     double box_w, 
                     double box_h, 
                     double sigma_aa,
                     double sigma_bb) 
{
    namespace plt = matplotlibcpp;

    if (particles.empty()) {
        return;
    }

    // ----------------------------
    // 1) Determine box if needed
    // ----------------------------
    double min_x = std::numeric_limits<double>::max();
    double max_x = -std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double max_y = -std::numeric_limits<double>::max();

    for (const auto& p : particles) {
        if (p.pos.x < min_x) min_x = p.pos.x;
        if (p.pos.x > max_x) max_x = p.pos.x;
        if (p.pos.y < min_y) min_y = p.pos.y;
        if (p.pos.y > max_y) max_y = p.pos.y;
    }

    bool draw_box = false;
    if (box_w <= 0.0 || box_h <= 0.0) {
        box_w = max_x - min_x;
        if (box_w <= 0.0) {
            box_w = 1.0;
        }

        box_h = max_y - min_y;
        if (box_h <= 0.0) {
            box_h = 1.0;
        }
    } else {
        draw_box = true;
    }

    const double x_left   = min_x;
    const double x_right  = x_left + box_w;
    const double y_bottom = min_y;
    const double y_top    = y_bottom + box_h;

    // ----------------------------
    // 2) Group particles by type
    //    type == 0 -> A (sigma_aa)
    //    others    -> B (sigma_bb)
    // ----------------------------
    std::vector<double> xs_a, ys_a;
    std::vector<double> xs_b, ys_b;

    xs_a.reserve(particles.size());
    ys_a.reserve(particles.size());
    xs_b.reserve(particles.size());
    ys_b.reserve(particles.size());

    for (const auto& p : particles) {
        if (p.type == 0) {
            xs_a.push_back(p.pos.x);
            ys_a.push_back(p.pos.y);
        } else {
            xs_b.push_back(p.pos.x);
            ys_b.push_back(p.pos.y);
        }
    }

    // ----------------------------
    // 3) Fix figure size in pixels
    //    and compute marker radii so that
    //
    //    2^(1/6)*sigma_aa / box_w = R_a_px / fig_w_px
    //    2^(1/6)*sigma_bb / box_w = R_b_px / fig_w_px
    //
    //    Then convert desired radius in pixels -> scatter size s
    //    using s [pt^2] ~ area of marker in points^2:
    //    r_px = sqrt(s/pi) * dpi / 72
    //    => s = pi * (r_px * 72 / dpi)^2
    // ----------------------------
    const double fig_width_in = 10.0;   // same as fig_width_in in Python
    const double dpi          = 300.0;  // same as dpi in Python

    // Figure size in pixels: width = fig_width_in * dpi
    const int fig_w_px = static_cast<int>(fig_width_in * dpi);
    int       fig_h_px = static_cast<int>(fig_width_in * (box_h / box_w) * dpi);
    if (fig_h_px <= 0) {
        fig_h_px = static_cast<int>(fig_width_in * dpi);
    }

    plt::figure_size(fig_w_px, fig_h_px);
    plt::figure();

    // ----------------------------------------
    // Marker sizes (in points^2), per type
    // Python logic:
    //   radius_points = (sigma * 0.5) * (fig_width_in / box_w) * 72.0
    //   marker_size   = radius_points ** 2
    // Here we do it per species using sigma_aa, sigma_bb.
    // ----------------------------------------
    const double sigma_a = sigma_aa;
    const double sigma_b = sigma_bb;

    // 1 unit in box corresponds to (fig_width_in / box_w) inches on the figure.
    // 1 inch = 72 points, so:
    //   radius_points = (sigma/2) * (fig_width_in / box_w) * 72
    const double radius_a_pts = (sigma_a * 0.5) * (fig_width_in / box_w) * 72.0;
    const double radius_b_pts = (sigma_b * 0.5) * (fig_width_in / box_w) * 72.0;

    // Scatter s argument = area in points^2
    const double size_a = radius_a_pts * radius_a_pts;
    const double size_b = radius_b_pts * radius_b_pts;

    // ----------------------------
    // 4) Plot
    // ----------------------------
    plt::xlim(x_left,  x_right);
    plt::ylim(y_bottom, y_top);

    if (!xs_a.empty()) {
        plt::scatter(xs_a, ys_a, size_a,
                    {{"facecolors", "red"},
                    {"edgecolors", "black"},
                    {"linewidths", "0.1"}});
    }
    if (!xs_b.empty()) {
        plt::scatter(xs_b, ys_b, size_b,
                    {{"facecolors", "blue"},
                    {"edgecolors", "black"},
                    {"linewidths", "0.1"}});
    }

    if (draw_box) {
        plt::plot({x_left, x_right, x_right, x_left, x_left},
                {y_bottom, y_bottom, y_top, y_top, y_bottom},
                {{"c", "black"}, {"linestyle", "--"}});
    }

    plt::xlabel("x");
    plt::ylabel("y");
    plt::title("Particles");
    plt::axis("equal");

    plt::save(filename);
    plt::close();
}
