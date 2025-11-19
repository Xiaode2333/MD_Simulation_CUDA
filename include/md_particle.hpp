#pragma once

#include <string>
#include <vector>
#include <limits>
#include <cmath>
#include <vector_types.h>
#include <fmt/core.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace delaunator {
class Delaunator;
}


// #ifdef __CUDACC__
//     #include <vector_types.h>
// #else
//     struct double2 {
//         double x;
//         double y;
//     };
// #endif

struct Particle {
    double2 pos;
    double2 vel;
    double2 acc;
    int type;
};

void print_particles_csv(const std::vector<Particle>& particles,
                         const std::string& filename,
                         const double box_w   = 0.0,
                         const double box_h   = 0.0,
                         const double sigma_aa = 1.0,
                         const double sigma_bb = 1.0);

void plot_particles_python(const std::vector<Particle>& particles,
                           const std::string& filename,
                           const std::string& csv_path,
                           const double box_w,
                           const double box_h,
                           const double sigma_aa,
                           const double sigma_bb);

void print_triangulation_csv(const std::vector<Particle>& particles,
                             const delaunator::Delaunator& triangulation,
                             const std::string& csv_name,
                             const double box_w   = 0.0,
                             const double box_h   = 0.0,
                             const double sigma_aa = 1.0,
                             const double sigma_bb = 1.0);

void plot_triangulation_python(const std::vector<Particle>& particles,
                               const delaunator::Delaunator& triangulation,
                               const std::string& filename,
                               const std::string& csv_path,
                               const double box_w,
                               const double box_h,
                               const double sigma_aa,
                               const double sigma_bb);

void print_interfaces_csv(const std::vector<Particle>& particles,
                          const std::vector<std::vector<double>>& interfaces,
                          const std::string& filename,
                          double box_w,
                          double box_h,
                          double sigma_aa,
                          double sigma_bb);

void plot_interfaces_python(const std::vector<Particle>& particles,
                            const std::vector<std::vector<double>>& interfaces,
                            const std::string& filename,
                            const std::string& csv_path,
                            const double box_w,
                            const double box_h,
                            const double sigma_aa,
                            const double sigma_bb);