#pragma once

#include <string>
#include <vector>
#include <limits>
#include <cmath>
#include <array>
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

// Write particle positions/types to a CSV snapshot compatible with python/particle_csv.py.
// First row stores metadata; subsequent rows are "x,<x>,y,<y>,type,<int>" per particle.
void print_particles_csv(const std::vector<Particle>& particles,
                         const std::string& filename,
                         const double box_w   = 0.0,
                         const double box_h   = 0.0,
                         const double sigma_aa = 1.0,
                         const double sigma_bb = 1.0);

// Generate a particle CSV (via print_particles_csv) and call the Python plotting script to save an image.
void plot_particles_python(const std::vector<Particle>& particles,
                           const std::string& filename,
                           const std::string& csv_path,
                           const double box_w,
                           const double box_h,
                           const double sigma_aa,
                           const double sigma_bb);

// Legacy triangulation path (CPU, delaunator) – kept for reference.
// Convert a delaunator triangulation + particles into a triangulation CSV (metadata, particles, then triangles).
void print_triangulation_csv(const std::vector<Particle>& particles,
                             const delaunator::Delaunator& triangulation,
                             const std::string& csv_name,
                             const double box_w   = 0.0,
                             const double box_h   = 0.0,
                             const double sigma_aa = 1.0,
                             const double sigma_bb = 1.0);

// GPU / generic triangulation path: triangles given as (x0,y0,x1,y1,x2,y2).
// Write a triangulation CSV from explicit triangle vertex coordinates, plus particle metadata.
void print_triangulation_csv_from_triangles(
    const std::vector<Particle>& particles,
    const std::vector<std::array<double, 6>>& triangles,
    const std::string& csv_name,
    double box_w,
    double box_h,
    double sigma_aa,
    double sigma_bb);

// Legacy: write CSV from delaunator triangulation and invoke the Python script to render the mesh image.
void plot_triangulation_python(const std::vector<Particle>& particles,
                               const delaunator::Delaunator& triangulation,
                               const std::string& filename,
                               const std::string& csv_path,
                               const double box_w,
                               const double box_h,
                               const double sigma_aa,
                               const double sigma_bb);

// Use an existing list of triangles to write CSV and call the Python triangulation plot script.
void plot_triangulation_python_from_triangles(
    const std::vector<Particle>& particles,
    const std::vector<std::array<double, 6>>& triangles,
    const std::string& filename,
    const std::string& csv_path,
    const double box_w,
    const double box_h,
    const double sigma_aa,
    const double sigma_bb);

// Write an interface CSV: metadata row, particle block, then interface segments (for python/particle_csv.load_interface_csv).
void print_interfaces_csv(const std::vector<Particle>& particles,
                          const std::vector<std::vector<double>>& interfaces,
                          const std::string& filename,
                          double box_w,
                          double box_h,
                          double sigma_aa,
                          double sigma_bb);

// Write an interface CSV and use the Python interface plotting script to generate an image.
void plot_interfaces_python(const std::vector<Particle>& particles,
                            const std::vector<std::vector<double>>& interfaces,
                            const std::string& filename,
                            const std::string& csv_path,
                            const double box_w,
                            const double box_h,
                            const double sigma_aa,
                            const double sigma_bb);
