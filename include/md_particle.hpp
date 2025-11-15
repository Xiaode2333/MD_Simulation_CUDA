#pragma once

#include "matplotlibcpp.h"

#include <string>
#include <vector>
#include <limits>
#include <cmath>
#include <vector_types.h>


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

void print_particles(const std::vector<Particle>& particles,
                            const std::string& filename,
                            const double box_w   = 0.0,
                            const double box_h   = 0.0,
                            const double sigma_aa = 1.0,
                            const double sigma_bb = 1.0);