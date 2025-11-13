#pragma once

#include "../external/matplotlibcpp/matplotlibcpp.h"

#include <string>
#include <vector>


#ifdef __CUDACC__
    #include <vector_types.h>
#else
    struct double2 {
        double x;
        double y;
    };
#endif

struct Particle {
    double2 pos;
    double2 vel;
    double2 acc;
    int type;
};

void print_particles(const std::vector<Particle>& particles, const std::string& filename,
     double box_w = NULL, double box_h = NULL, double sigma_aa = 1.0, double sigma_bb = 1.0);