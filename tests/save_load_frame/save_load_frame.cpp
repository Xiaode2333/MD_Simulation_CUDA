// g++ tests/save_load_frame/save_load_frame.cpp     src/md_common.cpp src/md_particle.cpp     -o tests/save_load_frame/save_load_frame     -I"${CONDA_PREFIX}/include"     -I"./include"     -L"${CONDA_PREFIX}/lib"     -Wl,-rpath,"${CONDA_PREFIX}/lib"     -lz -lfmt
#include "../../include/md_common.hpp"
#include "../../include/md_particle.hpp"

#include <vector>
#include <fmt/core.h>

int main() {
    std::vector<Particle> frame0(3);
    for (std::size_t i = 0; i < frame0.size(); i++) {
        frame0[i].pos = {double(i), double(i) * 2.0};
        frame0[i].vel = {0.1, 0.2};
        frame0[i].acc = {0.0, 0.0};
        frame0[i].type = int(i);
    }

    FileWriter writer("tests/save_load_frame/frames.bin", false);
    writer.write_frame(frame0.data(), frame0.size(), 0);

    std::vector<Particle> frame1(2);
    for (std::size_t i = 0; i < frame1.size(); i++) {
        frame1[i].pos = {double(i), double(i) * 3.0};
        frame1[i].vel = {0.3, 0.4};
        frame1[i].acc = {0.0, 0.0};
        frame1[i].type = int(10 + i);
    }
    writer.write_frame(frame1.data(), frame1.size(), 1);

    FileReader reader("tests/save_load_frame/frames.bin");
    std::vector<Particle> buf;
    std::uint64_t frame_index = 0;

    while (reader.next_frame(buf, frame_index)) {
        fmt::print("Frame {} has {} particles\n", frame_index, buf.size());
    }

    if (reader.corrupted()) {
        fmt::format_error("File is corrupted!\n");
    }
}