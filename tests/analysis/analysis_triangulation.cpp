// Stream a multi-frame XYZ trajectory, triangulate each frame with gDel2D, and
// write one compact binary file beside the trajectory.
//
// Output format (.tri2d):
// - File header: magic, versioned metadata, and the fixed rule used to derive
//   the PBC expansion width from each frame's XYZ box data
// - For each frame: frame metadata, the resolved per-frame PBC band width,
//   expanded PBC vertex mapping,
//   (particle index + periodic shift), then triangle connectivity as indices
//   into that expanded vertex list
//
// Expanded vertex coordinates can be reconstructed from the XYZ frame via:
//   x = particle_x + shift_x * Lx
//   y = particle_y + shift_y * Ly

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>
#include <fmt/core.h>
#include "external/gDel2D-Oct2015/src/gDel2D/GpuDelaunay.h"

namespace fs = std::filesystem;

namespace {

constexpr char kFileMagic[8] = {'T', 'R', 'I', '2', 'D', '0', '1', '\0'};
constexpr std::uint32_t kFrameMagic = 0x314d5246u; // "FRM1"
constexpr double kPbcBandMinBoxFraction = 0.25;

struct Options {
    fs::path xyz_path;
    fs::path output_path;
    bool overwrite = false;
};

struct ParticleSnapshot {
    double x = 0.0;
    double y = 0.0;
    int type = 0;
};

struct FrameMetadata {
    std::string comment;
    std::string phase;
    std::int64_t global_step = -1;
    std::int64_t phase_step = -1;
    double global_time = -1.0;
    double phase_time = -1.0;
    double Lx = 0.0;
    double Ly = 0.0;
};

struct TriangulationParameters {
    double pbc_band_min_box_fraction = kPbcBandMinBoxFraction;
};

struct ExpandedFrame {
    std::vector<Point2> points;
    std::vector<std::int32_t> particle_indices;
    std::vector<std::int8_t> shift_x;
    std::vector<std::int8_t> shift_y;
};

void print_usage(const char* argv0) {
    fmt::print(
        "Usage: {} --xyz PATH [--output PATH] [--overwrite]\n"
        "Writes PATH.tri2d by default.\n",
        argv0);
}

bool is_space(char c) {
    return std::isspace(static_cast<unsigned char>(c)) != 0;
}

std::string trim_copy(const std::string& value) {
    std::size_t begin = 0;
    while (begin < value.size() && is_space(value[begin])) {
        ++begin;
    }

    std::size_t end = value.size();
    while (end > begin && is_space(value[end - 1])) {
        --end;
    }

    return value.substr(begin, end - begin);
}

std::optional<std::int64_t> parse_int64(const std::string& value) {
    try {
        std::size_t pos = 0;
        const long long parsed = std::stoll(value, &pos);
        if (pos != value.size()) {
            return std::nullopt;
        }
        return static_cast<std::int64_t>(parsed);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<double> parse_double_string(const std::string& value) {
    try {
        std::size_t pos = 0;
        const double parsed = std::stod(value, &pos);
        if (pos != value.size()) {
            return std::nullopt;
        }
        return parsed;
    } catch (...) {
        return std::nullopt;
    }
}

Options parse_args(int argc, char** argv) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);

        if (arg == "--xyz") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--xyz requires a path");
            }
            options.xyz_path = argv[++i];
        } else if (arg == "--output") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--output requires a path");
            }
            options.output_path = argv[++i];
        } else if (arg == "--overwrite") {
            options.overwrite = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument(fmt::format("Unknown argument '{}'", arg));
        }
    }

    if (options.xyz_path.empty()) {
        throw std::invalid_argument("--xyz is required");
    }

    if (options.output_path.empty()) {
        options.output_path = fs::path(options.xyz_path.string() + ".tri2d");
    }

    return options;
}

int parse_particle_type(std::string_view token) {
    if (token == "A" || token == "a") {
        return 0;
    }
    if (token == "B" || token == "b") {
        return 1;
    }
    if (token == "X" || token == "x") {
        return 2;
    }

    try {
        return std::stoi(std::string(token));
    } catch (...) {
        return 2;
    }
}

bool parse_next_double(const char*& cursor, double& value) {
    while (*cursor != '\0' && is_space(*cursor)) {
        ++cursor;
    }
    if (*cursor == '\0') {
        return false;
    }

    char* end = nullptr;
    value = std::strtod(cursor, &end);
    if (end == cursor) {
        return false;
    }
    cursor = end;
    return true;
}

ParticleSnapshot parse_particle_line(const std::string& line, std::size_t line_number) {
    const char* cursor = line.c_str();
    while (*cursor != '\0' && is_space(*cursor)) {
        ++cursor;
    }

    const char* token_begin = cursor;
    while (*cursor != '\0' && !is_space(*cursor)) {
        ++cursor;
    }

    if (cursor == token_begin) {
        throw std::runtime_error(
            fmt::format("Malformed particle line {}: missing type token", line_number));
    }

    ParticleSnapshot particle;
    particle.type = parse_particle_type(
        std::string_view(token_begin, static_cast<std::size_t>(cursor - token_begin)));

    if (!parse_next_double(cursor, particle.x) || !parse_next_double(cursor, particle.y)) {
        throw std::runtime_error(
            fmt::format("Malformed particle line {}: expected x and y coordinates", line_number));
    }

    return particle;
}

void parse_comment_line(const std::string& comment, FrameMetadata& meta) {
    meta.comment = comment;

    std::istringstream iss(comment);
    std::string token;
    while (iss >> token) {
        const std::size_t eq_pos = token.find('=');
        if (eq_pos == std::string::npos) {
            continue;
        }

        const std::string key = token.substr(0, eq_pos);
        const std::string value = token.substr(eq_pos + 1);

        if (key == "phase") {
            meta.phase = value;
        } else if (key == "global_step") {
            if (const auto parsed = parse_int64(value)) {
                meta.global_step = *parsed;
            }
        } else if (key == "phase_step") {
            if (const auto parsed = parse_int64(value)) {
                meta.phase_step = *parsed;
            }
        } else if (key == "global_time") {
            if (const auto parsed = parse_double_string(value)) {
                meta.global_time = *parsed;
            }
        } else if (key == "phase_time") {
            if (const auto parsed = parse_double_string(value)) {
                meta.phase_time = *parsed;
            }
        } else if (key == "Lx") {
            if (const auto parsed = parse_double_string(value)) {
                meta.Lx = *parsed;
            }
        } else if (key == "Ly") {
            if (const auto parsed = parse_double_string(value)) {
                meta.Ly = *parsed;
            }
        }
    }
}

bool read_next_frame(std::istream& in,
                     std::vector<ParticleSnapshot>& particles,
                     FrameMetadata& meta,
                     std::size_t& line_number) {
    std::string line;
    while (std::getline(in, line)) {
        ++line_number;
        line = trim_copy(line);
        if (!line.empty()) {
            break;
        }
    }

    if (!in) {
        return false;
    }

    std::size_t n_particles = 0;
    try {
        n_particles = static_cast<std::size_t>(std::stoull(line));
    } catch (...) {
        throw std::runtime_error(
            fmt::format("Expected particle count at line {}, got '{}'", line_number, line));
    }

    if (!std::getline(in, line)) {
        throw std::runtime_error("Unexpected EOF while reading XYZ comment line");
    }
    ++line_number;

    meta = FrameMetadata{};
    parse_comment_line(trim_copy(line), meta);

    particles.clear();
    particles.reserve(n_particles);

    for (std::size_t i = 0; i < n_particles; ++i) {
        if (!std::getline(in, line)) {
            throw std::runtime_error(
                fmt::format("Unexpected EOF while reading particle {} of {}", i + 1, n_particles));
        }
        ++line_number;
        particles.push_back(parse_particle_line(line, line_number));
    }

    return true;
}

TriangulationParameters resolve_triangulation_parameters() {
    TriangulationParameters params;
    return params;
}

double compute_frame_pbc_band_width(std::size_t n_particles, double Lx, double Ly,
                                    const TriangulationParameters& params) {
    (void)n_particles;

    if (Lx <= 0.0 || Ly <= 0.0) {
        return 1.0;
    }

    return std::max(1.0, params.pbc_band_min_box_fraction * std::min(Lx, Ly));
}

ExpandedFrame build_expanded_frame(const std::vector<ParticleSnapshot>& particles,
                                   double Lx,
                                   double Ly,
                                   double band_width) {
    ExpandedFrame expanded;

    const std::size_t n = particles.size();
    expanded.points.reserve(n * 2);
    expanded.particle_indices.reserve(n * 2);
    expanded.shift_x.reserve(n * 2);
    expanded.shift_y.reserve(n * 2);

    auto add_vertex = [&](std::int32_t particle_idx,
                          std::int8_t sx,
                          std::int8_t sy,
                          double x,
                          double y) {
        Point2 p;
        p._p[0] = static_cast<RealType>(x);
        p._p[1] = static_cast<RealType>(y);
        expanded.points.push_back(p);
        expanded.particle_indices.push_back(particle_idx);
        expanded.shift_x.push_back(sx);
        expanded.shift_y.push_back(sy);
    };

    for (std::size_t i = 0; i < n; ++i) {
        add_vertex(static_cast<std::int32_t>(i), 0, 0, particles[i].x, particles[i].y);
    }

    for (std::size_t i = 0; i < n; ++i) {
        const double x0 = particles[i].x;
        const double y0 = particles[i].y;
        const auto idx = static_cast<std::int32_t>(i);

        const bool near_left = (x0 < band_width);
        const bool near_right = (x0 > (Lx - band_width));
        const bool near_bottom = (y0 < band_width);
        const bool near_top = (y0 > (Ly - band_width));

        if (near_left) add_vertex(idx, 1, 0, x0 + Lx, y0);
        if (near_right) add_vertex(idx, -1, 0, x0 - Lx, y0);
        if (near_bottom) add_vertex(idx, 0, 1, x0, y0 + Ly);
        if (near_top) add_vertex(idx, 0, -1, x0, y0 - Ly);

        if (near_left && near_bottom) add_vertex(idx, 1, 1, x0 + Lx, y0 + Ly);
        if (near_left && near_top) add_vertex(idx, 1, -1, x0 + Lx, y0 - Ly);
        if (near_right && near_bottom) add_vertex(idx, -1, 1, x0 - Lx, y0 + Ly);
        if (near_right && near_top) add_vertex(idx, -1, -1, x0 - Lx, y0 - Ly);
    }

    return expanded;
}

std::vector<std::int32_t> triangulate_frame(const ExpandedFrame& expanded, double Lx, double Ly, GpuDel& gpu_del) {
    if (expanded.points.size() < 3) {
        return {};
    }

    GDel2DInput input;
    input.pointVec = expanded.points;
    input.constraintVec.clear();
    input.insAll = false;
    input.noSort = true;
    input.noReorder = true;
    input.profLevel = ProfNone;

    GDel2DOutput output;
    gpu_del.compute(input, &output);

    const int inf_idx = static_cast<int>(input.pointVec.size());
    std::vector<std::int32_t> triangles;
    triangles.reserve(output.triVec.size() * 3);

    const auto in_base_box = [Lx, Ly](const Point2& p) {
        const double x = static_cast<double>(p._p[0]);
        const double y = static_cast<double>(p._p[1]);
        return (x >= 0.0 && x < Lx && y >= 0.0 && y < Ly);
    };

    for (const Tri& tri : output.triVec) {
        if (tri._v[0] == inf_idx || tri._v[1] == inf_idx || tri._v[2] == inf_idx) {
            continue;
        }

        const Point2& p0 = input.pointVec[tri._v[0]];
        const Point2& p1 = input.pointVec[tri._v[1]];
        const Point2& p2 = input.pointVec[tri._v[2]];

        if (!(in_base_box(p0) || in_base_box(p1) || in_base_box(p2))) {
            continue;
        }

        triangles.push_back(static_cast<std::int32_t>(tri._v[0]));
        triangles.push_back(static_cast<std::int32_t>(tri._v[1]));
        triangles.push_back(static_cast<std::int32_t>(tri._v[2]));
    }

    return triangles;
}

template <typename T>
void write_pod(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) {
        throw std::runtime_error("Failed while writing binary output");
    }
}

void write_string(std::ostream& out, const std::string& value) {
    const std::uint32_t length = static_cast<std::uint32_t>(value.size());
    write_pod(out, length);
    if (length > 0) {
        out.write(value.data(), static_cast<std::streamsize>(length));
        if (!out) {
            throw std::runtime_error("Failed while writing string payload");
        }
    }
}

template <typename T>
void write_vector(std::ostream& out, const std::vector<T>& values) {
    if (values.empty()) {
        return;
    }

    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!out) {
        throw std::runtime_error("Failed while writing vector payload");
    }
}

void write_file_header(std::ostream& out,
                       const fs::path& xyz_path,
                       const fs::path& output_path,
                       const TriangulationParameters& params,
                       std::streampos& frame_count_pos) {
    out.write(kFileMagic, sizeof(kFileMagic));
    if (!out) {
        throw std::runtime_error("Failed while writing file magic");
    }

    const std::uint32_t version = 1;
    write_pod(out, version);

    frame_count_pos = out.tellp();
    write_pod<std::uint64_t>(out, 0);
    write_pod(out, params.pbc_band_min_box_fraction);
    write_string(out, fs::absolute(xyz_path).string());
    write_string(out, fs::absolute(output_path).string());
}

void write_frame_record(std::ostream& out,
                        std::uint64_t frame_index,
                        const FrameMetadata& meta,
                        std::uint64_t n_particles,
                        double pbc_band_width,
                        const ExpandedFrame& expanded,
                        const std::vector<std::int32_t>& triangles) {
    write_pod(out, kFrameMagic);
    write_pod(out, frame_index);
    write_pod(out, n_particles);
    write_pod<std::uint64_t>(out, expanded.points.size());
    write_pod<std::uint64_t>(out, triangles.size() / 3);
    write_pod(out, pbc_band_width);
    write_pod(out, meta.Lx);
    write_pod(out, meta.Ly);
    write_pod(out, meta.global_step);
    write_pod(out, meta.phase_step);
    write_pod(out, meta.global_time);
    write_pod(out, meta.phase_time);
    write_string(out, meta.phase);
    write_string(out, meta.comment);
    write_vector(out, expanded.particle_indices);
    write_vector(out, expanded.shift_x);
    write_vector(out, expanded.shift_y);
    write_vector(out, triangles);
}

void finalize_output_file(std::ofstream& out, std::streampos frame_count_pos, std::uint64_t frame_count) {
    out.seekp(frame_count_pos);
    if (!out) {
        throw std::runtime_error("Failed to seek back to frame count in output file");
    }
    write_pod(out, frame_count);
    out.flush();
    if (!out) {
        throw std::runtime_error("Failed to finalize output file");
    }
}

void ensure_cuda_device() {
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess) {
        throw std::runtime_error(
            fmt::format("cudaGetDeviceCount failed: {}", cudaGetErrorString(count_status)));
    }
    if (device_count <= 0) {
        throw std::runtime_error("No CUDA devices are visible to the triangulation analyzer");
    }

    const cudaError_t set_status = cudaSetDevice(0);
    if (set_status != cudaSuccess) {
        throw std::runtime_error(
            fmt::format("cudaSetDevice(0) failed: {}", cudaGetErrorString(set_status)));
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::ios::sync_with_stdio(false);

        const Options options = parse_args(argc, argv);

        if (!fs::exists(options.xyz_path)) {
            throw std::runtime_error(fmt::format("XYZ file '{}' does not exist",
                                                 options.xyz_path.string()));
        }
        if (!fs::is_regular_file(options.xyz_path)) {
            throw std::runtime_error(fmt::format("'{}' is not a regular file",
                                                 options.xyz_path.string()));
        }

        if (fs::exists(options.output_path) && !options.overwrite) {
            fmt::print("[analysis_triangulation] Output already exists, skipping: {}\n",
                       options.output_path.string());
            return 0;
        }

        std::ifstream in(options.xyz_path);
        if (!in) {
            throw std::runtime_error(
                fmt::format("Failed to open XYZ file '{}'", options.xyz_path.string()));
        }

        std::vector<ParticleSnapshot> particles;
        FrameMetadata first_meta;
        std::size_t line_number = 0;
        if (!read_next_frame(in, particles, first_meta, line_number)) {
            throw std::runtime_error("XYZ file is empty");
        }
        if (first_meta.Lx <= 0.0 || first_meta.Ly <= 0.0) {
            throw std::runtime_error(
                "First XYZ frame is missing valid Lx/Ly metadata in the comment line");
        }

        const TriangulationParameters params = resolve_triangulation_parameters();

        fmt::print("[analysis_triangulation] XYZ: {}\n", options.xyz_path.string());
        fmt::print("[analysis_triangulation] Output: {}\n", options.output_path.string());
        fmt::print(
            "[analysis_triangulation] Using per-frame PBC band width = {:.3f} * min(Lx, Ly) from the XYZ header\n",
            params.pbc_band_min_box_fraction);

        fs::create_directories(options.output_path.parent_path());

        const fs::path tmp_output_path = fs::path(options.output_path.string() + ".tmp");
        {
            std::error_code ec;
            fs::remove(tmp_output_path, ec);
        }

        std::ofstream out(tmp_output_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error(
                fmt::format("Failed to open output file '{}'", tmp_output_path.string()));
        }

        std::streampos frame_count_pos{};
        write_file_header(out, options.xyz_path, options.output_path, params, frame_count_pos);

        ensure_cuda_device();
        GpuDel gpu_del;

        const auto start_time = std::chrono::steady_clock::now();

        std::uint64_t frame_count = 0;
        std::uint64_t total_triangles = 0;
        std::uint64_t total_vertices = 0;

        auto process_frame = [&](const FrameMetadata& meta_ref) {
            if (meta_ref.Lx <= 0.0 || meta_ref.Ly <= 0.0) {
                throw std::runtime_error(
                    fmt::format("Frame {} is missing valid Lx/Ly metadata", frame_count));
            }

            const double pbc_band_width =
                compute_frame_pbc_band_width(particles.size(), meta_ref.Lx, meta_ref.Ly, params);
            ExpandedFrame expanded =
                build_expanded_frame(particles, meta_ref.Lx, meta_ref.Ly, pbc_band_width);
            std::vector<std::int32_t> triangles =
                triangulate_frame(expanded, meta_ref.Lx, meta_ref.Ly, gpu_del);

            write_frame_record(out, frame_count, meta_ref, particles.size(), pbc_band_width,
                               expanded, triangles);

            total_vertices += static_cast<std::uint64_t>(expanded.points.size());
            total_triangles += static_cast<std::uint64_t>(triangles.size() / 3);

            if (frame_count == 0 || ((frame_count + 1) % 25 == 0)) {
                const std::chrono::duration<double> elapsed =
                    std::chrono::steady_clock::now() - start_time;
                fmt::print(
                    "[analysis_triangulation] frame {} phase={} global_step={} Lx={:.6f} Ly={:.6f} band={:.6f} vertices={} triangles={} elapsed={:.1f}s\n",
                    frame_count,
                    meta_ref.phase.empty() ? "UNKNOWN" : meta_ref.phase,
                    meta_ref.global_step,
                    meta_ref.Lx,
                    meta_ref.Ly,
                    pbc_band_width,
                    expanded.points.size(),
                    triangles.size() / 3,
                    elapsed.count());
            }

            ++frame_count;
        };

        process_frame(first_meta);

        FrameMetadata meta;
        while (read_next_frame(in, particles, meta, line_number)) {
            process_frame(meta);
        }

        finalize_output_file(out, frame_count_pos, frame_count);
        out.close();

        if (fs::exists(options.output_path)) {
            fs::remove(options.output_path);
        }
        fs::rename(tmp_output_path, options.output_path);

        const std::chrono::duration<double> total_elapsed =
            std::chrono::steady_clock::now() - start_time;
        const double avg_vertices = frame_count == 0
                                        ? 0.0
                                        : static_cast<double>(total_vertices) / static_cast<double>(frame_count);
        const double avg_triangles = frame_count == 0
                                         ? 0.0
                                         : static_cast<double>(total_triangles) / static_cast<double>(frame_count);

        fmt::print(
            "[analysis_triangulation] Done. frames={} avg_vertices={:.1f} avg_triangles={:.1f} elapsed={:.1f}s\n",
            frame_count,
            avg_vertices,
            avg_triangles,
            total_elapsed.count());
        return 0;
    } catch (const std::exception& ex) {
        fmt::print(stderr, "[analysis_triangulation] {}\n", ex.what());
        return 1;
    }
}
