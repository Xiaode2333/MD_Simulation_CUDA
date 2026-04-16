// Stream one XYZ trajectory and its matching TRI2D file, reconstruct unique
// physical triangles per frame, then emit one CSV row per mixed triangle
// (AAB/ABB) with its area and counts of neighboring triangle types.

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <fmt/core.h>

namespace fs = std::filesystem;

namespace {

constexpr char kFileMagic[8] = {'T', 'R', 'I', '2', 'D', '0', '1', '\0'};
constexpr std::uint32_t kFrameMagic = 0x314d5246u; // "FRM1"

enum class DeviceMode {
    Cpu,
    Gpu,
};

struct Options {
    fs::path xyz_path;
    fs::path tri2d_path;
    fs::path output_path;
    DeviceMode device = DeviceMode::Gpu;
    bool overwrite = false;
    std::uint64_t max_frames = 0;
};

struct XyzParticle {
    double x = 0.0;
    double y = 0.0;
    int type = 0;
};

struct Tri2dHeader {
    std::uint32_t version = 0;
    std::uint64_t frame_count = 0;
    std::uint8_t backend_code = 255;
    double pbc_band_min_box_fraction = 0.0;
    std::string xyz_path;
    std::string output_path;
};

struct Tri2dFrame {
    std::uint64_t frame_index = 0;
    std::uint64_t n_particles = 0;
    std::uint64_t n_expanded_vertices = 0;
    std::uint64_t n_triangles = 0;
    double pbc_band_width = 0.0;
    double Lx = 0.0;
    double Ly = 0.0;
    std::int64_t global_step = -1;
    std::int64_t phase_step = -1;
    double global_time = -1.0;
    double phase_time = -1.0;
    std::string phase;
    std::string comment;
    std::vector<std::int32_t> particle_indices;
    std::vector<std::int8_t> shift_x;
    std::vector<std::int8_t> shift_y;
    std::vector<std::int32_t> triangles;
};

struct TriangleKey {
    std::int32_t a = -1;
    std::int32_t b = -1;
    std::int32_t c = -1;

    bool operator==(const TriangleKey& other) const {
        return a == other.a && b == other.b && c == other.c;
    }

    bool operator<(const TriangleKey& other) const {
        if (a != other.a) return a < other.a;
        if (b != other.b) return b < other.b;
        return c < other.c;
    }
};

struct TriangleKeyHash {
    std::size_t operator()(const TriangleKey& key) const {
        std::size_t h = 1469598103934665603ull;
        auto mix = [&h](std::uint64_t value) {
            h ^= value + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        };
        mix(static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.a)));
        mix(static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.b)));
        mix(static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.c)));
        return h;
    }
};

struct EdgeKey {
    std::int32_t a = -1;
    std::int32_t b = -1;

    bool operator==(const EdgeKey& other) const {
        return a == other.a && b == other.b;
    }
};

struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& key) const {
        std::size_t h = 1469598103934665603ull;
        auto mix = [&h](std::uint64_t value) {
            h ^= value + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        };
        mix(static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.a)));
        mix(static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.b)));
        return h;
    }
};

struct CanonicalTriangle {
    TriangleKey key;
    int type_code = -1;
    double area = 0.0;
    std::array<int, 4> neighbor_type_counts{{0, 0, 0, 0}};
};

struct RawTriangleMetrics {
    std::vector<std::uint8_t> valid;
    std::vector<int> type_code;
    std::vector<double> area;
    std::vector<TriangleKey> key;
};

const char* device_name(DeviceMode mode) {
    switch (mode) {
    case DeviceMode::Cpu:
        return "cpu";
    case DeviceMode::Gpu:
        return "gpu";
    default:
        return "unknown";
    }
}

__host__ __device__ int triangle_type_from_a_count(int a_count) {
    switch (a_count) {
    case 3:
        return 0; // AAA
    case 2:
        return 1; // AAB
    case 1:
        return 2; // ABB
    case 0:
        return 3; // BBB
    default:
        return -1;
    }
}

const char* triangle_type_name(int type_code) {
    switch (type_code) {
    case 0:
        return "AAA";
    case 1:
        return "AAB";
    case 2:
        return "ABB";
    case 3:
        return "BBB";
    default:
        return "INVALID";
    }
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

void print_usage(const char* argv0) {
    fmt::print(
        "Usage: {} --xyz PATH [--tri2d PATH] [--output PATH] [--device cpu|gpu] [--overwrite] [--max-frames N]\n",
        argv0);
}

DeviceMode parse_device_mode(const std::string& value) {
    if (value == "cpu" || value == "CPU") {
        return DeviceMode::Cpu;
    }
    if (value == "gpu" || value == "GPU") {
        return DeviceMode::Gpu;
    }
    throw std::invalid_argument(
        fmt::format("Unsupported --device '{}'; expected cpu or gpu", value));
}

Options parse_args(int argc, char** argv) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--xyz") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--xyz requires a path");
            }
            options.xyz_path = fs::path(argv[++i]);
        } else if (arg == "--tri2d") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--tri2d requires a path");
            }
            options.tri2d_path = fs::path(argv[++i]);
        } else if (arg == "--output") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--output requires a path");
            }
            options.output_path = fs::path(argv[++i]);
        } else if (arg == "--device") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--device requires a value");
            }
            options.device = parse_device_mode(argv[++i]);
        } else if (arg == "--overwrite") {
            options.overwrite = true;
        } else if (arg == "--max-frames") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--max-frames requires a value");
            }
            options.max_frames = static_cast<std::uint64_t>(std::stoull(argv[++i]));
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
    if (options.tri2d_path.empty()) {
        options.tri2d_path = options.xyz_path;
        options.tri2d_path += ".tri2d";
    }
    if (options.output_path.empty()) {
        options.output_path = options.xyz_path.parent_path() / "aab_abb_triangle_neighbors.csv";
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

XyzParticle parse_particle_line(const std::string& line, std::size_t line_number) {
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
            fmt::format("Malformed XYZ particle line {}: missing type token", line_number));
    }

    XyzParticle particle;
    particle.type = parse_particle_type(
        std::string_view(token_begin, static_cast<std::size_t>(cursor - token_begin)));
    if (!parse_next_double(cursor, particle.x) || !parse_next_double(cursor, particle.y)) {
        throw std::runtime_error(
            fmt::format("Malformed XYZ particle line {}: expected x and y coordinates",
                        line_number));
    }
    return particle;
}

bool read_next_xyz_frame(std::istream& in,
                         std::vector<XyzParticle>& particles,
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
            fmt::format("Expected XYZ particle count at line {}, got '{}'",
                        line_number,
                        line));
    }

    if (!std::getline(in, line)) {
        throw std::runtime_error("Unexpected EOF while reading XYZ comment line");
    }
    ++line_number;

    particles.clear();
    particles.reserve(n_particles);

    for (std::size_t i = 0; i < n_particles; ++i) {
        if (!std::getline(in, line)) {
            throw std::runtime_error(
                fmt::format("Unexpected EOF while reading XYZ particle {} of {}",
                            i + 1,
                            n_particles));
        }
        ++line_number;
        particles.push_back(parse_particle_line(line, line_number));
    }

    return true;
}

template <typename T>
T read_pod(std::istream& in) {
    T value {};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("Unexpected EOF while reading binary payload");
    }
    return value;
}

std::string read_string(std::istream& in) {
    const std::uint32_t length = read_pod<std::uint32_t>(in);
    std::string value(length, '\0');
    if (length > 0) {
        in.read(value.data(), static_cast<std::streamsize>(length));
        if (!in) {
            throw std::runtime_error("Unexpected EOF while reading binary string");
        }
    }
    return value;
}

template <typename T>
std::vector<T> read_vector(std::istream& in, std::size_t count) {
    std::vector<T> values(count);
    if (count > 0) {
        in.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(count * sizeof(T)));
        if (!in) {
            throw std::runtime_error("Unexpected EOF while reading binary vector");
        }
    }
    return values;
}

Tri2dHeader read_tri2d_header(std::istream& in) {
    char magic[sizeof(kFileMagic)];
    in.read(magic, sizeof(magic));
    if (!in) {
        throw std::runtime_error("Failed to read TRI2D file magic");
    }
    if (std::memcmp(magic, kFileMagic, sizeof(kFileMagic)) != 0) {
        throw std::runtime_error("Input TRI2D file has an unexpected magic header");
    }

    Tri2dHeader header;
    header.version = read_pod<std::uint32_t>(in);
    header.frame_count = read_pod<std::uint64_t>(in);
    header.backend_code = read_pod<std::uint8_t>(in);
    header.pbc_band_min_box_fraction = read_pod<double>(in);
    header.xyz_path = read_string(in);
    header.output_path = read_string(in);
    return header;
}

bool read_tri2d_frame(std::istream& in, Tri2dFrame& frame) {
    if (in.peek() == std::char_traits<char>::eof()) {
        return false;
    }

    const std::uint32_t magic = read_pod<std::uint32_t>(in);
    if (magic != kFrameMagic) {
        throw std::runtime_error(
            fmt::format("Unexpected TRI2D frame magic 0x{:08x}", magic));
    }

    frame = Tri2dFrame {};
    frame.frame_index = read_pod<std::uint64_t>(in);
    frame.n_particles = read_pod<std::uint64_t>(in);
    frame.n_expanded_vertices = read_pod<std::uint64_t>(in);
    frame.n_triangles = read_pod<std::uint64_t>(in);
    frame.pbc_band_width = read_pod<double>(in);
    frame.Lx = read_pod<double>(in);
    frame.Ly = read_pod<double>(in);
    frame.global_step = read_pod<std::int64_t>(in);
    frame.phase_step = read_pod<std::int64_t>(in);
    frame.global_time = read_pod<double>(in);
    frame.phase_time = read_pod<double>(in);
    frame.phase = read_string(in);
    frame.comment = read_string(in);
    frame.particle_indices = read_vector<std::int32_t>(in, frame.n_expanded_vertices);
    frame.shift_x = read_vector<std::int8_t>(in, frame.n_expanded_vertices);
    frame.shift_y = read_vector<std::int8_t>(in, frame.n_expanded_vertices);
    frame.triangles = read_vector<std::int32_t>(in, frame.n_triangles * 3);
    return true;
}

template <typename T>
void write_csv_value(std::ostream& out, const T& value) {
    out << value;
}

std::string temperature_label_from_path(const fs::path& xyz_path) {
    const std::string parent_name = xyz_path.parent_path().filename().string();
    if (parent_name.rfind("T=", 0) == 0 && parent_name.size() > 2) {
        return parent_name.substr(2);
    }
    return parent_name.empty() ? "unknown" : parent_name;
}

void check_cuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            fmt::format("{} failed: {}", context, cudaGetErrorString(status)));
    }
}

void ensure_cuda_device() {
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        throw std::runtime_error("No CUDA devices are visible to the analyzer");
    }
    check_cuda(cudaSetDevice(0), "cudaSetDevice(0)");
}

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    std::size_t count = 0;

    void allocate(std::size_t n) {
        reset();
        count = n;
        if (count == 0) {
            return;
        }
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)),
                   "cudaMalloc");
    }

    void copy_from_host(const T* host_ptr) {
        if (count == 0) {
            return;
        }
        check_cuda(cudaMemcpy(ptr,
                              host_ptr,
                              count * sizeof(T),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpyHostToDevice");
    }

    void copy_to_host(T* host_ptr) const {
        if (count == 0) {
            return;
        }
        check_cuda(cudaMemcpy(host_ptr,
                              ptr,
                              count * sizeof(T),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpyDeviceToHost");
    }

    void reset() {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
        count = 0;
    }

    ~DeviceBuffer() {
        reset();
    }
};

__host__ __device__ inline void sort3_int32(std::int32_t& a,
                                            std::int32_t& b,
                                            std::int32_t& c) {
    if (a > b) {
        const auto t = a;
        a = b;
        b = t;
    }
    if (b > c) {
        const auto t = b;
        b = c;
        c = t;
    }
    if (a > b) {
        const auto t = a;
        a = b;
        b = t;
    }
}

__global__ void classify_raw_triangles_kernel(const double* particle_x,
                                              const double* particle_y,
                                              const int* particle_type,
                                              std::size_t particle_count,
                                              const std::int32_t* vertex_particle_indices,
                                              const std::int8_t* shift_x,
                                              const std::int8_t* shift_y,
                                              std::size_t vertex_count,
                                              const std::int32_t* triangles,
                                              std::size_t raw_triangle_count,
                                              double Lx,
                                              double Ly,
                                              std::uint8_t* out_valid,
                                              int* out_type_code,
                                              double* out_area,
                                              std::int32_t* out_sorted_particles) {
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= raw_triangle_count) {
        return;
    }

    const std::int32_t v0 = triangles[3 * tid + 0];
    const std::int32_t v1 = triangles[3 * tid + 1];
    const std::int32_t v2 = triangles[3 * tid + 2];

    out_valid[tid] = 0;
    out_type_code[tid] = -1;
    out_area[tid] = 0.0;
    out_sorted_particles[3 * tid + 0] = -1;
    out_sorted_particles[3 * tid + 1] = -1;
    out_sorted_particles[3 * tid + 2] = -1;

    if (v0 < 0 || v1 < 0 || v2 < 0 ||
        static_cast<std::size_t>(v0) >= vertex_count ||
        static_cast<std::size_t>(v1) >= vertex_count ||
        static_cast<std::size_t>(v2) >= vertex_count) {
        return;
    }

    const std::int32_t p0 = vertex_particle_indices[v0];
    const std::int32_t p1 = vertex_particle_indices[v1];
    const std::int32_t p2 = vertex_particle_indices[v2];
    if (p0 < 0 || p1 < 0 || p2 < 0 ||
        static_cast<std::size_t>(p0) >= particle_count ||
        static_cast<std::size_t>(p1) >= particle_count ||
        static_cast<std::size_t>(p2) >= particle_count) {
        return;
    }
    if (p0 == p1 || p1 == p2 || p0 == p2) {
        return;
    }

    const int t0 = particle_type[p0];
    const int t1 = particle_type[p1];
    const int t2 = particle_type[p2];
    if (!((t0 == 0 || t0 == 1) && (t1 == 0 || t1 == 1) && (t2 == 0 || t2 == 1))) {
        return;
    }

    const double x0 = particle_x[p0] + static_cast<double>(shift_x[v0]) * Lx;
    const double y0 = particle_y[p0] + static_cast<double>(shift_y[v0]) * Ly;
    const double x1 = particle_x[p1] + static_cast<double>(shift_x[v1]) * Lx;
    const double y1 = particle_y[p1] + static_cast<double>(shift_y[v1]) * Ly;
    const double x2 = particle_x[p2] + static_cast<double>(shift_x[v2]) * Lx;
    const double y2 = particle_y[p2] + static_cast<double>(shift_y[v2]) * Ly;

    const double twice_area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    const double area = 0.5 * fabs(twice_area);
    if (!(area > 0.0)) {
        return;
    }

    const int a_count = (t0 == 0) + (t1 == 0) + (t2 == 0);
    const int type_code = triangle_type_from_a_count(a_count);
    if (type_code < 0) {
        return;
    }

    std::int32_t k0 = p0;
    std::int32_t k1 = p1;
    std::int32_t k2 = p2;
    sort3_int32(k0, k1, k2);

    out_valid[tid] = 1;
    out_type_code[tid] = type_code;
    out_area[tid] = area;
    out_sorted_particles[3 * tid + 0] = k0;
    out_sorted_particles[3 * tid + 1] = k1;
    out_sorted_particles[3 * tid + 2] = k2;
}

__global__ void count_neighbor_types_kernel(const int* triangle_type,
                                            const int* neighbor_offsets,
                                            const int* neighbor_indices,
                                            std::size_t triangle_count,
                                            int* out_counts) {
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= triangle_count) {
        return;
    }

    int local_counts[4] = {0, 0, 0, 0};
    const int begin = neighbor_offsets[tid];
    const int end = neighbor_offsets[tid + 1];
    for (int idx = begin; idx < end; ++idx) {
        const int neighbor = neighbor_indices[idx];
        if (neighbor < 0) {
            continue;
        }
        const int type_code = triangle_type[neighbor];
        if (type_code >= 0 && type_code < 4) {
            ++local_counts[type_code];
        }
    }

    for (int t = 0; t < 4; ++t) {
        out_counts[4 * tid + t] = local_counts[t];
    }
}

RawTriangleMetrics analyze_raw_triangles_cpu(const std::vector<XyzParticle>& particles,
                                            const Tri2dFrame& frame) {
    RawTriangleMetrics metrics;
    const std::size_t raw_triangle_count = frame.n_triangles;
    metrics.valid.assign(raw_triangle_count, 0);
    metrics.type_code.assign(raw_triangle_count, -1);
    metrics.area.assign(raw_triangle_count, 0.0);
    metrics.key.assign(raw_triangle_count, TriangleKey {});

    for (std::size_t tri_idx = 0; tri_idx < raw_triangle_count; ++tri_idx) {
        const auto v0 = frame.triangles[3 * tri_idx + 0];
        const auto v1 = frame.triangles[3 * tri_idx + 1];
        const auto v2 = frame.triangles[3 * tri_idx + 2];
        if (v0 < 0 || v1 < 0 || v2 < 0 ||
            static_cast<std::size_t>(v0) >= frame.particle_indices.size() ||
            static_cast<std::size_t>(v1) >= frame.particle_indices.size() ||
            static_cast<std::size_t>(v2) >= frame.particle_indices.size()) {
            continue;
        }

        const auto p0 = frame.particle_indices[static_cast<std::size_t>(v0)];
        const auto p1 = frame.particle_indices[static_cast<std::size_t>(v1)];
        const auto p2 = frame.particle_indices[static_cast<std::size_t>(v2)];
        if (p0 < 0 || p1 < 0 || p2 < 0 ||
            static_cast<std::size_t>(p0) >= particles.size() ||
            static_cast<std::size_t>(p1) >= particles.size() ||
            static_cast<std::size_t>(p2) >= particles.size()) {
            continue;
        }
        if (p0 == p1 || p1 == p2 || p0 == p2) {
            continue;
        }

        const int t0 = particles[static_cast<std::size_t>(p0)].type;
        const int t1 = particles[static_cast<std::size_t>(p1)].type;
        const int t2 = particles[static_cast<std::size_t>(p2)].type;
        if (!((t0 == 0 || t0 == 1) && (t1 == 0 || t1 == 1) && (t2 == 0 || t2 == 1))) {
            continue;
        }

        const double x0 = particles[static_cast<std::size_t>(p0)].x +
                          static_cast<double>(frame.shift_x[static_cast<std::size_t>(v0)]) * frame.Lx;
        const double y0 = particles[static_cast<std::size_t>(p0)].y +
                          static_cast<double>(frame.shift_y[static_cast<std::size_t>(v0)]) * frame.Ly;
        const double x1 = particles[static_cast<std::size_t>(p1)].x +
                          static_cast<double>(frame.shift_x[static_cast<std::size_t>(v1)]) * frame.Lx;
        const double y1 = particles[static_cast<std::size_t>(p1)].y +
                          static_cast<double>(frame.shift_y[static_cast<std::size_t>(v1)]) * frame.Ly;
        const double x2 = particles[static_cast<std::size_t>(p2)].x +
                          static_cast<double>(frame.shift_x[static_cast<std::size_t>(v2)]) * frame.Lx;
        const double y2 = particles[static_cast<std::size_t>(p2)].y +
                          static_cast<double>(frame.shift_y[static_cast<std::size_t>(v2)]) * frame.Ly;

        const double twice_area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        const double area = 0.5 * std::abs(twice_area);
        if (!(area > 0.0)) {
            continue;
        }

        const int a_count = (t0 == 0) + (t1 == 0) + (t2 == 0);
        const int type_code = triangle_type_from_a_count(a_count);
        if (type_code < 0) {
            continue;
        }

        std::int32_t k0 = p0;
        std::int32_t k1 = p1;
        std::int32_t k2 = p2;
        sort3_int32(k0, k1, k2);

        metrics.valid[tri_idx] = 1;
        metrics.type_code[tri_idx] = type_code;
        metrics.area[tri_idx] = area;
        metrics.key[tri_idx] = TriangleKey {k0, k1, k2};
    }

    return metrics;
}

RawTriangleMetrics analyze_raw_triangles_gpu(const std::vector<XyzParticle>& particles,
                                            const Tri2dFrame& frame) {
    ensure_cuda_device();

    const std::size_t particle_count = particles.size();
    const std::size_t vertex_count = frame.particle_indices.size();
    const std::size_t raw_triangle_count = frame.n_triangles;

    if (raw_triangle_count == 0) {
        return RawTriangleMetrics {};
    }

    std::vector<double> particle_x(particle_count, 0.0);
    std::vector<double> particle_y(particle_count, 0.0);
    std::vector<int> particle_type(particle_count, -1);
    for (std::size_t i = 0; i < particle_count; ++i) {
        particle_x[i] = particles[i].x;
        particle_y[i] = particles[i].y;
        particle_type[i] = particles[i].type;
    }

    DeviceBuffer<double> d_particle_x;
    DeviceBuffer<double> d_particle_y;
    DeviceBuffer<int> d_particle_type;
    DeviceBuffer<std::int32_t> d_vertex_particle_indices;
    DeviceBuffer<std::int8_t> d_shift_x;
    DeviceBuffer<std::int8_t> d_shift_y;
    DeviceBuffer<std::int32_t> d_triangles;
    DeviceBuffer<std::uint8_t> d_valid;
    DeviceBuffer<int> d_type_code;
    DeviceBuffer<double> d_area;
    DeviceBuffer<std::int32_t> d_sorted_particles;

    d_particle_x.allocate(particle_count);
    d_particle_y.allocate(particle_count);
    d_particle_type.allocate(particle_count);
    d_vertex_particle_indices.allocate(vertex_count);
    d_shift_x.allocate(vertex_count);
    d_shift_y.allocate(vertex_count);
    d_triangles.allocate(frame.triangles.size());
    d_valid.allocate(raw_triangle_count);
    d_type_code.allocate(raw_triangle_count);
    d_area.allocate(raw_triangle_count);
    d_sorted_particles.allocate(raw_triangle_count * 3);

    d_particle_x.copy_from_host(particle_x.data());
    d_particle_y.copy_from_host(particle_y.data());
    d_particle_type.copy_from_host(particle_type.data());
    d_vertex_particle_indices.copy_from_host(frame.particle_indices.data());
    d_shift_x.copy_from_host(frame.shift_x.data());
    d_shift_y.copy_from_host(frame.shift_y.data());
    d_triangles.copy_from_host(frame.triangles.data());

    constexpr int kBlockSize = 256;
    const int grid_size =
        static_cast<int>((raw_triangle_count + kBlockSize - 1) / kBlockSize);
    classify_raw_triangles_kernel<<<grid_size, kBlockSize>>>(
        d_particle_x.ptr,
        d_particle_y.ptr,
        d_particle_type.ptr,
        particle_count,
        d_vertex_particle_indices.ptr,
        d_shift_x.ptr,
        d_shift_y.ptr,
        vertex_count,
        d_triangles.ptr,
        raw_triangle_count,
        frame.Lx,
        frame.Ly,
        d_valid.ptr,
        d_type_code.ptr,
        d_area.ptr,
        d_sorted_particles.ptr);
    check_cuda(cudaGetLastError(), "classify_raw_triangles_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "classify_raw_triangles_kernel sync");

    RawTriangleMetrics metrics;
    metrics.valid.resize(raw_triangle_count);
    metrics.type_code.resize(raw_triangle_count);
    metrics.area.resize(raw_triangle_count);
    std::vector<std::int32_t> sorted_particles(raw_triangle_count * 3, -1);
    metrics.key.resize(raw_triangle_count);

    d_valid.copy_to_host(metrics.valid.data());
    d_type_code.copy_to_host(metrics.type_code.data());
    d_area.copy_to_host(metrics.area.data());
    d_sorted_particles.copy_to_host(sorted_particles.data());

    for (std::size_t tri_idx = 0; tri_idx < raw_triangle_count; ++tri_idx) {
        metrics.key[tri_idx] = TriangleKey {
            sorted_particles[3 * tri_idx + 0],
            sorted_particles[3 * tri_idx + 1],
            sorted_particles[3 * tri_idx + 2],
        };
    }

    return metrics;
}

std::vector<CanonicalTriangle> deduplicate_triangles(const RawTriangleMetrics& raw_metrics) {
    std::unordered_map<TriangleKey, CanonicalTriangle, TriangleKeyHash> unique;
    unique.reserve(raw_metrics.key.size());

    for (std::size_t raw_idx = 0; raw_idx < raw_metrics.key.size(); ++raw_idx) {
        if (raw_metrics.valid[raw_idx] == 0) {
            continue;
        }

        const TriangleKey key = raw_metrics.key[raw_idx];
        auto [it, inserted] = unique.emplace(
            key,
            CanonicalTriangle {key, raw_metrics.type_code[raw_idx], raw_metrics.area[raw_idx]});
        if (!inserted) {
            if (it->second.type_code != raw_metrics.type_code[raw_idx]) {
                throw std::runtime_error(
                    fmt::format("Duplicate physical triangle ({}, {}, {}) had inconsistent type codes",
                                key.a,
                                key.b,
                                key.c));
            }
            it->second.area = std::min(it->second.area, raw_metrics.area[raw_idx]);
        }
    }

    std::vector<CanonicalTriangle> triangles;
    triangles.reserve(unique.size());
    for (auto& entry : unique) {
        triangles.push_back(entry.second);
    }
    std::sort(triangles.begin(),
              triangles.end(),
              [](const CanonicalTriangle& lhs, const CanonicalTriangle& rhs) {
                  return lhs.key < rhs.key;
              });
    return triangles;
}

EdgeKey make_edge(std::int32_t a, std::int32_t b) {
    if (a > b) {
        std::swap(a, b);
    }
    return EdgeKey {a, b};
}

std::vector<std::vector<int>> build_adjacency(const std::vector<CanonicalTriangle>& triangles) {
    std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash> edge_to_triangles;
    edge_to_triangles.reserve(triangles.size() * 3);

    for (std::size_t tri_idx = 0; tri_idx < triangles.size(); ++tri_idx) {
        const auto& key = triangles[tri_idx].key;
        edge_to_triangles[make_edge(key.a, key.b)].push_back(static_cast<int>(tri_idx));
        edge_to_triangles[make_edge(key.b, key.c)].push_back(static_cast<int>(tri_idx));
        edge_to_triangles[make_edge(key.a, key.c)].push_back(static_cast<int>(tri_idx));
    }

    std::vector<std::vector<int>> adjacency(triangles.size());
    for (const auto& entry : edge_to_triangles) {
        const auto& tri_indices = entry.second;
        for (std::size_t i = 0; i < tri_indices.size(); ++i) {
            for (std::size_t j = i + 1; j < tri_indices.size(); ++j) {
                adjacency[static_cast<std::size_t>(tri_indices[i])].push_back(tri_indices[j]);
                adjacency[static_cast<std::size_t>(tri_indices[j])].push_back(tri_indices[i]);
            }
        }
    }

    for (auto& neighbors : adjacency) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }

    return adjacency;
}

void count_neighbor_types_cpu(std::vector<CanonicalTriangle>& triangles,
                              const std::vector<std::vector<int>>& adjacency) {
    for (std::size_t tri_idx = 0; tri_idx < triangles.size(); ++tri_idx) {
        auto& counts = triangles[tri_idx].neighbor_type_counts;
        counts = {0, 0, 0, 0};
        for (const int neighbor_idx : adjacency[tri_idx]) {
            if (neighbor_idx < 0 ||
                static_cast<std::size_t>(neighbor_idx) >= triangles.size()) {
                continue;
            }
            const int neighbor_type = triangles[static_cast<std::size_t>(neighbor_idx)].type_code;
            if (neighbor_type >= 0 && neighbor_type < 4) {
                ++counts[static_cast<std::size_t>(neighbor_type)];
            }
        }
    }
}

void count_neighbor_types_gpu(std::vector<CanonicalTriangle>& triangles,
                              const std::vector<std::vector<int>>& adjacency) {
    ensure_cuda_device();

    const std::size_t triangle_count = triangles.size();
    if (triangle_count == 0) {
        return;
    }
    std::vector<int> triangle_type(triangle_count, -1);
    std::vector<int> offsets(triangle_count + 1, 0);
    std::size_t total_neighbors = 0;
    for (std::size_t tri_idx = 0; tri_idx < triangle_count; ++tri_idx) {
        triangle_type[tri_idx] = triangles[tri_idx].type_code;
        offsets[tri_idx] = static_cast<int>(total_neighbors);
        total_neighbors += adjacency[tri_idx].size();
    }
    offsets[triangle_count] = static_cast<int>(total_neighbors);

    std::vector<int> flat_neighbors(total_neighbors, -1);
    std::size_t cursor = 0;
    for (const auto& neighbors : adjacency) {
        for (const int neighbor : neighbors) {
            flat_neighbors[cursor++] = neighbor;
        }
    }

    DeviceBuffer<int> d_triangle_type;
    DeviceBuffer<int> d_offsets;
    DeviceBuffer<int> d_neighbors;
    DeviceBuffer<int> d_counts;
    d_triangle_type.allocate(triangle_count);
    d_offsets.allocate(offsets.size());
    d_neighbors.allocate(flat_neighbors.size());
    d_counts.allocate(triangle_count * 4);

    d_triangle_type.copy_from_host(triangle_type.data());
    d_offsets.copy_from_host(offsets.data());
    if (!flat_neighbors.empty()) {
        d_neighbors.copy_from_host(flat_neighbors.data());
    }

    constexpr int kBlockSize = 256;
    const int grid_size =
        static_cast<int>((triangle_count + kBlockSize - 1) / kBlockSize);
    count_neighbor_types_kernel<<<grid_size, kBlockSize>>>(
        d_triangle_type.ptr,
        d_offsets.ptr,
        d_neighbors.ptr,
        triangle_count,
        d_counts.ptr);
    check_cuda(cudaGetLastError(), "count_neighbor_types_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "count_neighbor_types_kernel sync");

    std::vector<int> host_counts(triangle_count * 4, 0);
    d_counts.copy_to_host(host_counts.data());
    for (std::size_t tri_idx = 0; tri_idx < triangle_count; ++tri_idx) {
        for (int t = 0; t < 4; ++t) {
            triangles[tri_idx].neighbor_type_counts[static_cast<std::size_t>(t)] =
                host_counts[4 * tri_idx + t];
        }
    }
}

void count_neighbor_types(DeviceMode device,
                          std::vector<CanonicalTriangle>& triangles,
                          const std::vector<std::vector<int>>& adjacency) {
    if (device == DeviceMode::Gpu) {
        count_neighbor_types_gpu(triangles, adjacency);
    } else {
        count_neighbor_types_cpu(triangles, adjacency);
    }
}

RawTriangleMetrics analyze_raw_triangles(DeviceMode device,
                                         const std::vector<XyzParticle>& particles,
                                         const Tri2dFrame& frame) {
    if (device == DeviceMode::Gpu) {
        return analyze_raw_triangles_gpu(particles, frame);
    }
    return analyze_raw_triangles_cpu(particles, frame);
}

void write_csv_header(std::ostream& out) {
    out << "temperature"
        << ",frame_index"
        << ",global_step"
        << ",phase_step"
        << ",global_time"
        << ",phase_time"
        << ",phase"
        << ",triangle_index"
        << ",triangle_type"
        << ",particle_i"
        << ",particle_j"
        << ",particle_k"
        << ",area"
        << ",n_neighbors_total"
        << ",near_AAA"
        << ",near_AAB"
        << ",near_ABB"
        << ",near_BBB"
        << '\n';
}

std::size_t write_mixed_triangle_rows(std::ostream& out,
                                      const std::string& temperature_label,
                                      const Tri2dFrame& frame,
                                      const std::vector<CanonicalTriangle>& triangles) {
    std::size_t written = 0;
    std::size_t mixed_index = 0;
    for (const auto& triangle : triangles) {
        if (!(triangle.type_code == 1 || triangle.type_code == 2)) {
            continue;
        }
        const int near_aaa = triangle.neighbor_type_counts[0];
        const int near_aab = triangle.neighbor_type_counts[1];
        const int near_abb = triangle.neighbor_type_counts[2];
        const int near_bbb = triangle.neighbor_type_counts[3];
        const int n_neighbors_total = near_aaa + near_aab + near_abb + near_bbb;

        out << temperature_label
            << ',' << frame.frame_index
            << ',' << frame.global_step
            << ',' << frame.phase_step
            << ',' << frame.global_time
            << ',' << frame.phase_time
            << ',' << frame.phase
            << ',' << mixed_index
            << ',' << triangle_type_name(triangle.type_code)
            << ',' << triangle.key.a
            << ',' << triangle.key.b
            << ',' << triangle.key.c
            << ',' << triangle.area
            << ',' << n_neighbors_total
            << ',' << near_aaa
            << ',' << near_aab
            << ',' << near_abb
            << ',' << near_bbb
            << '\n';
        ++mixed_index;
        ++written;
    }
    return written;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::ios::sync_with_stdio(false);

        const Options options = parse_args(argc, argv);
        if (!fs::exists(options.xyz_path)) {
            throw std::runtime_error(
                fmt::format("XYZ file '{}' does not exist", options.xyz_path.string()));
        }
        if (!fs::is_regular_file(options.xyz_path)) {
            throw std::runtime_error(
                fmt::format("'{}' is not a regular file", options.xyz_path.string()));
        }
        if (!fs::exists(options.tri2d_path)) {
            throw std::runtime_error(
                fmt::format("TRI2D file '{}' does not exist", options.tri2d_path.string()));
        }
        if (!fs::is_regular_file(options.tri2d_path)) {
            throw std::runtime_error(
                fmt::format("'{}' is not a regular file", options.tri2d_path.string()));
        }
        if (fs::exists(options.output_path) && !options.overwrite) {
            fmt::print("[analysis_triangle_neighborhoods] Output already exists, skipping: {}\n",
                       options.output_path.string());
            return 0;
        }

        std::ifstream xyz_in(options.xyz_path);
        if (!xyz_in) {
            throw std::runtime_error(
                fmt::format("Failed to open XYZ file '{}'", options.xyz_path.string()));
        }
        std::ifstream tri_in(options.tri2d_path, std::ios::binary);
        if (!tri_in) {
            throw std::runtime_error(
                fmt::format("Failed to open TRI2D file '{}'", options.tri2d_path.string()));
        }

        const Tri2dHeader header = read_tri2d_header(tri_in);
        if (header.version != 2) {
            throw std::runtime_error(
                fmt::format("Unsupported TRI2D version {}; expected 2", header.version));
        }

        if (options.device == DeviceMode::Gpu) {
            ensure_cuda_device();
        }

        if (!options.output_path.parent_path().empty()) {
            fs::create_directories(options.output_path.parent_path());
        }
        const fs::path tmp_output_path = options.output_path.string() + ".tmp";
        std::ofstream out(tmp_output_path, std::ios::out | std::ios::trunc);
        if (!out) {
            throw std::runtime_error(
                fmt::format("Failed to open output '{}'", tmp_output_path.string()));
        }
        write_csv_header(out);

        const std::string temperature_label = temperature_label_from_path(options.xyz_path);
        fmt::print("[analysis_triangle_neighborhoods] XYZ: {}\n", options.xyz_path.string());
        fmt::print("[analysis_triangle_neighborhoods] TRI2D: {}\n", options.tri2d_path.string());
        fmt::print("[analysis_triangle_neighborhoods] Output: {}\n", options.output_path.string());
        fmt::print("[analysis_triangle_neighborhoods] Device: {}\n", device_name(options.device));
        fmt::print("[analysis_triangle_neighborhoods] TRI2D frames: {}\n", header.frame_count);
        if (options.max_frames > 0) {
            fmt::print("[analysis_triangle_neighborhoods] Max frames: {}\n", options.max_frames);
        }

        std::vector<XyzParticle> particles;
        Tri2dFrame frame;
        std::size_t xyz_line_number = 0;
        std::uint64_t processed_frames = 0;
        std::uint64_t total_unique_triangles = 0;
        std::uint64_t total_mixed_rows = 0;
        std::uint64_t total_raw_triangles = 0;

        const auto start_time = std::chrono::steady_clock::now();

        const std::uint64_t target_frames =
            options.max_frames > 0 ? std::min(header.frame_count, options.max_frames)
                                   : header.frame_count;

        while (processed_frames < target_frames) {
            if (!read_next_xyz_frame(xyz_in, particles, xyz_line_number)) {
                throw std::runtime_error(
                    fmt::format("XYZ file ended early after {} frames", processed_frames));
            }
            if (!read_tri2d_frame(tri_in, frame)) {
                throw std::runtime_error(
                    fmt::format("TRI2D file ended early after {} frames", processed_frames));
            }
            if (particles.size() != frame.n_particles) {
                throw std::runtime_error(
                    fmt::format("Frame {} particle mismatch: XYZ has {}, TRI2D says {}",
                                processed_frames,
                                particles.size(),
                                frame.n_particles));
            }

            const RawTriangleMetrics raw_metrics =
                analyze_raw_triangles(options.device, particles, frame);
            std::vector<CanonicalTriangle> triangles = deduplicate_triangles(raw_metrics);
            const auto adjacency = build_adjacency(triangles);
            count_neighbor_types(options.device, triangles, adjacency);

            total_raw_triangles += frame.n_triangles;
            total_unique_triangles += triangles.size();
            total_mixed_rows += write_mixed_triangle_rows(out, temperature_label, frame, triangles);

            if (processed_frames == 0 || ((processed_frames + 1) % 25 == 0)) {
                const std::chrono::duration<double> elapsed =
                    std::chrono::steady_clock::now() - start_time;
                std::size_t mixed_this_frame = 0;
                for (const auto& triangle : triangles) {
                    if (triangle.type_code == 1 || triangle.type_code == 2) {
                        ++mixed_this_frame;
                    }
                }
                fmt::print(
                    "[analysis_triangle_neighborhoods] frame {} phase={} raw_triangles={} unique_triangles={} mixed_rows={} elapsed={:.1f}s\n",
                    frame.frame_index,
                    frame.phase.empty() ? "UNKNOWN" : frame.phase,
                    frame.n_triangles,
                    triangles.size(),
                    mixed_this_frame,
                    elapsed.count());
            }

            ++processed_frames;
        }

        if (target_frames == header.frame_count) {
            std::size_t trailing_xyz_line = xyz_line_number;
            std::vector<XyzParticle> trailing_particles;
            if (read_next_xyz_frame(xyz_in, trailing_particles, trailing_xyz_line)) {
                throw std::runtime_error("XYZ file contains more frames than the TRI2D header");
            }
        }

        out.flush();
        if (!out) {
            throw std::runtime_error("Failed while flushing CSV output");
        }
        out.close();
        if (fs::exists(options.output_path)) {
            fs::remove(options.output_path);
        }
        fs::rename(tmp_output_path, options.output_path);

        const std::chrono::duration<double> elapsed =
            std::chrono::steady_clock::now() - start_time;
        const double avg_unique =
            processed_frames == 0
                ? 0.0
                : static_cast<double>(total_unique_triangles) /
                      static_cast<double>(processed_frames);
        const double avg_mixed =
            processed_frames == 0
                ? 0.0
                : static_cast<double>(total_mixed_rows) /
                      static_cast<double>(processed_frames);

        fmt::print(
            "[analysis_triangle_neighborhoods] Done. frames={} raw_triangles={} unique_triangles={} mixed_rows={} avg_unique={:.1f} avg_mixed={:.1f} elapsed={:.1f}s\n",
            processed_frames,
            total_raw_triangles,
            total_unique_triangles,
            total_mixed_rows,
            avg_unique,
            avg_mixed,
            elapsed.count());
        return 0;
    } catch (const std::exception& ex) {
        fmt::print(stderr, "[analysis_triangle_neighborhoods] {}\n", ex.what());
        return 1;
    }
}
