#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <numeric>
#include <random>
#include <string>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <zlib.h>
#include <cstdint> 
#include <getopt.h>
#include <algorithm>
#include <cstring>
#include <climits>
#include <complex>
#include <cstdio>
#include <unordered_map>
#include <array>
#include <limits>
#include <tuple>
#include <sstream>
#include <sciplot/sciplot.hpp>
#include <unordered_set>
#include <cerrno>
#include <unistd.h>

namespace scp = sciplot;

struct Particle {
    double2 pos;
    double2 vel;
    double2 acc;
    int type;
};

#pragma pack(push,1)
struct FileHeader {
    char     magic[8] = {'M','D','G','Z','D','A','T','A'};
    uint32_t version  = 1;
    uint32_t n_particles;
    double    box_w, box_h;
};

struct FrameHeader {
    int32_t step;
    double   sim_time;
    double   dt;
};

struct PackedParticle {
    double   x, y, vx, vy, ax, ay;
    uint8_t type;
};
#pragma pack(pop)

static inline bool gz_read_exact(gzFile gz, void* dst, size_t nbytes) {
    unsigned char* p = static_cast<unsigned char*>(dst);
    size_t done = 0;
    while (done < nbytes) {
        const unsigned chunk = static_cast<unsigned>(std::min(nbytes - done, static_cast<size_t>(INT_MAX)));
        const int got = gzread(gz, p + done, chunk);
        if (got <= 0) return false;
        done += static_cast<size_t>(got);
    }
    return true;
}

struct MDGZFrameIterator {
    MDGZFrameIterator() : gz_(nullptr), out_(nullptr), opened_(false), idx_(0) {
        std::memset(&file_header_, 0, sizeof(file_header_));
        std::memset(&last_frame_hdr_, 0, sizeof(last_frame_hdr_));
    }
    ~MDGZFrameIterator() { close(); }

    bool open(const std::string& filepath, std::vector<Particle>& out_host_buffer) {
        close();
        path_ = filepath;
        out_  = &out_host_buffer;

        gz_ = gzopen(filepath.c_str(), "rb");
        if (!gz_) {
            std::fprintf(stderr, "MDGZFrameIterator::open: gzopen failed: %s\n", filepath.c_str());
            return false;
        }

        if (!gz_read_exact(gz_, &file_header_, sizeof(file_header_))) {
            std::fprintf(stderr, "MDGZFrameIterator::open: Failed to read FileHeader\n");
            close();
            return false;
        }
        static const char expected_magic[8] = {'M','D','G','Z','D','A','T','A'};
        if (std::memcmp(file_header_.magic, expected_magic, 8) != 0) {
            std::fprintf(stderr, "MDGZFrameIterator::open: Bad magic in header\n");
            close();
            return false;
        }
        if (file_header_.version != 1u) {
            std::fprintf(stderr, "MDGZFrameIterator::open: Unsupported version: %u\n", file_header_.version);
            close();
            return false;
        }
        if (file_header_.n_particles == 0u) {
            std::fprintf(stderr, "MDGZFrameIterator::open: Header says zero particles\n");
            close();
            return false;
        }

        opened_ = true;
        idx_ = 0;
        std::memset(&last_frame_hdr_, 0, sizeof(last_frame_hdr_));
        return true;
    }

    bool next() {
        if (!opened_ || !out_) return false;

        // FrameHeader
        FrameHeader fh{};
        if (!gz_read_exact(gz_, &fh, sizeof(fh))) {
            // EOF or truncated tail
            close();
            return false;
        }

        // Particle payload
        const uint32_t N = file_header_.n_particles;
        out_->resize(static_cast<size_t>(N));
        if (!gz_read_exact(gz_, out_->data(), static_cast<size_t>(N) * sizeof(Particle))) {
            // truncated payload; treat as EOF and close
            close();
            return false;
        }

        last_frame_hdr_ = fh;
        ++idx_;
        return true;
    }

    void close() {
        if (gz_) { gzclose(gz_); gz_ = nullptr; }
        opened_ = false;
        out_ = nullptr;
        idx_ = 0;
        path_.clear();
        std::memset(&file_header_, 0, sizeof(file_header_));
        std::memset(&last_frame_hdr_, 0, sizeof(last_frame_hdr_));
    }

    // Accessors
    bool is_open() const { return opened_; }
    uint64_t frame_index() const { return idx_; }                // 1-based after first next()
    const FileHeader& file_header() const { return file_header_; }
    const FrameHeader& frame_header() const { return last_frame_hdr_; }
    const std::string& path() const { return path_; }

private:
    gzFile gz_;
    std::vector<Particle>* out_;
    bool opened_;
    uint64_t idx_;
    std::string path_;
    FileHeader  file_header_;
    FrameHeader last_frame_hdr_;
};

struct EdgeKey {
    int a, b;
    EdgeKey() : a(0), b(0) {}
    EdgeKey(int u, int v) {
        if (u < v) { a = u; b = v; } else { a = v; b = u; }
    }
    bool operator==(const EdgeKey& o) const noexcept { return a == o.a && b == o.b; }
};
struct EdgeKeyHash {
    size_t operator()(const EdgeKey& e) const noexcept {
        return (static_cast<size_t>(e.a) * 1315423911u) ^ (static_cast<size_t>(e.b) * 2654435761u);
    }
};



__device__ __forceinline__ double2 circumcenter_dev(const double2& A, const double2& B, const double2& C) {
    const double ax = A.x, ay = A.y;
    const double bx = B.x, by = B.y;
    const double cx = C.x, cy = C.y;
    const double a2 = ax*ax + ay*ay;
    const double b2 = bx*bx + by*by;
    const double c2 = cx*cx + cy*cy;
    const double d  = 2.0 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by));
    if (fabs(d) < 1e-24) {
        return make_double2(NAN, NAN);
    }
    const double ux = (a2*(by - cy) + b2*(cy - ay) + c2*(ay - by)) / d;
    const double uy = (a2*(cx - bx) + b2*(ax - cx) + c2*(bx - ax)) / d;
    return make_double2(ux, uy);
}



__global__ void ab_dual_voronoi_length_kernel(
    const double2* __restrict__ d_pos,    // N
    const int3*    __restrict__ d_tris,   // M triangles
    const int2*    __restrict__ d_edge_tris, // E AB edges mapped to (tL, tR)
    int n_edges,
    double Lx, double Ly,
    double* __restrict__ d_sum_len
){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_edges) return;

    const int2 tt = d_edge_tris[i];
    const int t1 = tt.x;
    const int t2 = tt.y;
    if (t1 < 0 || t2 < 0) return;

    const int3 T1 = d_tris[t1];
    const int3 T2 = d_tris[t2];

    const double2 A1 = d_pos[T1.x];
    const double2 B1 = d_pos[T1.y];
    const double2 C1 = d_pos[T1.z];
    const double2 A2 = d_pos[T2.x];
    const double2 B2 = d_pos[T2.y];
    const double2 C2 = d_pos[T2.z];

    const double2 cc1 = circumcenter_dev(A1, B1, C1);
    const double2 cc2 = circumcenter_dev(A2, B2, C2);
    if (isnan(cc1.x) || isnan(cc2.x)) return;

    double dx = cc2.x - cc1.x;
    double dy = cc2.y - cc1.y;
    // minimum-image for box
    if (dx >  0.5 * Lx) dx -= Lx; else if (dx < -0.5 * Lx) dx += Lx;
    if (dy >  0.5 * Ly) dy -= Ly; else if (dy < -0.5 * Ly) dy += Ly;

    const double seg = sqrt(dx*dx + dy*dy);
    atomicAdd(d_sum_len, seg);
}





static inline double h_orient(const double2& a, const double2& b, const double2& c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}
static inline double h_incircle(const double2& a, const double2& b, const double2& c, const double2& p) {
    // Returns positive if p is inside circumcircle of (a,b,c) when (a,b,c) is CCW
    const double ax = a.x - p.x, ay = a.y - p.y;
    const double bx = b.x - p.x, by = b.y - p.y;
    const double cx = c.x - p.x, cy = c.y - p.y;
    const double a2 = ax*ax + ay*ay;
    const double b2 = bx*bx + by*by;
    const double c2 = cx*cx + cy*cy;
    const double det = ax * (by * c2 - b2 * cy)
                     - ay * (bx * c2 - b2 * cx)
                     + a2 * (bx * cy - by * cx);
    const double ori = h_orient(a, b, c);
    return (ori > 0.0) ? det : -det;
}


__global__ void zero_scalar(double* x) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *x = 0.0;
}

static void bowyer_watson_cpu(
    const std::vector<double2>& pts,
    std::vector<std::array<int,3>>& out_tris // CCW triples using original indices
){
    const int n = (int)pts.size();
    out_tris.clear();
    if (n < 3) return;

    // Super triangle
    double minx = pts[0].x, maxx = pts[0].x, miny = pts[0].y, maxy = pts[0].y;
    for (const auto& p : pts) {
        minx = std::min(minx, p.x); maxx = std::max(maxx, p.x);
        miny = std::min(miny, p.y); maxy = std::max(maxy, p.y);
    }
    const double dx = maxx - minx, dy = maxy - miny;
    const double delta = std::max(dx, dy);
    const double cx = 0.5 * (minx + maxx);
    const double cy = 0.5 * (miny + maxy);

    std::vector<double2> P = pts;
    const int iA = (int)P.size(); P.push_back(make_double2(cx - 2.0*delta, cy - delta));
    const int iB = (int)P.size(); P.push_back(make_double2(cx,             cy + 2.0*delta));
    const int iC = (int)P.size(); P.push_back(make_double2(cx + 2.0*delta, cy - delta));

    std::vector<std::array<int,3>> tris_ext;
    tris_ext.push_back({iA, iB, iC});

    // Insert each real point
    for (int pi = 0; pi < n; ++pi) {
        const double2& p = P[pi];

        // 1) find bad triangles
        std::vector<int> bad;
        bad.reserve(tris_ext.size()/2);
        for (int t = 0; t < (int)tris_ext.size(); ++t) {
            auto tri = tris_ext[t];
            if (h_incircle(P[tri[0]], P[tri[1]], P[tri[2]], p) > 0.0) {
                bad.push_back(t);
            }
        }

        // 2) boundary of cavity (edges that appear exactly once among bad tris)
        std::unordered_map<EdgeKey,int,EdgeKeyHash> edge_count;
        edge_count.reserve(bad.size()*3);
        auto add_edge = [&](int u, int v){
            EdgeKey e(u,v);
            auto it = edge_count.find(e);
            if (it == edge_count.end()) edge_count.emplace(e, 1);
            else it->second += 1;
        };
        for (int bi : bad) {
            auto tri = tris_ext[bi];
            add_edge(tri[0], tri[1]);
            add_edge(tri[1], tri[2]);
            add_edge(tri[2], tri[0]);
        }
        std::vector<EdgeKey> boundary;
        boundary.reserve(edge_count.size());
        for (auto& kv : edge_count) {
            if (kv.second == 1) boundary.push_back(kv.first);
        }

        // 3) remove bad triangles
        std::vector<std::array<int,3>> tris_keep;
        tris_keep.reserve(tris_ext.size());
        std::vector<char> is_bad(tris_ext.size(), 0);
        for (int bi : bad) is_bad[bi] = 1;
        for (int t = 0; t < (int)tris_ext.size(); ++t) {
            if (!is_bad[t]) tris_keep.push_back(tris_ext[t]);
        }
        tris_ext.swap(tris_keep);

        // 4) retriangulate cavity with new point
        for (const auto& e : boundary) {
            std::array<int,3> tri = { e.a, e.b, pi };
            // enforce CCW
            if (h_orient(P[tri[0]], P[tri[1]], P[tri[2]]) < 0.0) std::swap(tri[0], tri[1]);
            tris_ext.push_back(tri);
        }
    }

    // 5) remove tris referencing super vertices
    for (auto tri : tris_ext) {
        if (tri[0] < n && tri[1] < n && tri[2] < n) {
            // ensure CCW
            if (h_orient(P[tri[0]], P[tri[1]], P[tri[2]]) < 0.0) std::swap(tri[1], tri[2]);
            out_tris.push_back(tri);
        }
    }
}


enum class NHPhase : int { NONE = 0, PRE = 1, POST = 2 };

struct NHState {
    double zeta;                // friction variable
    double Q;                   // thermostat "mass"
    double target_kinetic_E;    // K_target = DOF * 0.5 * kT  (already computed on host)
    double dt;                  // integrator step
};


#define CUDA_CHECK(call)                                                \
do {                                                                    \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

namespace cg = cooperative_groups;

// --- Custom Atomic Function for double ---
__device__ double atomicMax_double(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;

    do {
        assumed = old;
        double cur = __longlong_as_double(assumed);
        double mx  = fmax(val, cur);
        unsigned long long desired = __double_as_longlong(mx);
        old = atomicCAS(address_as_ull, assumed, desired);
    } while (assumed != old);

    return __longlong_as_double(old);
}




// Butcher tableau for Runge-Kutta-Fehlberg 7(8) method
__constant__ double rkf78_c[13] = {
    0.0, 2.0/27.0, 1.0/9.0, 1.0/6.0, 5.0/12.0, 1.0/2.0, 5.0/6.0,
    1.0/6.0, 2.0/3.0, 1.0/3.0, 1.0, 0.0, 1.0
};

__constant__ double rkf78_b8[13] = {
    41.0/840.0, 0.0, 0.0, 0.0, 0.0, 34.0/105.0, 9.0/35.0,
    9.0/35.0, 9.0/280.0, 9.0/280.0, 41.0/840.0, 0.0, 0.0
};

__constant__ double rkf78_b7[13] = {
    0.0, 0.0, 0.0, 0.0, 0.0, 34.0/105.0, 9.0/35.0,
    9.0/35.0, 9.0/280.0, 9.0/280.0, 0.0, 41.0/840.0, 41.0/840.0
};

__constant__ double rkf78_a[12][12] = {
    {2.0/27.0,0,0,0,0,0,0,0,0,0,0,0},
    {1.0/36.0,3.0/36.0,0,0,0,0,0,0,0,0,0,0},
    {1.0/24.0,0,3.0/24.0,0,0,0,0,0,0,0,0,0},
    {5.0/12.0,0,-25.0/16.0,25.0/16.0,0,0,0,0,0,0,0,0},
    {1.0/20.0,0,0,1.0/4.0,1.0/5.0,0,0,0,0,0,0,0},
    {-25.0/108.0,0,0,125.0/108.0,-65.0/27.0,125.0/54.0,0,0,0,0,0,0},
    {31.0/300.0,0,0,0,61.0/225.0,-2.0/9.0,13.0/900.0,0,0,0,0,0},
    {2.0,0,0,-53.0/6.0,704.0/45.0,-107.0/9.0,67.0/90.0,3.0,0,0,0,0},
    {-91.0/108.0,0,0,23.0/108.0,-976.0/135.0,311.0/54.0,-19.0/60.0,17.0/6.0,-1.0/12.0,0,0,0},
    {2383.0/4100.0,0,0,-341.0/164.0,4496.0/1025.0,-301.0/82.0,2133.0/4100.0,45.0/82.0,45.0/164.0,18.0/41.0,0,0},
    {3.0/205.0,0,0,0,0,-6.0/41.0,-3.0/205.0,-3.0/41.0,3.0/41.0,6.0/41.0,0,0},
    {-1777.0/4100.0,0,0,-341.0/164.0,4496.0/1025.0,-289.0/82.0,2193.0/4100.0,51.0/82.0,33.0/164.0,12.0/41.0,0,1.0}
};

inline double copy_scalar_from_device(double* d_ptr) {
    double h = 0.0;
    cudaMemcpy(&h, d_ptr, sizeof(double), cudaMemcpyDeviceToHost);
    return h;
}

__global__ void calculateForces(
    Particle* __restrict__ particles_out,       // local i: writes acc (and reads vel for NH/K)
    const Particle* __restrict__ particles_in,  // full N (global j)
    const Particle* __restrict__ particles_local, // this GPU's local slice (state for i)
    int n_particles_local,
    int n_particles_total,
    int local_start_index,            // global index of local[0]
    double box_width,
    double box_height,
    double mass0,
    double mass1,
    const double SIGMA_AA,
    const double EPSILON_AA,
    const double SIGMA_BB,
    const double EPSILON_BB,
    const double SIGMA_AB,
    const double EPSILON_AB,
    double* __restrict__ d_total_U,   // device scalar: total potential energy (pairwise, counted once)
    double* __restrict__ d_total_K,   // device scalar: total kinetic energy
    int     use_nose_hoover,          // 0/1
    double  zeta_current              // Nose–Hoover friction variable ζ (host-updated)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_particles_local) return;

    // Unchanged: read i from local slice
    const int gi = local_start_index + idx;    // global index of i
    const Particle pi = particles_local[idx];
    const double2 pi_pos = pi.pos;
    const int     pi_type = pi.type;

    // ADD: per-thread local accumulators
    double2 f = make_double2(0.0, 0.0);
    double  Ui = 0.0; // potential contributions counted with gi < j to avoid double count
    // End of Add

    // Loop over all j in global array
    for (int j = 0; j < n_particles_total; ++j) {
        if (j == gi) continue;

        const Particle pj = particles_in[j];
        double2 dr;
        dr.x = pi_pos.x - pj.pos.x;
        dr.y = pi_pos.y - pj.pos.y;

        // minimum image
        if (dr.x >  box_width  * 0.5) dr.x -= box_width;
        if (dr.x < -box_width  * 0.5) dr.x += box_width;
        if (dr.y >  box_height * 0.5) dr.y -= box_height;
        if (dr.y < -box_height * 0.5) dr.y += box_height;

        // Single precision pair evaluation (forces & potential), FP64 accumulation
        float2 drf; drf.x = (float)dr.x; drf.y = (float)dr.y;
        float r2 = drf.x * drf.x + drf.y * drf.y;
        if (r2 <= 1e-12f) continue;

        float sigma, epsilon;
        if (pi_type == 0 && pj.type == 0) { sigma = (float)SIGMA_AA; epsilon = (float)EPSILON_AA; }
        else if (pi_type == 1 && pj.type == 1) { sigma = (float)SIGMA_BB; epsilon = (float)EPSILON_BB; }
        else { sigma = (float)SIGMA_AB; epsilon = (float)EPSILON_AB; }

        const float rc = 2.5f * sigma;
        const float rc2 = rc * rc;

        if (r2 < rc2) {
            const float inv_r2 = 1.0f / r2;
            const float sig2   = sigma * sigma;
            const float rinv2  = sig2 * inv_r2;            // (σ/r)^2
            const float rinv6  = rinv2 * rinv2 * rinv2;    // (σ/r)^6
            const float rinv12 = rinv6 * rinv6;

            // force magnitude (LJ)
            const float coeff  = 24.0f * epsilon * (2.0f * rinv12 - rinv6) * inv_r2;

            // accumulate forces in FP64
            f.x += (double)coeff * (double)drf.x;
            f.y += (double)coeff * (double)drf.y;

            // ADD: accumulate potential once per pair: only when gi < j
            if (gi < j) {
                const float Uij = 4.0f * epsilon * (rinv12 - rinv6); // unshifted LJ
                Ui += (double)Uij;
            }
            // End of Add
        }
    }

    // ADD: Nose–Hoover friction term a_fric = -zeta * v (applied to accelerations)
    if (use_nose_hoover) {
        f.x += -zeta_current * pi.vel.x;
        f.y += -zeta_current * pi.vel.y;
    }
    // End of Add

    // write acceleration = force / m(type)
    const double mi = (pi_type == 0 ? mass0 : mass1);
    particles_out[idx].acc.x = f.x / mi;
    particles_out[idx].acc.y = f.y / mi;
    particles_out[idx].type  = pi_type;   // keep type consistent
    particles_out[idx].pos   = pi.pos;    // passthrough if your out array mirrors in
    particles_out[idx].vel   = pi.vel;

    // ADD: accumulate kinetic energy (per i) and potential energy (per-pair)
    if (d_total_K) {
        const double v2 = pi.vel.x * pi.vel.x + pi.vel.y * pi.vel.y;
        const double Ki = 0.5 * mi * v2;
        atomicAdd(d_total_K, Ki);
    }
    if (d_total_U) {
        atomicAdd(d_total_U, Ui);
    }
    // End of Add
}


__global__ void reduceKineticEnergy(double* block_sums, const Particle* particles, int n,
                                    double mass0, double mass1) {
    extern __shared__ double s_K[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    s_K[tid] = 0.0;
    if (idx < n) {
        double m = (particles[idx].type == 0) ? mass0 : mass1;
        double vx = particles[idx].vel.x, vy = particles[idx].vel.y;
        s_K[tid] = 0.5 * m * (vx*vx + vy*vy);
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_K[tid] += s_K[tid + s];
        __syncthreads();
    }
    if (tid == 0) block_sums[blockIdx.x] = s_K[0];
}

__global__ void applyThermostatScaling(Particle* particles, int n, double scale_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        particles[idx].vel.x *= scale_factor;
        particles[idx].vel.y *= scale_factor;
    }
}

__global__ void vv_first_kick_drift(Particle* mid, const Particle* y, int n,
                                    double h, double box_w, double box_h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double2 v_half;
    v_half.x = y[idx].vel.x + 0.5 * h * y[idx].acc.x;
    v_half.y = y[idx].vel.y + 0.5 * h * y[idx].acc.y;

    double2 r_new;
    r_new.x = y[idx].pos.x + h * v_half.x;
    r_new.y = y[idx].pos.y + h * v_half.y;

    // periodic wrap
    r_new.x = fmod(fmod(r_new.x, box_w) + box_w, box_w);
    r_new.y = fmod(fmod(r_new.y, box_h) + box_h, box_h);

    mid[idx].pos  = r_new;
    mid[idx].vel  = v_half;   // store v_half here; second kick will finish velocity
    mid[idx].type = y[idx].type;
}

// Velocity-Verlet: second kick + finalize (uses accelerations evaluated at mid.pos)
__global__ void vv_second_kick_finalize(Particle* y_out,
                                        const Particle* mid_with_vhalf,
                                        const Particle* acc_src, // acc_src[idx].acc holds a(t+h)
                                        int n, double h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    y_out[idx].pos   = mid_with_vhalf[idx].pos;
    y_out[idx].vel.x = mid_with_vhalf[idx].vel.x + 0.5 * h * acc_src[idx].acc.x;
    y_out[idx].vel.y = mid_with_vhalf[idx].vel.y + 0.5 * h * acc_src[idx].acc.y;
    y_out[idx].acc   = acc_src[idx].acc; // keep a(t+h) handy
    y_out[idx].type  = mid_with_vhalf[idx].type;
}

// Error estimator for adaptive dt: compare 1 full step vs 2 half-steps
__global__ void verlet_error_norm(double* error_sq,
                                  const Particle* y_full,
                                  const Particle* y_half2,
                                  int n,
                                  double rel_tol, double abs_tol) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double dx = y_full[idx].pos.x - y_half2[idx].pos.x;
    double dy = y_full[idx].pos.y - y_half2[idx].pos.y;
    double dvx = y_full[idx].vel.x - y_half2[idx].vel.x;
    double dvy = y_full[idx].vel.y - y_half2[idx].vel.y;

    double sx  = fmax(fabs(y_full[idx].pos.x), fabs(y_half2[idx].pos.x));
    double sy  = fmax(fabs(y_full[idx].pos.y), fabs(y_half2[idx].pos.y));
    double svx = fmax(fabs(y_full[idx].vel.x), fabs(y_half2[idx].vel.x));
    double svy = fmax(fabs(y_full[idx].vel.y), fabs(y_half2[idx].vel.y));

    double tol_x  = fmax(abs_tol, rel_tol * sx);
    double tol_y  = fmax(abs_tol, rel_tol * sy);
    double tol_vx = fmax(abs_tol, rel_tol * svx);
    double tol_vy = fmax(abs_tol, rel_tol * svy);

    double ex = (dx*dx)  / (tol_x*tol_x);
    double ey = (dy*dy)  / (tol_y*tol_y);
    double evx= (dvx*dvx)/(tol_vx*tol_vx);
    double evy= (dvy*dvy)/(tol_vy*tol_vy);

    double e = fmax(fmax(ex, ey), fmax(evx, evy));
    atomicMax_double(error_sq, e);
}

__global__ void rk_prepare_stage_state(Particle* stage_particles, const Particle* initial_particles, 
                                        double2** k_vel, double2** k_acc, int stage_idx, int n, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    double2 pos_sum = make_double2(0.0, 0.0);
    double2 vel_sum = make_double2(0.0, 0.0);
    for (int j = 0; j < stage_idx; ++j) {
        float a = rkf78_a[stage_idx-1][j];
        pos_sum.x += a * k_vel[j][idx].x;
        pos_sum.y += a * k_vel[j][idx].y;
        vel_sum.x += a * k_acc[j][idx].x;
        vel_sum.y += a * k_acc[j][idx].y;
    }
    
    stage_particles[idx].pos.x = initial_particles[idx].pos.x + dt * pos_sum.x;
    stage_particles[idx].pos.y = initial_particles[idx].pos.y + dt * pos_sum.y;
    stage_particles[idx].vel.x = initial_particles[idx].vel.x + dt * vel_sum.x;
    stage_particles[idx].vel.y = initial_particles[idx].vel.y + dt * vel_sum.y;
    stage_particles[idx].type = initial_particles[idx].type;
}

__global__ void rk_final_solution_and_error(
    Particle* y_final, Particle* y_hat_final, double* error_sq,
    const Particle* initial_particles, double2** k_vel, double2** k_acc, int n, double dt,
    double rel_tol, double abs_tol
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    double2 y_pos_sum = make_double2(0.0, 0.0);
    double2 y_vel_sum = make_double2(0.0, 0.0);
    double2 y_hat_pos_sum = make_double2(0.0, 0.0);
    double2 y_hat_vel_sum = make_double2(0.0, 0.0);

    for (int j = 0; j < 13; ++j) {
        y_pos_sum.x += rkf78_b8[j] * k_vel[j][idx].x;
        y_pos_sum.y += rkf78_b8[j] * k_vel[j][idx].y;
        y_vel_sum.x += rkf78_b8[j] * k_acc[j][idx].x;
        y_vel_sum.y += rkf78_b8[j] * k_acc[j][idx].y;
        y_hat_pos_sum.x += rkf78_b7[j] * k_vel[j][idx].x;
        y_hat_pos_sum.y += rkf78_b7[j] * k_vel[j][idx].y;
        y_hat_vel_sum.x += rkf78_b7[j] * k_acc[j][idx].x;
        y_hat_vel_sum.y += rkf78_b7[j] * k_acc[j][idx].y;
    }
    
    y_final[idx].pos.x = initial_particles[idx].pos.x + dt * y_pos_sum.x;
    y_final[idx].pos.y = initial_particles[idx].pos.y + dt * y_pos_sum.y;
    y_final[idx].vel.x = initial_particles[idx].vel.x + dt * y_vel_sum.x;
    y_final[idx].vel.y = initial_particles[idx].vel.y + dt * y_vel_sum.y;
    y_hat_final[idx].pos.x = initial_particles[idx].pos.x + dt * y_hat_pos_sum.x;
    y_hat_final[idx].pos.y = initial_particles[idx].pos.y + dt * y_hat_pos_sum.y;
    y_hat_final[idx].vel.x = initial_particles[idx].vel.x + dt * y_hat_vel_sum.x;
    y_hat_final[idx].vel.y = initial_particles[idx].vel.y + dt * y_hat_vel_sum.y;
    
    double2 error_pos, error_vel;
    error_pos.x = y_final[idx].pos.x - y_hat_final[idx].pos.x;
    error_pos.y = y_final[idx].pos.y - y_hat_final[idx].pos.y;
    error_vel.x = y_final[idx].vel.x - y_hat_final[idx].vel.x;
    error_vel.y = y_final[idx].vel.y - y_hat_final[idx].vel.y;

    double sx = fmax(fabs(y_final[idx].pos.x), fabs(y_hat_final[idx].pos.x));
    double sy = fmax(fabs(y_final[idx].pos.y), fabs(y_hat_final[idx].pos.y));
    double svx = fmax(fabs(y_final[idx].vel.x), fabs(y_hat_final[idx].vel.x));
    double svy = fmax(fabs(y_final[idx].vel.y), fabs(y_hat_final[idx].vel.y));

    double tol_pos_x = fmax(abs_tol, rel_tol * sx);
    double tol_pos_y = fmax(abs_tol, rel_tol * sy);
    double tol_vel_x = fmax(abs_tol, rel_tol * svx);
    double tol_vel_y = fmax(abs_tol, rel_tol * svy);

    double err_val = fmax(
        fmax((error_pos.x * error_pos.x) / (tol_pos_x * tol_pos_x),
              (error_pos.y * error_pos.y) / (tol_pos_y * tol_pos_y)),
        fmax((error_vel.x * error_vel.x) / (tol_vel_x * tol_vel_x),
              (error_vel.y * error_vel.y) / (tol_vel_y * tol_vel_y))
    );

    // [CHANGED] atomic max vs -INF initializer
    atomicMax_double(error_sq, err_val);
}

__global__ void apply_final_state(Particle* particles, const Particle* y_final,
                                  int n, double box_w, double box_h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    particles[idx].pos = y_final[idx].pos;
    particles[idx].vel = y_final[idx].vel;

    particles[idx].pos.x = fmod(fmod(particles[idx].pos.x, box_w) + box_w, box_w);
    particles[idx].pos.y = fmod(fmod(particles[idx].pos.y, box_h) + box_h, box_h);
}

__global__ void extract_accel(double2* dest_acc, const Particle* src_particles, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dest_acc[idx] = src_particles[idx].acc;
}

__global__ void extract_vel(double2* dest_vel, const Particle* src_particles, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dest_vel[idx] = src_particles[idx].vel;
}

__global__ void copyAccelerations(Particle* dest, const Particle* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dest[idx].acc = src[idx].acc;
}

class MDSimulation {
public:
    MDSimulation(int n_particles, int n_particles_type0,
            double box_w, double box_h, 
            double temp_init, double temp_target,
            double SIGMA_AA, double SIGMA_BB, double SIGMA_AB,
            double EPSILON_AA, double EPSILON_BB, double EPSILON_AB,
            double devide_p,
            double dt,
            const std::string& run_name = "")
        : N_PARTICLES_TOTAL(n_particles), N_PARTICLES_TYPE0(n_particles_type0), 
        BOX_WIDTH(box_w), BOX_HEIGHT(box_h),
        TEMP_INIT(temp_init), TARGET_TEMP(temp_target),
        SIGMA_AA(SIGMA_AA), SIGMA_BB(SIGMA_BB), SIGMA_AB(SIGMA_AB),
        EPSILON_AA(EPSILON_AA), EPSILON_BB(EPSILON_BB), EPSILON_AB(EPSILON_AB), dt(dt){
        target_kinetic_energy = (double)DEGREES_OF_FREEDOM * TARGET_TEMP / 2.0;
        ramp_kinetic_energy = (double) DEGREES_OF_FREEDOM * TEMP_INIT / 2.0;
        nh_state = {0.0, 100, ramp_kinetic_energy, dt}; //  {zeta, Q, target_kinetic_E, dt}.
        setupOutputDirectory(run_name);
        initGPUs();
        allocateMemory();
        initParticles(devide_p);
    }

    ~MDSimulation() {
        cleanup();
    }

    void run(const double run_time) {
        std::cout << "Starting simulation..." << std::endl;
        std::cout << "Step | Sim Time     | Timestep (dt) \n";
        std::cout << "--------------------------------------\n";
        
        output_filepath = output_dir_base + "run.bin.gz";
        save(0); 

        for (int step = 1; true; ++step) {
            double t_stop = floor(sim_time / SAVE_DT_INTERVAL) * SAVE_DT_INTERVAL + SAVE_DT_INTERVAL;
            
            bool trigger_stop = false;
            performTimestep("NVT", t_stop, &trigger_stop);

            if (step % OUTPUT_FREQ == 0) {
                std::cout << std::setw(4) << step << " | " 
                          << std::fixed << std::setw(12) << std::setprecision(6) << sim_time << " | " 
                          << std::scientific << std::setw(12) << std::setprecision(6) << dt 
                          << std::defaultfloat << std::endl;
                plot(output_dir_base + "runstep_" + std::to_string(step) + ".png");
            }
            
            if (trigger_stop) {
                save(step);
            }
            
            if (sim_time >= run_time) {
                break;
            }
        }
        
        std::cout << "--------------------------------------\n";
        std::cout << "Simulation finished." << std::endl;
    }

    void equilibrate(const double equil_time) {
        std::cout << "Equilibrating..." << std::endl;
        std::cout << "Step | Sim Time     | Timestep (dt) \n";
        std::cout << "--------------------------------------\n";
        
        output_filepath = output_dir_base + "equilibrate.bin.gz";
        save(0);

        for (int step = 1; true; ++step) {
            double t_stop = floor(sim_time / SAVE_DT_INTERVAL) * SAVE_DT_INTERVAL + SAVE_DT_INTERVAL;
            
            bool trigger_stop = false;
            performTimestep("NVT", t_stop, &trigger_stop);

            if (step % OUTPUT_FREQ == 0) {
                std::cout << std::setw(4) << step << " | " 
                          << std::fixed << std::setw(12) << std::setprecision(6) << sim_time << " | " 
                          << std::scientific << std::setw(12) << std::setprecision(6) << dt 
                          << std::defaultfloat << std::endl;
                plot(output_dir_base + "eq_step_" + std::to_string(step) + ".png");
            }
            
            if (trigger_stop) {
                save(step);
            }
            
            if (sim_time > equil_time) {
                break;
            }
        }
        
        sim_time = 0.0;
        std::cout << "--------------------------------------\n";
        std::cout << "Equilibration finished." << std::endl;
    }

    void equilibrate_T_ramp(const double ramp_time, const double extra_time) {
        std::cout << "Equilibrating with temperature ramping..." << std::endl;
        std::cout << "Step | Sim Time     | Timestep (dt) \n";
        std::cout << "--------------------------------------\n";
        
        output_filepath = output_dir_base + "ramp.bin.gz";
        save(0);
        int total_steps = (int) ramp_time/dt;
        double dT_ramp = (TARGET_TEMP - TEMP_INIT)/total_steps;

        for (int step = 1; step < total_steps; ++step) {
            double t_stop = floor(sim_time / SAVE_DT_INTERVAL) * SAVE_DT_INTERVAL + SAVE_DT_INTERVAL;
            ramp_kinetic_energy = (double) DEGREES_OF_FREEDOM * (dT_ramp * step + TEMP_INIT) / 2.0;
            nh_state.target_kinetic_E = ramp_kinetic_energy;
            bool trigger_stop = false;
            performTimestep("NVT", t_stop, &trigger_stop);

            if (step % OUTPUT_FREQ == 0) {
                std::cout << std::setw(4) << step << " | " 
                          << std::fixed << std::setw(12) << std::setprecision(6) << sim_time << " | " 
                          << std::scientific << std::setw(12) << std::setprecision(6) << dt 
                          << std::defaultfloat << std::endl;
                plot(output_dir_base + "eq_step_" + std::to_string(step) + ".png");
            }
            
            if (trigger_stop) {
                save(step);
            }
        }
        sim_time = 0.0;
        std::cout << "--------------------------------------\n";
        std::cout << "Temperature ramp finished." << std::endl;

        nh_state.target_kinetic_E = target_kinetic_energy;
        equilibrate(extra_time);
    }

    // double capillary_wave(int mode_min, int mode_max, const std::string& plot_path)
    // {
    //     // Unchanged: gather current frame (pos, vel, type) from all GPUs to host
    //     for (int g = 0; g < n_gpus; ++g) { CUDA_CHECK(cudaSetDevice(g)); CUDA_CHECK(cudaDeviceSynchronize()); }
    //     for (int g = 0; g < n_gpus; ++g) {
    //         CUDA_CHECK(cudaSetDevice(g));
    //         CUDA_CHECK(cudaMemcpy(&h_particles[offsets[g]], d_particles[g],
    //                             static_cast<size_t>(counts[g]) * sizeof(Particle),
    //                             cudaMemcpyDeviceToHost));
    //     }

    //     const double Lx = this->BOX_WIDTH;
    //     const double Ly = this->BOX_HEIGHT;

    //     const int N = static_cast<int>(h_particles.size());
    //     if (N <= 3) {
    //         std::cerr << "[capillary_wave] Not enough particles: N=" << N << std::endl;
    //         return 0.0;
    //     }

    //     std::vector<double2> pts(N);
    //     std::vector<int>      types(N);
    //     for (int i = 0; i < N; ++i) {
    //         pts[i]   = h_particles[i].pos;   // assume already mapped into [0,L)
    //         types[i] = h_particles[i].type;  // 0 or 1
    //     }

    //     // ADD: instantaneous temperature (k_B=1); DOF excludes 2 for total momentum in 2D
    //     const int DOF = 2 * N - 2;
    //     double K_now = 0.0;
    //     for (int i = 0; i < N; ++i) {
    //         const double m  = (h_particles[i].type == 0 ? MASS_TYPE0 : MASS_TYPE1);
    //         const double vx = h_particles[i].vel.x;
    //         const double vy = h_particles[i].vel.y;
    //         K_now += 0.5 * m * (vx*vx + vy*vy);
    //     }
    //     const double T_now = (DOF > 0 ? (2.0 * K_now) / static_cast<double>(DOF) : TARGET_TEMP);
    //     // End of Add

    //     // Unchanged: Delaunay triangulation on CPU (Bowyer–Watson)
    //     std::vector<std::array<int,3>> tris;
    //     bowyer_watson_cpu(pts, tris);
    //     if (tris.empty()) {
    //         std::cerr << "[capillary_wave] Delaunay returned 0 triangles.\n";
    //         return 0.0;
    //     }

    //     // Unchanged: build edge -> two adjacent triangles map
    //     struct EdgeU {
    //         int a, b;
    //         EdgeU(int i=0,int j=0){ if(i<j){a=i;b=j;} else {a=j;b=i;} }
    //         bool operator==(const EdgeU& o) const noexcept { return a==o.a && b==o.b; }
    //     };
    //     struct EdgeUHash {
    //         size_t operator()(const EdgeU& e) const noexcept {
    //             return (static_cast<size_t>(e.a)*1315423911u) ^ (static_cast<size_t>(e.b)*2654435761u);
    //         }
    //     };

    //     std::unordered_map<EdgeU, std::array<int,2>, EdgeUHash> edge_tris;
    //     edge_tris.reserve(tris.size()*2 + 16);

    //     auto add_edge_adj = [&](int u, int v, int tri_idx){
    //         EdgeU e(u,v);
    //         auto it = edge_tris.find(e);
    //         if (it == edge_tris.end()) {
    //             edge_tris.emplace(e, std::array<int,2>{tri_idx, -1});
    //         } else {
    //             if (it->second[0] == -1) it->second[0] = tri_idx;
    //             else                     it->second[1] = tri_idx;
    //         }
    //     };

    //     for (int t = 0; t < (int)tris.size(); ++t) {
    //         const auto& tr = tris[t];
    //         add_edge_adj(tr[0], tr[1], t);
    //         add_edge_adj(tr[1], tr[2], t);
    //         add_edge_adj(tr[2], tr[0], t);
    //     }

    //     // Unchanged: triangle circumcenter
    //     auto circumcenter_host = [](const double2& A, const double2& B, const double2& C)->double2 {
    //         const double ax=A.x, ay=A.y, bx=B.x, by=B.y, cx=C.x, cy=C.y;
    //         const double a2 = ax*ax + ay*ay;
    //         const double b2 = bx*bx + by*by;
    //         const double c2 = cx*cx + cy*cy;
    //         const double d  = 2.0 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by));
    //         if (std::fabs(d) < 1e-24)
    //             return make_double2(std::numeric_limits<double>::quiet_NaN(),
    //                                 std::numeric_limits<double>::quiet_NaN());
    //         const double ux = (a2*(by - cy) + b2*(cy - ay) + c2*(ay - by)) / d;
    //         const double uy = (a2*(cx - bx) + b2*(ax - cx) + c2*(bx - ax)) / d;
    //         return make_double2(ux, uy);
    //     };

    //     // Unchanged: AB Voronoi segments (use ALL; no ignoring here)
    //     struct Seg { double x1,y1,x2,y2; double dx_min, dy_min; };
    //     std::vector<Seg> ab_voro;
    //     ab_voro.reserve(edge_tris.size());

    //     auto pbc_delta = [](double d, double L) {
    //         if (L <= 0.0) return d;
    //         d -= std::floor((d / L) + 0.5) * L;  // nearest image
    //         return d;
    //     };
    //     auto wrap01 = [](double x, double L) {
    //         double y = std::fmod(x, L);
    //         if (y < 0.0) y += L;
    //         return y;
    //     };

    //     int ab_edges = 0;
    //     for (const auto& kv : edge_tris) {
    //         const EdgeU& e = kv.first;
    //         const int t1 = kv.second[0];
    //         const int t2 = kv.second[1];
    //         if (t1 < 0 || t2 < 0) continue;
    //         if (types[e.a] == types[e.b]) continue;

    //         const auto& T1 = tris[t1];
    //         const auto& T2 = tris[t2];
    //         const double2 c1 = circumcenter_host(pts[T1[0]], pts[T1[1]], pts[T1[2]]);
    //         const double2 c2 = circumcenter_host(pts[T2[0]], pts[T2[1]], pts[T2[2]]);
    //         if (std::isnan(c1.x) || std::isnan(c2.x)) continue;

    //         const double dx = pbc_delta(c2.x - c1.x, Lx);
    //         const double dy = pbc_delta(c2.y - c1.y, Ly);

    //         // store canonical image: a=(wrapped c1), b=a+(dx_min,dy_min) wrapped back for convenience
    //         const double ax = wrap01(c1.x, Lx), ay = wrap01(c1.y, Ly);
    //         const double bx = wrap01(c1.x + dx, Lx), by = wrap01(c1.y + dy, Ly);

    //         ab_voro.push_back(Seg{ax, ay, bx, by, dx, dy});
    //         ++ab_edges;
    //     }

    //     if (ab_voro.empty()) {
    //         std::cerr << "[capillary_wave] No AB Voronoi segments found.\n";
    //         return 0.0;
    //     }

    //     // ADD: vertical slicing to recover the TWO spanning interfaces along x (ignore internal loops implicitly)
    //     //      We intersect ALL segments with vertical rays x = x_i and then pick the interface nearest to mid-plane
    //     //      from below (ylow) and from above (yupp). This isolates the TWO percolating interfaces.
    //     const int NBINS = 512;
    //     std::vector<double> xs(NBINS);
    //     const double dx_bin = Lx / NBINS;
    //     for (int i = 0; i < NBINS; ++i) xs[i] = (i + 0.5) * dx_bin;

    //     auto intersections_at_x = [&](double xi, std::vector<double>& ys){
    //         ys.clear();
    //         ys.reserve(8);
    //         for (const auto& s : ab_voro) {
    //             // Use the unwrapped direction (dx_min, dy_min) anchored at (x1,y1).
    //             const double x1 = s.x1;
    //             const double x2 = wrap01(s.x1 + s.dx_min, Lx); // displayed endpoint
    //             const double y1 = s.y1;
    //             const double y2 = wrap01(s.y1 + s.dy_min, Ly);

    //             // To decide if xi intersects, we need a consistent segment image spanning [min,max] in x.
    //             // Reconstruct a continuous copy passing near xi by shifting (x2) if needed.
    //             double x2c = x2;
    //             double dxc = x2c - x1;
    //             if (dxc >  Lx*0.5) x2c -= Lx;  // pull back
    //             if (dxc < -Lx*0.5) x2c += Lx;  // push forward

    //             const double xmin = std::min(x1, x2c);
    //             const double xmax = std::max(x1, x2c);
    //             if (xmax == xmin) continue;
    //             // also consider one extra copy shifted by ±Lx so xi near 0/ Lx is not missed
    //             for (int sh = -1; sh <= 1; ++sh) {
    //                 const double off = sh * Lx;
    //                 const double xa = x1  + off;
    //                 const double xb = x2c + off;
    //                 const double xm = std::min(xa, xb);
    //                 const double xM = std::max(xa, xb);
    //                 if (xi < xm || xi > xM) continue;
    //                 const double t = (xi - xa) / (xb - xa);
    //                 const double yb = y1 + t * ( (y2 - y1) );  // interpolate in wrapped display; small error if y wrapped
    //                 double yv = yb;
    //                 // keep y within [0,Ly)
    //                 yv = std::fmod(yv, Ly); if (yv < 0.0) yv += Ly;
    //                 if (yv >= 0.0 && yv <= Ly) ys.push_back(yv);
    //             }
    //         }
    //         // light dedup (segments can produce same hit twice via different shifts)
    //         if (ys.size() > 1) {
    //             std::sort(ys.begin(), ys.end());
    //             ys.erase(std::unique(ys.begin(), ys.end(), [](double a,double b){return std::fabs(a-b) < 1e-8;}), ys.end());
    //         }
    //     };

    //     std::vector<double> ylow(NBINS, std::numeric_limits<double>::quiet_NaN());
    //     std::vector<double> yupp(NBINS, std::numeric_limits<double>::quiet_NaN());

    //     for (int i = 0; i < NBINS; ++i) {
    //         std::vector<double> ys;
    //         intersections_at_x(xs[i], ys);
    //         if (ys.empty()) continue;

    //         // choose the crossing closest to Ly/2 from below and from above
    //         double best_low = -1e300;
    //         double best_upp = +1e300;
    //         for (double y : ys) {
    //             if (y <  Ly*0.5) { if (y > best_low) best_low = y; }
    //             else             { if (y < best_upp) best_upp = y; }
    //         }
    //         if (best_low > -1e200) ylow[i] = best_low;
    //         if (best_upp <  1e200) yupp[i] = best_upp;
    //     }

    //     auto fill_gaps = [&](std::vector<double>& arr){
    //         int last = -1;
    //         for (int i = 0; i < NBINS; ++i) {
    //             if (!std::isnan(arr[i])) last = i;
    //             else if (last >= 0) arr[i] = arr[last];
    //         }
    //         last = -1;
    //         for (int i = NBINS-1; i >= 0; --i) {
    //             if (!std::isnan(arr[i])) last = i;
    //             else if (last >= 0) arr[i] = arr[last];
    //         }
    //         // still NaN → fallback
    //         for (int i = 0; i < NBINS; ++i)
    //             if (std::isnan(arr[i])) arr[i] = (arr.data()==ylow.data()? Ly*0.25 : Ly*0.75);
    //     };
    //     fill_gaps(ylow);
    //     fill_gaps(yupp);

    //     // light smoothing to suppress residual small loops (median of 3)
    //     auto median3 = [](double a,double b,double c){ std::array<double,3> t{a,b,c}; std::sort(t.begin(),t.end()); return t[1]; };
    //     auto smooth1 = [&](std::vector<double>& y){
    //         if (y.size() < 3) return;
    //         std::vector<double> tmp(y.size());
    //         tmp[0] = y[0];
    //         for (size_t i=1;i+1<y.size();++i) tmp[i] = median3(y[i-1], y[i], y[i+1]);
    //         tmp.back() = y.back();
    //         y.swap(tmp);
    //     };
    //     smooth1(ylow);
    //     smooth1(yupp);
    //     // End of Add

    //     // ADD: build height fields h(x) by subtracting means
    //     auto mean_of = [&](const std::vector<double>& v){
    //         double s=0.0; for(double x:v) s+=x; return s/static_cast<double>(v.size());
    //     };
    //     const double y0bar = mean_of(ylow);
    //     const double y1bar = mean_of(yupp);

    //     std::vector<double> h0(NBINS), h1(NBINS);
    //     for (int i = 0; i < NBINS; ++i) {
    //         h0[i] = ylow[i] - y0bar;
    //         h1[i] = yupp[i] - y1bar;
    //     }
    //     // End of Add

    //     // ADD: compute discrete Fourier spectra (simple DFT is fine for NBINS=512)
    //     const int MMAX = NBINS/2;  // Nyquist excluded later by ignoring m=0 and using up to MMAX
    //     std::vector<double> q(MMAX+1, 0.0);
    //     std::vector<double> P0(MMAX+1, 0.0);
    //     std::vector<double> P1(MMAX+1, 0.0);
    //     std::vector<double> Pavg(MMAX+1, 0.0);

    //     auto dft_power = [&](const std::vector<double>& h, std::vector<double>& P){
    //         for (int m = 0; m <= MMAX; ++m) {
    //             if (m == 0) { P[m] = 0.0; continue; }
    //             std::complex<double> H(0.0, 0.0);
    //             for (int n = 0; n < NBINS; ++n) {
    //                 const double theta = -2.0 * M_PI * double(m) * double(n) / double(NBINS);
    //                 const std::complex<double> ph(std::cos(theta), std::sin(theta));
    //                 H += h[n] * ph;
    //             }
    //             H /= double(NBINS);
    //             P[m] = std::norm(H);
    //         }
    //     };

    //     dft_power(h0, P0);
    //     dft_power(h1, P1);
    //     for (int m = 0; m <= MMAX; ++m) {
    //         q[m]    = 2.0 * M_PI * double(m) / Lx;
    //         Pavg[m] = 0.5 * (P0[m] + P1[m]);
    //     }
    //     // End of Add

    //     // ADD: fit 1/P vs q^2 on [mode_min, mode_max] through origin
    //     mode_min = std::max(mode_min, 1);
    //     mode_max = std::min(mode_max, MMAX);
    //     if (mode_min > mode_max) { mode_min = std::min(2, MMAX); mode_max = std::min(8, MMAX); }

    //     double num=0.0, den=0.0;
    //     std::vector<double> q2_fit, invP_fit;
    //     q2_fit.reserve(std::max(0, mode_max - mode_min + 1));
    //     invP_fit.reserve(std::max(0, mode_max - mode_min + 1));

    //     for (int m = mode_min; m <= mode_max; ++m) {
    //         if (Pavg[m] <= 0.0) continue;
    //         const double q2   = q[m] * q[m];
    //         const double invP = 1.0 / Pavg[m];
    //         num += q2 * invP;
    //         den += q2 * q2;
    //         q2_fit.push_back(q2);
    //         invP_fit.push_back(invP);
    //     }

    //     const double slope = (den > 0.0 ? num / den : 0.0);   // slope = (gamma*Lx)/T
    //     const double gamma = slope * (T_now / Lx);            // => gamma
    //     // End of Add

    //     // ADD: optional plotting
    //     if (!plot_path.empty()) {
    //         try {
    //             // Panel 1: log–log PSD and fitted 1/q^2 band over fit window
    //             sciplot::Plot2D p1;
    //             p1.xlabel("q");
    //             p1.ylabel("<|h_q|^2>");
    //             p1.gnuplot("set logscale xy");
    //             p1.gnuplot("set grid back linewidth 0.6 linecolor rgb '#E6E6E6'");
    //             p1.gnuplot("unset key");
    //             p1.gnuplot("set border linewidth 1.0 linecolor rgb '#4D4D4D'");

    //             std::vector<double> qv, Pav, Plo, Pup;
    //             qv.reserve(MMAX-1); Pav.reserve(MMAX-1); Plo.reserve(MMAX-1); Pup.reserve(MMAX-1);
    //             for (int m = 1; m <= MMAX; ++m) {
    //                 if (Pavg[m] <= 0.0) continue;
    //                 qv.push_back(q[m]); Pav.push_back(Pavg[m]); Plo.push_back(P0[m]); Pup.push_back(P1[m]);
    //             }
    //             if (qv.empty()) { qv.push_back(1.0); Pav.push_back(1.0); }

    //             p1.drawPoints(qv, Pav).pointType(7).pointSize(0.8).lineColor("#6E6E6E");
    //             p1.drawPoints(qv, Plo).pointType(7).pointSize(0.6).lineColor("#4C78A8");
    //             p1.drawPoints(qv, Pup).pointType(7).pointSize(0.6).lineColor("#F58518");

    //             // Fitted curve over the fit window: P_fit = T / (gamma*Lx*q^2)
    //             std::vector<double> q_fit, Pfit;
    //             for (int m = mode_min; m <= mode_max; ++m) {
    //                 if (q[m] <= 0) continue;
    //                 q_fit.push_back(q[m]);
    //                 Pfit.push_back(T_now / (gamma * Lx * q[m]*q[m]));
    //             }
    //             if (!q_fit.empty()) p1.drawCurve(q_fit, Pfit).lineWidth(1.6).lineColor("#2F4F4F");

    //             // Panel 2: linear 1/P vs q^2 with line through origin
    //             sciplot::Plot2D p2;
    //             p2.xlabel("q^2");
    //             p2.ylabel("1 / <|h_q|^2>");
    //             p2.gnuplot("set grid back linewidth 0.6 linecolor rgb '#E6E6E6'");
    //             p2.gnuplot("unset key");
    //             p2.gnuplot("set border linewidth 1.0 linecolor rgb '#4D4D4D'");

    //             if (!q2_fit.empty()) {
    //                 p2.drawPoints(q2_fit, invP_fit).pointType(7).pointSize(0.8).lineColor("#6E6E6E");

    //                 const double xmax = *std::max_element(q2_fit.begin(), q2_fit.end());
    //                 std::vector<double> xf{0.0, xmax}, yf{0.0, slope * xmax};
    //                 p2.drawCurve(xf, yf).lineWidth(1.6).lineColor("#2F4F4F");
    //             } else {
    //                 std::vector<double> nx{1.0}, ny{1.0};
    //                 p2.drawCurve(nx, ny);
    //             }

    //             sciplot::Figure fig = {{ p1 }, { p2 }};
    //             std::ostringstream title;
    //             title << "Capillary waves (spanning interfaces): γ ≈ " << gamma
    //                 << "  [T=" << T_now << ", m∈[" << mode_min << "," << mode_max << "]]";
    //             fig.title(title.str());

    //             sciplot::Canvas canvas = {{ fig }};
    //             canvas.size(1100, 900);
    //             canvas.save(plot_path);
    //         } catch (const std::exception& e) {
    //             std::cerr << "[capillary_wave] Plotting failed: " << e.what() << std::endl;
    //         }
    //     }
    //     // End of Add

    //     return gamma;
    // }
    double capillary_wave(int mode_min, int mode_max, const std::string& plot_path)
    {
        // Unchanged: gather current frame (pos, vel, type) from all GPUs to host
        for (int g = 0; g < n_gpus; ++g) { CUDA_CHECK(cudaSetDevice(g)); CUDA_CHECK(cudaDeviceSynchronize()); }
        for (int g = 0; g < n_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMemcpy(&h_particles[offsets[g]], d_particles[g],
                                static_cast<size_t>(counts[g]) * sizeof(Particle),
                                cudaMemcpyDeviceToHost));
        }

        const double Lx = this->BOX_WIDTH;
        const double Ly = this->BOX_HEIGHT;

        const int N = static_cast<int>(h_particles.size());
        if (N <= 3) {
            std::cerr << "[capillary_wave] Not enough particles: N=" << N << std::endl;
            return 0.0;
        }

        std::vector<double2> pts(N);
        std::vector<int>      types(N);
        for (int i = 0; i < N; ++i) {
            pts[i]   = h_particles[i].pos;   // assume already mapped into [0,L)
            types[i] = h_particles[i].type;  // 0 or 1
        }

        // ADD: instantaneous temperature with COM drift removed (k_B=1); DOF excludes 2 total-momentum components in 2D
        double msum = 0.0, px = 0.0, py = 0.0;
        for (int i = 0; i < N; ++i) {
            const double m = (h_particles[i].type == 0 ? MASS_TYPE0 : MASS_TYPE1);
            msum += m;
            px   += m * h_particles[i].vel.x;
            py   += m * h_particles[i].vel.y;
        }
        const double vx_cm = (msum > 0.0 ? px / msum : 0.0);
        const double vy_cm = (msum > 0.0 ? py / msum : 0.0);

        const int DOF = 2 * N - 2;
        double K_now = 0.0;
        for (int i = 0; i < N; ++i) {
            const double m  = (h_particles[i].type == 0 ? MASS_TYPE0 : MASS_TYPE1);
            const double vx = h_particles[i].vel.x - vx_cm;
            const double vy = h_particles[i].vel.y - vy_cm;
            K_now += 0.5 * m * (vx*vx + vy*vy);
        }
        const double T_now = (DOF > 0 ? (2.0 * K_now) / static_cast<double>(DOF) : TARGET_TEMP);
        // End of Add

        // Unchanged: Delaunay triangulation on CPU (Bowyer–Watson)
        std::vector<std::array<int,3>> tris;
        bowyer_watson_cpu(pts, tris);
        if (tris.empty()) {
            std::cerr << "[capillary_wave] Delaunay returned 0 triangles.\n";
            return 0.0;
        }

        // Unchanged: build edge -> two adjacent triangles map
        struct EdgeU {
            int a, b;
            EdgeU(int i=0,int j=0){ if(i<j){a=i;b=j;} else {a=j;b=i;} }
            bool operator==(const EdgeU& o) const noexcept { return a==o.a && b==o.b; }
        };
        struct EdgeUHash {
            size_t operator()(const EdgeU& e) const noexcept {
                return (static_cast<size_t>(e.a)*1315423911u) ^ (static_cast<size_t>(e.b)*2654435761u);
            }
        };

        std::unordered_map<EdgeU, std::array<int,2>, EdgeUHash> edge_tris;
        edge_tris.reserve(tris.size()*2 + 16);

        auto add_edge_adj = [&](int u, int v, int tri_idx){
            EdgeU e(u,v);
            auto it = edge_tris.find(e);
            if (it == edge_tris.end()) {
                edge_tris.emplace(e, std::array<int,2>{tri_idx, -1});
            } else {
                if (it->second[0] == -1) it->second[0] = tri_idx;
                else                     it->second[1] = tri_idx;
            }
        };

        for (int t = 0; t < (int)tris.size(); ++t) {
            const auto& tr = tris[t];
            add_edge_adj(tr[0], tr[1], t);
            add_edge_adj(tr[1], tr[2], t);
            add_edge_adj(tr[2], tr[0], t);
        }

        // Unchanged: triangle circumcenter
        auto circumcenter_host = [](const double2& A, const double2& B, const double2& C)->double2 {
            const double ax=A.x, ay=A.y, bx=B.x, by=B.y, cx=C.x, cy=C.y;
            const double a2 = ax*ax + ay*ay;
            const double b2 = bx*bx + by*by;
            const double c2 = cx*cx + cy*cy;
            const double d  = 2.0 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by));
            if (std::fabs(d) < 1e-24)
                return make_double2(std::numeric_limits<double>::quiet_NaN(),
                                    std::numeric_limits<double>::quiet_NaN());
            const double ux = (a2*(by - cy) + b2*(cy - ay) + c2*(ay - by)) / d;
            const double uy = (a2*(cx - bx) + b2*(ax - cx) + c2*(bx - ax)) / d;
            return make_double2(ux, uy);
        };

        // Unchanged: AB Voronoi segments (use ALL; no ignoring here)
        struct Seg { double x1,y1,x2,y2; double dx_min, dy_min; };
        std::vector<Seg> ab_voro;
        ab_voro.reserve(edge_tris.size());

        auto pbc_delta = [](double d, double L) {
            if (L <= 0.0) return d;
            d -= std::floor((d / L) + 0.5) * L;  // nearest image
            return d;
        };
        auto wrap01 = [](double x, double L) {
            double y = std::fmod(x, L);
            if (y < 0.0) y += L;
            return y;
        };

        int ab_edges = 0;
        for (const auto& kv : edge_tris) {
            const EdgeU& e = kv.first;
            const int t1 = kv.second[0];
            const int t2 = kv.second[1];
            if (t1 < 0 || t2 < 0) continue;
            if (types[e.a] == types[e.b]) continue;

            const auto& T1 = tris[t1];
            const auto& T2 = tris[t2];
            const double2 c1 = circumcenter_host(pts[T1[0]], pts[T1[1]], pts[T1[2]]);
            const double2 c2 = circumcenter_host(pts[T2[0]], pts[T2[1]], pts[T2[2]]);
            if (std::isnan(c1.x) || std::isnan(c2.x)) continue;

            const double dx = pbc_delta(c2.x - c1.x, Lx);
            const double dy = pbc_delta(c2.y - c1.y, Ly);

            // store canonical image: a=(wrapped c1), b=a+(dx_min,dy_min) wrapped back for convenience
            const double ax = wrap01(c1.x, Lx), ay = wrap01(c1.y, Ly);
            const double bx = wrap01(c1.x + dx, Lx), by = wrap01(c1.y + dy, Ly);

            ab_voro.push_back(Seg{ax, ay, bx, by, dx, dy});
            ++ab_edges;
        }

        if (ab_voro.empty()) {
            std::cerr << "[capillary_wave] No AB Voronoi segments found.\n";
            return 0.0;
        }

        // ADD: vertical slicing to recover the TWO spanning interfaces along x (parallel to x-axis)
        const int NBINS = 512;
        std::vector<double> xs(NBINS);
        const double dx_bin = Lx / NBINS;
        for (int i = 0; i < NBINS; ++i) xs[i] = (i + 0.5) * dx_bin;

        auto intersections_at_x = [&](double xi, std::vector<double>& ys){
            ys.clear();
            ys.reserve(8);
            for (const auto& s : ab_voro) {
                // Use the unwrapped direction (dx_min, dy_min) anchored at (x1,y1).
                const double x1 = s.x1;
                const double y1 = s.y1;
                // rebuild a continuous copy of the segment end near x1 using minimal images
                double x2c = x1 + s.dx_min;  // unwrapped endpoint in x
                double y2c = y1 + s.dy_min;  // unwrapped endpoint in y

                // pull back/push forward x2c to be the nearest image to x1 (robust near boundaries)
                double dxc = x2c - x1;
                if (dxc >  Lx*0.5) x2c -= Lx;
                if (dxc < -Lx*0.5) x2c += Lx;

                // early skip if degenerate in x
                if (x1 == x2c) continue;

                // also consider copies shifted by ±Lx so xi near edges is caught
                for (int sh = -1; sh <= 1; ++sh) {
                    const double off = sh * Lx;
                    const double xa = x1  + off;
                    const double xb = x2c + off;
                    const double xm = std::min(xa, xb);
                    const double xM = std::max(xa, xb);
                    if (xi < xm || xi > xM) continue;

                    const double t = (xi - xa) / (xb - xa);  // 0..1 along the segment
                    // interpolate in the UNWRAPPED y using dy_min (so vertical wraps don't corrupt y)
                    const double yb_unwrapped = y1 + t * (y2c - y1);
                    double yv = std::fmod(yb_unwrapped, Ly);
                    if (yv < 0.0) yv += Ly;
                    if (yv >= 0.0 && yv <= Ly) ys.push_back(yv);
                }
            }
            // dedup close duplicates
            if (ys.size() > 1) {
                std::sort(ys.begin(), ys.end());
                ys.erase(std::unique(ys.begin(), ys.end(),
                                    [](double a,double b){return std::fabs(a-b) < 1e-8;}), ys.end());
            }
        };

        std::vector<double> ylow(NBINS, std::numeric_limits<double>::quiet_NaN());
        std::vector<double> yupp(NBINS, std::numeric_limits<double>::quiet_NaN());

        for (int i = 0; i < NBINS; ++i) {
            std::vector<double> ys;
            intersections_at_x(xs[i], ys);
            if (ys.empty()) continue;

            // choose the crossing closest to Ly/2 from below and from above to isolate the two spanning interfaces
            double best_low = -1e300;
            double best_upp = +1e300;
            for (double y : ys) {
                if (y <  Ly*0.5) { if (y > best_low) best_low = y; }
                else             { if (y < best_upp) best_upp = y; }
            }
            if (best_low > -1e200) ylow[i] = best_low;
            if (best_upp <  1e200) yupp[i] = best_upp;
        }

        auto fill_gaps = [&](std::vector<double>& arr){
            int last = -1;
            for (int i = 0; i < NBINS; ++i) {
                if (!std::isnan(arr[i])) last = i;
                else if (last >= 0) arr[i] = arr[last];
            }
            last = -1;
            for (int i = NBINS-1; i >= 0; --i) {
                if (!std::isnan(arr[i])) last = i;
                else if (last >= 0) arr[i] = arr[last];
            }
            for (int i = 0; i < NBINS; ++i)
                if (std::isnan(arr[i])) arr[i] = (arr.data()==ylow.data()? Ly*0.25 : Ly*0.75);
        };
        fill_gaps(ylow);
        fill_gaps(yupp);

        // light smoothing to suppress residual small loops (median of 3)
        auto median3 = [](double a,double b,double c){ std::array<double,3> t{a,b,c}; std::sort(t.begin(),t.end()); return t[1]; };
        auto smooth1 = [&](std::vector<double>& y){
            if (y.size() < 3) return;
            std::vector<double> tmp(y.size());
            tmp[0] = y[0];
            for (size_t i=1;i+1<y.size();++i) tmp[i] = median3(y[i-1], y[i], y[i+1]);
            tmp.back() = y.back();
            y.swap(tmp);
        };
        smooth1(ylow);
        smooth1(yupp);
        // End of Add

        // ADD: build height fields h(x) by subtracting means (interfaces expected parallel to x-axis)
        auto mean_of = [&](const std::vector<double>& v){
            double s=0.0; for(double x:v) s+=x; return s/static_cast<double>(v.size());
        };
        const double y0bar = mean_of(ylow);
        const double y1bar = mean_of(yupp);

        std::vector<double> h0(NBINS), h1(NBINS);
        for (int i = 0; i < NBINS; ++i) {
            h0[i] = ylow[i] - y0bar;
            h1[i] = yupp[i] - y1bar;
        }
        // End of Add

        // ADD: compute discrete Fourier spectra
        const int MMAX = NBINS/2;
        std::vector<double> q(MMAX+1, 0.0);
        std::vector<double> P0(MMAX+1, 0.0);
        std::vector<double> P1(MMAX+1, 0.0);
        std::vector<double> Pavg(MMAX+1, 0.0);

        auto dft_power = [&](const std::vector<double>& h, std::vector<double>& P){
            for (int m = 0; m <= MMAX; ++m) {
                if (m == 0) { P[m] = 0.0; continue; }
                std::complex<double> H(0.0, 0.0);
                for (int n = 0; n < NBINS; ++n) {
                    const double theta = -2.0 * M_PI * double(m) * double(n) / double(NBINS);
                    const std::complex<double> ph(std::cos(theta), std::sin(theta));
                    H += h[n] * ph;
                }
            #if 1
                H /= double(NBINS);
            #endif
                P[m] = std::norm(H);
            }
        };

        dft_power(h0, P0);
        dft_power(h1, P1);
        for (int m = 0; m <= MMAX; ++m) {
            q[m]    = 2.0 * M_PI * double(m) / Lx;
            Pavg[m] = 0.5 * (P0[m] + P1[m]);
        }
        // End of Add

        // ADD: fit 1/P vs q^2 on [mode_min, mode_max] through origin
        mode_min = std::max(mode_min, 1);
        mode_max = std::min(mode_max, MMAX);
        if (mode_min > mode_max) { mode_min = std::min(2, MMAX); mode_max = std::min(8, MMAX); }

        double num=0.0, den=0.0;
        std::vector<double> q2_fit, invP_fit;
        q2_fit.reserve(std::max(0, mode_max - mode_min + 1));
        invP_fit.reserve(std::max(0, mode_max - mode_min + 1));

        for (int m = mode_min; m <= mode_max; ++m) {
            if (Pavg[m] <= 0.0) continue;
            const double q2   = q[m] * q[m];
            const double invP = 1.0 / Pavg[m];
            num += q2 * invP;
            den += q2 * q2;
            q2_fit.push_back(q2);
            invP_fit.push_back(invP);
        }

        const double slope = (den > 0.0 ? num / den : 0.0);   // slope = (gamma*Lx)/T
        const double gamma = (Lx > 0.0 ? slope * (T_now / Lx) : 0.0); // => gamma
        // End of Add

        // ADD: optional plotting
        if (!plot_path.empty()) {
            try {
                sciplot::Plot2D p1;
                p1.xlabel("q");
                p1.ylabel("<|h_q|^2>");
                p1.gnuplot("set logscale xy");
                p1.gnuplot("set grid back linewidth 0.6 linecolor rgb '#E6E6E6'");
                p1.gnuplot("unset key");
                p1.gnuplot("set border linewidth 1.0 linecolor rgb '#4D4D4D'");

                std::vector<double> qv, Pav, Plo, Pup;
                qv.reserve(MMAX-1); Pav.reserve(MMAX-1); Plo.reserve(MMAX-1); Pup.reserve(MMAX-1);
                for (int m = 1; m <= MMAX; ++m) {
                    if (Pavg[m] <= 0.0) continue;
                    qv.push_back(q[m]); Pav.push_back(Pavg[m]); Plo.push_back(P0[m]); Pup.push_back(P1[m]);
                }
                if (qv.empty()) { qv.push_back(1.0); Pav.push_back(1.0); }

                p1.drawPoints(qv, Pav).pointType(7).pointSize(0.8).lineColor("#6E6E6E");
                p1.drawPoints(qv, Plo).pointType(7).pointSize(0.6).lineColor("#4C78A8");
                p1.drawPoints(qv, Pup).pointType(7).pointSize(0.6).lineColor("#F58518");

                // Fitted curve over the fit window: P_fit = T / (gamma*Lx*q^2)
                std::vector<double> q_fit, Pfit;
                for (int m = mode_min; m <= mode_max; ++m) {
                    if (q[m] <= 0) continue;
                    q_fit.push_back(q[m]);
                    const double denom = gamma * Lx * q[m]*q[m];
                    Pfit.push_back(denom > 0.0 ? T_now / denom : 0.0);
                }
                if (!q_fit.empty()) p1.drawCurve(q_fit, Pfit).lineWidth(1.6).lineColor("#2F4F4F");

                sciplot::Plot2D p2;
                p2.xlabel("q^2");
                p2.ylabel("1 / <|h_q|^2>");
                p2.gnuplot("set grid back linewidth 0.6 linecolor rgb '#E6E6E6'");
                p2.gnuplot("unset key");
                p2.gnuplot("set border linewidth 1.0 linecolor rgb '#4D4D4D'");

                if (!q2_fit.empty()) {
                    p2.drawPoints(q2_fit, invP_fit).pointType(7).pointSize(0.8).lineColor("#6E6E6E");
                    const double xmax = *std::max_element(q2_fit.begin(), q2_fit.end());
                    std::vector<double> xf{0.0, xmax}, yf{0.0, slope * xmax};
                    p2.drawCurve(xf, yf).lineWidth(1.6).lineColor("#2F4F4F");
                } else {
                    std::vector<double> nx{1.0}, ny{1.0};
                    p2.drawCurve(nx, ny);
                }

                sciplot::Figure fig = {{ p1 }, { p2 }};
                std::ostringstream title;
                title << "Capillary waves (spanning interfaces): γ ≈ " << gamma
                    << "  [T=" << T_now << ", m∈[" << mode_min << "," << mode_max << "]]";
                fig.title(title.str());

                sciplot::Canvas canvas = {{ fig }};
                canvas.size(1100, 900);
                canvas.save(plot_path);
            } catch (const std::exception& e) {
                std::cerr << "[capillary_wave] Plotting failed: " << e.what() << std::endl;
            }
        }
        // End of Add

        return gamma;
    }

        // ADD: full, robust statistical_capillary_wave() — centered field + zero-cross fallback
        // ADD: statistical_capillary_wave with gradient-peak boundary detection (robust, no zero-cross dependency)
    // ADD: statistical_capillary_wave that reuses the *per-frame Voronoi slicing* method from capillary_wave,
    //      but pools modes over many frames read via MDGZFrameIterator.
    double statistical_capillary_wave(int mode_min,
                                    int mode_max,
                                    const std::string& plot_path,
                                    int frame_min,
                                    int frame_max,
                                    const std::string& csv_path,
                                    const std::string& data_save_path)
    {
        // Unchanged: box
        const double Lx = this->BOX_WIDTH;
        const double Ly = this->BOX_HEIGHT;

        // Unchanged: CSV header (overwrite)
        std::ofstream csv(csv_path);
        if (csv) csv << "frame,m,q,P0,P1,Pavg,T,Lx\n";

        // Unchanged: pooled regression buffers (we will fit T/P = (gamma*Lx) * q^2)
        std::vector<double> pooled_x;    // q^2
        std::vector<double> pooled_y;    // T / Pavg
        std::vector<double> pooled_q;    // for plotting panel 1
        std::vector<double> pooled_Pavg; // for plotting panel 1
        std::vector<double> pooled_T;    // T of the corresponding samples

        // Unchanged: iterator open and fast-forward
        MDGZFrameIterator it;
        if (!it.open(data_save_path, h_particles)) {
            std::cerr << "[statistical_capillary_wave] iterator open() failed for " << data_save_path << std::endl;
            if (csv) csv << "summary,,gamma=0,reason=iterator_open_failed\n";
            return 0.0;
        }
        if (frame_min > 1) {
            for (int i = 1; i < frame_min; ++i) {
                if (!it.next()) {
                    std::cerr << "[statistical_capillary_wave] EOF before frame_min=" << frame_min << std::endl;
                    it.close();
                    if (csv) csv << "summary,,gamma=0,reason=eof_before_frame_min\n";
                    return 0.0;
                }
            }
        }

        // Unchanged: constants used by the per-frame method
        const int NBINS = 512;  // same as capillary_wave
        std::vector<double> xs(NBINS);
        const double dx_bin = Lx / NBINS;
        for (int i = 0; i < NBINS; ++i) xs[i] = (i + 0.5) * dx_bin;

        // Unchanged: helpers copied from capillary_wave (scoped locally)
        auto pbc_wrap = [](double x, double L) {
            double y = std::fmod(x, L);
            if (y < 0.0) y += L;
            return y;
        };
        auto pbc_delta = [](double d, double L) {
            if (L <= 0.0) return d;
            d -= std::floor((d / L) + 0.5) * L;  // nearest image
            return d;
        };
        auto wrap01 = [](double x, double L) {
            double y = std::fmod(x, L);
            if (y < 0.0) y += L;
            return y;
        };
        auto circumcenter_host = [](const double2& A, const double2& B, const double2& C)->double2 {
            const double ax=A.x, ay=A.y, bx=B.x, by=B.y, cx=C.x, cy=C.y;
            const double a2 = ax*ax + ay*ay;
            const double b2 = bx*bx + by*by;
            const double c2 = cx*cx + cy*cy;
            const double d  = 2.0 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by));
            if (std::fabs(d) < 1e-24)
                return make_double2(std::numeric_limits<double>::quiet_NaN(),
                                    std::numeric_limits<double>::quiet_NaN());
            const double ux = (a2*(by - cy) + b2*(cy - ay) + c2*(ay - by)) / d;
            const double uy = (a2*(cx - bx) + b2*(ax - cx) + c2*(bx - ax)) / d;
            return make_double2(ux, uy);
        };
        struct EdgeU {
            int a, b;
            EdgeU(int i=0,int j=0){ if(i<j){a=i;b=j;} else {a=j;b=i;} }
            bool operator==(const EdgeU& o) const noexcept { return a==o.a && b==o.b; }
        };
        struct EdgeUHash {
            size_t operator()(const EdgeU& e) const noexcept {
                return (static_cast<size_t>(e.a)*1315423911u) ^ (static_cast<size_t>(e.b)*2654435761u);
            }
        };

        // Unchanged: frame loop – for each frame, reuse *exactly* the capillary_wave pipeline
        for (int frame_idx = frame_min; frame_idx <= frame_max; ++frame_idx) {
            if (!it.next()) {
                std::cerr << "[statistical_capillary_wave] EOF/truncated before frame " << frame_idx << std::endl;
                break;
            }
            std::cout << "[Frame] " << frame_idx << std::endl;

            const int N = static_cast<int>(h_particles.size());
            if (N <= 3) {
                std::cerr << "[statistical_capillary_wave] Not enough particles: N=" << N << std::endl;
                continue;
            }

            // Unchanged: positions/types (ensure wrapped into [0,L))
            std::vector<double2> pts(N);
            std::vector<int>      types(N);
            for (int i = 0; i < N; ++i) {
                pts[i].x = pbc_wrap(h_particles[i].pos.x, Lx);
                pts[i].y = pbc_wrap(h_particles[i].pos.y, Ly);
                types[i] = h_particles[i].type;
            }

            // Unchanged: instantaneous temperature (finite-safe)
            int n_vel_ok = 0;
            double K_now = 0.0;
            for (int i = 0; i < N; ++i) {
                const double vx = h_particles[i].vel.x;
                const double vy = h_particles[i].vel.y;
                if (std::isfinite(vx) && std::isfinite(vy)) {
                    const double m  = (h_particles[i].type == 0 ? MASS_TYPE0 : MASS_TYPE1);
                    K_now += 0.5 * m * (vx*vx + vy*vy);
                    n_vel_ok += 1;
                }
            }
            const int DOF_eff = std::max(0, 2 * n_vel_ok - 2);
            const double T_now = (DOF_eff > 0 ? (2.0 * K_now) / static_cast<double>(DOF_eff) : this->TARGET_TEMP);

            // Unchanged: Delaunay triangulation on CPU (Bowyer–Watson)
            std::vector<std::array<int,3>> tris;
            bowyer_watson_cpu(pts, tris);
            if (tris.empty()) {
                std::cerr << "[statistical_capillary_wave] Delaunay returned 0 triangles.\n";
                continue;
            }

            // Unchanged: build edge -> two adjacent triangles map
            std::unordered_map<EdgeU, std::array<int,2>, EdgeUHash> edge_tris;
            edge_tris.reserve(tris.size()*2 + 16);
            auto add_edge_adj = [&](int u, int v, int tri_idx){
                EdgeU e(u,v);
                auto it2 = edge_tris.find(e);
                if (it2 == edge_tris.end()) {
                    edge_tris.emplace(e, std::array<int,2>{tri_idx, -1});
                } else {
                    if (it2->second[0] == -1) it2->second[0] = tri_idx;
                    else                      it2->second[1] = tri_idx;
                }
            };
            for (int t = 0; t < (int)tris.size(); ++t) {
                const auto& tr = tris[t];
                add_edge_adj(tr[0], tr[1], t);
                add_edge_adj(tr[1], tr[2], t);
                add_edge_adj(tr[2], tr[0], t);
            }

            // Unchanged: AB Voronoi segments
            struct Seg { double x1,y1,x2,y2; double dx_min, dy_min; };
            std::vector<Seg> ab_voro;
            ab_voro.reserve(edge_tris.size());

            int ab_edges = 0;
            for (const auto& kv : edge_tris) {
                const EdgeU& e = kv.first;
                const int t1 = kv.second[0];
                const int t2 = kv.second[1];
                if (t1 < 0 || t2 < 0) continue;
                if (types[e.a] == types[e.b]) continue;

                const auto& T1 = tris[t1];
                const auto& T2 = tris[t2];
                const double2 c1 = circumcenter_host(pts[T1[0]], pts[T1[1]], pts[T1[2]]);
                const double2 c2 = circumcenter_host(pts[T2[0]], pts[T2[1]], pts[T2[2]]);
                if (std::isnan(c1.x) || std::isnan(c2.x)) continue;

                const double dx = pbc_delta(c2.x - c1.x, Lx);
                const double dy = pbc_delta(c2.y - c1.y, Ly);

                const double ax = wrap01(c1.x, Lx), ay = wrap01(c1.y, Ly);
                const double bx = wrap01(c1.x + dx, Lx), by = wrap01(c1.y + dy, Ly);

                ab_voro.push_back(Seg{ax, ay, bx, by, dx, dy});
                ++ab_edges;
            }

            if (ab_voro.empty()) {
                std::cerr << "[statistical_capillary_wave] No AB Voronoi segments in frame " << frame_idx << ".\n";
                continue;
            }

            // Unchanged: vertical slicing using the SAME routine as capillary_wave
            auto intersections_at_x = [&](double xi, std::vector<double>& ys){
                ys.clear();
                ys.reserve(8);
                for (const auto& s : ab_voro) {
                    const double x1 = s.x1;
                    const double x2 = wrap01(s.x1 + s.dx_min, Lx);
                    const double y1 = s.y1;
                    const double y2 = wrap01(s.y1 + s.dy_min, Ly);

                    double x2c = x2;
                    double dxc = x2c - x1;
                    if (dxc >  Lx*0.5) x2c -= Lx;
                    if (dxc < -Lx*0.5) x2c += Lx;

                    if (x2c == x1) continue;
                    for (int sh = -1; sh <= 1; ++sh) {
                        const double off = sh * Lx;
                        const double xa = x1  + off;
                        const double xb = x2c + off;
                        const double xm = std::min(xa, xb);
                        const double xM = std::max(xa, xb);
                        if (xi < xm || xi > xM) continue;
                        const double t = (xi - xa) / (xb - xa);
                        const double yb = y1 + t * ( (y2 - y1) );
                        double yv = yb;
                        yv = std::fmod(yv, Ly); if (yv < 0.0) yv += Ly;
                        if (yv >= 0.0 && yv <= Ly) ys.push_back(yv);
                    }
                }
                if (ys.size() > 1) {
                    std::sort(ys.begin(), ys.end());
                    ys.erase(std::unique(ys.begin(), ys.end(), [](double a,double b){return std::fabs(a-b) < 1e-8;}), ys.end());
                }
            };

            std::vector<double> ylow(NBINS, std::numeric_limits<double>::quiet_NaN());
            std::vector<double> yupp(NBINS, std::numeric_limits<double>::quiet_NaN());
            for (int i = 0; i < NBINS; ++i) {
                std::vector<double> ys;
                intersections_at_x(xs[i], ys);
                if (ys.empty()) continue;
                double best_low = -1e300;
                double best_upp = +1e300;
                for (double y : ys) {
                    if (y <  Ly*0.5) { if (y > best_low) best_low = y; }
                    else             { if (y < best_upp) best_upp = y; }
                }
                if (best_low > -1e200) ylow[i] = best_low;
                if (best_upp <  1e200) yupp[i] = best_upp;
            }

            auto fill_gaps = [&](std::vector<double>& arr){
                int last = -1;
                for (int i = 0; i < NBINS; ++i) {
                    if (!std::isnan(arr[i])) last = i;
                    else if (last >= 0) arr[i] = arr[last];
                }
                last = -1;
                for (int i = NBINS-1; i >= 0; --i) {
                    if (!std::isnan(arr[i])) last = i;
                    else if (last >= 0) arr[i] = arr[last];
                }
                for (int i = 0; i < NBINS; ++i)
                    if (std::isnan(arr[i])) arr[i] = (arr.data()==ylow.data()? Ly*0.25 : Ly*0.75);
            };
            fill_gaps(ylow);
            fill_gaps(yupp);

            auto median3 = [](double a,double b,double c){ std::array<double,3> t{a,b,c}; std::sort(t.begin(),t.end()); return t[1]; };
            auto smooth1 = [&](std::vector<double>& y){
                if (y.size() < 3) return;
                std::vector<double> tmp(y.size());
                tmp[0] = y[0];
                for (size_t i=1;i+1<y.size();++i) tmp[i] = median3(y[i-1], y[i], y[i+1]);
                tmp.back() = y.back();
                y.swap(tmp);
            };
            smooth1(ylow);
            smooth1(yupp);

            auto mean_of = [&](const std::vector<double>& v){
                double s=0.0; for(double x:v) s+=x; return s/static_cast<double>(v.size());
            };
            const double y0bar = mean_of(ylow);
            const double y1bar = mean_of(yupp);

            std::vector<double> h0(NBINS), h1(NBINS);
            for (int i = 0; i < NBINS; ++i) {
                h0[i] = ylow[i] - y0bar;
                h1[i] = yupp[i] - y1bar;
            }

            const int MMAX = NBINS/2;
            std::vector<double> q(MMAX+1, 0.0);
            std::vector<double> P0(MMAX+1, 0.0);
            std::vector<double> P1(MMAX+1, 0.0);
            std::vector<double> Pavg(MMAX+1, 0.0);

            auto dft_power = [&](const std::vector<double>& h, std::vector<double>& P){
                for (int m = 0; m <= MMAX; ++m) {
                    if (m == 0) { P[m] = 0.0; continue; }
                    std::complex<double> H(0.0, 0.0);
                    for (int n = 0; n < NBINS; ++n) {
                        const double theta = -2.0 * M_PI * double(m) * double(n) / double(NBINS);
                        H += h[n] * std::complex<double>(std::cos(theta), std::sin(theta));
                    }
                    H /= double(NBINS);
                    P[m] = std::norm(H);
                }
            };
            dft_power(h0, P0);
            dft_power(h1, P1);
            for (int m = 0; m <= MMAX; ++m) {
                q[m]    = 2.0 * M_PI * double(m) / Lx;
                Pavg[m] = 0.5 * (P0[m] + P1[m]);
            }

            // Unchanged: write per-frame rows for inspection
            if (csv) {
                for (int m = 1; m <= MMAX; ++m) {
                    csv << frame_idx << "," << m << "," << q[m] << ","
                        << P0[m] << "," << P1[m] << "," << Pavg[m] << ","
                        << T_now << "," << Lx << "\n";
                }
            }

            // Unchanged: pool samples on requested mode window (skip non-positive P)
            const int mmin = std::max(1, std::min(mode_min, MMAX));
            const int mmax = std::max(mmin, std::min(mode_max, MMAX));
            for (int m = mmin; m <= mmax; ++m) {
                if (Pavg[m] > 0.0 && std::isfinite(Pavg[m]) && std::isfinite(T_now)) {
                    pooled_x.push_back(q[m] * q[m]);      // q^2
                    pooled_y.push_back(T_now / Pavg[m]);  // T / P
                    pooled_q.push_back(q[m]);
                    pooled_Pavg.push_back(Pavg[m]);
                    pooled_T.push_back(T_now);
                }
            }
        } // frames

        it.close();

        // Unchanged: guard
        if (pooled_x.empty()) {
            std::cerr << "[statistical_capillary_wave] No pooled samples.\n";
            if (csv) csv << "summary,,gamma=0,reason=no_samples\n";
            return 0.0;
        }

        // Unchanged: fit y = a + b x (robust to tiny offsets); gamma = b / Lx
        auto linreg = [](const std::vector<double>& x, const std::vector<double>& y) {
            long double Sx=0, Sy=0, Sxx=0, Sxy=0;
            const size_t n = x.size();
            for (size_t i=0;i<n;++i){ Sx+=x[i]; Sy+=y[i]; Sxx+=x[i]*x[i]; Sxy+=x[i]*y[i]; }
            const long double D = n*Sxx - Sx*Sx;
            double a=0.0, b=0.0;
            if (fabsl(D) > 1e-18L) {
                a = (double)((Sxx*Sy - Sx*Sxy)/D);
                b = (double)((n*Sxy - Sx*Sy)/D);
            } else {
                a = (double)(Sy/(n>0?n:1));
                b = 0.0;
            }
            return std::pair<double,double>(a,b);
        };
        auto [intercept_prime, slope_prime] = linreg(pooled_x, pooled_y);
        double gamma = slope_prime / Lx;
        if (!std::isfinite(gamma)) gamma = 0.0;

        // Unchanged: CSV summary
        if (csv) {
            csv << "summary," << "slope_prime=" << slope_prime << ",gamma=" << gamma
                << ",Lx=" << Lx << ",modes=" << mode_min << "-" << mode_max << "\n";
            csv << "diagnostics,"
                << "N_samples=" << pooled_x.size()
                << ",frames=" << frame_min << "-" << frame_max
                << "\n";
        }

        // Unchanged: optional plotting (pooled view)
        if (!plot_path.empty()) {
            try {
                sciplot::Plot2D p1;
                p1.xlabel("q");
                p1.ylabel("<|h_q|^2>");
                p1.gnuplot("set logscale xy");
                p1.gnuplot("set grid back linewidth 0.6 linecolor rgb '#E6E6E6'");
                p1.gnuplot("unset key");
                p1.gnuplot("set border linewidth 1.0 linecolor rgb '#4D4D4D'");

                if (!pooled_q.empty()) {
                    p1.drawPoints(pooled_q, pooled_Pavg).pointType(7).pointSize(0.7).lineColor("#6E6E6E");

                    // Use median T for a representative theory band
                    std::vector<double> Ttmp = pooled_T;
                    std::nth_element(Ttmp.begin(), Ttmp.begin()+Ttmp.size()/2, Ttmp.end());
                    const double T_rep = Ttmp[Ttmp.size()/2];

                    double qmin = *std::min_element(pooled_q.begin(), pooled_q.end());
                    double qmax = *std::max_element(pooled_q.begin(), pooled_q.end());
                    if (qmin <= 0.0) qmin = (2.0 * M_PI) / Lx;

                    std::vector<double> qfit, Pfit;
                    qfit.reserve(256); Pfit.reserve(256);
                    for (int i = 0; i < 256; ++i) {
                        double qq = qmin * std::pow(qmax / qmin, double(i) / 255.0);
                        qfit.push_back(qq);
                        Pfit.push_back(std::max(1e-24, T_rep / (std::max(1e-24, gamma) * Lx * qq * qq)));
                    }
                    p1.drawCurve(qfit, Pfit).lineWidth(1.6).lineColor("#2F4F4F");
                } else {
                    std::vector<double> nx{1.0}, ny{1.0};
                    p1.drawCurve(nx, ny);
                }

                sciplot::Plot2D p2;
                p2.xlabel("q^2");
                p2.ylabel("T / <|h_q|^2>");
                p2.gnuplot("set grid back linewidth 0.6 linecolor rgb '#E6E6E6'");
                p2.gnuplot("unset key");
                p2.gnuplot("set border linewidth 1.0 linecolor rgb '#4D4D4D'");

                if (!pooled_x.empty()) {
                    p2.drawPoints(pooled_x, pooled_y).pointType(7).pointSize(0.8).lineColor("#4C78A8");
                    const double xmax = *std::max_element(pooled_x.begin(), pooled_x.end());
                    std::vector<double> xf{0.0, xmax}, yf{intercept_prime, intercept_prime + slope_prime * xmax};
                    p2.drawCurve(xf, yf).lineWidth(1.6).lineColor("#2F4F4F");
                } else {
                    std::vector<double> nx{1.0}, ny{1.0};
                    p2.drawCurve(nx, ny);
                }

                sciplot::Figure fig = {{ p1 }, { p2 }};
                std::ostringstream title;
                title << "Statistical capillary waves (Voronoi slicing): γ ≈ " << gamma
                    << "  [m=" << mode_min << "–" << mode_max
                    << ", frames " << frame_min << "–" << frame_max << "]";
                fig.title(title.str());

                sciplot::Canvas canvas = {{ fig }};
                canvas.size(1100, 900);
                canvas.save(plot_path);
            } catch (const std::exception& e) {
                std::cerr << "[statistical_capillary_wave] Plotting failed: " << e.what() << std::endl;
            }
        }

        return gamma;
    }
    // End of Add








    void save_to_csv_for_plot(const std::string& filepath) {
        for (int i = 0; i < n_gpus; ++i) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaDeviceSynchronize()); }
        for (int i = 0; i < n_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaMemcpy(&h_particles[offsets[i]], d_particles[i],
                                counts[i]*sizeof(Particle), cudaMemcpyDeviceToHost));
        }
        std::ofstream out(filepath);
        if (!out) { std::cerr << "plot CSV open failed: " << filepath << std::endl; return; }
        out << "pos_x,pos_y,type\n";
        for (const auto& p : h_particles) {
            out << p.pos.x << "," << p.pos.y << "," << p.type << std::endl;
        }
    }

    void save_to_file_binary_gz(const std::string& filepath, int step, bool append=false) {
        for (int i = 0; i < n_gpus; ++i) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaDeviceSynchronize()); } // Wait for pervious work to finish
        for (int i = 0; i < n_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaMemcpy(&h_particles[offsets[i]], d_particles[i],
                                counts[i]*sizeof(Particle), cudaMemcpyDeviceToHost));
        }

        gzFile gz = gzopen(filepath.c_str(), append ? "ab" : "wb");
        if (!gz) { std::cerr << "gzopen failed: " << filepath << std::endl; return; }

        if (!append) {
            FileHeader fhdr{ {'M','D','G','Z','D','A','T','A'}, 1u,
                            (uint32_t)N_PARTICLES_TOTAL, BOX_WIDTH, BOX_HEIGHT };
            gzwrite(gz, &fhdr, sizeof(fhdr));
        }

        FrameHeader fh{ step== -1 ? 0 : step, sim_time, dt };
        gzwrite(gz, &fh, sizeof(fh));

        PackedParticle rec;
        for (const auto& p : h_particles) {
            rec.x=p.pos.x; rec.y=p.pos.y;
            rec.vx=p.vel.x; rec.vy=p.vel.y;
            rec.ax=p.acc.x; rec.ay=p.acc.y;
            rec.type = static_cast<uint8_t>(p.type);
            gzwrite(gz, &rec, sizeof(rec));
        }
        gzclose(gz);
    }

    void save(int step) {
        save_to_file_binary_gz(output_filepath, step, (step != 0));
    }

    bool load(int frame_count, const std::string& filepath) {
        if (frame_count == 0) {
            std::cerr << "load(): frame_count must be >=1 or -1 (last intact)\n";
            return false;
        }

        gzFile gz = gzopen(filepath.c_str(), "rb");
        if (!gz) {
            std::cerr << "gzopen failed: " << filepath << std::endl;
            return false;
        }

        // Read and validate file header
        FileHeader fh{};
        if (!gz_read_exact(gz, &fh, sizeof(fh))) {
            std::cerr << "Failed to read FileHeader\n";
            gzclose(gz);
            return false;
        }
        const char expected_magic[8] = {'M','D','G','Z','D','A','T','A'};
        if (std::memcmp(fh.magic, expected_magic, 8) != 0) {
            std::cerr << "Bad magic in header\n";
            gzclose(gz);
            return false;
        }
        if (fh.version != 1u) {
            std::cerr << "Unsupported version: " << fh.version << std::endl;
            gzclose(gz);
            return false;
        }
        const uint32_t N_from_file = fh.n_particles;
        if (N_from_file == 0) {
            std::cerr << "Header says zero particles\n";
            gzclose(gz);
            return false;
        }

        // CHANGED: if geometry mismatches, adopt file’s geometry and reallocate
        const bool need_reconfig =
            (static_cast<uint32_t>(N_PARTICLES_TOTAL) != N_from_file) ||
            (BOX_WIDTH  != fh.box_w) ||
            (BOX_HEIGHT != fh.box_h);

        if (need_reconfig) {
            reshuffle_for_loaded_geometry(N_from_file, fh.box_w, fh.box_h);
        } else {
            if ((int)h_particles.size() != N_PARTICLES_TOTAL)
                h_particles.resize(N_PARTICLES_TOTAL);
        }

        // Walk frames sequentially with sliding read.
        int target_index = (frame_count == -1 ? INT_MAX : frame_count);
        int idx = 0;

        FrameHeader last_good_fh{};
        bool have_last_good = false;

        while (idx < target_index) {
            FrameHeader fh_frame{};
            const bool ok = read_one_frame_into_host(gz, N_from_file, fh_frame);
            if (!ok) {
                if (frame_count == -1 && have_last_good) {
                    sim_time = last_good_fh.sim_time;
                    dt       = last_good_fh.dt;
                    break; // success with last intact
                } else {
                    std::cerr << "File ended before requested frame; possibly truncated.\n";
                    gzclose(gz);
                    return false;
                }
            }

            ++idx;
            last_good_fh = fh_frame;
            have_last_good = true;

            if (frame_count != -1 && idx == target_index) {
                sim_time = last_good_fh.sim_time;
                dt       = last_good_fh.dt;
                break;
            }
        }

        gzclose(gz);

        if (!have_last_good) {
            std::cerr << "No frame data found in file.\n";
            return false;
        }

        // Scatter the selected frame to device
        for (int g = 0; g < n_gpus; ++g) { CUDA_CHECK(cudaSetDevice(g)); }
        for (int g = 0; g < n_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            const int ni  = counts[g];
            const int off = offsets[g];
            if (ni <= 0) continue;

            // Use default stream (0) when streams vector is empty
            cudaStream_t s = (streams.empty() ? static_cast<cudaStream_t>(0) : streams[g]);

            // Explicit error check without using a bool-incompatible ternary condition
            do {
                cudaError_t err = cudaMemcpyAsync(
                    d_particles[g],
                    &h_particles[off],
                    static_cast<size_t>(ni) * sizeof(Particle),
                    cudaMemcpyHostToDevice,
                    s
                );
                if (err != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpyAsync failed on GPU %d: %s\n", g, cudaGetErrorString(err));
                    std::abort();
                }
            } while (0);
        }
        for (int g = 0; g < n_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));

            cudaStream_t s = (streams.empty() ? static_cast<cudaStream_t>(0) : streams[g]);

            do {
                cudaError_t err = cudaStreamSynchronize(s);
                if (err != cudaSuccess) {
                    fprintf(stderr, "cudaStreamSynchronize failed on GPU %d: %s\n", g, cudaGetErrorString(err));
                    std::abort();
                }
            } while (0);
        }

        return true;
    }

    void plot(const std::string& output_filename) {
        const double L = measure_interface_length(true, output_filename);
        std::cout << "[plot] Saved '" << output_filename << "', L_interface = " << L << std::endl;
    }

    double measure_interface_length(bool plot, const std::string& plot_filename) {
        return measure_interface_length_impl(plot, plot_filename);
    }

    double measure_interface_length_impl(bool plot, const std::string& plot_filename)
    {
        // ------------------------------------------------------------
        // Gather current frame from all GPUs to host
        // ------------------------------------------------------------
        for (int g = 0; g < n_gpus; ++g) { CUDA_CHECK(cudaSetDevice(g)); CUDA_CHECK(cudaDeviceSynchronize()); }
        for (int g = 0; g < n_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMemcpy(&h_particles[offsets[g]], d_particles[g],
                                static_cast<size_t>(counts[g]) * sizeof(Particle),
                                cudaMemcpyDeviceToHost));
        }

        const double Lx = this->BOX_WIDTH;
        const double Ly = this->BOX_HEIGHT;

        const int N = static_cast<int>(h_particles.size());
        if (N <= 3) {
            std::cerr << "[measure_interface_length_impl] Not enough particles: N=" << N << std::endl;
            return 0.0;
        }

        std::vector<double2> pts(N);
        std::vector<int>      types(N);
        for (int i = 0; i < N; ++i) {
            pts[i]   = h_particles[i].pos;   // assume already in [0,L)
            types[i] = h_particles[i].type;  // 0 or 1
        }

        // ------------------------------------------------------------
        // Delaunay triangulation on CPU (Bowyer–Watson)
        // `bowyer_watson_cpu(pts, tris)` must be provided elsewhere.
        // ------------------------------------------------------------
        std::vector<std::array<int,3>> tris;
        bowyer_watson_cpu(pts, tris);
        if (tris.empty()) {
            std::cerr << "[measure_interface_length_impl] Delaunay returned 0 triangles.\n";
            return 0.0;
        }

        // ------------------------------------------------------------
        // Build edge -> adjacent triangle indices map
        // ------------------------------------------------------------
        struct EdgeU {
            int a, b;
            EdgeU(int i = 0, int j = 0) {
                if (i < j) { a = i; b = j; } else { a = j; b = i; }
            }
            bool operator==(const EdgeU& o) const noexcept { return a == o.a && b == o.b; }
        };
        struct EdgeUHash {
            size_t operator()(const EdgeU& e) const noexcept {
                return (static_cast<size_t>(e.a) * 1315423911u) ^ (static_cast<size_t>(e.b) * 2654435761u);
            }
        };

        std::unordered_map<EdgeU, std::array<int,2>, EdgeUHash> edge_tris;
        edge_tris.reserve(tris.size() * 2 + 16);

        auto add_edge_adj = [&](int u, int v, int tri_idx) {
            EdgeU e(u, v);
            auto it = edge_tris.find(e);
            if (it == edge_tris.end()) {
                edge_tris.emplace(e, std::array<int,2>{tri_idx, -1});
            } else {
                if (it->second[0] == -1) it->second[0] = tri_idx;
                else                     it->second[1] = tri_idx;
            }
        };

        for (int t = 0; t < (int)tris.size(); ++t) {
            const auto& tr = tris[t];
            add_edge_adj(tr[0], tr[1], t);
            add_edge_adj(tr[1], tr[2], t);
            add_edge_adj(tr[2], tr[0], t);
        }

        // ------------------------------------------------------------
        // Triangle circumcenter (infinite if degenerate)
        // ------------------------------------------------------------
        auto circumcenter_host = [](const double2& A, const double2& B, const double2& C)->double2 {
            const double ax = A.x, ay = A.y, bx = B.x, by = B.y, cx = C.x, cy = C.y;
            const double a2 = ax*ax + ay*ay, b2 = bx*bx + by*by, c2 = cx*cx + cy*cy;
            const double d  = 2.0 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by));
            if (std::fabs(d) < 1e-24)
                return make_double2(std::numeric_limits<double>::quiet_NaN(),
                                    std::numeric_limits<double>::quiet_NaN());
            const double ux = (a2*(by - cy) + b2*(cy - ay) + c2*(ay - by)) / d;
            const double uy = (a2*(cx - bx) + b2*(ax - cx) + c2*(bx - ax)) / d;
            return make_double2(ux, uy);
        };

        // ------------------------------------------------------------
        // Build Voronoi segments separating AB pairs
        // ------------------------------------------------------------
        struct Seg { double2 a, b; };
        std::vector<Seg> ab_voro;
        ab_voro.reserve(edge_tris.size());

        int ab_edges = 0;
        for (const auto& kv : edge_tris) {
            const EdgeU& e = kv.first;
            const int t1 = kv.second[0];
            const int t2 = kv.second[1];
            if (t1 < 0 || t2 < 0) continue;          // boundary (non-closed) edge in non-periodic Delaunay
            if (types[e.a] == types[e.b]) continue;  // not an AB edge

            const auto& T1 = tris[t1];
            const auto& T2 = tris[t2];
            const double2 cc1 = circumcenter_host(pts[T1[0]], pts[T1[1]], pts[T1[2]]);
            const double2 cc2 = circumcenter_host(pts[T2[0]], pts[T2[1]], pts[T2[2]]);
            if (std::isnan(cc1.x) || std::isnan(cc2.x)) continue;

            ab_voro.push_back(Seg{cc1, cc2});
            ++ab_edges;
        }

        std::cout << "[measure_interface_length_impl] N=" << N
                << " tris=" << tris.size()
                << " edge_tris=" << edge_tris.size()
                << " AB_edges=" << ab_edges << std::endl;

        if (ab_voro.empty()) {
            std::cerr << "[measure_interface_length_impl] No AB Voronoi segments found.\n";
            return 0.0;
        }

        // ------------------------------------------------------------
        // Periodic distance utilities
        // ------------------------------------------------------------
        auto pbc_delta = [](double d, double L) {
            if (L <= 0.0) return d;
            d -= std::floor((d / L) + 0.5) * L; // round to nearest integer multiple
            return d;
        };
        auto wrap01 = [](double x, double L) {
            double y = std::fmod(x, L);
            if (y < 0.0) y += L;
            return y;
        };

        // ------------------------------------------------------------
        // Total interface length on the 2D torus: sum of min-image edge lengths
        // ------------------------------------------------------------
        double L_interface = 0.0;
        std::vector<double> draw_x0, draw_y0, draw_x1, draw_y1; // for optional plotting

        for (const auto& s : ab_voro) {
            // min-image vector from a->b
            const double dx = pbc_delta(s.b.x - s.a.x, Lx);
            const double dy = pbc_delta(s.b.y - s.a.y, Ly);
            const double segL = std::sqrt(dx*dx + dy*dy);
            if (!(segL > 0.0)) continue;

            L_interface += segL;

            if (plot) {
                // pick a consistent image for drawing: place 'a' in box, 'b' as a+min-image
                const double ax = wrap01(s.a.x, Lx);
                const double ay = wrap01(s.a.y, Ly);
                const double bx = wrap01(s.a.x + dx, Lx);
                const double by = wrap01(s.a.y + dy, Ly);
                draw_x0.push_back(ax); draw_y0.push_back(ay);
                draw_x1.push_back(bx); draw_y1.push_back(by);
            }
        }

        // ------------------------------------------------------------
        // Optional plot of AB Voronoi edges (requires sciplot / gnuplot)
        // ------------------------------------------------------------
        if (plot && !plot_filename.empty()) {
            try {
                sciplot::Plot2D p;
                p.xlabel("x"); p.ylabel("y");
                p.xrange(0.0, Lx);
                p.yrange(0.0, Ly);
                p.gnuplot("set size ratio -1");
                p.gnuplot("set grid back linewidth 0.6 linecolor rgb '#E6E6E6'");
                p.gnuplot("set border linewidth 1.0 linecolor rgb '#4D4D4D'");
                p.gnuplot("unset key");

                // draw particle points (optional; faint)
                {
                    std::vector<double> xa, ya, xb, yb;
                    xa.reserve(N); ya.reserve(N); xb.reserve(N); yb.reserve(N);
                    for (int i = 0; i < N; ++i) {
                        const double x = wrap01(pts[i].x, Lx);
                        const double y = wrap01(pts[i].y, Ly);
                        if (types[i] == 0) { xa.push_back(x); ya.push_back(y); }
                        else               { xb.push_back(x); yb.push_back(y); }
                    }
                    if (!xa.empty()) p.drawPoints(xa, ya).pointType(7).pointSize(0.5).lineColor("#7F7F7F");
                    if (!xb.empty()) p.drawPoints(xb, yb).pointType(7).pointSize(0.5).lineColor("#1F77B4");
                }

                // draw AB Voronoi segments
                for (size_t i = 0; i < draw_x0.size(); ++i) {
                    std::vector<double> xs{draw_x0[i], draw_x1[i]};
                    std::vector<double> ys{draw_y0[i], draw_y1[i]};
                    p.drawCurve(xs, ys).lineWidth(1.2).lineColor("#D62728");
                }

                sciplot::Figure fig = {{ p }};
                std::ostringstream tt;
                tt << "Interface length ≈ " << L_interface;
                fig.title(tt.str());

                sciplot::Canvas canvas = {{ fig }};
                canvas.size(1000, 900);
                canvas.save(plot_filename);
            } catch (const std::exception& e) {
                std::cerr << "[measure_interface_length_impl] Plotting failed: " << e.what() << std::endl;
            }
        }

        return L_interface;
    }









private:
    std::string output_dir_base;
    std::string output_filepath;
    std::vector<int> counts;
    std::vector<int> offsets;
    int N_PARTICLES_TOTAL;
    int N_PARTICLES_TYPE0;
    double MASS_TYPE0 = 1.0;
    double MASS_TYPE1 = 1.0;
    double BOX_WIDTH, BOX_HEIGHT;
    const double TEMP_INIT;
    // double zeta = 0.0;                
    // const double Q = 100.0;       
    NHState nh_state; // {zeta, Q, target_kinetic_E, dt}
    const double TARGET_TEMP;       
    int DEGREES_OF_FREEDOM = 2 * N_PARTICLES_TOTAL - 2;
    double target_kinetic_energy;
    double ramp_kinetic_energy;
    const double SAVE_DT_INTERVAL = 0.1;
    const double DT_INITIAL = 0.001;
    const double DT_MIN = 1e-9;
    const double DT_MAX = 1e-3;
    const int N_STEPS = 10000000;
    const int OUTPUT_FREQ = 1000;
    const int THREADS_PER_BLOCK = 256;
    const double REL_TOL = 1e-6;
    const double ABS_TOL = 1e-9;
    const double SAFETY_FACTOR = 0.99;
    const double SIGMA_AA;
    const double SIGMA_BB;
    const double SIGMA_AB;
    const double EPSILON_AA;
    const double EPSILON_BB;
    const double EPSILON_AB;

    int n_gpus, particles_per_gpu;
    double dt;
    double sim_time = 0.0;

    std::vector<Particle*> d_particles;           // current y_n (pos, vel, acc)
    std::vector<Particle*> d_particles_stage_in;  // global scratch for broadcasting
    std::vector<Particle*> d_particles_stage_out; // mid state (r_{n+1}, v_half) or scratch
    std::vector<Particle*> d_particles_y_final;   // 1 full step result
    std::vector<Particle*> d_particles_y_hat_final; // 2 half-steps result
    std::vector<double*>    d_error_sq;
    std::vector<cudaStream_t> streams;
    std::vector<Particle> h_particles;
    std::vector<double*> d_block_sums;
    std::vector<double*> d_U_sums;
    std::vector<double*> d_K_sums;

    void setupOutputDirectory(const std::string& run_name) {
        std::string folder_name = run_name;
        if (folder_name.empty()) {
            auto now = std::chrono::system_clock::now();
            auto in_time_t = std::chrono::system_clock::to_time_t(now);
            std::stringstream ss;
            ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
            folder_name = ss.str();
        }
        
        output_dir_base = "./data/" + folder_name + "/";
        
        std::error_code ec;
        std::filesystem::remove(output_dir_base + "equilibrate.bin.gz", ec);
        std::filesystem::remove(output_dir_base + "run.bin.gz", ec);

        try {
            std::filesystem::create_directories(output_dir_base);
            std::cout << "Output directory: " << output_dir_base << std::endl;
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error creating directory: " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void initGPUs() {
        CUDA_CHECK(cudaGetDeviceCount(&n_gpus));
        if (n_gpus == 0) { std::cerr << "No CUDA-enabled GPUs found.\n"; exit(1); }
        std::cout << "Found " << n_gpus << " CUDA GPU(s)\n";

        for (int i = 0; i < n_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            for (int j = 0; j < n_gpus; ++j) {
                if (i == j) continue;
                int can_access;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
                if (can_access) { CUDA_CHECK(cudaDeviceEnablePeerAccess(j, 0)); }
            }
        }

        particles_per_gpu = N_PARTICLES_TOTAL / n_gpus;
        int rem = N_PARTICLES_TOTAL % n_gpus;
        counts.resize(n_gpus);
        offsets.resize(n_gpus);
        int off = 0;
        for (int i = 0; i < n_gpus; ++i) {
            counts[i]  = particles_per_gpu + (i < rem ? 1 : 0);
            offsets[i] = off;
            off += counts[i];
        }
        std::cout << "Particles per GPU (with remainder):";
        for (int i = 0; i < n_gpus; ++i) std::cout << " " << counts[i];
        std::cout << std::endl;
    }


    void allocateMemory() {
        d_particles.resize(n_gpus);
        d_particles_stage_in.resize(n_gpus);
        d_particles_stage_out.resize(n_gpus);
        d_particles_y_final.resize(n_gpus);
        d_particles_y_hat_final.resize(n_gpus);
        d_error_sq.resize(n_gpus);
        streams.resize(n_gpus);
        d_block_sums.resize(n_gpus);
        h_particles.resize(N_PARTICLES_TOTAL);
        d_U_sums.resize(n_gpus);
        d_K_sums.resize(n_gpus);

        for (int i = 0; i < n_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));

            int ni = counts[i];

            CUDA_CHECK(cudaMalloc(&d_particles[i],            ni * sizeof(Particle)));
            CUDA_CHECK(cudaMalloc(&d_particles_stage_in[i],   N_PARTICLES_TOTAL * sizeof(Particle)));
            CUDA_CHECK(cudaMalloc(&d_particles_stage_out[i],  ni * sizeof(Particle)));
            CUDA_CHECK(cudaMalloc(&d_particles_y_final[i],    ni * sizeof(Particle)));
            CUDA_CHECK(cudaMalloc(&d_particles_y_hat_final[i],ni * sizeof(Particle)));
            CUDA_CHECK(cudaMalloc(&d_error_sq[i], sizeof(double)));
            CUDA_CHECK(cudaMemset(d_error_sq[i], 0, sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_U_sums[i], sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_K_sums[i], sizeof(double)));
            CUDA_CHECK(cudaMemset(d_U_sums[i], 0, sizeof(double)));
            CUDA_CHECK(cudaMemset(d_K_sums[i], 0, sizeof(double)));

            int max_blocks = (ni + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            CUDA_CHECK(cudaMalloc(&d_block_sums[i], max_blocks * sizeof(double)));

            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }
    }
    

    void initParticles(const double p)
    {
        // double p = 0.3;

        // Geometry
        const double Lx = BOX_WIDTH;
        const double Ly = BOX_HEIGHT;
        const double band_h = p * Ly;
        const double y0 = 0.5 * (Ly - band_h);   // bottom of central band
        const double y1 = y0 + band_h;           // top of central band

        const int N0 = N_PARTICLES_TYPE0;        // type 0 in outer slabs
        const int N1 = N_PARTICLES_TOTAL - N_PARTICLES_TYPE0;        // type 1 in central band
        const int NT = N_PARTICLES_TOTAL;
        if (N0 + N1 != NT) {
            std::cerr << "initParticles: N_TYPE0 + N_TYPE1 must equal N_TOTAL\n";
            std::abort();
        }

        // -----------------------------
        // Build grid for TYPE 1 (central band)
        // -----------------------------
        std::vector<double2> pos_type1;
        if (N1 > 0) {
            double spacing1 = std::sqrt((Lx * band_h) / static_cast<double>(N1));
            spacing1 = std::max(spacing1, 1e-6);

            int cols1 = std::max(1, static_cast<int>(std::floor(Lx / spacing1)));
            int rows1 = std::max(1, static_cast<int>(std::floor(band_h / spacing1)));
            while (static_cast<long long>(cols1) * static_cast<long long>(rows1) < static_cast<long long>(N1)) {
                spacing1 *= 0.99;
                cols1 = std::max(1, static_cast<int>(std::floor(Lx / spacing1)));
                rows1 = std::max(1, static_cast<int>(std::floor(band_h / spacing1)));
            }

            pos_type1.reserve(static_cast<size_t>(cols1) * static_cast<size_t>(rows1));
            for (int j = 0; j < rows1 && static_cast<int>(pos_type1.size()) < N1; ++j) {
                const double y = y0 + (j + 0.5) * spacing1;
                for (int i = 0; i < cols1 && static_cast<int>(pos_type1.size()) < N1; ++i) {
                    const double x = (i + 0.5) * spacing1;
                    pos_type1.push_back(make_double2(x, y));
                }
            }
        }

        // -----------------------------
        // Build grid for TYPE 0 (outer slabs: [0,y0) and (y1,Ly])
        // -----------------------------
        std::vector<double2> pos_type0;
        if (N0 > 0) {
            const double outer_h = Ly - band_h;  // combined height of bottom+top slabs
            double spacing0 = std::sqrt((Lx * outer_h) / static_cast<double>(N0));
            spacing0 = std::max(spacing0, 1e-6);

            int cols0 = std::max(1, static_cast<int>(std::floor(Lx / spacing0)));
            int rows_bottom = std::max(0, static_cast<int>(std::floor(y0 / spacing0)));
            int rows_top    = std::max(0, static_cast<int>(std::floor((Ly - y1) / spacing0)));
            while (static_cast<long long>(cols0) * static_cast<long long>(rows_bottom + rows_top) < static_cast<long long>(N0)) {
                spacing0 *= 0.99;
                cols0 = std::max(1, static_cast<int>(std::floor(Lx / spacing0)));
                rows_bottom = std::max(0, static_cast<int>(std::floor(y0 / spacing0)));
                rows_top    = std::max(0, static_cast<int>(std::floor((Ly - y1) / spacing0)));
            }

            pos_type0.reserve(static_cast<size_t>(cols0) * static_cast<size_t>(rows_bottom + rows_top));

            // bottom slab: y in [0, y0)
            for (int j = 0; j < rows_bottom && static_cast<int>(pos_type0.size()) < N0; ++j) {
                const double y = (j + 0.5) * spacing0;
                for (int i = 0; i < cols0 && static_cast<int>(pos_type0.size()) < N0; ++i) {
                    const double x = (i + 0.5) * spacing0;
                    pos_type0.push_back(make_double2(x, y));
                }
            }

            // top slab: y in (y1, Ly]
            for (int j = 0; j < rows_top && static_cast<int>(pos_type0.size()) < N0; ++j) {
                const double y = y1 + (j + 0.5) * spacing0;
                for (int i = 0; i < cols0 && static_cast<int>(pos_type0.size()) < N0; ++i) {
                    const double x = (i + 0.5) * spacing0;
                    pos_type0.push_back(make_double2(x, y));
                }
            }
        }

        // -----------------------------
        // Assign positions and types
        // -----------------------------
        if (static_cast<int>(pos_type1.size()) != N1 || static_cast<int>(pos_type0.size()) != N0) {
            std::cerr << "initParticles: could not place requested particle counts ("
                    << "type1=" << pos_type1.size() << "/" << N1 << ", "
                    << "type0=" << pos_type0.size() << "/" << N0 << ")\n";
            std::abort();
        }

        for (int i = 0; i < N1; ++i) {
            h_particles[i].pos = pos_type1[i];
            h_particles[i].acc = make_double2(0.0, 0.0);
            h_particles[i].type = 1;
        }
        for (int i = 0; i < N0; ++i) {
            h_particles[N1 + i].pos = pos_type0[i];
            h_particles[N1 + i].acc = make_double2(0.0, 0.0);
            h_particles[N1 + i].type = 0;
        }

        // -----------------------------
        // Initialize velocities (Maxwell–Boltzmann at TEMP_INIT), remove COM, rescale
        // -----------------------------
        std::mt19937 gen(42);
        std::normal_distribution<double> dist(0.0, 1.0);

        double2 total_momentum = make_double2(0.0, 0.0);
        double total_mass = 0.0;

        for (int i = 0; i < NT; ++i) {
            const int t = h_particles[i].type;
            const double mass = (t == 0) ? MASS_TYPE0 : MASS_TYPE1;
            const double vel_scale = std::sqrt(TEMP_INIT / mass);

            h_particles[i].vel.x = dist(gen) * vel_scale;
            h_particles[i].vel.y = dist(gen) * vel_scale;

            total_momentum.x += h_particles[i].vel.x * mass;
            total_momentum.y += h_particles[i].vel.y * mass;
            total_mass += mass;
        }

        const double2 com_v = make_double2(total_momentum.x / total_mass,
                                        total_momentum.y / total_mass);
        for (int i = 0; i < NT; ++i) {
            h_particles[i].vel.x -= com_v.x;
            h_particles[i].vel.y -= com_v.y;
        }

        double K = 0.0;
        for (int i = 0; i < NT; ++i) {
            const int t = h_particles[i].type;
            const double m = (t == 0) ? MASS_TYPE0 : MASS_TYPE1;
            K += 0.5 * m * (h_particles[i].vel.x * h_particles[i].vel.x +
                            h_particles[i].vel.y * h_particles[i].vel.y);
        }
        const double ideal_K = static_cast<double>(NT - 1) * TEMP_INIT; // keep your original convention
        const double lambda = std::sqrt(ideal_K / K);
        for (int i = 0; i < NT; ++i) {
            h_particles[i].vel.x *= lambda;
            h_particles[i].vel.y *= lambda;
        }

        // -----------------------------
        // Upload to GPUs and compute initial forces
        // -----------------------------
        for (int i = 0; i < n_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaMemcpy(d_particles[i],
                                &h_particles[offsets[i]],
                                counts[i] * sizeof(Particle),
                                cudaMemcpyHostToDevice));
        }

        // dt = DT_INITIAL;

        broadcastAndCalculateForces(
            /*locals (i)*/      d_particles.data(),
            /*globals (j)*/     d_particles_stage_in.data(),
            /*acc_out*/         d_particles_stage_out.data(),
            /*counts*/          counts.data(),
            /*starts*/          offsets.data(),
            /*n_gpus*/          n_gpus,
            /*box*/             BOX_WIDTH, BOX_HEIGHT,
            /*masses*/          MASS_TYPE0, MASS_TYPE1,
            /*LJ AA*/           SIGMA_AA, EPSILON_AA,
            /*LJ BB*/           SIGMA_BB, EPSILON_BB,
            /*LJ AB*/           SIGMA_AB, EPSILON_AB,
            /*ensemble*/        "NVT",
            /*NH phase*/        NHPhase::NONE,
            /*NH state*/        nh_state,
            /*energy sums*/     d_U_sums, d_K_sums,
            /*streams*/         streams
        );

        for (int i = 0; i < n_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            const int n_blocks = (counts[i] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            copyAccelerations<<<n_blocks, THREADS_PER_BLOCK, 0, streams[i]>>>(
                d_particles[i], d_particles_stage_out[i], counts[i]);
            CUDA_CHECK(cudaGetLastError());
        }
        for (int i = 0; i < n_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }

        std::cout << "Initialization complete (band p=" << p << ")\n";
    }

    void cleanup() {
        for (int i = 0; i < n_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaFree(d_particles[i]));
            CUDA_CHECK(cudaFree(d_particles_stage_in[i]));
            CUDA_CHECK(cudaFree(d_particles_stage_out[i]));
            CUDA_CHECK(cudaFree(d_particles_y_final[i]));
            CUDA_CHECK(cudaFree(d_particles_y_hat_final[i]));
            CUDA_CHECK(cudaFree(d_error_sq[i]));
            CUDA_CHECK(cudaFree(d_K_sums[i]));
            CUDA_CHECK(cudaFree(d_U_sums[i]));
            CUDA_CHECK(cudaFree(d_block_sums[i]));
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }
    }

    double calculateTotalKineticEnergy() {
        double total_K = 0.0;
        std::vector<double> h_block_sums;

        for (int i = 0; i < n_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            int ni = counts[i];
            int n_blocks = (ni + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            size_t shmem = THREADS_PER_BLOCK * sizeof(double);

            reduceKineticEnergy<<<n_blocks, THREADS_PER_BLOCK, shmem, streams[i]>>>(
                d_block_sums[i], d_particles[i], ni, MASS_TYPE0, MASS_TYPE1); // [CHANGED]

            h_block_sums.resize(n_blocks);
            CUDA_CHECK(cudaMemcpyAsync(h_block_sums.data(), d_block_sums[i],
                                    n_blocks * sizeof(double),
                                    cudaMemcpyDeviceToHost, streams[i]));
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));

            for (int j = 0; j < n_blocks; ++j) total_K += h_block_sums[j];
        }
        return total_K;
    }

    

    void reshuffle_for_loaded_geometry(uint32_t newN, double newW, double newH) {
        // update basics
        std::cout << "Loaded Width: "<< newW << ", Loaded Height: " << newH << std::endl;
        N_PARTICLES_TOTAL   = static_cast<int>(newN);
        BOX_WIDTH           = newW;
        BOX_HEIGHT          = newH;
        DEGREES_OF_FREEDOM  = 2 * N_PARTICLES_TOTAL - 2;
        target_kinetic_energy = static_cast<double>(DEGREES_OF_FREEDOM) * TARGET_TEMP / 2.0;

        // host storage
        h_particles.resize(N_PARTICLES_TOTAL);

        // recompute sharding
        particles_per_gpu = N_PARTICLES_TOTAL / n_gpus;
        int rem = N_PARTICLES_TOTAL % n_gpus;
        counts.resize(n_gpus);
        offsets.resize(n_gpus);
        int off = 0;
        for (int i = 0; i < n_gpus; ++i) {
            counts[i]  = particles_per_gpu + (i < rem ? 1 : 0);
            offsets[i] = off;
            off += counts[i];
        }

        // (re)allocate device buffers sized to new shard/global sizes
        for (int g = 0; g < n_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));

            auto safe_free = [&](void* p) {
                if (p) CUDA_CHECK(cudaFree(p));
            };

            safe_free(d_particles[g]);
            safe_free(d_particles_stage_in[g]);
            safe_free(d_particles_stage_out[g]);
            safe_free(d_particles_y_final[g]);
            safe_free(d_particles_y_hat_final[g]);
            safe_free(d_block_sums[g]);
            // keep d_U_sums[g], d_K_sums[g] (scalars) and streams[g]

            const int ni = counts[g];
            const int nb = (ni + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

            CUDA_CHECK(cudaMalloc(&d_particles[g],             ni * sizeof(Particle)));
            CUDA_CHECK(cudaMalloc(&d_particles_stage_in[g],    N_PARTICLES_TOTAL * sizeof(Particle))); // full system view
            CUDA_CHECK(cudaMalloc(&d_particles_stage_out[g],   ni * sizeof(Particle)));
            CUDA_CHECK(cudaMalloc(&d_particles_y_final[g],     ni * sizeof(Particle)));
            CUDA_CHECK(cudaMalloc(&d_particles_y_hat_final[g], ni * sizeof(Particle)));
            CUDA_CHECK(cudaMalloc(&d_block_sums[g],            nb * sizeof(double)));

            // zero energy accumulators
            if (d_U_sums[g]) CUDA_CHECK(cudaMemset(d_U_sums[g], 0, sizeof(double)));
            if (d_K_sums[g]) CUDA_CHECK(cudaMemset(d_K_sums[g], 0, sizeof(double)));
        }
    }

    void broadcastAndCalculateForces(
        Particle** d_locals,        // per-GPU: local slice used as "i"
        Particle** d_globals,       // per-GPU: full system buffer used as "j"
        Particle** d_acc_out,       // per-GPU: write accelerations for local slice
        int* counts,                // per-GPU local counts
        int* starts,                // per-GPU local start indices in global order
        int  n_gpus,
        double BOX_WIDTH, double BOX_HEIGHT,
        double mass0, double mass1,
        double SIGMA_AA, double EPSILON_AA,
        double SIGMA_BB, double EPSILON_BB,
        double SIGMA_AB, double EPSILON_AB,
        const std::string& ensemble, NHPhase phase, NHState& nh,
        std::vector<double*>& d_U_sums, std::vector<double*>& d_K_sums,
        std::vector<cudaStream_t>& streams
    ) {
        const bool use_NVT = (ensemble == "NVT");
        const int THREADS  = THREADS_PER_BLOCK;

        // --- Broadcast: assemble the full system y into each GPU's d_globals[g] ---
        for (int g_dst = 0; g_dst < n_gpus; ++g_dst) {
            CUDA_CHECK(cudaSetDevice(g_dst));
            for (int g_src = 0; g_src < n_gpus; ++g_src) {
                const size_t nbytes = (size_t)counts[g_src] * sizeof(Particle);
                Particle* dst_ptr   = d_globals[g_dst] + starts[g_src];
                if (g_src == g_dst) {
                    CUDA_CHECK(cudaMemcpyAsync(dst_ptr, d_locals[g_src], nbytes,
                                            cudaMemcpyDeviceToDevice, streams[g_dst]));
                } else {
                    CUDA_CHECK(cudaMemcpyPeerAsync(dst_ptr, g_dst, d_locals[g_src], g_src,
                                                nbytes, streams[g_dst]));
                }
            }
        }
        for (int g = 0; g < n_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaStreamSynchronize(streams[g]));
        }

        // --- Nose–Hoover PRE half-step: compute K, update ζ, scale v by exp(-ζ Δt/2) ---
        if (use_NVT && phase == NHPhase::PRE) {
            double K_total = 0.0;
            std::vector<double> h_block_sums; // reused per GPU

            for (int g = 0; g < n_gpus; ++g) {
                CUDA_CHECK(cudaSetDevice(g));
                const int ni = counts[g];
                const int nb = (ni + THREADS - 1) / THREADS;
                const size_t shmem = THREADS * sizeof(double);

                reduceKineticEnergy<<<nb, THREADS, shmem, streams[g]>>>(
                    d_block_sums[g], d_locals[g], ni, mass0, mass1);
                CUDA_CHECK(cudaGetLastError());

                h_block_sums.resize(nb);
                CUDA_CHECK(cudaMemcpyAsync(h_block_sums.data(), d_block_sums[g],
                                        nb * sizeof(double), cudaMemcpyDeviceToHost, streams[g]));
                CUDA_CHECK(cudaStreamSynchronize(streams[g]));
                for (int j = 0; j < nb; ++j) K_total += h_block_sums[j];
            }

            nh.zeta += (nh.dt * 0.5 / nh.Q) * (K_total - nh.target_kinetic_E);

            const double scale = std::exp(-nh.zeta * (nh.dt * 0.5));
            for (int g = 0; g < n_gpus; ++g) {
                CUDA_CHECK(cudaSetDevice(g));
                const int nb = (counts[g] + THREADS - 1) / THREADS;
                applyThermostatScaling<<<nb, THREADS, 0, streams[g]>>>(d_locals[g], counts[g], scale);
                CUDA_CHECK(cudaGetLastError());
            }
            for (int g = 0; g < n_gpus; ++g) {
                CUDA_CHECK(cudaSetDevice(g));
                CUDA_CHECK(cudaStreamSynchronize(streams[g]));
            }

            // velocities changed -> refresh broadcast of locals into globals
            for (int g_dst = 0; g_dst < n_gpus; ++g_dst) {
                CUDA_CHECK(cudaSetDevice(g_dst));
                for (int g_src = 0; g_src < n_gpus; ++g_src) {
                    const size_t nbytes = (size_t)counts[g_src] * sizeof(Particle);
                    Particle* dst_ptr   = d_globals[g_dst] + starts[g_src];
                    if (g_src == g_dst) {
                        CUDA_CHECK(cudaMemcpyAsync(dst_ptr, d_locals[g_src], nbytes,
                                                cudaMemcpyDeviceToDevice, streams[g_dst]));
                    } else {
                        CUDA_CHECK(cudaMemcpyPeerAsync(dst_ptr, g_dst, d_locals[g_src], g_src,
                                                    nbytes, streams[g_dst]));
                    }
                }
            }
            for (int g = 0; g < n_gpus; ++g) {
                CUDA_CHECK(cudaSetDevice(g));
                CUDA_CHECK(cudaStreamSynchronize(streams[g]));
            }
        }

        // --- Forces (+ U, K accumulation) using current ζ (friction term inside kernel) ---
        for (int g = 0; g < n_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            zero_scalar<<<1,1,0,streams[g]>>>(d_U_sums[g]);
            zero_scalar<<<1,1,0,streams[g]>>>(d_K_sums[g]);
            const int nb = (counts[g] + THREADS - 1) / THREADS;

            calculateForces<<<nb, THREADS, 0, streams[g]>>>(
                d_acc_out[g],          // write acc here
                d_globals[g],          // read all j here
                d_locals[g],           // read i (positions/velocities/types) here
                counts[g],
                N_PARTICLES_TOTAL,
                starts[g],
                BOX_WIDTH, BOX_HEIGHT,
                mass0,  mass1,
                SIGMA_AA, EPSILON_AA,
                SIGMA_BB, EPSILON_BB,
                SIGMA_AB, EPSILON_AB,
                d_U_sums[g], d_K_sums[g],
                (use_NVT ? 1 : 0),
                nh.zeta
            );
            CUDA_CHECK(cudaGetLastError());
        }
        for (int g = 0; g < n_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaStreamSynchronize(streams[g]));
        }

        // --- Host aggregation of energies (available for diagnostics and POST ζ update) ---
        double U_total = 0.0, K_total = 0.0;
        for (int g = 0; g < n_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            U_total += copy_scalar_from_device(d_U_sums[g]);
            K_total += copy_scalar_from_device(d_K_sums[g]);
        }

        // --- Nose–Hoover POST half-step: update ζ and scale v by exp(-ζ Δt/2) ---
        if (use_NVT && phase == NHPhase::POST) {
            nh.zeta += (nh.dt * 0.5 / nh.Q) * (K_total - nh.target_kinetic_E);

            const double scale = std::exp(-nh.zeta * (nh.dt * 0.5));
            for (int g = 0; g < n_gpus; ++g) {
                CUDA_CHECK(cudaSetDevice(g));
                const int nb = (counts[g] + THREADS - 1) / THREADS;
                applyThermostatScaling<<<nb, THREADS, 0, streams[g]>>>(d_locals[g], counts[g], scale);
                CUDA_CHECK(cudaGetLastError());
            }
            for (int g = 0; g < n_gpus; ++g) {
                CUDA_CHECK(cudaSetDevice(g));
                CUDA_CHECK(cudaStreamSynchronize(streams[g]));
            }
        }
    }


    void performTimestep(const std::string& ensemble, const double t_stop = -1.0, bool* trigger_stop = nullptr) {
        const double EPS_TIME = 1e-8;
        // dt = 1e-3;

        // Unchanged: 1) first kick + drift
        for (int i = 0; i < n_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            int ni = counts[i];
            int nb = (ni + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            vv_first_kick_drift<<<nb, THREADS_PER_BLOCK, 0, streams[i]>>>(d_particles_stage_out[i], d_particles[i], ni, dt, BOX_WIDTH, BOX_HEIGHT);
        }
        for (int i = 0; i < n_gpus; ++i) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaStreamSynchronize(streams[i])); }

        // ADD: 2) Nose–Hoover PRE half-step scaling + a(t+dt) at mid
        nh_state.dt = dt; // ensure set
        broadcastAndCalculateForces(
            /*locals*/      d_particles_stage_out.data(),
            /*globals*/     d_particles_stage_in.data(),
            /*acc_out*/     d_particles_stage_in.data(),
            counts.data(), offsets.data(), n_gpus,
            BOX_WIDTH, BOX_HEIGHT,
            MASS_TYPE0, MASS_TYPE1,
            SIGMA_AA, EPSILON_AA, SIGMA_BB, EPSILON_BB, SIGMA_AB, EPSILON_AB,
            ensemble, NHPhase::PRE, nh_state,
            d_U_sums, d_K_sums, streams
        );
        // End of Add

        // Unchanged: 3) second kick + finalize
        for (int i = 0; i < n_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            int ni = counts[i];
            int nb = (ni + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            vv_second_kick_finalize<<<nb, THREADS_PER_BLOCK, 0, streams[i]>>>(d_particles_y_final[i], d_particles_stage_out[i], d_particles_stage_in[i], ni, dt);
        }
        for (int i = 0; i < n_gpus; ++i) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaStreamSynchronize(streams[i])); }

        // Unchanged: 4) commit state + wrap
        for (int i = 0; i < n_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            int ni = counts[i];
            int nb = (ni + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            apply_final_state<<<nb, THREADS_PER_BLOCK, 0, streams[i]>>>(d_particles[i], d_particles_y_final[i], ni, BOX_WIDTH, BOX_HEIGHT);
        }
        for (int i = 0; i < n_gpus; ++i) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaStreamSynchronize(streams[i])); }

        // ADD: 5) POST half-step scaling + refresh a(t+dt) into d_particles[].acc
        broadcastAndCalculateForces(
            /*locals*/      d_particles.data(),
            /*globals*/     d_particles_stage_in.data(),
            /*acc_out*/     d_particles_stage_out.data(),
            counts.data(), offsets.data(), n_gpus,
            BOX_WIDTH, BOX_HEIGHT,
            MASS_TYPE0, MASS_TYPE1,
            SIGMA_AA, EPSILON_AA, SIGMA_BB, EPSILON_BB, SIGMA_AB, EPSILON_AB,
            ensemble, NHPhase::POST, nh_state,
            d_U_sums, d_K_sums, streams
        );
        // copy accelerations from stage_out into main state
        for (int i = 0; i < n_gpus; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            int ni = counts[i];
            int nb = (ni + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            copyAccelerations<<<nb, THREADS_PER_BLOCK, 0, streams[i]>>>(d_particles[i], d_particles_stage_out[i], ni);
        }
        for (int i = 0; i < n_gpus; ++i) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaStreamSynchronize(streams[i])); }
        // End of Add

        // ADD: disable legacy thermostat block (handled by NH)
        // (formerly here)
        // End of Add

        sim_time += dt;

        // Unchanged: trigger save/stop and device sync
        if (trigger_stop && t_stop > 0.0 && sim_time >= t_stop - EPS_TIME) *trigger_stop = true;
        for (int i = 0; i < n_gpus; ++i) { CUDA_CHECK(cudaSetDevice(i)); CUDA_CHECK(cudaDeviceSynchronize()); }
    }

    static bool gz_read_exact(gzFile gz, void* dst, size_t nbytes) {
        unsigned char* p = static_cast<unsigned char*>(dst);
        size_t got = 0;
        while (got < nbytes) {
            const size_t want = std::min(nbytes - got, (size_t)INT_MAX);
            int rd = gzread(gz, p + got, (unsigned int)want);
            if (rd <= 0) return false;
            got += (size_t)rd;
        }
        return true;
    }

    // reads one full frame into h_particles using a small sliding buffer. 
    // Returns true if an entire frame (header + N records) was read.
    bool read_one_frame_into_host(gzFile gz, uint32_t N, FrameHeader& fh_out) {
        if (!gz_read_exact(gz, &fh_out, sizeof(fh_out))) return false;

        constexpr uint32_t CHUNK = 8192;
        std::vector<PackedParticle> buf(std::min(CHUNK, N));

        uint32_t done = 0;
        while (done < N) {
            const uint32_t take = std::min<uint32_t>(CHUNK, N - done);
            if (!gz_read_exact(gz, buf.data(), sizeof(PackedParticle) * (size_t)take)) {
                return false; // truncated frame
            }
            for (uint32_t k = 0; k < take; ++k) {
                const PackedParticle& r = buf[k];
                Particle& p = h_particles[done + k];
                p.pos.x = r.x;   p.pos.y = r.y;
                p.vel.x = r.vx;  p.vel.y = r.vy;
                p.acc.x = r.ax;  p.acc.y = r.ay;
                p.type  = (int)r.type;
            }
            done += take;
        }
        return true;
    }

};

static bool parse_int(const char* s, int& out) {
    errno = 0; 
    char* end = nullptr; 
    long v = std::strtol(s, &end, 10); 
    if (errno != 0 || end == s || *end != '\0') return false;
    if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) return false;
    out = static_cast<int>(v);
    return true;
} 

static bool parse_double(const char* s, double& out) {
    errno = 0;
    char* end = nullptr;
    double v = std::strtod(s, &end);
    if (errno != 0 || end == s || *end != '\0') return false;
    out = v;
    return true;
}





// int main(int argc, char** argv) {
//     // Defaults to match argparse version
//     int   n_particles        = 1024;
//     int   n_particles_type0  = 512;
//     double temp_init          = 0.0; // required -> must be provided
//     double lambda_param       = 0.0; // required -> must be provided
//     std::string run_name     = "test";

//     bool have_temp = false;
//     bool have_lambda = false;

//     static option long_opts[] = {
//         {"n_particles",         required_argument, nullptr, 'p'},
//         {"n_particles_type0",   required_argument, nullptr, 't'},
//         {"temp_init",           required_argument, nullptr, 'T'},
//         {"lambda",              required_argument, nullptr, 'l'},
//         {"run_name",            required_argument, nullptr, 'r'},
//         {"help",                no_argument,       nullptr, 'h'},
//         {nullptr,               0,                 nullptr,  0 }
//     };

//     int opt, opt_index = 0;
//     // Short opts mirror the above: letters p t T l r h with required args where needed
//     while ((opt = getopt_long(argc, argv, "p:t:T:l:r:h", long_opts, &opt_index)) != -1) {
//         switch (opt) {
//             case 'p':
//                 if (!parse_int(optarg, n_particles) || n_particles <= 0) {
//                     std::cerr << "Error: --n_particles must be a positive integer.\n";
//                     print_usage(argv[0]); return 1;
//                 }
//                 break;
//             case 't':
//                 if (!parse_int(optarg, n_particles_type0) || n_particles_type0 < 0) {
//                     std::cerr << "Error: --n_particles_type0 must be a nonnegative integer.\n";
//                     print_usage(argv[0]); return 1;
//                 }
//                 break;
//             case 'T':
//                 if (!parse_double(optarg, temp_init)) {
//                     std::cerr << "Error: --temp_init must be a double.\n";
//                     print_usage(argv[0]); return 1;
//                 }
//                 have_temp = true;
//                 break;
//             case 'l':
//                 if (!parse_double(optarg, lambda_param) || lambda_param <= 0.0) {
//                     std::cerr << "Error: --lambda must be a positive double.\n";
//                     print_usage(argv[0]); return 1;
//                 }
//                 have_lambda = true;
//                 break;
//             case 'r':
//                 run_name = optarg ? std::string(optarg) : std::string{};
//                 break;
//             case 'h':
//                 print_usage(argv[0]); return 0;
//             default:
//                 print_usage(argv[0]); return 1;
//         }
//     }

//     // Enforce required args like argparse would
//     if (!have_temp || !have_lambda) {
//         if (!have_temp)   std::cerr << "Missing required argument: --temp_init\n";
//         if (!have_lambda) std::cerr << "Missing required argument: --lambda\n";
//         print_usage(argv[0]); return 1;
//     }

//     // Extra sanity: type0 count cannot exceed total
//     if (n_particles_type0 > n_particles) {
//         std::cerr << "Error: --n_particles_type0 (" << n_particles_type0
//                   << ") cannot exceed --n_particles (" << n_particles << ").\n";
//         return 1;
//     }

//     // Geometry from your original
//     // const double area = n_particles / 0.6;
//     const double area = 196747.0;
//     const double box_width  = std::sqrt(area / lambda_param);
//     const double box_height = std::sqrt(area * lambda_param);

//     MDSimulation sim(n_particles, n_particles_type0,
//             box_width, box_height, 
//             temp_init, temp_target,
//             SIGMA_AA, SIGMA_BB, SIGMA_AB,
//             EPSILON_AA, EPSILON_BB, EPSILON_AB,
//             devide_p,
//             dt,
//             run_name)

//     sim.equilibrate(1000.0);
//     sim.run(1000.0);

//     return 0;
// }

int main(int argc, char** argv) {
    // Defaults
    int         N_PARTICLES_TOTAL = -1;
    int         N_PARTICLES_TYPE0 = -1;
    double      BOX_WIDTH         = 0.0;
    double      BOX_HEIGHT        = 0.0;
    double      T_init            = 0.0;
    double      T_target          = 0.0;
    
    double      SIGMA_AA          = 1.0;
    double      SIGMA_BB          = 1.0;
    double      SIGMA_AB          = 1.0;
    
    double      EPSILON_AA        = 1.0;
    double      EPSILON_BB        = 1.0;
    double      EPSILON_AB        = 1.0;
    
    double      devide_p          = 0.5;
    double      dt                = 1e-3;
    int         save_interval     = 1000;
    
    std::string base_dir          = "test";
    
    int         mode_min          = -1;
    int         mode_max          = -1;
    int         frame_min         = -1;
    int         frame_max         = -1;
    
    // Help printer
    auto print_usage = [&](const char* prog){
        std::cerr
            << "Usage: " << prog << " [options]\n\n"
            << "Required:\n"
            << "  --NT <int>                 Total number of particles\n"
            << "  --N0 <int>                 Number of type-0 particles\n"
            << "  --BoxW <double>            Box width\n"
            << "  --BoxH <double>            Box height\n"
            << "  --Ti <double>              Initial temperature\n"
            << "  --Ttarget <double>         Target temperature\n"
            << "  --ModeMin <int>            Min Fourier mode (inclusive) for fit\n"
            << "  --ModeMax <int>            Max Fourier mode (inclusive) for fit\n"
            << "  --FrameMin <int>           First frame index (inclusive)\n"
            << "  --FrameMax <int>           Last frame index (inclusive)\n"
            << std::endl
            << "Optional (defaults in brackets):\n"
            << "  --SIGMA-AA <double>        Lennard-Jones sigma_AA [" << SIGMA_AA << "]\n"
            << "  --SIGMA-BB <double>        Lennard-Jones sigma_BB [" << SIGMA_BB << "]\n"
            << "  --SIGMA-AB <double>        Lennard-Jones sigma_AB [" << SIGMA_AB << "]\n"
            << "  --EPSILON_AA <double>      Lennard-Jones epsilon_AA [" << EPSILON_AA << "]\n"
            << "  --EPSILON-BB <double>      Lennard-Jones epsilon_BB [" << EPSILON_BB << "]\n"
            << "  --EPSILON-AB <double>      Lennard-Jones epsilon_AB [" << EPSILON_AB << "]\n"
            << "  --DevideP <double>         Central band fraction p in y [" << devide_p << "]\n"
            << "  --dt <double>              Timestep size [" << dt << "]\n"
            << "  --SaveInterval <int>       Save/plot interval in steps [" << save_interval << "]\n"
            << "  --BaseDir <string>         Output base directory [" << base_dir << "]\n"
            << "  -h, --help                 Show this help and exit\n";
    };
    
    // Long-option IDs
    enum : int {
        OPT_NT = 1000, OPT_N0,
        OPT_BOXW, OPT_BOXH,
        OPT_TI, OPT_TTARGET,
        OPT_SIGMA_AA, OPT_SIGMA_BB, OPT_SIGMA_AB,
        OPT_EPSILON_AA, OPT_EPSILON_BB, OPT_EPSILON_AB,
        OPT_DEVIDEP, OPT_DT, OPT_SAVEINTERVAL,
        OPT_BASEDIR,
        OPT_MODEMIN, OPT_MODEMAX,
        OPT_FRAMEMIN, OPT_FRAMEMAX,
    };
    
    static option long_opts[] = {
        {"NT",             required_argument, nullptr, OPT_NT},
        {"N0",             required_argument, nullptr, OPT_N0},
        {"BoxW",           required_argument, nullptr, OPT_BOXW},
        {"BoxH",           required_argument, nullptr, OPT_BOXH},
        {"Ti",             required_argument, nullptr, OPT_TI},
        {"Ttarget",        required_argument, nullptr, OPT_TTARGET},
    
        // Accept both hyphen and underscore variants where relevant
        {"SIGMA-AA",       required_argument, nullptr, OPT_SIGMA_AA},
        {"SIGMA-BB",       required_argument, nullptr, OPT_SIGMA_BB},
        {"SIGMA-AB",       required_argument, nullptr, OPT_SIGMA_AB},
        {"SIGMA_AA",       required_argument, nullptr, OPT_SIGMA_AA},
        {"SIGMA_BB",       required_argument, nullptr, OPT_SIGMA_BB},
        {"SIGMA_AB",       required_argument, nullptr, OPT_SIGMA_AB},
    
        {"EPSILON_AA",     required_argument, nullptr, OPT_EPSILON_AA},
        {"EPSILON-BB",     required_argument, nullptr, OPT_EPSILON_BB},
        {"EPSILON-AB",     required_argument, nullptr, OPT_EPSILON_AB},
        {"EPSILON-AA",     required_argument, nullptr, OPT_EPSILON_AA},
        {"EPSILON_BB",     required_argument, nullptr, OPT_EPSILON_BB},
        {"EPSILON_AB",     required_argument, nullptr, OPT_EPSILON_AB},
    
        {"DevideP",        required_argument, nullptr, OPT_DEVIDEP},
        {"dt",             required_argument, nullptr, OPT_DT},
        {"SaveInterval",   required_argument, nullptr, OPT_SAVEINTERVAL},
        {"BaseDir",        required_argument, nullptr, OPT_BASEDIR},
    
        {"ModeMin",        required_argument, nullptr, OPT_MODEMIN},
        {"ModeMax",        required_argument, nullptr, OPT_MODEMAX},
        {"FrameMin",       required_argument, nullptr, OPT_FRAMEMIN},
        {"FrameMax",       required_argument, nullptr, OPT_FRAMEMAX},
    
        {"help",           no_argument,       nullptr, 'h'},
        {nullptr,          0,                 nullptr,  0 }
    };
    
    // Parse
    int optc, optidx = 0;
    opterr = 0;  // we will print our own messages
    while ((optc = getopt_long(argc, argv, "h", long_opts, &optidx)) != -1) {
        switch (optc) {
            case 'h':
                print_usage(argv[0]);
                std::exit(0);
    
            case OPT_NT:
                if (!parse_int(optarg, N_PARTICLES_TOTAL) || N_PARTICLES_TOTAL <= 0) {
                    std::cerr << "Error: --NT requires positive int.\n"; print_usage(argv[0]); std::exit(1);
                }
                break;
            case OPT_N0:
                if (!parse_int(optarg, N_PARTICLES_TYPE0) || N_PARTICLES_TYPE0 < 0) {
                    std::cerr << "Error: --N0 requires nonnegative int.\n"; print_usage(argv[0]); std::exit(1);
                }
                break;
            case OPT_BOXW:
                if (!parse_double(optarg, BOX_WIDTH) || BOX_WIDTH <= 0.0) {
                    std::cerr << "Error: --BoxW requires positive double.\n"; print_usage(argv[0]); std::exit(1);
                }
                break;
            case OPT_BOXH:
                if (!parse_double(optarg, BOX_HEIGHT) || BOX_HEIGHT <= 0.0) {
                    std::cerr << "Error: --BoxH requires positive double.\n"; print_usage(argv[0]); std::exit(1);
                }
                break;
            case OPT_TI:
                if (!parse_double(optarg, T_init)) {
                    std::cerr << "Error: --Ti requires double.\n"; print_usage(argv[0]); std::exit(1);
                }
                break;
            case OPT_TTARGET:
                if (!parse_double(optarg, T_target)) {
                    std::cerr << "Error: --Ttarget requires double.\n"; print_usage(argv[0]); std::exit(1);
                }
                break;
    
            case OPT_SIGMA_AA:
                if (!parse_double(optarg, SIGMA_AA) || SIGMA_AA <= 0.0) {
                    std::cerr << "Error: --SIGMA-AA requires positive double.\n"; print_usage(argv[0]); std::exit(1);
                }
                break;
            case OPT_SIGMA_BB:
                if (!parse_double(optarg, SIGMA_BB) || SIGMA_BB <= 0.0) {
                    std::cerr << "Error: --SIGMA-BB requires positive double.\n"; print_usage(argv[0]); std::exit(1);
                }
                break;
            case OPT_SIGMA_AB:
                if (!parse_double(optarg, SIGMA_AB) || SIGMA_AB <= 0.0) {
                    std::cerr << "Error: --SIGMA-AB requires positive double.\n"; print_usage(argv[0]); std::exit(1);
                }
                break;
    
            case OPT_EPSILON_AA:
                if (!parse_double(optarg, EPSILON_AA) || EPSILON_AA <= 0.0) {
                    std::cerr << "Error: --EPSILON_AA/--EPSILON-AA requires positive double.\n"; print_usage(argv[0]); std::exit(1);
                }
                break;
            case OPT_EPSILON_BB:
                if (!parse_double(optarg, EPSILON_BB) || EPSILON_BB <= 0.0) {
                    std::cerr << "Error: --EPSILON-BB requires positive double.\n"; print_usage(argv[0]); std::exit(1);
                }
                break;
            case OPT_EPSILON_AB:
                if (!parse_double(optarg, EPSILON_AB) || EPSILON_AB <= 0.0) {
                    std::cerr << "Error: --EPSILON-AB requires positive double.\n"; print_usage(argv[0]); std::exit(1);
                }
                break;
    
            case OPT_DEVIDEP:
                if (!parse_double(optarg, devide_p) || !(devide_p >= 0.0 && devide_p <= 1.0)) {
                    std::cerr << "Error: --DevideP requires double in [0,1].\n"; print_usage(argv[0]); std::exit(1);
                }
                break;
            case OPT_DT:
                if (!parse_double(optarg, dt) || dt <= 0.0) {
                    std::cerr << "Error: --dt requires positive double.\n"; print_usage(argv[0]); std::exit(1);
                }
                break;
            case OPT_SAVEINTERVAL:
                if (!parse_int(optarg, save_interval) || save_interval <= 0) {
                    std::cerr << "Error: --SaveInterval requires positive int.\n"; print_usage(argv[0]); std::exit(1);
                }
                break;
            case OPT_BASEDIR:
                base_dir = (optarg ? std::string(optarg) : std::string{});
                if (base_dir.empty()) {
                    std::cerr << "Error: --BaseDir requires non-empty string.\n"; print_usage(argv[0]); std::exit(1);
                }
                break;
    
            case OPT_MODEMIN:
                if (!parse_int(optarg, mode_min) || mode_min < 1) {
                    std::cerr << "Error: --ModeMin requires int >= 1.\n"; print_usage(argv[0]); std::exit(1);
                }
                break;
            case OPT_MODEMAX:
                if (!parse_int(optarg, mode_max) || mode_max < 1) {
                    std::cerr << "Error: --ModeMax requires int >= 1.\n"; print_usage(argv[0]); std::exit(1);
                }
                break;
            case OPT_FRAMEMIN:
                if (!parse_int(optarg, frame_min) || frame_min < 1) {
                    std::cerr << "Error: --FrameMin requires int >= 1.\n"; print_usage(argv[0]); std::exit(1);
                }
                break;
            case OPT_FRAMEMAX:
                if (!parse_int(optarg, frame_max) || frame_max < 1) {
                    std::cerr << "Error: --FrameMax requires int >= 1.\n"; print_usage(argv[0]); std::exit(1);
                }
                break;
    
            case '?':
            default:
                std::cerr << "Error: unrecognized option.\n";
                print_usage(argv[0]);
                std::exit(1);
        }
    }
    
    // Post-parse validation of requireds
    auto missing = [&](const char* name)->bool{
        if      (std::strcmp(name,"NT")==0)       return N_PARTICLES_TOTAL <  0;
        else if (std::strcmp(name,"N0")==0)       return N_PARTICLES_TYPE0 <  0;
        else if (std::strcmp(name,"BoxW")==0)     return !(BOX_WIDTH  > 0.0);
        else if (std::strcmp(name,"BoxH")==0)     return !(BOX_HEIGHT > 0.0);
        else if (std::strcmp(name,"Ti")==0)       return false; // allow 0.0 if user wants
        else if (std::strcmp(name,"Ttarget")==0)  return false; // allow 0.0 if user wants
        else if (std::strcmp(name,"ModeMin")==0)  return mode_min  <  0;
        else if (std::strcmp(name,"ModeMax")==0)  return mode_max  <  0;
        else if (std::strcmp(name,"FrameMin")==0) return frame_min <  0;
        else if (std::strcmp(name,"FrameMax")==0) return frame_max <  0;
        return false;
    };
    
    bool bad = false;
    const char* reqs[] = {"NT","N0","BoxW","BoxH","ModeMin","ModeMax","FrameMin","FrameMax"};
    for (const char* r : reqs) {
        if (missing(r)) { std::cerr << "Missing required option: --" << r << std::endl; bad = true; }
    }
    if (bad) { print_usage(argv[0]); std::exit(1); }
    
    // Sanity: N0 cannot exceed NT
    if (N_PARTICLES_TYPE0 > N_PARTICLES_TOTAL) {
        std::cerr << "Error: --N0 (" << N_PARTICLES_TYPE0 << ") cannot exceed --NT ("
                  << N_PARTICLES_TOTAL << ").\n";
        std::exit(1);
    }


    const std::string run_name = "analysis";

    MDSimulation sim(N_PARTICLES_TOTAL, N_PARTICLES_TYPE0, // N_PARTICLES_TOTAL, N_PARTICLES_TYPE0,
            BOX_WIDTH, BOX_HEIGHT, // BOX_WIDTH, BOX_HEIGHT,
            T_init, T_target, //T_init, T_target,
            SIGMA_AA, SIGMA_BB, SIGMA_AB, //SIGMA_AA, SIGMA_BB, SIGMA_AB,
            EPSILON_AA, EPSILON_BB, EPSILON_AB, //EPSILON_AA, EPSILON_BB, EPSILON_AB,
            devide_p, // devide_p
            dt, //dt
            run_name); //run_name
    
    const std::string savepath = "./data/" + base_dir + "/run.bin.gz";
    const std::string outputpath = "./data/" + base_dir +"/plot_tmp/frame.png";
    const std::string spectrum_path = "./data/"+ base_dir+ "/plot_tmp/spectrum_single.png";
    const std::string csv_path = "./data/"+ base_dir+ "/plot_tmp/spectrum.csv";
    const std::string saved_datapath = "./data/"+ base_dir+ "/run.bin.gz";

    // int frame_idx = 1;
    for (int frame_idx = 100; frame_idx < 110; frame_idx ++){
        sim.load(frame_idx, saved_datapath);
        double gamma = sim.capillary_wave(mode_min, mode_max, spectrum_path);
        std::cout << "[Frame] "<<frame_idx<<" Measured gamma: " << gamma << std::endl;
    }
    // sim.load(frame_idx, saved_datapath);
    // // double gamma = sim.statistical_capillary_wave(mode_min, mode_max,
    // //                               spectrum_path,
    // //                               frame_min, frame_max,
    // //                               csv_path, saved_datapath);
    // double gamma = sim.capillary_wave(mode_min, mode_max, spectrum_path);

    

    return 0;
}