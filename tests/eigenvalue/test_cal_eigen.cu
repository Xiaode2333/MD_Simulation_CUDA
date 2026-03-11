#include "md_cuda_common.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdio>
#include <cmath>
#include <type_traits>

template <typename T>
bool run_case()
{
    using index_t = int32_t;

    constexpr int n_rows = 128;
    constexpr int nnz    = n_rows;  // diagonal
    constexpr int k      = 4;       // smallest eigenvalues

    // CSR for diagonal matrix diag(1, 2, 5, 10)
    thrust::host_vector<index_t> h_row_offsets(n_rows + 1);
    thrust::host_vector<index_t> h_col_indices(nnz);
    thrust::host_vector<T>       h_values(nnz);

    for (int i = 0; i <= n_rows; ++i) {
        h_row_offsets[i] = static_cast<index_t>(i);
        if (i < n_rows) {
            h_col_indices[i] = static_cast<index_t>(i);
            h_values[i]      = static_cast<T>(1 + i);  // strictly increasing spectrum
        }
    }

    thrust::device_vector<index_t> d_row_offsets = h_row_offsets;
    thrust::device_vector<index_t> d_col_indices = h_col_indices;
    thrust::device_vector<T>       d_values      = h_values;

    thrust::device_vector<T> d_eigvals(k);
    thrust::device_vector<T> d_eigvecs(static_cast<size_t>(n_rows) * k);

    int ncv = std::max(k * 2 + 1, 32);
    if (ncv >= n_rows) { ncv = n_rows - 1; }
    int max_iterations = 200;
    T   tol            = std::is_same<T, float>::value ? static_cast<T>(1e-3) : static_cast<T>(1e-8);
    uint64_t seed      = 12345ULL;

    cal_eigen<index_t, T>(
        n_rows, nnz,
        thrust::raw_pointer_cast(d_row_offsets.data()),
        thrust::raw_pointer_cast(d_col_indices.data()),
        thrust::raw_pointer_cast(d_values.data()),
        k, max_iterations, ncv, tol, seed,
        thrust::raw_pointer_cast(d_eigvals.data()),
        thrust::raw_pointer_cast(d_eigvecs.data()));

    thrust::host_vector<T> h_eigvals = d_eigvals;
    thrust::host_vector<T> h_eigvecs = d_eigvecs;

    const T tol_val = tol * static_cast<T>(10);
    bool ok = true;
    ok &= std::fabs(h_eigvals[0] - static_cast<T>(1)) < tol_val;
    ok &= std::fabs(h_eigvals[1] - static_cast<T>(2)) < tol_val;
    ok &= std::fabs(h_eigvals[2] - static_cast<T>(3)) < tol_val;
    ok &= std::fabs(h_eigvals[3] - static_cast<T>(4)) < tol_val;

    auto col_ok = [&](int col, int expected_row) {
        for (int r = 0; r < n_rows; ++r) {
            T v = h_eigvecs[static_cast<size_t>(r) + static_cast<size_t>(col) * n_rows];
            if (r == expected_row) {
                if (std::fabs(std::fabs(v) - static_cast<T>(1)) > tol_val) return false;
            } else {
                if (std::fabs(v) > tol_val) return false;
            }
        }
        return true;
    };

    ok &= col_ok(0, 0);
    ok &= col_ok(1, 1);
    ok &= col_ok(2, 2);
    ok &= col_ok(3, 3);
    return ok;
}

int main()
{
    bool ok_float  = run_case<float>();
    bool ok_double = run_case<double>();

    if (ok_float && ok_double) {
        std::printf("cal_eigen tests passed (float & double)\n");
        return 0;
    }
    std::fprintf(stderr, "cal_eigen tests FAILED (float=%d, double=%d)\n",
                 ok_float ? 1 : 0, ok_double ? 1 : 0);
    return 1;
}
