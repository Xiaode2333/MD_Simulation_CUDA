#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/solver/lanczos.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <vector>

int main()
{
	using index_t = int32_t;
	using value_t = float;

	constexpr index_t n_rows       = 10'000;
	constexpr index_t nnz          = n_rows;  // Diagonal matrix
	constexpr int eigen_components = 100;
	constexpr int max_iterations   = 400;
	constexpr value_t tolerance    = 1e-6f;

	raft::resources handle;
	auto stream = raft::resource::get_cuda_stream(handle);

	rmm::device_uvector<index_t> row_offsets(n_rows + 1, stream);
	rmm::device_uvector<index_t> col_indices(nnz, stream);
	rmm::device_uvector<value_t> values(nnz, stream);

	// Build a diagonally dominant positive-definite matrix: A = diag(10 + small variation)
	auto exec = thrust::cuda::par.on(stream.value());
	thrust::sequence(exec,
									 thrust::device_pointer_cast(row_offsets.data()),
									 thrust::device_pointer_cast(row_offsets.data() + row_offsets.size()),
									 0);
	thrust::sequence(exec,
									 thrust::device_pointer_cast(col_indices.data()),
									 thrust::device_pointer_cast(col_indices.data() + col_indices.size()),
									 0);
	thrust::transform(
		exec,
		thrust::make_counting_iterator<index_t>(0),
		thrust::make_counting_iterator<index_t>(nnz),
		thrust::device_pointer_cast(values.data()),
		[] __device__(index_t idx) { return 10.0f + 1e-3f * static_cast<value_t>(idx % 256); });

	auto rows_view = raft::make_device_vector_view<index_t, uint32_t>(
		row_offsets.data(), static_cast<uint32_t>(row_offsets.size()));
	auto cols_view = raft::make_device_vector_view<index_t, uint32_t>(
		col_indices.data(), static_cast<uint32_t>(col_indices.size()));
	auto vals_view = raft::make_device_vector_view<value_t, uint32_t>(
		values.data(), static_cast<uint32_t>(values.size()));

	rmm::device_uvector<value_t> eigenvalues(eigen_components, stream);
	rmm::device_uvector<value_t> eigenvectors(static_cast<size_t>(n_rows) * eigen_components, stream);

	auto eigenvalues_view =
		raft::make_device_vector_view<value_t, uint32_t, raft::layout_f_contiguous>(
			eigenvalues.data(), static_cast<uint32_t>(eigenvalues.size()));
	auto eigenvectors_view =
		raft::make_device_matrix_view<value_t, uint32_t, raft::layout_f_contiguous>(
			eigenvectors.data(), static_cast<uint32_t>(n_rows), static_cast<uint32_t>(eigen_components));

	int ncv = std::max(eigen_components * 2, 256);
	if (ncv >= n_rows) { ncv = n_rows - 1; }

	raft::sparse::solver::lanczos_solver_config<value_t> config{
		eigen_components,
		max_iterations,
		ncv,
		tolerance,
		raft::sparse::solver::LANCZOS_WHICH::SA,
		12345ULL};

	int status = raft::sparse::solver::lanczos_compute_smallest_eigenvectors<index_t, value_t>(
		handle, config, rows_view, cols_view, vals_view, std::nullopt, eigenvalues_view, eigenvectors_view);
	RAFT_EXPECTS(status == 0, "Lanczos solver failed with status %d", status);

	RAFT_CUDA_TRY(cudaStreamSynchronize(stream.value()));

	std::vector<value_t> host_eigenvalues(static_cast<size_t>(std::min(eigen_components, 10)));
	RAFT_CUDA_TRY(cudaMemcpy(host_eigenvalues.data(),
													 eigenvalues.data(),
													 host_eigenvalues.size() * sizeof(value_t),
													 cudaMemcpyDeviceToHost));

	std::cout << "Smallest eigenvalues (first " << host_eigenvalues.size() << "): ";
	for (auto v : host_eigenvalues) {
		std::cout << v << " ";
	}
	std::cout << "\n";

	return 0;
}
