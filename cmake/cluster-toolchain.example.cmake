# Example site-specific toolchain configuration.
#
# Copy this file outside the repository, replace the placeholder paths, and
# configure with:
#   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=/path/to/cluster-toolchain.cmake

set(CMAKE_C_COMPILER "/path/to/gcc" CACHE FILEPATH "")
set(CMAKE_CXX_COMPILER "/path/to/g++" CACHE FILEPATH "")
set(CMAKE_CUDA_COMPILER "/path/to/nvcc" CACHE FILEPATH "")
set(CMAKE_CUDA_HOST_COMPILER "/path/to/g++" CACHE FILEPATH "")
set(MPI_C_COMPILER "/path/to/mpicc" CACHE FILEPATH "")
set(MPI_CXX_COMPILER "/path/to/mpicxx" CACHE FILEPATH "")
set(CUDAToolkit_ROOT "/path/to/cuda" CACHE PATH "")
set(Python3_EXECUTABLE "/path/to/python" CACHE FILEPATH "")
