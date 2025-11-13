//mpi_cuda_test.cu
#include <mpi.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr,                                              \
                         "CUDA error %s:%d: %s\n",                            \
                         __FILE__, __LINE__, cudaGetErrorString(err__));      \
            std::abort();                                                     \
        }                                                                     \
    } while (0)

__global__ void simple_kernel(int* data, int rank) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *data = rank;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    char hostname[256];
    hostname[0] = '\0';
    gethostname(hostname, sizeof(hostname));

    const char* cvd = std::getenv("CUDA_VISIBLE_DEVICES");
    if (!cvd) {
        cvd = "CUDA_VISIBLE_DEVICES not set";
    }

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count < 1) {
        std::fprintf(stderr,
                     "Rank %d sees no GPU (device_count=%d). Check Slurm GPU options.\n",
                     world_rank, device_count);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int dev_id = 0;
    CUDA_CHECK(cudaSetDevice(dev_id));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev_id));

    int* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(int)));

    simple_kernel<<<1, 1>>>(d_data, world_rank);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_data = -1;
    CUDA_CHECK(cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost));

    std::printf("Rank %d / %d on host %s\n", world_rank, world_size, hostname);
    std::printf("  CUDA_VISIBLE_DEVICES = %s\n", cvd);
    std::printf("  device_count = %d, using device %d: %s\n",
                device_count, dev_id, prop.name);
    std::printf("  kernel wrote value = %d\n", h_data);
    std::fflush(stdout);

    CUDA_CHECK(cudaFree(d_data));

    MPI_Finalize();
    return 0;
}
// END OF ADD
