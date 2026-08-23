#include "solver_cuda.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace diffusion {

namespace {

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " +
                                 cudaGetErrorString(err));
    }
}

__global__ void diffusion_kernel(const double* current, double* next,
                                 int nx, int ny, double alpha) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx || j >= ny) return;

    const int idx = i * ny + j;
    const double centre = current[idx];

    const double up    = (i == 0)      ? centre : current[(i - 1) * ny + j];
    const double down  = (i == nx - 1) ? centre : current[(i + 1) * ny + j];
    const double left  = (j == 0)      ? centre : current[idx - 1];
    const double right = (j == ny - 1) ? centre : current[idx + 1];

    next[idx] = centre + alpha * (up + down + left + right - 4.0 * centre);
}

}  // namespace

void run_cuda(Field& field, const SimulationParams& params, int n_steps) {
    const int nx = static_cast<int>(field.nx());
    const int ny = static_cast<int>(field.ny());
    const std::size_t bytes = field.data().size() * sizeof(double);
    const double alpha =
        params.diffusivity * params.dt / (params.dx * params.dx);

    double* d_current = nullptr;
    double* d_next = nullptr;
    cuda_check(cudaMalloc(&d_current, bytes), "cudaMalloc current");
    cuda_check(cudaMalloc(&d_next, bytes), "cudaMalloc next");

    cuda_check(cudaMemcpy(d_current, field.data().data(), bytes,
                          cudaMemcpyHostToDevice), "copy to device");

    const dim3 block(16, 16);
    const dim3 grid((ny + block.x - 1) / block.x,
                    (nx + block.y - 1) / block.y);

    for (int s = 0; s < n_steps; ++s) {
        diffusion_kernel<<<grid, block>>>(d_current, d_next, nx, ny, alpha);
        cuda_check(cudaGetLastError(), "kernel launch");
        std::swap(d_current, d_next);
    }
    cuda_check(cudaDeviceSynchronize(), "sync");

    cuda_check(cudaMemcpy(field.data().data(), d_current, bytes,
                          cudaMemcpyDeviceToHost), "copy to host");

    cudaFree(d_current);
    cudaFree(d_next);
}

}  // namespace diffusion