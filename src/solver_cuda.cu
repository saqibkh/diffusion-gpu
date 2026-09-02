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
                                 int nx, int ny, double alpha,
                                 double u, double v, double dt_over_dx) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx || j >= ny) return;

    const int idx = i * ny + j;
    const double centre = current[idx];

    const double up    = (i == 0)      ? centre : current[(i - 1) * ny + j];
    const double down  = (i == nx - 1) ? centre : current[(i + 1) * ny + j];
    const double left  = (j == 0)      ? centre : current[idx - 1];
    const double right = (j == ny - 1) ? centre : current[idx + 1];

    double value = centre + alpha * (up + down + left + right - 4.0 * centre);

    if (u != 0.0) {
        const double grad_x = (u > 0.0) ? (centre - up) : (down - centre);
        value -= u * dt_over_dx * grad_x;
    }
    if (v != 0.0) {
        const double grad_y = (v > 0.0) ? (centre - left) : (right - centre);
        value -= v * dt_over_dx * grad_y;
    }

    next[idx] = value;
}

constexpr int kTile = 16;

__global__ void diffusion_kernel_tiled(const double* current, double* next,
                                       int nx, int ny, double alpha,
                                       double u, double v, double dt_over_dx) {
    __shared__ double tile[kTile + 2][kTile + 2];

    const int tj = threadIdx.x;
    const int ti = threadIdx.y;
    const int j = blockIdx.x * kTile + tj;
    const int i = blockIdx.y * kTile + ti;

    // Shared-memory coordinates are offset by the halo.
    const int si = ti + 1;
    const int sj = tj + 1;

    auto load = [&](int gi, int gj) -> double {
        gi = max(0, min(gi, nx - 1));
        gj = max(0, min(gj, ny - 1));
        return current[gi * ny + gj];
    };

    if (i < nx && j < ny) {
        tile[si][sj] = current[i * ny + j];
    } else {
        tile[si][sj] = 0.0;
    }

    // Halo cells: threads on the tile edges load the surrounding ring.
    if (ti == 0)          tile[0][sj]         = load(i - 1, j);
    if (ti == kTile - 1)  tile[kTile + 1][sj] = load(i + 1, j);
    if (tj == 0)          tile[si][0]         = load(i, j - 1);
    if (tj == kTile - 1)  tile[si][kTile + 1] = load(i, j + 1);

    __syncthreads();

    if (i >= nx || j >= ny) return;

    const double centre = tile[si][sj];
    const double up    = (i == 0)      ? centre : tile[si - 1][sj];
    const double down  = (i == nx - 1) ? centre : tile[si + 1][sj];
    const double left  = (j == 0)      ? centre : tile[si][sj - 1];
    const double right = (j == ny - 1) ? centre : tile[si][sj + 1];

    double value = centre + alpha * (up + down + left + right - 4.0 * centre);

    if (u != 0.0) {
        const double grad_x = (u > 0.0) ? (centre - up) : (down - centre);
        value -= u * dt_over_dx * grad_x;
    }
    if (v != 0.0) {
        const double grad_y = (v > 0.0) ? (centre - left) : (right - centre);
        value -= v * dt_over_dx * grad_y;
    }

    next[i * ny + j] = value;
}

}  // namespace

void run_cuda(Field& field, const SimulationParams& params, int n_steps) {
    const int nx = static_cast<int>(field.nx());
    const int ny = static_cast<int>(field.ny());
    const std::size_t bytes = field.data().size() * sizeof(double);
    const double alpha =
        params.diffusivity * params.dt / (params.dx * params.dx);
    const double dt_over_dx = params.dt / params.dx;

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
        diffusion_kernel<<<grid, block>>>(d_current, d_next, nx, ny, alpha, params.u, params.v, dt_over_dx);
        cuda_check(cudaGetLastError(), "kernel launch");
        std::swap(d_current, d_next);
    }
    cuda_check(cudaDeviceSynchronize(), "sync");

    cuda_check(cudaMemcpy(field.data().data(), d_current, bytes,
                          cudaMemcpyDeviceToHost), "copy to host");

    cudaFree(d_current);
    cudaFree(d_next);
}

void run_cuda_tiled(Field& field, const SimulationParams& params, int n_steps) {
    const int nx = static_cast<int>(field.nx());
    const int ny = static_cast<int>(field.ny());
    const std::size_t bytes = field.data().size() * sizeof(double);
    const double alpha =
        params.diffusivity * params.dt / (params.dx * params.dx);
    const double dt_over_dx = params.dt / params.dx;

    double* d_current = nullptr;
    double* d_next = nullptr;
    cuda_check(cudaMalloc(&d_current, bytes), "cudaMalloc current");
    cuda_check(cudaMalloc(&d_next, bytes), "cudaMalloc next");
    cuda_check(cudaMemcpy(d_current, field.data().data(), bytes,
                          cudaMemcpyHostToDevice), "copy to device");

    const dim3 block(kTile, kTile);
    const dim3 grid((ny + kTile - 1) / kTile, (nx + kTile - 1) / kTile);

    for (int s = 0; s < n_steps; ++s) {
        diffusion_kernel_tiled<<<grid, block>>>(d_current, d_next, nx, ny,
                                                alpha, params.u, params.v,
                                                dt_over_dx);
        cuda_check(cudaGetLastError(), "tiled kernel launch");
        std::swap(d_current, d_next);
    }
    cuda_check(cudaDeviceSynchronize(), "sync");

    cuda_check(cudaMemcpy(field.data().data(), d_current, bytes,
                          cudaMemcpyDeviceToHost), "copy to host");

    cudaFree(d_current);
    cudaFree(d_next);
}

}  // namespace diffusion


