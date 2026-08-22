#include "solver.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>


constexpr double kPi = 3.14159265358979323846;

namespace diffusion {

Field::Field(std::size_t nx, std::size_t ny)
    : nx_(nx), ny_(ny), data_(nx * ny, 0.0) {
    if (nx < 3 || ny < 3) {
        throw std::invalid_argument("Grid must be at least 3x3.");
    }
}

double Field::sum() const {
    return std::accumulate(data_.begin(), data_.end(), 0.0);
}

double max_stable_dt(double diffusivity, double dx) {
    // Explicit 2D scheme stability: dt <= dx^2 / (4D)
    return (dx * dx) / (4.0 * diffusivity);
}

void step(const Field& current, Field& next, const SimulationParams& params) {
    const std::size_t nx = current.nx();
    const std::size_t ny = current.ny();
    const double alpha =
        params.diffusivity * params.dt / (params.dx * params.dx);

    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            // Ghost cells reflect: out-of-domain neighbor = centre, so NO flux
            const double centre = current.at(i, j);
            const double up    = (i == 0)      ? centre : current.at(i - 1, j);
            const double down  = (i == nx - 1) ? centre : current.at(i + 1, j);
            const double left  = (j == 0)      ? centre : current.at(i, j - 1);
            const double right = (j == ny - 1) ? centre : current.at(i, j + 1);

            const double laplacian = up + down + left + right - 4.0 * centre;
            next.at(i, j) = centre + alpha * laplacian;
        }
    }
}

void run(Field& field, const SimulationParams& params, int n_steps) {
    Field buffer(field.nx(), field.ny());
    for (int s = 0; s < n_steps; ++s) {
        step(field, buffer, params);
        std::swap(field.data(), buffer.data());
    }
}


double gaussian_solution(double r, double t, double mass, double diffusivity) {
    const double denom = 4.0 * kPi * diffusivity * t;
    return (mass / denom) * std::exp(-(r*r)/(4.0*diffusivity*t));
}

double l2_error(const Field& field, double t, double mass, const SimulationParams& params) {
    const double cx = (field.nx() - 1)/2.0;
    const double cy = (field.ny() - 1)/2.0;

    double sum_sq = 0.0;
    for (std::size_t i = 0; i < field.nx(); ++i) {
        for (std::size_t j = 0; j < field.ny(); ++j) {
            const double dx = (i - cx)*params.dx;
            const double dy = (j - cy)*params.dx;
            const double r = std::sqrt(dx*dx + dy*dy);
            const double exact = gaussian_solution(r, t, mass, params.diffusivity);
            const double diff = field.at(i, j)/(params.dx*params.dx)-exact;
            sum_sq += diff*diff;
        }
    }
    return std::sqrt(sum_sq/(field.nx()*field.ny()));
}

}  // namespace diffusion