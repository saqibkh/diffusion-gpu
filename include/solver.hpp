#pragma once

#include <cstddef>
#include <vector>

namespace diffusion {

/// A 2D scalar field on a uniform grid, stored row-major.
class Field {
public:
    Field(std::size_t nx, std::size_t ny);

    std::size_t nx() const { return nx_; }
    std::size_t ny() const { return ny_; }

    double& at(std::size_t i, std::size_t j) { return data_[i * ny_ + j]; }
    double at(std::size_t i, std::size_t j) const { return data_[i * ny_ + j]; }

    const std::vector<double>& data() const { return data_; }
    std::vector<double>& data() { return data_; }

    /// Total of all cell values (conserved by pure diffusion with no-flux walls).
    double sum() const;

private:
    std::size_t nx_;
    std::size_t ny_;
    std::vector<double> data_;
};

/// Parameters for a diffusion simulation.
struct SimulationParams {
    double diffusivity = 1.0;  ///< D, the diffusion coefficient
    double dx = 1.0;           ///< grid spacing (assumed equal in x and y)
    double dt = 0.1;           ///< time step
};

/// Returns the largest stable dt for the explicit scheme (CFL condition).
double max_stable_dt(double diffusivity, double dx);

/// Advance the field one time step using the five-point explicit stencil.
/// Boundaries use no-flux (Neumann) conditions.
void step(const Field& current, Field& next, const SimulationParams& params);

/// Advance the field `n_steps` times, swapping buffers internally.
void run(Field& field, const SimulationParams& params, int n_steps);

}  // namespace diffusion