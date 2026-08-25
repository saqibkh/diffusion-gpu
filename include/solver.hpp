#pragma once

#include <cstddef>
#include <functional>
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
    double u = 0.0;            ///< advection velocity, x
    double v = 0.0;            ///< advection velocity, y
};

/// Returns the largest stable dt for the explicit scheme (CFL condition).
double max_stable_dt(double diffusivity, double dx);

/// Largest stable dt accounting for both diffusion and advection.
double max_stable_dt_advective(const SimulationParams& params);

/// Advance the field one time step using the five-point explicit stencil.
/// Boundaries use no-flux (Neumann) conditions.
void step(const Field& current, Field& next, const SimulationParams& params);

/// Advance the field `n_steps` times, swapping buffers internally.
void run(Field& field, const SimulationParams& params, int n_steps);

/// Exact solution for a point source of mass M at the domain centre,
/// diffusing in an unbounded 2D domain.
double gaussian_solution(double r, double t, double mass, double diffusivity);


/// L2 norm of the difference between a field and the analytical solution.
double l2_error(const Field& field, double t, double mass, const SimulationParams& params);

/// Run with a callback applied after each step (e.g. to hold inlet values).
void run_with_hook(Field& field, const SimulationParams& params, int n_steps, const std::function<void(Field&, int)>& hook);

}  // namespace diffusion

