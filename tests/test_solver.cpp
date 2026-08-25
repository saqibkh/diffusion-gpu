#include "solver.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

constexpr double kPi = 3.14159265358979323846;

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        ++failures;
    } else {
        std::cout << "pass: " << message << "\n";
    }
}

void check_near(double a, double b, double tol, const char* message) {
    check(std::fabs(a - b) < tol, message);
}

}  // namespace

int main() {
    using namespace diffusion;

    // A uniform field has zero Laplacian everywhere, so nothing changes.
    {
        Field f(16, 16);
        for (std::size_t i = 0; i < 16; ++i)
            for (std::size_t j = 0; j < 16; ++j) f.at(i, j) = 5.0;

        SimulationParams p;
        p.dt = max_stable_dt(p.diffusivity, p.dx) * 0.5;
        run(f, p, 10);

        check_near(f.at(8, 8), 5.0, 1e-9, "uniform field stays uniform");
    }

    // No-flux walls mean total mass is invariant.
    {
        Field f(32, 32);
        f.at(16, 16) = 100.0;
        const double initial = f.sum();

        SimulationParams p;
        p.dt = max_stable_dt(p.diffusivity, p.dx) * 0.5;
        run(f, p, 50);

        check_near(f.sum(), initial, 1e-6, "mass is conserved");
    }

    {
        Field f(32, 32);
        f.at(16, 16) = 100.0;

        SimulationParams p;
        p.dt = max_stable_dt(p.diffusivity, p.dx) * 0.5;
        run(f, p, 1);

        check(f.at(16, 16) < 100.0, "centre loses concentration");
        check(f.at(17, 16) > 0.0, "neighbour gains concentration");
    }

    {
        check_near(max_stable_dt(1.0, 1.0), 0.25, 1e-12,
                   "CFL limit for D=1, dx=1");
        check_near(max_stable_dt(2.0, 1.0), 0.125, 1e-12,
                   "CFL limit scales with D");
    }

    {
        bool threw = false;
        try {
            Field f(2, 2);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "tiny grids are rejected");
    }

    {
        const double peak = gaussian_solution(0.0, 1.0, 1.0, 1.0);
        check_near(peak, 1.0 / (4.0 * kPi), 1e-9,
                   "gaussian peak matches formula");

        const double near_val = gaussian_solution(1.0, 1.0, 1.0, 1.0);
        const double far_val = gaussian_solution(5.0, 1.0, 1.0, 1.0);
        check(far_val < near_val, "gaussian decays with distance");
    }

    // Second-order scheme: a finer grid must reduce error against the
    // analytical solution.
    {
        auto err_at = [](std::size_t n) {
            SimulationParams p;
            p.dx = 40.0 / (n - 1);
            p.dt = max_stable_dt(p.diffusivity, p.dx) * 0.25;

            const double t0 = 0.5;
            const double centre = (n - 1) / 2.0;

            Field f(n, n);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    const double x = (i - centre) * p.dx;
                    const double y = (j - centre) * p.dx;
                    const double r = std::sqrt(x * x + y * y);
                    f.at(i, j) =
                        gaussian_solution(r, t0, 1.0, p.diffusivity)
                        * p.dx * p.dx;
                }
            }

            const int steps = static_cast<int>((2.0 - t0) / p.dt);
            run(f, p, steps);
            return l2_error(f, t0 + steps * p.dt, 1.0, p);
        };

        check(err_at(101) < err_at(51), "refining the grid reduces error");
    }

    // Pure advection transports without creating or destroying mass.
    {
        Field f(64, 64);
        for (std::size_t j = 20; j < 30; ++j) f.at(32, j) = 10.0;
        const double initial = f.sum();

        SimulationParams p;
        p.diffusivity = 0.0;
        p.v = 0.5;
        p.dt = max_stable_dt_advective(p) * 0.5;
        run(f, p, 20);

        check_near(f.sum(), initial, 1e-6, "advection conserves mass");
    }

    // Flow in +v carries material toward higher j.
    {
        Field f(64, 64);
        f.at(32, 20) = 100.0;

        SimulationParams p;
        p.diffusivity = 0.01;
        p.v = 0.5;
        p.dt = max_stable_dt_advective(p) * 0.5;
        run(f, p, 30);

        double upstream = 0.0, downstream = 0.0;
        for (std::size_t i = 0; i < 64; ++i)
            for (std::size_t j = 0; j < 64; ++j)
                (j < 20 ? upstream : downstream) += f.at(i, j);

        check(downstream > upstream, "advection transports downstream");
    }

    {
        SimulationParams p;
        p.u = 10.0;
        check(max_stable_dt_advective(p) < max_stable_dt(p.diffusivity, p.dx),
              "advection tightens the stability limit");
    }

    std::cout << (failures == 0 ? "\nAll tests passed.\n" : "\nFAILURES.\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}