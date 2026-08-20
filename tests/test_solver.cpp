#include "solver.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

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

    // A uniform field should stay uniform (the Laplacian is zero everywhere).
    {
        Field f(16, 16);
        for (std::size_t i = 0; i < 16; ++i)
            for (std::size_t j = 0; j < 16; ++j) f.at(i, j) = 5.0;

        SimulationParams p;
        p.dt = max_stable_dt(p.diffusivity, p.dx) * 0.5;
        run(f, p, 10);

        check_near(f.at(8, 8), 5.0, 1e-9, "uniform field stays uniform");
    }

    // Mass is conserved under no-flux boundaries.
    {
        Field f(32, 32);
        f.at(16, 16) = 100.0;
        const double initial = f.sum();

        SimulationParams p;
        p.dt = max_stable_dt(p.diffusivity, p.dx) * 0.5;
        run(f, p, 50);

        check_near(f.sum(), initial, 1e-6, "mass is conserved");
    }

    // A point source spreads outward: neighbours gain, centre loses.
    {
        Field f(32, 32);
        f.at(16, 16) = 100.0;

        SimulationParams p;
        p.dt = max_stable_dt(p.diffusivity, p.dx) * 0.5;
        run(f, p, 1);

        check(f.at(16, 16) < 100.0, "centre loses concentration");
        check(f.at(17, 16) > 0.0, "neighbour gains concentration");
    }

    // The stability limit is what the theory says.
    {
        check_near(max_stable_dt(1.0, 1.0), 0.25, 1e-12, "CFL limit for D=1, dx=1");
        check_near(max_stable_dt(2.0, 1.0), 0.125, 1e-12, "CFL limit scales with D");
    }

    // Grids that are too small are rejected.
    {
        bool threw = false;
        try {
            Field f(2, 2);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "tiny grids are rejected");
    }

    std::cout << (failures == 0 ? "\nAll tests passed.\n" : "\nFAILURES.\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}