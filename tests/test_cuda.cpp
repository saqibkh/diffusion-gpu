#include "solver.hpp"
#include "solver_cuda.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

int main() {
    using namespace diffusion;

    bool ok = true;

    // Pure diffusion.
    {
        SimulationParams p;
        p.dt = max_stable_dt(p.diffusivity, p.dx) * 0.5;

        Field cpu(256, 256);
        Field gpu(256, 256);
        cpu.at(128, 128) = 1000.0;
        gpu.at(128, 128) = 1000.0;

        run(cpu, p, 50);
        run_cuda(gpu, p, 50);

        double worst = 0.0;
        for (std::size_t k = 0; k < cpu.data().size(); ++k) {
            worst = std::max(worst, std::fabs(cpu.data()[k] - gpu.data()[k]));
        }

        const bool pass = worst < 1e-12;
        ok = ok && pass;
        std::cout << (pass ? "pass" : "FAIL")
                  << ": GPU matches CPU (max diff " << worst << ")\n";
    }

    // Advection in both directions, exercising both upwind branches.
    {
        SimulationParams ap;
        ap.diffusivity = 0.5;
        ap.u = 0.3;
        ap.v = -0.2;
        ap.dt = max_stable_dt_advective(ap) * 0.5;

        Field cpu(128, 128);
        Field gpu(128, 128);
        cpu.at(64, 64) = 1000.0;
        gpu.at(64, 64) = 1000.0;

        run(cpu, ap, 40);
        run_cuda(gpu, ap, 40);

        double worst = 0.0;
        for (std::size_t k = 0; k < cpu.data().size(); ++k) {
            worst = std::max(worst, std::fabs(cpu.data()[k] - gpu.data()[k]));
        }

        const bool pass = worst < 1e-12;
        ok = ok && pass;
        std::cout << (pass ? "pass" : "FAIL")
                  << ": GPU matches CPU with advection (max diff "
                  << worst << ")\n";
    }

    // Tiled kernel: halo loading must reproduce the same result.
    {
        SimulationParams p;
        p.dt = max_stable_dt(p.diffusivity, p.dx) * 0.5;

        Field cpu(256, 256);
        Field gpu(256, 256);
        cpu.at(128, 128) = 1000.0;
        gpu.at(128, 128) = 1000.0;

        run(cpu, p, 50);
        run_cuda_tiled(gpu, p, 50);

        double worst = 0.0;
        for (std::size_t k = 0; k < cpu.data().size(); ++k) {
            worst = std::max(worst, std::fabs(cpu.data()[k] - gpu.data()[k]));
        }

        const bool pass = worst < 1e-12;
        ok = ok && pass;
        std::cout << (pass ? "pass" : "FAIL")
                  << ": tiled GPU matches CPU (max diff " << worst << ")\n";
    }

    // Tiled kernel with advection, on a grid that is not a tile multiple.
    {
        SimulationParams ap;
        ap.diffusivity = 0.5;
        ap.u = 0.3;
        ap.v = -0.2;
        ap.dt = max_stable_dt_advective(ap) * 0.5;

        Field cpu(100, 150);
        Field gpu(100, 150);
        cpu.at(50, 75) = 1000.0;
        gpu.at(50, 75) = 1000.0;

        run(cpu, ap, 40);
        run_cuda_tiled(gpu, ap, 40);

        double worst = 0.0;
        for (std::size_t k = 0; k < cpu.data().size(); ++k) {
            worst = std::max(worst, std::fabs(cpu.data()[k] - gpu.data()[k]));
        }

        const bool pass = worst < 1e-12;
        ok = ok && pass;
        std::cout << (pass ? "pass" : "FAIL")
                  << ": tiled GPU matches CPU, ragged grid (max diff "
                  << worst << ")\n";
    }

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}