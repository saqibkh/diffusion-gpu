#include "solver.hpp"
#include "solver_cuda.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

int main() {
    using namespace diffusion;

    SimulationParams p;
    p.dt = max_stable_dt(p.diffusivity, p.dx)*0.5;

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

    const bool ok = worst < 1e-12;
    std::cout << (ok ? "pass" : "FAIL")
              << ": GPU matches CPU (max diff " << worst << ")\n";
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}