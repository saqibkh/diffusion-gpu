#include "solver.hpp"
#include "solver_cuda.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace diffusion;
using Clock = std::chrono::steady_clock;

namespace {

Field make_initial(std::size_t n) {
    Field f(n, n);
    f.at(n / 2, n / 2) = 1000.0;
    return f;
}

double max_abs_diff(const Field& a, const Field& b) {
    double worst = 0.0;
    for (std::size_t k = 0; k < a.data().size(); ++k) {
        worst = std::max(worst, std::fabs(a.data()[k] - b.data()[k]));
    }
    return worst;
}

}  // namespace

int main() {
    const int steps = 200;

    // Absorb CUDA context initialisation so it isn't charged to the first
    // timed run.
    {
        Field warm(64, 64);
        SimulationParams wp;
        wp.dt = max_stable_dt(wp.diffusivity, wp.dx) * 0.5;
        run_cuda(warm, wp, 1);
    }

    std::ofstream out("benchmark.csv");
    out << "n,cpu_ms,gpu_ms,speedup,max_diff\n";

    std::cout << std::fixed << std::setprecision(2);

    for (std::size_t n : {128u, 256u, 512u, 1024u, 2048u}) {
        SimulationParams p;
        p.dt = max_stable_dt(p.diffusivity, p.dx) * 0.5;

        Field cpu_field = make_initial(n);
        auto t0 = Clock::now();
        run(cpu_field, p, steps);
        auto t1 = Clock::now();
        const double cpu_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        Field gpu_field = make_initial(n);
        t0 = Clock::now();
        run_cuda(gpu_field, p, steps);
        t1 = Clock::now();
        const double gpu_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        const double diff = max_abs_diff(cpu_field, gpu_field);

        std::cout << n << "x" << n
                  << "  CPU " << cpu_ms << " ms"
                  << "  GPU " << gpu_ms << " ms"
                  << "  speedup " << (cpu_ms / gpu_ms) << "x"
                  << "  max|diff| " << std::scientific << diff
                  << std::fixed << "\n";

        out << n << "," << cpu_ms << "," << gpu_ms << ","
            << (cpu_ms / gpu_ms) << "," << diff << "\n";
    }

    std::cout << "Wrote benchmark.csv\n";
    return 0;
}