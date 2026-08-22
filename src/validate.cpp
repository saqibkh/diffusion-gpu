#include "solver.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace diffusion;

int main() {
    const double D = 1.0;
    const double mass = 1.0;
    const double t_final = 2.0;
    const double domain = 40.0;  // physical width, held fixed

    std::ofstream out("validation.csv");
    out << "n,dx,l2_error\n";
    std::cout << std::scientific << std::setprecision(4);

    for (std::size_t n : {51u, 101u, 201u, 401u}) {
        SimulationParams p;
        p.diffusivity = D;
        p.dx = domain / (n-1);
        p.dt = max_stable_dt(D, p.dx)*0.25;

        // Start from the analytical profile at t0 rather than a delta spike,
        // which the grid cannot represent.
        const double t0 = 0.5;
        Field f(n, n);
        const double cx = (n-1)/2.0;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                const double x = (i - cx)*p.dx;
                const double y = (j - cx)*p.dx;
                const double r = std::sqrt(x*x + y*y);
                f.at(i, j) = gaussian_solution(r, t0, mass, D)*p.dx*p.dx;
            }
        }
        const int steps = static_cast<int>((t_final-t0)/p.dt);
        run(f, p, steps);
        
        const double t = t0 + steps*p.dt;
        const double err = l2_error(f, t, mass, p);

        std::cout << "n=" << n << "  dx=" << p.dx << "  L2=" << err << "\n";
        out << n << "," << p.dx << "," << err << "\n";
    }

    std::cout << "Wrote validation.csv\n";
    return 0;
}
