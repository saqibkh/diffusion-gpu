#include "solver.hpp"

#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    using namespace diffusion;

    std::size_t n = 128;
    int steps = 500;
    if (argc > 1) n = std::stoul(argv[1]);
    if (argc > 2) steps = std::stoi(argv[2]);

    Field field(n, n);
    // Initial condition: a point source at the centre.
    field.at(n / 2, n / 2) = 1000.0;

    SimulationParams params;
    params.dt = max_stable_dt(params.diffusivity, params.dx) * 0.5;

    std::cout << "Grid " << n << "x" << n << ", " << steps << " steps, dt="
              << params.dt << "\n";
    std::cout << "Initial mass: " << field.sum() << "\n";

    run(field, params, steps);

    std::cout << "Final mass:   " << field.sum() << "\n";

    // Dump the final field for plotting.
    std::ofstream out("field.csv");
    for (std::size_t i = 0; i < field.nx(); ++i) {
        for (std::size_t j = 0; j < field.ny(); ++j) {
            out << field.at(i, j);
            if (j + 1 < field.ny()) out << ",";
        }
        out << "\n";
    }
    std::cout << "Wrote field.csv\n";
    return 0;
}