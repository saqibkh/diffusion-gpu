#include "solver.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace diffusion;

namespace {

void write_frame(const Field& f, int index) {
    std::ostringstream name;
    name << "frames/frame_" << std::setw(4) << std::setfill('0') << index
         << ".csv";
    std::ofstream out(name.str());
    for (std::size_t i = 0; i < f.nx(); ++i) {
        for (std::size_t j = 0; j < f.ny(); ++j) {
            out << f.at(i, j);
            if (j + 1 < f.ny()) out << ",";
        }
        out << "\n";
    }
}

}  // namespace

int main() {
    // Channel: nx along the flow, ny across it.
    const std::size_t nx = 400;
    const std::size_t ny = 120;

    SimulationParams p;
    p.diffusivity = 0.6;
    p.dx = 1.0;
    p.u = 1.0;                     // flow along +x (down the channel)
    p.dt = max_stable_dt_advective(p) * 0.4;

    Field f(nx, ny);

    // Inlet: left half of the channel width at high concentration.
    auto impose_inlet = [ny](Field& field) {
        for (std::size_t j = 0; j < ny; ++j) {
            field.at(0, j) = (j < ny / 2) ? 1.0 : 0.0;
        }
    };
    impose_inlet(f);

    std::filesystem::create_directory("frames");

    const int steps = 3000;
    const int frame_every = 30;
    int frame = 0;

    run_with_hook(f, p, steps, [&](Field& field, int s) {
        impose_inlet(field);
        if (s % frame_every == 0) write_frame(field, frame++);
    });

    std::cout << "Wrote " << frame << " frames to frames/\n";

    // Outlet profile: the gradient across the channel at the far end.
    std::ofstream out("outlet_profile.csv");
    out << "j,concentration\n";
    for (std::size_t j = 0; j < ny; ++j) {
        out << j << "," << f.at(nx - 1, j) << "\n";
    }
    std::cout << "Wrote outlet_profile.csv\n";
    return 0;
}