#pragma once

#include "solver.hpp"

namespace diffusion {

/// GPU implementation of `run`, Same semantics, same results.
void run_cuda(Field& field, const SimulationParams& params, int n_steps);

void run_cuda_tiled(Field& field, const SimulationParams& params, int n_steps);
}  // namespace diffusion