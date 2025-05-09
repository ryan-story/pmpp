#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <cuda_runtime.h>

// Define a fixed seed for reproducibility
#define RANDOM_SEED 42

// Function to generate random atoms
void generate_atoms(float* atoms, int numatoms, float max_x, float max_y, float max_z);

// Function to compare two energy grids
int grids_allclose(const float* grid1, const float* grid2, dim3 grid_dimensions, 
                  float rtol, float atol, int verbose);

#endif // BENCHMARK_H