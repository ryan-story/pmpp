#ifndef CENERGY_H
#define CENERGY_H

#include <cuda_runtime.h>

// Define constants
#define BLOCK_SIZE 1024
#define CHUNK_SIZE 1024
#define COARSEN_FACTOR 1

// Sequential implementation
void cenergySequential(float* energygrid, dim3 grid, float gridspacing, float z, const float* atoms, int numatoms);

// Optimized sequential implementation
void cenergySequentialOptimized(float* energygrid, dim3 grid, float gridspacing, float z, const float* atoms, int numatoms);

// GPU implementations
void cenergyParallelScatter(float* host_energygrid, dim3 grid, float gridspacing, float z, const float* host_atoms, int numatoms);
void cenergyParallelGather(float* host_energygrid, dim3 grid, float gridspacing, float z, const float* host_atoms, int numatoms);
void cenergyParallelCoarsen(float* host_energygrid, dim3 grid, float gridspacing, float z, const float* host_atoms, int numatoms);
void cenergyParallelCoalescing(float* host_energygrid, dim3 grid, float gridspacing, float z, const float* host_atoms, int numatoms);

#endif // CENERGY_H