#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Define a fixed seed for reproducibility
#define RANDOM_SEED 42
#define BLOCK_SIZE 256
#define CHUNK_SIZE 256  // Maximum number of atoms to process in one chunk
#define COARSEN_FACTOR 4 // Thread coarsening factor

// CUDA kernel using constant memory
__constant__ float atoms[CHUNK_SIZE * 4];  // Each atom has x,y,z,charge

__global__ void cenergyScatterKernel(float* energygrid, dim3 grid, float gridspacing, float z, int atoms_in_chunk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = idx * 4;

    // Only process valid atoms in the chunk
    if (idx < atoms_in_chunk) {
        float dz = z - atoms[n + 2];  // all grid points in a slice have the same
        float dz2 = dz * dz;          // z value
        // starting position of the slice in the energy grid
        int grid_slice_offset = (grid.x * grid.y * z) / gridspacing;
        float charge = atoms[n + 3];
        for (int j = 0; j < grid.y; j++) {
            float y = gridspacing * (float)j;
            float dy = y - atoms[n + 1];  // all grid points in a row have the same
            float dy2 = dy * dy;          // y value
            // starting position of the row in the energy grid
            int grid_row_offset = grid_slice_offset + grid.x * j;
            for (int i = 0; i < grid.x; i++) {
                float x = gridspacing * (float)i;
                float dx = x - atoms[n];
                atomicAdd(&energygrid[grid_row_offset + i], charge / sqrtf(dx * dx + dy2 + dz2));
            }
        }
    }
}

// Chunked GPU implementation with scatter approach
void cenergyParallelScatter(float* host_energygrid, dim3 grid, float gridspacing, float z, const float* host_atoms,
                            int numatoms) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate device memory for the energy grid
    float* d_energygrid = NULL;
    size_t grid_size = grid.x * grid.y * grid.z * sizeof(float);
    err = cudaMalloc((void**)&d_energygrid, grid_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory for energy grid (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // Initialize energy grid to zero
    err = cudaMemset(d_energygrid, 0, grid_size);
    if (err != cudaSuccess) {
        printf("Failed to initialize device memory (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_energygrid);
        return;
    }

    // Process atoms in chunks
    int num_chunks = (numatoms + CHUNK_SIZE - 1) / CHUNK_SIZE;
    printf("Processing %d atoms in %d chunks of size %d\n", numatoms, num_chunks, CHUNK_SIZE);

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        // Calculate the number of atoms in this chunk
        int start_atom = chunk * CHUNK_SIZE;
        int atoms_in_chunk = (start_atom + CHUNK_SIZE <= numatoms) ? CHUNK_SIZE : (numatoms - start_atom);

        // Copy this chunk to constant memory
        size_t chunk_size = atoms_in_chunk * 4 * sizeof(float);
        err = cudaMemcpyToSymbol(atoms, &host_atoms[start_atom * 4], chunk_size);
        if (err != cudaSuccess) {
            printf("Failed to copy atoms chunk %d to constant memory (error code %s)!\n", chunk,
                   cudaGetErrorString(err));
            cudaFree(d_energygrid);
            return;
        }

        // Launch kernel for this chunk with the correct number of atoms
        int threadsPerBlock = BLOCK_SIZE;
        int blocksPerGrid = (atoms_in_chunk + threadsPerBlock - 1) / threadsPerBlock;

        // Pass atoms_in_chunk to the kernel to handle the non-full last chunk
        cenergyScatterKernel<<<blocksPerGrid, threadsPerBlock>>>(d_energygrid, grid, gridspacing, z, atoms_in_chunk);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to launch kernel for chunk %d (error code %s)!\n", chunk, cudaGetErrorString(err));
            cudaFree(d_energygrid);
            return;
        }

        // Wait for this chunk to complete
        cudaDeviceSynchronize();
    }

    // Copy the accumulated results back to host
    err = cudaMemcpy(host_energygrid, d_energygrid, grid_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Failed to copy energy grid from device to host (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_energygrid);
        return;
    }

    // Free device memory
    cudaFree(d_energygrid);
}

__global__ void cenergyGatherKernel(float* energygrid, dim3 grid_dim, float gridspacing, float z, int atoms_in_chunk,
                                    int chunk_start) {
    // Get 2D grid coordinates
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if this thread is within grid bounds
    if (i < grid_dim.x && j < grid_dim.y) {
        // Calculate actual grid point coordinates
        float x = gridspacing * (float)i;
        float y = gridspacing * (float)j;

        // Calculate z-index for the plane
        int k = (int)(z / gridspacing);

        // Grid point index in the energy grid
        int grid_point_idx = grid_dim.x * grid_dim.y * k + grid_dim.x * j + i;

        // For the first chunk, initialize the energy value
        if (chunk_start == 0) {
            energygrid[grid_point_idx] = 0.0f;
        }

        // Calculate energy contribution from all atoms in this chunk
        float energy = 0.0f;

        // Loop through all atoms in the current chunk
        for (int n = 0; n < atoms_in_chunk; n++) {
            int atom_idx = n * 4;  // Each atom has 4 values (x,y,z,charge)

            float dx = x - atoms[atom_idx];
            float dy = y - atoms[atom_idx + 1];
            float dz = z - atoms[atom_idx + 2];
            float charge = atoms[atom_idx + 3];

            energy += charge / sqrtf(dx * dx + dy * dy + dz * dz);
        }

        // Add this chunk's contribution to the energy grid (no atomic needed)
        energygrid[grid_point_idx] += energy;
    }
}

// Chunked GPU implementation with gather approach
void cenergyParallelGather(float* host_energygrid, dim3 grid_dim, float gridspacing, float z, const float* host_atoms,
                           int numatoms) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate device memory for the energy grid
    float* d_energygrid = NULL;
    size_t grid_size = grid_dim.x * grid_dim.y * grid_dim.z * sizeof(float);
    err = cudaMalloc((void**)&d_energygrid, grid_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory for energy grid (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // Initialize energy grid to zero
    err = cudaMemset(d_energygrid, 0, grid_size);
    if (err != cudaSuccess) {
        printf("Failed to initialize device memory (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_energygrid);
        return;
    }

    // Process atoms in chunks
    int num_chunks = (numatoms + CHUNK_SIZE - 1) / CHUNK_SIZE;
    printf("Processing %d atoms in %d chunks of size %d\n", numatoms, num_chunks, CHUNK_SIZE);

    // Define 2D thread block and grid
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
    dim3 blocksPerGrid((grid_dim.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (grid_dim.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("Gather: Launching kernel with grid: %d x %d blocks of %d x %d threads\n", blocksPerGrid.x, blocksPerGrid.y,
           threadsPerBlock.x, threadsPerBlock.y);

    // For each chunk of atoms
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        // Calculate the number of atoms in this chunk
        int start_atom = chunk * CHUNK_SIZE;
        int atoms_in_chunk = (start_atom + CHUNK_SIZE <= numatoms) ? CHUNK_SIZE : (numatoms - start_atom);

        // Copy this chunk to constant memory
        size_t chunk_size = atoms_in_chunk * 4 * sizeof(float);
        err = cudaMemcpyToSymbol(atoms, &host_atoms[start_atom * 4], chunk_size);
        if (err != cudaSuccess) {
            printf("Failed to copy atoms chunk %d to constant memory (error code %s)!\n", chunk,
                   cudaGetErrorString(err));
            cudaFree(d_energygrid);
            return;
        }

        // Launch kernel for this chunk
        cenergyGatherKernel<<<blocksPerGrid, threadsPerBlock>>>(d_energygrid, grid_dim, gridspacing, z, atoms_in_chunk,
                                                                start_atom);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to launch kernel for chunk %d (error code %s)!\n", chunk, cudaGetErrorString(err));
            cudaFree(d_energygrid);
            return;
        }

        // Wait for this chunk to complete
        cudaDeviceSynchronize();
    }

    // Copy the accumulated results back to host
    err = cudaMemcpy(host_energygrid, d_energygrid, grid_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Failed to copy energy grid from device to host (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_energygrid);
        return;
    }

    // Free device memory
    cudaFree(d_energygrid);
}

// Thread coarsening kernel - each thread processes COARSEN_FACTOR grid points
__global__ void cenergyCoarsenKernel(float* energygrid, dim3 grid, float gridspacing, float z, int atoms_in_chunk) {
    int base_i = blockIdx.x * blockDim.x * COARSEN_FACTOR + threadIdx.x * COARSEN_FACTOR;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (j >= grid.y) return;
    
    int k = z / gridspacing;
    float y = gridspacing * (float)j;
    
    // Process COARSEN_FACTOR points per thread
    for (int offset = 0; offset < COARSEN_FACTOR; offset++) {
        int i = base_i + offset;
        if (i >= grid.x) continue; // Skip if out of bounds
        
        float x = gridspacing * (float)i;
        float energy = 0.0f;
        
        // Calculate for all atoms
        for (int n = 0; n < atoms_in_chunk * 4; n += 4) {
            float dx = x - atoms[n];
            float dy = y - atoms[n + 1];
            float dz = z - atoms[n + 2];
            float dysqdzq = dy * dy + dz * dz;
            float charge = atoms[n + 3];
            energy += charge / sqrtf(dx * dx + dysqdzq);
        }
        
        // Write result
        energygrid[grid.x * grid.y * k + grid.x * j + i] += energy;
    }
}

// Chunked GPU implementation with thread coarsening approach
void cenergyParallelCoarsen(float* host_energygrid, dim3 grid_dim, float gridspacing, float z, const float* host_atoms,
                            int numatoms) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate device memory for the energy grid
    float* d_energygrid = NULL;
    size_t grid_size = grid_dim.x * grid_dim.y * grid_dim.z * sizeof(float);
    err = cudaMalloc((void**)&d_energygrid, grid_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory for energy grid (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // Initialize energy grid to zero
    err = cudaMemset(d_energygrid, 0, grid_size);
    if (err != cudaSuccess) {
        printf("Failed to initialize device memory (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_energygrid);
        return;
    }

    // Process atoms in chunks
    int num_chunks = (numatoms + CHUNK_SIZE - 1) / CHUNK_SIZE;
    printf("Processing %d atoms in %d chunks of size %d\n", numatoms, num_chunks, CHUNK_SIZE);

    // Set up execution configuration with coarsening factor
    dim3 blockDim(16, 16);
    dim3 gridDim(((grid_dim.x + COARSEN_FACTOR - 1) / COARSEN_FACTOR + blockDim.x - 1) / blockDim.x,
                 (grid_dim.y + blockDim.y - 1) / blockDim.y);

    printf("Thread Coarsening: Launching kernel with grid: %d x %d blocks of %d x %d threads\n", 
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // For each chunk of atoms
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        // Calculate the number of atoms in this chunk
        int start_atom = chunk * CHUNK_SIZE;
        int atoms_in_chunk = (start_atom + CHUNK_SIZE <= numatoms) ? CHUNK_SIZE : (numatoms - start_atom);

        // Copy this chunk to constant memory
        size_t chunk_size = atoms_in_chunk * 4 * sizeof(float);
        err = cudaMemcpyToSymbol(atoms, &host_atoms[start_atom * 4], chunk_size);
        if (err != cudaSuccess) {
            printf("Failed to copy atoms chunk %d to constant memory (error code %s)!\n", chunk,
                   cudaGetErrorString(err));
            cudaFree(d_energygrid);
            return;
        }

        // Launch kernel for this chunk
        cenergyCoarsenKernel<<<gridDim, blockDim>>>(d_energygrid, grid_dim, gridspacing, z, atoms_in_chunk);

        // Check for kernel launch errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error: %s\n", cudaGetErrorString(err));
            cudaFree(d_energygrid);
            return;
        }

        // Wait for kernel to complete
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Error synchronizing: %s\n", cudaGetErrorString(err));
            cudaFree(d_energygrid);
            return;
        }
    }

    // Copy results back to host
    err = cudaMemcpy(host_energygrid, d_energygrid, grid_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Failed to copy energy grid from device to host (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_energygrid);
        return;
    }

    // Free device memory
    cudaFree(d_energygrid);
}

void cenergySequential(float* energygrid, dim3 grid, float gridspacing, float z, const float* atoms, int numatoms) {
    int atomarrdim = numatoms * 4;  // x,y,z, and charge info for each atom
    for (int j = 0; j < grid.y; j++) {
        // calculate y coordinate of the grid point based on j
        float y = gridspacing * (float)j;
        for (int i = 0; i < grid.x; i++) {
            // calculate x coordinate based on i
            float x = gridspacing * (float)i;
            float energy = 0.0f;
            for (int n = 0; n < atomarrdim; n += 4) {
                float dx = x - atoms[n];
                float dy = y - atoms[n + 1];
                float dz = z - atoms[n + 2];
                energy += atoms[n + 3] / sqrtf(dx * dx + dy * dy + dz * dz);
            }
            // Fixed indexing - use integer z_index
            int z_index = (int)(z / gridspacing);
            energygrid[grid.x * grid.y * z_index + grid.x * j + i] = energy;
        }
    }
}

void cenergySequentialOptimized(float* energygrid, dim3 grid, float gridspacing, float z, const float* atoms,
                                int numatoms) {
    int atomarrdim = numatoms * 4;  // x,y,z, and charge info for each atom
    // starting point of the slice in the energy grid
    int grid_slice_offset = (grid.x * grid.y * z) / gridspacing;
    // calculate potential contribution of each atom
    for (int n = 0; n < atomarrdim; n += 4) {
        float dz = z - atoms[n + 2];  // all grid points in a slice have the same
        float dz2 = dz * dz;          // z value, no recalculation in inner loops
        float charge = atoms[n + 3];
        for (int j = 0; j < grid.y; j++) {
            float y = gridspacing * (float)j;
            float dy = y - atoms[n + 1];  // all grid points in a row have the same
            float dy2 = dy * dy;          // y value
            int grid_row_offset = grid_slice_offset + grid.x * j;
            for (int i = 0; i < grid.x; i++) {
                float x = gridspacing * (float)i;
                float dx = x - atoms[n];
                energygrid[grid_row_offset + i] += charge / sqrtf(dx * dx + dy2 + dz2);
            }
        }
    }
}

// Function to generate k synthetic atoms
void generate_atoms(float* atoms, int numatoms, float max_x, float max_y, float max_z) {
    // Seed random number generator with defined seed
    srand(RANDOM_SEED);

    for (int i = 0; i < numatoms; i++) {
        // Generate random positions within the grid boundaries
        atoms[i * 4] = (float)rand() / RAND_MAX * max_x;      // x position
        atoms[i * 4 + 1] = (float)rand() / RAND_MAX * max_y;  // y position
        atoms[i * 4 + 2] = (float)rand() / RAND_MAX * max_z;  // z position

        // Generate random charge between -2.0 and 2.0
        atoms[i * 4 + 3] = ((float)rand() / RAND_MAX * 4.0) - 2.0;  // charge
    }
}

// Uses the formula: |a - b| <= atol + rtol * |b|
int grids_allclose(const float* grid1, const float* grid2, dim3 grid_dimensions, float rtol, float atol, int verbose) {
    // Calculate the total number of grid points
    int total_grid_points = grid_dimensions.x * grid_dimensions.y * grid_dimensions.z;

    int mismatches = 0;
    float max_diff = 0.0f;
    int max_diff_idx = -1;

    // Compare each grid point
    for (int i = 0; i < total_grid_points; i++) {
        // Calculate absolute difference
        float abs_diff = fabsf(grid1[i] - grid2[i]);
        // Calculate tolerance threshold
        float threshold = atol + rtol * fabsf(grid2[i]);

        if (abs_diff > threshold) {
            mismatches++;

            // Track the location of maximum difference
            if (abs_diff > max_diff) {
                max_diff = abs_diff;
                max_diff_idx = i;
            }

            // If verbose and this is one of the first few mismatches, print details
            if (verbose && mismatches <= 10) {
                int z = i / (grid_dimensions.x * grid_dimensions.y);
                int remainder = i % (grid_dimensions.x * grid_dimensions.y);
                int y = remainder / grid_dimensions.x;
                int x = remainder % grid_dimensions.x;

                printf("Mismatch at grid point (%d, %d, %d): %f vs %f (diff: %f, threshold: %f)\n", x, y, z, grid1[i],
                       grid2[i], abs_diff, threshold);
            }
        }
    }

    // Print summary
    if (mismatches > 0) {
        printf("Found %d mismatches out of %d points (%.2f%%)\n", mismatches, total_grid_points,
               (float)mismatches * 100.0f / total_grid_points);

        // Print info about maximum difference
        if (max_diff_idx >= 0) {
            int z = max_diff_idx / (grid_dimensions.x * grid_dimensions.y);
            int remainder = max_diff_idx % (grid_dimensions.x * grid_dimensions.y);
            int y = remainder / grid_dimensions.x;
            int x = remainder % grid_dimensions.x;

            printf("Maximum difference: %f at point (%d, %d, %d)\n", max_diff, x, y, z);
        }

        return 0;  // Grids don't match
    }

    printf("All %d grid points match within tolerances (rtol=%e, atol=%e)\n", total_grid_points, rtol, atol);
    return 1;  // Grids match within tolerance
}

int main() {
    // Grid configuration
    dim3 grid(100, 100, 50);
    float gridspacing = 0.5;  // 0.5 Ångström between points
    float z = 10.0;           // Computing energy for z=10 plane

    // Number of atoms to generate
    int numatoms = 10000;  // Reduced for faster testing - adjust as needed

    // Allocate memory for atoms
    float* atoms = (float*)malloc(numatoms * 4 * sizeof(float));
    if (!atoms) {
        printf("Memory allocation for atoms failed\n");
        return 1;
    }

    // Generate random atoms
    float max_x = grid.x * gridspacing;
    float max_y = grid.y * gridspacing;
    float max_z = grid.z * gridspacing;
    generate_atoms(atoms, numatoms, max_x, max_y, max_z);

    // Print first few atoms for verification
    printf("First 3 atoms (using seed %d):\n", RANDOM_SEED);
    for (int i = 0; i < 3 && i < numatoms; i++) {
        printf("Atom %d: Position (%.2f, %.2f, %.2f), Charge: %.2f\n", i, atoms[i * 4], atoms[i * 4 + 1],
               atoms[i * 4 + 2], atoms[i * 4 + 3]);
    }

    // Allocate energy grids for each approach
    size_t grid_size = grid.x * grid.y * grid.z;
    float* energygrid_seq = (float*)calloc(grid_size, sizeof(float));
    float* energygrid_gpu_gather = (float*)calloc(grid_size, sizeof(float));
    float* energygrid_gpu_coarsen = (float*)calloc(grid_size, sizeof(float));

    if (!energygrid_seq || !energygrid_gpu_gather || !energygrid_gpu_coarsen) {
        printf("Memory allocation for energy grids failed\n");
        free(atoms);
        if (energygrid_seq) free(energygrid_seq);
        if (energygrid_gpu_gather) free(energygrid_gpu_gather);
        if (energygrid_gpu_coarsen) free(energygrid_gpu_coarsen);
        return 1;
    }

    // Run sequential approach and measure time
    clock_t start_seq, end_seq;
    start_seq = clock();
    cenergySequential(energygrid_seq, grid, gridspacing, z, atoms, numatoms);
    end_seq = clock();
    double time_seq = ((double)(end_seq - start_seq)) / CLOCKS_PER_SEC;

    // Run GPU gather approach with chunking and measure time
    clock_t start_gather, end_gather;
    start_gather = clock();
    cenergyParallelGather(energygrid_gpu_gather, grid, gridspacing, z, atoms, numatoms);
    end_gather = clock();
    double time_gather = ((double)(end_gather - start_gather)) / CLOCKS_PER_SEC;

    // Run GPU thread coarsening approach and measure time
    clock_t start_coarsen, end_coarsen;
    start_coarsen = clock();
    cenergyParallelCoarsen(energygrid_gpu_coarsen, grid, gridspacing, z, atoms, numatoms);
    end_coarsen = clock();
    double time_coarsen = ((double)(end_coarsen - start_coarsen)) / CLOCKS_PER_SEC;

    // Compare results
    printf("\nPerformance Results:\n");
    printf("Sequential Approach: %.6f seconds\n", time_seq);
    printf("GPU Gather Implementation: %.6f seconds\n", time_gather);
    printf("GPU Thread Coarsening Implementation: %.6f seconds\n", time_coarsen);

    if (time_seq > 0.0) {
        printf("GPU Gather Speedup: %.2fx\n", time_seq / time_gather);
        printf("GPU Thread Coarsening Speedup: %.2fx\n", time_seq / time_coarsen);
        printf("Thread Coarsening vs Gather: %.2fx\n", time_gather / time_coarsen);
    }

    // Print sample results from each approach
    int z_index = (int)(z / gridspacing);
    int sample_idx = grid.x * grid.y * z_index + grid.x * 50 + 50;
    printf("\nEnergy at grid point (50,50,%d):\n", z_index);
    printf("Sequential: %f\n", energygrid_seq[sample_idx]);
    printf("GPU Gather: %f\n", energygrid_gpu_gather[sample_idx]);
    printf("GPU Thread Coarsening: %f\n", energygrid_gpu_coarsen[sample_idx]);

    printf("\nComparing Sequential vs. GPU Gather:\n");
    grids_allclose(energygrid_seq, energygrid_gpu_gather, grid, 1e-2, 1e-3, 0);
    
    printf("\nComparing Sequential vs. GPU Thread Coarsening:\n");
    grids_allclose(energygrid_seq, energygrid_gpu_coarsen, grid, 1e-2, 1e-3, 0);
    
    printf("\nComparing GPU Gather vs. GPU Thread Coarsening:\n");
    grids_allclose(energygrid_gpu_gather, energygrid_gpu_coarsen, grid, 1e-2, 1e-3, 0);

    // Clean up
    free(energygrid_seq);
    free(energygrid_gpu_gather);
    free(energygrid_gpu_coarsen);
    free(atoms);

    return 0;
}