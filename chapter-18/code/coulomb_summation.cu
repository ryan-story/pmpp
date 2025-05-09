#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Define a fixed seed for reproducibility
#define RANDOM_SEED 42

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

int main() {
    // Grid configuration
    dim3 grid(100, 100, 50);
    float gridspacing = 0.5;  // 0.5 Ångström between points
    float z = 10.0;           // Computing energy for z=10 plane

    // Number of atoms to generate
    int numatoms = 10000;

    // Allocate memory for atoms
    float* atoms = (float*)malloc(numatoms * 4 * sizeof(float));
    if (!atoms) {
        printf("Memory allocation for atoms failed\n");
        return 1;
    }

    // float atoms[12] = {
    //     5.2,  7.1,  3.8,  1.0,   // First atom: position (5.2, 7.1, 3.8) with +1 charge
    //     12.4, 9.6,  10.2, -1.0,  // Second atom: position (12.4, 9.6, 10.2) with -1 charge
    //     8.3,  15.7, 6.9,  2.0    // Third atom: position (8.3, 15.7, 6.9) with +2 charge
    // };
    // int numatoms = 3;

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

    // Pre-allocated energy grid array
    float* energygrid = (float*)calloc(grid.x * grid.y * grid.z, sizeof(float));
    if (!energygrid) {
        printf("Memory allocation for energy grid failed\n");
        free(atoms);
        return 1;
    }

    // Function call
    cenergySequential(energygrid, grid, gridspacing, z, atoms, numatoms);
    // cenergySequentialOptimized(energygrid, grid, gridspacing, z, atoms, numatoms);

    // Print some sample results
    int z_index = (int)(z / gridspacing);
    printf("Energy at grid point (50,50,%d): %f\n", z_index, energygrid[grid.x * grid.y * z_index + grid.x * 50 + 50]);

    // Clean up
    free(energygrid);
    free(atoms);

    return 0;
}
