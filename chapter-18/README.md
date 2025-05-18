# Chapter 17

## Code

We implement all of the kernels discussed in Chapter 18. 

In particural we implement:
- Scatter kernel and its corresponding host code from `Fig. 18.5`
- Gather kernel and its corresponding host code from `Fig. 18.6`
- Thread coarsening kernel and its corresponding host code from `Fig. 18.8`
- Thread coarsening and memory coalescing kernel and its corresponding host code from `Fig. 18.10`

For each kernel we implement the corresponding host codebase. All implementations can be found in [cenergy.cu](./code/cenergy.cu). We implement a simple benchmark, comparing the performance between different approaches. The benchmark implementation can be found in [benchmark.cu](./code/benchmark.cu). Note that our benchmark includes copying the data from host memory to device constant memory via `cudaMemcpyToSymbol`. 

## Exercises

### Exercise 1

**Complete the host code for configuring the grid and calling the kernel in Fig. 18.6 with all the execution configuration parameters.**

Full implementation to be found in [cenergy.cu](./code/cenergy.cu). 

```cpp
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

    // Define 2D thread block and grid
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
    dim3 blocksPerGrid((grid_dim.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (grid_dim.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

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

        cenergyGatherKernel<<<blocksPerGrid, threadsPerBlock>>>(d_energygrid, grid_dim, gridspacing, z, atoms_in_chunk,
                                                                start_atom);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to launch kernel for chunk %d (error code %s)!\n", chunk, cudaGetErrorString(err));
            cudaFree(d_energygrid);
            return;
        }

        cudaDeviceSynchronize();
    }
```

### Exercise 2

**Compare the number of operations (memory loads, floating-point arithmetic, branches) executed in each iteration of the kernel in Fig. 18.8 with that in Fig. 18.6 for a coarsening factor of 8. Keep in mind that each iteration of the former corresponds to eight iterations of the latter.**

#### Fig. 18.6

```cpp
01 __constant__ float atoms[CHUNK_SIZE*4];
02 void __global__ cenergy(float *energygrid, dim3 grid, float gridspacing,
03                     float z, int numatoms) {
04     int i = blockIdx.x * blockDim.x + threadIdx.x;
05     int j = blockIdx.y * blockDim.y + threadIdx.y;
06     int atomarrdim = numatoms * 4;
07     int k = z / gridspacing;
08     float y = gridspacing * (float) j;
09     float x = gridspacing * (float) i;
10     float energy = 0.0f;
      // calculate potential contribution from all atoms
11     for (int n=0; n<atomarrdim; n+=4) {
12         float dx = x - atoms[n  ];
13         float dy = y - atoms[n+1];
14         float dz = z - atoms[n+2];
15         energy += atoms[n+3] / sqrtf(dx*dx + dy*dy + dz*dz);
16     }
17     energygrid[grid.x*grid.y*k + grid.x*j + i] += energy;
18 }
```

#### Fig. 18.8
We had to slighly modify the kernel in `Fig. 18.8`. The orignal one hardcoded the COARSEN_FACTOR (the `energy0`, `energy1` ... thing in lines 25-28). This one works conceptually in the same way, but it allows for more flexible thread coarseninging scheme. 

```cpp
01  __global__ void cenergyCoarsenKernel(float* *energygrid*, dim3 *grid*, float *gridspacing*, float *z*, int *atoms_in_chunk*) {
02      int base_i = blockIdx.x * blockDim.x * COARSEN_FACTOR + threadIdx.x * COARSEN_FACTOR;
03      int j = blockIdx.y * blockDim.y + threadIdx.y;
04      if (j >= *grid*.y) {
05          return;
06      }
07      int k = *z* / *gridspacing*;
08      float y = *gridspacing* * (float)j;
09      // Process COARSEN_FACTOR points per thread
10      for (int offset = 0; offset < COARSEN_FACTOR; offset++) {
11          int i = base_i + offset;
12          if (i >= *grid*.x) {
13              continue; // Skip if out of bounds
14          }
15          float x = *gridspacing* * (float)i;
16          float energy = 0.0f;
17          // Calculate for all atoms
18          for (int n = 0; n < *atoms_in_chunk* * 4; n += 4) {
19              float dx = x - atoms[n];
20              float dy = y - atoms[n + 1];
21              float dz = *z* - atoms[n + 2];
22              float dysqdzq = dy * dy + dz * dz;
23              float charge = atoms[n + 3];
24              energy += charge / sqrtf(dx * dx + dysqdzq);
25          }
26          // Write result
27          *energygrid*[*grid*.x * grid.y * k + *grid*.x * j + i] += energy;
28      }
29  }
```

### Exercise 3

**Give two potential disadvantages associated with increasing the amount of work done in each CUDA thread, as shown in Section 18.3.**

### Exercise 4

**Use Fig. 18.13 to explain how control divergence can arise when threads in a block process a bin in the neighborhood list.**


