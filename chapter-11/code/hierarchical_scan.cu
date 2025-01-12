// nvcc hierarchical_scan.cu -o hierarchical_scan

#include <assert.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#define SECTION_SIZE 1024  // Maximum threads per block
#define cdiv(x, y) (((x) + (y) - 1)/(y))

#define CUDA_CHECK(call)                                                                                 \
    do {                                                                                                 \
        cudaError_t error = call;                                                                        \
        if (error != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE);                                                                          \
        }                                                                                                \
    } while (0)

// Phase 1: Block-level scan and collect block sums
__global__ void hierarchical_kogge_stone_phase1(float *X, float *Y, float *S, unsigned int N) {
    extern __shared__ float buffer[];

    unsigned int tid = threadIdx.x;
    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    //load data into the shared memory, each thread loads its part
    if (global_idx < N){
        buffer[tid] = X[global_idx];
    }
    else{
        buffer[tid] = 0.0f;
    }

    //kogge stone within the block
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        float temp;
        __syncthreads();
        if (tid >= stride){
            temp = buffer[tid] + buffer[tid - stride];
        }
        __syncthreads();
        if (tid >= stride){
            buffer[tid] = temp;
        }
    }

    // write back to the global memory
    if (global_idx < N){
        Y[global_idx] = buffer[tid];
    }
    
    // store the final sum into the S array in global memory
    if (tid == blockDim.x - 1){
        S[blockIdx.x] = buffer[tid];
    }

}

// Phase 2: Scan block sums
__global__ void hierarchical_kogge_stone_phase2(float *S, unsigned int num_blocks) {
    extern __shared__ float buffer[];
    unsigned int tid = threadIdx.x;

    // load into the shared memory

    if (tid < num_blocks){
        buffer[tid] = S[tid];
    }
    else{
        buffer[tid] = 0.0f;
    }

    //kogge stone on the block sums
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        float temp;
        __syncthreads();
        
        if (tid >= stride){
            temp = buffer[tid] + buffer[tid - stride];
        }
        __syncthreads();

        if (tid >= stride){
            buffer[tid] = temp;
        }
    }

    //write the results back into S
    if (tid < num_blocks){
        S[tid] = buffer[tid];
    }
}

// Phase 3: Distribute block sums
__global__ void hierarchical_kogge_stone_phase3(float *Y, float *S, unsigned int N) {
    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx < N && blockIdx.x > 0){
        Y[global_idx] += S[blockIdx.x-1];
    }
}

// Host function to coordinate the hierarchical scan
void hierarchical_scan(float *X, float *Y, unsigned int N) {
    float *d_X, *d_Y, *d_S;
    
    unsigned int block_size = SECTION_SIZE;
    unsigned int num_blocks = cdiv(N, block_size);
    
    CUDA_CHECK(cudaMalloc((void**)&d_X, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_S, num_blocks * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Phase 1: Block-level scan and collect block sums
    hierarchical_kogge_stone_phase1<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        d_X, d_Y, d_S, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Phase 2: Scan block sums
    hierarchical_kogge_stone_phase2<<<1, num_blocks, num_blocks * sizeof(float)>>>(
        d_S, num_blocks);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Phase 3: Distribute block sums
    hierarchical_kogge_stone_phase3<<<num_blocks, block_size>>>(d_Y, d_S, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
    CUDA_CHECK(cudaFree(d_S));
}

void sequential_inclusive_scan(float *X, float *Y, unsigned int N) {
    Y[0] = X[0];
    for(unsigned int i = 1; i < N; i++) {
        Y[i] = X[i] + Y[i-1];
    }
}

bool allclose(float* a, float* b, int N, float rtol = 1e-5, float atol = 1e-8) {
    for(int i = 0; i < N; i++) {
        float allowed_error = atol + rtol * fabs(b[i]);
        if(fabs(a[i] - b[i]) > allowed_error) {
            printf("Arrays differ at index %d: %f != %f (allowed error: %f)\n", 
                   i, a[i], b[i], allowed_error);
            return false;
        }
    }
    return true;
}

int main() {
    unsigned int length = 2048; 
    float* X = (float*)malloc(length * sizeof(float));
    float* Y_hierarchical = (float*)malloc(length * sizeof(float));
    float* Y_sequential = (float*)malloc(length * sizeof(float));
    
    for (unsigned int i = 0; i < length; i++) {
        X[i] = 1.0f;  // Using simple values for testing
    }
    
    sequential_inclusive_scan(X, Y_sequential, length);
    hierarchical_scan(X, Y_hierarchical, length);
    
    printf("First 8 elements of hierarchical scan: [");
    for (unsigned int i = 0; i < 8 && i < length; i++) {
        printf("%.1f%s", Y_hierarchical[i], (i < 7 && i < length-1) ? ", " : "");
    }
    printf("]\n");
    
    if (length > 16) {
        printf("Last 8 elements of hierarchical scan: [");
        for (unsigned int i = length - 8; i < length; i++) {
            printf("%.1f%s", Y_hierarchical[i], (i < length-1) ? ", " : "");
        }
        printf("]\n");
    }

    // Compare results
    printf("\nComparing hierarchical scan with sequential scan:\n");
    if(allclose(Y_hierarchical, Y_sequential, length)) {
        printf("Arrays are close enough!\n");
    } else {
        printf("Arrays differ significantly!\n");
    }
    
    // Cleanup
    free(X);
    free(Y_hierarchical);
    free(Y_sequential);
    
    return 0;
}