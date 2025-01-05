// nvcc scan.cu -o scan

#include <assert.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#define SECTION_SIZE 16


#define CUDA_CHECK(call)                                                                                 \
    do {                                                                                                 \
        cudaError_t error = call;                                                                        \
        if (error != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE);                                                                          \
        }                                                                                                \
    } while (0)

__global__ void kogge_stone_scan_kernel(float *X, float *Y, unsigned int N){
    unsigned int tid = threadIdx.x;
    Y[tid] = 1.0 * (tid+1); 
}


void scan_via_kogge_stone(float *X, float *Y, unsigned int N){
    assert(N == SECTION_SIZE && "Length must be equal to SECTION_SIZE");

    float* d_X;
    float* d_Y;

    dim3 dimBlock(SECTION_SIZE); //for now we stick to a single section executed within a single block
    dim3 dimGrid(1);

    CUDA_CHECK(cudaMalloc((void**)&d_X, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));    

    kogge_stone_scan_kernel<<<dimGrid, dimBlock>>>(d_X, d_Y, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
}   


//inclusive scan
void sequential_inclusive_scan(float *X, float *Y, unsigned int N){
    Y[0] = X[0];
    for(unsigned int i=1; i<N; i++){
        Y[i] = X[i] + Y[i-1];
    }
}

int main() {
    unsigned int length = 16;
    float* X = (float*)malloc(length * sizeof(float));
    float Y[16] = {0.0f};
    
    srand(time(NULL));
    for (unsigned int i = 0; i < length; i++) {
        // x[i] = 9.0f * ((float)rand() / RAND_MAX);
        X[i] = 1.0;
        // printf("numbers[%d] = %.2f\n", i, X[i]);
    }

    scan_via_kogge_stone(X, Y, length);

    for (unsigned int i = 0; i < length; i++) {
        printf("numbers[%d] = %.2f\n", i, Y[i]);
    }

    free(X);
    return 0;
}