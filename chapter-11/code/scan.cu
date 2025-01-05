// nvcc scan.cu -o scan

#include <assert.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#define SECTION_SIZE 8


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
    __shared__ float buffer[SECTION_SIZE];

    if (tid < N){
        buffer[tid] = X[tid];
    }
    else{
        buffer[tid] = 0.0;
    }
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        float temp;
        __syncthreads();
        if (tid >= stride){
            //read
            temp = buffer[tid] + buffer[tid - stride]; 
        }
        //make sure reading is done
        __syncthreads();
        if (tid >= stride){
            //write the updated version
            buffer[tid] = temp;
        }
    }
    if (tid < N){
        Y[tid] = buffer[tid];
    }
}


__global__ void kogge_stone_scan_kernel_with_double_buffering(float *X, float *Y, unsigned int N){
    unsigned int tid = threadIdx.x;
    __shared__ float buffer1[SECTION_SIZE];
    __shared__ float buffer2[SECTION_SIZE];

    float *src_buffer = buffer1;
    float *trg_buffer = buffer2;

    if (tid < N){
        src_buffer[tid] = X[tid];
    }
    else{
        src_buffer[tid] = 0.0;
    }
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        __syncthreads();
        if (tid >= stride) {  // Add boundary check
            trg_buffer[tid] = src_buffer[tid] + src_buffer[tid - stride];
        } else {
            trg_buffer[tid] = src_buffer[tid];
        }
        
        float* temp;
        temp = src_buffer;
        src_buffer = trg_buffer;
        trg_buffer = temp;
    }

    if (tid < N){
        Y[tid] = src_buffer[tid];
    }
}


// PyTorch-like allclose function
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


void scan_via_kogge_stone(float *X, float *Y, unsigned int N){
    assert(N == SECTION_SIZE && "Length must be equal to SECTION_SIZE");

    float* d_X;
    float* d_Y;

    dim3 dimBlock(SECTION_SIZE); //for now we stick to a single section executed within a single block
    dim3 dimGrid(1);

    CUDA_CHECK(cudaMalloc((void**)&d_X, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));    

    kogge_stone_scan_kernel_with_double_buffering<<<dimGrid, dimBlock>>>(d_X, d_Y, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
}   

void scan_via_kogge_stone_with_double_buffering(float *X, float *Y, unsigned int N){
    assert(N == SECTION_SIZE && "Length must be equal to SECTION_SIZE");

    float* d_X;
    float* d_Y;

    dim3 dimBlock(SECTION_SIZE); //for now we stick to a single section executed within a single block
    dim3 dimGrid(1);

    CUDA_CHECK(cudaMalloc((void**)&d_X, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));    

    kogge_stone_scan_kernel_with_double_buffering<<<dimGrid, dimBlock>>>(d_X, d_Y, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
}   


void sequential_inclusive_scan(float *X, float *Y, unsigned int N){
    Y[0] = X[0];
    for(unsigned int i=1; i<N; i++){
        Y[i] = X[i] + Y[i-1];
    }
}

int main() {
    unsigned int length = SECTION_SIZE;
    float* X = (float*)malloc(length * sizeof(float));
    float* Y_kogge_stone = (float*)malloc(length * sizeof(float));
    float* Y_kogge_stone_double = (float*)malloc(length * sizeof(float));
    float* Y_sequential = (float*)malloc(length * sizeof(float));
    
    for (unsigned int i = 0; i < length; i++) {
        // X[i] = 9.0f * ((float)rand() / RAND_MAX);
        X[i] = 1.0*(i+1);
    }
    
    scan_via_kogge_stone(X, Y_kogge_stone, length);
    scan_via_kogge_stone_with_double_buffering(X, Y_kogge_stone_double, length);
    sequential_inclusive_scan(X, Y_sequential, length);
    
    printf("Kogge Stone Scan results:           [");
    for (unsigned int i = 0; i < length; i++) {
        printf("%.2f%s", Y_kogge_stone[i], (i < length-1) ? ", " : "");
    }
    printf("]\n");

    printf("Kogge Stone Double Buffer results:  [");
    for (unsigned int i = 0; i < length; i++) {
        printf("%.2f%s", Y_kogge_stone_double[i], (i < length-1) ? ", " : "");
    }
    printf("]\n");

    printf("Sequential Scan results:            [");
    for (unsigned int i = 0; i < length; i++) {
        printf("%.2f%s", Y_sequential[i], (i < length-1) ? ", " : "");
    }
    printf("]\n");
    
    printf("\nComparing results...\n");
    printf("Comparing regular Kogge Stone with sequential:\n");
    if(allclose(Y_kogge_stone, Y_sequential, length)) {
        printf("Arrays are close enough!\n");
    } else {
        printf("Arrays differ significantly!\n");
    }

    printf("\nComparing double buffered Kogge Stone with sequential:\n");
    if(allclose(Y_kogge_stone_double, Y_sequential, length)) {
        printf("Arrays are close enough!\n");
    } else {
        printf("Arrays differ significantly!\n");
    }
    
    free(X);
    free(Y_kogge_stone);
    free(Y_kogge_stone_double);
    free(Y_sequential);
    return 0;
}