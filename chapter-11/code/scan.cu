// nvcc scan.cu -o scan

#include <assert.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#define cdiv(x, y) (((x) + (y) - 1)/(y))

#define COARSE_FACTOR 2


#define CUDA_CHECK(call)                                                                                 \
    do {                                                                                                 \
        cudaError_t error = call;                                                                        \
        if (error != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE);                                                                          \
        }                                                                                                \
    } while (0)

__global__ void kogge_stone_scan_kernel(float *X, float *Y, unsigned int N){
    extern __shared__ float buffer[];
    unsigned int tid = threadIdx.x;

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
    // we do this trick to use one shared_mem as cuda doesn't allow to declare two extern shared memory fields
    extern __shared__ float shared_mem[];
    float* buffer1 = shared_mem;
    float* buffer2 = &shared_mem[N];

    unsigned int tid = threadIdx.x;

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
        if (tid >= stride) {
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


__global__ void brent_kung_scan_kernel(float *X, float *Y, unsigned int N){
    extern __shared__ float buffer[];
    //1 thread processes two elements
    unsigned int i = 2 * threadIdx.x;

    //move data from the global memory to the shared memory in a coalesced way
     if (i < N){
        buffer[i] = X[i];
        if (i + 1 < N){
            buffer[i + 1] = X[i +1];
        }
     }
     __syncthreads();

    //reduction tree
    for (unsigned int stride=1; stride <= blockDim.x; stride *= 2){
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < N){
            buffer[index] += buffer[index - stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        printf("Buffer after reduction tree:\n[");
        for (int j = 0; j < N; j++) {
            printf("%.2f", buffer[j]);
            if (j < N-1) printf(", ");
        }
        printf("]\n");
    }
    __syncthreads();

    // reversed tree
    for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2){
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if(index + stride < N){
            float temp = buffer[index];
            buffer[index] = buffer[index - stride];
            buffer[index + stride] += temp;
        }
        __syncthreads();
    }

    if (i < N){
        Y[i] = buffer[i];
        if (i + 1 < N){
           Y[i + 1] = buffer[i + 1];
        }
    }
}


__global__ void three_phases_scan_kernel(float *X, float *Y, unsigned int N){
    extern __shared__ float shared_mem[];
    float* buffer1 = shared_mem;
    float* sections_ends = &shared_mem[N];

    //load the data from the global memory into the shared memory of the block in a coeleased manner
    for (unsigned int i=0; i < COARSE_FACTOR; i++){
        buffer1[threadIdx.x + blockDim.x * i] = X[threadIdx.x + blockDim.x * i];
    }

    //Phase 1: partial sequenntial scan via coarsening
    for (unsigned int i=0; i < COARSE_FACTOR; i++){
    unsigned int idx = threadIdx.x + blockDim.x * i;
    if (idx > 0 && (idx/COARSE_FACTOR) == ((idx-1)/COARSE_FACTOR)) {  // Check if in same section
            buffer1[idx] += buffer1[idx-1];
        }
    }
    __syncthreads();

    //move the section ends into a seperate array
    unsigned int sections_ends_id = threadIdx.x * COARSE_FACTOR + COARSE_FACTOR-1;
    if (sections_ends_id < N){
        sections_ends[threadIdx.x] = buffer1[sections_ends_id];
    }
    __syncthreads();

    //Phase 2: perform the kogge stone scan at the end of the sections
    unsigned int tid = threadIdx.x;
    for (unsigned int stride = 1; stride < N/COARSE_FACTOR; stride *= 2){
        float temp;
        __syncthreads();
        if (tid >= stride){
            //read
            temp = sections_ends[tid] + sections_ends[tid - stride]; 
        }
        //make sure reading is done
        __syncthreads();
        if (tid >= stride){
            //write the updated version
            sections_ends[tid] = temp;
        }
    }
    __syncthreads();

    //Phase 3: distribute the calculated values back into the original array
    for (unsigned int i = 0; i < COARSE_FACTOR; ++i){
        unsigned int idx = threadIdx.x * COARSE_FACTOR + i;
        if (idx < N){
            unsigned int section = idx / COARSE_FACTOR;
            if (section > 0){
                buffer1[idx] += sections_ends[section-1];
            }
        }
    }
    __syncthreads();

    //back into the blobal memory
    for (unsigned int i=0; i < COARSE_FACTOR; i++){
        Y[threadIdx.x + blockDim.x * i] = buffer1[threadIdx.x + blockDim.x * i];
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


void scan_via_kogge_stone(float *X, float *Y, unsigned int N){
    assert(N <= 1024 && "Length must be less than or equal to 1024");

    float* d_X;
    float* d_Y;

    dim3 dimBlock(N); //for now we stick to a single section executed within a single block
    dim3 dimGrid(1);

    CUDA_CHECK(cudaMalloc((void**)&d_X, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));    

    kogge_stone_scan_kernel<<<dimGrid, dimBlock, N * sizeof(float) >>>(d_X, d_Y, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
}   

void scan_via_kogge_stone_with_double_buffering(float *X, float *Y, unsigned int N){
    assert(N <= 1024 && "Length must be less than or equal to 1024");

    float* d_X;
    float* d_Y;

    dim3 dimBlock(N); //for now we stick to a single section executed within a single block
    dim3 dimGrid(1);

    CUDA_CHECK(cudaMalloc((void**)&d_X, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));    

    kogge_stone_scan_kernel_with_double_buffering<<<dimGrid, dimBlock, 2 * N * sizeof(float)>>>(d_X, d_Y, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
}

void scan_via_brent_kung(float *X, float *Y, unsigned int N){
    assert(N <= 2048 && "Length must be less than or equal to 2048"); //twice kogge stone

    float* d_X;
    float* d_Y;

    dim3 dimBlock(cdiv(N, 2)); //every thread processing two elements
    dim3 dimGrid(1);

    CUDA_CHECK(cudaMalloc((void**)&d_X, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));    

    brent_kung_scan_kernel<<<dimGrid, dimBlock, N * sizeof(float)>>>(d_X, d_Y, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
}

void scan_via_three_phase_kernel(float *X, float *Y, unsigned int N){
    assert(N <= 1024 && "Length must be less than or equal to 1024");

    float* d_X;
    float* d_Y;

    dim3 dimBlock(cdiv(N, COARSE_FACTOR));
    dim3 dimGrid(1);

    CUDA_CHECK(cudaMalloc((void**)&d_X, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));        

    three_phases_scan_kernel<<<dimGrid, dimBlock,  (N + N/COARSE_FACTOR) * sizeof(float)>>>(d_X, d_Y, N);

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
    unsigned int length = 13;
    float* X = (float*)malloc(length * sizeof(float));
    float* Y_kogge_stone = (float*)malloc(length * sizeof(float));
    float* Y_kogge_stone_double = (float*)malloc(length * sizeof(float));
    float* Y_brent_kung = (float*)malloc(length * sizeof(float));
    float* Y_three_phases = (float*)malloc(length * sizeof(float));
    float* Y_sequential = (float*)malloc(length * sizeof(float));
    
    for (unsigned int i = 0; i < length; i++) {
        // X[i] = 9.0f * ((float)rand() / RAND_MAX);
        X[i] = 1.0*(i+1);
    }
    
    sequential_inclusive_scan(X, Y_sequential, length);
    // scan_via_kogge_stone(X, Y_kogge_stone, length);
    // scan_via_kogge_stone_with_double_buffering(X, Y_kogge_stone_double, length);
    scan_via_brent_kung(X, Y_brent_kung, length);
    // scan_via_three_phase_kernel(X, Y_three_phases, length);
    
    // printf("Kogge Stone Scan results:           [");
    // for (unsigned int i = 0; i < length; i++) {
    //     printf("%.2f%s", Y_kogge_stone[i], (i < length-1) ? ", " : "");
    // }
    // printf("]\n");

    // printf("Kogge Stone Double Buffer results:  [");
    // for (unsigned int i = 0; i < length; i++) {
    //     printf("%.2f%s", Y_kogge_stone_double[i], (i < length-1) ? ", " : "");
    // }
    // printf("]\n");

    printf("Brent-Kung Scan results:           [");
    for (unsigned int i = 0; i < length; i++) {
        printf("%.2f%s", Y_brent_kung[i], (i < length-1) ? ", " : "");
    }
    printf("]\n");

    // printf("Three Phases Scan results:         [");
    // for (unsigned int i = 0; i < length; i++) {
    //     printf("%.2f%s", Y_three_phases[i], (i < length-1) ? ", " : "");
    // }
    // printf("]\n");

    printf("Sequential Scan results:            [");
    for (unsigned int i = 0; i < length; i++) {
        printf("%.2f%s", Y_sequential[i], (i < length-1) ? ", " : "");
    }
    printf("]\n");
    
    printf("\nComparing results...\n");
    // printf("Comparing regular Kogge Stone with sequential:\n");
    // if(allclose(Y_kogge_stone, Y_sequential, length)) {
    //     printf("Arrays are close enough!\n");
    // } else {
    //     printf("Arrays differ significantly!\n");
    // }

    // printf("\nComparing double buffered Kogge Stone with sequential:\n");
    // if(allclose(Y_kogge_stone_double, Y_sequential, length)) {
    //     printf("Arrays are close enough!\n");
    // } else {
    //     printf("Arrays differ significantly!\n");
    // }

    printf("\nComparing Brent-Kung with sequential:\n");
    if(allclose(Y_brent_kung, Y_sequential, length)) {
        printf("Arrays are close enough!\n");
    } else {
        printf("Arrays differ significantly!\n");
    }

    // printf("\nComparing Three Phases with sequential:\n");
    // if(allclose(Y_three_phases, Y_sequential, length)) {
    //     printf("Arrays are close enough!\n");
    // } else {
    //     printf("Arrays differ significantly!\n");
    // }
    
    free(X);
    free(Y_kogge_stone);
    free(Y_kogge_stone_double);
    free(Y_brent_kung);
    free(Y_three_phases);
    free(Y_sequential);
    return 0;
}