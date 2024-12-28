// nvcc histogram.cu -o histogram

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define BIN_SIZE 4
#define NUM_BINS ((26 + BIN_SIZE - 1) / BIN_SIZE)  

#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
            __FILE__, __LINE__, \
            cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void clear_l2() {
    static int l2_clear_size = 0;
    static unsigned char* gpu_scratch_l2_clear = NULL;
    if (!gpu_scratch_l2_clear) {
        cudaDeviceGetAttribute(&l2_clear_size, cudaDevAttrL2CacheSize, 0);
        l2_clear_size *= 2;
        gpuErrchk(cudaMalloc(&gpu_scratch_l2_clear, l2_clear_size));
    }
    gpuErrchk(cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size));
}

char* generate_random_text(unsigned int length) {
    char* text = (char*)malloc((length + 1) * sizeof(char));
    if (text == NULL) {
        printf("Memory allocation failed!\n");
        return NULL;
    }
    srand(time(NULL));
    for (unsigned int i = 0; i < length; i++) {
        text[i] = 'a' + (rand() % 26);
    }
    text[length] = '\0';
    return text;
}

__global__ void histo_kernel(char *data, unsigned int length, unsigned int *histo) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&histo[alphabet_position/BIN_SIZE], 1);
        }
    }
}


__global__ void histo_private_kernel(char *data, unsigned int length, unsigned int *histo) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&histo[blockIdx.x*NUM_BINS + alphabet_position/BIN_SIZE], 1);
        }
    }

    if (blockIdx.x > 0){
        __syncthreads();
        for(unsigned int bin=threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
            unsigned int binValue = histo[blockIdx.x*NUM_BINS + bin];
            if(binValue > 0){
                atomicAdd(&histo[bin], binValue);
            }
        }
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

void histogram_parallel(char *data, unsigned int length, unsigned int *histo) {
    char *d_data;
    unsigned int *d_histo;
    
    CUDA_CHECK(cudaMalloc((void**)&d_data, length * sizeof(char)));
    CUDA_CHECK(cudaMemcpy(d_data, data, length * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&d_histo, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_histo, 0, NUM_BINS * sizeof(unsigned int)));
    
    dim3 dimBlock(1024);
    dim3 dimGrid(cdiv(length, dimBlock.x));
    
    histo_kernel<<<dimGrid, dimBlock>>>(d_data, length, d_histo);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(histo, d_histo, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_histo));
}

void histogram_parallel_private(char *data, unsigned int length, unsigned int *histo) {
    char *d_data;
    unsigned int *d_histo;
    
    CUDA_CHECK(cudaMalloc((void**)&d_data, length * sizeof(char)));
    CUDA_CHECK(cudaMemcpy(d_data, data, length * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&d_histo, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_histo, 0, NUM_BINS * sizeof(unsigned int)));
    
    dim3 dimBlock(1024);
    dim3 dimGrid(cdiv(length, dimBlock.x));
    
    histo_private_kernel<<<dimGrid, dimBlock>>>(d_data, length, d_histo);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(histo, d_histo, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_histo));
}


void histogram_sequential(char *data, unsigned int length, unsigned int *histo) {
    for(unsigned int i = 0; i < length; ++i) {
        int alphabet_position = data[i] - 'a';
        if(alphabet_position >= 0 && alphabet_position < 26)
            histo[alphabet_position/BIN_SIZE]++;
    }
}

float benchmark_histogram(void (*func)(char*, unsigned int, unsigned int*),
                        char* data, unsigned int length, unsigned int* histo,
                        int warmup = 25, int reps = 100) {
    memset(histo, 0, NUM_BINS * sizeof(unsigned int));
    
    for (int i = 0; i < warmup; ++i) {
        func(data, length, histo);
        memset(histo, 0, NUM_BINS * sizeof(unsigned int));
    }

    cudaEvent_t iterStart, iterStop;
    cudaEventCreate(&iterStart);
    cudaEventCreate(&iterStop);
    
    float totalTime_ms = 0.0f;
    
    for (int i = 0; i < reps; ++i) {
        clear_l2();
        memset(histo, 0, NUM_BINS * sizeof(unsigned int));
        cudaEventRecord(iterStart);
        func(data, length, histo);
        cudaEventRecord(iterStop);
        cudaEventSynchronize(iterStop);
        
        float iterTime = 0.0f;
        cudaEventElapsedTime(&iterTime, iterStart, iterStop);
        totalTime_ms += iterTime;
    }
    
    cudaEventDestroy(iterStart);
    cudaEventDestroy(iterStop);
    
    return totalTime_ms / reps;
}

int main(int argc, char const *argv[]) {
    unsigned int length = 10000000;
    unsigned int histo_parallel[NUM_BINS] = {0};
    unsigned int histo_sequential[NUM_BINS] = {0};
    unsigned int histo_private[NUM_BINS] = {0};  // Added for private implementation
    
    char* data = generate_random_text(length);
    if (data == NULL) {
        return 1;
    }
    
    printf("Configuration:\n");
    printf("BIN_SIZE: %d\n", BIN_SIZE);
    printf("NUM_BINS: %d\n\n", NUM_BINS);
    
    // Benchmark parallel version
    printf("Benchmarking parallel histogram...\n");
    float parallel_time = benchmark_histogram(histogram_parallel, data, length, histo_parallel);
    
    // Benchmark parallel private version
    printf("Benchmarking parallel private histogram...\n");
    float private_time = benchmark_histogram(histogram_parallel_private, data, length, histo_private);
    
    // Benchmark sequential version
    printf("Benchmarking sequential histogram...\n");
    float sequential_time = benchmark_histogram(histogram_sequential, data, length, histo_sequential);
    
    // Print results
    printf("\nResults:\n");
    printf("Parallel Implementation:\n");
    printf("Average time: %.3f ms\n", parallel_time);
    printf("Histogram values:\n");
    for (int i = 0; i < NUM_BINS; i++) {
        printf("Bin %d (letters %c-%c): %u\n", 
               i, 
               'a' + (i * BIN_SIZE), 
               'a' + min(25, (i + 1) * BIN_SIZE - 1), 
               histo_parallel[i]);
    }
    
    printf("\nParallel Private Implementation:\n");
    printf("Average time: %.3f ms\n", private_time);
    printf("Histogram values:\n");
    for (int i = 0; i < NUM_BINS; i++) {
        printf("Bin %d (letters %c-%c): %u\n", 
               i, 
               'a' + (i * BIN_SIZE), 
               'a' + min(25, (i + 1) * BIN_SIZE - 1), 
               histo_private[i]);
    }
    
    printf("\nSequential Implementation:\n");
    printf("Average time: %.3f ms\n", sequential_time);
    printf("Histogram values:\n");
    for (int i = 0; i < NUM_BINS; i++) {
        printf("Bin %d (letters %c-%c): %u\n", 
               i, 
               'a' + (i * BIN_SIZE), 
               'a' + min(25, (i + 1) * BIN_SIZE - 1), 
               histo_sequential[i]);
    }
    
    printf("\nSpeedups:\n");
    printf("Parallel vs Sequential: %.2fx\n", sequential_time / parallel_time);
    printf("Parallel Private vs Sequential: %.2fx\n", sequential_time / private_time);
    printf("Parallel Private vs Parallel: %.2fx\n", parallel_time / private_time);
    
    bool results_match = true;
    for (int i = 0; i < NUM_BINS; i++) {
        if (histo_parallel[i] != histo_sequential[i] || histo_private[i] != histo_sequential[i]) {
            results_match = false;
            break;
        }
    }
    printf("Results match: %s\n", results_match ? "Yes" : "No");
    
    free(data);
    return 0;
}