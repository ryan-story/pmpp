// nvcc reduction_sum.cu -o reduction_sum
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define BIN_SIZE 4
#define NUM_BINS ((26 + BIN_SIZE - 1) / BIN_SIZE)  
#define CFACTOR 32
#define BLOCKS_PER_SM 32

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

int getNumSMs() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // Get properties of device 0
    return prop.multiProcessorCount;
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

float simple_sequential_reduce_sum(float *data, int length){
    float total = 0.0f;
    for(unsigned int i = 0; i < length; ++i){
        total += data[i];
    }
    return total;
}


int main(int argc, char const *argv[])
{
    unsigned int length = 2048;

    float numbers[length];
    for(unsigned int i = 0; i < length; i++) {
        numbers[i] = 1.0f;
    }

    float sequential_sum = simple_sequential_reduce_sum(numbers, length);

    printf("Sequential sum %.2f\n", sequential_sum);

    return 0;
}
