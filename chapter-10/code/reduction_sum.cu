// nvcc reduction_sum.cu -o reduction_sum
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <assert.h>

#define BLOCK_DIM 1024

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

bool is_close(float a, float b, float rtol = 1e-5, float atol = 1e-8) {
    float diff = fabs(a - b);
    float tolerance = atol + rtol * fabs(b);
    return diff <= tolerance;
}

float benchmark_sum_reduction(float (*func)(float *, int),
                            float *data, unsigned int length,
                            int warmup = 25, int reps = 100){

    for (int i = 0; i < warmup; ++i){
        func(data, length);
    }

    cudaEvent_t iterStart, iterStop;
    cudaEventCreate(&iterStart);
    cudaEventCreate(&iterStop);

    float totalTime_ms = 0.0f;

    for(int i = 0; i < reps; ++i){
        clear_l2();
        cudaEventRecord(iterStart);
        func(data, length);
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


__global__ void simple_sum_reduction_kernel(float* input, float* output){
    unsigned int i = 2 * threadIdx.x;

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2){
        if (threadIdx.x % stride == 0){
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        *output = input[0];
}

__global__ void covergent_sum_reduction_kernel(float* input, float* output){
    unsigned int i = threadIdx.x;

    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2){
        if (threadIdx.x < stride){
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0){
        *output = input[0];
    }
}

__global__ void shared_memory_sum_reduction_kernel(float* input, float* output){
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;

    input_s[t] = input[t] + input[t+BLOCK_DIM];

    //we already did the first iteration compared to covergent_sum_reduction_kernel so we start with the 2nd one
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2){
        __syncthreads();
        if (t < stride){
            input_s[t] += input_s[t+stride];
        }
    }

    if (t == 0)
        *output = input_s[0];
}

__global__ void segmented_sum_reduction_kernel(float* input, float* output, int length){
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    if (i >= length || i + BLOCK_DIM >= length) return;

    input_s[t] = input[i] + input[i+BLOCK_DIM];

    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2){
        __syncthreads();
        if (t < stride){
            input_s[t] += input_s[t+stride];
        }
    }

    if (t == 0)
        atomicAdd(output, input_s[0]);
}



float simple_parallel_sum_reduction(float *data, int length){
    assert(length == 2 * BLOCK_DIM && "Length must be equal to 2 * BLOCK_DIM");

    float total;
    float *d_total;
    float* d_data;

     dim3 dimBlock(BLOCK_DIM); //we always run this with as much threads in block as possible
     dim3 dimGrid(1); //since the blocks can't communicate we are stuck for now with a single block

     CUDA_CHECK(cudaMalloc((void**)&d_data, length * sizeof(float)));
     CUDA_CHECK(cudaMalloc((void**)&d_total, sizeof(float)));

     CUDA_CHECK(cudaMemcpy(d_data, data, length * sizeof(float), cudaMemcpyHostToDevice));

    simple_sum_reduction_kernel<<<dimGrid, dimBlock>>>(d_data, d_total);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_total));

    return total;
}

float covergent_parallel_sum_reduction(float *data, int length){
    assert(length == 2 * BLOCK_DIM && "Length must be equal to 2 * BLOCK_DIM");

    float total;
    float *d_total;
    float* d_data;

     dim3 dimBlock(BLOCK_DIM); //we always run this with as much threads in block as possible
     dim3 dimGrid(1); //since the blocks can't communicate we are stuck for now with a single block

     CUDA_CHECK(cudaMalloc((void**)&d_data, length * sizeof(float)));
     CUDA_CHECK(cudaMalloc((void**)&d_total, sizeof(float)));

     CUDA_CHECK(cudaMemcpy(d_data, data, length * sizeof(float), cudaMemcpyHostToDevice));

    covergent_sum_reduction_kernel<<<dimGrid, dimBlock>>>(d_data, d_total);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_total));

    return total;
}

float shared_memory_sum_reduction(float *data, int length){
    assert(length == 2 * BLOCK_DIM && "Length must be equal to 2 * BLOCK_DIM");

    float total;
    float *d_total;
    float* d_data;

     dim3 dimBlock(BLOCK_DIM); //we always run this with as much threads in block as possible
     dim3 dimGrid(1); //since the blocks can't communicate we are stuck for now with a single block

     CUDA_CHECK(cudaMalloc((void**)&d_data, length * sizeof(float)));
     CUDA_CHECK(cudaMalloc((void**)&d_total, sizeof(float)));

     CUDA_CHECK(cudaMemcpy(d_data, data, length * sizeof(float), cudaMemcpyHostToDevice));

    shared_memory_sum_reduction_kernel<<<dimGrid, dimBlock>>>(d_data, d_total);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_total));

    return total;
}


float segmented_sum_reduction(float *data, int length){
    float total;
    float *d_total;
    float* d_data;

     dim3 dimBlock(BLOCK_DIM); //we always run this with as much threads in block as possible
     dim3 dimGrid((length + BLOCK_DIM - 1)/BLOCK_DIM); //since the blocks can't communicate we are stuck for now with a single block

     CUDA_CHECK(cudaMalloc((void**)&d_data, length * sizeof(float)));
     CUDA_CHECK(cudaMalloc((void**)&d_total, sizeof(float)));

     CUDA_CHECK(cudaMemcpy(d_data, data, length * sizeof(float), cudaMemcpyHostToDevice));

    segmented_sum_reduction_kernel<<<dimGrid, dimBlock>>>(d_data, d_total, length);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_total));

    return total;
}

float sequential_sum_reduction(float *data, int length){
    double total = 0.0;
    for(unsigned int i = 0; i < length; ++i){
        total += data[i];
    }
    return (float)total;
}

int main(int argc, char const *argv[]) {
    unsigned int length = 20480000;
    float* numbers = (float*)malloc(length * sizeof(float));
    
    for(unsigned int i = 0; i < length; i++) {
        numbers[i] = 1.0f;
    }

    // printf("Benchmarking simple parallel sum reduction...\n");
    // float simple_parallel_time = benchmark_sum_reduction(simple_parallel_sum_reduction, numbers, length);
    // float simple_parallel_sum = simple_parallel_sum_reduction(numbers, length);
    
    // printf("Benchmarking convergent parallel sum reduction...\n");
    // float convergent_parallel_time = benchmark_sum_reduction(covergent_parallel_sum_reduction, numbers, length);
    // float convergent_parallel_sum = covergent_parallel_sum_reduction(numbers, length);
    
    // printf("Benchmarking shared memory parallel sum reduction...\n");
    // float shared_memory_time = benchmark_sum_reduction(shared_memory_sum_reduction, numbers, length);
    // float shared_memory_sum = shared_memory_sum_reduction(numbers, length);

    printf("Benchmarking segmented sum reduction...\n");
    float segmented_time = benchmark_sum_reduction(segmented_sum_reduction, numbers, length);
    float segmented_sum = segmented_sum_reduction(numbers, length);
    
    printf("Benchmarking sequential sum reduction...\n");
    float sequential_time = benchmark_sum_reduction(sequential_sum_reduction, numbers, length, 10, 10);
    float sequential_sum = sequential_sum_reduction(numbers, length);
    
    printf("\nResults:\n");
    // printf("Simple Parallel Implementation:\n");
    // printf("Sum: %.2f\n", simple_parallel_sum);
    // printf("Average time: %.3f ms\n", simple_parallel_time);
    
    // printf("\nConvergent Parallel Implementation:\n");
    // printf("Sum: %.2f\n", convergent_parallel_sum);
    // printf("Average time: %.3f ms\n", convergent_parallel_time);
    
    // printf("\nShared Memory Parallel Implementation:\n");
    // printf("Sum: %.2f\n", shared_memory_sum);
    // printf("Average time: %.3f ms\n", shared_memory_time);

    printf("\nSegmented Sum Implementation:\n");
    printf("Sum: %.2f\n", segmented_sum);
    printf("Average time: %.3f ms\n", segmented_time);
    
    printf("\nSequential Implementation:\n");
    printf("Sum: %.2f\n", sequential_sum);
    printf("Average time: %.3f ms\n", sequential_time);
    
    printf("\nSpeedup:\n");
    // printf("Simple Parallel vs Sequential: %.2fx\n", sequential_time / simple_parallel_time);
    // printf("Convergent Parallel vs Sequential: %.2fx\n", sequential_time / convergent_parallel_time);
    // printf("Shared Memory Parallel vs Sequential: %.2fx\n", sequential_time / shared_memory_time);
    printf("Segmented Sum vs Sequential: %.2fx\n", sequential_time / segmented_time);
    
    // bool results_match = is_close(sequential_sum, simple_parallel_sum) &&
    //                     is_close(simple_parallel_sum, convergent_parallel_sum) &&
    //                     is_close(convergent_parallel_sum, shared_memory_sum) &&
    //                     is_close(shared_memory_sum, segmented_sum);
    bool results_match = is_close(sequential_sum, segmented_sum);                    
    printf("\nAll results match: %s\n", results_match ? "Yes" : "No");
    
    free(numbers);
    return 0;
}
