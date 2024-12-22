#include <torch/extension.h>
#include "conv2d_kernels.cuh"
#include <c10/cuda/CUDAStream.h>

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor conv2d_torch(torch::Tensor input, torch::Tensor kernel, int padding) {
    TORCH_CHECK(input.device().type() == torch::kCUDA, "Input must be a CUDA tensor");
    TORCH_CHECK(kernel.device().type() == torch::kCUDA, "Kernel must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(kernel.dtype() == torch::kFloat32, "Kernel must be float32");

    int height = input.size(0);
    int width = input.size(1);
    int r = padding;

    auto output = torch::empty_like(input);

    const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    conv2d_kernel<<<dimGrid, dimBlock, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        kernel.data_ptr<float>(),
        output.data_ptr<float>(),
        r, height, width
    );

    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}

torch::Tensor conv2d_torch_with_constant_memory(torch::Tensor input, torch::Tensor kernel, int r) {
    TORCH_CHECK(input.device().type() == torch::kCUDA, "Input must be a CUDA tensor");
    TORCH_CHECK(kernel.device().type() == torch::kCUDA, "Kernel must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(kernel.dtype() == torch::kFloat32, "Kernel must be float32");

    int height = input.size(0);
    int width = input.size(1);
    auto output = torch::empty_like(input);
    cudaError_t error;

    error = initConstFilter(kernel.data_ptr<float>(), r);
    if (error != cudaSuccess) {
        std::cout << "Failed to initialize constant memory: " << cudaGetErrorString(error) << std::endl;
        return output;
    }

    const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    conv2d_kernel_with_constant_memory<<<dimGrid, dimBlock, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        r, height, width
    );

    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}

#define IN_TILE_SIZE 32
#define OUT_TILE_SIZE (IN_TILE_SIZE - 2*FILTER_RADIUS)
__global__ void tiled_convolution_kernel(float *M, float *P, int r, int height, int width) {
    int row = blockDim.y *IN_TILE_SIZE + threadIdx.y - FILTER_RADIUS;
    int col = blockDim.x * IN_TILE_SIZE + threadIdx.x - FILTER_RADIUS;

    __shared__ float M_s[IN_TILE_SIZE][IN_TILE_SIZE];

    if (row >= 0 && row < height && col >= 0 && col < width)
        M_s[threadIdx.y][threadIdx.x] = M[row * width + col];
    else {
        M_s[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();

    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    int filterIndex;

    if (row >= 0 && row < height && col >= 0 && col < width){
        if (tileRow >= 0 && tileRow < OUT_TILE_SIZE && tileCol >= 0 && tileCol < OUT_TILE_SIZE){
            float Pvalue = 0.0f;

            for(int fRow = 0; fRow < 2 * FILTER_RADIUS +1; fRow++){
                for(int fCol = 0; fCol < 2 * FILTER_RADIUS +1; fCol++){
                    filterIndex = fRow * (2*FILTER_RADIUS+1) + fCol;
                    // Pvalue = M_s[fRow][fCol] * constFilter[filterIndex];
                    Pvalue += M_s[threadIdx.y + fRow - FILTER_RADIUS][threadIdx.x + fCol - FILTER_RADIUS] * constFilter[filterIndex];
                }
            }
            P[row * width + col] = Pvalue;

            // if (threadIdx.x == 10 && threadIdx.y == 10 && blockIdx.x == 0 && blockIdx.y == 0) {
            //        printf("HELLO from here :)\n");
            //        printf("Pvlue %d\n", P[row * width + col]);
            //     }

            
        }
    }
}


torch::Tensor conv2d_torch_with_tiled_convolution(torch::Tensor input, torch::Tensor kernel, int r) {
    TORCH_CHECK(input.device().type() == torch::kCUDA, "Input must be a CUDA tensor");
    TORCH_CHECK(kernel.device().type() == torch::kCUDA, "Kernel must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(kernel.dtype() == torch::kFloat32, "Kernel must be float32");

    int height = input.size(0);
    int width = input.size(1);
    auto output = torch::empty_like(input);
    cudaError_t error;

    error = initConstFilter(kernel.data_ptr<float>(), r);
    if (error != cudaSuccess) {
        std::cout << "Failed to initialize constant memory: " << cudaGetErrorString(error) << std::endl;
        return output;
    }

    const dim3 dimBlock(IN_TILE_SIZE, IN_TILE_SIZE);
    dim3 dimGrid(cdiv(width+2*FILTER_RADIUS, OUT_TILE_SIZE), cdiv(height+2*FILTER_RADIUS, OUT_TILE_SIZE));

    tiled_convolution_kernel<<<dimGrid, dimBlock, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        r, height, width
    );

    C10_CUDA_CHECK(cudaGetLastError());
    return output;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_torch", &conv2d_torch, "2D Convolution with CUDA");
    m.def("conv2d_torch_with_constant_memory", &conv2d_torch_with_constant_memory, "2D Convolution with CUDA utilizing constant memory");
    m.def("conv2d_torch_with_tiled_convolution", &conv2d_torch_with_tiled_convolution, "2D Convolution with tiling");
}