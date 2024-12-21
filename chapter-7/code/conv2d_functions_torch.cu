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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_torch", &conv2d_torch, "2D Convolution with CUDA");
    m.def("conv2d_torch_with_constant_memory", &conv2d_torch_with_constant_memory, "2D Convolution with CUDA utilizing constant memory");
}