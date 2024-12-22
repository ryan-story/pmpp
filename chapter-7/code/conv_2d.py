from functools import partial
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9' #4090
os.environ['MAX_JOBS'] = '4'
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from triton.testing import do_bench

def load_conv2d_extension():
    return load(
        name="conv2d_torch_with_constant_memory",
        sources=[
            str(Path(__file__).parent / "conv2d_kernels.cu"),
            str(Path(__file__).parent / "conv2d_functions_torch.cu")
        ],
        verbose=True,
        with_cuda=True
    )
@torch.inference_mode
def main():
    conv2d_extension = load_conv2d_extension()
    
    device = "cuda"
    height = 4096*2
    width = 4096*2
    r = 9 #TODO make sure FILTER_RADIOUS is set correctly in the kernel header
    kernel_size = 2 * r + 1
    
    input_tensor = torch.randn(height, width, device=device, dtype=torch.float32)
    kernel_tensor = torch.randn(kernel_size, kernel_size, device=device, dtype=torch.float32)
    
    input_torch = input_tensor.unsqueeze(0).unsqueeze(0)  
    kernel_torch = kernel_tensor.unsqueeze(0).unsqueeze(0)
    
    custom_output = conv2d_extension.conv2d_torch_with_tiled_convolution(input_tensor, kernel_tensor, r)
    
    torch_output = F.conv2d(input_torch, kernel_torch, padding=r).squeeze()

    assert torch.allclose(custom_output, torch_output, rtol=1e-5, atol=1e-5), "Your function output differs from torch."

    custom_conv2d = partial(conv2d_extension.conv2d_torch_with_tiled_convolution,
                        input_tensor,
                        kernel_tensor,
                        r,
    )


    torch_conv2d = partial(F.conv2d,
                        input=input_torch,
                        weight=kernel_torch,
                        padding=r,
    )

    custom_conv2d_time = do_bench(custom_conv2d, warmup=25, rep=100)
    torch_conv2d_time = do_bench(torch_conv2d, warmup=25, rep=100)

    print(f"Custom Conv2d kernel time: {custom_conv2d_time:.4f} ms")
    print(f"Torch Conv2d kernel time: {torch_conv2d_time:.4f} ms")


if __name__ == "__main__":
    main()