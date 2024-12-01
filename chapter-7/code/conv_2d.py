import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9' #4090
os.environ['MAX_JOBS'] = '4'

from pathlib import Path
import torch
from torch.utils.cpp_extension import load

def load_conv2d_extension():
    return load(
        name="conv2d_torch",
        sources=[
            str(Path(__file__).parent / "conv2d_functions.cu"),
            str(Path(__file__).parent / "conv2d_functions_torch.cu")
        ],
        verbose=True,
        with_cuda=True
    )

def main():
    conv2d_extension = load_conv2d_extension()
    
    device = "cuda"
    batch_size = 1
    channels = 1
    height = 28
    width = 28
    padding = 1
    kernel_size = 2 * padding + 1
    
    input_tensor = torch.randn(height, width, device=device, dtype=torch.float32)
    kernel_tensor = torch.randn(kernel_size, kernel_size, device=device, dtype=torch.float32)
    
    output = conv2d_extension.conv2d_torch(input_tensor, kernel_tensor, padding)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()