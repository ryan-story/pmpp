import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9' #4090
os.environ['MAX_JOBS'] = '4'
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import numpy as np

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

def compare_outputs(custom_output, torch_output, rtol=1e-5, atol=1e-5):
    """Compare the outputs of the two implementations."""
    is_close = torch.allclose(custom_output, torch_output, rtol=rtol, atol=atol)
    max_diff = torch.max(torch.abs(custom_output - torch_output)).item()
    mean_diff = torch.mean(torch.abs(custom_output - torch_output)).item()
    return {
        "is_close": is_close,
        "max_difference": max_diff,
        "mean_difference": mean_diff
    }

def main():
    conv2d_extension = load_conv2d_extension()
    
    device = "cuda"
    height = 28
    width = 28
    padding = 1
    kernel_size = 2 * padding + 1
    
    input_tensor = torch.randn(height, width, device=device, dtype=torch.float32)
    kernel_tensor = torch.randn(kernel_size, kernel_size, device=device, dtype=torch.float32)
    
    input_torch = input_tensor.unsqueeze(0).unsqueeze(0)  
    kernel_torch = kernel_tensor.unsqueeze(0).unsqueeze(0)
    
    custom_output = conv2d_extension.conv2d_torch(input_tensor, kernel_tensor, padding)
    
    torch_output = F.conv2d(input_torch, kernel_torch, padding=padding).squeeze()
    comparison = compare_outputs(custom_output, torch_output)

    print(f"Custom implementation output shape: {custom_output.shape}")
    print(f"PyTorch implementation output shape: {torch_output.shape}")
    print("\nComparison Results:")
    print(f"Outputs match within tolerance: {comparison['is_close']}")
    print(f"Maximum absolute difference: {comparison['max_difference']:.8f}")
    print(f"Mean absolute difference: {comparison['mean_difference']:.8f}")
    
if __name__ == "__main__":
    main()