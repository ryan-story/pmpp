import random

import numpy as np
import torch
from cuda_wrappers.cublas_wrapper import cleanup_cublas
from cuda_wrappers.cudnn_wrapper import cleanup_cudnn
from examples.xor_example import xor_example

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


def main():
    try:
        # print("Running MNIST CNN example with CUDA acceleration...")
        # mnist_cnn_example()
        # print("MNIST CNN example completed successfully")
        xor_example()

    finally:
        # Clean up cuBLAS and cuDNN resources
        cleanup_cublas()
        cleanup_cudnn()
        print("Cleaned up cuBLAS and cuDNN resources")


if __name__ == "__main__":
    main()
