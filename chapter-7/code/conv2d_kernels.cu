#define DEFINE_CONSTANT_MEMORY
#include "conv2d_kernels.cuh"
#include <stdio.h>

__global__ void conv2d_kernel(float *M, float *F, float *P, int r, int height, int width) {
    int outRow = blockDim.y * blockIdx.y + threadIdx.y;
    int outCol = blockDim.x * blockIdx.x + threadIdx.x;
    
    float Pvalue = 0.0f;
    for (int Frow = 0; Frow < r*2+1; Frow++) {
        for (int Fcol = 0; Fcol < r*2+1; Fcol++) {
            int inRow = outRow - r + Frow;
            int inCol = outCol - r + Fcol;
            if ((inRow >= 0) && (inRow < height) && (inCol >= 0) && (inCol < width))
                Pvalue += M[inRow * width + inCol] * F[Frow * (2*r+1) + Fcol];
        }
    }
    
    if ((outRow < height) && (outCol < width))
        P[outRow*width + outCol] = Pvalue;
}

cudaError_t initConstFilter(const float* filter, int r) {
    return cudaMemcpyToSymbol(constFilter, filter, (2*r+1) * (2*r+1) * sizeof(float));
}

__global__ void conv2d_kernel_with_constant_memory(float *M, float *P, int r, int height, int width) {
    int outRow = blockDim.y * blockIdx.y + threadIdx.y;
    int outCol = blockDim.x * blockIdx.x + threadIdx.x;

    if (outRow < height && outCol < width) {
        float sum = 0.0f;
        for (int i = -r; i <= r; i++) {
            for (int j = -r; j <= r; j++) {
                int inRow = outRow + i;
                int inCol = outCol + j;
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                    int filterIndex = (i + r) * (2*r + 1) + (j + r);
                    sum += M[inRow * width + inCol] * constFilter[filterIndex];
                }
            }
        }
        P[outRow * width + outCol] = sum;
    }
}