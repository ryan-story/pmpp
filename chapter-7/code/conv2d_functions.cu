#include "conv2d_functions.cuh"

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

__global__ void conv2d_kernel_with_constant_memory(float *M, float *P, int r, int height, int width) {
    int outRow = blockDim.y * blockIdx.y + threadIdx.y;
    int outCol = blockDim.x * blockIdx.x + threadIdx.x;
    
    float Pvalue = 0.0f;
    for (int Frow = 0; Frow < r*2+1; Frow++) {
        for (int Fcol = 0; Fcol < r*2+1; Fcol++) {
            int inRow = outRow - r + Frow;
            int inCol = outCol - r + Fcol;
            if ((inRow >= 0) && (inRow < height) && (inCol >= 0) && (inCol < width))
                Pvalue += M[inRow * width + inCol] * constFilter[Frow * (2*r+1) + Fcol];
        }
    }
    
    if ((outRow < height) && (outCol < width))
        P[outRow*width + outCol] = Pvalue;
}