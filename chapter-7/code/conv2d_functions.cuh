#pragma once
#ifndef CONV2D_KERNEL_H
#define CONV2D_KERNEL_H

#include <cuda_runtime.h>

#define FILTER_RADIUS 9
#define BLOCK_SIZE 32

// Declare constant memory
__constant__ float constFilter[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];

// Function declarations
inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

__global__ void conv2d_kernel(float *M, float *F, float *P, int r, int height, int width);
__global__ void conv2d_kernel_with_constant_memory(float *M, float *P, int r, int height, int width);

void conv2d_with_constant_memory(float *M, float *F, float *P, int r, int height, int width);

#endif