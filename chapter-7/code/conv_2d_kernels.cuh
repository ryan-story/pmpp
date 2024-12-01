#pragma once
#ifndef CONV2D_KERNEL_H
#define CONV2D_KERNEL_H

#include <cuda_runtime.h>

__global__ void conv2d_kernel(float *M, float *F, float *P, int r, int height, int width);
void conv2d(float *M, float *F, float *P, int r, int height, int width);
inline unsigned int cdiv(unsigned int a, unsigned int b);

#endif