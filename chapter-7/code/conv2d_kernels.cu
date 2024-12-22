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


__global__ void tiled_convolution_kernel(float *M, float *P, int r, int height, int width) {
    int row = blockIdx.y * OUT_TILE_SIZE + threadIdx.y - FILTER_RADIUS;
    int col = blockIdx.x * OUT_TILE_SIZE + threadIdx.x - FILTER_RADIUS;

    __shared__ float M_s[IN_TILE_SIZE][IN_TILE_SIZE];

    // printf("(%d %d) Row: %d Col: %d\n", threadIdx.y, threadIdx.x, row, col);

    if (row >= 0 && row < height && col >= 0 && col < width)
        M_s[threadIdx.y][threadIdx.x] = M[row * width + col];
    else {
        M_s[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();

    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 1) {
    //     printf("M_s matrix contents:\n");
    //     for(int i = 0; i < IN_TILE_SIZE; i++) {
    //         for(int j = 0; j < IN_TILE_SIZE; j++) {
    //             printf("%.2f ", M_s[i][j]);
    //         }
    //         printf("\n");
    //     }
    // }

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