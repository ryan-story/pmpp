// nvcc -o basic_conv2d_with_constant_memory basic_conv2d_with_constant_memory.cu

#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>
#include <stdio.h>

#define BLOCK_SIZE 4
#define FILTER_RADIUS 1

__constant__ float constFilter[(2*FILTER_RADIUS+1)][(2*FILTER_RADIUS+1)];

void printMatrix(float *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            std::cout << std::setw(6) << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

__global__ void conv2d_kernel_with_constant_memory(float *M, float *P, int r, int height, int width){
    //we run it independently for each output pixel
    int outRow = blockDim.y * blockIdx.y + threadIdx.y;
    int outCol = blockDim.x * blockIdx.x + threadIdx.x;

    float Pvalue = 0.0f;

    for (int Frow=0; Frow < r*2+1; Frow++){
        for (int Fcol=0; Fcol < r*2+1; Fcol++){
            int inRow = outRow - r + Frow;
            int inCol = outCol - r + Fcol;

            if((inRow >= 0) && (inRow < height) && (inCol >=0) && (inCol < width))
                Pvalue += M[inRow * width + inCol] * constFilter[Frow][Fcol];
        }
    }

    if((outRow < height) && (outCol < width))
        // printf("Thread (%d,%d): Pvalue = %f\n", outRow, outCol, Pvalue);
        P[outRow*width + outCol] = Pvalue;  
}

void conv2d(float *M, float *F, float *P, int r, int height, int width){

    float *d_M, *d_P;

    cudaMalloc((void**)&d_M, width * height * sizeof(float));
    cudaMalloc((void**)&d_P, width * height *sizeof(float));

    cudaMemcpy(d_M, M, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P, M, width * height * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(constFilter, F, (2*r+1) * (2*r+1) * sizeof(float));

    dim3 dimBloc(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(cdiv(width, dimBloc.x), cdiv(height, dimBloc.y));

    conv2d_kernel_with_constant_memory<<<dimGrid, dimBloc>>>(d_M, d_P, r, height, width);
    cudaMemcpy(P, d_P, width * height *sizeof(float), cudaMemcpyDeviceToHost);
}


int main(int argc, char const *argv[])
{
    int size = 4;
    float M[size*size] = {1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0
    };

    float P[size*size] = {0.0f};

    int r = FILTER_RADIUS;
    // float gaussian_kernel[(2*r+1)*(2*r+1)] = {
    //     1/16.f, 2/16.f, 1/16.f,
    //     2/16.f, 4/16.f, 2/16.f,
    //     1/16.f, 2/16.f, 1/16.f
    // };

    float identity_kernel[(2*r+1)*(2*r+1)] = {
    0, 0, 0,
    0, 1, 0,
    0, 0, 0
    };

    // float *M, float *F, float *P, int r, int height, int width
    conv2d(M, identity_kernel, P, r, size, size);

    printMatrix(P, size, size);

}
