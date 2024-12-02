// nvcc run_conv2d.cu conv2d_functions.cu -o conv2d_program
#include <iostream>
#include <iomanip>
#include "conv2d_functions.cuh"

void printMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(6) << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char const *argv[]) {
    int size = 4;
    float M[size*size] = {
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0
    };
    float P[size*size] = {0.0f};
    int r = 1;
    float gaussian_kernel[(2*r+1)*(2*r+1)] = {
        1/16.f, 2/16.f, 1/16.f,
        2/16.f, 4/16.f, 2/16.f,
        1/16.f, 2/16.f, 1/16.f
    };
    
    conv2d_with_constant_memory(M, gaussian_kernel, P, r, size, size);
    printMatrix(P, size, size);
    
    return 0;
}