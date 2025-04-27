# Chapter 16

## Code

### Pooling

```bash
cd code/pooling

python setup.py build_ext --inplace

python main.py
```

```bash
=== Testing Pooling Implementation ===
Testing Max Pooling...
Max Pooling - Maximum Difference: 0.0

Testing Average Pooling...
Average Pooling - Maximum Difference: 0.0

✓ Tests passed!

=== Performance Benchmarks ===

Input size: [2, 16, 32, 32]
Max Pooling - Custom: 0.058ms, PyTorch: 0.002ms
Avg Pooling - Custom: 0.002ms, PyTorch: 0.001ms

Input size: [8, 64, 128, 128]
Max Pooling - Custom: 82.433ms, PyTorch: 9.454ms
Avg Pooling - Custom: 47.322ms, PyTorch: 3.836ms
```


## Exercises

### Exercise 1

**Implement the forward pass for the pooling layer described in Section 16.2.**

We implement this function in [pooling.c](./code/pooling/pooling.c). See [Pooling](#code-pooling)

```cpp
void poolingLayer_forward(int M, int H, int W, int K, float* Y, float* S, const char* pooling_type) {
    for(int m = 0; m < M; m++)              // for each output feature map
        for(int h = 0; h < H/K; h++)        // for each output element,
            for(int w = 0; w < W/K; w++) {  // this code assumes that H and W
                // Initialize based on pooling type
                if(strcmp(pooling_type, "max") == 0)
                    S[m*(H/K)*(W/K) + h*(W/K) + w] = -FLT_MAX;  // For max pooling
                else
                    S[m*(H/K)*(W/K) + h*(W/K) + w] = 0.0f;      // For avg pooling
                
                // Loop over KxK input window
                for(int p = 0; p < K; p++) {
                    for(int q = 0; q < K; q++) {
                        float val = Y[m*H*W + (K*h + p)*W + (K*w + q)];
                        
                        if(strcmp(pooling_type, "max") == 0) {
                            // Max pooling
                            if(val > S[m*(H/K)*(W/K) + h*(W/K) + w])
                                S[m*(H/K)*(W/K) + h*(W/K) + w] = val;
                        }
                        else {
                            // Average pooling (same as original)
                            S[m*(H/K)*(W/K) + h*(W/K) + w] += val / (K*K);
                        }
                    }
                }
            }
}
```

### Exercise 2

**We used an [N x C x H x W] layout for input and output features. Can we reduce the memory bandwidth by changing it to an [N x H x W x C] layout? What are potential benefits of using a [C x H x W x N] layout?**

### Exercise 3

**Implement the backward pass for the convolutional layer described in Section 16.2.**

```cpp
void convLayer_backward_x_grad(int M, int C, int H_in, int W_in, int K,
                             float* dE_dY, float* W, float* dE_dX) {
   int H_out = H_in - K + 1;
   int W_out = W_in - K + 1;
   for(int c = 0; c < C; c++)
       for(int h = 0; h < H_in; h++)
           for(int w = 0; w < W_in; w++)
               dE_dX[c, h, w] = 0;
               
   for(int m = 0; m < M; m++)
       for(int h = 0; h < H-1; h++)
           for(int w = 0; w < W-1; w++)
               for(int c = 0; c < C; c++)
                   for(int p = 0; p < K; p++)
                       for(int q = 0; q < K; q++)
                           if(h-p >= 0 && w-p >=0 && h-p < H_out && w-p < W_OUT)
                               dE_dX[c, h, w] += dE_dY[m, h-p, w-p] * W[m, c, k-p, k-q];
}
```

### Exercise 4

**Analyze the read access pattern to X in the unroll_Kernel in Fig. 16.18 and show whether the memory reads that are done by adjacent threads can be coalesced.**

```cpp
01  __global__ void
02  unroll_Kernel(int C, int H, int W, int K, float* X, float* X_unroll) {
03      int t = blockIdx.x * blockDim.x + threadIdx.x;
04      int H_out = H - K + 1;
05      int W_out = W - K + 1;
06      // Width of the unrolled input feature matrix
07      int W_unroll = H_out * W_out;
08      if (t < C * W_unroll) {
09          // Channel of the input feature map being collected by the thread
10          int c = t / W_unroll;
11          // Column index of the unrolled matrix to write a strip of
12          // input elements into (also, the linearized index of the output
13          // element for which the thread is collecting input elements)
14          int w_unroll = t % W_unroll;
15          // Horizontal and vertical indices of the output element
16          int h_out = w_unroll / W_out;
17          int w_out = w_unroll % W_out;
18      }
19  }
```