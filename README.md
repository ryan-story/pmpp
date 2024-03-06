# pmpp
Solutions to the Programming Massively Parallel Processors book



# CUDA Exercises and Solutions

This README provides exercises and solutions related to using threads and blocks in CUDA for vector addition operations.

## Exercises

### Exercise 1
**Question:**
If we want to use each thread in a grid to calculate one output element of a vector addition, what would be the expression for mapping the thread/block indices to the data index (i)?

**Choices:**
- A. `i = threadIdx.x + threadIdx.y;`
- B. `i = blockIdx.x + threadIdx.x;`
- C. `i = blockIdx.x * blockDim.x + threadIdx.x;`
- D. `i = blockIdx.x * threadIdx.x;`


**Solution:**
  
<details>
  
**C** it would be i = blockIdx.x * blockDim.x + threadIdx.x We need all three of these, we need `blockIdx.x` to identify the block and the `blockDim.x` to identify how big each block is. Each of the blocks has the same lengt (e.g. 256). So assuming we want to run the kernel for the 128th element of block in block one we would assign i to `1 * 256 + 128 = 384`

</details>

### Exercise 2
**Question:**
Assume that we want to use each thread to calculate two adjacent elements of a vector addition. What would be the expression for mapping the thread/block indices to the data index (i) of the first element to be processed by a thread?

**Choices:**
- A. `i = blockIdx.x * blockDim.x + threadIdx.x * 2;`
- B. `i = blockIdx.x * threadIdx.x * 2;`
- C. `i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;`
- D. `i = blockIdx.x * blockDim.x * 2 + threadIdx.x;`

**Solution:**

**TODO**

### Exercise 3
**Question:**
We want to use each thread to calculate two elements of a vector addition. Each thread block processes 2 * blockDim.x consecutive elements that form two sections. All threads in each block will process a section first, each processing one element. They will then all move to the next section, each processing one element. What would be the correct expression for the data index (i)? processing one element. Assume that variable i should be the index for the first element to be processed by a thread. What would be the expression for mapping the thread/block indices to data index of the first element?

**Solution:**

<details>

</details>




### Exercise 4

**Question:**

For a vector addition, assume that the vector length is 8000, each thread calculates one output element, and the thread block size is 1024 threads. The programmer configures the kernel call to have a minimum number of thread blocks to cover all output elements. How many threads will be in the grid?



**Choices:**

- A. 8000

- B. 8196

- C. 8192

- D. 8200



**Solution:**

<details>



</details>



### Exercise 5

**Question:**

If we want to allocate an array of v integer elements in the CUDA device global memory, what would be an appropriate expression for the second argument of the `cudaMalloc` call?



**Choices:**

- A. n

- B. v

- C. n * sizeof(int)

- D. v * sizeof(int)



**Solution:**

<details>



</details>



### Exercise 6

**Question:**

If we want to allocate an array of n floating-point elements and have a floating-point pointer variable A_d to point to the allocated memory, what would be an appropriate expression for the first argument of the `cudaMalloc` call?



**Choices:**

- A. n

- B. (void*) A_d

- C. *A_d

- D. (void**) &A_d



**Solution:**

<details>



</details>



### Exercise 7

**Question:**

If we want to copy 3000 bytes of data from host array A_h (A_h is a pointer to element 0 of the source array) to device array A_d (A_d is a pointer to element 0 of the destination array), what would be an appropriate API call for this data copy in CUDA?



**Choices:**

- A. cudaMemcpy(3000, A_h, A_d, cudaMemcpyHostToDevice);

- B. cudaMemcpy(A_h, A_d, 3000, cudaMemcpyDeviceToHost);

- C. cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);

- D. cudaMemcpy(3000, A_d, A_h, cudaMemcpyHostToDevice);



**Solution:**

<details>



</details>



### Exercise 8

**Question:**

How would one declare a variable err that can appropriately receive the returned value of a CUDA API call?



**Choices:**

- A. int err;

- B. cudaError err;

- C. cudaError_t err;

- D. cudaSuccess_t err;



**Solution:**

<details>



</details>
