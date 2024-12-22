### Exercise 1

**Calculate the P[0] value in Fig. 7.3.**

`x = [0, 0, 8, 2, 5]` `*` `f = [1, 3, 5, 3, 1]` = `[0, 0, 40, 6, 5]` -> `51`

### Exercise 2

**Consider performing a 1D convolution on array N = {4,1,3,2,3} with filter F = {2,1,4}. What is the resulting output array?**

`N_with_ghost_cells = [0, 4, 1, 3, 2, 3, 0]`

`P[0]` = `[0, 4, 1] * [2, 1, 4] = [0, 4, 4] -> 8`
`P[1]` = `[4, 1, 3] * [2, 1, 4] = [8, 1, 12] -> 21`
`P[2]` = `[1, 3, 2] * [2, 1, 4] = [2, 3, 8] -> 13`
`P[3]` = `[3, 2, 3] * [2, 1, 4] = [6, 2, 12] -> 20`
`P[4]` = `[2, 3, 0] * [2, 1, 4] = [4, 3, 0] -> 7`

`P = [8, 21, 13, 20, 7]`

### Exercise 3

**What do you think the following 1D convolution filters are doing?**

**a. [0 1 0]**

A spike detector or peak detector, it isolates each signal from its neighbours. The output signal is effectively the same as the input signal.

**b. [0 0 1]**

[0 0 1] is a right shift kernel - it shifts the entire signal one position to the right. 

**c. [1 0 0]**

[0 0 1] is a left shift kernel - it shifts the entire signal one position to the left. 

**d. [-1/2 0 1/2]**

It is an edge detection kernel - it "fires" is there is a rapid value change between the neigbouring cells. If the cells are similar the values cancel out and it is 0. 

**e. [1/3 1/3 1/3]**

Signal averages the value based on the cell neighbours - we relace the value with the average of its neighborhood. This smoothing effect reduces noise and local variations in the signal by averaging each value with its immediate neighbors.

### Exercise 4

**Consider performing a 1D convolution on an array of size N with a filter of size M:**

**a. How many ghost cells are there in total?** 

The filter is of size `M` or in other words of size `2r + 1`. Meaning `r=(M-1)/2`. There are `r` ghost cells on the left of the array `N` and `r` ghost cells on the right of the array `N` - `2r`in total or `(M-1)/2 * 2 = (M-1)` ghost cells.  

**b. How many multiplications are performed if ghost cells are treated as multiplications (by 0)?**

For each element of the array we perform `M` mutiplications, there are `N` elements in the array so `M*N` multiplications. 

**c. How many multiplications are performed if ghost cells are not treated as multiplications?**

This gets slightly more tricky to calculate. As we showed above, if the ghost cells are treated as multiplication, we have `NxM` multiplications, but now, for the first element of an array, we "lose" `r` multiplications (two on the left); for the second element, we lose `r-1` multiplications; for the third, `r-2`, etc., all the way to `r-r=0` for the `r+1`th element. On the other hand, we have the same situation. So overall we have

`NxM - (r + r-1 + r-2 + ... + r-r) x 2`. This is a sum of an arithmetic sequence where:

- First term (a₁) = r
- Last term (aₙ) = 0
- Number of terms = r + 1
- Common difference (d) = -1

We can apply the following formula to arithmetic sequences:
Sum = n(a₁ + aₙ)/2, where n is the number of terms. Substituting our values: Sum = (r + 1)(r + 0)/2 = (r + 1)(r)/2 = r(r + 1)/2 on the left and the same on the right. 

So in total, `NxM - (r(r+1)/2) x 2 =  NxM - r(r+1)` multiplications.

This formula makes sense because when `M = 1` (filter size = 1), `r = 0`, so we get NxM multiplications as expected



### Exercise 5
**Consider performing a 2D convolution on a square matrix of size N × N with a square filter of size M × M:**

**a. How many ghost cells are there in total?**

The filter is of size `M` in each direction or in other worlds `M = 2r + 1`. `r = (M-1)/2`. There are `Nxr` ghost cells at each of 4 sides of the matix, so `4xNxr` in total, plus there is `rxr` ghost cells at each corner. So overall there are `4r x (N + r)` ghost cells in total.


**b. How many multiplications are performed if ghost cells are treated as multiplications (by 0)?**

There are `NxN` elements in the matrix. For each element we multiply the filter values by the element and its 
neighbours. Each filter is `MxM` so there is in total `NxN x MxM` multiplications. 


**c. How many multiplications are performed if ghost cells are not treated as multiplications?**

### Exercise 6
**Consider performing a 2D convolution on a rectangular matrix of size N₁ × N₂ with a rectangular mask of size M₁ × M₂:**

**a. How many ghost cells are there in total?**

**b. How many multiplications are performed if ghost cells are treated as multiplications (by 0)?**

**c. How many multiplications are performed if ghost cells are not treated as multiplications?**

### Exercise 7
**Consider performing a 2D tiled convolution with the kernel shown in Fig. 7.12 on an array of size N × N with a filter of size M × M using an output tile of size T × T:**

**a. How many thread blocks are needed?**

**b. How many threads are needed per block?**

**c. How much shared memory is needed per block?**

**d. Repeat the same questions if you were using the kernel in Fig. 7.15.**


### Exercise 8
**Revise the 2D kernel in Fig. 7.7 to perform 3D convolution.**


### Exercise 9
**Revise the 2D kernel in Fig. 7.9 to perform 3D convolution.**


### Exercise 10
**Revise the tiled 2D kernel in Fig. 7.12 to perform 3D convolution.**


