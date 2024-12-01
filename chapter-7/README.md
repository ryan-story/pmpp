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

Identity kernel - the output signal is the same as the input signal.

**b. [0 0 1]**

The kernel shifts signal by 1 left.

**c. [1 0 0]**

The kernel shifts signal by 1 right.

**d. [-1/2 0 1/2]**

It is an edge detection kernel - it "fires" is there is a rapid value change between the neigbouring cells. If the cells are similar the values cancel out and it is 0. 

**e. [1/3 1/3 1/3]**

Signal averages the value based on the cell neighbours - we relace the value with the average of its neighborhood. 

### Exercise 4

**Consider performing a 1D convolution on an array of size N with a filter of size M:**

**a. How many ghost cells are there in total?** 

The filter is of size `M` or in other words of size `2r + 1`. Meaning `r=(M-1)/2`. There are `r` ghost cells on the rigth of the array `N` and `r` ghost cells on the right of the array `N` - `2r`in total or `(M-1)/2 * 2 = (M-1)` ghost cells.  

**b. How many multiplications are performed if ghost cells are treated as multiplications (by 0)?**

For each element of the array we perform `M` mutiplications, there are `M` elements in the array so `M*N` multiplications. 

**c. How many multiplications are performed if ghost cells are not treated as multiplications?**



### Exercise 5

### Exercise 6

### Exercise 7

### Exercise 8


