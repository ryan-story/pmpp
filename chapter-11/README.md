# Chapter 11

## Code

We implement all of the kernels described in this chapter. This includes being

- Kogge-Stone scan
- Kogge-Stone with double buffering scan
- Brent-Kung scan
- Three-phase scan

The above implementations are located in [scan.cu](code/scan.cu). To execute the code, run:

```bash
nvcc scan.cu -o scan
./scan
```

We also implemented the hierarchical kernel, capable of processing arrays of arbitrary length, and we benchmarked it against the traditional sequential scan kernel. Additionally, we implemented a kernel with a domino-like synchronization mechanism between the blocks. The implementation can be found in [hierarchical_scan.cu]. (code/hierarchical_scan.cu)

To execute the code, run:

```bash
nvcc hierarchical_scan.cu -o hierarchical_scan
./hierarchical_scan
```

The output should resemble this:

```bash
Benchmarking Scan Operations
----------------------------
Size       Sequential(ms)  Hierarchical(ms) Domino(ms)      Speedup-H  Speedup-D  Match   
16384      0.002           0.135           0.135           0.01       0.01       ✓✓
65536      0.088           0.196           0.220           0.45       0.40       ✓✓
262144     0.790           0.521           0.589           1.52       1.34       ✓✓
1048576    3.537           1.582           2.114           2.24       1.67       ✓✓
```

## Exercises


### Exercise 1
**Consider the following array: [4 6 7 1 2 8 5 2]. Perform a parallel inclusive prefix scan on the array, using the Kogge-Stone algorithm. Report the intermediate states of the array after each step.**

### Exercise 2
**Modify the Kogge-Stone parallel scan kernel in Fig. 11.3 to use double-buffering instead of a second call to `__syncthreads()` to overcome the write-after-read race condition.**

### Exercise 3
**Analyze the Kogge-Stone parallel scan kernel in Fig. 11.3. Show that control divergence occurs only in the first warp of each block for stride values up to half of the warp size. That is, for warp size 32, control divergence will occur 5 iterations for stride values 1, 2, 4, 8, and 16.**

### Exercise 4
**For the Kogge-Stone scan kernel based on reduction trees, assume that we have 2048 elements. Which of the following gives the closest approximation of how many add operations will be performed?**

### Exercise 5
**Consider the following array: [4 6 7 1 2 8 5 2]. Perform a parallel inclusive prefix scan on the array, using the Brent-Kung algorithm. Report the intermediate states of the array after each step.**

### Exercise 6
**For the Brent-Kung scan kernel, assume that we have 2048 elements. How many add operations will be performed in both the reduction tree phase and the inverse reduction tree phase?**

### Exercise 7
**Use the algorithm in Fig. 11.4 to complete an exclusive scan kernel.**

### Exercise 8
**Complete the host code and all three kernels for the segmented parallel scan algorithm in Fig. 11.9.**