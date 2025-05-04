# Chapter 17

## Code

## Exercises

### Exercise 1

**Loop fission splits a loop into two loops. Use the F H D code in Fig. 17.4 and enumerate the execution order of the two parts of the outer loop body: (1) the statements before the inner loop and (2) the inner loop.**

#### Original code

```cpp
01  for (int m = 0; m < M; m++) {
02    rMu[m] = rPhi[m]*rD[m] + iPhi[m]*iD[m];
03    iMu[m] = rPhi[m]*iD[m] - iPhi[m]*rD[m];
04    for (int n = 0; n < N; n++) {
05      float expFhD = 2*PI*(kx[m]*x[n] + ky[m]*y[n] + kz[m]*z[n]);
06      float cArg = cos(expFhD);
07      float sArg = sin(expFhD);
08      rFhD[n] += rMu[m]*cArg - iMu[m]*sArg;
09      iFhD[n] += iMu[m]*cArg + rMu[m]*sArg;
10    }
11  }
```

#### Loop fission on the FHD computation.

```cpp
01  for (int m = 0; m < M; m++) {
02    rMu[m] = rPhi[m]*rD[m] + iPhi[m]*iD[m];
03    iMu[m] = rPhi[m]*iD[m] - iPhi[m]*rD[m];
04  }
05  for (int m = 0; m < M; m++) {
06    for (int n = 0; n < N; n++) {
07      float expFhD = 2*PI*(kx[m]*x[n] + ky[m]*y[n] + kz[m]*z[n]);
08      float cArg = cos(expFhD);
09      float sArg = sin(expFhD);
10      rFhD[n] += rMu[m]*cArg - iMu[m]*sArg;
11      iFhD[n] += iMu[m]*cArg + rMu[m]*sArg;
12    }
13  }
```

**a. List the execution order of these parts from different iterations of the outer loop before fission.**

In the original loop, for each iteration of the outer lopp (m from 0 to M-1), the execution happens in the following order:
1. First part (statements before inner loop):
- Calculate rMu[m] (line 02)
- Calculate iMu[m] (line 03)

2. Second part (inner loop):
- For each n from 0 to N-1:
    - Calculate expFhD (line 05)
    - Calculate cArg (line 06)
    - Calculate sArg (line 07)
    - Update rFhD[n] (line 08)
    - Update iFhD[n] (line 09)

The execution order is as follows:
a. `m=0`
- First part (m = 0): lines 02-03
- Second part (m = 0): inner loop lines 05-09 for all n (0 to N-1)

b. `m=1`
- First part (m = 1): lines 02-03
- Second part (m = 1): inner loop lines 05-09 for all n (0 to N-1)

c. `m=2`
- First part (`m = 2`): lines 02-03
- Second part (`m = 2`): inner loop lines 05-09 for all n (0 to N-1)

... and so on until `m = M -1`

**b. List the execution order of these parts from the two loops after fission.**

1. First loop (lines 01-04): All `rMu` and `iMu` calculations are completed first
- `m=0`: Calculate `rMu[0]` and `iMu[0]`
- `m=1`: Calculate `rMu[1]` and `iMu[1]`
- `m=2`: Calculate `rMu[2]` and `iMu[2]`

... and so on until `m=M-1`

2. Second loop (lines 05-13): All inner loop calculations are completed after all `rMu/iMu` values are available
- `m=0`: Run inner loop for all `n` values (using pre-calculated `rMu[0]` and `iMu[0]`)
- `m=1`: Run inner loop for all `n` values (using pre-calculated `rMu[1]` and `iMu[1]`)
- `m=2`: Run inner loop for all `n` values (using pre-calculated `rMu[2]` and `iMu[2]`)

...and so on until m=M-1

Overall the big change is that all of the `rMu/iMu` values are already pre-computed when we execute the inner loop. 

**c. Determine whether the execution results in parts (a) and (b) of this exercise will be identical. The execution results are identical if all data required by a part are properly generated and preserved for its consumption before that part executes and the execution result of the part is not overwritten by other parts that should come after the part in the original execution order.**

To answer this question we need to analyze two aspects: 
1. Are all data required by the inner loop are properly generated and preserved before the inner loop execution starts?
2. Are the results of the inner loop not overwritten in the inner looop. 

Let's try to answer this:
- 1: To calculate the second part of the fission code (lines `05-12`) we require the values of `rMu[m]` and `iMu[m]` to be already pre-calculated. This happens in the first part of the fission cde (lines `01-04`). So the first **requirement is satisfied**
- 2: We don't overwrite values fo `rMu` and `iMu anywhere. The only override is (+=) on rFhD[n] and iFhD[n] in lines 10-11 and it happens in exactly the same order as in the orignal code, so **this requirement is satisfied as well.**

Hence we can determine that yes the result of `a and b part of this exericse will be identical`.

### Exercise 2

**Loop interchange swaps the inner loop into the outer loop and vice versa. Use the loops from Fig. 17.9 and enumerate the execution order of the instances of loop body before and after the loop exchange.**

**a. List the execution order of the loop body from different iterations before loop interchange. Identify these iterations with the values of m and n.** 

**b. List the execution order of the loop body from different iterations after loop interchange. Identify these iterations with the values of m and n.**

**c. Determine whether the execution results in parts (a) and (b) of this exercise will be identical. The execution results are identical if all data required by a part are properly generated and preserved for its consumption before that part executes and the execution result of the part is not overwritten by other parts that should come after the part in the original execution order.**

### Exercise 3

**In Fig. 17.11, identify the difference between the access to x[] and kx[] in the nature of indices used. Use the difference to explain why it does not make sense to try to load kx[n] into a register for the kernel shown in Fig. 17.11**





