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

**b. List the execution order of these parts from the two loops after fission.**

**c. Determine whether the execution results in parts (a) and (b) of this exercise will be identical. The execution results are identical if all data required by a part are properly generated and preserved for its consumption before that part executes and the execution result of the part is not overwritten by other parts that should come after the part in the original execution order.**

### Exercise 2

**Loop interchange swaps the inner loop into the outer loop and vice versa. Use the loops from Fig. 17.9 and enumerate the execution order of the instances of loop body before and after the loop exchange.**

**a. List the execution order of the loop body from different iterations before loop interchange. Identify these iterations with the values of m and n.** 

**b. List the execution order of the loop body from different iterations after loop interchange. Identify these iterations with the values of m and n.**

**c. Determine whether the execution results in parts (a) and (b) of this exercise will be identical. The execution results are identical if all data required by a part are properly generated and preserved for its consumption before that part executes and the execution result of the part is not overwritten by other parts that should come after the part in the original execution order.**

### Exercise 3

**In Fig. 17.11, identify the difference between the access to x[] and kx[] in the nature of indices used. Use the difference to explain why it does not make sense to try to load kx[n] into a register for the kernel shown in Fig. 17.11**





