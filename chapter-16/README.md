# Chapter 16

## Code

## Exercises

### Exercise 1

**Implement the forward pass for the pooling layer described in Section 16.2.**

### Exercise 2

**We used an [N x C x H x W] layout for input and output features. Can we reduce the memory bandwidth by changing it to an [N x H x W x C] layout? What are potential benefits of using a [C x H x W x N] layout?**

### Exercise 3

**Implement the backward pass for the convolutional layer described in Section 16.2.**

### Exercise 4

**Analyze the read access pattern to X in the unroll_Kernel in Fig. 16.18 and show whether the memory reads that are done by adjacent threads can be coalesced.**