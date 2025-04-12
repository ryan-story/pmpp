#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/graph_structures.h"
#include "../include/bfs_parallel.h"
#include "../include/device_memory.h"

// KERNEL DEFINITIONS
__global__ void bsf_push_vertex_centric_kernel(CSRGraph graph, int* levels, int* newVertexVisitd,
                                               unsigned int currLevel) {
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < graph.numVertices) {
        if (levels[vertex] == currLevel - 1) {
            // iterate over all the vertices at the current level
            for (unsigned int edge = graph.srcPtrs[vertex]; edge < graph.srcPtrs[vertex + 1]; edge++) {
                unsigned int neigbour = graph.dst[edge];
                // not yet visited
                if (levels[neigbour] == -1) {
                    levels[neigbour] = currLevel;
                    *newVertexVisitd = 1;  // set the flag to 1 so we know we need to run it again
                }
            }
        }
    }
}

__global__ void bsf_pull_vertex_centric_kernel(CSCGraph graph, int* levels, int* newVertexVisitd,
                                               unsigned int currLevel) {
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;

    if (vertex < graph.numVertices) {
        if (levels[vertex] == -1) {  // Vertext not yet visited
            for (unsigned int edge = graph.dstPtrs[vertex]; edge < graph.dstPtrs[vertex + 1]; edge++) {
                unsigned int neighbour = graph.src[edge];

                if (levels[neighbour] == currLevel - 1) {
                    levels[vertex] = currLevel;
                    *newVertexVisitd = 1;
                    break;  // if any of the neighbour at the prev level we reached our point
                }
            }
        }
    }
}

__global__ void bsf_edge_centric_kernel(COOGraph cooGraph, int* levels, int* newVertexVisitd, unsigned int currLevel) {
    unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;

    if (edge < cooGraph.numEdges) {
        unsigned int vertex = cooGraph.scr[edge];
        if (levels[vertex] == currLevel - 1) {
            unsigned int neighbour = cooGraph.dst[edge];
            if (levels[neighbour] == -1) {  // neighbour not yet visited
                levels[neighbour] = currLevel;
                *newVertexVisitd = 1;
            }
        }
    }
}

__global__ void bsf_frontier_vertex_centric_kernel(CSRGraph csrGraph, int* levels, int* prevFrontier, int* currFrontier,
                                                   int numPrevFrontier, int* numCurrentFrontier,
                                                   unsigned int currLevel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; edge++) {
            unsigned int neighbour = csrGraph.dst[edge];
            // neighbour not visited
            if (atomicCAS(&levels[neighbour], -1, currLevel) == -1) {
                unsigned int currFrontierIdx = atomicAdd(numCurrentFrontier, 1);
                currFrontier[currFrontierIdx] = neighbour;
            }
        }
    }
}

// ORIGINAL HOST-BASED IMPLEMENTATIONS

int* bfsParallelPushVertexCentric(const CSRGraph& hostGraph, int startingNode) {
    // Initialize host levels array
    int* hostLevels = (int*)malloc(sizeof(int) * hostGraph.numVertices);
    for (int i = 0; i < hostGraph.numVertices; i++) {
        hostLevels[i] = -1;
    }
    hostLevels[startingNode] = 0;

    // Calculate size needed for graph arrays
    size_t scrPtrsSize = sizeof(int) * (hostGraph.numVertices + 1);
    size_t dstSize = sizeof(int) * hostGraph.srcPtrs[hostGraph.numVertices];
    size_t vertexSize = sizeof(int) * hostGraph.numVertices;

    // Allocate device memory
    int *d_srcPtrs, *d_dst, *d_values, *d_levels, *d_newVertexVisited;
    cudaMalloc(&d_srcPtrs, scrPtrsSize);
    cudaMalloc(&d_dst, dstSize);
    cudaMalloc(&d_values, dstSize);
    cudaMalloc(&d_levels, vertexSize);
    cudaMalloc(&d_newVertexVisited, sizeof(int));

    // Copy values to cuda
    cudaMemcpy(d_srcPtrs, hostGraph.srcPtrs, scrPtrsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, hostGraph.dst, dstSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, hostGraph.values, dstSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_levels, hostLevels, vertexSize, cudaMemcpyHostToDevice);

    CSRGraph deviceGraph = {
        .srcPtrs = d_srcPtrs, .dst = d_dst, .values = d_values, .numVertices = hostGraph.numVertices};

    int threadsPerBlock = 256;
    int blocksPerGrid = (hostGraph.numVertices + threadsPerBlock - 1) / threadsPerBlock;

    int currLevel = 1;
    int hostNewVertexVisited = 1;

    // we iterate over the levels as long as any vertex reports finding a new unvisited neighbour
    while (hostNewVertexVisited != 0) {
        hostNewVertexVisited = 0;
        cudaMemcpy(d_newVertexVisited, &hostNewVertexVisited, sizeof(int), cudaMemcpyHostToDevice);

        bsf_push_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(deviceGraph, d_levels, d_newVertexVisited,
                                                                           currLevel);

        cudaDeviceSynchronize();

        cudaMemcpy(&hostNewVertexVisited, d_newVertexVisited, sizeof(int), cudaMemcpyDeviceToHost);
        currLevel++;
    }

    cudaMemcpy(hostLevels, d_levels, vertexSize, cudaMemcpyDeviceToHost);

    cudaFree(d_srcPtrs);
    cudaFree(d_dst);
    cudaFree(d_values);
    cudaFree(d_levels);
    cudaFree(d_newVertexVisited);

    return hostLevels;
}

int* bfsParallelPullVertexCentric(const CSCGraph& hostGraph, int startingNode) {
    // Initialize host levels array
    int* hostLevels = (int*)malloc(sizeof(int) * hostGraph.numVertices);
    for (int i = 0; i < hostGraph.numVertices; i++) {
        hostLevels[i] = -1;
    }
    hostLevels[startingNode] = 0;

    // Calculate size needed for graph arrays
    size_t dstPtrsSize = sizeof(int) * (hostGraph.numVertices + 1);
    size_t srcSize =
        sizeof(int) * hostGraph.dstPtrs[hostGraph.numVertices];  // look where the last dstPtrs is poinintg - than we
                                                                 // know how many scr elements there are
    size_t vertexSize = sizeof(int) * hostGraph.numVertices;

    // Allocate device memory
    int *d_dstPtrs, *d_src, *d_values, *d_levels, *d_newVertexVisited;
    cudaMalloc(&d_dstPtrs, dstPtrsSize);
    cudaMalloc(&d_src, srcSize);
    cudaMalloc(&d_values, srcSize);
    cudaMalloc(&d_levels, vertexSize);
    cudaMalloc(&d_newVertexVisited, sizeof(int));

    // Copy values to cuda
    cudaMemcpy(d_dstPtrs, hostGraph.dstPtrs, dstPtrsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_src, hostGraph.src, srcSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, hostGraph.values, srcSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_levels, hostLevels, vertexSize, cudaMemcpyHostToDevice);

    CSCGraph deviceGraph = {
        .dstPtrs = d_dstPtrs, .src = d_src, .values = d_values, .numVertices = hostGraph.numVertices};

    int threadsPerBlock = 256;
    int blocksPerGrid = (hostGraph.numVertices + threadsPerBlock - 1) / threadsPerBlock;

    int currLevel = 1;
    int hostNewVertexVisited = 1;

    // we iterate over the levels as long as any vertex reports finding a new unvisited neighbour
    while (hostNewVertexVisited != 0) {
        hostNewVertexVisited = 0;
        cudaMemcpy(d_newVertexVisited, &hostNewVertexVisited, sizeof(int), cudaMemcpyHostToDevice);

        bsf_pull_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(deviceGraph, d_levels, d_newVertexVisited,
                                                                           currLevel);

        cudaDeviceSynchronize();

        cudaMemcpy(&hostNewVertexVisited, d_newVertexVisited, sizeof(int), cudaMemcpyDeviceToHost);
        currLevel++;
    }

    cudaMemcpy(hostLevels, d_levels, vertexSize, cudaMemcpyDeviceToHost);

    cudaFree(d_dstPtrs);
    cudaFree(d_src);
    cudaFree(d_values);
    cudaFree(d_levels);
    cudaFree(d_newVertexVisited);

    return hostLevels;
}

int* bfsParallelEdgeCentric(const COOGraph& hostGraph, int startingNode) {
    // Initialize host levels array
    int* hostLevels = (int*)malloc(sizeof(int) * hostGraph.numVertices);
    for (int i = 0; i < hostGraph.numVertices; i++) {
        hostLevels[i] = -1;
    }
    hostLevels[startingNode] = 0;

    // Calculate size needed for graph arrays
    size_t edgeArraysSize = sizeof(int) * hostGraph.numEdges;
    size_t vertexSize = sizeof(int) * hostGraph.numVertices;

    // Allocate device memory
    int *d_src, *d_dst, *d_values, *d_levels, *d_newVertexVisited;
    cudaMalloc(&d_src, edgeArraysSize);
    cudaMalloc(&d_dst, edgeArraysSize);
    cudaMalloc(&d_values, edgeArraysSize);
    cudaMalloc(&d_levels, vertexSize);
    cudaMalloc(&d_newVertexVisited, sizeof(int));

    // Copy values to cuda
    cudaMemcpy(d_src, hostGraph.scr, edgeArraysSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, hostGraph.dst, edgeArraysSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, hostGraph.values, edgeArraysSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_levels, hostLevels, vertexSize, cudaMemcpyHostToDevice);

    COOGraph deviceGraph = {.scr = d_src,
                            .dst = d_dst,
                            .values = d_values,
                            .numEdges = hostGraph.numEdges,
                            .numVertices = hostGraph.numVertices};

    int threadsPerBlock = 256;
    int blocksPerGrid = (hostGraph.numEdges + threadsPerBlock - 1) / threadsPerBlock;

    int currLevel = 1;
    int hostNewVertexVisited = 1;

    // we iterate over the levels as long as any vertex reports finding a new unvisited neighbour
    while (hostNewVertexVisited != 0) {
        hostNewVertexVisited = 0;
        cudaMemcpy(d_newVertexVisited, &hostNewVertexVisited, sizeof(int), cudaMemcpyHostToDevice);

        bsf_edge_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(deviceGraph, d_levels, d_newVertexVisited,
                                                                    currLevel);

        cudaDeviceSynchronize();

        cudaMemcpy(&hostNewVertexVisited, d_newVertexVisited, sizeof(int), cudaMemcpyDeviceToHost);
        currLevel++;
    }

    cudaMemcpy(hostLevels, d_levels, vertexSize, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_values);
    cudaFree(d_levels);
    cudaFree(d_newVertexVisited);

    return hostLevels;
}

int* bfsParallelFrontierVertexCentric(const CSRGraph& hostGraph, int startingNode) {
    // Initialize host levels array
    int* hostLevels = (int*)malloc(sizeof(int) * hostGraph.numVertices);
    for (int i = 0; i < hostGraph.numVertices; i++) {
        hostLevels[i] = -1;
    }
    hostLevels[startingNode] = 0;

    // Initialize the first frontier hosting just the starting node
    int* hostPrevFrontier = (int*)malloc(sizeof(int) * hostGraph.numVertices);
    hostPrevFrontier[0] = startingNode;
    int hostNumPrevFrontier = 1;

    // Allocate memory for the current frontier
    int* hostCurrFrontier = (int*)malloc(sizeof(int) * hostGraph.numVertices);
    int hostNumCurrFrontier = 0;

    // Calculate size needed for graph arrays
    size_t scrPtrsSize = sizeof(int) * (hostGraph.numVertices + 1);
    size_t dstSize = sizeof(int) * hostGraph.srcPtrs[hostGraph.numVertices];
    size_t vertexSize = sizeof(int) * hostGraph.numVertices;

    // Allocate device memory
    int *d_srcPtrs, *d_dst, *d_values, *d_levels;
    int *d_prevFrontier, *d_currFrontier, *d_numCurrFrontier;

    cudaMalloc(&d_srcPtrs, scrPtrsSize);
    cudaMalloc(&d_dst, dstSize);
    cudaMalloc(&d_values, dstSize);
    cudaMalloc(&d_levels, vertexSize);
    cudaMalloc(&d_prevFrontier, vertexSize);
    cudaMalloc(&d_currFrontier, vertexSize);
    cudaMalloc(&d_numCurrFrontier, sizeof(int));

    // Copy graph structure to device
    cudaMemcpy(d_srcPtrs, hostGraph.srcPtrs, scrPtrsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, hostGraph.dst, dstSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, hostGraph.values, dstSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_levels, hostLevels, vertexSize, cudaMemcpyHostToDevice);

    CSRGraph deviceGraph = {
        .srcPtrs = d_srcPtrs, .dst = d_dst, .values = d_values, .numVertices = hostGraph.numVertices};

    int threadsPerBlock = 256;
    int blocksPerGrid = (hostGraph.numVertices + threadsPerBlock - 1) / threadsPerBlock;

    int currLevel = 1;

    // Continue BFS as long as there are vertices in the frontier
    while (hostNumPrevFrontier > 0) {
        // Copy frontier data to device
        cudaMemcpy(d_prevFrontier, hostPrevFrontier, sizeof(int) * hostNumPrevFrontier, cudaMemcpyHostToDevice);

        // Reset the current frontier counter
        hostNumCurrFrontier = 0;
        cudaMemcpy(d_numCurrFrontier, &hostNumCurrFrontier, sizeof(int), cudaMemcpyHostToDevice);

        // Calculate grid dimensions based on frontier size
        int threadsPerBlock = 256;
        int blocksPerGrid = (hostNumPrevFrontier + threadsPerBlock - 1) / threadsPerBlock;

        // Launch kernel to process the frontier
        bsf_frontier_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            deviceGraph, d_levels, d_prevFrontier, d_currFrontier, hostNumPrevFrontier, d_numCurrFrontier, currLevel);

        cudaDeviceSynchronize();

        // Get the size of the new frontier
        cudaMemcpy(&hostNumCurrFrontier, d_numCurrFrontier, sizeof(int), cudaMemcpyDeviceToHost);

        // Check if the new frontier is empty
        if (hostNumCurrFrontier > 0) {
            // Copy the new frontier back to host
            cudaMemcpy(hostCurrFrontier, d_currFrontier, sizeof(int) * hostNumCurrFrontier, cudaMemcpyDeviceToHost);

            // Swap frontiers for next iteration
            int* tempFrontier = hostPrevFrontier;
            hostPrevFrontier = hostCurrFrontier;
            hostCurrFrontier = tempFrontier;

            hostNumPrevFrontier = hostNumCurrFrontier;
        } else {
            // No more vertices to explore
            hostNumPrevFrontier = 0;
        }

        currLevel++;
    }

    // Copy final levels back to host
    cudaMemcpy(hostLevels, d_levels, vertexSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_srcPtrs);
    cudaFree(d_dst);
    cudaFree(d_values);
    cudaFree(d_levels);
    cudaFree(d_prevFrontier);
    cudaFree(d_currFrontier);
    cudaFree(d_numCurrFrontier);

    // Free host frontier memory
    free(hostPrevFrontier);
    free(hostCurrFrontier);

    return hostLevels;
}

// NEW DEVICE-BASED IMPLEMENTATIONS

int* bfsParallelPushVertexCentricDevice(const CSRGraph& deviceGraph, int startingNode) {
    // Create device levels array
    int* d_levels = allocateAndInitLevelsOnDevice(deviceGraph.numVertices, startingNode);
    
    // Allocate new vertex visited flag
    int* d_newVertexVisited;
    cudaMalloc(&d_newVertexVisited, sizeof(int));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (deviceGraph.numVertices + threadsPerBlock - 1) / threadsPerBlock;
    
    int currLevel = 1;
    int hostNewVertexVisited = 1;
    
    // BFS iteration loop
    while (hostNewVertexVisited != 0) {
        hostNewVertexVisited = 0;
        cudaMemcpy(d_newVertexVisited, &hostNewVertexVisited, sizeof(int), cudaMemcpyHostToDevice);
        
        bsf_push_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            deviceGraph, d_levels, d_newVertexVisited, currLevel);
        
        cudaDeviceSynchronize();
        
        cudaMemcpy(&hostNewVertexVisited, d_newVertexVisited, sizeof(int), cudaMemcpyDeviceToHost);
        currLevel++;
    }
    
    // Copy results to host
    int* hostLevels = copyLevelsToHost(d_levels, deviceGraph.numVertices);
    
    // Free device memory
    cudaFree(d_levels);
    cudaFree(d_newVertexVisited);
    
    return hostLevels;
}

int* bfsParallelPullVertexCentricDevice(const CSCGraph& deviceGraph, int startingNode) {
    // Create device levels array
    int* d_levels = allocateAndInitLevelsOnDevice(deviceGraph.numVertices, startingNode);
    
    // Allocate new vertex visited flag
    int* d_newVertexVisited;
    cudaMalloc(&d_newVertexVisited, sizeof(int));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (deviceGraph.numVertices + threadsPerBlock - 1) / threadsPerBlock;
    
    int currLevel = 1;
    int hostNewVertexVisited = 1;
    
    // BFS iteration loop
    while (hostNewVertexVisited != 0) {
        hostNewVertexVisited = 0;
        cudaMemcpy(d_newVertexVisited, &hostNewVertexVisited, sizeof(int), cudaMemcpyHostToDevice);
        
        bsf_pull_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            deviceGraph, d_levels, d_newVertexVisited, currLevel);
        
        cudaDeviceSynchronize();
        
        cudaMemcpy(&hostNewVertexVisited, d_newVertexVisited, sizeof(int), cudaMemcpyDeviceToHost);
        currLevel++;
    }
    
    // Copy results to host
    int* hostLevels = copyLevelsToHost(d_levels, deviceGraph.numVertices);
    
    // Free device memory
    cudaFree(d_levels);
    cudaFree(d_newVertexVisited);
    
    return hostLevels;
}

int* bfsParallelEdgeCentricDevice(const COOGraph& deviceGraph, int startingNode) {
    // Create device levels array
    int* d_levels = allocateAndInitLevelsOnDevice(deviceGraph.numVertices, startingNode);
    
    // Allocate new vertex visited flag
    int* d_newVertexVisited;
    cudaMalloc(&d_newVertexVisited, sizeof(int));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (deviceGraph.numEdges + threadsPerBlock - 1) / threadsPerBlock;
    
    int currLevel = 1;
    int hostNewVertexVisited = 1;
    
    // BFS iteration loop
    while (hostNewVertexVisited != 0) {
        hostNewVertexVisited = 0;
        cudaMemcpy(d_newVertexVisited, &hostNewVertexVisited, sizeof(int), cudaMemcpyHostToDevice);
        
        bsf_edge_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            deviceGraph, d_levels, d_newVertexVisited, currLevel);
        
        cudaDeviceSynchronize();
        
        cudaMemcpy(&hostNewVertexVisited, d_newVertexVisited, sizeof(int), cudaMemcpyDeviceToHost);
        currLevel++;
    }
    
    // Copy results to host
    int* hostLevels = copyLevelsToHost(d_levels, deviceGraph.numVertices);
    
    // Free device memory
    cudaFree(d_levels);
    cudaFree(d_newVertexVisited);
    
    return hostLevels;
}

int* bfsParallelFrontierVertexCentricDevice(const CSRGraph& deviceGraph, int startingNode) {
    // Create device levels array
    int* d_levels = allocateAndInitLevelsOnDevice(deviceGraph.numVertices, startingNode);
    
    // Initialize the first frontier with the starting node
    int* hostPrevFrontier = (int*)malloc(sizeof(int) * deviceGraph.numVertices);
    hostPrevFrontier[0] = startingNode;
    int hostNumPrevFrontier = 1;
    
    // Allocate memory for the current frontier
    int* hostCurrFrontier = (int*)malloc(sizeof(int) * deviceGraph.numVertices);
    int hostNumCurrFrontier = 0;
    
    // Allocate device memory for frontiers
    int *d_prevFrontier, *d_currFrontier, *d_numCurrFrontier;
    cudaMalloc(&d_prevFrontier, sizeof(int) * deviceGraph.numVertices);
    cudaMalloc(&d_currFrontier, sizeof(int) * deviceGraph.numVertices);
    cudaMalloc(&d_numCurrFrontier, sizeof(int));
    
    int currLevel = 1;
    
    // Continue BFS as long as there are vertices in the frontier
    while (hostNumPrevFrontier > 0) {
        // Copy frontier data to device
        cudaMemcpy(d_prevFrontier, hostPrevFrontier, sizeof(int) * hostNumPrevFrontier, cudaMemcpyHostToDevice);
        
        // Reset the current frontier counter
        hostNumCurrFrontier = 0;
        cudaMemcpy(d_numCurrFrontier, &hostNumCurrFrontier, sizeof(int), cudaMemcpyHostToDevice);
        
        // Calculate grid dimensions based on frontier size
        int threadsPerBlock = 256;
        int blocksPerGrid = (hostNumPrevFrontier + threadsPerBlock - 1) / threadsPerBlock;
        
        // Launch kernel to process the frontier
        bsf_frontier_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            deviceGraph, d_levels, d_prevFrontier, d_currFrontier, 
            hostNumPrevFrontier, d_numCurrFrontier, currLevel);
        
        cudaDeviceSynchronize();
        
        // Get the size of the new frontier
        cudaMemcpy(&hostNumCurrFrontier, d_numCurrFrontier, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Check if the new frontier is empty
        if (hostNumCurrFrontier > 0) {
            // Copy the new frontier back to host
            cudaMemcpy(hostCurrFrontier, d_currFrontier, sizeof(int) * hostNumCurrFrontier, cudaMemcpyDeviceToHost);
            
            // Swap frontiers for next iteration
            int* tempFrontier = hostPrevFrontier;
            hostPrevFrontier = hostCurrFrontier;
            hostCurrFrontier = tempFrontier;
            
            hostNumPrevFrontier = hostNumCurrFrontier;
        } else {
            // No more vertices to explore
            hostNumPrevFrontier = 0;
        }
        
        currLevel++;
    }
    
    // Copy results to host
    int* hostLevels = copyLevelsToHost(d_levels, deviceGraph.numVertices);
    
    // Free device memory
    cudaFree(d_levels);
    cudaFree(d_prevFrontier);
    cudaFree(d_currFrontier);
    cudaFree(d_numCurrFrontier);
    
    // Free host frontier memory
    free(hostPrevFrontier);
    free(hostCurrFrontier);
    
    return hostLevels;
}