#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/graph_structures.h"
#include "../include/graph_conversions.h"
#include "../include/graph_generators.h"
#include "../include/bfs_sequential.h"
#include "../include/bfs_parallel.h"
#include "../include/device_memory.h"
#include "../include/utils.h"

int main() {
    // Generate a scale-free graph with 10000 vertices and 300 edges per new vertex
    printf("Generating scale-free graph...\n");
    COOGraph scaleFreeCOO = generateScaleFreeGraphCOO(10000, 300);
    CSRGraph scaleFreeCSR = convertCOOtoCSR(scaleFreeCOO);
    
    // Generate a small-world graph with 10000 vertices, 400 neighbors, 0.1 rewiring probability
    printf("Generating small-world graph...\n");
    COOGraph smallWorldCOO = generateSmallWorldGraphCOO(10000, 400, 0.1);
    CSRGraph smallWorldCSR = convertCOOtoCSR(smallWorldCOO);
    
    // Allocate graphs on device
    printf("Allocating graphs on device...\n");
    COOGraph deviceScaleFreeCOO = allocateCOOGraphOnDevice(scaleFreeCOO);
    CSRGraph deviceScaleFreeCSR = allocateCSRGraphOnDevice(scaleFreeCSR);
    COOGraph deviceSmallWorldCOO = allocateCOOGraphOnDevice(smallWorldCOO);
    CSRGraph deviceSmallWorldCSR = allocateCSRGraphOnDevice(smallWorldCSR);
    
    // Run BFS on scale-free graph
    printf("Running BFS on Scale-Free Graph:\n");
    int* seqScaleFree = bfs(scaleFreeCSR, 0);
    
    // Using device-based BFS
    int* parScaleFree = bfsParallelEdgeCentricDevice(deviceScaleFreeCOO, 0);
    compareBFSResults(seqScaleFree, parScaleFree, scaleFreeCSR.numVertices, false);
    
    // Run BFS on small-world graph
    printf("\nRunning BFS on Small-World Graph:\n");
    int* seqSmallWorld = bfs(smallWorldCSR, 0);
    
    // Using device-based BFS
    int* parSmallWorld = bfsParallelEdgeCentricDevice(deviceSmallWorldCOO, 0);
    compareBFSResults(seqSmallWorld, parSmallWorld, smallWorldCSR.numVertices, false);
    
    // Run other BFS implementations on device graphs
    printf("\nRunning Push-based BFS on Scale-Free Graph:\n");
    int* pushBFS = bfsParallelPushVertexCentricDevice(deviceScaleFreeCSR, 0);
    compareBFSResults(seqScaleFree, pushBFS, scaleFreeCSR.numVertices, false);
    
    printf("\nRunning Frontier-based BFS on Small-World Graph:\n");
    int* frontierBFS = bfsParallelFrontierVertexCentricDevice(deviceSmallWorldCSR, 0);
    compareBFSResults(seqSmallWorld, frontierBFS, smallWorldCSR.numVertices, false);
    
    // Clean up result arrays
    free(seqScaleFree);
    free(parScaleFree);
    free(seqSmallWorld);
    free(parSmallWorld);
    free(pushBFS);
    free(frontierBFS);
    
    // Free device graph memory
    freeCSRGraphOnDevice(deviceScaleFreeCSR);
    freeCOOGraphOnDevice(deviceScaleFreeCOO);
    freeCSRGraphOnDevice(deviceSmallWorldCSR);
    freeCOOGraphOnDevice(deviceSmallWorldCOO);
    
    // Free host graph memory
    free(scaleFreeCOO.scr);
    free(scaleFreeCOO.dst);
    free(scaleFreeCOO.values);
    free(scaleFreeCSR.srcPtrs);
    free(scaleFreeCSR.dst);
    free(scaleFreeCSR.values);
    free(smallWorldCOO.scr);
    free(smallWorldCOO.dst);
    free(smallWorldCOO.values);
    free(smallWorldCSR.srcPtrs);
    free(smallWorldCSR.dst);
    free(smallWorldCSR.values);
    
    return 0;
}