#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "../include/graph_structures.h"
#include "../include/graph_conversions.h"
#include "../include/graph_generators.h"
#include "../include/bfs_sequential.h"
#include "../include/bfs_parallel.h"
#include "../include/device_memory.h"
#include "../include/utils.h"

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

void clear_l2() {
    static int l2_clear_size = 0;
    static unsigned char* gpu_scratch_l2_clear = NULL;
    if (!gpu_scratch_l2_clear) {
        int dev;
        gpuErrchk(cudaGetDevice(&dev));
        gpuErrchk(cudaDeviceGetAttribute(&l2_clear_size, cudaDevAttrL2CacheSize, dev));
        l2_clear_size *= 2;  // Use a buffer twice the L2 size
        gpuErrchk(cudaMalloc(&gpu_scratch_l2_clear, l2_clear_size));
    }
    gpuErrchk(cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size));
}

typedef int* (*BFSSequentialFunction)(const CSRGraph&, int);
typedef int* (*BFSParallelCSRFunction)(const CSRGraph&, int);
typedef int* (*BFSParallelCSCFunction)(const CSCGraph&, int);
typedef int* (*BFSParallelCOOFunction)(const COOGraph&, int);

float benchmark_sequential_bfs(BFSSequentialFunction bfs_func, const CSRGraph& graph, int start_vertex, 
                            int warmup, int reps) {
    // Warmup runs
    for (int i = 0; i < warmup; i++) {
        int* result = bfs_func(graph, start_vertex);
        free(result);
    }
    
    // Timing runs
    clock_t start_time, end_time;
    double total_time_ms = 0.0;
    
    for (int i = 0; i < reps; i++) {
        start_time = clock();
        int* result = bfs_func(graph, start_vertex);
        end_time = clock();
        
        double elapsed_ms = 1000.0 * (double)(end_time - start_time) / CLOCKS_PER_SEC;
        total_time_ms += elapsed_ms;
        
        free(result);
    }
    
    return total_time_ms / reps;
}

float benchmark_parallel_csr_bfs(BFSParallelCSRFunction bfs_func, const CSRGraph& deviceGraph, 
                               int start_vertex, int warmup, int reps) {
    // Warmup runs
    for (int i = 0; i < warmup; i++) {
        int* result = bfs_func(deviceGraph, start_vertex);
        free(result);
    }
    
    // Timing runs
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    
    float total_time_ms = 0.0f;
    
    for (int i = 0; i < reps; i++) {
        clear_l2();  // Clear L2 cache to reduce inter-run effects
        
        gpuErrchk(cudaEventRecord(start, 0));
        int* result = bfs_func(deviceGraph, start_vertex);
        gpuErrchk(cudaEventRecord(stop, 0));
        gpuErrchk(cudaEventSynchronize(stop));
        
        float elapsed_ms = 0.0f;
        gpuErrchk(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_time_ms += elapsed_ms;
        
        free(result);
    }
    
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
    
    return total_time_ms / reps;
}

float benchmark_parallel_coo_bfs(BFSParallelCOOFunction bfs_func, const COOGraph& deviceGraph, 
                               int start_vertex, int warmup, int reps) {
    // Warmup runs
    for (int i = 0; i < warmup; i++) {
        int* result = bfs_func(deviceGraph, start_vertex);
        free(result);
    }
    
    // Timing runs
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    
    float total_time_ms = 0.0f;
    
    for (int i = 0; i < reps; i++) {
        clear_l2();  // Clear L2 cache to reduce inter-run effects
        
        gpuErrchk(cudaEventRecord(start, 0));
        int* result = bfs_func(deviceGraph, start_vertex);
        gpuErrchk(cudaEventRecord(stop, 0));
        gpuErrchk(cudaEventSynchronize(stop));
        
        float elapsed_ms = 0.0f;
        gpuErrchk(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_time_ms += elapsed_ms;
        
        free(result);
    }
    
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
    
    return total_time_ms / reps;
}

// Main function with benchmarking
int main() {
    // Parameters for benchmarking
    int warmup_runs = 3;
    int timing_runs = 10;
    int start_vertex = 3;
    
    // Graph sizes to benchmark
    int graph_sizes[] = {1000, 5000, 10000, 20000};
    int num_sizes = sizeof(graph_sizes) / sizeof(graph_sizes[0]);
    
    printf("BFS Benchmark Results\n");
    printf("=====================\n\n");

    // Verify correctness on a small graph before benchmarking
    printf("Verifying BFS implementations correctness...\n");
    COOGraph verifyGraphCOO = generateScaleFreeGraphCOO(2000, 50);
    CSRGraph verifyGraphCSR = convertCOOtoCSR(verifyGraphCOO);
    
    // Allocate on device for testing
    COOGraph deviceVerifyGraphCOO = allocateCOOGraphOnDevice(verifyGraphCOO);
    CSRGraph deviceVerifyGraphCSR = allocateCSRGraphOnDevice(verifyGraphCSR);
    
    // Get reference results from sequential BFS
    int* reference_result = bfs(verifyGraphCSR, start_vertex);
    
    // Check push-based vertex-centric BFS
    int* push_result = bfsParallelPushVertexCentricDevice(deviceVerifyGraphCSR, start_vertex);
    bool push_correct = compareBFSResults(reference_result, push_result, verifyGraphCSR.numVertices, false);
    
    // Check edge-centric BFS
    int* edge_result = bfsParallelEdgeCentricDevice(deviceVerifyGraphCOO, start_vertex);
    bool edge_correct = compareBFSResults(reference_result, edge_result, verifyGraphCSR.numVertices, false);
    
    // Check frontier-based BFS
    int* frontier_result = bfsParallelFrontierVertexCentricDevice(deviceVerifyGraphCSR, start_vertex);
    bool frontier_correct = compareBFSResults(reference_result, frontier_result, verifyGraphCSR.numVertices, false);
    
    bool all_correct = push_correct && edge_correct && frontier_correct;
    
    // Clean up verification resources
    free(reference_result);
    free(push_result);
    free(edge_result);
    free(frontier_result);
    freeCSRGraphOnDevice(deviceVerifyGraphCSR);
    freeCOOGraphOnDevice(deviceVerifyGraphCOO);
    free(verifyGraphCOO.scr);
    free(verifyGraphCOO.dst);
    free(verifyGraphCOO.values);
    free(verifyGraphCSR.srcPtrs);
    free(verifyGraphCSR.dst);
    free(verifyGraphCSR.values);
    
    if (!all_correct) {
        printf("ERROR: Some BFS implementations are not producing correct results!\n");
        printf("Please fix implementation errors before running benchmarks.\n");
        return 1; // Exit with error code
    }
    
    printf("All BFS implementations passed correctness verification!\n\n");
    
    for (int i = 0; i < num_sizes; i++) {
        int size = graph_sizes[i];
        printf("Graph Size: %d vertices\n", size);
        printf("--------------------\n");
        
        // Generate a scale-free graph
        printf("Generating scale-free graph...\n");
        COOGraph scaleFreeCOO = generateScaleFreeGraphCOO(size, 100);
        CSRGraph scaleFreeCSR = convertCOOtoCSR(scaleFreeCOO);
        CSCGraph scaleFreeCSC = convertCOOtoCSC(scaleFreeCOO);
        
        // Allocate graphs on device
        printf("Allocating graphs on device...\n");
        COOGraph deviceScaleFreeCOO = allocateCOOGraphOnDevice(scaleFreeCOO);
        CSRGraph deviceScaleFreeCSR = allocateCSRGraphOnDevice(scaleFreeCSR);
        
        // Verify correctness for this specific graph
        printf("Verifying BFS correctness for %d vertices graph... ", size);
        int* seq_result = bfs(scaleFreeCSR, start_vertex);
        
        int* par_push_result = bfsParallelPushVertexCentricDevice(deviceScaleFreeCSR, start_vertex);
        bool push_ok = compareBFSResults(seq_result, par_push_result, scaleFreeCSR.numVertices, false);
        
        int* par_edge_result = bfsParallelEdgeCentricDevice(deviceScaleFreeCOO, start_vertex);
        bool edge_ok = compareBFSResults(seq_result, par_edge_result, scaleFreeCSR.numVertices, false);
        
        int* par_frontier_result = bfsParallelFrontierVertexCentricDevice(deviceScaleFreeCSR, start_vertex);
        bool frontier_ok = compareBFSResults(seq_result, par_frontier_result, scaleFreeCSR.numVertices, false);
        
        bool correct = push_ok && edge_ok && frontier_ok;
        
        // Free the verification results
        free(seq_result);
        free(par_push_result);
        free(par_edge_result);
        free(par_frontier_result);
        
        if (!correct) {
            printf("ERROR: Skipping this graph size due to correctness issues!\n\n");
            
            // Free memory and continue to next size
            freeCSRGraphOnDevice(deviceScaleFreeCSR);
            freeCOOGraphOnDevice(deviceScaleFreeCOO);
            free(scaleFreeCOO.scr);
            free(scaleFreeCOO.dst);
            free(scaleFreeCOO.values);
            free(scaleFreeCSR.srcPtrs);
            free(scaleFreeCSR.dst);
            free(scaleFreeCSR.values);
            free(scaleFreeCSC.dstPtrs);
            free(scaleFreeCSC.src);
            free(scaleFreeCSC.values);
            
            continue;
        }
        printf("Passed!\n");
        
        // Benchmark sequential BFS
        printf("Sequential BFS: ");
        float seq_time = benchmark_sequential_bfs(bfs, scaleFreeCSR, start_vertex, warmup_runs, timing_runs);
        printf("%.2f ms\n", seq_time);
        
        // Benchmark push-based vertex-centric BFS
        printf("Push Vertex-Centric BFS: ");
        float push_time = benchmark_parallel_csr_bfs(bfsParallelPushVertexCentricDevice, 
                                                  deviceScaleFreeCSR, start_vertex, warmup_runs, timing_runs);
        printf("%.2f ms (%.2fx speedup)\n", push_time, seq_time / push_time);
        
        // Benchmark edge-centric BFS
        printf("Edge-Centric BFS: ");
        float edge_time = benchmark_parallel_coo_bfs(bfsParallelEdgeCentricDevice, 
                                                 deviceScaleFreeCOO, start_vertex, warmup_runs, timing_runs);
        printf("%.2f ms (%.2fx speedup)\n", edge_time, seq_time / edge_time);
        
        // Benchmark frontier-based BFS
        printf("Frontier-based BFS: ");
        float frontier_time = benchmark_parallel_csr_bfs(bfsParallelFrontierVertexCentricDevice, 
                                                      deviceScaleFreeCSR, start_vertex, warmup_runs, timing_runs);
        printf("%.2f ms (%.2fx speedup)\n", frontier_time, seq_time / frontier_time);
        
        printf("\n");
        
        // Free device graph memory
        freeCSRGraphOnDevice(deviceScaleFreeCSR);
        freeCOOGraphOnDevice(deviceScaleFreeCOO);
        
        // Free host graph memory
        free(scaleFreeCOO.scr);
        free(scaleFreeCOO.dst);
        free(scaleFreeCOO.values);
        free(scaleFreeCSR.srcPtrs);
        free(scaleFreeCSR.dst);
        free(scaleFreeCSR.values);
        free(scaleFreeCSC.dstPtrs);
        free(scaleFreeCSC.src);
        free(scaleFreeCSC.values);
    }
    
    // Generate a small-world graph with 10000 vertices as an additional benchmark
    printf("Small-World Graph (10000 vertices)\n");
    printf("--------------------\n");
    
    // Generate a small-world graph
    printf("Generating small-world graph...\n");
    COOGraph smallWorldCOO = generateSmallWorldGraphCOO(10000, 50, 0.1);
    CSRGraph smallWorldCSR = convertCOOtoCSR(smallWorldCOO);
    
    // Verify correctness for the small-world graph
    printf("Verifying BFS correctness for small-world graph... ");
    
    // Allocate graphs on device
    COOGraph deviceSmallWorldCOO = allocateCOOGraphOnDevice(smallWorldCOO);
    CSRGraph deviceSmallWorldCSR = allocateCSRGraphOnDevice(smallWorldCSR);
    
    // Get sequential reference result
    int* sw_seq_result = bfs(smallWorldCSR, start_vertex);
    
    // Check implementations
    int* sw_push_result = bfsParallelPushVertexCentricDevice(deviceSmallWorldCSR, start_vertex);
    bool sw_push_ok = compareBFSResults(sw_seq_result, sw_push_result, smallWorldCSR.numVertices, false);
    
    int* sw_edge_result = bfsParallelEdgeCentricDevice(deviceSmallWorldCOO, start_vertex);
    bool sw_edge_ok = compareBFSResults(sw_seq_result, sw_edge_result, smallWorldCSR.numVertices, false);
    
    int* sw_frontier_result = bfsParallelFrontierVertexCentricDevice(deviceSmallWorldCSR, start_vertex);
    bool sw_frontier_ok = compareBFSResults(sw_seq_result, sw_frontier_result, smallWorldCSR.numVertices, false);
    
    bool small_world_correct = sw_push_ok && sw_edge_ok && sw_frontier_ok;
    
    // Free verification results
    free(sw_seq_result);
    free(sw_push_result);
    free(sw_edge_result);
    free(sw_frontier_result);
    
    if (!small_world_correct) {
        printf("ERROR: Skipping small-world graph benchmark due to correctness issues!\n");
        
        // Free device graph memory
        freeCSRGraphOnDevice(deviceSmallWorldCSR);
        freeCOOGraphOnDevice(deviceSmallWorldCOO);
        
        // Free host graph memory
        free(smallWorldCOO.scr);
        free(smallWorldCOO.dst);
        free(smallWorldCOO.values);
        free(smallWorldCSR.srcPtrs);
        free(smallWorldCSR.dst);
        free(smallWorldCSR.values);
        
        return 1;
    }
    printf("Passed!\n");
    
    // Benchmark sequential BFS
    printf("Sequential BFS: ");
    float seq_time = benchmark_sequential_bfs(bfs, smallWorldCSR, start_vertex, warmup_runs, timing_runs);
    printf("%.2f ms\n", seq_time);
    
    // Benchmark push-based vertex-centric BFS
    printf("Push Vertex-Centric BFS: ");
    float push_time = benchmark_parallel_csr_bfs(bfsParallelPushVertexCentricDevice, 
                                              deviceSmallWorldCSR, start_vertex, warmup_runs, timing_runs);
    printf("%.2f ms (%.2fx speedup)\n", push_time, seq_time / push_time);
    
    // Benchmark edge-centric BFS
    printf("Edge-Centric BFS: ");
    float edge_time = benchmark_parallel_coo_bfs(bfsParallelEdgeCentricDevice, 
                                             deviceSmallWorldCOO, start_vertex, warmup_runs, timing_runs);
    printf("%.2f ms (%.2fx speedup)\n", edge_time, seq_time / edge_time);
    
    // Benchmark frontier-based BFS
    printf("Frontier-based BFS: ");
    float frontier_time = benchmark_parallel_csr_bfs(bfsParallelFrontierVertexCentricDevice, 
                                                  deviceSmallWorldCSR, start_vertex, warmup_runs, timing_runs);
    printf("%.2f ms (%.2fx speedup)\n", frontier_time, seq_time / frontier_time);
    
    // Free device graph memory
    freeCSRGraphOnDevice(deviceSmallWorldCSR);
    freeCOOGraphOnDevice(deviceSmallWorldCOO);
    
    // Free host graph memory
    free(smallWorldCOO.scr);
    free(smallWorldCOO.dst);
    free(smallWorldCOO.values);
    free(smallWorldCSR.srcPtrs);
    free(smallWorldCSR.dst);
    free(smallWorldCSR.values);
    
    return 0;
}