#include <stdio.h>
#include <stdlib.h>
#include <queue>
#include <vector>

struct CSRGraph{
    int *srcPtr;
    int *dst;
    int *values;
    int numVertices;
};

struct CSSGraph{
    int *colPtr;
    int *rowIdx;
    int *values;
    int numVertices;
};

// BFS returning a pointer to the list of levels for all vertices
int* bfs(const CSRGraph& graph, int startingNode) {
    int* levels = (int*)malloc(sizeof(int) * graph.numVertices);
    std::vector<bool> visited(graph.numVertices, false);

    //set the default level to -1 meaning it is not yet visited
    for (int i = 0; i < graph.numVertices; i++){
        levels[i] = -1;
    }

    std::queue<int> queue;

    levels[startingNode] = 0;
    visited[startingNode] = true;
    queue.push(startingNode);

    while (!queue.empty())
    {   
        int vertex = queue.front();
        queue.pop();

        for(int edge = graph.srcPtr[vertex]; edge < graph.srcPtr[vertex+1]; edge++){
            int neigbour = graph.dst[edge];
            if (!visited[neigbour]){
                levels[neigbour] = levels[vertex] + 1;
                visited[neigbour] = true;
                queue.push(neigbour);
            }
        }
    }
    return levels;
}

__global__ void bsf_push_vertex_centric_kernel(CSRGraph graph, int* levels, int* newVertexVisitd, unsigned int currLevel){
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < graph.numVertices){
        if (levels[vertex] == currLevel - 1){
            //iterate over all the vertices at the current level
            for (unsigned int edge = graph.srcPtr[vertex]; edge < graph.srcPtr[vertex+1]; edge++){
                unsigned int neigbour = graph.dst[edge];
                //not yet visited
                if (levels[neigbour] == -1){
                    levels[neigbour] = currLevel;
                    *newVertexVisitd = 1; //set the flag to 1 so we know we need to run it again
                }
            }
        }
    }
}

int* bfsParallelPushVertexCentric(const CSRGraph& hostGraph, int startingNode) {
    // Initialize host levels array
    int* hostLevels = (int*)malloc(sizeof(int) * hostGraph.numVertices);
    for (int i = 0; i < hostGraph.numVertices; i++) {
        hostLevels[i] = -1;
    }
    hostLevels[startingNode] = 0;
    
    // Calculate size needed for graph arrays
    size_t rowPtrSize = sizeof(int) * (hostGraph.numVertices + 1);
    size_t edgeSize = sizeof(int) * hostGraph.srcPtr[hostGraph.numVertices];
    size_t vertexSize = sizeof(int) * hostGraph.numVertices;
    
    // Allocate device memory
    int *d_rowPtr, *d_colIdx, *d_values, *d_levels, *d_newVertexVisited;
    cudaMalloc(&d_rowPtr, rowPtrSize);
    cudaMalloc(&d_colIdx, edgeSize);
    cudaMalloc(&d_values, edgeSize);
    cudaMalloc(&d_levels, vertexSize);
    cudaMalloc(&d_newVertexVisited, sizeof(int));
    
    cudaMemcpy(d_rowPtr, hostGraph.srcPtr, rowPtrSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, hostGraph.dst, edgeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, hostGraph.values, edgeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_levels, hostLevels, vertexSize, cudaMemcpyHostToDevice);
    
    CSRGraph deviceGraph = {
        .srcPtr = d_rowPtr,
        .dst = d_colIdx,
        .values = d_values,
        .numVertices = hostGraph.numVertices
    };
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (hostGraph.numVertices + threadsPerBlock - 1) / threadsPerBlock;
    
    int currLevel = 1;
    int hostNewVertexVisited = 1;
    
    while (hostNewVertexVisited != 0) {
        hostNewVertexVisited = 0;
        cudaMemcpy(d_newVertexVisited, &hostNewVertexVisited, sizeof(int), cudaMemcpyHostToDevice);
        
        bsf_push_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            deviceGraph, d_levels, d_newVertexVisited, currLevel);
        
        cudaDeviceSynchronize();
        
        cudaMemcpy(&hostNewVertexVisited, d_newVertexVisited, sizeof(int), cudaMemcpyDeviceToHost);
        currLevel++;
    }
    
    cudaMemcpy(hostLevels, d_levels, vertexSize, cudaMemcpyDeviceToHost);
    
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_levels);
    cudaFree(d_newVertexVisited);
    
    return hostLevels;
}


bool compareBFSResults(int* sequentialLevels, int* parallelLevels, int numVertices, bool printDetails = false) {
    bool resultsMatch = true;
    
    if (printDetails) {
        printf("Comparing BFS results:\n");
        printf("%-10s %-15s %-15s %-10s\n", "Vertex", "Sequential", "Parallel", "Match");
        printf("------------------------------------------------\n");
    }
    
    for (int i = 0; i < numVertices; i++) {
        bool vertexMatch = (sequentialLevels[i] == parallelLevels[i]);
        
        if (!vertexMatch) {
            resultsMatch = false;
            
            // Always print mismatches even if printDetails is false
            if (!printDetails) {
                printf("Mismatch at vertex %d: Sequential=%d, Parallel=%d\n", 
                       i, sequentialLevels[i], parallelLevels[i]);
            }
        }
        
        if (printDetails) {
            printf("%-10d %-15d %-15d %-10s\n", 
                   i, 
                   sequentialLevels[i], 
                   parallelLevels[i], 
                   vertexMatch ? "✓" : "✗");
        }
    }
    
    if (printDetails) {
        printf("------------------------------------------------\n");
        printf("Overall result: %s\n", resultsMatch ? "Both implementations match!" : "Implementations differ!");
    } else if (resultsMatch) {
        printf("All BFS results match between sequential and parallel implementations.\n");
    }
    
    return resultsMatch;
}


int main() {
    // CRS representation from the image
    int rowPtrData[] = {0, 2, 5, 6, 8, 9, 11, 12, 16};
    int colIdxData[] = {2, 5, 0, 4, 7, 3, 0, 6, 3, 1, 7, 4, 1, 2, 4, 6};
    int valuesData[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    int numVertices = 8;

    struct CSRGraph graph = {
        .srcPtr = rowPtrData,
        .dst = colIdxData,
        .values = valuesData,
        .numVertices = numVertices
    };
    

    // int* levels = bfs(graph, 0);
    int* sequentialLevels = bfs(graph, 0);
    int* parallelLevels = bfsParallelPushVertexCentric(graph, 0);


    bool resultsMatch = compareBFSResults(sequentialLevels, parallelLevels, numVertices, true);

    return 0;
    
}

