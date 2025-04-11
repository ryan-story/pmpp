#include <stdio.h>
#include <stdlib.h>
#include <queue>
#include <vector>

struct CSRGraph{
    int *srcPtrs;
    int *dst;
    int *values;
    int numVertices;
};

struct CSCGraph{
    int *dstPtrs;
    int *src;
    int *values;
    int numVertices;
};

struct COOGraph
{
    int *scr;
    int *dst;
    int *values;
    int numEdges;
    int numVertices;
};


COOGraph convertCSRtoCOO(const CSRGraph& csrGraph) {
    int numVertices = csrGraph.numVertices;
    int numEdges = csrGraph.srcPtrs[numVertices];
    
    // Allocate memory for COO graph
    COOGraph cooGraph;
    cooGraph.numVertices = numVertices;
    cooGraph.numEdges = numEdges;
    cooGraph.scr = (int*)malloc(sizeof(int) * numEdges);
    cooGraph.dst = (int*)malloc(sizeof(int) * numEdges);
    cooGraph.values = (int*)malloc(sizeof(int) * numEdges);
    
    // Populate the COO arrays
    int edgeIdx = 0;
    for (int i = 0; i < numVertices; i++) {
        for (int j = csrGraph.srcPtrs[i]; j < csrGraph.srcPtrs[i + 1]; j++) {
            cooGraph.scr[edgeIdx] = i;                // Source vertex
            cooGraph.dst[edgeIdx] = csrGraph.dst[j];  // Destination vertex
            cooGraph.values[edgeIdx] = csrGraph.values[j]; // Edge value
            edgeIdx++;
        }
    }
    
    return cooGraph;
}

CSCGraph convertCOOtoCSC(const COOGraph& cooGraph) {
    CSCGraph cscGraph;
    cscGraph.numVertices = cooGraph.numVertices;
    cscGraph.dstPtrs = (int*)malloc(sizeof(int) * (cooGraph.numVertices + 1));
    cscGraph.src = (int*)malloc(sizeof(int) * cooGraph.numEdges);
    cscGraph.values = (int*)malloc(sizeof(int) * cooGraph.numEdges);
    
    // Initialize dstPtrs array with zeros
    for (int i = 0; i <= cooGraph.numVertices; i++) {
        cscGraph.dstPtrs[i] = 0;
    }
    
    // Count occurrences of each destination (column)
    for (int i = 0; i < cooGraph.numEdges; i++) {
        cscGraph.dstPtrs[cooGraph.dst[i] + 1]++;
    }
    
    // Cumulative sum to get column pointers
    for (int i = 1; i <= cooGraph.numVertices; i++) {
        cscGraph.dstPtrs[i] += cscGraph.dstPtrs[i - 1];
    }
    
    // Copy data from COO to CSC
    int* pos = (int*)malloc(sizeof(int) * cooGraph.numVertices);
    memcpy(pos, cscGraph.dstPtrs, sizeof(int) * cooGraph.numVertices);
    
    for (int i = 0; i < cooGraph.numEdges; i++) {
        int col = cooGraph.dst[i];
        int idx = pos[col]++;
        
        cscGraph.src[idx] = cooGraph.scr[i];
        cscGraph.values[idx] = cooGraph.values[i];
    }
    
    free(pos);
    return cscGraph;
}

CSCGraph convertCSRtoCSC(const CSRGraph& csrGraph) {
    COOGraph cooGraph = convertCSRtoCOO(csrGraph);
    CSCGraph cscGraph = convertCOOtoCSC(cooGraph);
    
    free(cooGraph.scr);
    free(cooGraph.dst);
    free(cooGraph.values);
    
    return cscGraph;
}



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

        for(int edge = graph.srcPtrs[vertex]; edge < graph.srcPtrs[vertex+1]; edge++){
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
            for (unsigned int edge = graph.srcPtrs[vertex]; edge < graph.srcPtrs[vertex+1]; edge++){
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
        .srcPtrs = d_srcPtrs,
        .dst = d_dst,
        .values = d_values,
        .numVertices = hostGraph.numVertices
    };
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (hostGraph.numVertices + threadsPerBlock - 1) / threadsPerBlock;
    
    int currLevel = 1;
    int hostNewVertexVisited = 1;
    
    //we iterate over the levels as long as any vertex reports finding a new unvisited neighbour
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
    
    cudaFree(d_srcPtrs);
    cudaFree(d_dst);
    cudaFree(d_values);
    cudaFree(d_levels);
    cudaFree(d_newVertexVisited);
    
    return hostLevels;
}

__global__ void bsf_pull_vertex_centric_kernel(CSCGraph graph, int* levels, int* newVertexVisitd, unsigned int currLevel){
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;

    if (vertex < graph.numVertices){
        if (levels[vertex] == -1) { //Vertext not yet visited
            for (unsigned int edge = graph.dstPtrs[vertex]; edge < graph.dstPtrs[vertex+1]; edge++){
                unsigned int neighbour = graph.src[edge];

                if (levels[neighbour] == currLevel - 1){
                    levels[vertex] = currLevel; 
                    *newVertexVisitd = 1;
                    break; //if any of the neighbour at the prev level we reached our point
                }
            }
        }
    }
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
    size_t srcSize = sizeof(int) * hostGraph.dstPtrs[hostGraph.numVertices]; // look where the last dstPtrs is poinintg - than we know how many scr elements there are
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
        .dstPtrs = d_dstPtrs,
        .src = d_src,
        .values = d_values,
        .numVertices = hostGraph.numVertices
    };
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (hostGraph.numVertices + threadsPerBlock - 1) / threadsPerBlock;
    
    int currLevel = 1;
    int hostNewVertexVisited = 1;
    
    //we iterate over the levels as long as any vertex reports finding a new unvisited neighbour
    while (hostNewVertexVisited != 0) {
        hostNewVertexVisited = 0;
        cudaMemcpy(d_newVertexVisited, &hostNewVertexVisited, sizeof(int), cudaMemcpyHostToDevice);
        
        bsf_pull_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            deviceGraph, d_levels, d_newVertexVisited, currLevel);
        
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

__global__ void bsf_edge_centric_kernel(COOGraph cooGraph, int* levels, int* newVertexVisitd, unsigned int currLevel){
    unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;

    if (edge < cooGraph.numEdges){
        unsigned int vertex = cooGraph.scr[edge];
        if (levels[vertex] == currLevel - 1){
            unsigned int neighbour = cooGraph.dst[edge];
            if (levels[neighbour] == -1){//neighbour not yet visited
                levels[neighbour] = currLevel;
                *newVertexVisitd = 1;
            }
        }
    }
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
    
    COOGraph deviceGraph = {
        .scr = d_src,
        .dst = d_dst,
        .values = d_values,
        .numEdges = hostGraph.numEdges,
        .numVertices = hostGraph.numVertices
    };
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (hostGraph.numEdges + threadsPerBlock - 1) / threadsPerBlock;
    
    int currLevel = 1;
    int hostNewVertexVisited = 1;
    
    //we iterate over the levels as long as any vertex reports finding a new unvisited neighbour
    while (hostNewVertexVisited != 0) {
        hostNewVertexVisited = 0;
        cudaMemcpy(d_newVertexVisited, &hostNewVertexVisited, sizeof(int), cudaMemcpyHostToDevice);
        
        bsf_edge_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            deviceGraph, d_levels, d_newVertexVisited, currLevel);
        
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

__global__ void bsf_frontier_vertex_centric_kernel(CSRGraph csrGraph, int* levels, int* prevFrontier,
                                                    int* currFrontier, int numPrevFrontier, int* numCurrentFrontier,
                                                    unsigned int currLevel){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numPrevFrontier){
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex+1]; edge++){
            unsigned int neighbour = csrGraph.dst[edge];
            //neighbour not visited
            if (atomicCAS(&levels[neighbour], -1, currLevel) == -1){
                unsigned int currFrontierIdx = atomicAdd(numCurrentFrontier, 1);
                currFrontier[currFrontierIdx] = neighbour;
            }
        }
    }
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
        .srcPtrs = d_srcPtrs,
        .dst = d_dst,
        .values = d_values,
        .numVertices = hostGraph.numVertices
    };
    
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

    struct CSRGraph csrGraph = {
        .srcPtrs = rowPtrData,
        .dst = colIdxData,
        .values = valuesData,
        .numVertices = numVertices
    };

    CSCGraph cscGraph = convertCSRtoCSC(csrGraph);
    COOGraph cooGraph = convertCSRtoCOO(csrGraph);
    
    // int* levels = bfs(graph, 0);
    int* sequentialLevels = bfs(csrGraph, 0);
    // int* parallelLevels = bfsParallelPushVertexCentric(csrGraph, 0);
    // int* parallelLevels = bfsParallelPullVertexCentric(cscGraph, 0);
    // int* parallelLevels = bfsParallelEdgeCentric(cooGraph, 0);
    int* parallelLevels = bfsParallelFrontierVertexCentric(csrGraph, 0);


    bool resultsMatch = compareBFSResults(sequentialLevels, parallelLevels, numVertices, true);

    return 0;
    
}

