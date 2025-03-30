// 1. Code cpu sequential BFS - for each node have a level
// 2. Use it to find the closest path between two nodes
// 2. Code up the 


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>

// BFS on a graph represented in CRS format that tracks levels/distances
void bfs(int startVertex, int* rowPtr, int* colIdx, int* values, int numVertices) {
    // Create a queue for BFS
    int queue[100];  // Assuming max 100 vertices
    int front = 0, rear = 0;
    
    // Create a visited array
    bool* visited = (bool*)calloc(numVertices, sizeof(bool));
    
    // Create an array to store levels/distances from start vertex
    int* level = (int*)malloc(numVertices * sizeof(int));
    for (int i = 0; i < numVertices; i++) {
        level[i] = INT_MAX;  // Initialize to "infinity"
    }
    
    // Mark the start vertex as visited, enqueue it, and set its level to 0
    visited[startVertex] = true;
    queue[rear++] = startVertex;
    level[startVertex] = 0;
    
    while (front < rear) {
        // Dequeue a vertex from queue
        int currentVertex = queue[front++];
        
        // Get all adjacent vertices of the dequeued vertex
        int startIdx = rowPtr[currentVertex];
        int endIdx = rowPtr[currentVertex + 1];
        
        for (int i = startIdx; i < endIdx; i++) {
            int adjacentVertex = colIdx[i];
            
            if (!visited[adjacentVertex]) {
                visited[adjacentVertex] = true;
                queue[rear++] = adjacentVertex;
                
                // Set level of adjacent vertex to current vertex's level + 1
                level[adjacentVertex] = level[currentVertex] + 1;
            }
        }
    }
    
    // Print the levels
    printf("Distances from vertex %d:\n", startVertex);
    for (int i = 0; i < numVertices; i++) {
        if (level[i] == INT_MAX) {
            printf("Vertex %d: Unreachable\n", i);
        } else {
            printf("Vertex %d: Level %d\n", i, level[i]);
        }
    }
    
    // Free allocated memory
    free(visited);
    free(level);
    
    // Return the level array if needed for further processing
    // You could modify this function to return the level array instead of printing it
}

int main() {
    // CRS representation from the image
    int rowPtr[] = {0, 2, 5, 6, 7, 8, 10, 11};
    int colIdx[] = {2, 5, 0, 4, 7, 3, 0, 6, 3, 1, 7, 4, 2, 4, 6};
    int values[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    
    int numVertices = sizeof(rowPtr)/sizeof(rowPtr[0]) - 1;
    
    // Perform BFS starting from vertex 0
    bfs(0, rowPtr, colIdx, values, numVertices);
    
    return 0;
}

