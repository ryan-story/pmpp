#include <stdio.h>
#include <stdlib.h>
#include <queue>
#include <vector>

struct CSRGraph{
    int *rowPtr;
    int *colIdx;
    int *values;
};

// BFS returning a pointer to the list of levels for all vertices
int* bfs(const CSRGraph& graph, int numVertices, int startingNode) {
    int* levels = (int*)malloc(sizeof(int) * numVertices);
    std::vector<bool> visited(numVertices, false);

    //set the default level to -1 meaning it is not yet visited
    for (int i = 0; i < numVertices; i++){
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

        for(int i = graph.rowPtr[vertex]; i < graph.rowPtr[vertex+1]; i++){
            int neigbour = graph.colIdx[i];
            if (!visited[neigbour]){
                levels[neigbour] = levels[vertex] + 1;
                visited[neigbour] = true;
                queue.push(neigbour);
            }
        }
    }
    return levels;
}

int main() {
    // CRS representation from the image
    int rowPtrData[] = {0, 2, 5, 6, 8, 9, 11, 12, 16};
    int colIdxData[] = {2, 5, 0, 4, 7, 3, 0, 6, 3, 1, 7, 4, 1, 2, 4, 6};
    int valuesData[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    int numVertices = 8;

    struct CSRGraph graph = {
        .rowPtr = rowPtrData,
        .colIdx = colIdxData,
        .values = valuesData
    };

    int* levels = bfs(graph, numVertices, 0);

    for(int i=0; i<numVertices; i++){
        printf("%d: level %d \n", i, levels[i]);
    }

    return 0;
    
}

