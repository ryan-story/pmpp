#ifndef BFS_PARALLEL_H
#define BFS_PARALLEL_H

#include "graph_structures.h"

// Functions taking graph on the host
int* bfsParallelPushVertexCentric(const CSRGraph& hostGraph, int startingNode);
int* bfsParallelPullVertexCentric(const CSCGraph& hostGraph, int startingNode);
int* bfsParallelEdgeCentric(const COOGraph& hostGraph, int startingNode);
int* bfsParallelFrontierVertexCentric(const CSRGraph& hostGraph, int startingNode);

// Functions taking graph already on the device
int* bfsParallelPushVertexCentricDevice(const CSRGraph& deviceGraph, int startingNode);
int* bfsParallelPullVertexCentricDevice(const CSCGraph& deviceGraph, int startingNode);
int* bfsParallelEdgeCentricDevice(const COOGraph& deviceGraph, int startingNode);
int* bfsParallelFrontierVertexCentricDevice(const CSRGraph& deviceGraph, int startingNode);

#endif // BFS_PARALLEL_H