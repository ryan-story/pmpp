#ifndef GRAPH_STRUCTURES_H
#define GRAPH_STRUCTURES_H

struct CSRGraph {
    int* srcPtrs;
    int* dst;
    int* values;
    int numVertices;
};

struct CSCGraph {
    int* dstPtrs;
    int* src;
    int* values;
    int numVertices;
};

struct COOGraph {
    int* scr;
    int* dst;
    int* values;
    int numEdges;
    int numVertices;
};

#endif  // GRAPH_STRUCTURES_H
