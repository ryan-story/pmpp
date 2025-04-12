#ifndef GRAPH_GENERATORS_H
#define GRAPH_GENERATORS_H

#include "graph_structures.h"

COOGraph generateScaleFreeGraphCOO(int numVertices, int edgesPerNewVertex);
COOGraph generateSmallWorldGraphCOO(int numVertices, int k, float rewireProbability);

#endif // GRAPH_GENERATORS_H