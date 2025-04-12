#ifndef GRAPH_CONVERSIONS_H
#define GRAPH_CONVERSIONS_H

#include "graph_structures.h"

CSRGraph convertCOOtoCSR(const COOGraph& cooGraph);
COOGraph convertCSRtoCOO(const CSRGraph& csrGraph);
CSCGraph convertCOOtoCSC(const COOGraph& cooGraph);
CSCGraph convertCSRtoCSC(const CSRGraph& csrGraph);

#endif // GRAPH_CONVERSIONS_H