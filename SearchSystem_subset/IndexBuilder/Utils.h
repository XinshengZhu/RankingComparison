/* Utils.h */
#ifndef UTILS_H
#define UTILS_H

#include "MergeHeap.h"

/* Function prototypes */
int computeAvgDocLength(const int *docLengths, int totalDocCount);  // Compute the average document length
int computeTermDocCount(ParsedItem **parsedItems);  // Compute the total number of documents containing the term
double calculateBM25ImpactScore(int totalDocCount, int termDocCount, int termFrequency, int docLength, int avgDocLength);   // Calculate the BM25 impact score

#endif
