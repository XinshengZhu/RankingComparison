/* Utils.c */
#include "Utils.h"
#include <math.h>

/**
 * Computes the average document length across all documents
 *
 * This function is used in BM25 scoring to normalize document lengths.
 * The average length helps account for the fact that longer documents
 * naturally contain more term occurrences.
 *
 * @param docLengths Array containing length of each document
 * @param totalDocCount Total number of documents in collection
 * @return Average document length (rounded to nearest integer)
 *
 * Note: Integer division is used for efficiency, as exact precision
 * is not critical for BM25 calculations.
 */
int computeAvgDocLength(const int *docLengths, int totalDocCount) {
    int totalDocLength = 0;
    for (int docIndex = 0; docIndex < totalDocCount; docIndex++) {
        totalDocLength += docLengths[docIndex];
    }
    return totalDocLength / totalDocCount;
}

/**
 * Computes the total number of documents containing a term
 * across all parsed items (used during merge phase)
 *
 * This function aggregates document counts when merging multiple
 * intermediate files. The count is used in BM25's IDF calculation.
 *
 * @param parsedItems Array of parsed items from different files
 * @return Total number of documents containing the term
 *
 * Example:
 * If a term appears in:
 * - 5 documents in file 1
 * - 3 documents in file 2
 * - NULL in file 3
 * The function returns 8
 */
int computeTermDocCount(ParsedItem **parsedItems) {
    int termDocCount = 0;
    for (int itemIndex = 0; itemIndex < MERGE_HEAP_SIZE; itemIndex++) {
        if (parsedItems[itemIndex] != NULL) {
            termDocCount += parsedItems[itemIndex]->postingCount;
        }
    }
    return termDocCount;
}

/**
 * Calculates BM25 impact score for a term-document pair
 *
 * Implements the BM25 scoring function:
 * score = IDF * ((k1 + 1) * tf) / (K + tf)
 * where:
 * - IDF = log((N - n + 0.5) / (n + 0.5))
 * - K = k1 * ((1 - b) + b * (docLen / avgDocLen))
 *
 * Constants:
 * - k1 = 1.2 (term frequency saturation parameter)
 * - b = 0.75 (length normalization impact)
 *
 * @param totalDocCount Total documents in collection (N)
 * @param termDocCount Documents containing the term (n)
 * @param termFrequency Frequency of term in document (tf)
 * @param docLength Length of current document
 * @param avgDocLength Average document length in collection
 * @return BM25 impact score
 *
 * The formula can be broken down into:
 * 1. IDF component: log((N - n + 0.5) / (n + 0.5))
 *    - Measures how rare/common the term is
 *    - Higher for rare terms, lower for common terms
 *
 * 2. TF component: ((k1 + 1) * tf) / (K + tf)
 *    - Measures term importance in document
 *    - Saturates as term frequency increases
 *    - Normalized by document length
 *
 * Note: The score combines both components to balance:
 * - Term rarity in collection (IDF)
 * - Term prominence in document (TF)
 * - Document length normalization
 */
double calculateBM25ImpactScore(int totalDocCount, int termDocCount, int termFrequency, int docLength, int avgDocLength) {
    double k1 = 1.2;
    double b = 0.75;
    double idf = log((totalDocCount - termDocCount + 0.5) / (termDocCount + 0.5));
    double tf = ((k1 + 1) * termFrequency) / (k1 * ((1 - b) + b * ((double)docLength / (double)avgDocLength)) + termFrequency);
    return idf * tf;
}
