/* InvertedList.h */
#ifndef INVERTED_LIST_H
#define INVERTED_LIST_H

#include <stdio.h>
#include <stdint.h>

/* Maximum postings per chunk */
#define MAX_POSTING_COUNT 128
/* Maximum chunks per block */
#define MAX_CHUNK_COUNT 64

/**
 * Structure representing a word's inverted list
 * Manages chunk-based reading and decompression
 */
typedef struct InvertedList {
    char *word;                          // Word string
    FILE *listPointer;                   // File pointer to index
    int currentChunkIndex;               // Current chunk being processed
    int remainingChunkCount;             // Chunks left to process
    int chunkSizes[MAX_CHUNK_COUNT];     // Size of each chunk in bytes
    int lastDocIds[MAX_CHUNK_COUNT];     // Last docID in each chunk
    int currentPostingIndex;             // Current posting position
    int postingCount;                    // Number of postings in current chunk
    uint8_t *postings;                   // Compressed posting data
    int docIds[MAX_POSTING_COUNT];       // Decompressed document IDs
    double impactScores[MAX_POSTING_COUNT]; // Decompressed impact scores
} InvertedList;

/* Function prototypes */
InvertedList * createInvertedList(FILE *listPointer, const char *word, int remainingChunkToStart, int remainingChunkToEnd); // Create an inverted list
void freeInvertedList(InvertedList *invertedList);        // Free inverted list
void updateInvertedListByChunk(InvertedList *invertedList); // Update inverted list by chunk
void updateInvertedListByBlock(InvertedList *invertedList); // Update inverted list by block

#endif
