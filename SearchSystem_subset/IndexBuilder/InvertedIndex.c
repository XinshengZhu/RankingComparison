/* InvertedIndex.c */
#include "InvertedIndex.h"
#include "Compression.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * Creates and initializes a new index block
 *
 * An index block is a container for multiple chunks of postings.
 * It includes metadata for efficient searching and compression.
 *
 * Memory layout:
 * - Array of chunk sizes for compressed data
 * - Array of last document IDs for query processing
 * - Array of chunks, each containing:
 *   * Document IDs
 *   * Impact scores
 *   * Posting count
 *
 * @return Pointer to newly created index block
 *
 * Note: All arrays are initialized to invalid/zero values:
 * - Document IDs: -1 (invalid ID)
 * - Impact scores: 0.0
 * - Chunk sizes: 0
 * - Posting counts: 0
 */
IndexBlock *createIndexBlock() {
    // Allocate block structure
    IndexBlock *newBlock = (IndexBlock *)malloc(sizeof(IndexBlock));
    if (newBlock == NULL) {
        printf("Error allocating memory!\n");
        exit(1);
    }
    // Initialize all chunks and metadata
    newBlock->chunkCount = 0;
     for (int chunkIndex = 0; chunkIndex < MAX_CHUNK_COUNT; chunkIndex++) {
         // Initialize block metadata for chunk
         newBlock->chunkSizes[chunkIndex] = 0;
         newBlock->lastDocIds[chunkIndex] = -1;
         // Initialize postings in chunk
         for (int postingIndex = 0; postingIndex < MAX_POSTING_COUNT; postingIndex++) {
             newBlock->indexChunks[chunkIndex].postingCount = 0;
             newBlock->indexChunks[chunkIndex].docIds[postingIndex] = -1;
             newBlock->indexChunks[chunkIndex].impactScores[postingIndex] = 0.0;
         }
     }
    newBlock->nextIndexBlock = NULL;
    return newBlock;
}

/**
 * Creates and initializes a new inverted index
 *
 * The inverted index is the main data structure for storing term-document relationships.
 * It maintains a linked list of blocks, where each block contains multiple chunks of postings.
 * This structure allows for:
 * - Efficient storage of variable-length posting lists
 * - Quick query processing capabilities
 *
 * @return Pointer to newly created inverted index
 *
 * Structure organization:
 * InvertedIndex
 * └── Block 1
 *     ├── Chunk 1 (128 postings max)
 *     ├── Chunk 2
 *     └── ... (64 chunks max)
 * └── Block 2
 * └── ... (24000 blocks max)
 */
InvertedIndex *createInvertedIndex() {
    InvertedIndex *newIndex = (InvertedIndex *)malloc(sizeof(InvertedIndex));
    if (newIndex == NULL) {
        printf("Error allocating memory!\n");
        exit(1);
    }
    // Initialize index metadata
    newIndex->chunkNumber = 0;
    newIndex->blockCount = 0;
    newIndex->headIndexBlock = NULL;
    newIndex->tailIndexBlock = NULL;
    return newIndex;
}

/**
 * Frees all memory associated with an inverted index
 *
 * Systematically releases memory by:
 * 1. Resetting all counters and metadata
 * 2. Traversing the block linked list
 * 3. Cleaning each block's chunks and postings
 * 4. Freeing block structures
 * 5. Finally freeing the index structure
 *
 * @param invertedIndex Pointer to inverted index to be freed
 *
 * Note: This function ensures complete cleanup without memory leaks
 * by methodically resetting and freeing all nested structures.
 */
void freeInvertedIndex(InvertedIndex *invertedIndex) {
    invertedIndex->chunkNumber = 0;
    invertedIndex->blockCount = 0;
    // Traverse and free blocks
    IndexBlock *currentBlock = invertedIndex->headIndexBlock;
    while (currentBlock != NULL) {
        IndexBlock *nextBlock = currentBlock->nextIndexBlock;
        // Reset and clean block data
        currentBlock->chunkCount = 0;
        for (int chunkIndex = 0; chunkIndex < MAX_CHUNK_COUNT; chunkIndex++) {
            // Reset block metadata for chunk
            currentBlock->chunkSizes[chunkIndex] = 0;
            currentBlock->lastDocIds[chunkIndex] = 0;
            // Reset postings in chunk
            for (int postingIndex = 0; postingIndex < MAX_POSTING_COUNT; postingIndex++) {
                currentBlock->indexChunks[chunkIndex].postingCount = 0;
                currentBlock->indexChunks[chunkIndex].docIds[postingIndex] = 0;
                currentBlock->indexChunks[chunkIndex].impactScores[postingIndex] = 0.0;
            }
        }
        // Free block structure
        free(currentBlock);
        currentBlock = nextBlock;
    }
    // Free index structure
    free(invertedIndex);
}
