/* InvertedIndex.h */
#ifndef INVERTED_INDEX_H
#define INVERTED_INDEX_H

/* Maximum postings per chunk */
#define MAX_POSTING_COUNT 128
/* Maximum chunks per block */
#define MAX_CHUNK_COUNT 64

/**
 * Represents a chunk of postings in the inverted index
 * A chunk stores document IDs and their corresponding impact scores
 * for a portion of a term's posting list
 */
typedef struct IndexChunk {
    int postingCount;   // Current number of postings in the chunk
    int docIds[MAX_POSTING_COUNT];  // Array of document IDs
    double impactScores[MAX_POSTING_COUNT]; // Array of impact scores
} IndexChunk;

/**
 * Represents a block in the inverted index
 * A block contains multiple chunks and maintains metadata about them
 * Blocks are linked together to handle large posting lists
 */
typedef struct IndexBlock {
    int chunkCount; // Current number of chunks in the block
    int chunkSizes[MAX_CHUNK_COUNT];    // Size (in bytes) of each chunk after compression (for jumping)
    int lastDocIds[MAX_CHUNK_COUNT];    // Last document ID in each chunk (for query processing)
    IndexChunk indexChunks[MAX_CHUNK_COUNT];    // Array of chunks
    struct IndexBlock *nextIndexBlock;  // Pointer to the next block in the index
} IndexBlock;

/**
 * Represents the main inverted index structure
 * Maintains a linked list of blocks and tracks global statistics
 */
typedef struct InvertedIndex {
    int chunkNumber;    // Current number of chunks in the index, used for recording chunk allocation in lexicon
    int blockCount; // Current number of blocks in the index
    IndexBlock *headIndexBlock; // Pointer to the first block in the index
    IndexBlock *tailIndexBlock; // Pointer to the last block in the index
} InvertedIndex;

/* Function prototypes */
IndexBlock *createIndexBlock(); // Create new index block
InvertedIndex *createInvertedIndex();   // Create new inverted index
void freeInvertedIndex(InvertedIndex *invertedIndex);   // Free memory allocated for inverted index

#endif
