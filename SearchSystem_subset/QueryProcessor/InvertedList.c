/* InvertedList.c */
#include "InvertedList.h"
#include <stdlib.h>
#include <string.h>

/**
 * Creates and initializes a new inverted list for a word
 * Reads initial chunk and block metadata from file
 * @param listPointer File pointer to index
 * @param word Word string
 * @param remainingChunkToStart Starting chunk number
 * @param remainingChunkToEnd Ending chunk number
 * @return Initialized inverted list
 */
InvertedList * createInvertedList(FILE *listPointer, const char *word, int remainingChunkToStart, int remainingChunkToEnd) {
    InvertedList * invertedList = (InvertedList *)malloc(sizeof(InvertedList));
    if (invertedList == NULL) {
        printf("Error allocating memory!\n");
        exit(1);
    }
    // Initialize word info
    invertedList->word = (char *)malloc(strlen(word) + 1);
    strcpy(invertedList->word, word);
    invertedList->listPointer = listPointer;
    invertedList->currentChunkIndex = remainingChunkToStart;
    invertedList->remainingChunkCount = remainingChunkToEnd;
    // Read block metadata where the initial chunk is located
    fread(invertedList->chunkSizes, sizeof(int), MAX_CHUNK_COUNT, listPointer);
    fread(invertedList->lastDocIds, sizeof(int), MAX_CHUNK_COUNT, listPointer);
    // Skip to the initial chunk
    int sumSize = 0;
    for (int chunkIndex = 0; chunkIndex < MAX_CHUNK_COUNT; chunkIndex++) {
        if (chunkIndex == remainingChunkToStart) {
            break;
        }
        sumSize += invertedList->chunkSizes[chunkIndex];
    }
    fseek(listPointer, sumSize, SEEK_CUR);
    // Initialize postings data of the initial chunk
    invertedList->currentPostingIndex = 0;
    invertedList->postingCount = 0;
    invertedList->postings = malloc(invertedList->chunkSizes[remainingChunkToStart]);
    fread(invertedList->postings, 1, invertedList->chunkSizes[remainingChunkToStart], listPointer);
    for (int postingIndex = 0; postingIndex < MAX_POSTING_COUNT; postingIndex++) {
        invertedList->docIds[postingIndex] = -1;
        invertedList->impactScores[postingIndex] = 0.0;
    }
    return invertedList;
}

/**
 * Frees all memory associated with inverted list
 * @param invertedList List to free
 */
void freeInvertedList(InvertedList *invertedList) {
    free(invertedList->word);
    fclose(invertedList->listPointer);
    for (int chunkIndex = 0; chunkIndex < MAX_CHUNK_COUNT; chunkIndex++) {
        invertedList->chunkSizes[chunkIndex] = 0;
        invertedList->lastDocIds[chunkIndex] = 0;
    }
    free(invertedList->postings);
    for (int postingIndex = 0; postingIndex < MAX_POSTING_COUNT; postingIndex++) {
        invertedList->docIds[postingIndex] = 0;
        invertedList->impactScores[postingIndex] = 0.0;
    }
    free(invertedList);
}

/**
 * Updates inverted list to next chunk
 * Reads new chunk data from file
 * @param invertedList List to update
 */
void updateInvertedListByChunk(InvertedList *invertedList) {
    // Handle block boundary
    if (invertedList->currentChunkIndex == MAX_CHUNK_COUNT - 1) {
        invertedList->currentChunkIndex = 0;
        // Read new block metadata
        fread(invertedList->chunkSizes, sizeof(int), MAX_CHUNK_COUNT, invertedList->listPointer);
        fread(invertedList->lastDocIds, sizeof(int), MAX_CHUNK_COUNT, invertedList->listPointer);
    } else {
        invertedList->currentChunkIndex++;
    }
    // Update chunk tracking info
    invertedList->remainingChunkCount--;
    // Read new compressed postings data
    invertedList->postingCount = 0;
    free(invertedList->postings);
    invertedList->postings = malloc(invertedList->chunkSizes[invertedList->currentChunkIndex]);
    fread(invertedList->postings, 1, invertedList->chunkSizes[invertedList->currentChunkIndex], invertedList->listPointer);
    // Reset decompressed postings arrays
    for (int postingIndex = 0; postingIndex < MAX_POSTING_COUNT; postingIndex++) {
        invertedList->docIds[postingIndex] = -1;
        invertedList->impactScores[postingIndex] = 0.0;
    }
}

/**
 * Updates inverted list to next block
 * Skips remaining chunks in current block
 * @param invertedList List to update
 */
void updateInvertedListByBlock(InvertedList *invertedList) {
    // Skip remaining chunks in current block
    int sumSize = 0;
    for (int chunkIndex = invertedList->currentChunkIndex + 1; chunkIndex < MAX_CHUNK_COUNT; chunkIndex++) {
        sumSize += invertedList->chunkSizes[chunkIndex];
    }
    fseek(invertedList->listPointer, sumSize, SEEK_CUR);
    // Update block tracking info
    invertedList->remainingChunkCount -= MAX_CHUNK_COUNT - invertedList->currentChunkIndex;
    invertedList->currentChunkIndex = 0;
    // Read new block metadata
    fread(invertedList->chunkSizes, sizeof(int), MAX_CHUNK_COUNT, invertedList->listPointer);
    fread(invertedList->lastDocIds, sizeof(int), MAX_CHUNK_COUNT, invertedList->listPointer);
    // Read new compressed postings data
    invertedList->postingCount = 0;
    free(invertedList->postings);
    invertedList->postings = malloc(invertedList->chunkSizes[invertedList->currentChunkIndex]);
    fread(invertedList->postings, 1, invertedList->chunkSizes[invertedList->currentChunkIndex], invertedList->listPointer);
    // Reset decompressed postings arrays
    for (int postingIndex = 0; postingIndex < MAX_POSTING_COUNT; postingIndex++) {
        invertedList->docIds[postingIndex] = -1;
        invertedList->impactScores[postingIndex] = 0.0;
    }
}
