/* IndexBuilder.c */
#include "IndexBuilder.h"
#include "Compression.h"
#include "Utils.h"
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>

/**
 * Reads document length data from binary file
 * This data is used in BM25 score calculations
 *
 * @return Array of document lengths, caller must free
 * @note Exits on file/memory errors
 */
int *loadDocLengthsFromDisk() {
    FILE *docLengthsFile = fopen("SearchSystem_subset/build/DocLengths.bin", "rb");
    if (docLengthsFile == NULL) {
        printf("Error opening file %s!\n", "SearchSystem_subset/build/DocLengths.bin");
        exit(1);
    }
    fseek(docLengthsFile, 0, SEEK_END);
    size_t fileSize = ftell(docLengthsFile);
    fseek(docLengthsFile, 0, SEEK_SET);
    int *docLengths = (int *)malloc(fileSize);
    if (docLengths == NULL) {
        printf("Error allocating memory!\n");
        exit(1);
    }
    fread(docLengths, fileSize, 1, docLengthsFile);
    fclose(docLengthsFile);
    return docLengths;
}

/**
 * Maps a portion of intermediate file content to memory
 * Uses memory mapping for efficient reading of large files
 *
 * @param file File to read from
 * @param fileSize Total file size
 * @param offset Current file offset
 * @param remainingFileSize Remaining bytes in file
 * @param remainingBuffer Previous unprocessed content
 * @param remainingBufferSize Size of remaining buffer
 * @return Newly allocated buffer containing mapped data
 * @note Handles concatenation with previous unprocessed data
 */
void *mapIntermediateContentFromDisk(FILE *file, const size_t fileSize, size_t *offset, size_t *remainingFileSize, void **remainingBuffer, size_t *remainingBufferSize) {
    // Check if we've reached end of file
    if (*offset >= fileSize) {
        return NULL;
    }
    // Calculate size to map
    *remainingFileSize = fileSize - *offset;
    size_t mapSize = (*remainingFileSize < READ_SIZE) ? *remainingFileSize : READ_SIZE;
    // Map file segment to memory
    void *mappedContent = mmap(NULL, mapSize, PROT_READ, MAP_PRIVATE, fileno(file), (off_t)*offset);
    if (mappedContent == MAP_FAILED) {
        printf("Error mapping file!\n");
        exit(1);
    }
    *offset += mapSize;
    // Handle remaining buffer concatenation
    void *newBuffer = NULL;
    if (*remainingBuffer != NULL && *remainingBufferSize > 0) {
        // Concatenate with existing buffer
        newBuffer = malloc(mapSize + *remainingBufferSize);
        memcpy(newBuffer, *remainingBuffer, *remainingBufferSize);
        memcpy(newBuffer + *remainingBufferSize, mappedContent, mapSize);
        *remainingBuffer = newBuffer;
        *remainingBufferSize = mapSize + *remainingBufferSize;
    } else {
        // Create new buffer
        newBuffer = malloc(mapSize);
        memcpy(newBuffer, mappedContent, mapSize);
        *remainingBuffer = newBuffer;
        *remainingBufferSize = mapSize;
    }
    // Clean up mapped content
    if (munmap(mappedContent, mapSize) == -1) {
        printf("Error unmapping file!\n");
        exit(1);
    }
    return newBuffer;
}

/**
 * Adds parsed items to inverted index and lexicon
 * Handles block creation, chunk management, and impact score calculation
 *
 * @param parsedItems Array of parsed items of a word from different files
 * @param invertedIndex Inverted index to add to
 * @param lexicon Lexicon to update
 * @param docLengths Array of document lengths for BM25 score calculation
 * @param totalDocCount Total number of documents for BM25 score calculation
 * @param avgDocLength Average document length for BM25 score calculation
 */
void addParsedItemsToInvertedIndex(ParsedItem **parsedItems, InvertedIndex *invertedIndex, Lexicon *lexicon, const int *docLengths, int totalDocCount, int avgDocLength) {
    // Calculate term document count for BM25
    int termDocCount = computeTermDocCount(parsedItems);
    // Retrieve or create current block
    IndexBlock *currentBlock = invertedIndex->tailIndexBlock;
    if (currentBlock == NULL || currentBlock->chunkCount == MAX_CHUNK_COUNT) {
        IndexBlock *newBlock = createIndexBlock();
        if (currentBlock == NULL) {
            invertedIndex->headIndexBlock = newBlock;
            invertedIndex->tailIndexBlock = newBlock;
        } else {
            currentBlock->nextIndexBlock = newBlock;
            invertedIndex->tailIndexBlock = newBlock;
        }
        invertedIndex->blockCount++;
        currentBlock = newBlock;
    }
    // Initialize variables for tracking
    int chunkIndex = currentBlock->chunkCount;
    int postingCount = currentBlock->indexChunks[chunkIndex].postingCount;
    currentBlock->chunkCount++;
    invertedIndex->chunkNumber++;
    int prevDocId = -1;
    char *word = NULL;
    int startChunk = invertedIndex->chunkNumber;
    // Process each parsed item
    for (int itemIndex = 0; itemIndex < MERGE_HEAP_SIZE; itemIndex++) {
        if (parsedItems[itemIndex] == NULL) {
            continue;
        }
        for (int postingIndex = 0; postingIndex < parsedItems[itemIndex]->postingCount; postingIndex++) {
            word = parsedItems[itemIndex]->word;
            int docId = parsedItems[itemIndex]->docIds[postingIndex];
            int frequency = parsedItems[itemIndex]->frequencies[postingIndex];
            // Calculate BM25 impact score
            double impactScore = calculateBM25ImpactScore(totalDocCount, termDocCount, frequency, docLengths[docId - 1], avgDocLength);
            // Move to next chunk if current chunk is full
            if (postingCount == MAX_POSTING_COUNT) {
                if (chunkIndex == MAX_CHUNK_COUNT - 1) {
                    // Create new block if needed
                    IndexBlock *newBlock = createIndexBlock();
                    currentBlock->nextIndexBlock = newBlock;
                    invertedIndex->tailIndexBlock = newBlock;
                    invertedIndex->blockCount++;
                    currentBlock = newBlock;
                }
                chunkIndex = currentBlock->chunkCount;
                postingCount = currentBlock->indexChunks[chunkIndex].postingCount;
                currentBlock->chunkCount++;
                invertedIndex->chunkNumber++;
                prevDocId = -1;
            }
            // Store impact score and delta-encoded document ID, also update block metadata
            if (prevDocId != -1) {
                currentBlock->indexChunks[chunkIndex].docIds[postingCount] = docId - prevDocId;
                currentBlock->chunkSizes[chunkIndex] += computeVarByteLength(docId - prevDocId);
            } else {
                currentBlock->indexChunks[chunkIndex].docIds[postingCount] = docId;
                currentBlock->chunkSizes[chunkIndex] += computeVarByteLength(docId);
            }
            currentBlock->lastDocIds[chunkIndex] = docId;
            prevDocId = docId;
            currentBlock->indexChunks[chunkIndex].impactScores[postingCount] = impactScore;
            currentBlock->chunkSizes[chunkIndex] += 1;
            currentBlock->indexChunks[chunkIndex].postingCount++;
            postingCount = currentBlock->indexChunks[chunkIndex].postingCount;
        }
    }
    // Add a new node for the word to lexicon with chunk range
    int endChunk = invertedIndex->chunkNumber;
    addNodeToLexicon(lexicon, word, startChunk, endChunk);
}

/**
 * Writes inverted index to binary file
 * Format: For each block:
 * 1. Chunk sizes array (int)
 * 2. Last docIDs array (int)
 * 3. For each chunk:
 *    - VByte compressed delta-encoded docIDs array (1-5 bytes each)
 *    - Log compressed impact scores array (1 byte each)
 *
 * @param invertedIndex Index to write
 * @param outputFileName Target file name
 */
void writeInvertedIndexToDisk(const InvertedIndex *invertedIndex, const char *outputFileName) {
    FILE *invertedIndexFile = fopen(outputFileName, "wb");
    if (invertedIndexFile == NULL) {
        printf("Error opening file %s!\n", outputFileName);
        exit(1);
    }
    IndexBlock *currentBlock = invertedIndex->headIndexBlock;
    while (currentBlock != NULL) {
        // Write block metadata
        fwrite(currentBlock->chunkSizes, sizeof(int), MAX_CHUNK_COUNT, invertedIndexFile);
        fwrite(currentBlock->lastDocIds, sizeof(int), MAX_CHUNK_COUNT, invertedIndexFile);
        // Write each chunk's data
        for (int chunkIndex = 0; chunkIndex < currentBlock->chunkCount; chunkIndex++) {
            // Write compressed docIDs
            for (int postingIndex = 0; postingIndex < currentBlock->indexChunks[chunkIndex].postingCount; postingIndex++) {
                uint8_t byteBuffer[5];
                size_t byteCount = varByteCompressInt(currentBlock->indexChunks[chunkIndex].docIds[postingIndex], byteBuffer);
                fwrite(byteBuffer, 1, byteCount, invertedIndexFile);
            }
            // Write compressed impact scores
            for (int postingIndex = 0; postingIndex < currentBlock->indexChunks[chunkIndex].postingCount; postingIndex++) {
                uint8_t compressedImpactScore = linearCompressDouble(currentBlock->indexChunks[chunkIndex].impactScores[postingIndex]);
                fwrite(&compressedImpactScore, 1, 1, invertedIndexFile);
            }
        }
        currentBlock = currentBlock->nextIndexBlock;
    }
    printf("File %s written with %d blocks in inverted list.\n", outputFileName, invertedIndex->blockCount);
    fclose(invertedIndexFile);
}

/**
 * Writes lexicon to text file (ASCII format)
 * Format: <word> <start_chunk> <end_chunk>
 * One entry per line
 *
 * @param lexicon Lexicon to write
 */
void writeLexiconToDisk(const Lexicon *lexicon) {
    FILE *lexiconFile = fopen("SearchSystem_subset/build/Lexicon.txt", "w");
    if (lexiconFile == NULL) {
        printf("Error opening file %s!\n", "SearchSystem_subset/build/Lexicon.txt");
        exit(1);
    }
    LexiconNode *currentNode = lexicon->headNode;
    while (currentNode != NULL) {
        fprintf(lexiconFile, "%s %d %d\n", currentNode->word, currentNode->startChunk, currentNode->endChunk);
        currentNode = currentNode->next;
    }
    printf("File %s written with %d words in lexicon.\n", "SearchSystem_subset/build/Lexicon.txt", lexicon->nodeCount);
    fclose(lexiconFile);
}

/**
 * Main index building function
 * Merges intermediate files and builds final index structure
 * Process:
 * 1. Open all intermediate files
 * 2. Use min-heap for merging
 * 3. Build index incrementally
 * 4. Write results to disk
 */
void buildIndex() {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    // Initialize file handling arrays
    FILE *intermediateFiles[INTERMEDIATE_FILE_COUNT];
    size_t fileSizes[INTERMEDIATE_FILE_COUNT];
    size_t offsets[INTERMEDIATE_FILE_COUNT];
    size_t remainingFileSizes[INTERMEDIATE_FILE_COUNT];
    void *buffer[INTERMEDIATE_FILE_COUNT];
    size_t remainingBufferSizes[INTERMEDIATE_FILE_COUNT];
    void *remainingBuffer[INTERMEDIATE_FILE_COUNT];
    // Create merge heap
    MergeHeap *heap = createHeap();
    // Open and initialize all intermediate files
    for (int fileIndex = 0; fileIndex < INTERMEDIATE_FILE_COUNT; fileIndex++) {
        // Open file
        char *intermediateFileName = (char *)malloc(50);
        sprintf(intermediateFileName, "SearchSystem_subset/build/Intermediate%d.bin", fileIndex);
        intermediateFiles[fileIndex] = fopen(intermediateFileName, "r");
        if (intermediateFiles[fileIndex] == NULL) {
            printf("Error opening file %s!\n", intermediateFileName);
            exit(1);
        }
        // Get file size
        fseek(intermediateFiles[fileIndex], 0, SEEK_END);
        fileSizes[fileIndex] = ftell(intermediateFiles[fileIndex]);
        fseek(intermediateFiles[fileIndex], 0, SEEK_SET);
        // Initialize buffer tracking variables
        offsets[fileIndex] = 0;
        remainingFileSizes[fileIndex] = fileSizes[fileIndex];
        remainingBuffer[fileIndex] = NULL;
        remainingBufferSizes[fileIndex] = 0;
        // Map initial intermediate content
        buffer[fileIndex] = mapIntermediateContentFromDisk(intermediateFiles[fileIndex], fileSizes[fileIndex], &offsets[fileIndex], &remainingFileSizes[fileIndex], &remainingBuffer[fileIndex], &remainingBufferSizes[fileIndex]);
        // Add first parsed item to heap
        ParsedItem *parsedItem = convertBinaryToParsedItem(&remainingBuffer[fileIndex], &remainingBufferSizes[fileIndex]);
        if (parsedItem != NULL) {
            MergeHeapNode heapNode;
            heapNode.fileNumber = fileIndex;
            heapNode.parsedItem = parsedItem;
            insertHeapNode(heap, heapNode);
        }
    }
    // Load document statistics
    int *docLengths = loadDocLengthsFromDisk();
    int totalDocCount = DOC_COUNT;
    int avgDocLength = computeAvgDocLength(docLengths, totalDocCount);
    // Initialize index structures
    Lexicon *lexicon = createLexicon();
    InvertedIndex *invertedIndex = createInvertedIndex();
    // Initialize merge tracking variables
    ParsedItem *parsedItems[MERGE_HEAP_SIZE];
    ParsedItem *newParsedItem = NULL;
    void* tempBuffer = NULL;
    // Main merge loop
    while (heap->nodeCount > 0) {
        // Get minimum item
        MergeHeapNode min = extractMin(heap);
        parsedItems[min.fileNumber] = min.parsedItem;
        // Try to get next item from same file
        newParsedItem = convertBinaryToParsedItem(&remainingBuffer[min.fileNumber], &remainingBufferSizes[min.fileNumber]);
        if (newParsedItem != NULL) {
            // Add to heap if available
            MergeHeapNode heapNode;
            heapNode.fileNumber = min.fileNumber;
            heapNode.parsedItem = newParsedItem;
            insertHeapNode(heap, heapNode);
        } else {
            // Try to map more content
            tempBuffer = mapIntermediateContentFromDisk(intermediateFiles[min.fileNumber], fileSizes[min.fileNumber], &offsets[min.fileNumber], &remainingFileSizes[min.fileNumber], &remainingBuffer[min.fileNumber], &remainingBufferSizes[min.fileNumber]);
            free(buffer[min.fileNumber]);
            buffer[min.fileNumber] = tempBuffer;
            if (buffer[min.fileNumber] == NULL) {
                continue;
            }
            newParsedItem = convertBinaryToParsedItem(&remainingBuffer[min.fileNumber], &remainingBufferSizes[min.fileNumber]);
            if (newParsedItem != NULL) {
                MergeHeapNode heapNode;
                heapNode.fileNumber = min.fileNumber;
                heapNode.parsedItem = newParsedItem;
                insertHeapNode(heap, heapNode);
            }
        }
        // Collect all items with same word
        while(heap->nodeCount > 0 && strcmp(parsedItems[min.fileNumber]->word, heap->heapNodes[0].parsedItem->word) == 0) {
            min = extractMin(heap);
            parsedItems[min.fileNumber] = min.parsedItem;
            // Similar process for getting next item from same file
            newParsedItem = convertBinaryToParsedItem(&remainingBuffer[min.fileNumber], &remainingBufferSizes[min.fileNumber]);
            if (newParsedItem != NULL) {
                MergeHeapNode heapNode;
                heapNode.fileNumber = min.fileNumber;
                heapNode.parsedItem = newParsedItem;
                insertHeapNode(heap, heapNode);
            } else {
                tempBuffer = mapIntermediateContentFromDisk(intermediateFiles[min.fileNumber], fileSizes[min.fileNumber], &offsets[min.fileNumber], &remainingFileSizes[min.fileNumber], &remainingBuffer[min.fileNumber], &remainingBufferSizes[min.fileNumber]);
                free(buffer[min.fileNumber]);
                buffer[min.fileNumber] = tempBuffer;
                if (buffer[min.fileNumber] == NULL) {
                    continue;
                }
                newParsedItem = convertBinaryToParsedItem(&remainingBuffer[min.fileNumber], &remainingBufferSizes[min.fileNumber]);
                if (newParsedItem != NULL) {
                    MergeHeapNode heapNode;
                    heapNode.fileNumber = min.fileNumber;
                    heapNode.parsedItem = newParsedItem;
                    insertHeapNode(heap, heapNode);
                }
            }
        }
        // Add collected items for the same word to index
        addParsedItemsToInvertedIndex(parsedItems, invertedIndex, lexicon, docLengths, totalDocCount, avgDocLength);
        freeParsedItems(parsedItems);
    }
    // Write out the index and clean up memory
    char *outputFileName = (char *)malloc(60);
    sprintf(outputFileName, "SearchSystem_subset/build/InvertedIndex.bin");
    writeInvertedIndexToDisk(invertedIndex, outputFileName);
    freeInvertedIndex(invertedIndex);
    // Write out the lexicon and clean up memory
    writeLexiconToDisk(lexicon);
    freeLexicon(lexicon);
    freeHeap(heap);
    // Close files and free buffers
    for (int fileIndex = 0; fileIndex < INTERMEDIATE_FILE_COUNT; fileIndex++) {
        fclose(intermediateFiles[fileIndex]);
        free(buffer[fileIndex]);
    }
    free(docLengths);
    gettimeofday(&end, NULL);
    double elapsed_time = ((double)end.tv_sec - (double)start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("Index built in %.6f seconds.\n", elapsed_time);
}

int main() {
    buildIndex();
    return 0;
}
