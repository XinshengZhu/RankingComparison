/* QueryProcessor.c */
#include "QueryProcessor.h"
#include "Decompression.h"
#include <ctype.h>
#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/errno.h>
#include <sys/mman.h>
#include <sys/time.h>

/**
 * Memory maps lexicon file for efficient access
 * @return Buffer containing lexicon content
 */
char *mapLexiconFileFromDisk() {
    FILE *lexiconFile = fopen("SearchSystem_subset/build/Lexicon.txt", "r");
    if (lexiconFile == NULL) {
        printf("Error opening file %s!\n", "SearchSystem_subset/build/Lexicon.txt");
        return NULL;
    }
    // Get file size
    fseek(lexiconFile, 0, SEEK_END);
    size_t fileSize = ftell(lexiconFile);
    fseek(lexiconFile, 0, SEEK_SET);
    // Map file to memory
    char *mappedLexiconFile = mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fileno(lexiconFile), 0);
    if (mappedLexiconFile == MAP_FAILED) {
        printf("Error mapping file to memory!\n");
        return NULL;
    }
    // Copy to buffer
    char *buffer = (char *)malloc(fileSize + 1);
    memcpy(buffer, mappedLexiconFile, fileSize);
    buffer[fileSize] = '\0';
    // Clean up mapped content
    if (munmap(mappedLexiconFile, fileSize) == -1) {
        printf("Error unmapping file from memory!\n");
        return NULL;
    }
    fclose(lexiconFile);
    return buffer;
}

/**
 * Locates correct inverted index file and block position for a word
 * @param remainingChunkToStart Chunks to skip (modified in place)
 * @return File pointer positioned at start of the block where the word is located
 */
FILE *getListPointerForWord(int *remainingChunkToStart) {
    int fileNumber = 0;
    FILE *filePointer = NULL;
    // Search through inverted index files
    while (fileNumber < INDEX_FILE_COUNT) {
        char *indexFileName = malloc(60);
        sprintf(indexFileName, "SearchSystem_subset/build/InvertedIndex.bin");
        filePointer = fopen(indexFileName, "rb");
        if (filePointer == NULL) {
            printf("Error opening file %s!\n", indexFileName);
            return NULL;
        }
        free(indexFileName);
        // Get file size and prepare for reading
        fseek(filePointer, 0, SEEK_END);
        size_t fileSize = ftell(filePointer);
        fseek(filePointer, 0, SEEK_SET);
        // Skip chunks until target block is found
        int *chunkSizes = malloc(MAX_CHUNK_COUNT * sizeof(int));
        while (*remainingChunkToStart / MAX_CHUNK_COUNT > 0) {
            fread(chunkSizes, sizeof(int), MAX_CHUNK_COUNT, filePointer);
            fseek(filePointer, MAX_CHUNK_COUNT * sizeof(int), SEEK_CUR);
            int sumSize = 0;
            int chunkEndIndex = MAX_CHUNK_COUNT;
            for (int chunkIndex = 0; chunkIndex < MAX_CHUNK_COUNT; chunkIndex++) {
                if (chunkSizes[chunkIndex] == 0) {
                    chunkEndIndex = chunkIndex;
                    break;
                }
                sumSize += chunkSizes[chunkIndex];
            }
            fseek(filePointer, sumSize, SEEK_CUR);
            // Check if end of file reached, adjust remaining chunks specially if so
            if (ftell(filePointer) == fileSize) {
                *remainingChunkToStart -= chunkEndIndex;
                break;
            }
            *remainingChunkToStart -= MAX_CHUNK_COUNT;
        }
        free(chunkSizes);
        // Check if end of file reached, move to next file if so
        if (ftell(filePointer) == fileSize) {
            fclose(filePointer);
            fileNumber++;
        } else {
            break;
        }
    }
    return filePointer;
}

/**
 * Finds next document ID greater than or equal to target
 * Uses block-level and chunk-level skipping
 * @param invertedList List to search
 * @param docId Target document ID
 * @return Next GEQ document ID or -1 if none exists
 */
int getNextGEQDocId(InvertedList *invertedList, int docId) {
    // Skip blocks if possible
    while (invertedList->remainingChunkCount - (MAX_CHUNK_COUNT - (invertedList->currentChunkIndex + 1)) > 0 && invertedList->lastDocIds[MAX_CHUNK_COUNT - 1] < docId) {
        updateInvertedListByBlock(invertedList);
    }
    // Skip chunks if possible
    while (invertedList->remainingChunkCount > 0 && invertedList->lastDocIds[invertedList->currentChunkIndex] < docId) {
        updateInvertedListByChunk(invertedList);
    }
    // Decompress and search current chunk
    decompressPostings(invertedList);
    for (int postingIndex = 0; postingIndex < invertedList->postingCount; postingIndex++) {
        if (invertedList->docIds[postingIndex] >= docId) {
            invertedList->currentPostingIndex = postingIndex;
            return invertedList->docIds[postingIndex];
        }
    }
    return -1;
}

/**
* Performs disjunctive (OR) document-at-a-time query processing
* Returns top-K documents containing ANY query terms, ranked by impact score
*
* @param lexiconTable Hash table containing term dictionary
* @param words Array of query terms
* @param wordCount Number of query terms
* @return Heap containing top-K results sorted by impact score
*/
QueryHeap *disjunctiveDocumentAtATime(const LexiconTable *lexiconTable, char **words, int wordCount) {
    QueryHeap *heap = createHeap();
    // Allocate and initialize inverted lists for each query term
    InvertedList **invertedLists = (InvertedList **)malloc(wordCount * sizeof(InvertedList *));
    for (int wordIndex = 0; wordIndex < wordCount; wordIndex++) {
        // Look up term in lexicon table
        LexiconEntry *lexiconEntry = findWordInLexiconTable(lexiconTable, words[wordIndex]);
        if (lexiconEntry != NULL) {
            // Term found - create its inverted list
            int remainingChunkToStart = lexiconEntry->startChunk - 1;
            int remainingChunkToEnd = lexiconEntry->endChunk - lexiconEntry->startChunk;
            FILE *lp = getListPointerForWord(&remainingChunkToStart);
            invertedLists[wordIndex] = createInvertedList(lp, words[wordIndex], remainingChunkToStart, remainingChunkToEnd);
        } else {
            // Term not found - mark list as NULL
            invertedLists[wordIndex] = NULL;
        }
    }
    int resultCount = 0;    // Number of results found
    // Arrays to track current state of each list
    int *currentDocIds = (int *)malloc(wordCount * sizeof(int));    // Current docId in each list
    bool *listExhausted = (bool *)malloc(wordCount * sizeof(bool)); // Whether list is exhausted
    bool allListsExhausted = false; // Whether all lists are exhausted
    // Initialize each list's state
    for (int i = 0; i < wordCount; i++) {
        if (invertedLists[i] != NULL) {
            // Get first document from list
            currentDocIds[i] = getNextGEQDocId(invertedLists[i], 0);
            listExhausted[i] = (currentDocIds[i] == -1);
        } else {
            listExhausted[i] = true;
        }
    }
    // Main processing loop
    while (!allListsExhausted) {
        // Find minimum document ID among all current positions
        int minDocId = INT_MAX;
        for (int i = 0; i < wordCount; i++) {
            if (!listExhausted[i] && currentDocIds[i] < minDocId) {
                minDocId = currentDocIds[i];
            }
        }
        // Exit if no more documents
        if (minDocId == INT_MAX) {
            break;
        }
        // Calculate total impact score for current document
        double totalImpactScore = 0.0;
        for (int i = 0; i < wordCount; i++) {
            if (!listExhausted[i] && currentDocIds[i] == minDocId) {
                totalImpactScore += invertedLists[i]->impactScores[invertedLists[i]->currentPostingIndex];
            }
        }
        // Update top-K heap
        if (resultCount < QUERY_HEAP_SIZE) {
            // Heap not full - add directly
            QueryHeapNode heapNode;
            heapNode.docId = minDocId;
            heapNode.impactScore = totalImpactScore;
            insertHeapNode(heap, heapNode);
        } else if (totalImpactScore > heap->heapNodes[0].impactScore) {
            // Heap full but new score better than minimum - replace minimum
            extractMin(heap);
            QueryHeapNode heapNode;
            heapNode.docId = minDocId;
            heapNode.impactScore = totalImpactScore;
            insertHeapNode(heap, heapNode);
        }
        resultCount++;
        // Advance lists that matched current document
        allListsExhausted = true;
        for (int i = 0; i < wordCount; i++) {
            if (!listExhausted[i] && currentDocIds[i] == minDocId) {
                currentDocIds[i] = getNextGEQDocId(invertedLists[i], minDocId + 1);
                listExhausted[i] = (currentDocIds[i] == -1);
            }
            if (!listExhausted[i]) {
                allListsExhausted = false;
            }
        }
    }
    // Clean up
    free(currentDocIds);
    free(listExhausted);
    for (int wordIndex = 0; wordIndex < wordCount; wordIndex++) {
        if (invertedLists[wordIndex] != NULL) {
            freeInvertedList(invertedLists[wordIndex]);
        }
    }
    free(invertedLists);
    // Sort results by impact score
    heapSort(heap);
    return heap;
}

/**
* Splits string into an array of unique words, replacing non-alphanumeric characters with spaces
* @param input Input string (will be modified)
* @param wordCount Pointer to store the final word count
* @return Array of unique word strings
*/
char **splitIntoWords(char *input, int *wordCount) {
    char **words = NULL;    // Final array to return
    char **tempWords = NULL;    // Temporary array for deduplication
    int tempCount = 0;  // Current count in temporary array
    // Replace all non-alphanumeric characters with spaces
    for (int i = 0; input[i] != '\0'; i++) {
        if (!isalnum(input[i])) {
            input[i] = ' ';
        }
    }
    // First pass: count total words
    char *inputCopy = strdup(input);
    char *token = strtok(inputCopy, " \t\n");
    int maxWords = 0;
    while (token != NULL) {
        maxWords++;
        token = strtok(NULL, " \t\n");
    }
    // Allocate temporary array
    tempWords = (char **)malloc(maxWords * sizeof(char *));
    // Second pass: tokenize and remove duplicates
    strcpy(inputCopy, input);
    token = strtok(inputCopy, " \t\n");
    while (token != NULL) {
        // Check if word already exists
        bool isDuplicate = false;
        for (int i = 0; i < tempCount; i++) {
            if (strcmp(tempWords[i], token) == 0) {
                isDuplicate = true;
                break;
            }
        }
        // Add to temporary array if not a duplicate
        if (!isDuplicate) {
            tempWords[tempCount] = strdup(token);
            tempCount++;
        }
        token = strtok(NULL, " \t\n");
    }
    // Set final word count
    *wordCount = tempCount;
    // Allocate final array
    words = (char **)malloc(*wordCount * sizeof(char *));
    // Copy words from temporary to final array
    for (int i = 0; i < *wordCount; i++) {
        words[i] = tempWords[i];
    }
    // Clean up temporary memory
    free(tempWords);
    free(inputCopy);
    return words;
}

/**
* Main query processing function that handles user interaction and search execution
* Provides an interactive interface for users to perform document searches
*/
void queryProcessor() {
    // Initialize lexicon for term lookup
    LexiconTable *lexiconTable = createLexiconTable();
    char *buffer = mapLexiconFileFromDisk();
    convertLexiconFileToLexiconTable(lexiconTable, buffer);
    // Array for query files to process
    char *queryFiles[] = {"generated_data/queries.dev.subset.tsv", "generated_data/queries.eval.one.subset.tsv", "generated_data/queries.eval.two.subset.tsv"};
    char *resultFiles[] = {"generated_data/bm25_result_dev.tsv", "generated_data/bm25_result_eval_one.tsv", "generated_data/bm25_result_eval_two.tsv"};
    // Process each query file
    for (int fileNumber = 0; fileNumber < 3; fileNumber++) {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        // Open query file
        FILE *queryFile = fopen(queryFiles[fileNumber], "r");
        if (queryFile == NULL) {
            printf("Error opening file %s!\n", queryFiles[fileNumber]);
            return;
        }
        // Read query file line by line
        char *line = NULL;
        size_t len = 0;
        while ((getline(&line, &len, queryFile)) != -1) {
            // Remove trailing newline
            line[strcspn(line, "\n")] = 0;
            // Validate input
            if (strlen(line) == 0) {
                printf("Empty input. Please enter some search terms.\n");
                continue;
            }
            // Split line into ID and content
            int id = (int)strtol(strtok(line, "\t"), NULL, 10);
            char *content = strtok(NULL, "\t");
            // Split content into words and validate
            int wordCount;
            char **words = splitIntoWords(content, &wordCount);
            if (words == NULL || wordCount == 0) {
                printf("No valid search terms found.\n");
                continue;
            }
            // Perform disjunctive search
            QueryHeap *heap = NULL;
            heap = disjunctiveDocumentAtATime(lexiconTable, words, wordCount);
            // Display results
            if (heap->nodeCount == 0) {
                printf("No results found.\n");
            } else {
                // Write results to output file
                FILE *resultFile = fopen(resultFiles[fileNumber], "a");
                if (resultFile == NULL) {
                    printf("Error opening file %s!\n", resultFiles[fileNumber]);
                    return;
                }
                for (int i = 0; i < heap->nodeCount; i++) {
                    fprintf(resultFile, "%d\t%d\n", id, heap->heapNodes[i].docId);
                }
                fclose(resultFile);
            }
            // Clean up allocated memory
            for (int i = 0; i < wordCount; i++) {
                free(words[i]);
            }
            free(words);
            freeHeap(heap);
        }
        free(line);
        fclose(queryFile);
        gettimeofday(&end, NULL);
        double elapsed_time = ((double)end.tv_sec - (double)start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
        printf("Search completed for query file %s in %.6f seconds.\n", queryFiles[fileNumber], elapsed_time);
    }
    free(buffer);
    freeLexiconTable(lexiconTable);
}

int main() {
    queryProcessor();
    return 0;
}
