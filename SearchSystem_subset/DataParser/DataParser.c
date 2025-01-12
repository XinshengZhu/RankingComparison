/* DataParser.c */
#include "DataParser.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>

/**
 * Maps a portion of file content to memory and processes special characters
 * @param file File pointer to read from
 * @param mapSize Number of bytes to map
 * @param offset Starting position in file (has to be a multiple of page size)
 * @param remainingContent Previous unprocessed content (used for concatenation, if any)
 * @return Processed content buffer
 */
char *mapRawContentFromDisk(FILE *file, size_t mapSize, size_t offset, char *remainingContent) {
    // Memory map the file segment
    char *mapContent = mmap(NULL, mapSize, PROT_READ, MAP_PRIVATE, fileno(file), (off_t)offset);
    if (mapContent == MAP_FAILED) {
        printf("Error mapping file!\n");
        exit(1);
    }
    // Process special characters, keeping only alphanumeric, tab, and newline
    char *tempBuffer = (char *)malloc(mapSize + 1);
    for (size_t i = 0; i < mapSize; i++) {
        if (mapContent[i] == '\t' || mapContent[i] == '\n' || isalnum(mapContent[i])) {
            tempBuffer[i] = mapContent[i];
        } else {
            tempBuffer[i] = ' ';
        }
    }
    tempBuffer[mapSize] = '\0';
    // Combine with remaining content if exists
    char *buffer = NULL;
    if (remainingContent != NULL) {
        buffer = (char *)malloc(strlen(remainingContent) + mapSize + 1);
        strcpy(buffer, remainingContent);
        strcat(buffer, tempBuffer);
        free(remainingContent);
    } else {
        buffer = (char *)malloc(mapSize + 1);
        strcpy(buffer, tempBuffer);
    }
    free(tempBuffer);
    // Clean up mapped content
    if (munmap(mapContent, mapSize) == -1) {
        printf("Error unmapping file!\n");
    }
    return buffer;
}

/**
 * Writes hash table contents to a binary file in alphabetically sorted order of words
 * @param table Hash table to write
 * @param outputFileName Name of output file
 */
void writeHashTableToDisk(const HashTable *table, const char *outputFileName) {
    FILE *file = fopen(outputFileName, "wb");
    if (file == NULL) {
        printf("Error opening file %s!\n", outputFileName);
        exit(1);
    }
    // Get sorted list of words from hash table
    char **words = getSortedWordsFromHashTable(table);
    // Write each word's data in binary format
    for (int wordIndex = 0; wordIndex < table->wordCount; wordIndex++) {
        HashTableEntry *currentEntry = table->slots[hashFunction(words[wordIndex])];
        while (currentEntry != NULL) {
            if (strcmp(currentEntry->word, words[wordIndex]) == 0) {
                LinkedList *list = currentEntry->list;
                // Write word length and word
                fwrite(&currentEntry->wordLength, sizeof(int), 1, file);
                fwrite(currentEntry->word, sizeof(char), strlen(currentEntry->word), file);
                // Write number of postings
                fwrite(&list->nodeCount, sizeof(int), 1, file);
                // Write docId for each posting, not interleaved
                LinkedListNode *currentNode = list->headNode;
                while (currentNode != NULL) {
                    fwrite(&currentNode->docId, sizeof(int), 1, file);
                    currentNode = currentNode->next;
                }
                // Write frequency for each posting, not interleaved
                currentNode = list->headNode;
                while (currentNode != NULL) {
                    fwrite(&currentNode->frequency, sizeof(int), 1, file);
                    currentNode = currentNode->next;
                }
                break;
            }
            currentEntry = currentEntry->next;
        }
    }
    printf("File %s written with %d words in hash table.\n", outputFileName, table->wordCount);
    free(words);
    fclose(file);
}

/**
 * Writes document lengths to a binary file
 * @param docLengths Array of document lengths
 */
void writeDocLengthsToDisk(const int *docLengths) {
    FILE *file = fopen("SearchSystem_subset/build/DocLengths.bin", "wb");
    if (file == NULL) {
        printf("Error opening file %s!\n", "SearchSystem_subset/build/DocLengths.bin");
        exit(1);
    }
    // Write document lengths in binary format
    for (int docId = 0; docId < DOC_COUNT; docId++) {
        fwrite(&docLengths[docId], sizeof(int), 1, file);
    }
    printf("File %s written with %d document lengths.\n", "SearchSystem_subset/build/DocLengths.bin", DOC_COUNT);
    fclose(file);
}

/**
 * Main function for parsing data
 * Parses input data file and creates intermediate binary files
 */
void parseData() {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    FILE *file = fopen("generated_data/collection_subset.tsv", "r");
    if (file == NULL) {
        printf("Error opening file %s!\n", "generated_data/collection_subset.tsv");
        exit(1);
    }
    // Get file size
    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);
    // Initialize processing variables
    size_t offset = 0;
    char *remainingContent = NULL;
    int fileNumber = 0;
    int *docLengths = (int *)malloc(sizeof(int) * DOC_COUNT);
    // Process file in segments of READ_SIZE bytes
    while (offset < fileSize) {
        // Determine size of next segment to read
        size_t remainingFileSize = fileSize - offset;
        size_t mapSize = (remainingFileSize < READ_SIZE) ? remainingFileSize : READ_SIZE;
        // Map the segment of file content to memory and process
        char *buffer = mapRawContentFromDisk(file, mapSize, offset, remainingContent);
        char *lineStart = buffer;
        HashTable *table = createHashTable();
        // Process each line in the segment
        while (*lineStart != '\0') {
            char *lineEnd = strchr(lineStart, '\n');
            if (lineEnd != NULL) {
                *lineEnd = '\0';
            } else {
                // If lineEnd is NULL, then save the incomplete line for next segment
                remainingContent = (char *)malloc(strlen(lineStart) + 1);
                strcpy(remainingContent, lineStart);
                break;
            }
            // Split line by tab character, separating docId and document content
            char *tabPos = strchr(lineStart, '\t');
            if (tabPos != NULL) {
                *tabPos = '\0';
                int docId = (int)strtol(lineStart, NULL, 10);
                int wordCount = 0;
                // Split words by space character and process each word
                char *wordsStart = tabPos + 1;
                char *wordStart = wordsStart;
                while (*wordStart != '\0') {
                    char *spacePos = strchr(wordStart, ' ');
                    if (spacePos != NULL) {
                        *spacePos = '\0';
                        if (spacePos == wordStart) {
                            wordStart = spacePos + 1;
                            continue;
                        }
                    }
                    // Update hash table with word and docId
                    updateHashTable(table, wordStart, docId);
                    wordCount++;
                    if (spacePos != NULL) {
                        wordStart = spacePos + 1;
                    } else {
                        wordStart = lineEnd;
                    }
                }
                docLengths[docId] = wordCount;
            }
            lineStart = lineEnd + 1;
        }
        // Write the hash table as intermediate results to disk
        char *outputFileName = (char *)malloc(60);
        sprintf(outputFileName, "SearchSystem_subset/build/Intermediate%d.bin", fileNumber);
        fileNumber++;
        writeHashTableToDisk(table, outputFileName);
        // Clean up and prepare for next segment
        freeHashTable(table);
        offset += mapSize;
        free(buffer);
    }
    fclose(file);
    // Write document lengths to disk
    writeDocLengthsToDisk(docLengths);
    free(docLengths);
    gettimeofday(&end, NULL);
    double elapsed_time = ((double)end.tv_sec - (double)start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("Data parsed in %.6f seconds.\n", elapsed_time);
    // Create page table for documents using sqlite3 database
    gettimeofday(&start, NULL);
}

int main() {
    parseData();
    return 0;
}
