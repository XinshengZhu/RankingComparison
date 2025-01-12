/* DataParser.h */
#ifndef DATA_PARSER_H
#define DATA_PARSER_H

#include "HashTable.h"
#include <stdio.h>

/* Buffer size for reading the dataset from disk (192MB) */
#define READ_SIZE (192 * 1024 * 1024)
/* Total number of documents in the collection.tsv dataset, hardcoded */
#define DOC_COUNT 1000000

/* Function prototypes */
char *mapRawContentFromDisk(FILE *file, size_t mapSize, size_t offset, char *remainingContent); // Map raw file content to memory
void writeHashTableToDisk(const HashTable *table, const char *outputFileName);  // Write hash table to binary file
void writeDocLengthsToDisk(const int *docLengths);  // Write document lengths to binary file
void parseData();  // Parse input data file and create intermediate binary files

#endif
