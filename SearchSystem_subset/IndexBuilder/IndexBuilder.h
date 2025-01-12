/* IndexBuilder.h */
#ifndef INDEX_BUILDER_H
#define INDEX_BUILDER_H

#include "MergeHeap.h"
#include "InvertedIndex.h"
#include "Lexicon.h"
#include <stdio.h>

/* Number of intermediate files to merge, depends on the number of the intermediate files from the previous phase */
#define INTERMEDIATE_FILE_COUNT 2
/* Buffer size for reading intermediate file from disk (48MB) */
#define READ_SIZE (48 * 1024 * 1024)
/* Total number of documents in the collection.tsv dataset, hardcoded */
#define DOC_COUNT 1000000

/* Function prototypes */
int *loadDocLengthsFromDisk();  // Load document lengths from disk
void *mapIntermediateContentFromDisk(FILE *file, const size_t fileSize, size_t *offset, size_t *remainingFileSize, void **remainingBuffer, size_t *remainingBufferSize);    // Map intermediate file content from disk
void addParsedItemsToInvertedIndex(ParsedItem **parsedItems, InvertedIndex *invertedIndex, Lexicon *lexicon, const int *docLengths, int totalDocCount, int avgDocLength);   // Add parsed items of a word to inverted index, update lexicon
void writeInvertedIndexToDisk(const InvertedIndex *invertedIndex, const char *outputFileName);  // Write inverted index to disk
void writeLexiconToDisk(const Lexicon *lexicon);    // Write lexicon to disk
void buildIndex();  // Build inverted index from intermediate files, compress and write to disk

#endif
