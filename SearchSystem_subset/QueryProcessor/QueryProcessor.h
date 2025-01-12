/* QueryProcessor.h */
#ifndef QUERY_PROCESSOR_H
#define QUERY_PROCESSOR_H

#include "InvertedList.h"
#include "LexiconTable.h"
#include "QueryHeap.h"

/* Number of index files to search through, depends on the number of the inverted index files from the previous phase */
#define INDEX_FILE_COUNT 1

/* Function prototypes */
char *mapLexiconFileFromDisk(); // Load lexicon file from disk
FILE *getListPointerForWord(int *remainingChunkToStart);    // Get the file pointer for the word
int getNextGEQDocId(InvertedList *invertedList, int docId); // Get the next GEQ docId in the inverted list
QueryHeap *conjunctiveDocumentAtATime(const LexiconTable *lexiconTable, char **words, int wordCount);   // Perform conjunctive query processing, aka AND query
QueryHeap *disjunctiveDocumentAtATime(const LexiconTable *lexiconTable, char **words, int wordCount);   // Perform disjunctive query processing, aka OR query
void queryProcessor();  // Main function for query processing

#endif
