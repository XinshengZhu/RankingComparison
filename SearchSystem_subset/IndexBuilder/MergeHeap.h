#ifndef HEAP_H
#define HEAP_H

#include <stdlib.h>

/* Maximum number of files to merge at once, depends on the number of the intermediate files from the previous phase */
#define MERGE_HEAP_SIZE 2

/* Structure for parsed items from binary intermediate files, contains a word and all postings' information */
typedef struct ParsedItem {
    char *word; // The word string
    int postingCount;   // Number of postings, i.e., number of documents containing the word
    int *docIds;    // Array of document IDs
    int *frequencies;   // Array of frequencies of the word in the corresponding documents
} ParsedItem;


/* Structure for heap nodes, contains the intermediate file number and the parsed item, associates parsed items with their source intermediate file */
typedef struct MergeHeapNode {
    int fileNumber; // Source intermediate file number of the parsed item
    ParsedItem *parsedItem; // Parsed item from the source intermediate file
} MergeHeapNode;

/* Heap structure for merging multiple files, contains the number of nodes and the array of heap nodes */
typedef struct MergeHeap {
    int nodeCount;  // Current number of nodes in the heap
    MergeHeapNode heapNodes[MERGE_HEAP_SIZE];    // Array of heap nodes
} MergeHeap;

/* Function prototypes */
ParsedItem *createParsedItem(char *word, int postingCount, int *docIds, int *frequencies);  // Create a parsed item
void freeParsedItem(ParsedItem *parsedItem);    // Free memory allocated for a parsed item
void freeParsedItems(ParsedItem **parsedItems); // Free memory allocated for an array of parsed items
ParsedItem *convertBinaryToParsedItem(void **remainingBuffer, size_t *remainingBufferSize);   // Convert binary data from source intermediate file to a parsed item
MergeHeap *createHeap();    // Create a merge heap
void freeHeap(MergeHeap *heap); // Free memory allocated for a merge heap
void swapHeapNodes(MergeHeapNode *heapNodes, int i, int j);   // Swap two heap nodes
void heapify(MergeHeap *heap, int i);   // Heapify the heap
void buildHeap(MergeHeap *heap);    // Build a heap from an array of heap nodes
MergeHeapNode extractMin(MergeHeap *heap);  // Extract the minimum node from the heap
void insertHeapNode(MergeHeap *heap, MergeHeapNode heapNode);   // Insert a new node to the heap

#endif
