/* MergeHeap.c */
#include "MergeHeap.h"
#include <stdio.h>
#include <string.h>

/**
 * Creates a parsed item structure
 * @param word Word string
 * @param postingCount Number of postings, i.e., number of documents containing the word
 * @param docIds Array of document IDs
 * @param frequencies Array of frequencies of the word in the corresponding documents
 * @return Newly created parsed item
 */
ParsedItem *createParsedItem(char *word, int postingCount, int *docIds, int *frequencies) {
    ParsedItem *parsedItem = (ParsedItem *)malloc(sizeof(ParsedItem));
    if (parsedItem == NULL) {
        printf("Error allocating memory!\n");
        exit(1);
    }
    parsedItem->word = word;
    parsedItem->postingCount = postingCount;
    parsedItem->docIds = docIds;
    parsedItem->frequencies = frequencies;
    return parsedItem;
}

/**
 * Frees memory associated with a parsed item
 * @param parsedItem Item to be freed
 */
void freeParsedItem(ParsedItem *parsedItem) {
    free(parsedItem->word);
    for (int postingIndex = 0; postingIndex < parsedItem->postingCount; postingIndex++) {
        parsedItem->docIds[postingIndex] = 0;
        parsedItem->frequencies[postingIndex] = 0;
    }
    parsedItem->postingCount = 0;
    free(parsedItem->docIds);
    free(parsedItem->frequencies);
    free(parsedItem);
}

/**
 * Frees memory associated with an array of parsed items
 * @param parsedItems Array of parsed items to be freed
 */
void freeParsedItems(ParsedItem **parsedItems) {
    for (int itemIndex = 0; itemIndex < MERGE_HEAP_SIZE; itemIndex++) {
        if (parsedItems[itemIndex] != NULL) {
            freeParsedItem(parsedItems[itemIndex]);
            parsedItems[itemIndex] = NULL;
        }
    }
}

/**
 * Converts binary data from the source intermediate file to a parsed item structure
 * @param remainingBuffer Pointer to the pointer of current position in buffer, i.e., remaining buffer to be parsed
 * @param remainingBufferSize Pointer to the size of the remaining buffer
 * @return Newly created parsed item or NULL if insufficient data
 */
ParsedItem *convertBinaryToParsedItem(void **remainingBuffer, size_t *remainingBufferSize) {
    // Check if enough data for word length
    if (*remainingBufferSize < sizeof(int)) {
        return NULL;
    }
    int wordLength = *((int *)*remainingBuffer);
    // Check if enough data for word
    if (*remainingBufferSize < sizeof(int) + wordLength) {
        return NULL;
    }
    char *word = (char *)malloc(wordLength + 1);
    memcpy(word, *remainingBuffer + sizeof(int), wordLength);
    word[wordLength] = '\0';
    // Check if enough data for posting count
    if (*remainingBufferSize < sizeof(int) + wordLength + sizeof(int)) {
        free(word);
        return NULL;
    }
    int postingCount = *((int *)(*remainingBuffer + sizeof(int) + wordLength));
    // Check if enough data for doc IDs and frequencies
    if (*remainingBufferSize < sizeof(int) + wordLength + sizeof(int) + postingCount * sizeof(int) * 2) {
        free(word);
        return NULL;
    }
    // Parse document IDs and frequencies
    int *docIds = (int *)malloc(postingCount * sizeof(int));
    int *frequencies = (int *)malloc(postingCount * sizeof(int));
    memcpy(docIds, *remainingBuffer + sizeof(int) + wordLength + sizeof(int), postingCount * sizeof(int));
    memcpy(frequencies, *remainingBuffer + sizeof(int) + wordLength + sizeof(int) + postingCount * sizeof(int), postingCount * sizeof(int));
    // Create and initialize parsed item
    ParsedItem *parsedItem = createParsedItem(word, postingCount, docIds, frequencies);
    // Update buffer position and size
    *remainingBuffer = *remainingBuffer + sizeof(int) + wordLength + sizeof(int) + postingCount * sizeof(int) * 2;
    *remainingBufferSize = *remainingBufferSize - (sizeof(int) + wordLength + sizeof(int) + postingCount * sizeof(int) * 2);
    return parsedItem;
}

/**
 * Creates and initializes a new merge heap
 * @return Pointer to newly created heap
 */
MergeHeap *createHeap() {
    MergeHeap *heap = (MergeHeap *)malloc(sizeof(MergeHeap));
    if (heap == NULL) {
        printf("Error allocating memory!\n");
        exit(1);
    }
    heap->nodeCount = 0;
    return heap;
}

/**
 * Frees memory associated with merge heap
 * @param heap Heap to be freed
 */
void freeHeap(MergeHeap *heap) {
    free(heap);
}

/**
 * Swaps two nodes in the heap
 * @param heapNodes Array of heap nodes
 * @param i First node index
 * @param j Second node index
 */
void swapHeapNodes(MergeHeapNode *heapNodes, int i, int j) {
    MergeHeapNode temp = heapNodes[i];
    heapNodes[i] = heapNodes[j];
    heapNodes[j] = temp;
}

/**
 * Maintains heap property by moving node down the tree
 * @param heap Heap to heapify
 * @param i Index to start from
 */
void heapify(MergeHeap *heap, int i) {
    while (i < heap->nodeCount) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        int smallest = i;
        // Compare with left child, word in alphabetical order first, then file number in ascending order
        if (left < heap->nodeCount) {
            int cmpLeft = strcmp(heap->heapNodes[left].parsedItem->word, heap->heapNodes[smallest].parsedItem->word);
            if (cmpLeft < 0 || (cmpLeft == 0 && heap->heapNodes[left].fileNumber < heap->heapNodes[smallest].fileNumber)) {
                smallest = left;
            }
        }
        // Compare with right child, word in alphabetical order first, then file number in ascending order
        if (right < heap->nodeCount) {
            int cmpRight = strcmp(heap->heapNodes[right].parsedItem->word, heap->heapNodes[smallest].parsedItem->word);
            if (cmpRight < 0 || (cmpRight == 0 && heap->heapNodes[right].fileNumber < heap->heapNodes[smallest].fileNumber)) {
                smallest = right;
            }
        }
        // Swap if needed and continue
        if (smallest != i) {
            swapHeapNodes(heap->heapNodes, i, smallest);
            i = smallest;
        } else {
            break;
        }
    }
}

/**
 * Builds heap from unordered array of nodes
 * @param heap Heap to build
 */
void buildHeap(MergeHeap *heap) {
    for (int i = heap->nodeCount / 2 - 1; i >= 0; i--) {
        heapify(heap, i);
    }
}

/**
 * Extracts minimum element from heap
 * @param heap Heap to extract from
 * @return Minimum heap node
 */
MergeHeapNode extractMin(MergeHeap *heap) {
    MergeHeapNode min = heap->heapNodes[0];
    heap->heapNodes[0] = heap->heapNodes[heap->nodeCount - 1];
    heap->nodeCount--;
    heapify(heap, 0);
    return min;
}

/**
 * Inserts new node into heap
 * @param heap Heap to insert into
 * @param heapNode Node to insert
 */
void insertHeapNode(MergeHeap *heap, MergeHeapNode heapNode) {
    heap->heapNodes[heap->nodeCount] = heapNode;
    heap->nodeCount++;
    // Bubble up new node if needed, word in alphabetical order first, then file number in ascending order
    int i = heap->nodeCount - 1;
    while (i > 0) {
        int parentIndex = (i - 1) / 2;
        int cmp = strcmp(heap->heapNodes[i].parsedItem->word, heap->heapNodes[parentIndex].parsedItem->word);
        if (cmp < 0 || (cmp == 0 && heap->heapNodes[i].fileNumber < heap->heapNodes[parentIndex].fileNumber)) {
            swapHeapNodes(heap->heapNodes, i, parentIndex);
            i = parentIndex;
        } else {
            break;
        }
    }
}
