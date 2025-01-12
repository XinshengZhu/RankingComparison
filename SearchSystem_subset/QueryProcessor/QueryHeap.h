/* QueryHeap.h */
#ifndef QUERY_HEAP_H
#define QUERY_HEAP_H

/* Maximum number of results of query processing to track, changeable */
#define QUERY_HEAP_SIZE 1000

/* Node in query result heap, stores document ID and its impact score */
typedef struct QueryHeapNode {
    int docId;  // Document ID
    double impactScore; // Impact score of the document
} QueryHeapNode;

/* Min-heap for tracking top query results, contains the current number of nodes and the array of heap nodes */
typedef struct QueryHeap {
    int nodeCount;
    QueryHeapNode heapNodes[QUERY_HEAP_SIZE];
} QueryHeap;

/* Function prototypes */
QueryHeap *createHeap();    // Create a query heap
void freeHeap(QueryHeap *heap); // Free memory allocated for a query heap
void swapHeapNodes(QueryHeapNode *heapNodes, int i, int j);  // Swap two heap nodes
void heapify(QueryHeap *heap, int i);   // Heapify the heap
void buildHeap(QueryHeap *heap);    // Build a heap from an array of heap nodes
QueryHeapNode extractMin(QueryHeap *heap);  // Extract the minimum node from the heap
void insertHeapNode(QueryHeap *heap, QueryHeapNode heapNode);   // Insert a new node to the heap
void heapSort(QueryHeap *heap); // Sort the heap in descending order

#endif
