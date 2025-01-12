/* QueryHeap.c */
#include "QueryHeap.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * Creates and initializes new query result heap
 * @return Pointer to newly created heap
 */
QueryHeap *createHeap() {
    QueryHeap *heap = (QueryHeap *)malloc(sizeof(QueryHeap));
    if (heap == NULL) {
        printf("Error allocating memory!\n");
        exit(1);
    }
    heap->nodeCount = 0;
    return heap;
}

/**
 * Frees memory associated with query result heap
 * @param heap Heap to be freed
 */
void freeHeap(QueryHeap *heap) {
    free(heap);
}

/**
 * Swaps two nodes in heap array
 * @param heapNodes Array of nodes
 * @param i First node index
 * @param j Second node index
 */
void swapHeapNodes(QueryHeapNode *heapNodes, int i, int j) {
    QueryHeapNode temp = heapNodes[i];
    heapNodes[i] = heapNodes[j];
    heapNodes[j] = temp;
}

/**
 * Maintains min-heap property by bubbling down
 * @param heap Heap to heapify
 * @param i Index to start from
 */
void heapify(QueryHeap *heap, int i) {
    while (i < heap->nodeCount) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        int smallest = i;
        // Compare with left child, impact score in ascending order
        if (left < heap->nodeCount) {
            if (heap->heapNodes[left].impactScore < heap->heapNodes[smallest].impactScore) {
                smallest = left;
            }
        }
        // Compare with right child, impact score in ascending order
        if (right < heap->nodeCount) {
            if (heap->heapNodes[right].impactScore < heap->heapNodes[smallest].impactScore) {
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
 * Builds a min-heap from an array of nodes
 * @param heap Heap to build
 */
void buildHeap(QueryHeap *heap) {
    for (int i = heap->nodeCount / 2 - 1; i >= 0; i--) {
        heapify(heap, i);
    }
}

/**
 * Extracts the minimum node from the heap
 * @param heap Heap to extract from
 * @return Minimum node
 */
QueryHeapNode extractMin(QueryHeap *heap) {
    QueryHeapNode min = heap->heapNodes[0];
    heap->heapNodes[0] = heap->heapNodes[heap->nodeCount - 1];
    heap->nodeCount--;
    heapify(heap, 0);
    return min;
}

/**
 * Inserts a new node to the heap
 * @param heap Heap to insert to
 * @param heapNode Node to insert
 */
void insertHeapNode(QueryHeap *heap, QueryHeapNode heapNode) {
    heap->heapNodes[heap->nodeCount] = heapNode;
    heap->nodeCount++;
    // Bubble up new node if needed, impact score in ascending order
    int i = heap->nodeCount - 1;
    while (i > 0) {
        int parentIndex = (i - 1) / 2;
        if (heap->heapNodes[i].impactScore < heap->heapNodes[parentIndex].impactScore) {
            swapHeapNodes(heap->heapNodes, i, parentIndex);
            i = parentIndex;
        } else {
            break;
        }
    }
}

/**
 * Sorts heap contents by impact score
 * After sorting, heapNodes[0] has the highest score, heapNodes[nodeCount-1] has the lowest score
 * @param heap Heap to sort
 */
void heapSort(QueryHeap *heap) {
    buildHeap(heap);
    int originalSize = heap->nodeCount;
    for (int i = heap->nodeCount - 1; i > 0; i--) {
        swapHeapNodes(heap->heapNodes, 0, i);
        heap->nodeCount--;
        heapify(heap, 0);
    }
    heap->nodeCount = originalSize;
}
