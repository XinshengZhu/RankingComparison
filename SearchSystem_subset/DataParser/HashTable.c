/* HashTable.c */
#include "HashTable.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Adds a new node to the linked list
 * @param list Linked list to add to
 * @param docId Document ID for the new node
 */
void addLinkedListNode(LinkedList *list, const int docId) {
    LinkedListNode *newNode = (LinkedListNode *)malloc(sizeof(LinkedListNode));
    newNode->docId = docId;
    newNode->frequency = 1;
    newNode->next = NULL;
    // Add to empty list or append to existing list
    if (list->nodeCount == 0) {
        list->headNode = newNode;
        list->tailNode = newNode;
    } else {
        list->tailNode->next = newNode;
        list->tailNode = newNode;
    }
    list->nodeCount++;
}

/**
 * Updates frequency if document exists in list, otherwise adds new node
 * @param list Linked list to update
 * @param docId Document ID to update/add
 */
void updateLinkedList(LinkedList *list, const int docId) {
    LinkedListNode *currentNode = list->tailNode;
    // If same document as last node, increment frequency
    if (currentNode->docId == docId) {
        currentNode->frequency += 1;
        return;
    }
    // Otherwise add new node
    addLinkedListNode(list, docId);
}

/**
 * Calculates hash value for a word string using the DJB2 algorithm
 * @param word Input word string to be hashed
 * @return Computed hash value (modulo table size)
 */
int hashFunction (const char *word) {
    int hash = 5381;    // Initial value for DJB2 algorithm
    for (int i = 0; word[i] != '\0'; i++) {
        hash = ((hash << 5) + hash) + word[i];  // hash * 33 + c
    }
    int result = hash % HASHTABLE_SLOT_SIZE;
    // Handle negative hash values
    if (result < 0) {
        result += HASHTABLE_SLOT_SIZE;
    }
    return result;
}

/**
 * Creates and initializes a new hash table
 * @return Pointer to the newly created hash table
 */
HashTable *createHashTable() {
    HashTable *newTable = (HashTable *)malloc(sizeof(HashTable));
    if (newTable == NULL) {
        printf("Error allocating memory!\n");
        exit(1);
    }
    newTable->wordCount = 0;
    memset(newTable->slots, 0, sizeof(HashTableEntry *) * HASHTABLE_SLOT_SIZE);
    return newTable;
}

/**
 * Frees all memory associated with the hash table
 * @param table Pointer to the hash table to be freed
 */
void freeHashTable(HashTable *table) {
    for (int slotIndex = 0; slotIndex < HASHTABLE_SLOT_SIZE; slotIndex++) {
        HashTableEntry *currentEntry = table->slots[slotIndex];
        while (currentEntry != NULL) {
            HashTableEntry *tempEntry = currentEntry;
            currentEntry = currentEntry->next;
            free(tempEntry->word);
            // Free the linked list nodes
            LinkedListNode *currentNode = tempEntry->list->headNode;
            while (currentNode != NULL) {
                LinkedListNode *tempNode = currentNode;
                currentNode = currentNode->next;
                free(tempNode);
            }
            free(tempEntry->list);
            free(tempEntry);
        }
    }
    free(table);
}

/**
 * Adds a new entry to the hash table
 * @param table Hash table to add to
 * @param word Word to be added
 * @param docId Document ID where the word appears
 */
void addHashTableEntry(HashTable *table, const char *word, const int docId) {
    int slotIndex = hashFunction(word);
    // Create new linked list for the entry
    LinkedList *newList = (LinkedList *)malloc(sizeof(LinkedList));
    newList->nodeCount = 0;
    newList->headNode = NULL;
    newList->tailNode = NULL;
    addLinkedListNode(newList, docId);
    // Create and initialize new hash table entry
    HashTableEntry *newEntry = (HashTableEntry *)malloc(sizeof(HashTableEntry));
    newEntry->wordLength = (int)strlen(word);
    newEntry->word = (char *)malloc(strlen(word) + 1);
    strcpy(newEntry->word, word);
    // Add to front of the chain at the computed slot
    newEntry->list = newList;
    newEntry->next = table->slots[slotIndex];
    table->slots[slotIndex] = newEntry;
    table->wordCount++;
}

/**
 * Updates existing entry or adds new entry to hash table
 * @param table Hash table to update
 * @param word Word to be updated/added
 * @param docId Document ID where the word appears
 */
void updateHashTable(HashTable *table, const char *word, const int docId) {
    int slotIndex = hashFunction(word);
    // Search for existing entry in the chain
    HashTableEntry *currentEntry = table->slots[slotIndex];
    while (currentEntry != NULL) {
        if (strcmp(currentEntry->word, word) == 0) {
            updateLinkedList(currentEntry->list, docId);
            return;
        }
        currentEntry = currentEntry->next;
    }
    // If not found, add new entry
    addHashTableEntry(table, word, docId);
}

int compare(const void *a, const void *b) {
    return strcmp(*(const char **)a, *(const char **)b);
}

/**
 * Returns an array of all words in the hash table, sorted alphabetically
 * @param table Hash table to extract words from
 * @return Array of pointers to words
 */
char **getSortedWordsFromHashTable(const HashTable *table) {
    char **words = (char **)malloc(sizeof(char *) * table->wordCount);
    int wordIndex = 0;
    // Collect all words from the hash table
    for (int slotIndex = 0; slotIndex < HASHTABLE_SLOT_SIZE; slotIndex++) {
        HashTableEntry *currentEntry = table->slots[slotIndex];
        while (currentEntry != NULL) {
            words[wordIndex] = currentEntry->word;
            wordIndex++;
            currentEntry = currentEntry->next;
        }
    }
    // Sort words alphabetically using quicksort
    qsort(words, wordIndex, sizeof(char *), compare);
    return words;
}
