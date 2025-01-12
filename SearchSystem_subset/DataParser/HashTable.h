/* HashTable.h */
#ifndef HASH_TABLE_H
#define HASH_TABLE_H

/* Size of hash table slots, chosen as a prime number to minimize collisions */
#define HASHTABLE_SLOT_SIZE 50001

/* Node structure for storing document occurrence information */
typedef struct LinkedListNode {
    int docId;  // Document identifier
    int frequency;  // Number of occurrences in the document
    struct LinkedListNode *next;    // Pointer to the next node
} LinkedListNode;

/* Linked list structure for managing document occurrence information */
typedef struct LinkedList {
    int nodeCount;  // Number of nodes in the linked list
    LinkedListNode *headNode;   // Pointer to the head node
    LinkedListNode *tailNode;   // Pointer to the tail node
} LinkedList;

/* Hash table entry structure for storing words and their occurrences */
typedef struct HashTableEntry {
    int wordLength; // Length of the word
    char *word; // The word string
    LinkedList *list;   // Linked list storing document IDs and frequencies
    struct HashTableEntry *next;    // Pointer to the next entry for handling hash collisions in the slot
} HashTableEntry;

/* Hash table structure for storing word entries */
typedef struct HashTable {
    int wordCount;  // Number of unique words in the hash table
    HashTableEntry *slots[HASHTABLE_SLOT_SIZE];   // Array of hash table slots for storing word entries
} HashTable;

/* Function prototypes */
void addLinkedListNode(LinkedList *list, const int docId);  // Add a new node to the linked list
void updateLinkedList(LinkedList *list, const int docId);   // Update the linked list with document occurrence information
int hashFunction(const char *word); // Hash function for generating slot index from a word
HashTable *createHashTable();   // Create a new hash table
void freeHashTable(HashTable *table);   // Free the memory allocated for the hash table
void addHashTableEntry(HashTable *table, const char *word, const int docId);    // Add a new word entry to the hash table
void updateHashTable(HashTable *table, const char *word, const int docId);  // Update an existing word entry or add a new one in the hash table
char **getSortedWordsFromHashTable(const HashTable *table); // Get an array of sorted words from the hash table

#endif
