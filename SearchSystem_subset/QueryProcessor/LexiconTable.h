/* LexiconTable.h */
#ifndef LEXICON_TABLE_H
#define LEXICON_TABLE_H

/* Prime number chosen for hash table size to minimize collisions */
#define LEXICON_SIZE 1999993

/* Entry in lexicon hash table, stores word and its chunk location in index file */
typedef struct LexiconEntry {
    char *word; // Word string
    int startChunk; // Start chunk containing word in index file, starting from 1
    int endChunk;   // End chunk containing word in index file
    struct LexiconEntry *next;  // Pointer to next entry in case of collision
} LexiconEntry;

/* Lexicon hash table for fast lexicon lookups */
typedef struct LexiconTable {
    int wordCount;  // Total number of words in lexicon
    LexiconEntry *slots[LEXICON_SIZE];  // Array of pointers to lexicon entries in each slot
} LexiconTable;

/* Function prototypes */
int hashFunction (const char *word);    // Hash function for generating slot index from a word
LexiconTable *createLexiconTable(); // Create a new lexicon table
void freeLexiconTable(LexiconTable *lexiconTable);  // Free memory allocated for lexicon table
void addEntryToLexiconTable(LexiconTable *lexiconTable, const char *word, int startChunk, int endChunk);    // Add a new word entry to lexicon table
LexiconTable *convertLexiconFileToLexiconTable(LexiconTable *lexiconTable, char *buffer);   // Convert lexicon file in ASCII format to lexicon table
LexiconEntry *findWordInLexiconTable(const LexiconTable *lexiconTable, const char *word);   // Find a word in the lexicon table

#endif
