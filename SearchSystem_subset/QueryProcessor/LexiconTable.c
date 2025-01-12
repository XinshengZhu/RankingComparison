/* LexiconTable.c */
#include "LexiconTable.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    int result = hash % LEXICON_SIZE;
    // Handle negative hash values
    if (result < 0) {
        result += LEXICON_SIZE;
    }
    return result;
}

/**
 * Creates and initializes a new lexicon table
 * @return Pointer to the newly created lexicon table
 */
LexiconTable *createLexiconTable() {
    LexiconTable *lexiconTable = (LexiconTable *)malloc(sizeof(LexiconTable));
    if (lexiconTable == NULL) {
        printf("Error allocating memory!\n");
        exit(1);
    }
    lexiconTable->wordCount = 0;
    memset(lexiconTable->slots, 0, sizeof(LexiconEntry *) * LEXICON_SIZE);
    return lexiconTable;
}

/**
 * Frees all memory associated with the lexicon table
 * @param lexiconTable Pointer to the lexicon table to be freed
 */
void freeLexiconTable(LexiconTable *lexiconTable) {
    for (int slotIndex = 0; slotIndex < LEXICON_SIZE; slotIndex++) {
        LexiconEntry *currentEntry = lexiconTable->slots[slotIndex];
        while (currentEntry != NULL) {
            LexiconEntry *tempEntry = currentEntry;
            currentEntry = currentEntry->next;
            free(tempEntry->word);
            tempEntry->startChunk = 0;
            tempEntry->endChunk = 0;
            free(tempEntry);
        }
    }
    free(lexiconTable);
}

/**
 * Adds new entry to lexicon table
 * Uses chaining for collision resolution
 *
 * @param lexiconTable Table to add to
 * @param word Word to add
 * @param startChunk First chunk containing word
 * @param endChunk Last chunk containing word
 */
void addEntryToLexiconTable(LexiconTable *lexiconTable, const char *word, int startChunk, int endChunk) {
    int slotIndex = hashFunction(word);
    LexiconEntry *newEntry = (LexiconEntry *)malloc(sizeof(LexiconEntry));
    char *newWord = (char *)malloc(strlen(word) + 1);
    strcpy(newWord, word);
    newEntry->word = newWord;
    newEntry->startChunk = startChunk;
    newEntry->endChunk = endChunk;
    // Add to front of chain
    newEntry->next = lexiconTable->slots[slotIndex];
    lexiconTable->slots[slotIndex] = newEntry;
    lexiconTable->wordCount++;
}

/**
 * Parses lexicon file content and builds hash table for lexicon
 * File format: <word> <startChunk> <endChunk>\n
 *
 * @param lexiconTable Table to populate
 * @param buffer File content buffer
 * @return Populated lexicon table
 */
LexiconTable *convertLexiconFileToLexiconTable(LexiconTable *lexiconTable, char *buffer) {
    char *lineStart = buffer;
    char *lineEnd;
    while ((lineEnd = strchr(lineStart, '\n')) != NULL) {
        *lineEnd = '\0';    // Temporarily terminate line
        char *word = lineStart;
        char *wordEnd = strchr(word, ' ');
        if (wordEnd) {
            *wordEnd = '\0';    // Temporarily terminate word
            char *startChunkStr = wordEnd + 1;
            char *endChunkStr = strchr(startChunkStr, ' ');
            if (endChunkStr) {
                *endChunkStr = '\0';
                endChunkStr++;
                // Convert chunk numbers
                int startChunk = (int)strtoll(startChunkStr, NULL, 10);
                int endChunk = (int)strtoll(endChunkStr, NULL, 10);
                addEntryToLexiconTable(lexiconTable, word, startChunk, endChunk);
                // Restore spaces
                *wordEnd = ' ';
                *(endChunkStr - 1) = ' ';
            }
        }
        // Restore newline
        *lineEnd = '\n';
        lineStart = lineEnd + 1;
    }
    return lexiconTable;
}

/**
 * Finds word entry in lexicon table
 *
 * @param lexiconTable Table to search
 * @param word Word to find
 * @return Entry pointer if found, NULL if not found
 */
LexiconEntry *findWordInLexiconTable(const LexiconTable *lexiconTable, const char *word) {
    int slotIndex = hashFunction(word);
    LexiconEntry *lexiconEntry = lexiconTable->slots[slotIndex];
    while (lexiconEntry != NULL) {
        if (strcmp(lexiconEntry->word, word) == 0) {
            return lexiconEntry;
        }
        lexiconEntry = lexiconEntry->next;
    }
    return NULL;
}
