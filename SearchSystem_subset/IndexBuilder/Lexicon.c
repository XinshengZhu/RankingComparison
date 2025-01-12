/* Lexicon.c */
#include "Lexicon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Creates and initializes a new lexicon structure
 * @return Pointer to newly created lexicon
 */
Lexicon *createLexicon() {
    Lexicon *lexicon = (Lexicon *)malloc(sizeof(Lexicon));
    if (lexicon == NULL) {
        printf("Error allocating memory!\n");
        exit(1);
    }
    lexicon->nodeCount = 0;
    lexicon->headNode = NULL;
    lexicon->tailNode = NULL;
    return lexicon;
}

/**
 * Frees memory allocated for the lexicon
 * @param lexicon Pointer to the lexicon structure to be freed
 */
void freeLexicon(Lexicon *lexicon) {
    LexiconNode *currentNode = lexicon->headNode;
    while (currentNode != NULL) {
        LexiconNode *nextNode = currentNode->next;
        free(currentNode->word);
        currentNode->startChunk = 0;
        currentNode->endChunk = 0;
        free(currentNode);
        currentNode = nextNode;
    }
    free(lexicon);
}

/**
 * Adds a new node (word) to the lexicon
 * @param lexicon Pointer to the lexicon structure to add the node
 * @param word Word string to be added
 * @param startChunk Start chunk number in the inverted index for the word
 * @param endChunk End chunk number in the inverted index for the word
 */
void addNodeToLexicon(Lexicon *lexicon, const char *word, int startChunk, int endChunk) {
    // Create and initialize new node
    LexiconNode *newNode = (LexiconNode *)malloc(sizeof(LexiconNode));
    char *newWord = (char *)malloc(strlen(word) + 1);
    strcpy(newWord, word);
    newNode->word = newWord;
    newNode->startChunk = startChunk;
    newNode->endChunk = endChunk;
    newNode->next = NULL;
    // Add to empty lexicon or append to the end of the list
    if (lexicon->headNode == NULL) {
        lexicon->headNode = newNode;
        lexicon->tailNode = newNode;
    } else {
        lexicon->tailNode->next = newNode;
        lexicon->tailNode = newNode;
    }
    lexicon->nodeCount++;
}
