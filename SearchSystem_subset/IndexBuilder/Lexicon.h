/* Lexicon.h */
#ifndef LEXICON_H
#define LEXICON_H

/* Node structure for lexicon entries, stores word and its location information in the inverted index */
typedef struct LexiconNode {
    char *word; // Word string in the lexicon
    int startChunk; // Start chunk number in the inverted index
    int endChunk;   // End chunk number in the inverted index
    struct LexiconNode *next; // Pointer to the next node in the linked list
} LexiconNode;

/* Lexicon structure to maintain a linked list of word entries*/
typedef struct Lexicon {
    int nodeCount;  // Number of nodes (words) in the lexicon
    LexiconNode *headNode;  // Pointer to the head node of the linked list
    LexiconNode *tailNode;  // Pointer to the tail node of the linked list
} Lexicon;

/* Function declarations */
Lexicon *createLexicon();   // Create a new lexicon
void freeLexicon(Lexicon *lexicon); // Free memory allocated for the lexicon
void addNodeToLexicon(Lexicon *lexicon, const char *word, int startChunk, int endChunk);    // Add a new node (word) to the lexicon

#endif
