/**
  @brief trie-based dtype to store possible vocabs
  
  * trie-based functions for creating, inserting & deleting tries & entries
  * word-count & printing functions
*/

#ifndef __TRIE__H__
#define __TRIE__H__

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define NUM_CHARS 256
#define TRIE_CHILDREN 256
#define MAX_TOKEN_LENGTH 1024

typedef struct TrieNode {
  struct TrieNode *children[TRIE_CHILDREN];
  bool is_token;
  int freq;
} TrieNode;

typedef struct SubwordTrie {
  TrieNode *root;
  int total_tokens;
} SubwordTrie;

extern "C" {
  SubwordTrie* trieCreate();
  void trieDestroy(SubwordTrie *trie);

  bool trieInsert(SubwordTrie *trie, const char *token, int freq);
  int trieSearch(SubwordTrie *trie, const char *token);
  bool trieContains(SubwordTrie *trie, const char *token);
  bool trieRemove(SubwordTrie *trie, const char *token);
  bool trieUpdateFreq(SubwordTrie *trie, const char *token, int new_freq);

  void trieGetAllTokens(SubwordTrie *trie, char ***tokens, int **freqs, int *count);
  void trieFreeTokens(char **tokens, int count);
  int trieGetTokenCount(SubwordTrie *trie);
}

#endif