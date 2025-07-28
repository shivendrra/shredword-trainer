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

#define NUM_CHARS 256   // Maximum initial letters, since UTF-8 so 0-255
#define TRIE_CHILDREN 256

typedef struct TrieNode {
  struct TrieNode *children[TRIE_CHILDREN];
  int is_token, freq;
} TrieNode;

typedef struct SubwordTrie {
  TrieNode *root;
} SubwordTrie;

extern "C" {
  SubwordTrie* trie_create();
  void trie_destroy(SubwordTrie *trie);
  void trie_insert(SubwordTrie *trie, const char *token, int freq);
  int trie_search(SubwordTrie *trie, const char *token);
  void trie_get_all_tokens(SubwordTrie *trie, char ***tokens, int **freqs, int *count);
}

#endif