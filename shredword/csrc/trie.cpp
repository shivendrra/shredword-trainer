#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "trie.h"
#include "inc/hash.h"

// TrieNode functions
static TrieNode* trie_node_create() {
  TrieNode *node = (TrieNode*)calloc(1, sizeof(TrieNode));
  return node;
}

static void trie_node_destroy(TrieNode *node) {
  if (!node) return;
  for (int i = 0; i < TRIE_CHILDREN; i++) {
    if (node->children[i]) trie_node_destroy(node->children[i]);
  }
  free(node);
}

SubwordTrie* trie_create() {
  SubwordTrie *trie = (SubwordTrie*)malloc(sizeof(SubwordTrie));
  trie->root = trie_node_create();
  return trie;
}

void trie_destroy(SubwordTrie *trie) {
  if (!trie) return;
  trie_node_destroy(trie->root);
  free(trie);
}

void trie_insert(SubwordTrie *trie, const char *token, int freq) {
  if (!trie || !token) return;
  TrieNode *node = trie->root;
  for (const char *p = token; *p; p++) {
    unsigned char c = (unsigned char)*p;
    if (!node->children[c]) node->children[c] = trie_node_create();
    node = node->children[c];
  }
  node->is_token = 1;
  node->freq = freq;
}

int trie_search(SubwordTrie *trie, const char *token) {
  if (!trie || !token) return -1;
  TrieNode *node = trie->root;
  for (const char *p = token; *p; p++) {
    unsigned char c = (unsigned char)*p;
    if (!node->children[c]) return -1;
    node = node->children[c];
  }
  return node->is_token ? node->freq : -1;
}

static void trie_collect_tokens(TrieNode *node, char *prefix, int depth, char ***tokens, int **freqs, int *count, int *capacity) {
  if (node->is_token) {
    if (*count >= *capacity) {
      *capacity *= 2;
      *tokens = (char**)realloc(*tokens, sizeof(char*) * (*capacity));
      *freqs = (int*)realloc(*freqs, sizeof(int) * (*capacity));
    }
    (*tokens)[*count] = strdup(prefix);
    (*freqs)[*count] = node->freq;
    (*count)++;
  }
  for (int i = 0; i < TRIE_CHILDREN; i++) {
    if (node->children[i]) {
      prefix[depth] = (char)i;
      prefix[depth + 1] = '\0';
      trie_collect_tokens(node->children[i], prefix, depth + 1, tokens, freqs, count, capacity);
    }
  }
}

void trie_get_all_tokens(SubwordTrie *trie, char ***tokens, int **freqs, int *count) {
  if (!trie) return;
  *count = 0;
  int capacity = 1000;
  *tokens = (char**)malloc(sizeof(char*) * capacity);
  *freqs = (int*)malloc(sizeof(int) * capacity);
  char prefix[NUM_CHARS];
  prefix[0] = '\0';
  trie_collect_tokens(trie->root, prefix, 0, tokens, freqs, count, &capacity);
}