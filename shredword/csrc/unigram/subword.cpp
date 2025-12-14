#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "subword.h"
#include "cache.h"
#include "hashmap.h"
#include "../inc/hash.h"

SubwordSet* subwordSetCreate(int initial_capacity) {
  SubwordSet* set = (SubwordSet*)malloc(sizeof(SubwordSet));
  if (!set) return NULL;
  set->subwords = (char**)malloc(sizeof(char*) * initial_capacity);
  if (!set->subwords) { free(set); return NULL; }
  set->count = 0, set->capacity = initial_capacity;
  return set;
}

bool subwordSetResize(SubwordSet* set) {
  if (!set) return false;
  int new_capacity = set->capacity * 2;
  char** new_subwords = (char**)realloc(set->subwords, sizeof(char*) * new_capacity);
  if (!new_subwords) return false;
  set->subwords = new_subwords, set->capacity = new_capacity;
  return true;
}

bool subwordSetContains(SubwordSet* set, const char* subword) {
  if (!set || !subword) return false;
  for (int i = 0; i < set->count; i++) {
    if (strcmp(set->subwords[i], subword) == 0) return true;
  }
  return false;
}

bool subwordSetAdd(SubwordSet* set, const char* subword) {
  if (!set || !subword) return false;
  if (subwordSetContains(set, subword)) return true;
  if (set->count >= set->capacity && !subwordSetResize(set)) return false;
  set->subwords[set->count] = strdup(subword);
  if (!set->subwords[set->count]) return false;
  set->count++;
  return true;
}

void subwordSetDestroy(SubwordSet* set) {
  if (!set) return;
  for (int i = 0; i < set->count; i++) free(set->subwords[i]);
  free(set->subwords);
  free(set);
}

uint64_t stringHash64(const char* str) {
  if (!str) return 0;
  uint64_t hash = 14695981039346656037ULL;
  while (*str) { hash ^= (uint64_t)*str++; hash *= 1099511628211ULL; }
  return hash;
}

char* createCacheKey(const char* text, int max_len) {
  if (!text) return NULL;
  char* key = (char*)malloc(64);
  if (!key) return NULL;
  uint64_t text_hash = stringHash64(text);
  snprintf(key, 64, "%llu_%d", (unsigned long long)text_hash, max_len);
  return key;
}

SubwordExtractor* subwordExtractorCreate() {
  SubwordExtractor* extractor = (SubwordExtractor*)malloc(sizeof(SubwordExtractor));
  if (!extractor) return NULL;
  extractor->cache = cacheCreate(SUBWORD_CACHE_SIZE);
  return extractor;
}

void subwordExtractorDestroy(SubwordExtractor* extractor) {
  if (!extractor) return;
  if (extractor->cache) cacheFree(extractor->cache);
  free(extractor);
}

SubwordSet* extractSubwords(SubwordExtractor* extractor, const char* text, int max_len) {
  if (!extractor || !text || max_len <= 0) return NULL;
  int text_len = strlen(text);
  if (text_len == 0) return NULL;
  if (text_len >= MAX_TEXT_LEN) text_len = MAX_TEXT_LEN - 1;
  if (max_len > MAX_TOKEN_LEN) max_len = MAX_TOKEN_LEN;
  char* cache_key = createCacheKey(text, max_len);
  if (!cache_key) return NULL;
  uint32_t key_hash = murmur3_hash(cache_key, strlen(cache_key));
  int cached_result = cacheGet(extractor->cache, (int)key_hash);
  if (cached_result != -1) { free(cache_key); return NULL; }
  int estimated_size = text_len * 2;
  if (estimated_size > 50000) estimated_size = 50000;
  if (estimated_size < 100) estimated_size = 100;
  SubwordSet* subwords = subwordSetCreate(estimated_size);
  if (!subwords) { free(cache_key); return NULL; }
  for (int i = 0; i < text_len; i++) {
    int max_j = i + max_len + 1;
    if (max_j > text_len + 1) max_j = text_len + 1;
    for (int j = i + 1; j < max_j; j++) {
      int substr_len = j - i;
      if (substr_len >= MAX_TOKEN_LEN) continue;
      char substr[MAX_TOKEN_LEN];
      memcpy(substr, text + i, substr_len);
      substr[substr_len] = '\0';
      if (!subwordSetAdd(subwords, substr)) {
        subwordSetDestroy(subwords);
        free(cache_key);
        return NULL;
      }
    }
  }
  cachePut(extractor->cache, (int)key_hash, 1);
  free(cache_key);
  return subwords;
}

CharFreqResult* getCharFrequencies(const char** texts, int text_count) {
  if (!texts || text_count <= 0) return NULL;
  FastHashMap* char_map = hashmapCreate(256);
  if (!char_map) return NULL;
  for (int t = 0; t < text_count; t++) {
    if (!texts[t]) continue;
    const char* text = texts[t];
    for (int i = 0; text[i]; i++) {
      char char_key[2] = {text[i], '\0'};
      int* freq = (int*)hashMapGet(char_map, char_key);
      if (freq) { (*freq)++; }
      else {
        int* new_freq = (int*)malloc(sizeof(int));
        if (new_freq) { *new_freq = 1; hashMapSet(char_map, char_key, new_freq); }
      }
    }
  }
  CharFreqResult* result = (CharFreqResult*)malloc(sizeof(CharFreqResult));
  if (!result) { hashMapDestroy(char_map); return NULL; }
  result->count = 0, result->chars = NULL, result->frequencies = NULL;
  hashMapDestroy(char_map);
  return result;
}

void charFreqResultDestroy(CharFreqResult* result) {
  if (!result) return;
  free(result->chars);
  free(result->frequencies);
  free(result);
}

ViterbiDecoder* viterbiDecoderCreate() {
  ViterbiDecoder* decoder = (ViterbiDecoder*)malloc(sizeof(ViterbiDecoder));
  if (!decoder) return NULL;
  decoder->cache = cacheCreate(VITERBI_CACHE_SIZE);
  return decoder;
}

void viterbiDecoderDestroy(ViterbiDecoder* decoder) {
  if (!decoder) return;
  if (decoder->cache) cacheFree(decoder->cache);
  free(decoder);
}

TokenList* tokenListCreate(int initial_capacity) {
  TokenList* list = (TokenList*)malloc(sizeof(TokenList));
  if (!list) return NULL;
  list->tokens = (char**)malloc(sizeof(char*) * initial_capacity);
  if (!list->tokens) { free(list); return NULL; }
  list->count = 0, list->capacity = initial_capacity;
  return list;
}

bool tokenListAdd(TokenList* list, const char* token) {
  if (!list || !token) return false;
  if (list->count >= list->capacity) {
    int new_capacity = list->capacity * 2;
    char** new_tokens = (char**)realloc(list->tokens, sizeof(char*) * new_capacity);
    if (!new_tokens) return false;
    list->tokens = new_tokens, list->capacity = new_capacity;
  }
  list->tokens[list->count] = strdup(token);
  if (!list->tokens[list->count]) return false;
  list->count++;
  return true;
}

void tokenListDestroy(TokenList* list) {
  if (!list) return;
  for (int i = 0; i < list->count; i++) free(list->tokens[i]);
  free(list->tokens);
  free(list);
}

TokenList* viterbiDecode(ViterbiDecoder* decoder, const char* text, FastHashMap* vocab) {
  if (!decoder || !text || !vocab) return NULL;
  int text_len = strlen(text);
  if (text_len == 0) return tokenListCreate(1);
  if (text_len >= MAX_TEXT_LEN) return NULL;
  double* dp = (double*)calloc(text_len + 1, sizeof(double));
  int* parent = (int*)malloc(sizeof(int) * (text_len + 1));
  if (!dp || !parent) { free(dp); free(parent); return NULL; }
  for (int i = 1; i <= text_len; i++) dp[i] = -1e9, parent[i] = -1;
  dp[0] = 0.0;
  for (int i = 0; i < text_len; i++) {
    if (dp[i] < -1e8) continue;
    int max_j = (i + 21 < text_len + 1) ? i + 21 : text_len + 1;
    for (int j = i + 1; j < max_j; j++) {
      int token_len = j - i;
      if (token_len >= MAX_TOKEN_LEN) continue;
      char token[MAX_TOKEN_LEN];
      strncpy(token, text + i, token_len);
      token[token_len] = '\0';
      double* vocab_score = (double*)hashMapGet(vocab, token);
      if (vocab_score) {
        double score = dp[i] + *vocab_score;
        if (score > dp[j]) { dp[j] = score; parent[j] = i; }
      }
    }
  }
  if (parent[text_len] == -1) {
    TokenList* result = tokenListCreate(text_len);
    if (result) {
      for (int i = 0; i < text_len; i++) {
        char single[2] = {text[i], '\0'};
        tokenListAdd(result, single);
      }
    }
    free(dp); free(parent);
    return result;
  }
  TokenList* path = tokenListCreate(text_len / 2 + 1);
  int pos = text_len;
  while (pos > 0 && parent[pos] != -1) {
    int start = parent[pos];
    int token_len = pos - start;
    char token[MAX_TOKEN_LEN];
    strncpy(token, text + start, token_len);
    token[token_len] = '\0';
    if (!tokenListAdd(path, token)) {
      tokenListDestroy(path);
      free(dp); free(parent);
      return NULL;
    }
    pos = start;
  }
  for (int i = 0; i < path->count / 2; i++) {
    char* temp = path->tokens[i];
    path->tokens[i] = path->tokens[path->count - 1 - i];
    path->tokens[path->count - 1 - i] = temp;
  }
  free(dp); free(parent);
  return path;
}