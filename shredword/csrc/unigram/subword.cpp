#include <float.h>
#include "utils.h"
#include "subword.h"
#include "../inc/hash.h"
#include "cache.h"

// SubwordSet functions
SubwordSet* subword_set_create() {
  SubwordSet *set = (SubwordSet*)malloc(sizeof(SubwordSet));
  set->capacity = 1000;
  set->size = 0;
  set->items = (char**)malloc(sizeof(char*) * set->capacity);
  return set;
}

void subword_set_destroy(SubwordSet *set) {
  if (!set) return;
  for (int i = 0; i < set->size; i++) free(set->items[i]);
  free(set->items);
  free(set);
}

void subword_set_add(SubwordSet *set, const char *item) {
  if (!set || !item) return;
  if (subword_set_contains(set, item)) return;

  if (set->size >= set->capacity) {
    set->capacity *= 2;
    set->items = (char**)realloc(set->items, sizeof(char*) * set->capacity);
  }
  set->items[set->size++] = strdup(item);
}

int subword_set_contains(SubwordSet *set, const char *item) {
  if (!set || !item) return 0;
  for (int i = 0; i < set->size; i++) { if (strcmp(set->items[i], item) == 0) return 1; }
  return 0;
}

// SubwordExtractor functions
SubwordExtractor* extractor_create() {
  SubwordExtractor *extractor = (SubwordExtractor*)malloc(sizeof(SubwordExtractor));
  extractor->cache = cache_create(50000);
  return extractor;
}

void extractor_destroy(SubwordExtractor *extractor) {
  if (!extractor) return;
  free(extractor->cache);
  free(extractor);
}

SubwordSet* extract_subwords(SubwordExtractor *extractor, const char *text, int max_len) {
  if (!extractor || !text) return NULL;
  
  int text_len = strlen(text);
  uint32_t cache_key = djb2_hash(text) ^ max_len;
  
  SubwordSet *subwords = subword_set_create();
  for (int i = 0; i < text_len; i++) {
    for (int j = i + 1; j <= text_len && j - i <= max_len; j++) {
      char *subword = (char*)malloc(j - i + 1);
      strncpy(subword, text + i, j - i);
      subword[j - i] = '\0';
      subword_set_add(subwords, subword);
      free(subword);
    }
  }
  return subwords;
}

// ViterbiDecoder functions
ViterbiDecoder* viterbi_create() {
  ViterbiDecoder *decoder = (ViterbiDecoder*)malloc(sizeof(ViterbiDecoder));
  decoder->cache = cache_create(20000);
  return decoder;
}

void viterbi_destroy(ViterbiDecoder *decoder) {
  if (!decoder) return;
  free(decoder->cache);
  free(decoder);
}

ViterbiResult* viterbi_decode(ViterbiDecoder *decoder, const char *text, FastHashMap *vocab) {
  if (!decoder || !text || !vocab) return NULL;
  
  int n = strlen(text);
  ViterbiState *dp = (ViterbiState*)malloc(sizeof(ViterbiState) * (n + 1));
  
  // Initialize DP table
  for (int i = 0; i <= n; i++) {
    dp[i].score = -FLT_MAX;
    dp[i].parent = -1;
  }
  dp[0].score = 0.0f;
  
  // Dynamic programming
  for (int i = 0; i < n; i++) {
    if (dp[i].score == -FLT_MAX) continue;
    for (int j = i + 1; j <= n && j - i <= 21; j++) {
      char *token = (char*)malloc(j - i + 1);
      strncpy(token, text + i, j - i);
      token[j - i] = '\0';

      if (hashmap_contains(vocab, token)) {
        float score = dp[i].score + hashmap_get(vocab, token);
        if (score > dp[j].score) {
          dp[j].score = score;
          dp[j].parent = i;
        }
      }
      free(token);
    }
  }
  
  ViterbiResult *result = (ViterbiResult*)malloc(sizeof(ViterbiResult));
  
  // If no valid path found, return original text
  if (dp[n].score == -FLT_MAX) {
    result->count = 1;
    result->tokens = (char**)malloc(sizeof(char*));
    result->tokens[0] = strdup(text);
    free(dp);
    return result;
  }

  // Reconstruct path
  int *path = (int*)malloc(sizeof(int) * (n + 1));
  int path_len = 0;
  int pos = n;
  
  while (pos > 0) {
    int start = dp[pos].parent;
    path[path_len++] = start;
    pos = start;
  }

  result->count = path_len;
  result->tokens = (char**)malloc(sizeof(char*) * path_len);

  for (int i = 0; i < path_len; i++) {
    int start = path[path_len - 1 - i];
    int end = (i == 0) ? n : path[path_len - i];
    result->tokens[i] = (char*)malloc(end - start + 1);
    strncpy(result->tokens[i], text + start, end - start);
    result->tokens[i][end - start] = '\0';
  }

  free(dp);
  free(path);
  return result;
}

void viterbi_result_destroy(ViterbiResult *result) {
  if (!result) return;
  for (int i = 0; i < result->count; i++) free(result->tokens[i]);
  free(result->tokens);
  free(result);
}

FastHashMap* get_char_frequencies(SubwordExtractor *extractor, char **texts, int text_count) {
  if (!extractor || !texts) return NULL;

  FastHashMap *char_freq = hashmap_create(256);
  for (int i = 0; i < text_count; i++) {
    if (!texts[i]) continue;
    for (const char *p = texts[i]; *p; p++) {
      char char_str[2] = {*p, '\0'};
      float current = hashmap_get(char_freq, char_str);
      if (current == -FLT_MAX) current = 0;
      hashmap_put(char_freq, char_str, current + 1);
    }
  }
  return char_freq;
}