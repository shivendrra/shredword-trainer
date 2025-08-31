#ifndef __SUBWORD_H__
#define __SUBWORD_H__

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <float.h>
#include "cache.h"
#include "hashmap.h"

#define MAX_TEXT_LEN 8192
#define MAX_TOKEN_LEN 256
#define DEFAULT_MAX_LEN 20
#define SUBWORD_CACHE_SIZE 50000
#define VITERBI_CACHE_SIZE 20000

typedef struct SubwordSet {
  char **subwords;
  int count, capacity;
} SubwordSet;

typedef struct SubwordExtractor {
  LRUCache* cache;
} SubwordExtractor;

typedef struct ViterbiDecoder {
  LRUCache* cache;
} ViterbiDecoder;

typedef struct CharFreqResult {
  char *chars;
  int *frequencies;
  int count;
} CharFreqResult;

typedef struct TokenList {
  char **tokens;
  int count, capacity;
} TokenList;

extern "C" {
  // SubwordExtractor functions
  SubwordExtractor* subwordExtractorCreate();
  void subwordExtractorDestroy(SubwordExtractor* extractor);
  SubwordSet* extractSubwords(SubwordExtractor* extractor, const char* text, int max_len);
  CharFreqResult* getCharFrequencies(const char** texts, int text_count);
  void subwordSetDestroy(SubwordSet* set);
  void charFreqResultDestroy(CharFreqResult* result);

  // ViterbiDecoder functions  
  ViterbiDecoder* viterbiDecoderCreate();
  void viterbiDecoderDestroy(ViterbiDecoder* decoder);
  TokenList* viterbiDecoder(ViterbiDecoder* decoder, const char* text, FastHashMap* vocab);
  void tokenListDestroy(TokenList* list);

  // Utility functions
  bool subwordSetContains(SubwordSet* set, const char* subword);
  bool subwordSetAdd(SubwordSet* set, const char* subword);
  uint64_t stringHash64(const char* str);
  char* createCacheKey(const char* text, int max_len);

}

#endif