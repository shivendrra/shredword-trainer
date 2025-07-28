#ifndef __SUBWORD__H__
#define __SUBWORD__H__

#include <stdint.h>
#include "utils.h"

typedef struct SubwordSet {
  char **items;
  size_t size, capacity;
} SubwordSet;

typedef struct ViterbiState {
  float score;
  size_t parent;
} ViterbiState;

typedef struct ViterbiResult {
  char **tokens;
  size_t count;
} ViterbiResult;

typedef struct SubwordExtractor { void *cache; } SubwordExtractor;
typedef struct ViterbiDecoder { void *cache; } ViterbiDecoder;

extern "C" {
  // SubwordSet functions
  SubwordSet* subword_set_create();
  void subword_set_destroy(SubwordSet *set);
  void subword_set_add(SubwordSet *set, const char *item);
  int subword_set_contains(SubwordSet *set, const char *item);

  // SubwordExtractor functions
  SubwordExtractor* extractor_create();
  void extractor_destroy(SubwordExtractor *extractor);
  SubwordSet* extract_subwords(SubwordExtractor *extractor, const char *text, int max_len);
  FastHashMap* get_char_frequencies(SubwordExtractor *extractor, char **texts, int text_count);

  // ViterbiDecoder functions
  ViterbiDecoder* viterbi_create();
  void viterbi_destroy(ViterbiDecoder *decoder);
  ViterbiResult* viterbi_decode(ViterbiDecoder *decoder, const char *text, FastHashMap *vocab);
  void viterbi_result_destroy(ViterbiResult *result); 
}

#endif  //!__SUBWORD__H__