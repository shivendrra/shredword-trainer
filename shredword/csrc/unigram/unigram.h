#ifndef __UNIGRAM_H__
#define __UNIGRAM_H__

#include "../trie.h"
#include "../inc/normalizer.h"
#include "hashmap.h"
#include "heap.h"
#include "cache.h"
#include "subword.h"

#define DEFAULT_VOCAB_SIZE 32000
#define DEFAULT_CHARACTER_COVERAGE 0.9995
#define DEFAULT_MAX_SENTENCEPIECE_LENGTH 16
#define DEFAULT_SEED_SIZE 1000000
#define MAX_TEXTS_FOR_TRAINING 50000
#define MAX_TEXTS_FOR_SAMPLING 10000
#define MAX_TEXTS_FOR_SCORING 5000
#define MAX_TEXTS_FOR_LOSS 2000
#define MAX_TEXTS_FOR_TOKEN_LOSS 1000
#define DEFAULT_REDUCTION_RATIO 0.8
#define CONVERGENCE_THRESHOLD 0.001
#define MIN_TOKEN_FREQ 1
#define UNKNOWN_TOKEN_SCORE -20.0

typedef struct UnigramTrainer {
  int vocab_size, seed_size, max_len, total_chars;
  float character_coverage;

  TokenFreqHeap* vocab_heap;
  FastHashMap *token_freqs, *vocab, *final_vocab;
  SubwordTrie* subword_trie;
  SubwordExtractor* extractor;
  ViterbiDecoder* decoder;
  LRUCache* loss_cache;

  char** texts;
  int text_count, text_capacity;
} UnigramTrainer;

typedef struct RemovalCandidate {
  double loss_increase;
  char* token;
} RemovalCandidate;

typedef struct TokenScore {
  char* token;
  double score;
} TokenScore;

extern "C" {
  UnigramTrainer* trainerCreate(int vs, float cc, int msl, int sss);
  void trainerDestroy(UnigramTrainer* trainer);
  bool addTextToTrainer(UnigramTrainer* trainer, const char* text);

  bool preprocessTexts(UnigramTrainer* trainer);
  bool extractInitialSubwords(UnigramTrainer* trainer);
  float computeLoss(UnigramTrainer* trainer, const char** texts, int text_count);
  double computeTokenLoss(UnigramTrainer* trainer, const char* token, const char** texts, int text_count);

  bool pruneVocabStep(UnigramTrainer* trainer, const char** texts, int text_count, double reduction_ratio);
  bool updateTokenScores(UnigramTrainer* trainer, const char** texts, int text_count);
  int compareTokenScores(const void* a, const void* b);
  bool trainUnigram(UnigramTrainer* trainer, const char** texts, int text_count, int num_iterations);

  bool getVocab(UnigramTrainer* trainer, char*** tokens, double** scores, int* count);
  bool saveVocab(UnigramTrainer* trainer, const char* filepath);
  bool loadVocab(UnigramTrainer* trainer, const char* filepath);
}

#endif