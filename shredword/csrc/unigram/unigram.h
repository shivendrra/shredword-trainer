#ifndef __UNIGRAM_H__
#define __UNIGRAM_H__

#include "utils.h"
#include "cache.h"
#include "heap.h"
#include "subword.h"
#include "../trie.h"

#define MAX_TEXTS 50000
#define MAX_ITERATIONS 20
#define SEED_TEXTS 10000
#define LOSS_TEXTS 2000
#define SCORE_TEXTS 5000
#define PRUNE_TEXTS 1000
#define DEFAULT_VOCAB_SIZE 32000
#define DEFAULT_COVERAGE 0.9995f
#define DEFAULT_MAX_LEN 16
#define DEFAULT_SEED_SIZE 1000000

typedef struct TextArray {
  char **texts;
  int count, capacity;
} TextArray;

typedef struct CharCount {
  char ch;
  int count;
} CharCount;

typedef struct {
  char *token;
  float score;
} TokenScore;

typedef struct {
  float loss_increase;
  char *token;
} RemovalCandidate;

typedef struct {
  float freq;
  char *token;
} FreqToken;

typedef struct UnigramTrainer {
  int vocab_size, max_len, seed_size;
  float character_coverage;
  TokenFreqHeap *vocab_heap;
  FastHashMap *token_freqs;
  FastHashMap *vocab;
  FastHashMap *final_vocab;
  SubwordTrie *subword_trie;
  SubwordExtractor *extractor;
  ViterbiDecoder *decoder;
  LRUCache *loss_cache;
  TextArray *texts;
  int total_chars;
} UnigramTrainer;

extern "C" {
  UnigramTrainer* trainer_create(int vocab_size, float character_coverage, int max_len, int seed_size);
  void trainer_destroy(UnigramTrainer *trainer);

  TextArray* preprocess_texts(UnigramTrainer *trainer, char **input_texts, int text_count);
  void initialize_seed_vocab(UnigramTrainer *trainer, TextArray *texts);
  float compute_loss(UnigramTrainer *trainer, TextArray *texts);
  float compute_token_loss(UnigramTrainer *trainer, const char *token, TextArray *texts);
  void prune_vocab_step(UnigramTrainer *trainer, TextArray *texts, float reduction_ratio);
  void update_token_scores(UnigramTrainer *trainer, TextArray *texts);

  FastHashMap* train_unigram(UnigramTrainer *trainer, char **input_texts, int text_count, int num_iterations);
  FastHashMap* get_final_vocab(UnigramTrainer *trainer);
  void save_vocab(UnigramTrainer *trainer, const char *filepath);
  FastHashMap* load_vocab(UnigramTrainer *trainer, const char *filepath);

  TextArray* text_array_create(int capacity);
  void text_array_destroy(TextArray *array);
  void text_array_add(TextArray *array, const char *text);

  int compare_char_counts(const void *a, const void *b);
  int compare_freq_tokens(const void *a, const void *b);
}

#endif