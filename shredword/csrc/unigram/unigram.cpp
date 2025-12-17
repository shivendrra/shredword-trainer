#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <float.h>
#include <stdint.h>
#include "../inc/hash.h"
#include "unigram.h"

UnigramTrainer* trainerCreate(int vs, float cc, int msl, int sss) {
  UnigramTrainer* trainer = (UnigramTrainer*)malloc(sizeof(UnigramTrainer));
  if (!trainer) return NULL;
  trainer->vocab_size = vs, trainer->character_coverage = cc, trainer->max_len = msl, trainer->seed_size = sss;
  trainer->vocab_heap = heapCreate();
  trainer->token_freqs = hashmapCreate(INITIAL_SIZE);
  trainer->subword_trie = trieCreate();
  trainer->extractor = subwordExtractorCreate();
  trainer->decoder = viterbiDecoderCreate();
  trainer->loss_cache = cacheCreate(100000);
  trainer->vocab = hashmapCreate(INITIAL_SIZE);
  trainer->final_vocab = hashmapCreate(INITIAL_SIZE);
  trainer->text_capacity = 16, trainer->text_count = 0, trainer->total_chars = 0;
  trainer->texts = (char**)malloc(trainer->text_capacity * sizeof(char*));
  if (!trainer->texts) { free(trainer); return NULL; }
  for (int i = 0; i < trainer->text_capacity; i++) trainer->texts[i] = NULL;
  return trainer;
}

void trainerDestroy(UnigramTrainer* trainer) {
  if (!trainer) return;
  heapFree(trainer->vocab_heap);
  hashMapDestroy(trainer->token_freqs);
  trieDestroy(trainer->subword_trie);
  subwordExtractorDestroy(trainer->extractor);
  viterbiDecoderDestroy(trainer->decoder);
  cacheFree(trainer->loss_cache);
  hashMapDestroy(trainer->vocab);
  hashMapDestroy(trainer->final_vocab);
  for (int i = 0; i < trainer->text_capacity; i++) {
    if (trainer->texts[i]) free(trainer->texts[i]);
  }
  free(trainer->texts);
  free(trainer);
}

bool addTextToTrainer(UnigramTrainer* trainer, const char* text) {
  if (!trainer || !text) return false;
  if (trainer->text_count >= trainer->text_capacity) {
    int new_capacity = trainer->text_capacity * 2;
    char** new_texts = (char**)realloc(trainer->texts, new_capacity * sizeof(char*));
    if (!new_texts) return false;
    for (int i = trainer->text_capacity; i < new_capacity; i++) new_texts[i] = NULL;
    trainer->texts = new_texts, trainer->text_capacity = new_capacity;
  }
  trainer->texts[trainer->text_count] = strdup(text);
  if (!trainer->texts[trainer->text_count]) return false;
  trainer->text_count++;
  return true;
}

bool preprocessTexts(UnigramTrainer* trainer) {
  if (!trainer || trainer->text_count == 0) return false;
  printf("  Allocating memory for preprocessing...\n");
  fflush(stdout);
  FastHashMap* char_counts = hashmapCreate(256);
  if (!char_counts) { printf("  ERROR: Failed to create char_counts hashmap\n"); return false; }
  char** processed_texts = (char**)malloc(trainer->text_count * sizeof(char*));
  if (!processed_texts) { printf("  ERROR: Failed to allocate processed_texts\n"); hashMapDestroy(char_counts); return false; }
  int processed_count = 0, skipped = 0, failed_norm = 0;
  trainer->total_chars = 0;
  printf("  Testing normalization on first text...\n");
  fflush(stdout);
  bool use_normalization = true;
  if (trainer->texts[0] && strlen(trainer->texts[0]) > 0) {
    NormalizedText* test_nt = create_normalized_text(1000);
    if (!test_nt) { printf("  WARNING: Normalization unavailable, using raw text\n"); use_normalization = false; }
    else {
      char test_text[101];
      strncpy(test_text, trainer->texts[0], 100);
      test_text[100] = '\0';
      if (normalize_text_fast(test_text, test_nt) != 0) { printf("  WARNING: Normalization failed, using raw text\n"); use_normalization = false; }
      free_normalized_text(test_nt);
    }
  }
  printf("  Processing %d texts (normalization: %s)...\n", trainer->text_count, use_normalization ? "enabled" : "disabled");
  fflush(stdout);
  for (int i = 0; i < trainer->text_count; i++) {
    if (i % 1000 == 0) { printf("    Processed %d/%d texts (skipped %d)\r", i, trainer->text_count, skipped); fflush(stdout); }
    if (!trainer->texts[i]) { skipped++; continue; }
    int orig_len = strlen(trainer->texts[i]);
    if (orig_len == 0 || orig_len > 50000) { skipped++; continue; }
    char* final_text = NULL;
    if (use_normalization) {
      int text_len = orig_len < 10000 ? orig_len : 10000;
      int buffer_size = text_len * 3 + 100;
      NormalizedText* nt = create_normalized_text(buffer_size);
      if (nt) {
        char temp_text[10001];
        memset(temp_text, 0, sizeof(temp_text));
        strncpy(temp_text, trainer->texts[i], 10000);
        temp_text[10000] = '\0';
        if (normalize_text_fast(temp_text, nt) == 0 && nt->length > 0 && nt->data) {
          final_text = strdup(nt->data);
        } else { failed_norm++; }
        free_normalized_text(nt);
      } else { failed_norm++; }
    }
    if (!final_text) {
      int copy_len = orig_len < 10000 ? orig_len : 10000;
      final_text = (char*)malloc(copy_len + 1);
      if (final_text) { strncpy(final_text, trainer->texts[i], copy_len); final_text[copy_len] = '\0'; }
    }
    if (final_text) {
      processed_texts[processed_count] = final_text;
      int text_len = strlen(final_text);
      for (int j = 0; j < text_len; j++) {
        char key[2] = {final_text[j], '\0'};
        int* count = (int*)hashMapGet(char_counts, key);
        if (count) { (*count)++; }
        else {
          int* new_count = (int*)malloc(sizeof(int));
          if (new_count) { *new_count = 1; hashMapSet(char_counts, key, new_count); }
        }
        trainer->total_chars++;
      }
      processed_count++;
    } else { skipped++; }
  }
  printf("\n  Processed %d texts successfully (skipped %d, normalization failed %d)\n", processed_count, skipped, failed_norm);
  if (processed_count == 0) { printf("  ERROR: No texts were normalized successfully\n"); free(processed_texts); hashMapDestroy(char_counts); return false; }
  for (int i = 0; i < trainer->text_count; i++) {
    if (trainer->texts[i]) free(trainer->texts[i]);
  }
  free(trainer->texts);
  trainer->texts = processed_texts;
  trainer->text_count = processed_count;
  trainer->text_capacity = processed_count > 0 ? processed_count : 1;
  HashMapIterator* iter = hashMapIteratorCreate(char_counts);
  if (iter) {
    const char* k; void* v;
    while (hashMapIteratorNext(iter, &k, &v)) free(v);
    hashMapIteratorDestroy(iter);
  }
  hashMapDestroy(char_counts);
  return processed_count > 0;
}

bool extractInitialSubwords(UnigramTrainer* trainer) {
  if (!trainer) return false;
  int sample_limit = 1000;
  if (trainer->text_count < sample_limit) sample_limit = trainer->text_count;
  printf("  Sampling %d texts for initial vocabulary...\n", sample_limit);
  FastHashMap* token_freq_map = hashmapCreate(INITIAL_SIZE);
  if (!token_freq_map) { printf("  ERROR: Failed to create token_freq_map\n"); return false; }
  printf("  Extracting character frequencies...\n");
  for (int i = 0; i < trainer->text_count; i++) {
    if (i % 1000 == 0) printf("    Processing text %d/%d\r", i, trainer->text_count);
    const char* text = trainer->texts[i];
    for (int j = 0; text[j]; j++) {
      char char_key[2] = {text[j], '\0'};
      int* count = (int*)hashMapGet(token_freq_map, char_key);
      if (count) { (*count)++; }
      else {
        int* new_count = (int*)malloc(sizeof(int));
        if (new_count) { *new_count = 1; hashMapSet(token_freq_map, char_key, new_count); }
      }
    }
  }
  printf("\n  Extracted %d unique characters\n", hashMapSize(token_freq_map));
  printf("  Extracting subword candidates from sample...\n");
  int subword_count = 0, max_subwords = trainer->seed_size;
  for (int i = 0; i < sample_limit && subword_count < max_subwords; i++) {
    if (i % 100 == 0) printf("    Sampling text %d/%d (found %d subwords)\r", i, sample_limit, subword_count);
    const char* text = trainer->texts[i];
    int text_len = strlen(text);
    if (text_len > 500) text_len = 500;
    for (int start = 0; start < text_len && subword_count < max_subwords; start++) {
      int max_end = start + trainer->max_len + 1;
      if (max_end > text_len + 1) max_end = text_len + 1;
      for (int end = start + 2; end < max_end; end++) {
        int token_len = end - start;
        if (token_len >= MAX_TOKEN_LEN) continue;
        char token[MAX_TOKEN_LEN];
        memcpy(token, text + start, token_len);
        token[token_len] = '\0';
        if (!hashMapContains(token_freq_map, token)) {
          int* new_count = (int*)malloc(sizeof(int));
          if (new_count) { *new_count = 1; hashMapSet(token_freq_map, token, new_count); subword_count++; }
        }
      }
    }
  }
  printf("\n  Collected %d candidate subwords\n", hashMapSize(token_freq_map));
  printf("  Counting frequencies in full dataset...\n");
  HashMapIterator* candidate_iter = hashMapIteratorCreate(token_freq_map);
  if (candidate_iter) {
    const char* token; void* dummy;
    while (hashMapIteratorNext(candidate_iter, &token, &dummy)) {
      int* freq_ptr = (int*)dummy;
      *freq_ptr = 0;
    }
    hashMapIteratorDestroy(candidate_iter);
  }
  for (int i = 0; i < trainer->text_count; i++) {
    if (i % 1000 == 0) printf("    Counting in text %d/%d\r", i, trainer->text_count);
    const char* text = trainer->texts[i];
    int text_len = strlen(text);
    for (int start = 0; start < text_len; start++) {
      int max_end = start + trainer->max_len + 1;
      if (max_end > text_len + 1) max_end = text_len + 1;
      for (int end = start + 1; end < max_end; end++) {
        int token_len = end - start;
        if (token_len >= MAX_TOKEN_LEN) continue;
        char token[MAX_TOKEN_LEN];
        memcpy(token, text + start, token_len);
        token[token_len] = '\0';
        int* count = (int*)hashMapGet(token_freq_map, token);
        if (count) (*count)++;
      }
    }
  }
  printf("\n  Building initial vocabulary...\n");
  int added = 0;
  HashMapIterator* freq_iter = hashMapIteratorCreate(token_freq_map);
  if (freq_iter) {
    const char* token; void* freq_value;
    while (hashMapIteratorNext(freq_iter, &token, &freq_value)) {
      int freq = *(int*)freq_value;
      if (freq > MIN_TOKEN_FREQ && added < trainer->seed_size) {
        heapPush(trainer->vocab_heap, token, freq);
        double* log_freq = (double*)malloc(sizeof(double));
        if (log_freq) { *log_freq = log((double)freq); hashMapSet(trainer->vocab, token, log_freq); }
        trieInsert(trainer->subword_trie, token, freq);
        int* freq_copy = (int*)malloc(sizeof(int));
        if (freq_copy) { *freq_copy = freq; hashMapSet(trainer->token_freqs, token, freq_copy); }
        added++;
      }
    }
    hashMapIteratorDestroy(freq_iter);
  }
  printf("  Added %d tokens to initial vocabulary\n", added);
  HashMapIterator* cleanup_iter = hashMapIteratorCreate(token_freq_map);
  if (cleanup_iter) {
    const char* token; void* freq_value;
    while (hashMapIteratorNext(cleanup_iter, &token, &freq_value)) free(freq_value);
    hashMapIteratorDestroy(cleanup_iter);
  }
  hashMapDestroy(token_freq_map);
  return added > 0;
}

float computeLoss(UnigramTrainer* trainer, const char** texts, int text_count) {
  if (!trainer || !texts || text_count <= 0) return 0.0f;
  double total_loss = 0.0;
  int total_len = 0;
  for (int i = 0; i < text_count; i++) {
    if (!texts[i]) continue;
    uint64_t cache_key = stringHash64(texts[i]);
    int cached_loss = cacheGet(trainer->loss_cache, (int)(cache_key % INT32_MAX));
    if (cached_loss != -1) {
      total_loss += (double)cached_loss / MAX_TEXTS_FOR_TOKEN_LOSS;
      total_len += (int)strlen(texts[i]);
      continue;
    }
    TokenList* segmentation = viterbiDecode(trainer->decoder, texts[i], trainer->vocab);
    if (!segmentation) continue;
    double text_loss = 0.0;
    for (int j = 0; j < segmentation->count; j++) {
      double* token_score = (double*)hashMapGet(trainer->vocab, segmentation->tokens[j]);
      double score = token_score ? *token_score : -UNKNOWN_TOKEN_SCORE;
      text_loss -= score;
    }
    cachePut(trainer->loss_cache, (int)(cache_key % INT32_MAX), (int)(text_loss * MAX_TEXTS_FOR_TOKEN_LOSS));
    total_loss += text_loss;
    total_len += (int)strlen(texts[i]);
    tokenListDestroy(segmentation);
  }
  return total_len > 0 ? (float)(total_loss / total_len) : 0.0f;
}

double computeTokenLoss(UnigramTrainer* trainer, const char* token, const char** texts, int text_count) {
  if (!trainer || !token) return 0.0;

  int* freq = (int*)hashMapGet(trainer->token_freqs, token);
  if (!freq) return 0.0;

  double* score = (double*)hashMapGet(trainer->vocab, token);
  double token_score = score ? *score : 0.0;

  return (double)(*freq) * fabs(token_score);
}


int compareRemovalCandidates(const void* a, const void* b) {
  const RemovalCandidate* ca = (const RemovalCandidate*)a;
  const RemovalCandidate* cb = (const RemovalCandidate*)b;
  if (ca->loss_increase < cb->loss_increase) return -1;
  if (ca->loss_increase > cb->loss_increase) return 1;
  return 0;
}

void shuffleVocabItems(char** tokens, double** scores, int count) {
  srand((unsigned int)time(NULL));
  for (int i = count - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    char* temp_token = tokens[i]; double* temp_score = scores[i];
    tokens[i] = tokens[j], scores[i] = scores[j];
    tokens[j] = temp_token, scores[j] = temp_score;
  }
}

bool pruneVocabStep(UnigramTrainer* trainer, const char** texts, int text_count, double reduction_ratio) {
  if (!trainer || hashMapSize(trainer->vocab) <= trainer->vocab_size) return true;
  printf("  Pruning vocabulary...\n");
  int current_size = hashMapSize(trainer->vocab);
  int target_size = (int)(current_size * reduction_ratio);
  if (target_size < trainer->vocab_size) target_size = trainer->vocab_size;
  int tokens_to_remove = current_size - target_size;
  if (tokens_to_remove <= 0) return true;
  char** vocab_tokens = (char**)malloc(current_size * sizeof(char*));
  double** vocab_scores = (double**)malloc(current_size * sizeof(double*));
  if (!vocab_tokens || !vocab_scores) {
    if (vocab_tokens) free(vocab_tokens);
    if (vocab_scores) free(vocab_scores);
    return false;
  }
  HashMapIterator* iter = hashMapIteratorCreate(trainer->vocab);
  int vocab_count = 0;
  if (iter) {
    const char* key; void* value;
    while (hashMapIteratorNext(iter, &key, &value) && vocab_count < current_size) {
      vocab_tokens[vocab_count] = strdup(key);
      vocab_scores[vocab_count] = (double*)value;
      vocab_count++;
    }
    hashMapIteratorDestroy(iter);
  }
  shuffleVocabItems(vocab_tokens, vocab_scores, vocab_count);
  int candidates_limit = (vocab_count < tokens_to_remove * 2) ? vocab_count : tokens_to_remove * 2;
  RemovalCandidate* candidates = (RemovalCandidate*)malloc(candidates_limit * sizeof(RemovalCandidate));
  if (!candidates) {
    for (int i = 0; i < vocab_count; i++) free(vocab_tokens[i]);
    free(vocab_tokens); free(vocab_scores);
    return false;
  }
  int candidate_count = 0;
  int text_limit = (text_count < 500) ? text_count : 500;
  for (int i = 0; i < candidates_limit && candidate_count < candidates_limit; i++) {
    if (strlen(vocab_tokens[i]) == 1) continue;
    double loss_increase = computeTokenLoss(trainer, vocab_tokens[i], texts, text_limit);
    candidates[candidate_count].loss_increase = loss_increase;
    candidates[candidate_count].token = strdup(vocab_tokens[i]);
    candidate_count++;
  }
  qsort(candidates, candidate_count, sizeof(RemovalCandidate), compareRemovalCandidates);
  int actual_removals = (candidate_count < tokens_to_remove) ? candidate_count : tokens_to_remove;
  for (int i = 0; i < actual_removals; i++) {
    if (hashMapContains(trainer->vocab, candidates[i].token)) {
      hashMapRemove(trainer->vocab, candidates[i].token);
      heapRemove(trainer->vocab_heap, candidates[i].token);
      hashMapRemove(trainer->token_freqs, candidates[i].token);
      trieRemove(trainer->subword_trie, candidates[i].token);
    }
  }
  for (int i = 0; i < vocab_count; i++) free(vocab_tokens[i]);
  free(vocab_tokens); free(vocab_scores);
  for (int i = 0; i < candidate_count; i++) free(candidates[i].token);
  free(candidates);
  return true;
}

bool updateTokenScores(UnigramTrainer* trainer, const char** texts, int text_count) {
  if (!trainer || !texts || text_count <= 0) return false;
  FastHashMap* token_context_freq = hashmapCreate(hashMapSize(trainer->vocab));
  if (!token_context_freq) return false;
  int text_limit = (text_count < 3000) ? text_count : 3000;
  for (int i = 0; i < text_limit; i++) {
    if (!texts[i]) continue;
    TokenList* segmentation = viterbiDecode(trainer->decoder, texts[i], trainer->vocab);
    if (!segmentation) continue;
    for (int j = 0; j < segmentation->count; j++) {
      if (hashMapContains(trainer->vocab, segmentation->tokens[j])) {
        int* count = (int*)hashMapGet(token_context_freq, segmentation->tokens[j]);
        if (count) { (*count)++; }
        else {
          int* new_count = (int*)malloc(sizeof(int));
          if (new_count) { *new_count = 1; hashMapSet(token_context_freq, segmentation->tokens[j], new_count); }
        }
      }
    }
    tokenListDestroy(segmentation);
  }
  int total_freq = 0;
  HashMapIterator* freq_iter = hashMapIteratorCreate(token_context_freq);
  if (freq_iter) {
    const char* key; void* value;
    while (hashMapIteratorNext(freq_iter, &key, &value)) total_freq += *(int*)value;
    hashMapIteratorDestroy(freq_iter);
  }
  if (total_freq == 0) total_freq = 1;
  HashMapIterator* vocab_iter = hashMapIteratorCreate(trainer->vocab);
  if (vocab_iter) {
    const char* key; void* value;
    while (hashMapIteratorNext(vocab_iter, &key, &value)) {
      int* freq_ptr = (int*)hashMapGet(token_context_freq, key);
      int freq = freq_ptr ? *freq_ptr : 1;
      double new_score = log((double)freq) - log((double)total_freq);
      double* score_ptr = (double*)value;
      *score_ptr = new_score;
      if (hashMapContains(trainer->token_freqs, key)) {
        heapUpdateFreq(trainer->vocab_heap, key, freq);
        int* token_freq = (int*)hashMapGet(trainer->token_freqs, key);
        if (token_freq) *token_freq = freq;
      }
    }
    hashMapIteratorDestroy(vocab_iter);
  }
  HashMapIterator* titer = hashMapIteratorCreate(token_context_freq);
  if (titer) {
    const char* key; void* value;
    while (hashMapIteratorNext(titer, &key, &value)) free(value);
    hashMapIteratorDestroy(titer);
  }
  hashMapDestroy(token_context_freq);
  return true;
}

int compareTokenScores(const void* a, const void* b) {
  const TokenScore* ta = (const TokenScore*)a;
  const TokenScore* tb = (const TokenScore*)b;
  if (ta->score > tb->score) return -1;
  if (ta->score < tb->score) return 1;
  return 0;
}

bool trainUnigram(UnigramTrainer* trainer, const char** texts, int text_count, int num_iterations) {
  if (!trainer || !texts || text_count <= 0) return false;
  if (trainer->text_count == 0) {
    if (trainer->texts) {
      for (int i = 0; i < trainer->text_capacity; i++) {
        if (trainer->texts[i]) { free(trainer->texts[i]); trainer->texts[i] = NULL; }
      }
      free(trainer->texts);
    }
    trainer->text_capacity = text_count > 16 ? text_count : 16;
    trainer->texts = (char**)malloc(trainer->text_capacity * sizeof(char*));
    if (!trainer->texts) return false;
    for (int i = 0; i < trainer->text_capacity; i++) trainer->texts[i] = NULL;
    for (int i = 0; i < text_count; i++) {
      trainer->texts[i] = strdup(texts[i] ? texts[i] : "");
      if (!trainer->texts[i]) {
        for (int j = 0; j < i; j++) free(trainer->texts[j]);
        free(trainer->texts);
        trainer->texts = NULL;
        return false;
      }
    }
    trainer->text_count = text_count;
  }
  
  printf("Preprocessing %d texts...\n", trainer->text_count);
  if (!preprocessTexts(trainer)) { printf("Failed in preprocessTexts\n"); return false; }
  
  int train_text_limit = (trainer->text_count < 10000) ? trainer->text_count : 10000;
  trainer->text_count = train_text_limit;
  printf("Initializing seed vocabulary (using %d texts)...\n", trainer->text_count);
  
  if (!extractInitialSubwords(trainer)) { printf("Failed in extractInitialSubwords\n"); return false; }
  printf("Initial vocabulary size: %d\n", hashMapSize(trainer->vocab));
  
  int max_initial = trainer->vocab_size * 4;
  if (hashMapSize(trainer->vocab) > max_initial) {
    printf("Hard pruning initial vocab to %d tokens...\n", max_initial);
    pruneVocabStep(trainer, (const char**)trainer->texts, trainer->text_count < 200 ? trainer->text_count : 200, (double)max_initial / hashMapSize(trainer->vocab));
    printf("Initial vocab pruned to %d tokens\n", hashMapSize(trainer->vocab));
  }
  double prev_loss = DBL_MAX;
  for (int iteration = 0; iteration < num_iterations; iteration++) {
    printf("\nIteration %d/%d\n", iteration + 1, num_iterations);
    int loss_text_limit = (trainer->text_count < 1000) ? trainer->text_count : 1000;
    double current_loss = computeLoss(trainer, (const char**)trainer->texts, loss_text_limit);

    printf("  Current loss: %.4f\n", current_loss);

    if (fabs(prev_loss - current_loss) < CONVERGENCE_THRESHOLD) { printf("  Convergence reached\n"); break; }
    prev_loss = current_loss;
    updateTokenScores(trainer, (const char**)trainer->texts, trainer->text_count);
    printf("  Updated token scores\n");

    if (hashMapSize(trainer->vocab) > trainer->vocab_size) {
      pruneVocabStep(trainer, (const char**)trainer->texts, trainer->text_count, DEFAULT_REDUCTION_RATIO);
      printf("  Pruned vocabulary to %d tokens\n", hashMapSize(trainer->vocab));
    }

    cacheFree(trainer->loss_cache);
    trainer->loss_cache = cacheCreate(100000);
  }
  printf("\nFinalizing vocabulary...\n");
  FastHashMap* char_tokens = hashmapCreate(256);
  FastHashMap* other_tokens = hashmapCreate(hashMapSize(trainer->vocab));
  if (!char_tokens || !other_tokens) {
    if (char_tokens) hashMapDestroy(char_tokens);
    if (other_tokens) hashMapDestroy(other_tokens);
    return false;
  }
  HashMapIterator* iter = hashMapIteratorCreate(trainer->vocab);
  if (iter) {
    const char* key; void* value;
    while (hashMapIteratorNext(iter, &key, &value)) {
      if (strlen(key) == 1) {
        double* score = (double*)malloc(sizeof(double));
        if (score) { *score = *(double*)value; hashMapSet(char_tokens, key, score); }
      } else {
        double* score = (double*)malloc(sizeof(double));
        if (score) { *score = *(double*)value; hashMapSet(other_tokens, key, score); }
      }
    }
    hashMapIteratorDestroy(iter);
  }
  int other_count = hashMapSize(other_tokens);
  TokenScore* sorted_tokens = (TokenScore*)malloc(other_count * sizeof(TokenScore));
  if (!sorted_tokens) {
    hashMapDestroy(char_tokens);
    hashMapDestroy(other_tokens);
    return false;
  }
  HashMapIterator* other_iter = hashMapIteratorCreate(other_tokens);
  int idx = 0;
  if (other_iter) {
    const char* key; void* value;
    while (hashMapIteratorNext(other_iter, &key, &value) && idx < other_count) {
      sorted_tokens[idx].token = strdup(key);
      sorted_tokens[idx].score = *(double*)value;
      idx++;
    }
    hashMapIteratorDestroy(other_iter);
  }
  qsort(sorted_tokens, other_count, sizeof(TokenScore), compareTokenScores);
  hashMapClear(trainer->final_vocab);
  int char_count = hashMapSize(char_tokens);
  int final_other_limit = trainer->vocab_size - char_count;
  if (final_other_limit > other_count) final_other_limit = other_count;
  for (int i = 0; i < final_other_limit; i++) {
    double* score = (double*)malloc(sizeof(double));
    if (score) { *score = sorted_tokens[i].score; hashMapSet(trainer->final_vocab, sorted_tokens[i].token, score); }
  }
  HashMapIterator* char_iter = hashMapIteratorCreate(char_tokens);
  if (char_iter) {
    const char* key; void* value;
    while (hashMapIteratorNext(char_iter, &key, &value)) {
      double* score = (double*)malloc(sizeof(double));
      if (score) { *score = *(double*)value; hashMapSet(trainer->final_vocab, key, score); }
    }
    hashMapIteratorDestroy(char_iter);
  }
  printf("Training completed. Final vocabulary size: %d\n", hashMapSize(trainer->final_vocab));
  for (int i = 0; i < other_count; i++) free(sorted_tokens[i].token);
  free(sorted_tokens);
  HashMapIterator* ct = hashMapIteratorCreate(char_tokens);
  if (ct) {
    const char* key; void* value;
    while (hashMapIteratorNext(ct, &key, &value)) free(value);
    hashMapIteratorDestroy(ct);
  }
  HashMapIterator* ot = hashMapIteratorCreate(other_tokens);
  if (ot) {
    const char* key; void* value;
    while (hashMapIteratorNext(ot, &key, &value)) free(value);
    hashMapIteratorDestroy(ot);
  }
  hashMapDestroy(char_tokens);
  hashMapDestroy(other_tokens);
  return true;
}

bool getVocab(UnigramTrainer* trainer, char*** tokens, double** scores, int* count) {
  if (!trainer || !tokens || !scores || !count) return false;
  *count = hashMapSize(trainer->final_vocab);
  if (*count == 0) return true;
  *tokens = (char**)malloc(*count * sizeof(char*));
  *scores = (double*)malloc(*count * sizeof(double));
  if (!*tokens || !*scores) return false;
  HashMapIterator* iter = hashMapIteratorCreate(trainer->final_vocab);
  int idx = 0;
  if (iter) {
    const char* key; void* value;
    while (hashMapIteratorNext(iter, &key, &value) && idx < *count) {
      (*tokens)[idx] = strdup(key);
      (*scores)[idx] = *(double*)value;
      idx++;
    }
    hashMapIteratorDestroy(iter);
  }
  return true;
}

bool saveVocab(UnigramTrainer* trainer, const char* filepath) {
  if (!trainer || !filepath) return false;
  FILE* file = fopen(filepath, "w");
  if (!file) return false;
  int count = hashMapSize(trainer->final_vocab);
  TokenScore* sorted_vocab = (TokenScore*)malloc(count * sizeof(TokenScore));
  if (!sorted_vocab) { fclose(file); return false; }
  HashMapIterator* iter = hashMapIteratorCreate(trainer->final_vocab);
  int idx = 0;
  if (iter) {
    const char* key; void* value;
    while (hashMapIteratorNext(iter, &key, &value) && idx < count) {
      sorted_vocab[idx].token = strdup(key);
      sorted_vocab[idx].score = *(double*)value;
      idx++;
    }
    hashMapIteratorDestroy(iter);
  }
  qsort(sorted_vocab, count, sizeof(TokenScore), compareTokenScores);
  for (int i = 0; i < count; i++) {
    fprintf(file, "%s\t%.6f\n", sorted_vocab[i].token, sorted_vocab[i].score);
    free(sorted_vocab[i].token);
  }
  free(sorted_vocab);
  fclose(file);
  return true;
}

bool loadVocab(UnigramTrainer* trainer, const char* filepath) {
  if (!trainer || !filepath) return false;
  FILE* file = fopen(filepath, "r");
  if (!file) return false;
  hashMapClear(trainer->final_vocab);
  char line[MAX_TOKEN_LEN * 2];
  while (fgets(line, sizeof(line), file)) {
    char* tab_pos = strchr(line, '\t');
    if (!tab_pos) continue;
    *tab_pos = '\0';
    char* score_str = tab_pos + 1;
    char* newline = strchr(score_str, '\n');
    if (newline) *newline = '\0';
    double* score_ptr = (double*)malloc(sizeof(double));
    if (score_ptr) { *score_ptr = atof(score_str); hashMapSet(trainer->final_vocab, line, score_ptr); }
  }
  fclose(file);
  return true;
}