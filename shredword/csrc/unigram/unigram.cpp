#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include "../inc/hash.h"
#include "unigram.h"

UnigramTrainer* trainerCreate(int vs, float cc, int msl, int sss) {
  UnigramTrainer* trainer = (UnigramTrainer*)malloc(sizeof(UnigramTrainer));
  trainer->vocab_size = vs;
  trainer->character_coverage = cc;
  trainer->max_len = msl;
  trainer->seed_size = sss;
  trainer->vocab_heap = heapCreate();
  trainer->token_freqs = hashmapCreate(INITIAL_SIZE);
  trainer->subword_trie = trieCreate();
  trainer->extractor = subwordExtractorCreate();
  trainer->decoder = viterbiDecoderCreate();
  trainer->loss_cache = cacheCreate(INITIAL_SIZE);
  trainer->vocab = hashmapCreate(INITIAL_SIZE);
  trainer->final_vocab = hashmapCreate(INITIAL_SIZE);

  trainer->texts = (char**)malloc(trainer->text_capacity * sizeof(char*));
  if (!trainer->texts) {
    fprintf(stderr, "Couldn't initialize the Unigram Trainer!");
    exit(EXIT_FAILURE);
  }
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
    free(trainer->texts[i]);
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
    trainer->texts = new_texts;
    trainer->text_capacity = new_capacity;
  }

  trainer->texts[trainer->text_count] = strdup(text);
  if (!trainer->texts[trainer->text_count]) return false;

  trainer->text_count++;
  return true;
}

bool preprocessTexts(UnigramTrainer* trainer) {
  if (!trainer || trainer->text_count == 0) return false;

  FastHashMap* char_counts = hashmapCreate(256);
  char** processed_texts = (char**)malloc(trainer->text_count * sizeof(char*));
  int processed_count = 0;

  for (int i = 0; i < trainer->text_count; i++) {
    NormalizedText* nt = create_normalized_text(strlen(trainer->texts[i]) * 2);
    if (!nt) continue;

    if (normalize_text_fast(trainer->texts[i], nt) == 0) {
      processed_texts[processed_count] = strdup(nt->data);
      if (processed_texts[processed_count]) {
        for (int j = 0; j < nt->length; j++) {
          char key[2] = {nt->data[j], '\0'};
          int* count = (int*)hashMapGet(char_counts, key);
          if (count) { (*count)++; }
          else {
            int* new_count = (int*)malloc(sizeof(int));
            *new_count = 1;
            hashMapSet(char_counts, key, new_count);
          } trainer->total_chars++;
        } processed_count++;
      } } free_normalized_text(nt);
  }

  int covered_chars = 0;
  int required_chars = (int)(trainer->total_chars * trainer->character_coverage);
  for (int i = 0; i < trainer->text_count; i++) free(trainer->texts[i]);
  free(trainer->texts);

  trainer->texts = processed_texts;
  trainer->text_count = processed_count;
  trainer->text_capacity = processed_count;
  hashMapDestroy(char_counts);
  return true;
}

bool extractInitialSubwords(UnigramTrainer* trainer) {
  if (!trainer) return false;

  FastHashMap* all_subwords = hashmapCreate(INITIAL_SIZE * 4);
  FastHashMap* char_freq = hashmapCreate(256);

  int text_limit = (trainer->text_count < 10000) ? trainer->text_count : 10000;
  for (int i = 0; i < text_limit; i++) {
    SubwordSet* subwords = extractSubwords(trainer->extractor, trainer->texts[i], trainer->max_len);
    if (!subwords) continue;

    for (int j = 0; j < subwords->count; j++) hashMapSet(all_subwords, subwords->subwords[j], (void*)1);
    for (int k = 0; trainer->texts[i][k]; k++) {
      char char_key[2] = {trainer->texts[i][k], '\0'};
      int* count = (int*)hashMapGet(char_freq, char_key);
      if (count) (*count)++; else {
        int* new_count = (int*)malloc(sizeof(int));
        *new_count = 1;
        hashMapSet(char_freq, char_key, new_count);
      }
    }
    subwordSetDestroy(subwords);
  }

  HashMapIterator* char_iter = hashMapIteratorCreate(char_freq);
  const char* char_key;
  void* char_value;
  while (hashMapIteratorNext(char_iter, &char_key, &char_value)) { hashMapSet(all_subwords, char_key, (void*)1); }
  hashMapIteratorDestroy(char_iter);

  for (int i = 0; i < trainer->text_count; i++) {
    SubwordSet* text_subwords = extractSubwords(trainer->extractor, trainer->texts[i], trainer->max_len);
    if (!text_subwords) continue;
    
    for (int j = 0; j < text_subwords->count; j++) {
      if (hashMapContains(all_subwords, text_subwords->subwords[j])) {
        int* count = (int*)hashMapGet(trainer->token_freqs, text_subwords->subwords[j]);
        if (count) (*count)++; else {
          int* new_count = (int*)malloc(sizeof(int));
          *new_count = 1;
          hashMapSet(trainer->token_freqs, text_subwords->subwords[j], new_count);
        }
      }
    }
    subwordSetDestroy(text_subwords);
  }

  HashMapIterator* freq_iter = hashMapIteratorCreate(trainer->token_freqs);
  const char* token;
  void* freq_value;
  while (hashMapIteratorNext(freq_iter, &token, &freq_value)) {
    int freq = *(int*)freq_value;
    if (freq > 1 && heapSize(trainer->vocab_heap) < trainer->seed_size) {
      heapPush(trainer->vocab_heap, token, freq);
      double* log_freq = (double*)malloc(sizeof(double));
      *log_freq = log((double)freq);
      hashMapSet(trainer->vocab, token, log_freq);
      trieInsert(trainer->subword_trie, token, freq);
    }
  }
  hashMapIteratorDestroy(freq_iter);

  hashMapDestroy(all_subwords);
  hashMapDestroy(char_freq);
  return true;
}

float computeLoss(UnigramTrainer* trainer, const char** texts, int text_count) {
  if (!trainer || !texts || text_count <= 0) return 0.0;

  double total_loss = 0.0;
  int total_len = 0;

  for (int i = 0; i < text_count; i++) {
    if (!texts[i]) continue;

    uint64_t cache_key = stringHash64(texts[i]);
    int cached_loss = cacheGet(trainer->loss_cache, cache_key);
    if (cached_loss == -1) {
      total_loss += (double)cached_loss / MAX_TEXTS_FOR_TOKEN_LOSS;
      total_len += strlen(texts[i]);
      continue;
    }
    TokenList* segmenetation = viterbiDecode(trainer->decoder, texts[i], trainer->vocab);
    if (!segmenetation) continue;
    double text_loss = 0.0;
    for (int j = 0; j < segmenetation->count; j++) {
      double* token_score = (double*)hashMapGet(trainer->vocab, segmenetation->tokens[j]);
      double score = token_score ? *token_score : - UNKNOWN_TOKEN_SCORE;
      text_loss -= score;
    }

    cachePut(trainer->loss_cache, (int)cache_key, (int)text_loss * MAX_TEXTS_FOR_TOKEN_LOSS);
    total_loss += text_loss;
    total_len += strlen(texts[i]);
    tokenListDestroy(segmenetation);
  }
  return total_len > 0 ? total_loss / total_len : 0.0;
}

double computeTokenLoss(UnigramTrainer* trainer, const char* token, const char** texts, int text_count) {
  if (!trainer || !token || !texts || text_count <= 0) return 0.0;

  FastHashMap* temp_vocab = hashmapCreate(hashMapSize(trainer->vocab));
  if (!temp_vocab) return 0.0;

  HashMapIterator* iter = hashMapIteratorCreate(trainer->vocab);
  const char* key;
  void* value;
  while (hashMapIteratorNext(iter, &key, &value)) {
    if (strcmp(key, token) != 0) {
      double* original_score = (double*)value;
      double* new_score = (double*)malloc(sizeof(double));
      *new_score = *original_score;
      hashMapSet(temp_vocab, key, new_score);
    }
  }
  hashMapIteratorDestroy(iter);
  ViterbiDecoder* temp_decoder = viterbiDecoderCreate();
  if (!temp_decoder) {
    hashMapDestroy(temp_vocab);
    return 0.0;
  }

  double total_loss = 0.0;
  for (int i = 0; i < text_count; i++) {
    if (!texts[i] || !strstr(texts[i], token)) continue;
    TokenList* segmentation = viterbiDecode(temp_decoder, texts[i], temp_vocab);
    if (!segmentation) continue;

    for (int j = 0; j < segmentation->count; j++) {
      double* token_score = (double*)hashMapGet(temp_vocab, segmentation->tokens[j]);
      double score = token_score ? *token_score : -20.0;
      total_loss -= score;
    }    
    tokenListDestroy(segmentation);
  }
  viterbiDecoderDestroy(temp_decoder);
  hashMapDestroy(temp_vocab);
  return total_loss;
}

int compareRemovalCandidates(const void* a, const void* b) {
  RemovalCandidate* ca = (RemovalCandidate*)a;
  RemovalCandidate* cb = (RemovalCandidate*)b;
  if (ca->loss_increase < cb->loss_increase) return -1;
  if (ca->loss_increase > cb->loss_increase) return 1;
  return 0;
}

void shuffleVocabItems(char** tokens, double** scores, int count) {
  srand((unsigned int)time(NULL));
  for (int i = count - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    char* temp_token = tokens[i];
    double* temp_score = scores[i];
    tokens[i] = tokens[j];
    scores[i] = scores[j];
    tokens[j] = temp_token;
    scores[j] = temp_score;
  }
}

bool pruneVocabStep(UnigramTrainer* trainer, const char** texts, int text_count, double reduction_ratio) {
  if (!trainer || hashMapSize(trainer->vocab) <= trainer->vocab_size) return true;

  int current_size = hashMapSize(trainer->vocab);
  int target_size = trainer->vocab_size > (int)(current_size * reduction_ratio) ? trainer->vocab_size : (int)(current_size * reduction_ratio);
  int tokens_to_remove = current_size - target_size;

  char** vocab_tokens = (char**)malloc(current_size * sizeof(char*));
  double** vocab_scores = (double**)malloc(current_size * sizeof(double*));
  if (!vocab_tokens || !vocab_scores) return false;

  HashMapIterator* iter = hashMapIteratorCreate(trainer->vocab);
  const char* key;
  void* value;
  int vocab_count = 0;

  while (hashMapIteratorNext(iter, &key, &value)) {
    vocab_tokens[vocab_count] = strdup(key);
    vocab_scores[vocab_count] = (double*)value;
    vocab_count++;
  }
  hashMapIteratorDestroy(iter);  
  shuffleVocabItems(vocab_tokens, vocab_scores, vocab_count);

  int candidates_limit = (vocab_count < tokens_to_remove * 3) ? vocab_count : tokens_to_remove * 3;
  RemovalCandidate* candidates = (RemovalCandidate*)malloc(candidates_limit * sizeof(RemovalCandidate));
  if (!candidates) {
    for (int i = 0; i < vocab_count; i++) free(vocab_tokens[i]);
    free(vocab_tokens);
    free(vocab_scores);
    return false;
  }  
  int candidate_count = 0, text_limit = (text_count < 1000) ? text_count : 1000;
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
  free(vocab_tokens);
  free(vocab_scores);
  for (int i = 0; i < candidate_count; i++) free(candidates[i].token);
  free(candidates);
  return true;
}

bool updateTokenScores(UnigramTrainer* trainer, const char** texts, int text_count) {
  if (!trainer || !texts || text_count <= 0) return false;
  
  FastHashMap* token_context_freq = hashmapCreate(hashMapSize(trainer->vocab));
  if (!token_context_freq) return false;  
  int text_limit = (text_count < 5000) ? text_count : 5000;
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
          *new_count = 1;
          hashMapSet(token_context_freq, segmentation->tokens[j], new_count);
        }
      }
    }  tokenListDestroy(segmentation);
  }

  int total_freq = 0;
  HashMapIterator* freq_iter = hashMapIteratorCreate(token_context_freq);
  const char* key;
  void* value;
  while (hashMapIteratorNext(freq_iter, &key, &value)) total_freq += *(int*)value;
  hashMapIteratorDestroy(freq_iter);

  if (total_freq == 0) {
    hashMapDestroy(token_context_freq);
    return true;
  }
  
  HashMapIterator* vocab_iter = hashMapIteratorCreate(trainer->vocab);
  while (hashMapIteratorNext(vocab_iter, &key, &value)) {
    int* freq_ptr = (int*)hashMapGet(token_context_freq, key);
    int freq = freq_ptr ? *freq_ptr : 1;

    double new_score = log((double)freq / total_freq) + log((double)total_freq);
    double* score_ptr = (double*)value;
    *score_ptr = new_score;

    if (hashMapContains(trainer->token_freqs, key)) {
      heapUpdateFreq(trainer->vocab_heap, key, freq);
      int* token_freq = (int*)hashMapGet(trainer->token_freqs, key);
      if (token_freq) *token_freq = freq;
    }
  }
  hashMapIteratorDestroy(vocab_iter);
  hashMapDestroy(token_context_freq);
  return true;
}

int compareTokenScores(const void* a, const void* b) {
  TokenScore* ta = (TokenScore*)a;
  TokenScore* tb = (TokenScore*)b;
  if (ta->score > tb->score) return -1;
  if (ta->score < tb->score) return 1;
  return 0;
}

bool trainUnigram(UnigramTrainer* trainer, const char** texts, int text_count, int num_iterations) {
  if (!trainer || !texts || text_count <= 0) return false;  
  printf("Preprocessing %d texts...\n", text_count);
  if (!preprocessTexts(trainer)) return false;

  int train_text_limit = (trainer->text_count < 50000) ? trainer->text_count : 50000;
  trainer->text_count = train_text_limit;
  printf("Initializing seed vocabulary...\n");
  if (!extractInitialSubwords(trainer)) return false;
  printf("Initial vocabulary size: %d\n", hashMapSize(trainer->vocab));  
  double prev_loss = DBL_MAX;

  for (int iteration = 0; iteration < num_iterations; iteration++) {
    printf("Iteration %d/%d\n", iteration + 1, num_iterations);
    int loss_text_limit = (trainer->text_count < 2000) ? trainer->text_count : 2000;
    const char** loss_texts = (const char**)trainer->texts;
    double current_loss = computeLoss(trainer, loss_texts, loss_text_limit);
    printf("  Current loss: %.4f\n", current_loss);

    if (fabs(prev_loss - current_loss) < 0.001) {
      printf("  Convergence reached\n");
      break;
    }
    prev_loss = current_loss;

    const char** update_texts = (const char**)trainer->texts;
    updateTokenScores(trainer, update_texts, trainer->text_count);
    printf("  Updated token scores\n");
    if (hashMapSize(trainer->vocab) > trainer->vocab_size) {
      const char** prune_texts = (const char**)trainer->texts;
      pruneVocabStep(trainer, prune_texts, trainer->text_count, 0.8);
      printf("  Pruned vocabulary to %d tokens\n", hashMapSize(trainer->vocab));
    }
    cacheFree(trainer->loss_cache);
    trainer->loss_cache = cacheCreate(100000);
  }

  FastHashMap* char_tokens = hashmapCreate(256);
  FastHashMap* other_tokens = hashmapCreate(hashMapSize(trainer->vocab));  
  HashMapIterator* iter = hashMapIteratorCreate(trainer->vocab);
  const char* key;
  void* value;

  while (hashMapIteratorNext(iter, &key, &value)) {
    if (strlen(key) == 1) {
      double* score = (double*)malloc(sizeof(double));
      *score = *(double*)value;
      hashMapSet(char_tokens, key, score);
    } else {
      double* score = (double*)malloc(sizeof(double));
      *score = *(double*)value;
      hashMapSet(other_tokens, key, score);
    }
  }
  hashMapIteratorDestroy(iter);
  int other_count = hashMapSize(other_tokens);
  TokenScore* sorted_tokens = (TokenScore*)malloc(other_count * sizeof(TokenScore));
  if (!sorted_tokens) {
    hashMapDestroy(char_tokens);
    hashMapDestroy(other_tokens);
    return false;
  }
  HashMapIterator* other_iter = hashMapIteratorCreate(other_tokens);
  int idx = 0;
  while (hashMapIteratorNext(other_iter, &key, &value)) {
    sorted_tokens[idx].token = strdup(key);
    sorted_tokens[idx].score = *(double*)value;
    idx++;
  }
  hashMapIteratorDestroy(other_iter);

  qsort(sorted_tokens, other_count, sizeof(TokenScore), compareTokenScores);
  hashMapClear(trainer->final_vocab);
  int char_count = hashMapSize(char_tokens);
  int final_other_limit = trainer->vocab_size - char_count;
  if (final_other_limit > other_count) final_other_limit = other_count;

  for (int i = 0; i < final_other_limit; i++) {
    double* score = (double*)malloc(sizeof(double));
    *score = sorted_tokens[i].score;
    hashMapSet(trainer->final_vocab, sorted_tokens[i].token, score);
  }

  HashMapIterator* char_iter = hashMapIteratorCreate(char_tokens);
  while (hashMapIteratorNext(char_iter, &key, &value)) {
    double* score = (double*)malloc(sizeof(double));
    *score = *(double*)value;
    hashMapSet(trainer->final_vocab, key, score);
  }
  hashMapIteratorDestroy(char_iter);
  printf("Training completed. Final vocabulary size: %d\n", hashMapSize(trainer->final_vocab));
  for (int i = 0; i < other_count; i++) free(sorted_tokens[i].token);
  free(sorted_tokens);
  hashMapDestroy(char_tokens);
  hashMapDestroy(other_tokens);
  return true;
}

bool getVocab(UnigramTrainer* trainer, char*** tokens, double** scores, int* count) {
  if (!trainer || !tokens || !scores || !count) return false;

  *count = hashMapSize(trainer->final_vocab);
  if (*count == 0) return true;
  *tokens = (char**)malloc(*count * sizeof(char*)); *scores = (double*)malloc(*count * sizeof(double));
  if (!*tokens || !*scores) return false;

  HashMapIterator* iter = hashMapIteratorCreate(trainer->final_vocab);
  const char* key;
  void* value;
  int idx = 0;  
  while (hashMapIteratorNext(iter, &key, &value) && idx < *count) {
    (*tokens)[idx] = strdup(key);
    (*scores)[idx] = *(double*)value;
    idx++;
  }
  hashMapIteratorDestroy(iter);
  return true;
}

bool saveVocab(UnigramTrainer* trainer, const char* filepath) {
  if (!trainer || !filepath) return false;

  FILE* file = fopen(filepath, "w");
  if (!file) return false;
  int count = hashMapSize(trainer->final_vocab);
  TokenScore* sorted_vocab = (TokenScore*)malloc(count * sizeof(TokenScore));
  if (!sorted_vocab) {
    fclose(file);
    return false;
  }

  HashMapIterator* iter = hashMapIteratorCreate(trainer->final_vocab);
  const char* key;
  void* value;
  int idx = 0;

  while (hashMapIteratorNext(iter, &key, &value) && idx < count) {
    sorted_vocab[idx].token = strdup(key);
    sorted_vocab[idx].score = *(double*)value;
    idx++;
  }
  hashMapIteratorDestroy(iter);
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
    char* token = line;
    char* score_str = tab_pos + 1;

    char* newline = strchr(score_str, '\n');
    if (newline) *newline = '\0';
    double score = atof(score_str);
    double* score_ptr = (double*)malloc(sizeof(double));
    if (!score_ptr) continue;

    *score_ptr = score;
    hashMapSet(trainer->final_vocab, token, score_ptr);
  }

  fclose(file);
  return true;
}