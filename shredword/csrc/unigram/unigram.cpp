#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include "unigram.h"
#include "../inc/hash.h"

UnigramTrainer* trainer_create(int vocab_size, float character_coverage, int max_len, int seed_size) {
  UnigramTrainer *trainer = (UnigramTrainer*)malloc(sizeof(UnigramTrainer));
  trainer->vocab_size = vocab_size ? vocab_size : DEFAULT_VOCAB_SIZE;
  trainer->character_coverage = character_coverage ? character_coverage : DEFAULT_COVERAGE;
  trainer->max_len = max_len ? max_len : DEFAULT_MAX_LEN;
  trainer->seed_size = seed_size ? seed_size : DEFAULT_SEED_SIZE;

  trainer->vocab_heap = heap_create();
  trainer->token_freqs = hashmap_create(INITIAL_HASH_SIZE);
  trainer->vocab = hashmap_create(INITIAL_HASH_SIZE);
  trainer->final_vocab = hashmap_create(INITIAL_HASH_SIZE);
  trainer->subword_trie = trie_create();
  trainer->extractor = extractor_create();
  trainer->decoder = viterbi_create();
  trainer->loss_cache = cache_create(100000);
  trainer->texts = NULL, trainer->total_chars = 0;

  return trainer;
}

void trainer_destroy(UnigramTrainer *trainer) {
  if (!trainer) return;
  if (trainer->texts) text_array_destroy(trainer->texts);
  hashmap_destroy(trainer->token_freqs);
  hashmap_destroy(trainer->vocab);
  hashmap_destroy(trainer->final_vocab);
  trie_destroy(trainer->subword_trie);
  extractor_destroy(trainer->extractor);
  viterbi_destroy(trainer->decoder);
  free(trainer);
}

TextArray* text_array_create(int capacity) {
  TextArray *array = (TextArray*)malloc(sizeof(TextArray));
  array->texts = (char**)malloc(capacity * sizeof(char*));
  array->count = 0;
  array->capacity = capacity;
  return array;
}

void text_array_destroy(TextArray *array) {
  if (!array) return;
  for (int i = 0; i < array->count; i++) { if (array->texts[i]) free(array->texts[i]); }
  free(array->texts);
  free(array);
}

void text_array_add(TextArray *array, const char *text) {
  if (array->count >= array->capacity) return;
  array->texts[array->count] = strdup(text);
  array->count++;
}

int is_printable_char(char c) { return c >= 32 && c <= 126; }

TextArray* preprocess_texts(UnigramTrainer *trainer, char **input_texts, int text_count) {
  TextArray *processed = text_array_create(text_count);
  FastHashMap *char_counts = hashmap_create(256);

  for (int i = 0; i < text_count; i++) {
    if (!input_texts[i]) continue;

    int len = strlen(input_texts[i]);
    char *clean_text = (char*)malloc(len * 3 + 2);
    int pos = 0;

    clean_text[pos++] = '\xE2';
    clean_text[pos++] = '\x96';
    clean_text[pos++] = '\x81';

    for (int j = 0; j < len; j++) {
      char c = input_texts[i][j];
      if (is_printable_char(c) && c != ' ') {
        clean_text[pos++] = c;
        char key[2] = {c, '\0'};
        float count = hashmap_get(char_counts, key);
        hashmap_put(char_counts, key, count + 1);
      } else {
        clean_text[pos++] = '\xE2';
        clean_text[pos++] = '\x96';
        clean_text[pos++] = '\x81';
      }
    }
    clean_text[pos] = '\0';

    if (pos > 3) { text_array_add(processed, clean_text); }
    free(clean_text);
  }

  hashmap_destroy(char_counts);
  return processed;
}

void initialize_seed_vocab(UnigramTrainer *trainer, TextArray *texts) {
  FastHashMap *subword_freq = hashmap_create(INITIAL_HASH_SIZE * 4);
  FastHashMap *char_freq = hashmap_create(256);

  int process_count = texts->count < SEED_TEXTS ? texts->count : SEED_TEXTS;
  for (int i = 0; i < process_count; i++) {
    SubwordSet *subwords = extract_subwords(trainer->extractor, texts->texts[i], trainer->max_len);

    for (size_t j = 0; j < subwords->size; j++) {
      float count = hashmap_get(subword_freq, subwords->items[j]);
      hashmap_put(subword_freq, subwords->items[j], count + 1);
    }

    subword_set_destroy(subwords);    
    for (int j = 0; texts->texts[i][j]; j++) {
      char key[2] = {texts->texts[i][j], '\0'};
      float count = hashmap_get(char_freq, key);
      hashmap_put(char_freq, key, count + 1);
    }
  }

  FreqToken *candidates = (FreqToken*)malloc(trainer->seed_size * sizeof(FreqToken));
  int candidate_count = 0;

  for (int i = 0; i < texts->count && candidate_count < trainer->seed_size; i++) {
    SubwordSet *subwords = extract_subwords(trainer->extractor, texts->texts[i], trainer->max_len);
    for (size_t j = 0; j < subwords->size && candidate_count < trainer->seed_size; j++) {
      float freq = hashmap_get(subword_freq, subwords->items[j]);
      if (freq > 1) {
        candidates[candidate_count].freq = freq;
        candidates[candidate_count].token = strdup(subwords->items[j]);
        candidate_count++;
      }
    }
    subword_set_destroy(subwords);
  }

  qsort(candidates, candidate_count, sizeof(FreqToken), compare_freq_tokens);
  for (int i = 0; i < candidate_count && i < trainer->seed_size; i++) {
    float score = logf(candidates[i].freq);
    hashmap_put(trainer->vocab, candidates[i].token, score);
    hashmap_put(trainer->token_freqs, candidates[i].token, candidates[i].freq);
    heap_push(trainer->vocab_heap, candidates[i].token, (int)candidates[i].freq);
    trie_insert(trainer->subword_trie, candidates[i].token, (int)candidates[i].freq);
    free(candidates[i].token);
  }
  free(candidates);
  hashmap_destroy(subword_freq);
  hashmap_destroy(char_freq);
}

float compute_loss(UnigramTrainer *trainer, TextArray *texts) {
  float total_loss = 0.0f;
  int total_len = 0;
  int process_count = texts->count < LOSS_TEXTS ? texts->count : LOSS_TEXTS;
  for (int i = 0; i < process_count; i++) {
    int cache_key = djb2_hash(texts->texts[i]) % 100000;
    int cached_loss = cache_get(trainer->loss_cache, cache_key);

    if (cached_loss != -1) {
      total_loss += cached_loss;
      total_len += strlen(texts->texts[i]);
      continue;
    }

    ViterbiResult *result = viterbi_decode(trainer->decoder, texts->texts[i], trainer->vocab);
    float text_loss = 0.0f;
    for (size_t j = 0; j < result->count; j++) {
      float score = hashmap_get(trainer->vocab, result->tokens[j]);
      text_loss -= (score != 0.0f) ? score : -20.0f;
    }

    cache_put(trainer->loss_cache, cache_key, (int)text_loss);
    total_loss += text_loss;
    total_len += strlen(texts->texts[i]);
    viterbi_result_destroy(result);
  }
  
  return total_len > 0 ? total_loss / total_len : 0.0f;
}

float compute_token_loss(UnigramTrainer *trainer, const char *token, TextArray *texts) {
  FastHashMap *temp_vocab = hashmap_create(trainer->vocab->size);

  for (int i = 0; i < trainer->vocab->size; i++) {
    HashEntry *entry = trainer->vocab->buckets[i];
    while (entry) {
      if (strcmp(entry->key, token) != 0) {
        hashmap_put(temp_vocab, entry->key, entry->value);
      }
      entry = entry->next;
    }
  }
  float total_loss = 0.0f;
  ViterbiDecoder *temp_decoder = viterbi_create();

  int process_count = texts->count < PRUNE_TEXTS ? texts->count : PRUNE_TEXTS;
  for (int i = 0; i < process_count; i++) {
    if (!strstr(texts->texts[i], token)) continue;
    ViterbiResult *result = viterbi_decode(temp_decoder, texts->texts[i], temp_vocab);

    for (size_t j = 0; j < result->count; j++) {
      float score = hashmap_get(temp_vocab, result->tokens[j]);
      total_loss -= (score != 0.0f) ? score : -20.0f;
    }
    viterbi_result_destroy(result);
  }
  
  viterbi_destroy(temp_decoder);
  hashmap_destroy(temp_vocab);
  return total_loss;
}

void prune_vocab_step(UnigramTrainer *trainer, TextArray *texts, float reduction_ratio) {
  if (trainer->vocab->count <= trainer->vocab_size) return;  
  int target_size = trainer->vocab_size > (int)(trainer->vocab->count * reduction_ratio) ? trainer->vocab_size : (int)(trainer->vocab->count * reduction_ratio);
  int tokens_to_remove = trainer->vocab->count - target_size;

  RemovalCandidate *candidates = (RemovalCandidate*)malloc(tokens_to_remove * 3 * sizeof(RemovalCandidate));
  int candidate_count = 0;

  for (int i = 0; i < trainer->vocab->size && candidate_count < tokens_to_remove * 3; i++) {
    for (HashEntry *entry = trainer->vocab->buckets[i]; entry && candidate_count < tokens_to_remove * 3; entry = entry->next) {
      if (strlen(entry->key) <= 1) continue;
      float loss_increase = compute_token_loss(trainer, entry->key, texts);
      candidates[candidate_count].loss_increase = loss_increase;
      candidates[candidate_count].token = strdup(entry->key);
      candidate_count++;
    }
  }

  qsort(candidates, candidate_count, sizeof(RemovalCandidate), [](const void *a, const void *b) {
    float diff = ((RemovalCandidate*)a)->loss_increase - ((RemovalCandidate*)b)->loss_increase;
    return (diff > 0) ? 1 : (diff < 0) ? -1 : 0;
  });

  for (int i = 0; i < tokens_to_remove && i < candidate_count; i++) {
    if (hashmap_contains(trainer->vocab, candidates[i].token)) {
      hashmap_remove(trainer->vocab, candidates[i].token);
      heap_remove(trainer->vocab_heap, candidates[i].token);
      hashmap_remove(trainer->token_freqs, candidates[i].token);
    }
    free(candidates[i].token);
  }
  free(candidates);
}

void update_token_scores(UnigramTrainer *trainer, TextArray *texts) {
  FastHashMap *token_context_freq = hashmap_create(trainer->vocab->size);
  int process_count = texts->count < SCORE_TEXTS ? texts->count : SCORE_TEXTS;
  for (int i = 0; i < process_count; i++) {
    ViterbiResult *result = viterbi_decode(trainer->decoder, texts->texts[i], trainer->vocab);

    for (size_t j = 0; j < result->count; j++) {
      if (hashmap_contains(trainer->vocab, result->tokens[j])) {
        float count = hashmap_get(token_context_freq, result->tokens[j]);
        hashmap_put(token_context_freq, result->tokens[j], count + 1);
      }
    }
    viterbi_result_destroy(result);
  }

  float total_freq = 0.0f;
  for (int i = 0; i < token_context_freq->size; i++) {
    HashEntry *entry = token_context_freq->buckets[i];
    while (entry) {
      total_freq += entry->value;
      entry = entry->next;
    }
  }

  if (total_freq == 0.0f) {
    hashmap_destroy(token_context_freq);
    return;
  }

  for (int i = 0; i < trainer->vocab->size; i++) {
    HashEntry *entry = trainer->vocab->buckets[i];
    while (entry) {
      float freq = hashmap_get(token_context_freq, entry->key);
      if (freq == 0.0f) freq = 1.0f;
      float new_score = logf(freq / total_freq) + logf(total_freq);
      hashmap_put(trainer->vocab, entry->key, new_score);
      if (hashmap_contains(trainer->token_freqs, entry->key)) {
        heap_update_freq(trainer->vocab_heap, entry->key, (int)freq);
        hashmap_put(trainer->token_freqs, entry->key, freq);
      }
      entry = entry->next;
    }
  }
  hashmap_destroy(token_context_freq);
}

FastHashMap* train_unigram(UnigramTrainer *trainer, char **input_texts, int text_count, int num_iterations) {
  printf("Preprocessing %d texts...\n", text_count);
  TextArray *processed_texts = preprocess_texts(trainer, input_texts, text_count);

  trainer->texts = text_array_create(MAX_TEXTS);
  int copy_count = processed_texts->count < MAX_TEXTS ? processed_texts->count : MAX_TEXTS;
  for (int i = 0; i < copy_count; i++) { text_array_add(trainer->texts, processed_texts->texts[i]); }

  printf("Initializing seed vocabulary...\n");
  initialize_seed_vocab(trainer, trainer->texts);
  printf("Initial vocabulary size: %d\n", trainer->vocab->count);

  float prev_loss = INFINITY;
  for (int iteration = 0; iteration < num_iterations; iteration++) {
    printf("Iteration %d/%d\n", iteration + 1, num_iterations);

    float current_loss = compute_loss(trainer, trainer->texts);
    printf("  Current loss: %.4f\n", current_loss);

    if (fabsf(prev_loss - current_loss) < 0.001f) {
      printf("  Convergence reached\n");
      break;
    }
    prev_loss = current_loss;

    update_token_scores(trainer, trainer->texts);
    printf("  Updated token scores\n");

    if (trainer->vocab->count > trainer->vocab_size) {
      prune_vocab_step(trainer, trainer->texts, 0.8f);
      printf("  Pruned vocabulary to %d tokens\n", trainer->vocab->count);
    }
  }

  FastHashMap *char_tokens = hashmap_create(256);
  FastHashMap *other_tokens = hashmap_create(trainer->vocab->size);

  for (int i = 0; i < trainer->vocab->size; i++) {
    HashEntry *entry = trainer->vocab->buckets[i];
    while (entry) {
      if (strlen(entry->key) == 1) { hashmap_put(char_tokens, entry->key, entry->value); }
      else { hashmap_put(other_tokens, entry->key, entry->value); }
      entry = entry->next;
    }
  }

  TokenScore *sorted_tokens = (TokenScore*)malloc(other_tokens->count * sizeof(TokenScore));
  int token_count = 0;

  for (int i = 0; i < other_tokens->size; i++) {
    HashEntry *entry = other_tokens->buckets[i];
    while (entry) {
      sorted_tokens[token_count].token = strdup(entry->key);
      sorted_tokens[token_count].score = entry->value;
      token_count++;
      entry = entry->next;
    }
  }

  qsort(sorted_tokens, token_count, sizeof(TokenScore), [](const void *a, const void *b) {
    float diff = ((TokenScore*)b)->score - ((TokenScore*)a)->score;
    return (diff > 0) ? 1 : (diff < 0) ? -1 : 0;
  });

  int final_other_count = (trainer->vocab_size - char_tokens->count) < token_count ? (trainer->vocab_size - char_tokens->count) : token_count;
  for (int i = 0; i < final_other_count; i++) { hashmap_put(trainer->final_vocab, sorted_tokens[i].token, sorted_tokens[i].score); }
  for (int i = 0; i < char_tokens->size; i++) {
    HashEntry *entry = char_tokens->buckets[i];
    while (entry) {
      hashmap_put(trainer->final_vocab, entry->key, entry->value);
      entry = entry->next;
    }
  }

  for (int i = 0; i < token_count; i++) { free(sorted_tokens[i].token); }
  free(sorted_tokens);
  printf("Training completed. Final vocabulary size: %d\n", trainer->final_vocab->count);

  text_array_destroy(processed_texts);
  hashmap_destroy(char_tokens);
  hashmap_destroy(other_tokens);  
  return trainer->final_vocab;
}

FastHashMap* get_final_vocab(UnigramTrainer *trainer) {
  FastHashMap *copy = hashmap_create(trainer->final_vocab->size);
  for (int i = 0; i < trainer->final_vocab->size; i++) {
    HashEntry *entry = trainer->final_vocab->buckets[i];
    while (entry) {
      hashmap_put(copy, entry->key, entry->value);
      entry = entry->next;
    }
  }
  return copy;
}

void save_vocab(UnigramTrainer *trainer, const char *filepath) {
  FILE *f = fopen(filepath, "w");
  if (!f) return;
  
  TokenScore *sorted = (TokenScore*)malloc(trainer->final_vocab->count * sizeof(TokenScore));
  int count = 0;

  for (int i = 0; i < trainer->final_vocab->size; i++) {
    HashEntry *entry = trainer->final_vocab->buckets[i];
    while (entry) {
      sorted[count].token = entry->key;
      sorted[count].score = entry->value;
      count++;
      entry = entry->next;
    }
  }

  qsort(sorted, count, sizeof(TokenScore), [](const void *a, const void *b) {
    float diff = ((TokenScore*)b)->score - ((TokenScore*)a)->score;
    return (diff > 0) ? 1 : (diff < 0) ? -1 : 0;
  });

  for (int i = 0; i < count; i++) { fprintf(f, "%s\t%f\n", sorted[i].token, sorted[i].score); }
  free(sorted);
  fclose(f);
}

FastHashMap* load_vocab(UnigramTrainer *trainer, const char *filepath) {
  FILE *f = fopen(filepath, "r");
  if (!f) return NULL;

  char line[1024];
  while (fgets(line, sizeof(line), f)) {
    char *tab = strchr(line, '\t');
    if (tab) {
      *tab = '\0';
      float score = atof(tab + 1);
      hashmap_put(trainer->final_vocab, line, score);
    }
  }

  fclose(f);
  return trainer->final_vocab;
}

int compare_char_counts(const void *a, const void *b) { return ((CharCount*)b)->count - ((CharCount*)a)->count; }
int compare_freq_tokens(const void *a, const void *b) {
  typedef struct { float freq; char *token; } FreqToken;
  float diff = ((FreqToken*)b)->freq - ((FreqToken*)a)->freq;
  return (diff > 0) ? 1 : (diff < 0) ? -1 : 0;
}