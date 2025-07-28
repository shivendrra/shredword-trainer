#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <ctype.h>
#include <time.h>
#include "unigram.h"
#include "../inc/hash.h"

UnigramTrainer* trainer_create(int vocab_size, float character_coverage, int max_len, int seed_size) {
  UnigramTrainer *trainer = (UnigramTrainer*)malloc(sizeof(UnigramTrainer));
  if (!trainer) return NULL;

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
  trainer->texts = NULL;
  trainer->total_chars = 0;
  
  if (!trainer->vocab_heap || !trainer->token_freqs || !trainer->vocab || !trainer->final_vocab || !trainer->loss_cache || !trainer->subword_trie || !trainer->extractor || !trainer->decoder) {
    trainer_destroy(trainer);
    return NULL;
  }
  return trainer;
}

void trainer_destroy(UnigramTrainer *trainer) {
  if (!trainer) return;
  if (trainer->texts) text_array_destroy(trainer->texts);
  if (trainer->token_freqs) hashmap_destroy(trainer->token_freqs);
  if (trainer->vocab) hashmap_destroy(trainer->vocab);
  if (trainer->final_vocab) hashmap_destroy(trainer->final_vocab);
  if (trainer->subword_trie) trie_destroy(trainer->subword_trie);
  if (trainer->extractor) extractor_destroy(trainer->extractor);
  if (trainer->decoder) viterbi_destroy(trainer->decoder);
  if (trainer->loss_cache) cache_destroy(trainer->loss_cache);
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
int compare_removal_candidates(const void *a, const void *b) {
  float diff = ((RemovalCandidate*)a)->loss_increase - ((RemovalCandidate*)b)->loss_increase;
  return (diff > 0) ? 1 : (diff < 0) ? -1 : 0;
}

int compare_token_scores_desc(const void *a, const void *b) {
  float diff = ((TokenScore*)b)->score - ((TokenScore*)a)->score;
  return (diff > 0) ? 1 : (diff < 0) ? -1 : 0;
}

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
  FastHashMap *char_freq = hashmap_create(256);    
  FastHashMap *subword_freq = hashmap_create(50000);
  
  // Add special tokens first
  hashmap_put(trainer->vocab, "▁", 10000.0f);
  hashmap_put(trainer->token_freqs, "▁", 10000.0f);
  if (trainer->vocab_heap) heap_push(trainer->vocab_heap, "▁", 10000);
  if (trainer->subword_trie) trie_insert(trainer->subword_trie, "▁", 10000);
  
  // Collect character frequencies  
  for (int i = 0; i < texts->count && i < SEED_TEXTS; i++) {
    char *text = texts->texts[i];
    for (int j = 0; text[j]; j++) {
      if (text[j] >= 32 && text[j] <= 126) {
        char key[2] = {text[j], '\0'};
        float count = hashmap_get(char_freq, key);
        hashmap_put(char_freq, key, count == -FLT_MAX ? 1.0f : count + 1.0f);
      }
    }
  }
  
  // Add all characters to vocab
  for (int i = 0; i < char_freq->size; i++) {
    HashEntry *entry = char_freq->buckets[i];
    while (entry) {
      if (entry->value > 0) {
        float score = logf(entry->value);
        hashmap_put(trainer->vocab, entry->key, score);
        hashmap_put(trainer->token_freqs, entry->key, entry->value);      
        if (trainer->vocab_heap) heap_push(trainer->vocab_heap, entry->key, (int)entry->value);
        if (trainer->subword_trie) trie_insert(trainer->subword_trie, entry->key, (int)entry->value);
      }
      entry = entry->next;
    }
  }
  
  // Generate meaningful subwords from word boundaries
  for (int i = 0; i < texts->count && i < SEED_TEXTS && trainer->vocab->count < trainer->seed_size; i++) {
    char *text = texts->texts[i];
    int len = strlen(text);
    
    // Split by word boundaries (▁ character)
    int word_start = 0;
    for (int j = 0; j <= len; j++) {
      if (j == len || (j > 0 && text[j-3] == '\xE2' && text[j-2] == '\x96' && text[j-1] == '\x81')) {
        if (j > word_start + 3) { // Skip the ▁ marker itself
          int word_len = j - word_start;
          if (word_len > 16) word_len = 16;
          
          // Extract prefixes and suffixes
          for (int prefix_len = 2; prefix_len <= word_len && prefix_len <= trainer->max_len; prefix_len++) {
            char prefix[17];
            strncpy(prefix, text + word_start, prefix_len);
            prefix[prefix_len] = '\0';
            
            // Only add if it starts with ▁ or is alphabetic
            if ((prefix[0] == '\xE2' && prefix[1] == '\x96' && prefix[2] == '\x81') || 
                isalnum(prefix[0])) {
              float count = hashmap_get(subword_freq, prefix);
              hashmap_put(subword_freq, prefix, count == -FLT_MAX ? 1.0f : count + 1.0f);
            }
          }
          
          // Extract suffixes (without ▁)
          if (word_len > 3) { // Skip the ▁ marker
            for (int suffix_len = 2; suffix_len <= word_len - 3 && suffix_len <= trainer->max_len; suffix_len++) {
              char suffix[17];
              int start_pos = j - suffix_len;
              if (start_pos >= word_start + 3) { // Ensure we don't include ▁
                strncpy(suffix, text + start_pos, suffix_len);
                suffix[suffix_len] = '\0';
                
                if (isalnum(suffix[0])) {
                  float count = hashmap_get(subword_freq, suffix);
                  hashmap_put(subword_freq, suffix, count == -FLT_MAX ? 1.0f : count + 1.0f);
                }
              }
            }
          }
        }
        word_start = j;
      }
    }
  }
  
  // Add frequent subwords to vocabulary
  typedef struct {
    char *token;
    float freq;
  } FreqSubword;
  
  FreqSubword *subwords = (FreqSubword*)malloc(subword_freq->count * sizeof(FreqSubword));
  int subword_count = 0;
  
  for (int i = 0; i < subword_freq->size; i++) {
    HashEntry *entry = subword_freq->buckets[i];
    while (entry) {
      if (entry->value >= 5.0f) { // Only frequent subwords
        subwords[subword_count].token = strdup(entry->key);
        subwords[subword_count].freq = entry->value;
        subword_count++;
      }
      entry = entry->next;
    }
  }
  
  // Sort by frequency
  qsort(subwords, subword_count, sizeof(FreqSubword), [](const void *a, const void *b) {
    float diff = ((FreqSubword*)b)->freq - ((FreqSubword*)a)->freq;
    return (diff > 0) ? 1 : (diff < 0) ? -1 : 0;
  });
  
  // Add top subwords to vocabulary
  int added = 0;
  for (int i = 0; i < subword_count && trainer->vocab->count < trainer->seed_size && added < 10000; i++) {
    if (!hashmap_contains(trainer->vocab, subwords[i].token)) {
      float score = logf(subwords[i].freq);
      hashmap_put(trainer->vocab, subwords[i].token, score);
      hashmap_put(trainer->token_freqs, subwords[i].token, subwords[i].freq);      
      if (trainer->vocab_heap) heap_push(trainer->vocab_heap, subwords[i].token, (int)subwords[i].freq);
      if (trainer->subword_trie) trie_insert(trainer->subword_trie, subwords[i].token, (int)subwords[i].freq);
      added++;
    }
    free(subwords[i].token);
  }
  
  // Free remaining
  for (int i = added; i < subword_count; i++) {
    free(subwords[i].token);
  }
  free(subwords);
  
  hashmap_destroy(char_freq);
  hashmap_destroy(subword_freq);
  printf("Initialized vocabulary with %d tokens\n", trainer->vocab->count);
}

float compute_loss(UnigramTrainer *trainer, TextArray *texts) {
  if (!trainer->decoder || !trainer->vocab) return 0.0f;
  
  float total_loss = 0.0f;
  int total_tokens = 0;
  
  // Process only first 100 texts for speed
  int process_count = texts->count < 100 ? texts->count : 100;
  
  for (int i = 0; i < process_count; i++) {
    if (!texts->texts[i] || strlen(texts->texts[i]) == 0) continue;
    
    // Simple hash-based caching
    int cache_key = djb2_hash(texts->texts[i]) % 100000;
    int cached_loss = cache_get(trainer->loss_cache, cache_key);

    if (cached_loss != -1) {
      total_loss += cached_loss;
      total_tokens += 10; // Assume average of 10 tokens per text
      continue;
    }

    ViterbiResult *result = viterbi_decode(trainer->decoder, texts->texts[i], trainer->vocab);
    if (!result) continue;
    
    float text_loss = 0.0f;
    for (size_t j = 0; j < result->count; j++) {
      float score = hashmap_get(trainer->vocab, result->tokens[j]);
      if (score != -FLT_MAX) {
        text_loss -= score;
      } else {
        text_loss += 10.0f; // Penalty for unknown tokens
      }
    }

    cache_put(trainer->loss_cache, cache_key, (int)(text_loss * 100)); // Scale for integer storage
    total_loss += text_loss;
    total_tokens += result->count;
    viterbi_result_destroy(result);
  }
  
  return total_tokens > 0 ? total_loss / total_tokens : 0.0f;
}

// Simplified token loss computation
float compute_token_loss(UnigramTrainer *trainer, const char *token, TextArray *texts) {
  if (!trainer->decoder || !token) return 0.0f;
  
  // Simple approximation: tokens that appear more frequently have lower loss when removed
  float token_freq = hashmap_get(trainer->token_freqs, token);
  if (token_freq == -FLT_MAX) token_freq = 1.0f;
  
  // Return inverse frequency as loss increase estimate
  return 1000.0f / (token_freq + 1.0f);
}

// Simplified update_token_scores to avoid hanging
void update_token_scores(UnigramTrainer *trainer, TextArray *texts) {
  if (!trainer->vocab || !texts) return;
  
  FastHashMap *token_context_freq = hashmap_create(trainer->vocab->size);
  
  // Process only first 200 texts for speed
  int process_count = texts->count < 200 ? texts->count : 200;
  
  for (int i = 0; i < process_count; i++) {
    if (!texts->texts[i]) continue;
    
    ViterbiResult *result = viterbi_decode(trainer->decoder, texts->texts[i], trainer->vocab);
    if (!result) continue;

    for (size_t j = 0; j < result->count; j++) {
      if (hashmap_contains(trainer->vocab, result->tokens[j])) {
        float count = hashmap_get(token_context_freq, result->tokens[j]);
        hashmap_put(token_context_freq, result->tokens[j], 
                   count == -FLT_MAX ? 1.0f : count + 1.0f);
      }
    }
    viterbi_result_destroy(result);
  }

  // Calculate total frequency
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

  // Update scores based on frequency
  for (int i = 0; i < trainer->vocab->size; i++) {
    HashEntry *entry = trainer->vocab->buckets[i];
    while (entry) {
      float freq = hashmap_get(token_context_freq, entry->key);
      if (freq == -FLT_MAX) freq = 0.1f; // Small frequency for unseen tokens
      
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

void prune_vocab_step(UnigramTrainer *trainer, TextArray *texts, float reduction_ratio) {
  if (trainer->vocab->count <= trainer->vocab_size) return;  
  
  int target_size = (int)(trainer->vocab->count * reduction_ratio);
  if (target_size < trainer->vocab_size) target_size = trainer->vocab_size;
  int tokens_to_remove = trainer->vocab->count - target_size;

  if (tokens_to_remove <= 0) return;

  // Collect all tokens with their frequencies for removal
  typedef struct {
    char *token;
    float freq;
    int len;
  } TokenCandidate;
  
  TokenCandidate *candidates = (TokenCandidate*)malloc(trainer->vocab->count * sizeof(TokenCandidate));
  int candidate_count = 0;

  for (int i = 0; i < trainer->vocab->size; i++) {
    for (HashEntry *entry = trainer->vocab->buckets[i]; entry; entry = entry->next) {
      if (strlen(entry->key) > 1) { // Don't remove single characters
        candidates[candidate_count].token = strdup(entry->key);
        candidates[candidate_count].freq = hashmap_get(trainer->token_freqs, entry->key);
        candidates[candidate_count].len = strlen(entry->key);
        candidate_count++;
      }
    }
  }

  // Sort by frequency (ascending) and length (descending for ties)
  qsort(candidates, candidate_count, sizeof(TokenCandidate), [](const void *a, const void *b) {
    TokenCandidate *ta = (TokenCandidate*)a;
    TokenCandidate *tb = (TokenCandidate*)b;
    
    if (ta->freq != tb->freq) {
      return (ta->freq < tb->freq) ? -1 : 1;
    }
    return tb->len - ta->len; // Prefer removing longer tokens when frequency is same
  });

  // Remove the least frequent tokens
  int removed = 0;
  for (int i = 0; i < candidate_count && removed < tokens_to_remove; i++) {
    if (hashmap_contains(trainer->vocab, candidates[i].token)) {
      hashmap_remove(trainer->vocab, candidates[i].token);
      if (trainer->vocab_heap) heap_remove(trainer->vocab_heap, candidates[i].token);
      hashmap_remove(trainer->token_freqs, candidates[i].token);
      removed++;
    }
    free(candidates[i].token);
  }
  
  // Free remaining tokens
  for (int i = removed; i < candidate_count; i++) {
    free(candidates[i].token);
  }
  free(candidates);
}
// void update_token_scores(UnigramTrainer *trainer, TextArray *texts) {
//   FastHashMap *token_context_freq = hashmap_create(trainer->vocab->size);
//   int process_count = texts->count < SCORE_TEXTS ? texts->count : SCORE_TEXTS;
//   for (int i = 0; i < process_count; i++) {
//     ViterbiResult *result = viterbi_decode(trainer->decoder, texts->texts[i], trainer->vocab);

//     for (size_t j = 0; j < result->count; j++) {
//       if (hashmap_contains(trainer->vocab, result->tokens[j])) {
//         float count = hashmap_get(token_context_freq, result->tokens[j]);
//         hashmap_put(token_context_freq, result->tokens[j], count + 1);
//       }
//     }
//     viterbi_result_destroy(result);
//   }

//   float total_freq = 0.0f;
//   for (int i = 0; i < token_context_freq->size; i++) {
//     HashEntry *entry = token_context_freq->buckets[i];
//     while (entry) {
//       total_freq += entry->value;
//       entry = entry->next;
//     }
//   }

//   if (total_freq == 0.0f) {
//     hashmap_destroy(token_context_freq);
//     return;
//   }

//   for (int i = 0; i < trainer->vocab->size; i++) {
//     HashEntry *entry = trainer->vocab->buckets[i];
//     while (entry) {
//       float freq = hashmap_get(token_context_freq, entry->key);
//       if (freq == 0.0f) freq = 1.0f;
//       float new_score = logf(freq / total_freq) + logf(total_freq);
//       hashmap_put(trainer->vocab, entry->key, new_score);
//       if (hashmap_contains(trainer->token_freqs, entry->key)) {
//         heap_update_freq(trainer->vocab_heap, entry->key, (int)freq);
//         hashmap_put(trainer->token_freqs, entry->key, freq);
//       }
//       entry = entry->next;
//     }
//   }
//   hashmap_destroy(token_context_freq);
// }

FastHashMap* train_unigram(UnigramTrainer *trainer, char **input_texts, int text_count, int num_iterations) {
  printf("Preprocessing %d texts...\n", text_count);
  TextArray *processed_texts = preprocess_texts(trainer, input_texts, text_count);

  trainer->texts = text_array_create(MAX_TEXTS);
  int copy_count = processed_texts->count < MAX_TEXTS ? processed_texts->count : MAX_TEXTS;
  for (int i = 0; i < copy_count; i++) { 
    text_array_add(trainer->texts, processed_texts->texts[i]); 
  }

  printf("Initializing seed vocabulary...\n");
  initialize_seed_vocab(trainer, trainer->texts);
  printf("Initial vocabulary size: %d\n", trainer->vocab->count);

  float prev_loss = INFINITY;
  for (int iteration = 0; iteration < num_iterations; iteration++) {
    printf("Iteration %d/%d\n", iteration + 1, num_iterations);

    float current_loss = compute_loss(trainer, trainer->texts);
    printf("  Current loss: %.4f\n", current_loss);

    if (fabsf(prev_loss - current_loss) < 0.001f && iteration > 2) {
      printf("  Convergence reached\n");
      break;
    }
    prev_loss = current_loss;

    update_token_scores(trainer, trainer->texts);
    printf("  Updated token scores\n");

    // Always try to prune if vocab is larger than target
    if (trainer->vocab->count > trainer->vocab_size) {
      int before_prune = trainer->vocab->count;
      prune_vocab_step(trainer, trainer->texts, 0.9f);
      printf("  Pruned vocabulary from %d to %d tokens\n", before_prune, trainer->vocab->count);
    }
  }

  // Final pruning to exact target size
  while (trainer->vocab->count > trainer->vocab_size) {
    prune_vocab_step(trainer, trainer->texts, 0.95f);
  }

  // Copy final vocabulary
  for (int i = 0; i < trainer->vocab->size; i++) {
    HashEntry *entry = trainer->vocab->buckets[i];
    while (entry) {
      hashmap_put(trainer->final_vocab, entry->key, entry->value);
      entry = entry->next;
    }
  }

  printf("Training completed. Final vocabulary size: %d\n", trainer->final_vocab->count);

  text_array_destroy(processed_texts);
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