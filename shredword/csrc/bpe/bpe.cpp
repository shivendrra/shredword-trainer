#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hash.h"
#include "heap.h"
#include "histogram.h"
#include "bpe.h"

typedef struct FreqChange {
  uint64_t pair_hash;
  int64_t delta;
  struct FreqChange* next;
} FreqChange;

#define FREQ_CHANGE_BUCKETS 1024

typedef struct FreqChangeMap {
  FreqChange* buckets[FREQ_CHANGE_BUCKETS];
} FreqChangeMap;

static void freq_change_init(FreqChangeMap* map) {
  for (int i = 0; i < FREQ_CHANGE_BUCKETS; i++) map->buckets[i] = NULL;
}

static void freq_change_add(FreqChangeMap* map, uint64_t pair_hash, int64_t delta) {
  size_t bucket = pair_hash % FREQ_CHANGE_BUCKETS;
  for (FreqChange* fc = map->buckets[bucket]; fc; fc = fc->next) {
    if (fc->pair_hash == pair_hash) {
      fc->delta += delta;
      return;
    }
  }
  FreqChange* new_fc = (FreqChange*)malloc(sizeof(FreqChange));
  new_fc->pair_hash = pair_hash;
  new_fc->delta = delta;
  new_fc->next = map->buckets[bucket];
  map->buckets[bucket] = new_fc;
}

static void freq_change_free(FreqChangeMap* map) {
  for (int i = 0; i < FREQ_CHANGE_BUCKETS; i++) {
    FreqChange* fc = map->buckets[i];
    while (fc) {
      FreqChange* next = fc->next;
      free(fc);
      fc = next;
    }
    map->buckets[i] = NULL;
  }
}

uint64_t recompute_freq(PairKey key, Info* info, Trainer* trainer) {
  if (key.first == trainer->config.unk_id || key.second == trainer->config.unk_id) return 0;
  uint64_t freq = 0;
  size_t vocab_size = trainer->corpus.vocab_size;
  for (size_t wi = 0; wi < vocab_size; ++wi) {
    Symbol* s = trainer->corpus.words[wi];
    uint64_t count = trainer->corpus.word_counts[wi];
    while (s && s->next) {
      if (!s->deleted && !s->next->deleted && s->id == key.first && s->next->id == key.second) freq += count;
      s = s->next;
    }
  }
  return freq;
}

Trainer* create_trainer(const BPEConfig* config) {
  if (config == NULL) {
    fprintf(stderr, "[ERROR]\t Config pointer is NULL\n");
    exit(EXIT_FAILURE);
  }
  Trainer* trainer = (Trainer*)malloc(sizeof(Trainer));
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t Couldn't allocate Memory to Trainer\n");
    exit(EXIT_FAILURE);
  }
  trainer->config = *config;
  if (trainer->config.character_coverage <= 0.0 || trainer->config.character_coverage >= 1.0) trainer->config.character_coverage = 0.995;
  if (trainer->config.min_pair_freq == 0) trainer->config.min_pair_freq = MIN_PAIR_FREQ;
  trainer->num_merges = 0;
  trainer->merge_ops = (PairKey*)malloc(sizeof(PairKey) * trainer->config.target_vocab_size);
  heap_init(&trainer->heap, MIN_HEAP_SIZE);
  printf("[INFO]\t BPE trainer initialized. Heap initialized successfully.\n");
  return trainer;
}

void bpe_trainer_destroy(Trainer* trainer) {
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t No Trainer pointer found to destroy!\n");
    exit(EXIT_FAILURE);
  }
  free(trainer->corpus.words);
  free(trainer->corpus.word_counts);
  heap_free(&trainer->heap);
  free(trainer);
}

void bpe_init(Trainer* trainer) {
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t NULL trainer pointer\n");
    exit(EXIT_FAILURE);
  }
  bimap_free(&trainer->bigram_map);
  bimap_init(&trainer->bigram_map, MIN_HEAP_SIZE);
  heap_free(&trainer->heap);
  heap_init(&trainer->heap, MIN_HEAP_SIZE);
  bpe_count_bigrams(trainer);
}

int bpe_load_corpus(Trainer* trainer, const char* input_path) {
  if (!trainer || !input_path) {
    fprintf(stderr, "[ERROR]\t NULL trainer or input path pointers\n");
    return -1;
  }
  StrMap freq_map;
  strmap_init(&freq_map, INITIAL_STR_BUFFER);
  FILE* fp = fopen(input_path, "r");
  if (!fp) {
    fprintf(stderr, "[ERROR]\t Couldn't open file: %s\n", input_path);
    strmap_free(&freq_map);
    return -1;
  }
  char* line = (char*)malloc(INITIAL_STR_BUFFER);
  if (!line) {
    fprintf(stderr, "[ERROR]\t Memory allocation failed\n");
    fclose(fp);
    strmap_free(&freq_map);
    return -1;
  }
  size_t line_cap = INITIAL_STR_BUFFER;
  while (fgets(line, line_cap, fp)) {
    size_t len = strlen(line);
    while (len == line_cap - 1 && line[len-1] != '\n') {
      line_cap *= 2;
      char* new_line = (char*)realloc(line, line_cap);
      if (!new_line) {
        fprintf(stderr, "[ERROR]\t Memory reallocation failed\n");
        free(line);
        fclose(fp);
        strmap_free(&freq_map);
        return -1;
      }
      line = new_line;
      if (!fgets(line + len, line_cap - len, fp)) break;
      len = strlen(line);
    }
    if (len > 0 && line[len-1] == '\n') { line[len-1] = '\0'; }
    char* tok = strtok(line, "\t\r\n ");
    while (tok) {
      strmap_increment(&freq_map, tok);
      tok = strtok(NULL, "\t\r\n ");
    }
  }
  free(line);
  fclose(fp);
  StrMap char_map;
  strmap_init(&char_map, INITIAL_VOCAB_SIZE);
  strmap_iter(&freq_map, char_hist, &char_map);
  CharCount* counts = (CharCount*)malloc(INITIAL_VOCAB_SIZE * sizeof(CharCount));
  if (!counts) {
    fprintf(stderr, "[ERROR]\t Failed allocation of character counts\n");
    exit(EXIT_FAILURE);
  }
  CharCountCtx ctx = {counts, 0};
  strmap_iter(&char_map, collect_char, &ctx);
  size_t c = ctx.idx;
  qsort(counts, c, sizeof(CharCount), charcount_cmp);
  printf("[DEBUG]\t Character histogram built with %zu unique characters.\n", c);
  size_t keep = (size_t)(c * trainer->config.character_coverage);
  bool keep_char[INITIAL_VOCAB_SIZE] = {0};
  for (size_t i = 0; i < keep; i++) keep_char[(unsigned char)counts[i].c] = true;
  free(counts);
  strmap_free(&char_map);
  size_t N = 0;
  strmap_iter(&freq_map, [](const char* k, uint64_t v, void* u){(*(size_t*)u)++;}, &N);
  trainer->corpus.vocab_size = N;
  trainer->corpus.words = (Symbol**)malloc(N * sizeof(Symbol*));
  trainer->corpus.word_counts = (uint64_t*)malloc(N * sizeof(uint64_t));
  size_t idx = 0;
  BuildCtx c_btx = { trainer, &idx, keep_char };
  strmap_iter(&freq_map, build_symbol_cb, &c_btx);
  strmap_free(&freq_map);
  bimap_init(&trainer->bigram_map, MIN_HEAP_SIZE);
  return 0;
}

void bpe_count_bigrams(Trainer* trainer) {
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t NULL trainer pointer\n");
    exit(EXIT_FAILURE);
  }
  size_t v = trainer->corpus.vocab_size;
  uint64_t min_freq = trainer->config.min_pair_freq;
  uint64_t total_pairs = 0;
  size_t unique_pairs = 0;
  printf("[INFO]\t Counting bigrams from %zu words...\n", v);
  for (size_t wi = 0; wi < v; wi++) {
    Symbol* s = trainer->corpus.words[wi];
    uint64_t wcount = trainer->corpus.word_counts[wi];
    while (s && s->next) {
      if (s->deleted || s->next->deleted || s->id == trainer->config.unk_id || s->next->id == trainer->config.unk_id) {
        s = s->next;
        continue;
      }
      PairKey key = { s->id, s->next->id };
      Info* info = bimap_get(&trainer->bigram_map, key);
      if (info->freq == 0) {
        unique_pairs++;
        info->version = 0;
      }
      info->freq += wcount;
      total_pairs += wcount;  
      s = s->next;
    }
    if (wi % 10000 == 0 && wi > 0) {
      printf("[DEBUG]\t Processed %zu/%zu words, found %zu unique pairs\n", wi, v, unique_pairs);
    }
  }
  size_t heap_entries = 0;
  for (size_t i = 0; i < trainer->bigram_map.nbuckets; i++) {
    for (BIEntry* e = trainer->bigram_map.buckets[i]; e; e = e->next) {
      if (e->info.freq >= min_freq) {
        heap_push(&trainer->heap, e->key, e->info.freq, e->info.version);
        heap_entries++;
      }
    }
  }
  printf("[INFO]\t Counted %llu total bigram occurrences, %zu unique pairs\n", (unsigned long long)total_pairs, unique_pairs);
  printf("[INFO]\t Added %zu pairs to heap (freq >= %llu)\n", heap_entries, (unsigned long long)min_freq);
}

int bpe_merge_batch(Trainer* trainer, int batch_size) {
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t Trainer pointer is NULL!\n");
    return -1;
  }
  if (heap_empty(&trainer->heap)) {
    printf("[INFO]\t Heap is empty, no more merges possible\n");
    return 0;
  }
  int merges_done = 0, stale_entries = 0;
  uint64_t min_freq = trainer->config.min_pair_freq;
  while (merges_done < batch_size && !heap_empty(&trainer->heap)) {
    HeapEntry top = heap_pop(&trainer->heap);
    PairKey key = top.key;
    Info* info = bimap_get(&trainer->bigram_map, key);
    if (top.version != info->version) {
      stale_entries++;
      continue;
    }
    uint64_t actual_freq = recompute_freq(key, info, trainer);
    if (actual_freq != info->freq) {
      info->freq = actual_freq;
      info->version++;
      if (actual_freq >= min_freq) heap_push(&trainer->heap, key, actual_freq, info->version);
      continue;
    }
    if (actual_freq < min_freq) continue;
    int32_t new_id = INITIAL_VOCAB_SIZE + trainer->num_merges;
    printf("[MERGE]\t Merging (%d,%d) freq=%llu -> new_id=%d (merge %zu)\n", key.first, key.second, (unsigned long long)actual_freq, new_id, trainer->num_merges + 1);
    if (trainer->num_merges < trainer->config.target_vocab_size) trainer->merge_ops[trainer->num_merges] = key;
    FreqChangeMap freq_changes;
    freq_change_init(&freq_changes);
    uint64_t total_merge_count = 0;
    for (size_t wi = 0; wi < trainer->corpus.vocab_size; ++wi) {
      Symbol* s = trainer->corpus.words[wi];
      uint64_t word_count = trainer->corpus.word_counts[wi];
      while (s && s->next) {
        if (s->deleted || s->next->deleted || s->id != key.first || s->next->id != key.second) {
          s = s->next;
          continue;
        }
        total_merge_count += word_count;
        if (s->prev && !s->prev->deleted) {
          PairKey old_left = {s->prev->id, key.first};
          PairKey new_left = {s->prev->id, new_id};
          uint64_t old_hash = ((uint64_t)old_left.first << 32) | (uint64_t)old_left.second;
          uint64_t new_hash = ((uint64_t)new_left.first << 32) | (uint64_t)new_left.second;
          freq_change_add(&freq_changes, old_hash, -(int64_t)word_count);
          freq_change_add(&freq_changes, new_hash, (int64_t)word_count);
        }
        Symbol* b = s->next;
        if (b->next && !b->next->deleted) {
          PairKey old_right = {key.second, b->next->id};
          PairKey new_right = {new_id, b->next->id};
          uint64_t old_hash = ((uint64_t)old_right.first << 32) | (uint64_t)old_right.second;
          uint64_t new_hash = ((uint64_t)new_right.first << 32) | (uint64_t)new_right.second;
          freq_change_add(&freq_changes, old_hash, -(int64_t)word_count);
          freq_change_add(&freq_changes, new_hash, (int64_t)word_count);
        }
        s->id = new_id;
        s->next = b->next;
        if (b->next) { b->next->prev = s; }
        b->deleted = true;
      }
    }
    for (int i = 0; i < FREQ_CHANGE_BUCKETS; i++) {
      for (FreqChange* fc = freq_changes.buckets[i]; fc; fc = fc->next) {
        uint64_t pair_hash = fc->pair_hash;
        int64_t delta = fc->delta;
        PairKey pk = {(int32_t)(pair_hash >> 32), (int32_t)(pair_hash & 0xFFFFFFFF)};
        if (pk.first == key.first && pk.second == key.second) continue;
        Info* pair_info = bimap_get(&trainer->bigram_map, pk);
        if (delta < 0) {
          uint64_t abs_delta = (uint64_t)(-delta);
          if (pair_info->freq >= abs_delta) { pair_info->freq -= abs_delta; } else { pair_info->freq = 0; }
        } else { pair_info->freq += (uint64_t)delta; }
        if (pair_info->freq >= min_freq) {
          pair_info->version++;
          heap_push(&trainer->heap, pk, pair_info->freq, pair_info->version);
        }
      }
    }
    freq_change_free(&freq_changes);
    info->freq = 0;
    info->version++;
    trainer->num_merges++;
    merges_done++;
    printf("[DEBUG]\t Merged %llu occurrences in corpus\n", (unsigned long long)total_merge_count);
  }
  if (stale_entries > 0) { printf("[DEBUG]\t Skipped %d stale heap entries\n", stale_entries); }
  return merges_done;
}

void free_deleted_symbols(Trainer* trainer) {
  if (!trainer) return;
  for (size_t wi = 0; wi < trainer->corpus.vocab_size; ++wi) {
    Symbol* s = trainer->corpus.words[wi];
    Symbol* prev = NULL;
    while (s) {
      if (s->deleted) {
        Symbol* to_free = s;
        if (prev) { prev->next = s->next; } else { trainer->corpus.words[wi] = s->next; }
        if (s->next) { s->next->prev = prev; }
        s = s->next;
        free(to_free);
      } else {
        prev = s;
        s = s->next;
      }
    }
  }
}

int bpe_train(Trainer* trainer) {
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t Trainer pointer is NULL!\n");
    return -1;
  }
  printf("[INFO]\t Starting BPE training (target vocab size: %zu)\n", trainer->config.target_vocab_size);  
  bpe_init(trainer);
  int total_merges = 0;
  int target_merges = (int)trainer->config.target_vocab_size - INITIAL_VOCAB_SIZE;
  printf("[INFO]\t Need to perform %d merges to reach target vocab size\n", target_merges);
  while (total_merges < target_merges) {
    if (heap_empty(&trainer->heap)) {
      printf("[INFO]\t Heap exhausted, stopping training\n");
      break;
    }
    HeapEntry top = trainer->heap.data[0];
    uint64_t top_freq = top.freq;
    int batch_size;
    if (top_freq > 50000) batch_size = 10;
    else if (top_freq > 20000) batch_size = 5;
    else if (top_freq > 10000) batch_size = 3;
    else if (top_freq > 5000) batch_size = 2;
    else batch_size = 1;
    batch_size = (batch_size > target_merges - total_merges) ? target_merges - total_merges : batch_size;
    printf("[INFO]\t Processing batch of %d merges (completed: %d/%d, heap size: %zu, top freq: %llu)\n", batch_size, total_merges, target_merges, trainer->heap.size, (unsigned long long)top_freq);
    int merged = bpe_merge_batch(trainer, batch_size);
    if (merged <= 0) {
      printf("[WARNING]\t No merges performed, stopping\n");
      break;
    }
    total_merges += merged;
    if (total_merges % 100 == 0) {
      printf("[DEBUG]\t Cleaning up deleted symbols after %d merges\n", total_merges);
      free_deleted_symbols(trainer);
    }
    if (total_merges % 50 == 0 || merged < batch_size) { printf("[PROGRESS]\t Completed %d/%d merges (%.1f%%)\n", total_merges, target_merges, 100.0 * total_merges / target_merges); }
  }
  printf("[INFO]\t Final cleanup of deleted symbols\n");
  free_deleted_symbols(trainer);
  printf("[INFO]\t Training completed. Performed %d merges\n", total_merges);
  return total_merges;
}

void bpe_save(const Trainer* trainer, const char* model_path, const char* vocab_path) {
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t Trainer pointer is NULL!\n");
    exit(EXIT_FAILURE);
  }
  size_t M = trainer->num_merges, T = INITIAL_VOCAB_SIZE + M;
  char** toks = (char**)calloc(T, sizeof(char*));
  for (size_t i = 0; i < INITIAL_VOCAB_SIZE; ++i) {
    toks[i] = (char*)malloc(2);
    toks[i][0] = (char)i; 
    toks[i][1] = '\0';
  }
  for (size_t m = 0; m < M; ++m) {
    PairKey ops = trainer->merge_ops[m];
    size_t id = INITIAL_VOCAB_SIZE + m;
    char *A = toks[ops.first], *B = toks[ops.second];
    size_t aL = strlen(A), bL = strlen(B);
    toks[id] = (char*)malloc(aL + bL + 1);
    memcpy(toks[id], A, aL);
    memcpy(toks[id] + aL, B, bL + 1);
  }
  uint64_t* freq = (uint64_t*)calloc(T, sizeof(uint64_t));
  for (size_t w = 0; w < trainer->corpus.vocab_size; ++w) {
    uint64_t wc = trainer->corpus.word_counts[w];
    for (Symbol* s = trainer->corpus.words[w]; s; s = s->next) {
      if (!s->deleted) freq[s->id] += wc;
    }
  }
  FILE* vf = fopen(vocab_path, "w");
  for (size_t i = 0; i < T; ++i) fprintf(vf, "%s %llu\n", toks[i], (unsigned long long)freq[i]);
  fclose(vf);
  FILE* mf = fopen(model_path, "wb");
  for (size_t m = 0; m < M; ++m) {
    PairKey op = trainer->merge_ops[m];
    int32_t a = (int32_t)op.first, b = (int32_t)op.second, new_id = (int32_t)(INITIAL_VOCAB_SIZE + m);
    fwrite(&a, sizeof(int32_t), 1, mf);
    fwrite(&b, sizeof(int32_t), 1, mf);
    fwrite(&new_id, sizeof(int32_t), 1, mf);
  }
  fclose(mf);
  for (size_t i = 0; i < T; ++i) free(toks[i]);
  free(toks);
  free(freq);
  printf("[INFO]\tSaved %zu-token vocab to %s and %zu merges to %s\n", T, vocab_path, M, model_path);
}