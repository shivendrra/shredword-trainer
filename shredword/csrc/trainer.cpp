/**
 * trainer.cpp
 * main CLI interface for training vocabs directly, by selecting b/w the bpe or unigram models
 * 
 * compile this file:
 *    - windows: g++ -o trainer.exe trainer.cpp bpe/bpe.cpp bpe/histogram.cpp bpe/hash.cpp bpe/heap.cpp unigram/unigram.cpp unigram/heap.cpp unigram/cache.cpp unigram/hashmap.cpp unigram/subword.cpp trie.cpp -I. -std=c++11
 *    - linux: g++ -o trainer.exe trainer.cpp bpe/bpe.cpp bpe/histogram.cpp bpe/hash.cpp bpe/heap.cpp unigram/unigram.cpp unigram/heap.cpp unigram/cache.cpp unigram/hashmap.cpp unigram/subword.cpp trie.cpp
 * 
 * run:
 *    - as bpe: trainer.exe input=corpus.txt model_type=bpe output_model=model.bin output_vocab=vocab.txt vocab_size=32000
 *    - as unigram: trainer.exe input=corpus.txt model_type=unigram output_model=model.bin output_vocab=vocab.txt vocab_size=32000 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bpe/bpe.h"
#include "bpe/heap.h"
#include "unigram/unigram.h"
#include "unigram/heap.h"

typedef struct CLIConfig {
  char *input_path, *output_model, *output_vocab, *model_type;
  int vocab_size, num_iterations, seed_size, max_piece_length;
  float character_coverage;
  uint64_t min_pair_freq;
  int32_t unk_id;
} CLIConfig;

void print_usage(const char* program_name) {
  printf("Usage: %s <args>\n\n", program_name);
  printf("Arguments (use: key=value format):\n");
  printf("  input=<path>              Input corpus file\n");
  printf("  model_type=<bpe|unigram>  Model type\n");
  printf("  output_model=<path>       Output model file\n");
  printf("  output_vocab=<path>       Output vocab file\n");
  printf("  vocab_size=<int>          Target vocab size (default: 32000)\n");
  printf("  character_coverage=<float> Coverage 0.0-1.0 (default: 0.9995)\n");
  printf("  min_pair_freq=<int>       Min pair freq BPE (default: 2000)\n");
  printf("  num_iterations=<int>      Iterations Unigram (default: 10)\n");
}

void init_config(CLIConfig* config) {
  config->input_path = config->output_model = config->output_vocab = config->model_type = NULL;
  config->vocab_size = 32000, config->num_iterations = 10, config->seed_size = 1000000;
  config->max_piece_length = 16, config->character_coverage = 0.9995f;
  config->min_pair_freq = 2000, config->unk_id = -1;
}

int parse_args(int argc, char** argv, CLIConfig* config) {
  for (int i = 1; i < argc; i++) {
    char* arg = argv[i];
    char* eq = strchr(arg, '=');
    if (!eq) continue;
    *eq = '\0';
    char *key = arg, *value = eq + 1;
    
    if (strcmp(key, "input") == 0) config->input_path = strdup(value);
    else if (strcmp(key, "model_type") == 0) config->model_type = strdup(value);
    else if (strcmp(key, "output_model") == 0) config->output_model = strdup(value);
    else if (strcmp(key, "output_vocab") == 0) config->output_vocab = strdup(value);
    else if (strcmp(key, "vocab_size") == 0) config->vocab_size = atoi(value);
    else if (strcmp(key, "character_coverage") == 0) config->character_coverage = atof(value);
    else if (strcmp(key, "min_pair_freq") == 0) config->min_pair_freq = (uint64_t)atoll(value);
    else if (strcmp(key, "num_iterations") == 0) config->num_iterations = atoi(value);
    else if (strcmp(key, "seed_size") == 0) config->seed_size = atoi(value);
    else if (strcmp(key, "max_piece_length") == 0) config->max_piece_length = atoi(value);
  }

  if (!config->input_path || !config->model_type || !config->output_model || !config->output_vocab) {
    fprintf(stderr, "[ERROR] Missing required arguments\n\n");
    print_usage(argv[0]);
    return -1;
  }

  if (strcmp(config->model_type, "bpe") != 0 && strcmp(config->model_type, "unigram") != 0) {
    fprintf(stderr, "[ERROR] Invalid model_type. Must be 'bpe' or 'unigram'\n");
    return -1;
  }

  return 1;
}

int train_bpe(const CLIConfig* config) {
  printf("\n========== BPE Training ==========\n");
  printf("[CONFIG] Vocab Size: %d\n", config->vocab_size);
  printf("[CONFIG] Character Coverage: %.4f\n", config->character_coverage);
  printf("[CONFIG] Min Pair Freq: %llu\n", (unsigned long long)config->min_pair_freq);

  BPEConfig bpe_config = {(size_t)config->vocab_size, config->unk_id, config->character_coverage, config->min_pair_freq};
  Trainer* trainer = create_trainer(&bpe_config);
  if (!trainer) { fprintf(stderr, "[ERROR] Failed to create BPE trainer\n"); return -1; }

  printf("\n[STEP 1] Loading corpus from: %s\n", config->input_path);
  if (bpe_load_corpus(trainer, config->input_path) != 0) {
    fprintf(stderr, "[ERROR] Failed to load corpus\n");
    bpe_trainer_destroy(trainer);
    return -1;
  }
  printf("[INFO] Corpus loaded successfully. Vocabulary: %zu words\n", trainer->corpus.vocab_size);

  printf("\n[STEP 2] Training BPE model...\n");
  int merges = bpe_train(trainer);
  if (merges < 0) {
    fprintf(stderr, "[ERROR] Training failed\n");
    bpe_trainer_destroy(trainer);
    return -1;
  }
  printf("[SUCCESS] Training completed with %d merges\n", merges);

  printf("\n[STEP 3] Saving model and vocabulary...\n");
  bpe_save(trainer, config->output_model, config->output_vocab);
  printf("[SUCCESS] Saved to:\n  Model: %s\n  Vocab: %s\n", config->output_model, config->output_vocab);

  bpe_trainer_destroy(trainer);
  printf("\n========== Training Complete ==========\n");
  return 0;
}

int train_unigram(const CLIConfig* config) {
  printf("\n========== Unigram Training ==========\n");
  printf("[CONFIG] Vocab Size: %d\n", config->vocab_size);
  printf("[CONFIG] Character Coverage: %.4f\n", config->character_coverage);
  printf("[CONFIG] Max Piece Length: %d\n", config->max_piece_length);
  printf("[CONFIG] Iterations: %d\n", config->num_iterations);

  UnigramTrainer* trainer = trainerCreate(config->vocab_size, config->character_coverage, config->max_piece_length, config->seed_size);
  if (!trainer) { fprintf(stderr, "[ERROR] Failed to create Unigram trainer\n"); return -1; }

  printf("\n[STEP 1] Loading corpus from: %s\n", config->input_path);
  FILE* fp = fopen(config->input_path, "r");
  if (!fp) {
    fprintf(stderr, "[ERROR] Cannot open input file: %s\n", config->input_path);
    trainerDestroy(trainer);
    return -1;
  }

  char line[10000];
  int text_count = 0;
  while (fgets(line, sizeof(line), fp) && text_count < MAX_TEXTS_FOR_TRAINING) {
    size_t len = strlen(line);
    if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';
    if (strlen(line) > 0 && addTextToTrainer(trainer, line)) text_count++;
  }
  fclose(fp);

  if (text_count == 0) {
    fprintf(stderr, "[ERROR] No texts loaded from corpus\n");
    trainerDestroy(trainer);
    return -1;
  }
  printf("[INFO] Loaded %d texts from corpus\n", text_count);

  printf("\n[STEP 2] Training Unigram model...\n");
  if (!trainUnigram(trainer, (const char**)trainer->texts, trainer->text_count, config->num_iterations)) {
    fprintf(stderr, "[ERROR] Training failed\n");
    trainerDestroy(trainer);
    return -1;
  }

  printf("\n[STEP 3] Saving vocabulary...\n");
  if (!saveVocab(trainer, config->output_vocab)) {
    fprintf(stderr, "[ERROR] Failed to save vocabulary\n");
    trainerDestroy(trainer);
    return -1;
  }
  printf("[SUCCESS] Saved vocabulary to: %s\n", config->output_vocab);

  char** tokens = NULL;
  double* scores = NULL;
  int count = 0;
  if (getVocab(trainer, &tokens, &scores, &count)) {
    FILE* model_file = fopen(config->output_model, "w");
    if (model_file) {
      fprintf(model_file, "vocab_size=%d\nmodel_type=unigram\n", count);
      fclose(model_file);
      printf("[SUCCESS] Saved model metadata to: %s\n", config->output_model);
    }
    for (int i = 0; i < count; i++) free(tokens[i]);
    free(tokens), free(scores);
  }

  trainerDestroy(trainer);
  printf("\n========== Training Complete ==========\n");
  return 0;
}

int main(int argc, char** argv) {
  printf("Tokenizer Trainer CLI v1.0\n==========================\n");
  if (argc < 2) { print_usage(argv[0]); return 0; }

  CLIConfig config;
  init_config(&config);

  if (parse_args(argc, argv, &config) <= 0) {
    if (config.input_path) free(config.input_path);
    if (config.model_type) free(config.model_type);
    if (config.output_model) free(config.output_model);
    if (config.output_vocab) free(config.output_vocab);
    return 1;
  }

  int result = 0;
  if (strcmp(config.model_type, "bpe") == 0) result = train_bpe(&config);
  else if (strcmp(config.model_type, "unigram") == 0) result = train_unigram(&config);

  if (config.input_path) free(config.input_path);
  if (config.model_type) free(config.model_type);
  if (config.output_model) free(config.output_model);
  if (config.output_vocab) free(config.output_vocab);

  return result;
}