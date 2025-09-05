// g++ -o run run.cpp unigram/unigram.cpp unigram/cache.cpp unigram/heap.cpp unigram/hashmap.cpp unigram/subword.cpp trie.cpp -lm
// run test.txt --vocab_size 1200 --coverage 0.9995 --iterations 10 --output base_12k.model

#include "unigram/unigram.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_LINE_LENGTH 8192
#define DEFAULT_NUM_ITERATIONS 20
#define MAX_TEXT_LINES 100000

void printUsage(const char* program_name) {
  printf("Usage: %s <input_file> [options]\n", program_name);
  printf("Options:\n");
  printf("  --vocab_size <size>        Vocabulary size (default: 32000)\n");
  printf("  --coverage <coverage>      Character coverage (default: 0.9995)\n");
  printf("  --max_len <length>         Max token length (default: 16)\n");
  printf("  --seed_size <size>         Seed vocabulary size (default: 1000000)\n");
  printf("  --iterations <num>         Training iterations (default: 20)\n");
  printf("  --output <file>            Output vocabulary file (default: vocab.txt)\n");
  printf("  --help                     Show this help message\n");
  printf("\nExample:\n");
  printf("  %s train.txt --vocab_size 16000 --output my_vocab.txt\n", program_name);
}

char** readTextFile(const char* filename, int* text_count) {
  FILE* file = fopen(filename, "r");
  if (!file) {
    printf("Error: Cannot open file '%s'\n", filename);
    return NULL;
  }
  char** texts = (char**)malloc(MAX_TEXT_LINES * sizeof(char*));
  if (!texts) {
    fclose(file);
    return NULL;
  }
  char line[MAX_LINE_LENGTH];
  int count = 0;
  while (fgets(line, sizeof(line), file) && count < MAX_TEXT_LINES) {
    size_t len = strlen(line);
    if (len > 0 && line[len - 1] == '\n') {
      line[len - 1] = '\0';
      len--;
    }
    if (len > 0) {
      texts[count] = (char*)malloc(len + 1);
      if (texts[count]) {
        strcpy(texts[count], line);
        count++;
      }
    }
  }
  fclose(file);
  *text_count = count;
  if (count == 0) {
    free(texts);
    return NULL;
  }
  return texts;
}

void freeTexts(char** texts, int text_count) {
  if (!texts) return;
  for (int i = 0; i < text_count; i++) { if (texts[i]) free(texts[i]); }
  free(texts);
}

int parseArgs(int argc, char* argv[], char** input_file, const char** output_file, int* vocab_size, float* coverage, int* max_len, int* seed_size, int* iterations) {
  if (argc < 2) return 0;

  *input_file = argv[1];
  *output_file = "vocab.txt";
  *vocab_size = DEFAULT_VOCAB_SIZE;
  *coverage = DEFAULT_CHARACTER_COVERAGE;
  *max_len = DEFAULT_MAX_SENTENCEPIECE_LENGTH;
  *seed_size = DEFAULT_SEED_SIZE;
  *iterations = DEFAULT_NUM_ITERATIONS;

  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--help") == 0) { return 0; }
    else if (strcmp(argv[i], "--vocab_size") == 0 && i + 1 < argc) { *vocab_size = atoi(argv[++i]); }
    else if (strcmp(argv[i], "--coverage") == 0 && i + 1 < argc) { *coverage = atof(argv[++i]); }
    else if (strcmp(argv[i], "--max_len") == 0 && i + 1 < argc) { *max_len = atoi(argv[++i]); }
    else if (strcmp(argv[i], "--seed_size") == 0 && i + 1 < argc) { *seed_size = atoi(argv[++i]); }
    else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) { *iterations = atoi(argv[++i]); }
    else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) { *output_file = argv[++i]; }
    else { printf("Error: Unknown option '%s'\n", argv[i]); return 0; }
  }
  return 1;
}

void printConfig(const char* input_file, const char* output_file, int vocab_size, float coverage, int max_len, int seed_size, int iterations) {
  printf("=== Unigram Tokenizer Training Configuration ===\n");
  printf("Input file: %s\n", input_file);
  printf("Output file: %s\n", output_file);
  printf("Vocabulary size: %d\n", vocab_size);
  printf("Character coverage: %.4f\n", coverage);
  printf("Max token length: %d\n", max_len);
  printf("Seed vocab size: %d\n", seed_size);
  printf("Training iterations: %d\n", iterations);
  printf("=================================================\n\n");
}

void printVocabStats(UnigramTrainer* trainer) {
  char** tokens;
  double* scores;
  int vocab_count;
  if (!getVocab(trainer, &tokens, &scores, &vocab_count)) return;

  int single_char = 0, short_tokens = 0, long_tokens = 0;
  double total_score = 0.0;
  for (int i = 0; i < vocab_count; i++) {
    int len = strlen(tokens[i]);
    total_score += scores[i];
    if (len == 1) single_char++;
    else if (len <= 4) short_tokens++;
    else long_tokens++;
  }

  printf("\n=== Vocabulary Statistics ===\n");
  printf("Total tokens: %d\n", vocab_count);
  printf("Single characters: %d (%.1f%%)\n", single_char, 100.0 * single_char / vocab_count);
  printf("Short tokens (2-4): %d (%.1f%%)\n", short_tokens, 100.0 * short_tokens / vocab_count);
  printf("Long tokens (5+): %d (%.1f%%)\n", long_tokens, 100.0 * long_tokens / vocab_count);
  printf("Average score: %.4f\n", total_score / vocab_count);
  printf("=============================\n");
}

int main(int argc, char* argv[]) {
  char* input_file;
  const char* output_file;
  int vocab_size, max_len, seed_size, iterations;
  float coverage;

  if (!parseArgs(argc, argv, &input_file, &output_file, &vocab_size, &coverage, &max_len, &seed_size, &iterations)) {
    printUsage(argv[0]);
    return 1;
  }

  printConfig(input_file, output_file, vocab_size, coverage, max_len, seed_size, iterations);
  printf("Reading training data from '%s'...\n", input_file);
  int text_count;
  char** texts = readTextFile(input_file, &text_count);

  if (!texts) {
    printf("Error: Failed to read training data\n");
    return 1;
  }
  printf("Loaded %d lines of text\n", text_count);

  printf("Creating unigram trainer...\n");
  UnigramTrainer* trainer = trainerCreate(vocab_size, coverage, max_len, seed_size);
  if (!trainer) {
    printf("Error: Failed to create trainer\n");
    freeTexts(texts, text_count);
    return 1;
  }

  printf("Starting training...\n");
  clock_t start_time = clock();
  bool success = trainUnigram(trainer, (const char**)texts, text_count, iterations);
  clock_t end_time = clock();
  double training_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

  if (!success) {
    printf("Error: Training failed\n");
    trainerDestroy(trainer);
    freeTexts(texts, text_count);
    return 1;
  }

  printf("\nTraining completed in %.2f seconds\n", training_time);
  printVocabStats(trainer);
  printf("Saving vocabulary to '%s'...\n", output_file);
  if (saveVocab(trainer, output_file)) { printf("Vocabulary saved successfully!\n"); }
  else { printf("Error: Failed to save vocabulary\n"); }

  printf("\nCleaning up...\n");
  trainerDestroy(trainer);
  freeTexts(texts, text_count);

  printf("Done!\n");
  return 0;
}