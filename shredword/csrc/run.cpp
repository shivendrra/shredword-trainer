#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "unigram/unigram.h"

#define MAX_LINE_LENGTH 2048
#define MAX_RUN_TEXTS 100000

char** read_text_file(const char* filepath, int* text_count) {
  FILE* file = fopen(filepath, "r");
  if (!file) {
    printf("Error: Could not open file %s\n", filepath);
    return NULL;
  }

  char** texts = (char**)malloc(MAX_RUN_TEXTS * sizeof(char*));
  char line[MAX_LINE_LENGTH];
  int count = 0;

  while (fgets(line, sizeof(line), file) && count < MAX_RUN_TEXTS) {
    size_t len = strlen(line);
    if (len > 0 && line[len-1] == '\n') {
      line[len-1] = '\0';
      len--;
    }
    
    if (len > 0) {
      texts[count] = (char*)malloc((len + 1) * sizeof(char));
      strcpy(texts[count], line);
      count++;
    }
  }

  fclose(file);
  *text_count = count;
  printf("Read %d lines from %s\n", count, filepath);
  return texts;
}

void free_texts(char** texts, int count) {
  if (!texts) return;
  for (int i = 0; i < count; i++) {
    if (texts[i]) free(texts[i]);
  }
  free(texts);
}

void print_sample_vocab(FastHashMap* vocab, int sample_size) {
  printf("\nSample vocabulary (first %d tokens):\n", sample_size);
  printf("Token\t\tScore\n");
  printf("-----\t\t-----\n");

  int printed = 0;
  for (int i = 0; i < vocab->size && printed < sample_size; i++) {
    HashEntry *entry = vocab->buckets[i];
    while (entry && printed < sample_size) {
      printf("'%s'\t\t%.4f\n", entry->key, entry->value);
      printed++;
      entry = entry->next;
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Usage: %s <input_file> [vocab_size] [output_vocab_file]\n", argv[0]);
    printf("Example: %s corpus.txt 8000 vocab.txt\n", argv[0]);
    return 1;
  }

  const char* input_file = argv[1];
  int vocab_size = (argc > 2) ? atoi(argv[2]) : DEFAULT_VOCAB_SIZE;
  const char* output_file = (argc > 3) ? argv[3] : NULL;

  printf("Starting Unigram vocabulary training...\n");
  printf("Input file: %s\n", input_file);
  printf("Target vocabulary size: %d\n", vocab_size);

  int text_count = 0;
  char** texts = read_text_file(input_file, &text_count);
  
  if (!texts || text_count == 0) {
    printf("Error: No texts loaded from file\n");
    return 1;
  }

  UnigramTrainer* trainer = trainer_create(vocab_size, DEFAULT_COVERAGE, DEFAULT_MAX_LEN, DEFAULT_SEED_SIZE);
  if (!trainer) {
    printf("Error: Failed to create trainer\n");
    free_texts(texts, text_count);
    return 1;
  }

  printf("\nTraining parameters:\n");
  printf("  Vocabulary size: %d\n", trainer->vocab_size);
  printf("  Character coverage: %.4f\n", trainer->character_coverage);
  printf("  Max token length: %d\n", trainer->max_len);
  printf("  Seed size: %d\n", trainer->seed_size);
  printf("  Training texts: %d\n", text_count);

  FastHashMap* final_vocab = train_unigram(trainer, texts, text_count, MAX_ITERATIONS);
  
  if (!final_vocab) {
    printf("Error: Training failed\n");
    trainer_destroy(trainer);
    free_texts(texts, text_count);
    return 1;
  }

  printf("\nTraining completed successfully!\n");
  printf("Final vocabulary size: %d\n", final_vocab->count);

  print_sample_vocab(final_vocab, 20);

  if (output_file) {
    save_vocab(trainer, output_file);
    printf("\nVocabulary saved to: %s\n", output_file);
  }

  trainer_destroy(trainer);
  free_texts(texts, text_count);
  
  printf("\nTraining session completed.\n");
  return 0;
}