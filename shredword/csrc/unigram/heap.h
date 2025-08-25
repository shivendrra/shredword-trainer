#ifndef __HEAP__H__
#define __HEAP__H__

typedef struct {
  int freq;
  char *token;
} HeapEntry;

typedef struct {
  char *key;
  int freq, removed;
} HeapHashEntry;

typedef struct {
  HeapEntry *heap;
  int heap_size, heap_cap;
  HeapHashEntry *hash;
  int hash_size, hash_cap;
} TokenFreqHeap;

extern "C" {
  TokenFreqHeap* heap_create();
  void heap_destroy(TokenFreqHeap *h);
  void heap_swap(HeapEntry *a, HeapEntry *b);
  void heap_up(TokenFreqHeap *h, int idx);
  void heap_down(TokenFreqHeap *h, int idx);
  HeapHashEntry* find_hash(TokenFreqHeap *h, const char *token);
  void heap_push(TokenFreqHeap *h, const char *token, int freq);
  int heap_pop(TokenFreqHeap *h, int *freq, char **token);
  void heap_remove(TokenFreqHeap *h, const char *token);
  void heap_update_freq(TokenFreqHeap *h, const char *token, int new_freq);
  int heap_len(TokenFreqHeap *h);
  int heap_contains(TokenFreqHeap *h, const char *token);
}

#endif  //!__HEAP__H__