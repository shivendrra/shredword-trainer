#ifndef __HEAP__H__
#define __HEAP__H__

#define MAX_TOKEN_LEN 256
#define INITIAL_CAPACITY 1024

typedef struct HeapEntry {
  int freq;
  char token[MAX_TOKEN_LEN];
} HeapEntry;

typedef struct TokenMap {
  char token[MAX_TOKEN_LEN];
  int freq; 
  bool removed;
  struct TokenMap* next;
} TokenMap;

typedef struct TokenFreqHeap {
  HeapEntry* heap;
  int heap_size, heap_capacity;
  TokenMap** token_map;
  int map_capacity;
  int active_tokens;
} TokenFreqHeap;

extern "C" {
  TokenFreqHeap* heapCreate();
  void heapSwap(HeapEntry *a, HeapEntry *b);
  void heapifyUp(TokenFreqHeap *h, int idx);
  void heapifyDown(TokenFreqHeap *h, int idx);
  void heapFree(TokenFreqHeap* h);
  
  TokenMap* findToken(TokenFreqHeap* h, const char* token);
  int heapSize(TokenFreqHeap* heap);

  bool heapResize(TokenFreqHeap* h);
  bool heapPush(TokenFreqHeap *h, const char *token, int freq);
  bool heapPop(TokenFreqHeap* heap, int* freq, char* token);
  bool heapRemove(TokenFreqHeap *h, const char *token);
  bool heapUpdateFreq(TokenFreqHeap *h, const char *token, int new_freq);
  bool heapContains(TokenFreqHeap *h, const char *token);
  bool heapEmpty(TokenFreqHeap* h);
}

#endif  //!__HEAP__H__