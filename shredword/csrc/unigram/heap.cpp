#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../inc/hash.h"
#include "heap.h"

TokenFreqHeap* heapCreate() {
  TokenFreqHeap* heap = (TokenFreqHeap*)malloc(sizeof(TokenFreqHeap));

  heap->heap = (HeapEntry*)malloc(INITIAL_CAPACITY * sizeof(HeapEntry));
  heap->token_map = (TokenMap**)calloc(INITIAL_CAPACITY, sizeof(TokenMap*));

  heap->heap_size = 0;
  heap->heap_capacity = INITIAL_CAPACITY;
  heap->map_capacity = INITIAL_CAPACITY;
  heap->active_tokens = 0;
  return heap;
}

void heapSwap(HeapEntry* a, HeapEntry* b) {
  HeapEntry* temp = a;
  *a = *b;
  *b = *temp;
}

void heapifyUp(TokenFreqHeap* h, int idx) {
  while (idx > 0) {
    int parent = (idx - 1) / 2;
    if (h->heap[idx].freq >= h->heap[parent].freq) break;
    heapSwap(&h->heap[idx], &h->heap[parent]);
    idx = parent;
  }
}

void heapifyDown(TokenFreqHeap* h, int idx) {
  while (true) {
    int smallest = idx, left = 2 * idx + 1, right = 2 * idx + 2;
    if (left < h->heap_size && h->heap[left].freq < h->heap[smallest].freq) smallest = left;
    if (right < h->heap_size && h->heap[right].freq < h->heap[smallest].freq) smallest = right;
    if (smallest == idx) break;
    heapSwap(&h->heap[idx], &h->heap[smallest]);
    idx = smallest;
  }
}

bool heapResize(TokenFreqHeap* h) {
  int new_capacity = h->heap_capacity * 2;
  HeapEntry* new_heap = (HeapEntry*)realloc(h->heap, new_capacity * sizeof(HeapEntry));
  if (!new_heap) return false;
  h->heap = new_heap;
  h->heap_capacity = new_capacity;
  return true;
}

TokenMap* findToken(TokenFreqHeap* h, char const* token) {
  unsigned int idx = heap_hash(token, h->map_capacity);
  TokenMap* current = h->token_map[idx];

  while (current) {
    if (strcmp(current->token, token) == 0) return current;
    current = current->next;
  }
  return NULL;
}

bool heapPush(TokenFreqHeap* h, const char* token, int freq) {
  if (!h || !token || strlen(token) >= MAX_TOKEN_LEN) return false;
  TokenMap* existing = findToken(h, token);
  if (existing && existing->removed) {
    existing->removed = false;
    h->active_tokens++;
  }
  if (h->heap_size >= h->heap_capacity && !heapResize(h)) return false;
  strcpy(h->heap[h->heap_size].token, token);
  h->heap[h->heap_size].freq = freq;
  heapifyUp(h, h->heap_size++);

  unsigned int idx = heap_hash(token, h->map_capacity);
  if (!existing) {
    TokenMap* new_entry = (TokenMap*)malloc(sizeof(TokenMap));
    if (!new_entry) return false;
    strcpy(new_entry->token, token);
    new_entry->next = h->token_map[idx];
    h->token_map[idx] = new_entry;
    existing = new_entry;
    h->active_tokens++;
  }

  existing->freq = freq;
  existing->removed = false;
  return true;
}

bool heapPop(TokenFreqHeap* h, int* freq, char* token) {
  if (!h || !freq || !token) return false;

  while (h->heap_size > 0) {
    *freq = h->heap[0].freq;
    strcpy(token, h->heap[0].token);

    TokenMap* token_entry = findToken(h, token);
    if (!token_entry || token_entry->removed || token_entry->freq != *freq) {
      h->heap[0] = h->heap[--h->heap_size];
      if (h->heap_size > 0) heapifyDown(h, 0);
      continue;
    }

    token_entry->removed = true;
    h->active_tokens--;
    h->heap[0] = h->heap[--h->heap_size];
    if (h->heap_size > 0) heapifyDown(h, 0);
    return true;
  }
  return false;
}

bool heapRemove(TokenFreqHeap* h, const char* token) {
  if (!h || !token) return false;

  TokenMap* token_entry = findToken(h, token);
  if (!token_entry || token_entry->removed) return false;

  token_entry->removed = true;
  h->active_tokens--;
  return true;
}

bool heapUpdateFreq(TokenFreqHeap* h, const char* token, int new_freq) {
  if (!h || !token) return false;
  heapRemove(h, token);
  return heapPush(h, token, new_freq);
}

bool heapContains(TokenFreqHeap* h, const char* token) {
  if (!h || !token) return false;
  TokenMap* entry = findToken(h, token);
  return entry && !entry->removed;
}

bool heapEmpty(TokenFreqHeap* h) { return !h || h->active_tokens == 0; }
int heapSize(TokenFreqHeap* h) { return h ? h->active_tokens : 0; }

void heapFree(TokenFreqHeap* h) {
  if (!h) return;

  for (int i = 0; i < h->map_capacity; i++) {
    TokenMap* current = h->token_map[i];
    while (current) {
      TokenMap* next = current->next;
      free(current);
      current = next;
    }
  }

  free(h->heap);
  free(h->token_map);
  free(h);
}