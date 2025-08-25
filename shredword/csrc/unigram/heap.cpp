#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../inc/hash.h"
#include "heap.h"

TokenFreqHeap* heap_create() {
  TokenFreqHeap *h = (TokenFreqHeap*)malloc(sizeof(TokenFreqHeap));
  h->heap = (HeapEntry*)malloc(sizeof(HeapEntry) * 1000);
  h->heap_size = 0;
  h->heap_cap = 1000;
  h->hash_cap = 10007;
  h->hash = (HeapHashEntry*)calloc(h->hash_cap, sizeof(HeapHashEntry));
  h->hash_size = 0;
  return h;
}

void heap_destroy(TokenFreqHeap *h) {
  if (!h) return;
  for (int i = 0; i < h->heap_size; i++) { if (h->heap[i].token) free(h->heap[i].token); }
  for (int i = 0; i < h->hash_cap; i++) { if (h->hash[i].key) free(h->hash[i].key); }
  free(h->heap);
  free(h->hash);
  free(h);
}

void heap_swap(HeapEntry *a, HeapEntry *b) {
  HeapEntry temp = *a;
  *a = *b;
  *b = temp;
}

void heap_up(TokenFreqHeap *h, int idx) {
  while (idx && h->heap[idx].freq < h->heap[(idx-1)/2].freq) {
    heap_swap(&h->heap[idx], &h->heap[(idx-1)/2]);
    idx = (idx-1)/2;
  }
}

void heap_down(TokenFreqHeap *h, int idx) {
  int min = idx, left = 2*idx+1, right = 2*idx+2;
  if (left < h->heap_size && h->heap[left].freq < h->heap[min].freq) min = left;
  if (right < h->heap_size && h->heap[right].freq < h->heap[min].freq) min = right;
  if (min != idx) {
    heap_swap(&h->heap[idx], &h->heap[min]);
    heap_down(h, min);
  }
}

HeapHashEntry* find_hash(TokenFreqHeap *h, const char *token) {
  int idx = heap_hash(token, h->hash_cap);
  while (h->hash[idx].key && strcmp(h->hash[idx].key, token)) { idx = (idx+1) % h->hash_cap; }
  return &h->hash[idx];
}

void heap_push(TokenFreqHeap *h, const char *token, int freq) {
  if (!h || !token) return;
  HeapHashEntry *entry = find_hash(h, token);
  if (entry->key && entry->removed) {
    entry->removed = 0;
    h->hash_size++;
  }

  if (h->heap_size >= h->heap_cap) {
    h->heap_cap *= 2;
    h->heap = (HeapEntry*)realloc(h->heap, sizeof(HeapEntry) * h->heap_cap);
  }

  h->heap[h->heap_size].freq = freq;
  h->heap[h->heap_size].token = strdup(token);
  heap_up(h, h->heap_size++);

  if (!entry->key) {
    entry->key = strdup(token);
    h->hash_size++;
  }
  entry->freq = freq;
  entry->removed = 0;
}

int heap_pop(TokenFreqHeap *h, int *freq, char **token) {
  if (!h || h->heap_size == 0) return 0;

  while (h->heap_size) {
    *freq = h->heap[0].freq;
    *token = strdup(h->heap[0].token);
    HeapHashEntry *entry = find_hash(h, h->heap[0].token);
    int is_valid = entry->key && !entry->removed && entry->freq == *freq;

    free(h->heap[0].token);
    h->heap[0] = h->heap[--h->heap_size];
    if (h->heap_size) heap_down(h, 0);

    if (is_valid) {
      free(entry->key);
      entry->key = NULL;
      entry->removed = 0;
      h->hash_size--;
      return 1;
    }
    free(*token);
  }
  return 0;
}

void heap_remove(TokenFreqHeap *h, const char *token) {
  if (!h || !token) return;
  HeapHashEntry *entry = find_hash(h, token);
  if (entry->key && !entry->removed) {
    entry->removed = 1;
    h->hash_size--;
  }
}

void heap_update_freq(TokenFreqHeap *h, const char *token, int new_freq) {
  if (!h || !token) return;
  HeapHashEntry *entry = find_hash(h, token);
  if (entry->key && !entry->removed) {
    entry->removed = 1;
    h->hash_size--;
  }
  heap_push(h, token, new_freq);
}

int heap_len(TokenFreqHeap *h) { return h ? h->hash_size : 0; }

int heap_contains(TokenFreqHeap *h, const char *token) {
  if (!h || !token) return 0;
  HeapHashEntry *entry = find_hash(h, token);
  return entry->key && !entry->removed;
}