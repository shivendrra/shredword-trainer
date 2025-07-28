#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "cache.h"
#include "../inc/hash.h"

LRUCache* cache_create(size_t capacity) {
  LRUCache* self = (LRUCache*)malloc(sizeof(LRUCache));
  self->capacity = capacity;
  self->size = 0;
  self->hash_size = capacity * 2;
  self->head = (Node*)malloc(sizeof(Node));
  self->tail = (Node*)malloc(sizeof(Node));
  self->head->next = self->tail;
  self->tail->prev = self->head;
  self->hash = (Node**)calloc(self->hash_size, sizeof(Node*));
  return self;
}

void move_to_end(LRUCache *cache, Node *node) {
  node->prev->next = node->next;
  node->next->prev = node->prev;
  node->prev = cache->tail->prev;
  node->next = cache->tail;
  cache->tail->prev->next = node;
  cache->tail->prev = node;
}

int cache_get(LRUCache* cache, int key) {
  int index = cache_hash(key, cache->hash_size);
  Node* node = cache->hash[index];
  while (node && node->key != key) node = node->next;
  if (!node) return -1;
  move_to_end(cache, node);
  return node->value;
}

void cache_put(LRUCache* cache, int key, int value) {
  int index = cache_hash(key, cache->hash_size);
  Node* node = cache->hash[index];
  while (node && node->key != key) node = node->next;
  if (node) {
    node->value = value;
    move_to_end(cache, node);
  } else {
    if (cache->size >= cache->capacity) {
      Node* lru = cache->head->next;
      int lru_idx = cache_hash(lru->key, cache->hash_size);
      if (cache->hash[lru_idx] == lru) cache->hash[lru_idx] = lru->next;
      lru->prev->next = lru->next;
      lru->next->prev = lru->prev;
      free(lru);
      cache->size--;
    }
    node = (Node*)malloc(sizeof(Node));
    node->key = key;
    node->value = value;
    node->next = cache->hash[index];
    cache->hash[index] = node;

    node->prev = cache->tail->prev;
    node->next = cache->tail;
    cache->tail->prev->next = node;
    cache->tail->prev = node;
    cache->size++;
  }
}

void cache_destroy(LRUCache* cache) {
  if (!cache) return;
  Node* current = cache->head;
  while (current) {
    Node* next = current->next;
    free(current);
    current = next;
  }
  if (cache->hash) free(cache->hash);
  free(cache);
}