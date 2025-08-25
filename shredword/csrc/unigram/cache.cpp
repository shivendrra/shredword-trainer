#include <stdlib.h>
#include <string.h>
#include "cache.h"
#include "../inc/hash.h"

LRUCache* cache_create(size_t capacity) {
  LRUCache* cache = (LRUCache*)malloc(sizeof(LRUCache));
  cache->capacity = capacity;
  cache->size = 0;
  cache->hash_size = capacity > MIN_HASH_SIZE ? capacity * 2 : MIN_HASH_SIZE;

  cache->head = (Node*)malloc(sizeof(Node));
  cache->tail = (Node*)malloc(sizeof(Node));
  cache->head->prev = NULL;
  cache->head->next = cache->tail;
  cache->tail->prev = cache->head;
  cache->tail->next = NULL;

  cache->hash = (Node**)calloc(cache->hash_size, sizeof(Node*));
  return cache;
}

void move_to_end(LRUCache* cache, Node* node) {
  if (!cache || !node) return;  
  if (node->prev) node->prev->next = node->next;
  if (node->next) node->next->prev = node->prev;

  node->prev = cache->tail->prev;
  node->next = cache->tail;
  cache->tail->prev->next = node;
  cache->tail->prev = node;
}

int cache_get(LRUCache* cache, int key) {
  if (!cache) return -1;

  uint32_t hash_key = cache_hash(key, cache->hash_size);
  Node* node = cache->hash[hash_key];

  while (node) {
    if (node->key == key) {
      move_to_end(cache, node);
      return node->value;
    }
    node = node->next;
  }
  return -1;
}

void cache_put(LRUCache* cache, int key, int value) {
  if (!cache) return;
  uint32_t hash_key = cache_hash(key, cache->hash_size);
  Node* node = cache->hash[hash_key];

  while (node) {
    if (node->key == key) {
      node->value = value;
      move_to_end(cache, node);
      return;
    }
    node = node->next;
  }

  if (cache->size >= cache->capacity) {
    Node* lru = cache->head->next;
    uint32_t old_hash = cache_hash(lru->key, cache->hash_size);

    Node** hash_node = &cache->hash[old_hash];
    while (*hash_node && (*hash_node)->key != lru->key) { hash_node = &(*hash_node)->next; }
    if (*hash_node) *hash_node = (*hash_node)->next;

    cache->head->next = lru->next;
    lru->next->prev = cache->head;
    free(lru);
    cache->size--;
  }

  Node* new_node = (Node*)malloc(sizeof(Node));
  new_node->key = key;
  new_node->value = value;
  new_node->next = cache->hash[hash_key];
  new_node->prev = NULL;
  cache->hash[hash_key] = new_node;

  move_to_end(cache, new_node);
  cache->size++;
}

void cache_destroy(LRUCache* cache) {
  if (!cache) return;
  Node* current = cache->head;
  while (current) {
    Node* next = current->next;
    free(current);
    current = next;
  }

  free(cache->hash);
  free(cache);
}