#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "cache.h"
#include "../inc/hash.h"

LRUCache* cacheCreate(size_t capacity) {
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

  cache->hash_table = (Node**)calloc(cache->hash_size, sizeof(Node*));
  return cache;
}

void addNode(LRUCache* cache, Node* node) {
  node->prev = cache->head;
  node->next = cache->head->next;
  cache->head->next->prev = node;
  cache->head->next = node;
}

void removeNode(Node* node) {
  node->prev->next = node->next;
  node->next->prev = node->prev;
}

void moveToHead(LRUCache* cache, Node* node) {
  removeNode(node);
  addNode(cache, node);
}

Node* popTail(LRUCache* cache) {
  Node* node = cache->tail->prev;
  removeNode(node);
  return node;
}

int cacheGet(LRUCache* cache, int key) {
  int index = cache_hash(key, cache->hash_size);
  Node* node = cache->hash_table[index];

  while (node) {
    if (node->key == key) {
      moveToHead(cache, node);
      return node->value;
    }
    node = node->hnext;
  }
  return -1;
}

void cachePut(LRUCache* cache, int key, int value) {
  int index = cache_hash(key, cache->hash_size);
  Node* node = cache->hash_table[index];

  while (node) {
    if (node->key == key) {
      node->value = value;
      moveToHead(cache, node);
      return;
    }
    node = node->hnext;
  }

  Node* new_node = (Node*)malloc(sizeof(Node));
  new_node->key = key;
  new_node->value = value;
  new_node->hnext = cache->hash_table[index];
  cache->hash_table[index] = new_node;

  if (cache->size < cache->capacity) {
    cache->size++;
    addNode(cache, new_node);
  } else {
    Node* tail = popTail(cache);
    int tidx = cache_hash(tail->key, cache->hash_size);

    Node* cur = cache->hash_table[tidx];
    Node* prev = NULL;
    while (cur) {
      if (cur == tail) {
        if (prev) prev->hnext = cur->hnext;
        else cache->hash_table[tidx] = cur->hnext;
        break;
      }
      prev = cur;
      cur = cur->hnext;
    }

    free(tail);
    addNode(cache, new_node);
  }
}

void cacheFree(LRUCache* cache) {
  Node* cur = cache->head;
  while (cur) {
    Node* next = cur->next;
    free(cur);
    cur = next;
  }

  free(cache->hash_table);
  free(cache);
}
