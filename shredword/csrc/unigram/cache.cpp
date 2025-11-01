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
  Node* last_node = cache->tail->prev;
  removeNode(last_node);
  return last_node;
}

int cacheGet(LRUCache* cache, int key) {
  if (!cache) {
    fprintf(stderr, "Invalid input, NULL value of LRUCache Pointer!\n");
    exit(EXIT_FAILURE);
  }
  int index = cache_hash(key, cache->hash_size);
  Node* node = cache->hash_table[index];
  while (node) {
    if (node->key == key) {
      moveToHead(cache, node);
      return node->value;
    }
    node = node->next;
  }
  return -1;
}

void cachePut(LRUCache* cache, int key, int value) {
  if (!cache) {
    fprintf(stderr, "Invalid input, NULL value of LRUCache Pointer!\n");
    exit(EXIT_FAILURE);
  }
  int index = cache_hash(key, cache->hash_size);
  Node* node = cache->hash_table[index];
  Node* prev = NULL;

  while (node) {
    if (node->key == key) {
      node->value = value;
      moveToHead(cache, node);
      return;
    }
    prev = node;
    node = node->next;
  }

  Node* new_node = (Node*)malloc(sizeof(Node));
  if (!new_node) return;
  new_node->key = key;
  new_node->value = value;
  new_node->next = NULL;

  if (cache->size < cache->capacity) {
    cache->size++;
    addNode(cache, new_node);
  } else {
    Node* tail = popTail(cache);
    int tail_index = cache_hash(tail->key, cache->hash_size);
    Node* hash_node = cache->hash_table[tail_index];
    Node* hash_prev = NULL;
    while (hash_node && hash_node->key != tail->key) {
      hash_prev = hash_node;
      hash_node = hash_node->next;
    }
    if (hash_node) {
      if (hash_prev) hash_prev->next = hash_node->next;
      else cache->hash_table[tail_index] = hash_node->next;
    }
    free(tail);
    addNode(cache, new_node);
  }

  if (cache->hash_table[index]) {
    Node* temp = cache->hash_table[index];
    while (temp->next) temp = temp->next;
    temp->next = new_node;
  } else { cache->hash_table[index] = new_node; }
}

void cacheFree(LRUCache* cache) {
  if (!cache) {
    fprintf(stderr, "Invalid input, NULL value of LRUCache Pointer!\n");
    exit(EXIT_FAILURE);
  }
  Node* current = cache->head;
  while (current) {
    Node* next = current->next;
    free(current);
    current = next;
  }
  for (int i = 0; i < cache->hash_size; i++) {
    Node* node = cache->hash_table[i];
    while (node) {
      Node* next = node->next;
      node = next;
    }
  }
  free(cache->hash_table);
  free(cache);
}