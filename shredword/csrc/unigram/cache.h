#ifndef __CACHE__H__
#define __CACHE__H__

#include <stdlib.h>
#define  MIN_HASH_SIZE  100000

typedef struct Node {
  int key, value;
  struct Node *prev, *next;
} Node;

typedef struct LRUCache {
  size_t capacity, size;
  Node *head, *tail;
  Node** hash;
} LRUCache;

extern "C" {
  LRUCache* cache_create(size_t capacity);
  void move_to_end(LRUCache* cache, Node* node);
  int cache_get(LRUCache* cache, int key);
  void cache_put(LRUCache* cache, int key, int value);
}

#endif  //!__CACHE__H__