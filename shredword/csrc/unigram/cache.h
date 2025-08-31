#ifndef __CACHE__H__
#define __CACHE__H__

#include <stdlib.h>
#define  MIN_HASH_SIZE  100000

typedef struct Node {
  int key, value;
  struct Node *prev, *next;
} Node;

typedef struct LRUCache {
  size_t capacity, size, hash_size;
  Node *head, *tail;
  Node** hash_table;
} LRUCache;

extern "C" {
  LRUCache* cache_create(size_t capacity);
  void addNode(LRUCache* cache, Node* node);
  void removeNode(Node* node);
  void moveToHead(LRUCache* cache, Node* node);
  Node* popTail(LRUCache* cache);
  int cacheGet(LRUCache* cache, int key);
}

#endif  //!__CACHE__H__