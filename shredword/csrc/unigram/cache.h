#ifndef __CACHE__H__
#define __CACHE__H__

#include <stdlib.h>
#define  MIN_HASH_SIZE  100000

typedef struct Node {
  int key;
  int value;
  struct Node* prev;
  struct Node* next;
  struct Node* hnext;
} Node;

typedef struct LRUCache {
  size_t capacity, size, hash_size;
  Node *head, *tail;
  Node** hash_table;
} LRUCache;

extern "C" {
  LRUCache* cacheCreate(size_t capacity);
  void addNode(LRUCache* cache, Node* node);
  void removeNode(Node* node);
  void moveToHead(LRUCache* cache, Node* node);
  Node* popTail(LRUCache* cache);
  int cacheGet(LRUCache* cache, int key);
  void cachePut(LRUCache* cache, int key, int value);
  void cacheFree(LRUCache* cache);
}

#endif  //!__CACHE__H__