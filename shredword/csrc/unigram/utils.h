#ifndef __SUBWORD_H__
#define __SUBWORD_H__

#include <stdint.h>
#include <stdlib.h>

#define MAX_TOKEN_LEN 256
#define MAX_SUBWORD_LEN 20
#define TRIE_CHILDREN 256
#define INITIAL_HASH_SIZE 16384
#define LOAD_FACTOR 0.75

typedef struct HashEntry {
  char *key;
  float value;
  struct HashEntry *next;
} HashEntry;

typedef struct FastHashMap {
  HashEntry **buckets;
  int size, count;
} FastHashMap;

extern "C" {
  // HashMap functions
  FastHashMap* hashmap_create(int initial_size);
  void hashmap_destroy(FastHashMap *map);
  void hashmap_put(FastHashMap *map, const char *key, float value);
  float hashmap_get(FastHashMap *map, const char *key);
  int hashmap_contains(FastHashMap *map, const char *key);
  void hashmap_remove(FastHashMap *map, const char *key);
  void resize_hashmap(FastHashMap *map);
}

#endif