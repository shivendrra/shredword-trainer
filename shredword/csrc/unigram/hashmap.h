#ifndef __SUBWORD_H__
#define __SUBWORD_H__

#include <stdint.h>
#include <stdlib.h>

#define INITIAL_SIZE 16384
#define LOAD_FACTOR_THRESHOLD 0.75
#define MAX_KEY_LEN 512

typedef struct HashEntry {
  char *key;
  void *value;
  struct HashEntry *next;
} HashEntry;

typedef struct FastHashMap {
  HashEntry **buckets;
  int size, count;
  void (*value_destructor)(void *);
} FastHashMap;

typedef struct HashMapIterator {
  FastHashMap* map;
  int bucket_idx;
  HashEntry* current_entry;
} HashMapIterator;

extern "C" {
  // HashMap functions
  FastHashMap* hashmapCreate(int initial_size);
  void hashMapSetDestructor(FastHashMap* map, void (*destructor)(void*));
  HashEntry* createEntry(const char* key, void* value);
  void destroyEntry(HashEntry* entry, void (*destructor)(void*));
  bool hashMapResize(FastHashMap* map);
  bool hashMapSet(FastHashMap* map, const char* key, void* value);
  void* hashMapGet(FastHashMap* map, const char* key);
  void* hashMapGetDefault(FastHashMap* map, const char* key, void* default_value);
  bool hashMapContains(FastHashMap* map, const char* key);
  bool hashMapRemove(FastHashMap* map, const char* key);
  int hashMapSize(FastHashMap* map);
  bool hashMapEmpty(FastHashMap* map);

  HashMapIterator* hashMapIteratorCreate(FastHashMap* map);
  bool hashMapIteratorNext(HashMapIterator* iter, const char** key, void** value);
  void hashMapIteratorDestroy(HashMapIterator* iter);
  void hashMapClear(FastHashMap* map);
  void hashMapDestroy(FastHashMap* map);
  void hashMapPrint(FastHashMap* map, void (*print_value)(const char*, void*));
}

#endif