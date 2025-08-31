#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "hashmap.h"
#include "inc/hash.h"

FastHashMap* hashmapCreate(int initial_size) {
  if (initial_size <= 0) initial_size = INITIAL_SIZE;

  FastHashMap* map = (FastHashMap*)malloc(sizeof(FastHashMap));
  map->buckets = (HashEntry**)calloc(initial_size, sizeof(HashEntry*));
  map->size = initial_size;
  map->count = 0;
  map->value_destructor = NULL;
  return map;
}

static HashEntry* createEntry(const char* key, void* value) {
  if (!key || strlen(key) >= MAX_KEY_LEN) return NULL;
  HashEntry* entry = (HashEntry*)malloc(sizeof(HashEntry));
  if (!entry) return NULL;
  entry->key = strdup(key);  
  entry->value = value;
  entry->next = NULL;
  return entry;
}

static void destroyEntry(HashEntry* entry, void (*destructor)(void*)) {
  if (!entry) return;
  free(entry->key);
  if (destructor && entry->value) destructor(entry->value);
  free(entry);
}

static bool hashMapResize(FastHashMap* map) {
  if (!map) return false;

  HashEntry** old_buckets = map->buckets;
  int old_size = map->size;  
  map->size *= 2;
  map->buckets = (HashEntry**)calloc(map->size, sizeof(HashEntry*));
  if (!map->buckets) {
    map->buckets = old_buckets;
    map->size = old_size;
    return false;
  }

  int old_count = map->count;
  map->count = 0;
  for (int i = 0; i < old_size; i++) {
    HashEntry* entry = old_buckets[i];
    while (entry) {
      HashEntry* next = entry->next;
      uint32_t new_idx = murmur3_hash(entry->key, map->size);

      entry->next = map->buckets[new_idx];
      map->buckets[new_idx] = entry;
      map->count++;
      entry = next;
    }
  }

  free(old_buckets);
  return true;
}

bool hashMapSet(FastHashMap* map, const char* key, void* value) {
  if (!map || !key) return false;

  if (map->count >= map->size * LOAD_FACTOR_THRESHOLD) { if (!hashMapResize(map)) return false; }
  uint32_t idx = murmur3_hash(key, map->size);
  HashEntry* entry = map->buckets[idx];
  while (entry) {
    if (strcmp(entry->key, key) == 0) {
      if (map->value_destructor && entry->value) map->value_destructor(entry->value);
      entry->value = value;
      return true;
    }
    entry = entry->next;
  }

  HashEntry* new_entry = createEntry(key, value);
  if (!new_entry) return false;  
  new_entry->next = map->buckets[idx];
  map->buckets[idx] = new_entry;
  map->count++;
  return true;
}

void* hashMapGet(FastHashMap* map, const char* key) {
  if (!map || !key) return NULL;

  uint32_t idx = murmur3_hash(key, map->size);
  HashEntry* entry = map->buckets[idx];  
  while (entry) {
    if (strcmp(entry->key, key) == 0) return entry->value;
    entry = entry->next;
  }
  return NULL;
}

void* hashMapGetDefault(FastHashMap* map, const char* key, void* default_value) {
  void* result = hashMapGet(map, key);
  return result ? result : default_value;
}

bool hashMapRemove(FastHashMap* map, const char* key) {
  if (!map || !key) return false;
  uint32_t idx = murmur3_hash(key, map->size);
  HashEntry* entry = map->buckets[idx];
  HashEntry* prev = NULL;
  while (entry) {
    if (strcmp(entry->key, key) == 0) {
      if (prev) prev->next = entry->next;
      else map->buckets[idx] = entry->next;
      destroyEntry(entry, map->value_destructor);
      map->count--;
      return true;
    }
    prev = entry;
    entry = entry->next;
  }
  
  return false;
}

void hashMapSetDestructor(FastHashMap* map, void (*destructor)(void*)) { if (map) map->value_destructor = destructor; }
bool hashMapContains(FastHashMap* map, const char* key) { return hashMapGet(map, key) != NULL; }
int hashMapSize(FastHashMap* map) { return map ? map->count : 0; }
bool hashMapEmpty(FastHashMap* map) { return !map || map->count == 0; }
static void print_int_value(const char* key, void* value) { printf("  %s: %d\n", key, *(int*)value); }  // helper
void hashMapIteratorDestroy(HashMapIterator* iter) { free(iter); }

HashMapIterator* hashMapIteratorCreate(FastHashMap* map) {
  if (!map) return NULL;

  HashMapIterator* iter = (HashMapIterator*)malloc(sizeof(HashMapIterator));
  iter->map = map;
  iter->bucket_idx = 0;
  iter->current_entry = NULL;
  while (iter->bucket_idx < map->size && !map->buckets[iter->bucket_idx]) iter->bucket_idx++;
  if (iter->bucket_idx < map->size) iter->current_entry = map->buckets[iter->bucket_idx];
  return iter;
}

bool hashMapIteratorNext(HashMapIterator* iter, const char** key, void** value) {
  if (!iter || !iter->map || !key || !value) return false;

  if (!iter->current_entry) return false;
  *key = iter->current_entry->key;
  *value = iter->current_entry->value;
  iter->current_entry = iter->current_entry->next;
  if (!iter->current_entry) {
    iter->bucket_idx++;
    while (iter->bucket_idx < iter->map->size && !iter->map->buckets[iter->bucket_idx]) iter->bucket_idx++;
    if (iter->bucket_idx < iter->map->size) iter->current_entry = iter->map->buckets[iter->bucket_idx];
  }
  return true;
}


void hashMapClear(FastHashMap* map) {
  if (!map) return;
  
  for (int i = 0; i < map->size; i++) {
    HashEntry* entry = map->buckets[i];
    while (entry) {
      HashEntry* next = entry->next;
      destroyEntry(entry, map->value_destructor);
      entry = next;
    }
    map->buckets[i] = NULL;
  }
  map->count = 0;
}

void hashMapDestroy(FastHashMap* map) {
  if (!map) return;
  hashMapClear(map);
  free(map->buckets);
  free(map);
}

void hashMapPrint(FastHashMap* map, void (*print_value)(const char*, void*)) {
  if (!map || !print_value) return;
  
  printf("HashMap size: %d/%d (%.2f%% load)\n", map->count, map->size, (float)map->count / map->size * 100);
  HashMapIterator* iter = hashMapIteratorCreate(map);
  if (!iter) return;  
  const char* key;
  void* value;
  while (hashMapIteratorNext(iter, &key, &value)) { print_value(key, value); }
  
  hashMapIteratorDestroy(iter);
}