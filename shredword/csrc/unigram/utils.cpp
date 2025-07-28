#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "utils.h"
#include "../inc/hash.h"

// FastHashMap functions
FastHashMap* hashmap_create(int initial_size) {
  FastHashMap *map = (FastHashMap*)malloc(sizeof(FastHashMap));
  map->size = initial_size, map->count = 0;
  map->buckets = (HashEntry**)calloc(map->size, sizeof(HashEntry*));
  return map;
}

void hashmap_destroy(FastHashMap *map) {
  if (!map) return;
  for (int i = 0; i < map->size; i++) {
    HashEntry *entry = map->buckets[i];
    while (entry) {
      HashEntry *next = entry->next;
      free(entry->key);
      free(entry);
      entry = next;
    }
  }
  free(map->buckets);
  free(map);
}

static void resize_hashmap(FastHashMap *map) {
  HashEntry **old_buckets = map->buckets;
  int old_size = map->size;
  map->size *= 2, map->count = 0;
  map->buckets = (HashEntry**)calloc(map->size, sizeof(HashEntry*));

  for (int i = 0; i < old_size; i++) {
    HashEntry *entry = old_buckets[i];
    while (entry) {
      HashEntry *next = entry->next;
      hashmap_put(map, entry->key, entry->value);
      free(entry->key);
      free(entry);
      entry = next;
    }
  }
  free(old_buckets);
}

void hashmap_put(FastHashMap *map, const char *key, float value) {
  if (!map || !key) return;
  if (map->count >= map->size * LOAD_FACTOR) resize_hashmap(map);

  uint32_t idx = djb2_hash(key) % map->size;
  HashEntry *entry = map->buckets[idx];

  while (entry) {
    if (strcmp(entry->key, key) == 0) { entry->value = value; return; }
    entry = entry->next;
  }

  entry = (HashEntry*)malloc(sizeof(HashEntry));
  entry->key = strdup(key);
  entry->value = value;
  entry->next = map->buckets[idx];
  map->buckets[idx] = entry;
  map->count++;
}

float hashmap_get(FastHashMap *map, const char *key) {
  if (!map || !key) return -FLT_MAX;
  uint32_t idx = djb2_hash(key) % map->size;
  HashEntry *entry = map->buckets[idx];

  while (entry) {
    if (strcmp(entry->key, key) == 0) return entry->value;
    entry = entry->next;
  }
  return -FLT_MAX;
}

int hashmap_contains(FastHashMap *map, const char *key) {
  return hashmap_get(map, key) != -FLT_MAX;
}

void hashmap_remove(FastHashMap *map, const char *key) {
  if (!map || !key) return;
  uint32_t idx = djb2_hash(key) % map->size;
  HashEntry **entry = &map->buckets[idx];

  while (*entry) {
    if (strcmp((*entry)->key, key) == 0) {
      HashEntry *to_remove = *entry;
      *entry = (*entry)->next;
      free(to_remove->key);
      free(to_remove);
      map->count--;
      return;
    }
    entry = &(*entry)->next;
  }
}