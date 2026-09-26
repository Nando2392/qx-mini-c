#include "qx_expert_cache.h"

#include <limits.h>
#include <string.h>

struct qx_expert_cache_entry {
    qx_expert_cache_key key;
    uint8_t *data;
    uint64_t pin_count;
    struct qx_expert_cache_entry *older;
    struct qx_expert_cache_entry *newer;
};

static int qx_expert_cache_fail(char *err, uint64_t err_len, const char *message) {
    if (err != NULL && err_len != 0u) {
        size_t count = strlen(message);
        if ((uint64_t)count >= err_len) count = (size_t)(err_len - 1u);
        memcpy(err, message, count);
        err[count] = '\0';
    }
    return 0;
}

static int qx_expert_cache_add(uint64_t left, uint64_t right, uint64_t *out) {
    if (left > UINT64_MAX - right) return 0;
    *out = left + right;
    return 1;
}

static int qx_expert_cache_key_equal(const qx_expert_cache_key *left,
                                     const qx_expert_cache_key *right) {
    return left->tensor_offset == right->tensor_offset &&
           left->slice_offset == right->slice_offset &&
           left->slice_bytes == right->slice_bytes &&
           left->expert == right->expert;
}

static void qx_expert_cache_unlink(qx_expert_cache *cache,
                                   struct qx_expert_cache_entry *entry) {
    if (entry->older != NULL) entry->older->newer = entry->newer;
    else cache->lru = entry->newer;
    if (entry->newer != NULL) entry->newer->older = entry->older;
    else cache->mru = entry->older;
    entry->older = NULL;
    entry->newer = NULL;
}

static void qx_expert_cache_append_mru(qx_expert_cache *cache,
                                       struct qx_expert_cache_entry *entry) {
    entry->older = cache->mru;
    entry->newer = NULL;
    if (cache->mru != NULL) cache->mru->newer = entry;
    else cache->lru = entry;
    cache->mru = entry;
}

static void qx_expert_cache_promote(qx_expert_cache *cache,
                                    struct qx_expert_cache_entry *entry) {
    if (cache->mru == entry) return;
    qx_expert_cache_unlink(cache, entry);
    qx_expert_cache_append_mru(cache, entry);
}

static void qx_expert_cache_free_entry(qx_expert_cache *cache,
                                       struct qx_expert_cache_entry *entry) {
    cache->ops.free(cache->ops.context, entry->data);
    cache->ops.free(cache->ops.context, entry);
}

static void qx_expert_cache_evict(qx_expert_cache *cache,
                                  struct qx_expert_cache_entry *entry) {
    qx_expert_cache_unlink(cache, entry);
    cache->counters.current_resident_packed_bytes -= entry->key.slice_bytes;
    cache->counters.evictions++;
    qx_expert_cache_free_entry(cache, entry);
}

int qx_expert_cache_init(qx_expert_cache *cache, uint64_t budget_bytes,
                         const qx_expert_cache_ops *ops,
                         char *err, uint64_t err_len) {
    if (cache == NULL) return qx_expert_cache_fail(err, err_len, "invalid expert cache");
    memset(cache, 0, sizeof(*cache));
    if (budget_bytes == 0u) return qx_expert_cache_fail(err, err_len, "expert cache budget must be positive");
    if (ops == NULL || ops->alloc == NULL || ops->free == NULL || ops->read_exact == NULL)
        return qx_expert_cache_fail(err, err_len, "invalid expert cache operations");
    cache->budget_bytes = budget_bytes;
    cache->ops = *ops;
    cache->initialized = 1u;
    return 1;
}

int qx_expert_cache_acquire(qx_expert_cache *cache, qx_expert_cache_key key,
                            qx_expert_cache_borrow *borrow,
                            char *err, uint64_t err_len) {
    struct qx_expert_cache_entry *entry;
    uint64_t absolute_offset;
    uint64_t end_offset;
    uint64_t actual_read = 0u;
    uint64_t next_value;
    int read_ok;

    if (borrow != NULL) memset(borrow, 0, sizeof(*borrow));
    if (cache == NULL || cache->initialized == 0u)
        return qx_expert_cache_fail(err, err_len, "expert cache is not initialized");
    if (borrow == NULL) return qx_expert_cache_fail(err, err_len, "invalid expert cache borrow");
    if (key.reserved != 0u || key.slice_bytes == 0u)
        return qx_expert_cache_fail(err, err_len, "invalid expert cache key");
    if (!qx_expert_cache_add(key.tensor_offset, key.slice_offset, &absolute_offset) ||
        !qx_expert_cache_add(absolute_offset, key.slice_bytes, &end_offset))
        return qx_expert_cache_fail(err, err_len, "expert cache key offset overflow");
    (void)end_offset;
    if (key.slice_bytes > (uint64_t)SIZE_MAX)
        return qx_expert_cache_fail(err, err_len, "expert cache slice exceeds address space");

    for (entry = cache->lru; entry != NULL; entry = entry->newer) {
        if (qx_expert_cache_key_equal(&entry->key, &key)) {
            if (cache->counters.requests == UINT64_MAX ||
                cache->counters.hits == UINT64_MAX ||
                !qx_expert_cache_add(cache->counters.buffered_bytes_avoided,
                                     key.slice_bytes, &next_value))
                return qx_expert_cache_fail(err, err_len, "expert cache counter overflow");
            cache->counters.requests++;
            cache->counters.hits++;
            cache->counters.buffered_bytes_avoided = next_value;
            if (entry->pin_count == UINT64_MAX)
                return qx_expert_cache_fail(err, err_len, "expert cache pin count overflow");
            entry->pin_count++;
            qx_expert_cache_promote(cache, entry);
            borrow->data = entry->data;
            borrow->size = entry->key.slice_bytes;
            borrow->entry = entry;
            borrow->owner = cache;
            return 1;
        }
    }

    if (cache->counters.requests == UINT64_MAX || cache->counters.misses == UINT64_MAX)
        return qx_expert_cache_fail(err, err_len, "expert cache counter overflow");
    cache->counters.requests++;
    cache->counters.misses++;
    if (key.slice_bytes > cache->budget_bytes)
        return qx_expert_cache_fail(err, err_len, "expert cache slice exceeds budget");

    while (cache->counters.current_resident_packed_bytes > cache->budget_bytes - key.slice_bytes) {
        struct qx_expert_cache_entry *victim = cache->lru;
        while (victim != NULL && victim->pin_count != 0u) victim = victim->newer;
        if (victim == NULL)
            return qx_expert_cache_fail(err, err_len, "expert cache admission blocked by pinned entries");
        if (cache->counters.evictions == UINT64_MAX)
            return qx_expert_cache_fail(err, err_len, "expert cache counter overflow");
        qx_expert_cache_evict(cache, victim);
    }

    entry = (struct qx_expert_cache_entry *)cache->ops.alloc(cache->ops.context, sizeof(*entry));
    if (entry == NULL) return qx_expert_cache_fail(err, err_len, "expert cache metadata allocation failed");
    memset(entry, 0, sizeof(*entry));
    entry->data = (uint8_t *)cache->ops.alloc(cache->ops.context, (size_t)key.slice_bytes);
    if (entry->data == NULL) {
        cache->ops.free(cache->ops.context, entry);
        return qx_expert_cache_fail(err, err_len, "expert cache packed allocation failed");
    }

    read_ok = cache->ops.read_exact(cache->ops.context, absolute_offset, entry->data,
                                    key.slice_bytes, &actual_read);
    if (actual_read > key.slice_bytes) actual_read = key.slice_bytes;
    if (!qx_expert_cache_add(cache->counters.buffered_bytes_read, actual_read, &next_value)) {
        qx_expert_cache_free_entry(cache, entry);
        return qx_expert_cache_fail(err, err_len, "expert cache counter overflow");
    }
    cache->counters.buffered_bytes_read = next_value;
    if (!read_ok) {
        qx_expert_cache_free_entry(cache, entry);
        return qx_expert_cache_fail(err, err_len, "expert cache read failed");
    }
    if (actual_read != key.slice_bytes) {
        qx_expert_cache_free_entry(cache, entry);
        return qx_expert_cache_fail(err, err_len, "expert cache truncated read");
    }
    if (cache->counters.loads == UINT64_MAX) {
        qx_expert_cache_free_entry(cache, entry);
        return qx_expert_cache_fail(err, err_len, "expert cache counter overflow");
    }

    entry->key = key;
    entry->pin_count = 1u;
    qx_expert_cache_append_mru(cache, entry);
    cache->counters.loads++;
    cache->counters.current_resident_packed_bytes += key.slice_bytes;
    if (cache->counters.current_resident_packed_bytes > cache->counters.peak_resident_packed_bytes)
        cache->counters.peak_resident_packed_bytes = cache->counters.current_resident_packed_bytes;
    borrow->data = entry->data;
    borrow->size = entry->key.slice_bytes;
    borrow->entry = entry;
    borrow->owner = cache;
    return 1;
}

void qx_expert_cache_release(qx_expert_cache *cache,
                             qx_expert_cache_borrow *borrow) {
    struct qx_expert_cache_entry *entry;
    if (cache == NULL || borrow == NULL || borrow->owner != cache) return;
    if (cache->initialized != 0u && borrow->entry != NULL) {
        for (entry = cache->lru; entry != NULL; entry = entry->newer) {
            if (entry == borrow->entry) {
                if (entry->pin_count != 0u) entry->pin_count--;
                break;
            }
        }
    }
    memset(borrow, 0, sizeof(*borrow));
}

void qx_expert_cache_snapshot(const qx_expert_cache *cache,
                              qx_expert_cache_counters *out_counters) {
    if (out_counters == NULL) return;
    memset(out_counters, 0, sizeof(*out_counters));
    if (cache != NULL && cache->initialized != 0u) *out_counters = cache->counters;
}

void qx_expert_cache_destroy(qx_expert_cache *cache) {
    struct qx_expert_cache_entry *entry;
    if (cache == NULL || cache->initialized == 0u) return;
    entry = cache->lru;
    while (entry != NULL) {
        struct qx_expert_cache_entry *next = entry->newer;
        qx_expert_cache_free_entry(cache, entry);
        entry = next;
    }
    memset(cache, 0, sizeof(*cache));
}
