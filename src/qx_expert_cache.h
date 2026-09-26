#ifndef QX_EXPERT_CACHE_H
#define QX_EXPERT_CACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t tensor_offset;
    uint64_t slice_offset;
    uint64_t slice_bytes;
    uint32_t expert;
    uint32_t reserved;
} qx_expert_cache_key;

typedef struct {
    void *context;
    void *(*alloc)(void *context, size_t size);
    void (*free)(void *context, void *ptr);
    int (*read_exact)(void *context, uint64_t offset, void *dst,
                      uint64_t size, uint64_t *actual_read);
} qx_expert_cache_ops;

typedef struct {
    uint64_t requests;
    uint64_t hits;
    uint64_t misses;
    uint64_t loads;
    uint64_t evictions;
    uint64_t current_resident_packed_bytes;
    uint64_t peak_resident_packed_bytes;
    uint64_t buffered_bytes_read;
    uint64_t buffered_bytes_avoided;
} qx_expert_cache_counters;

struct qx_expert_cache_entry;
typedef struct qx_expert_cache qx_expert_cache;

typedef struct {
    const uint8_t *data;
    uint64_t size;
    struct qx_expert_cache_entry *entry;
    qx_expert_cache *owner;
} qx_expert_cache_borrow;

struct qx_expert_cache {
    uint64_t budget_bytes;
    qx_expert_cache_ops ops;
    qx_expert_cache_counters counters;
    struct qx_expert_cache_entry *lru;
    struct qx_expert_cache_entry *mru;
    uint32_t initialized;
};

/*
 * This is a per-run cache; callers own the qx_expert_cache object and there is
 * no process-global state. A successful acquire pins the returned entry.
 * borrow->data remains valid until its matching release. Runtime callers are
 * sequential and must release a borrow before their next acquire.
 */
int qx_expert_cache_init(qx_expert_cache *cache, uint64_t budget_bytes,
                         const qx_expert_cache_ops *ops,
                         char *err, uint64_t err_len);
int qx_expert_cache_acquire(qx_expert_cache *cache, qx_expert_cache_key key,
                            qx_expert_cache_borrow *borrow,
                            char *err, uint64_t err_len);
void qx_expert_cache_release(qx_expert_cache *cache,
                             qx_expert_cache_borrow *borrow);
void qx_expert_cache_snapshot(const qx_expert_cache *cache,
                              qx_expert_cache_counters *out_counters);
void qx_expert_cache_destroy(qx_expert_cache *cache);

#ifdef __cplusplus
}
#endif

#endif
