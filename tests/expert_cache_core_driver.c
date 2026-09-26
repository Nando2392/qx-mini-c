#include "qx_expert_cache.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #condition); return 1; } } while (0)

typedef struct {
    uint8_t source[256];
    uint64_t reads;
    uint64_t actual_total;
    uint64_t allocations;
    uint64_t frees;
    uint64_t fail_alloc_call;
    uint64_t partial_bytes;
    int read_result;
    void *blocks[32];
    size_t block_sizes[32];
    uint8_t block_freed[32];
} fake_io;

static void *fake_alloc(void *opaque, size_t size) {
    fake_io *io = (fake_io *)opaque;
    void *ptr;
    io->allocations++;
    if (io->fail_alloc_call != 0u && io->allocations == io->fail_alloc_call) return NULL;
    ptr = malloc(size);
    if (ptr != NULL && io->allocations <= 32u) { io->blocks[io->allocations - 1u] = ptr; io->block_sizes[io->allocations - 1u] = size; }
    return ptr;
}

static void fake_free(void *opaque, void *ptr) {
    fake_io *io = (fake_io *)opaque;
    uint64_t i;
    if (ptr == NULL) return;
    io->frees++;
    for (i = 0u; i < 32u; ++i) if (io->blocks[i] == ptr) { memset(ptr, 0xa5, io->block_sizes[i]); io->block_freed[i] = 1u; return; }
    abort();
}

static int freed_blocks_are_poisoned(const fake_io *io) {
    uint64_t i;
    for (i = 0u; i < 32u; ++i) if (io->block_freed[i] != 0u) {
        const uint8_t *bytes = (const uint8_t *)io->blocks[i]; size_t j;
        for (j = 0u; j < io->block_sizes[i]; ++j) if (bytes[j] != 0xa5u) return 0;
    }
    return 1;
}

static void discard_blocks(fake_io *io) { uint64_t i; for (i = 0u; i < 32u; ++i) free(io->blocks[i]); }

static int fake_read(void *opaque, uint64_t offset, void *dst, uint64_t size, uint64_t *actual_read) {
    fake_io *io = (fake_io *)opaque;
    uint64_t amount = io->partial_bytes != 0u && io->partial_bytes < size ? io->partial_bytes : size;
    io->reads++;
    if (offset > sizeof(io->source) || amount > (uint64_t)sizeof(io->source) - offset) amount = 0u;
    if (amount != 0u) memcpy(dst, io->source + (size_t)offset, (size_t)amount);
    io->actual_total += amount;
    *actual_read = amount;
    return io->read_result;
}

static qx_expert_cache_ops ops_for(fake_io *io) {
    qx_expert_cache_ops ops;
    ops.context = io;
    ops.alloc = fake_alloc;
    ops.free = fake_free;
    ops.read_exact = fake_read;
    return ops;
}

static qx_expert_cache_key key(uint64_t tensor, uint64_t slice, uint64_t bytes, uint32_t expert) {
    qx_expert_cache_key result;
    result.tensor_offset = tensor;
    result.slice_offset = slice;
    result.slice_bytes = bytes;
    result.expert = expert;
    result.reserved = 0u;
    return result;
}

static void fill(fake_io *io) {
    uint32_t i;
    memset(io, 0, sizeof(*io));
    io->read_result = 1;
    for (i = 0u; i < (uint32_t)sizeof(io->source); ++i) io->source[i] = (uint8_t)i;
}

static int acquire_ok(qx_expert_cache *cache, qx_expert_cache_key k, qx_expert_cache_borrow *borrow) {
    char err[128] = {0};
    if (!qx_expert_cache_acquire(cache, k, borrow, err, sizeof(err))) {
        fprintf(stderr, "unexpected acquire error: %s\n", err);
        return 0;
    }
    return 1;
}

static int scenario_lru(void) {
    fake_io io; qx_expert_cache cache; qx_expert_cache_ops ops; qx_expert_cache_borrow borrow = {0}; qx_expert_cache_counters c; char err[128] = {0};
    qx_expert_cache_key a = key(0u, 0u, 4u, 1u), b = key(0u, 4u, 4u, 2u), d = key(0u, 8u, 4u, 3u);
    fill(&io); ops = ops_for(&io); CHECK(qx_expert_cache_init(&cache, 8u, &ops, err, sizeof(err)));
    CHECK(acquire_ok(&cache, a, &borrow)); CHECK(borrow.size == 4u && borrow.data[0] == 0u); qx_expert_cache_release(&cache, &borrow);
    CHECK(acquire_ok(&cache, a, &borrow)); qx_expert_cache_release(&cache, &borrow);
    CHECK(acquire_ok(&cache, b, &borrow)); qx_expert_cache_release(&cache, &borrow);
    CHECK(acquire_ok(&cache, a, &borrow)); qx_expert_cache_release(&cache, &borrow);
    CHECK(acquire_ok(&cache, d, &borrow)); qx_expert_cache_release(&cache, &borrow);
    CHECK(acquire_ok(&cache, b, &borrow)); qx_expert_cache_release(&cache, &borrow);
    qx_expert_cache_snapshot(&cache, &c);
    printf("requests=%llu hits=%llu misses=%llu loads=%llu evictions=%llu resident=%llu peak=%llu read=%llu avoided=%llu\n",
        (unsigned long long)c.requests, (unsigned long long)c.hits, (unsigned long long)c.misses,
        (unsigned long long)c.loads, (unsigned long long)c.evictions,
        (unsigned long long)c.current_resident_packed_bytes, (unsigned long long)c.peak_resident_packed_bytes,
        (unsigned long long)c.buffered_bytes_read, (unsigned long long)c.buffered_bytes_avoided);
    qx_expert_cache_destroy(&cache); return 0;
}

static int scenario_pinning(void) {
    fake_io io; qx_expert_cache cache; qx_expert_cache_ops ops; qx_expert_cache_borrow a = {0}, b = {0}, d = {0}; qx_expert_cache_counters c; char err[128] = {0};
    fill(&io); ops = ops_for(&io); CHECK(qx_expert_cache_init(&cache, 8u, &ops, err, sizeof(err)));
    CHECK(acquire_ok(&cache, key(0u, 0u, 4u, 1u), &a)); CHECK(acquire_ok(&cache, key(0u, 4u, 4u, 2u), &b));
    CHECK(a.data[0] == 0u); CHECK(!qx_expert_cache_acquire(&cache, key(0u, 8u, 4u, 3u), &d, err, sizeof(err)));
    CHECK(strcmp(err, "expert cache admission blocked by pinned entries") == 0); CHECK(io.allocations == 4u && io.reads == 2u);
    qx_expert_cache_snapshot(&cache, &c);
    printf("requests=%llu hits=%llu misses=%llu loads=%llu evictions=%llu resident=%llu peak=%llu read=%llu avoided=%llu\n",
        (unsigned long long)c.requests, (unsigned long long)c.hits, (unsigned long long)c.misses,
        (unsigned long long)c.loads, (unsigned long long)c.evictions,
        (unsigned long long)c.current_resident_packed_bytes, (unsigned long long)c.peak_resident_packed_bytes,
        (unsigned long long)c.buffered_bytes_read, (unsigned long long)c.buffered_bytes_avoided);
    qx_expert_cache_release(&cache, &a); qx_expert_cache_release(&cache, &b); qx_expert_cache_destroy(&cache); return 0;
}

static int scenario_failures(void) {
    fake_io io; qx_expert_cache cache; qx_expert_cache_ops ops; qx_expert_cache_borrow borrow = {0}; qx_expert_cache_counters c; char err[128] = {0}; uint64_t partial_read, truncated_read, allocation_reads;
    fill(&io); io.read_result = 0; io.partial_bytes = 2u; ops = ops_for(&io); CHECK(qx_expert_cache_init(&cache, 8u, &ops, err, sizeof(err)));
    CHECK(!qx_expert_cache_acquire(&cache, key(0u, 0u, 4u, 1u), &borrow, err, sizeof(err))); CHECK(strcmp(err, "expert cache read failed") == 0);
    qx_expert_cache_snapshot(&cache, &c); CHECK(c.loads == 0u && c.current_resident_packed_bytes == 0u); partial_read = c.buffered_bytes_read; qx_expert_cache_destroy(&cache);
    fill(&io); io.partial_bytes = 3u; ops = ops_for(&io); CHECK(qx_expert_cache_init(&cache, 8u, &ops, err, sizeof(err)));
    CHECK(!qx_expert_cache_acquire(&cache, key(0u, 0u, 4u, 1u), &borrow, err, sizeof(err))); CHECK(strcmp(err, "expert cache truncated read") == 0);
    qx_expert_cache_snapshot(&cache, &c); truncated_read = c.buffered_bytes_read; qx_expert_cache_destroy(&cache);
    fill(&io); io.fail_alloc_call = 1u; ops = ops_for(&io); CHECK(qx_expert_cache_init(&cache, 8u, &ops, err, sizeof(err)));
    CHECK(!qx_expert_cache_acquire(&cache, key(0u, 0u, 4u, 1u), &borrow, err, sizeof(err))); CHECK(io.reads == 0u); qx_expert_cache_destroy(&cache);
    fill(&io); io.fail_alloc_call = 2u; ops = ops_for(&io); CHECK(qx_expert_cache_init(&cache, 8u, &ops, err, sizeof(err)));
    CHECK(!qx_expert_cache_acquire(&cache, key(0u, 0u, 4u, 1u), &borrow, err, sizeof(err))); CHECK(io.reads == 0u && io.frees == 1u); allocation_reads = io.reads; qx_expert_cache_snapshot(&cache, &c); qx_expert_cache_destroy(&cache);
    printf("partial_read=%llu truncated_read=%llu allocation_reads=%llu resident=%llu\n", (unsigned long long)partial_read, (unsigned long long)truncated_read, (unsigned long long)allocation_reads, (unsigned long long)c.current_resident_packed_bytes); return 0;
}

static int scenario_validation(void) {
    fake_io io; qx_expert_cache cache; qx_expert_cache_ops ops; qx_expert_cache_borrow borrow = {0}; qx_expert_cache_counters c; char err[128] = {0}; uint64_t invalid_requests, oversize_requests, oversize_reads;
    fill(&io); ops = ops_for(&io); CHECK(!qx_expert_cache_init(&cache, 0u, &ops, err, sizeof(err))); CHECK(strcmp(err, "expert cache budget must be positive") == 0);
    CHECK(qx_expert_cache_init(&cache, 4u, &ops, err, sizeof(err)));
    CHECK(!qx_expert_cache_acquire(&cache, key(UINT64_MAX - 1u, 2u, 1u, 1u), &borrow, err, sizeof(err)));
    CHECK(!qx_expert_cache_acquire(&cache, key(UINT64_MAX - 2u, 1u, 2u, 1u), &borrow, err, sizeof(err)));
    { qx_expert_cache_key bad = key(0u, 0u, 1u, 1u); bad.reserved = 1u; CHECK(!qx_expert_cache_acquire(&cache, bad, &borrow, err, sizeof(err))); }
    CHECK(!qx_expert_cache_acquire(&cache, key(0u, 0u, 0u, 1u), &borrow, err, sizeof(err)));
    qx_expert_cache_snapshot(&cache, &c); invalid_requests = c.requests;
    CHECK(!qx_expert_cache_acquire(&cache, key(0u, 0u, 5u, 1u), &borrow, err, sizeof(err))); CHECK(strcmp(err, "expert cache slice exceeds budget") == 0);
    qx_expert_cache_snapshot(&cache, &c); oversize_requests = c.requests; oversize_reads = io.reads;
    qx_expert_cache_destroy(&cache); printf("invalid_requests=%llu oversize_requests=%llu oversize_reads=%llu\n", (unsigned long long)invalid_requests, (unsigned long long)oversize_requests, (unsigned long long)oversize_reads); return 0;
}

static int scenario_identity(void) {
    fake_io io; qx_expert_cache cache; qx_expert_cache_ops ops; qx_expert_cache_borrow borrow = {0}; qx_expert_cache_counters c; char err[128] = {0};
    fill(&io); ops = ops_for(&io); CHECK(qx_expert_cache_init(&cache, 8u, &ops, err, sizeof(err)));
    CHECK(acquire_ok(&cache, key(0u, 0u, 4u, 1u), &borrow)); qx_expert_cache_release(&cache, &borrow);
    CHECK(acquire_ok(&cache, key(0u, 0u, 4u, 2u), &borrow)); qx_expert_cache_release(&cache, &borrow);
    qx_expert_cache_snapshot(&cache, &c); printf("requests=%llu hits=%llu misses=%llu loads=%llu evictions=%llu resident=%llu peak=%llu read=%llu avoided=%llu\n", (unsigned long long)c.requests, (unsigned long long)c.hits, (unsigned long long)c.misses, (unsigned long long)c.loads, (unsigned long long)c.evictions, (unsigned long long)c.current_resident_packed_bytes, (unsigned long long)c.peak_resident_packed_bytes, (unsigned long long)c.buffered_bytes_read, (unsigned long long)c.buffered_bytes_avoided); qx_expert_cache_destroy(&cache); return 0;
}

static int scenario_isolation(void) {
    fake_io io1, io2; qx_expert_cache c1, c2; qx_expert_cache_ops o1, o2; qx_expert_cache_borrow b = {0}; qx_expert_cache_counters s1, s2; char err[128] = {0};
    fill(&io1); fill(&io2); o1 = ops_for(&io1); o2 = ops_for(&io2); CHECK(qx_expert_cache_init(&c1, 4u, &o1, err, sizeof(err))); CHECK(qx_expert_cache_init(&c2, 4u, &o2, err, sizeof(err)));
    CHECK(acquire_ok(&c1, key(0u, 0u, 4u, 1u), &b)); qx_expert_cache_release(&c1, &b); CHECK(acquire_ok(&c2, key(0u, 0u, 4u, 1u), &b)); qx_expert_cache_release(&c2, &b);
    qx_expert_cache_snapshot(&c1, &s1); qx_expert_cache_snapshot(&c2, &s2); printf("first_loads=%llu second_loads=%llu first_reads=%llu second_reads=%llu\n", (unsigned long long)s1.loads, (unsigned long long)s2.loads, (unsigned long long)s1.buffered_bytes_read, (unsigned long long)s2.buffered_bytes_read); qx_expert_cache_destroy(&c1); qx_expert_cache_destroy(&c2); return 0;
}

static int scenario_destroy(void) {
    fake_io io; qx_expert_cache cache; qx_expert_cache_ops ops; qx_expert_cache_borrow b = {0}; qx_expert_cache_counters c; char err[128] = {0}; uint64_t resident;
    fill(&io); ops = ops_for(&io); CHECK(qx_expert_cache_init(&cache, 8u, &ops, err, sizeof(err))); CHECK(acquire_ok(&cache, key(0u, 0u, 4u, 1u), &b)); qx_expert_cache_release(&cache, &b); CHECK(acquire_ok(&cache, key(0u, 4u, 4u, 2u), &b)); qx_expert_cache_release(&cache, &b); qx_expert_cache_snapshot(&cache, &c); resident = c.current_resident_packed_bytes; qx_expert_cache_destroy(&cache); qx_expert_cache_destroy(&cache); printf("allocations=%llu frees=%llu resident_before_destroy=%llu\n", (unsigned long long)io.allocations, (unsigned long long)io.frees, (unsigned long long)resident); return 0;
}

static int scenario_lifetime(void) {
    fake_io io; qx_expert_cache cache; qx_expert_cache_ops ops; qx_expert_cache_borrow live = {0}, forged = {0}, admitted = {0}; char err[128] = {0};
    fill(&io); ops = ops_for(&io); CHECK(qx_expert_cache_init(&cache, 4u, &ops, err, sizeof(err))); CHECK(acquire_ok(&cache, key(0u, 0u, 4u, 1u), &live));
    CHECK(!qx_expert_cache_acquire(&cache, key(0u, 4u, 4u, 2u), &admitted, err, sizeof(err)));
    forged.owner = &cache; forged.entry = (struct qx_expert_cache_entry *)(uintptr_t)1u; forged.data = (const uint8_t *)(uintptr_t)1u; forged.size = 9u;
    qx_expert_cache_release(&cache, &forged); CHECK(forged.entry == NULL && forged.owner == NULL); qx_expert_cache_release(&cache, &forged);
    qx_expert_cache_release(&cache, &live); CHECK(acquire_ok(&cache, key(0u, 4u, 4u, 2u), &admitted));
    qx_expert_cache_destroy(&cache); CHECK(io.frees == 4u && freed_blocks_are_poisoned(&io)); qx_expert_cache_release(&cache, &admitted);
    CHECK(admitted.entry == NULL && admitted.owner == NULL && admitted.data == NULL && admitted.size == 0u); CHECK(freed_blocks_are_poisoned(&io));
    printf("frees=%llu poison=1 stale_cleared=1\n", (unsigned long long)io.frees); discard_blocks(&io); return 0;
}

int main(int argc, char **argv) {
    CHECK(argc == 2);
    if (strcmp(argv[1], "lru") == 0) return scenario_lru();
    if (strcmp(argv[1], "pinning") == 0) return scenario_pinning();
    if (strcmp(argv[1], "failures") == 0) return scenario_failures();
    if (strcmp(argv[1], "validation") == 0) return scenario_validation();
    if (strcmp(argv[1], "identity") == 0) return scenario_identity();
    if (strcmp(argv[1], "isolation") == 0) return scenario_isolation();
    if (strcmp(argv[1], "destroy") == 0) return scenario_destroy();
    if (strcmp(argv[1], "lifetime") == 0) return scenario_lifetime();
    fprintf(stderr, "unknown scenario\n"); return 2;
}
