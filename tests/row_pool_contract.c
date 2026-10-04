#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "contract failure line %d: %s\n", __LINE__, #x); return 1; } } while (0)

#ifdef QX_ROW_POOL_SERIAL_BASELINE
/* Deliberately serial test-local baseline: numeric correctness is insufficient. */
typedef struct qx_row_pool { uint32_t workers; uint64_t jobs; } qx_row_pool;
typedef int (*qx_row_pool_fn)(void *, uint32_t, uint32_t);
static qx_row_pool *qx_row_pool_create(uint32_t n, char *e, uint64_t z) {
    qx_row_pool *p; (void)e; (void)z;
    if (n < 2 || n > 64) return NULL;
    p = calloc(1, sizeof(*p)); if (p) p->workers = n; return p;
}
static int qx_row_pool_run(qx_row_pool *p, uint32_t n, qx_row_pool_fn f, void *c, char *e, uint64_t z) {
    (void)e; (void)z; if (!p || !n || !f) return 0;
    ++p->jobs; return f(c, 0, n) == 0;
}
static void qx_row_pool_destroy(qx_row_pool *p) { free(p); }
static uint32_t qx_row_pool_workers(const qx_row_pool *p) { return p ? p->workers : 0; }
static uint64_t qx_row_pool_jobs(const qx_row_pool *p) { return p ? p->jobs : 0; }
#else
static int allocation_calls, fail_allocation, thread_calls, fail_thread;
static LONG allocations_live, handles_live, threads_live;
static void *test_calloc(size_t n, size_t s) {
    void *p; ++allocation_calls;
    if (allocation_calls == fail_allocation) return NULL;
    p = calloc(n, s); if (p) InterlockedIncrement(&allocations_live); return p;
}
static void test_free(void *p) { if (p) InterlockedDecrement(&allocations_live); free(p); }
/* Wrapping thread entry also proves partial-create shutdown joins live threads. */
typedef struct start_context { LPTHREAD_START_ROUTINE fn; void *arg; } start_context;
static DWORD WINAPI test_entry(void *arg) {
    start_context *s = arg; LPTHREAD_START_ROUTINE fn = s->fn; void *context = s->arg;
    DWORD result; free(s); InterlockedIncrement(&threads_live);
    result = fn(context); InterlockedDecrement(&threads_live); return result;
}
static HANDLE test_create_thread(LPTHREAD_START_ROUTINE fn, void *arg) {
    start_context *s; HANDLE h; ++thread_calls;
    if (thread_calls == fail_thread) return NULL;
    s = malloc(sizeof(*s)); if (!s) return NULL; s->fn = fn; s->arg = arg;
    h = CreateThread(NULL, 0, test_entry, s, 0, NULL);
    if (!h) free(s); else InterlockedIncrement(&handles_live); return h;
}
static void test_close(HANDLE h) { if (CloseHandle(h)) InterlockedDecrement(&handles_live); }
#define QX_ROW_POOL_TESTING
#define QX_ROW_POOL_CALLOC test_calloc
#define QX_ROW_POOL_FREE test_free
#define QX_ROW_POOL_CREATE_THREAD test_create_thread
#define QX_ROW_POOL_CLOSE_THREAD test_close
#include "qx_row_pool.h"
#include "qx_row_pool.h"
#endif

typedef struct work {
    uint32_t rows;
    DWORD coordinator;
    DWORD ids[257];
    LONG visits[257];
    uint64_t values[257];
    LONG callbacks;
    LONG finished;
    int fail;
    HANDLE gate;
} work;
static uint64_t reference(uint32_t row) {
    return ((uint64_t)row + 3) * 304u;
}
static int calculate(void *context, uint32_t begin, uint32_t end) {
    work *w = context; uint32_t row;
    InterlockedIncrement(&w->callbacks);
    for (row = begin; row < end; ++row) {
        uint32_t col;
        uint64_t sum = 0;
        w->ids[row] = GetCurrentThreadId();
        for (col = 0; col < 19; ++col) sum += ((uint64_t)row + 3) * (col + 7);
        w->values[row] = sum; InterlockedIncrement(&w->visits[row]);
    }
    if (w->fail && begin == 0) { SetEvent(w->gate); InterlockedIncrement(&w->finished); return 7; }
    if (w->fail) { if (WaitForSingleObject(w->gate, 5000) != WAIT_OBJECT_0) return 9; Sleep(20); }
    InterlockedIncrement(&w->finished); return 0;
}
static int numeric(void) {
    char err[128]; qx_row_pool *p = qx_row_pool_create(4, err, sizeof(err));
    work w; DWORD first[257]; uint32_t i, job;
    CHECK(p); CHECK(qx_row_pool_workers(p) == 4); CHECK(qx_row_pool_jobs(p) == 0);
    for (job = 0; job < 100; ++job) {
        memset(&w, 0, sizeof(w)); w.rows = 257; w.coordinator = GetCurrentThreadId();
        CHECK(qx_row_pool_run(p, w.rows, calculate, &w, err, sizeof(err)));
        CHECK(w.callbacks == 4); CHECK(w.finished == 4);
        for (i = 0; i < w.rows; ++i) {
            CHECK(w.visits[i] == 1); CHECK(w.values[i] == reference(i));
            CHECK(w.ids[i] != w.coordinator);
            if (job == 0) first[i] = w.ids[i]; else CHECK(first[i] == w.ids[i]);
        }
        CHECK(w.ids[0] != w.ids[64]); CHECK(w.ids[64] != w.ids[128]);
        CHECK(w.ids[128] != w.ids[192]); CHECK(w.ids[0] != w.ids[192]);
    }
    CHECK(qx_row_pool_jobs(p) == 100); qx_row_pool_destroy(p);
    puts("numeric persistent 100 jobs OK"); return 0;
}
static int boundaries(void) {
    uint32_t n, rows, i; char err[128];
    CHECK(!qx_row_pool_create(0, err, sizeof(err))); CHECK(!qx_row_pool_create(1, err, sizeof(err)));
    CHECK(!qx_row_pool_create(65, err, sizeof(err)));
    CHECK(!qx_row_pool_workers(NULL)); CHECK(!qx_row_pool_jobs(NULL)); qx_row_pool_destroy(NULL);
    CHECK(!qx_row_pool_run(NULL, 1, calculate, NULL, err, sizeof(err)));
    for (n = 2; n <= 64; n += 62) {
        qx_row_pool *p = qx_row_pool_create(n, NULL, 0); CHECK(p);
        CHECK(!qx_row_pool_run(p, 0, calculate, NULL, err, sizeof(err))); CHECK(err[0]);
        CHECK(!qx_row_pool_run(p, 1, NULL, NULL, err, sizeof(err)));
        CHECK(qx_row_pool_jobs(p) == 0);
        for (rows = 1; rows <= 67; ++rows) {
            work w; memset(&w, 0, sizeof(w));
            CHECK(qx_row_pool_run(p, rows, calculate, &w, NULL, 0));
            CHECK(w.callbacks == (LONG)(rows < n ? rows : n));
            for (i = 0; i < rows; ++i) CHECK(w.visits[i] == 1);
        }
        qx_row_pool_destroy(p);
    }
    puts("boundaries workers 2/64 rows 1..67 OK"); return 0;
}
typedef struct partition_work {
    LONG count;
    uint32_t begin[64], end[64];
} partition_work;
static int capture_partition(void *context, uint32_t begin, uint32_t end) {
    partition_work *w = context;
    LONG slot = InterlockedIncrement(&w->count) - 1;
    if (slot >= 64 || begin >= end) return 1;
    w->begin[slot] = begin; w->end[slot] = end; return 0;
}
static int partition(void) {
    char err[128]; partition_work w; uint32_t i, k;
    qx_row_pool *a = qx_row_pool_create(3, err, sizeof(err));
    qx_row_pool *b = qx_row_pool_create(7, err, sizeof(err)); CHECK(a); CHECK(b);
    memset(&w, 0, sizeof(w));
    CHECK(qx_row_pool_run(a, UINT32_MAX, capture_partition, &w, err, sizeof(err)));
    CHECK(w.count == 3);
    for (i = 0; i < 3; ++i) {
        uint32_t begin = (uint32_t)((uint64_t)UINT32_MAX * i / 3);
        uint32_t end = (uint32_t)((uint64_t)UINT32_MAX * (i + 1u) / 3);
        int found = 0;
        for (k = 0; k < 3; ++k) if (w.begin[k] == begin && w.end[k] == end) ++found;
        CHECK(found == 1);
    }
    CHECK(qx_row_pool_jobs(a) == 1); CHECK(qx_row_pool_jobs(b) == 0);
    qx_row_pool_destroy(a); memset(&w, 0, sizeof(w));
    CHECK(qx_row_pool_run(b, UINT32_MAX, capture_partition, &w, err, sizeof(err)));
    CHECK(w.count == 7); CHECK(qx_row_pool_jobs(b) == 1); qx_row_pool_destroy(b);
    puts("UINT32_MAX partition and independent pool lifetimes OK"); return 0;
}
static int drain(void) {
    char err[128]; work w; qx_row_pool *p = qx_row_pool_create(4, err, sizeof(err)); CHECK(p);
    memset(&w, 0, sizeof(w)); w.fail = 1; w.gate = CreateEvent(NULL, TRUE, FALSE, NULL); CHECK(w.gate);
    CHECK(!qx_row_pool_run(p, 257, calculate, &w, err, sizeof(err)));
    CHECK(w.finished == 4); CHECK(w.callbacks == 4); CHECK(err[0]); CHECK(qx_row_pool_jobs(p) == 1);
    CloseHandle(w.gate); memset(&w, 0, sizeof(w));
    CHECK(qx_row_pool_run(p, 257, calculate, &w, err, sizeof(err))); CHECK(w.finished == 4);
    qx_row_pool_destroy(p); puts("callback failure drained and reusable OK"); return 0;
}
#ifndef QX_ROW_POOL_SERIAL_BASELINE
static int faults(void) {
    int i; char err[128]; qx_row_pool *p;
    allocation_calls = 0; fail_allocation = 1;
    CHECK(!qx_row_pool_create(4, err, sizeof(err))); CHECK(err[0]); CHECK(!allocations_live);
    fail_allocation = 0;
    for (i = 1; i <= 4; ++i) {
        thread_calls = 0; fail_thread = i;
        p = qx_row_pool_create(4, err, sizeof(err)); CHECK(!p); CHECK(err[0]);
        CHECK(!allocations_live); CHECK(!handles_live); CHECK(!threads_live);
    }
    fail_thread = 0; thread_calls = 0;
    p = qx_row_pool_create(4, err, sizeof(err)); CHECK(p); CHECK(thread_calls == 4);
    { work w; memset(&w, 0, sizeof(w)); CHECK(qx_row_pool_run(p, 257, calculate, &w, err, sizeof(err))); }
    CHECK(thread_calls == 4); qx_row_pool_destroy(p);
    CHECK(!allocations_live); CHECK(!handles_live); CHECK(!threads_live);
    puts("allocation and thread faults 1..4 cleaned; no per-job spawn OK"); return 0;
}
#endif
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    if (!strcmp(argv[1], "numeric")) return numeric();
    if (!strcmp(argv[1], "boundaries")) return boundaries();
    if (!strcmp(argv[1], "drain")) return drain();
    if (!strcmp(argv[1], "partition")) return partition();
#ifndef QX_ROW_POOL_SERIAL_BASELINE
    if (!strcmp(argv[1], "faults")) return faults();
#endif
    return 2;
}
