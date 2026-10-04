#ifndef QX_ROW_POOL_H
#define QX_ROW_POOL_H

#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>

typedef struct qx_row_pool qx_row_pool;
/* Zero means success. Nonzero fails the job after every range has drained. */
typedef int (*qx_row_pool_fn)(void *context, uint32_t begin, uint32_t end);

static inline void qx_row_pool_error(char *err, uint64_t len, const char *text)
{
    uint64_t i = 0;
    if (!err || !len) return;
    while (i + 1 < len && text[i]) { err[i] = text[i]; ++i; }
    err[i] = '\0';
}

#ifdef _WIN32
#include <windows.h>

/* Fault seams are only available to explicitly opted-in contract fixtures. */
#ifndef QX_ROW_POOL_TESTING
#define QX_ROW_POOL_CALLOC calloc
#define QX_ROW_POOL_FREE free
#define QX_ROW_POOL_CREATE_THREAD(fn, arg) CreateThread(NULL, 0, (fn), (arg), 0, NULL)
#define QX_ROW_POOL_CLOSE_THREAD(h) ((void)CloseHandle(h))
#else
#ifndef QX_ROW_POOL_CALLOC
#define QX_ROW_POOL_CALLOC calloc
#endif
#ifndef QX_ROW_POOL_FREE
#define QX_ROW_POOL_FREE free
#endif
#ifndef QX_ROW_POOL_CREATE_THREAD
#define QX_ROW_POOL_CREATE_THREAD(fn, arg) CreateThread(NULL, 0, (fn), (arg), 0, NULL)
#endif
#ifndef QX_ROW_POOL_CLOSE_THREAD
#define QX_ROW_POOL_CLOSE_THREAD(h) ((void)CloseHandle(h))
#endif
#endif

typedef struct qx_row_pool_worker {
    qx_row_pool *pool;
    uint32_t index;
    HANDLE thread;
} qx_row_pool_worker;

struct qx_row_pool {
    CRITICAL_SECTION lock;
    CONDITION_VARIABLE ready;
    CONDITION_VARIABLE drained;
    uint32_t workers;
    uint32_t created;
    uint32_t remaining;
    uint32_t rows;
    uint64_t generation;
    uint64_t jobs;
    int stopping;
    int failed;
    qx_row_pool_fn fn;
    void *context;
    qx_row_pool_worker worker[64];
};

static DWORD WINAPI qx_row_pool_entry(void *argument)
{
    qx_row_pool_worker *worker = (qx_row_pool_worker *)argument;
    qx_row_pool *pool = worker->pool;
    uint64_t seen = 0;
    EnterCriticalSection(&pool->lock);
    for (;;) {
        uint32_t begin, end;
        int result = 0;
        qx_row_pool_fn fn;
        void *context;
        while (!pool->stopping && pool->generation == seen)
            (void)SleepConditionVariableCS(&pool->ready, &pool->lock, INFINITE);
        if (pool->stopping) break;
        seen = pool->generation;
        /* Widen before multiplying: UINT32_MAX rows are supported. */
        begin = (uint32_t)((uint64_t)pool->rows * worker->index / pool->workers);
        end = (uint32_t)((uint64_t)pool->rows * (worker->index + 1u) / pool->workers);
        fn = pool->fn;
        context = pool->context;
        LeaveCriticalSection(&pool->lock);
        if (begin != end) result = fn(context, begin, end);
        EnterCriticalSection(&pool->lock);
        if (result != 0) pool->failed = 1;
        --pool->remaining;
        if (!pool->remaining) WakeConditionVariable(&pool->drained);
    }
    LeaveCriticalSection(&pool->lock);
    return 0;
}

/* The coordinator exclusively owns run/destroy; callbacks must not reenter it. */
static inline void qx_row_pool_destroy(qx_row_pool *pool)
{
    uint32_t i;
    if (!pool) return;
    EnterCriticalSection(&pool->lock);
    pool->stopping = 1;
    WakeAllConditionVariable(&pool->ready);
    LeaveCriticalSection(&pool->lock);
    for (i = 0; i < pool->created; ++i) {
        (void)WaitForSingleObject(pool->worker[i].thread, INFINITE);
        QX_ROW_POOL_CLOSE_THREAD(pool->worker[i].thread);
    }
    DeleteCriticalSection(&pool->lock);
    QX_ROW_POOL_FREE(pool);
}

static inline qx_row_pool *qx_row_pool_create(uint32_t workers, char *err, uint64_t err_len)
{
    qx_row_pool *pool;
    uint32_t i;
    qx_row_pool_error(err, err_len, "");
    if (workers < 2 || workers > 64) {
        qx_row_pool_error(err, err_len, "row pool workers must be 2..64");
        return NULL;
    }
    pool = (qx_row_pool *)QX_ROW_POOL_CALLOC(1, sizeof(*pool));
    if (!pool) {
        qx_row_pool_error(err, err_len, "row pool allocation failed");
        return NULL;
    }
    if (!InitializeCriticalSectionEx(&pool->lock, 0, 0)) {
        QX_ROW_POOL_FREE(pool);
        qx_row_pool_error(err, err_len, "row pool lock initialization failed");
        return NULL;
    }
    InitializeConditionVariable(&pool->ready);
    InitializeConditionVariable(&pool->drained);
    pool->workers = workers;
    for (i = 0; i < workers; ++i) {
        pool->worker[i].pool = pool;
        pool->worker[i].index = i;
        pool->worker[i].thread = QX_ROW_POOL_CREATE_THREAD(qx_row_pool_entry, &pool->worker[i]);
        if (!pool->worker[i].thread) {
            qx_row_pool_destroy(pool);
            qx_row_pool_error(err, err_len, "row pool thread creation failed");
            return NULL;
        }
        ++pool->created;
    }
    return pool;
}

static inline int qx_row_pool_run(qx_row_pool *pool, uint32_t rows, qx_row_pool_fn fn,
                                  void *context, char *err, uint64_t err_len)
{
    int failed;
    qx_row_pool_error(err, err_len, "");
    if (!pool || !rows || !fn) {
        qx_row_pool_error(err, err_len, "invalid row pool job");
        return 0;
    }
    EnterCriticalSection(&pool->lock);
    pool->rows = rows;
    pool->fn = fn;
    pool->context = context;
    pool->remaining = pool->workers;
    pool->failed = 0;
    ++pool->generation;
    ++pool->jobs;
    WakeAllConditionVariable(&pool->ready);
    while (pool->remaining)
        (void)SleepConditionVariableCS(&pool->drained, &pool->lock, INFINITE);
    failed = pool->failed;
    /* Clear borrowed pointers only after every callback has returned. */
    pool->fn = NULL;
    pool->context = NULL;
    LeaveCriticalSection(&pool->lock);
    if (failed) qx_row_pool_error(err, err_len, "row pool callback failed");
    return !failed;
}

static inline uint32_t qx_row_pool_workers(const qx_row_pool *pool)
{
    return pool ? pool->workers : 0;
}
/* Accepted jobs, including callback failures; invalid calls are not jobs. */
static inline uint64_t qx_row_pool_jobs(const qx_row_pool *pool)
{
    return pool ? pool->jobs : 0;
}

#undef QX_ROW_POOL_CALLOC
#undef QX_ROW_POOL_FREE
#undef QX_ROW_POOL_CREATE_THREAD
#undef QX_ROW_POOL_CLOSE_THREAD
#else
static inline qx_row_pool *qx_row_pool_create(uint32_t workers, char *err, uint64_t err_len)
{
    (void)workers;
    qx_row_pool_error(err, err_len, "Windows row pool unavailable on this platform");
    return NULL;
}
static inline int qx_row_pool_run(qx_row_pool *pool, uint32_t rows, qx_row_pool_fn fn,
                                  void *context, char *err, uint64_t err_len)
{
    (void)pool; (void)rows; (void)fn; (void)context;
    qx_row_pool_error(err, err_len, "Windows row pool unavailable on this platform");
    return 0;
}
static inline void qx_row_pool_destroy(qx_row_pool *pool) { (void)pool; }
static inline uint32_t qx_row_pool_workers(const qx_row_pool *pool) { (void)pool; return 0; }
static inline uint64_t qx_row_pool_jobs(const qx_row_pool *pool) { (void)pool; return 0; }
#endif
#endif
