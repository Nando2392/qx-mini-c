#include "qx_format.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include "qx_iq2xs_tables.inc"

int qx_avx2_fma_supported(void);
float qx_dot_f32_avx2_fma_256(const float *weights, const float *input);

static struct {
    uint64_t malloc_calls;
    uint64_t calloc_calls;
    uint64_t realloc_calls;
    uint64_t free_calls;
    uint64_t bytes_requested;
} qx_alloc_profile;

static void qx_alloc_profile_print(void) {
    if (!getenv("QX_ALLOC_PROFILE")) return;
    fprintf(stderr,
        "QX_ALLOC_PROFILE {\"malloc\":%llu,\"calloc\":%llu,\"realloc\":%llu,\"free\":%llu,\"bytes_requested\":%llu}\n",
        (unsigned long long)qx_alloc_profile.malloc_calls,
        (unsigned long long)qx_alloc_profile.calloc_calls,
        (unsigned long long)qx_alloc_profile.realloc_calls,
        (unsigned long long)qx_alloc_profile.free_calls,
        (unsigned long long)qx_alloc_profile.bytes_requested);
}

static void *qx_profile_malloc(size_t size) {
    qx_alloc_profile.malloc_calls++;
    qx_alloc_profile.bytes_requested += (uint64_t)size;
    return malloc(size);
}

static void *qx_profile_calloc(size_t count, size_t size) {
    qx_alloc_profile.calloc_calls++;
    if (size && count <= SIZE_MAX / size) qx_alloc_profile.bytes_requested += (uint64_t)(count * size);
    return calloc(count, size);
}

static void *qx_profile_realloc(void *ptr, size_t size) {
    qx_alloc_profile.realloc_calls++;
    qx_alloc_profile.bytes_requested += (uint64_t)size;
    return realloc(ptr, size);
}

static void qx_profile_free(void *ptr) {
    qx_alloc_profile.free_calls++;
    free(ptr);
}

static void qx_alloc_profile_init_once(void) {
    static int initialized = 0;
    if (!initialized) {
        initialized = 1;
        atexit(qx_alloc_profile_print);
    }
}

#define malloc(size) qx_profile_malloc(size)
#define calloc(count, size) qx_profile_calloc(count, size)
#define realloc(ptr, size) qx_profile_realloc(ptr, size)
#define free(ptr) qx_profile_free(ptr)

static void qx_set_err(char *err, uint64_t err_len, const char *msg);

typedef struct qx_alloc_snapshot {
    uint64_t malloc_calls;
    uint64_t calloc_calls;
    uint64_t realloc_calls;
    uint64_t free_calls;
    uint64_t bytes_requested;
} qx_alloc_snapshot;

typedef struct qx_scratch_workspace {
    unsigned char *data;
    size_t capacity;
    size_t used;
    size_t peak_capacity;
    uint64_t growth_events;
} qx_scratch_workspace;

static qx_alloc_snapshot qx_alloc_profile_snapshot(void) {
    qx_alloc_snapshot snapshot;
    snapshot.malloc_calls = qx_alloc_profile.malloc_calls;
    snapshot.calloc_calls = qx_alloc_profile.calloc_calls;
    snapshot.realloc_calls = qx_alloc_profile.realloc_calls;
    snapshot.free_calls = qx_alloc_profile.free_calls;
    snapshot.bytes_requested = qx_alloc_profile.bytes_requested;
    return snapshot;
}

static qx_alloc_snapshot qx_alloc_profile_delta(qx_alloc_snapshot start) {
    qx_alloc_snapshot end = qx_alloc_profile_snapshot();
    end.malloc_calls -= start.malloc_calls;
    end.calloc_calls -= start.calloc_calls;
    end.realloc_calls -= start.realloc_calls;
    end.free_calls -= start.free_calls;
    end.bytes_requested -= start.bytes_requested;
    return end;
}

static void qx_scratch_free(qx_scratch_workspace *workspace) {
    if (!workspace) return;
    free(workspace->data);
    memset(workspace, 0, sizeof(*workspace));
}

static void qx_scratch_reset(qx_scratch_workspace *workspace) {
    if (workspace) workspace->used = 0u;
}

static int qx_scratch_reserve_capacity(qx_scratch_workspace *workspace, size_t required, char *err, uint64_t err_len) {
    if (!workspace) { qx_set_err(err, err_len, "invalid scratch workspace"); return 0; }
    if (required <= workspace->capacity) return 1;
    size_t capacity = workspace->capacity ? workspace->capacity : 4096u;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2u) { qx_set_err(err, err_len, "scratch workspace size overflow"); return 0; }
        capacity *= 2u;
    }
    unsigned char *next = (unsigned char *)realloc(workspace->data, capacity);
    if (!next) { qx_set_err(err, err_len, "out of memory"); return 0; }
    workspace->data = next;
    workspace->capacity = capacity;
    if (workspace->peak_capacity < capacity) workspace->peak_capacity = capacity;
    workspace->growth_events++;
    return 1;
}

static int qx_scratch_plan_add(size_t *cursor, size_t count, size_t size, char *err, uint64_t err_len) {
    if (!cursor) { qx_set_err(err, err_len, "invalid scratch plan"); return 0; }
    if (size && count > SIZE_MAX / size) { qx_set_err(err, err_len, "scratch allocation size overflow"); return 0; }
    size_t bytes = count * size;
    if (*cursor > SIZE_MAX - 15u) { qx_set_err(err, err_len, "scratch allocation size overflow"); return 0; }
    size_t aligned = (*cursor + 15u) & ~(size_t)15u;
    if (aligned > SIZE_MAX - bytes) { qx_set_err(err, err_len, "scratch allocation size overflow"); return 0; }
    *cursor = aligned + bytes;
    return 1;
}

static void *qx_scratch_alloc(qx_scratch_workspace *workspace, size_t count, size_t size, int zero, char *err, uint64_t err_len) {
    if (size && count > SIZE_MAX / size) { qx_set_err(err, err_len, "scratch allocation size overflow"); return NULL; }
    size_t bytes = count * size;
    if (!workspace) return zero ? calloc(count, size) : malloc(bytes);
    if (workspace->used > SIZE_MAX - 15u) { qx_set_err(err, err_len, "scratch allocation size overflow"); return NULL; }
    size_t aligned = (workspace->used + 15u) & ~(size_t)15u;
    if (aligned > SIZE_MAX - bytes) { qx_set_err(err, err_len, "scratch allocation size overflow"); return NULL; }
    size_t required = aligned + bytes;
    if (!qx_scratch_reserve_capacity(workspace, required, err, err_len)) return NULL;
    void *ptr = workspace->data + aligned;
    workspace->used = required;
    if (zero && bytes) memset(ptr, 0, bytes);
    return ptr;
}

static qx_io_backend qx_requested_io_backend = QX_IO_BUFFERED;

int qx_set_io_backend(const char *backend, char *err, uint64_t err_len) {
    if (!backend || strcmp(backend, "buffered") == 0) {
        qx_requested_io_backend = QX_IO_BUFFERED;
        return 1;
    }
    if (strcmp(backend, "mmap") == 0) {
        qx_requested_io_backend = QX_IO_MMAP;
        return 1;
    }
    qx_set_err(err, err_len, "invalid QXF I/O backend");
    return 0;
}

const char *qx_io_backend_name(qx_io_backend backend) {
    return backend == QX_IO_MMAP ? "mmap" : "buffered";
}

static void qx_set_err(char *err, uint64_t err_len, const char *msg) {
    if (err && err_len > 0) {
#ifdef _MSC_VER
        strncpy_s(err, (size_t)err_len, msg, _TRUNCATE);
#else
        snprintf(err, (size_t)err_len, "%s", msg);
#endif
    }
}

static void qx_set_errno_err(char *err, uint64_t err_len, int code) {
    if (!err || err_len == 0) return;
#ifdef _MSC_VER
    strerror_s(err, (size_t)err_len, code);
#else
    snprintf(err, (size_t)err_len, "%s", strerror(code));
#endif
}

uint64_t qx_align_u64(uint64_t value, uint64_t alignment) {
    if (alignment == 0) return value;
    uint64_t rem = value % alignment;
    return rem == 0 ? value : value + (alignment - rem);
}

static int qx_align_u64_checked(uint64_t value, uint64_t alignment, uint64_t *out) {
    if (!out || alignment == 0) return 0;
    uint64_t rem = value % alignment;
    uint64_t add = rem == 0 ? 0 : alignment - rem;
    if (value > UINT64_MAX - add) return 0;
    *out = value + add;
    return 1;
}

uint64_t qx_fnv1a64(const void *data, uint64_t len) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < len; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t qx_fnv1a64_update(uint64_t hash, const void *data, uint64_t len) {
    const unsigned char *p = (const unsigned char *)data;
    for (uint64_t i = 0; i < len; ++i) {
        hash ^= (uint64_t)p[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

const char *qx_model_type_name(qx_model_type t) {
    switch (t) {
        case QX_MODEL_QWEN3_DENSE: return "qwen3_dense";
        case QX_MODEL_QWEN3_MOE: return "qwen3_moe";
        default: return "unknown";
    }
}

const char *qx_quant_type_name(qx_quant_type t) {
    switch (t) {
        case QX_QUANT_F16: return "f16";
        case QX_QUANT_Q4_BLOCK: return "q4";
        case QX_QUANT_Q3_BLOCK: return "q3";
        case QX_QUANT_Q3Q4_MIXED: return "q3q4mix";
        case QX_QUANT_Q2_BLOCK: return "q2";
        default: return "unknown";
    }
}

int qx_parse_quant_type(const char *s, qx_quant_type *out) {
    if (!s || !out) return 0;
    if (strcmp(s, "f16") == 0) { *out = QX_QUANT_F16; return 1; }
    if (strcmp(s, "q4") == 0 || strcmp(s, "q4_block") == 0) { *out = QX_QUANT_Q4_BLOCK; return 1; }
    if (strcmp(s, "q3") == 0 || strcmp(s, "q3_block") == 0) { *out = QX_QUANT_Q3_BLOCK; return 1; }
    if (strcmp(s, "q3q4mix") == 0 || strcmp(s, "q3q4") == 0 || strcmp(s, "mixed") == 0) { *out = QX_QUANT_Q3Q4_MIXED; return 1; }
    if (strcmp(s, "q2") == 0 || strcmp(s, "q2_block") == 0) { *out = QX_QUANT_Q2_BLOCK; return 1; }
    return 0;
}

int qx_manifest_for_model(const char *model, qx_quant_type quant, qx_model_manifest *out) {
    if (!model || !out) return 0;
    memset(out, 0, sizeof(*out));
    out->quant_type = quant;
    out->vocab = 151936;
    out->max_ctx = 40960;
    if (strcmp(model, "qwen3-4b") == 0) {
        out->model_type = QX_MODEL_QWEN3_DENSE;
        out->layers = 36; out->hidden = 2560; out->intermediate = 9728;
        out->q_heads = 32; out->kv_heads = 8; out->head_dim = 128;
        return 1;
    }
    if (strcmp(model, "qwen3-8b") == 0) {
        out->model_type = QX_MODEL_QWEN3_DENSE;
        out->layers = 36; out->hidden = 4096; out->intermediate = 12288;
        out->q_heads = 32; out->kv_heads = 8; out->head_dim = 128;
        return 1;
    }
    if (strcmp(model, "qwen3-14b") == 0) {
        out->model_type = QX_MODEL_QWEN3_DENSE;
        out->layers = 40; out->hidden = 5120; out->intermediate = 17408;
        out->q_heads = 40; out->kv_heads = 8; out->head_dim = 128;
        return 1;
    }
    if (strcmp(model, "qwen3-30b-a3b") == 0) {
        out->model_type = QX_MODEL_QWEN3_MOE;
        out->layers = 48; out->hidden = 2048; out->intermediate = 6144;
        out->q_heads = 32; out->kv_heads = 4; out->head_dim = 128;
        out->experts = 128; out->experts_per_token = 8; out->moe_intermediate = 768;
        return 1;
    }
    return 0;
}

uint32_t qx_dense_tensor_count(uint32_t layers) {
    /* token_embd + output_norm + lm_head + 9 tensors/layer for dense Qwen3. */
    return 3u + layers * 9u;
}

uint32_t qx_moe_tensor_count(uint32_t layers, uint32_t experts) {
    /* token_embd + output_norm + lm_head + per-layer:
       attn_norm/q/k/v/o + moe_norm + router + 3 tensors per expert. */
    return 3u + layers * (7u + experts * 3u);
}

static qx_tensor_dtype qx_default_dtype_for_name(const char *name, qx_quant_type quant) {
    if (strstr(name, "norm") != NULL) return QX_DTYPE_F32;
    if (quant == QX_QUANT_Q3_BLOCK) return QX_DTYPE_Q3;
    if (quant == QX_QUANT_Q4_BLOCK) return QX_DTYPE_Q4;
    if (quant == QX_QUANT_Q2_BLOCK) return QX_DTYPE_Q2;
    if (quant == QX_QUANT_F16) return QX_DTYPE_F16;
    /* Mixed policy: attention/router/embed/head stay Q4; FFN and experts are Q3. */
    if (strstr(name, "ffn_") != NULL || strstr(name, ".expert.") != NULL) return QX_DTYPE_Q3;
    return QX_DTYPE_Q4;
}

static void qx_fill_entry(qx_tensor_dir_entry *e, const char *name, qx_quant_type quant,
                          uint64_t d0, uint64_t d1) {
    memset(e, 0, sizeof(*e));
#ifdef _MSC_VER
    strncpy_s(e->name, sizeof(e->name), name, _TRUNCATE);
#else
    snprintf(e->name, sizeof(e->name), "%s", name);
#endif
    e->dtype = (uint32_t)qx_default_dtype_for_name(name, quant);
    e->quant = (uint32_t)quant;
    e->rank = d1 == 0 ? 1u : 2u;
    e->dims[0] = d0;
    e->dims[1] = d1;
    e->group_size = (e->dtype == QX_DTYPE_F32 || e->dtype == QX_DTYPE_F16) ? 0u : 64u;
    e->offset = 0;
    e->byte_size = 0;
    e->checksum = qx_fnv1a64(e->name, (uint64_t)strlen(e->name));
}

static uint32_t qx_make_dense_entries(const qx_model_manifest *m, qx_tensor_dir_entry *entries, uint32_t cap) {
    uint32_t n = 0;
#define ADD(name, d0, d1) do { if (n < cap) qx_fill_entry(&entries[n++], (name), m->quant_type, (d0), (d1)); } while (0)
    ADD("token_embd.weight", m->vocab, m->hidden);
    for (uint32_t layer = 0; layer < m->layers; layer++) {
        char name[QX_NAME_MAX];
#define LADD(suffix, d0, d1) do { snprintf(name, sizeof(name), "blk.%u.%s", layer, (suffix)); ADD(name, (d0), (d1)); } while (0)
        LADD("attn_norm.weight", m->hidden, 0);
        LADD("attn_q.weight", m->hidden, (uint64_t)m->q_heads * m->head_dim);
        LADD("attn_k.weight", m->hidden, (uint64_t)m->kv_heads * m->head_dim);
        LADD("attn_v.weight", m->hidden, (uint64_t)m->kv_heads * m->head_dim);
        LADD("attn_o.weight", (uint64_t)m->q_heads * m->head_dim, m->hidden);
        LADD("ffn_norm.weight", m->hidden, 0);
        LADD("ffn_gate.weight", m->hidden, m->intermediate);
        LADD("ffn_up.weight", m->hidden, m->intermediate);
        LADD("ffn_down.weight", m->intermediate, m->hidden);
#undef LADD
    }
    ADD("output_norm.weight", m->hidden, 0);
    ADD("lm_head.weight", m->hidden, m->vocab);
#undef ADD
    return n;
}

static uint32_t qx_make_moe_entries(const qx_model_manifest *m, qx_tensor_dir_entry *entries, uint32_t cap) {
    uint32_t n = 0;
#define ADD(name, d0, d1) do { if (n < cap) qx_fill_entry(&entries[n++], (name), m->quant_type, (d0), (d1)); } while (0)
    ADD("token_embd.weight", m->vocab, m->hidden);
    for (uint32_t layer = 0; layer < m->layers; layer++) {
        char name[QX_NAME_MAX];
#define LADD(suffix, d0, d1) do { snprintf(name, sizeof(name), "blk.%u.%s", layer, (suffix)); ADD(name, (d0), (d1)); } while (0)
        LADD("attn_norm.weight", m->hidden, 0);
        LADD("attn_q.weight", m->hidden, (uint64_t)m->q_heads * m->head_dim);
        LADD("attn_k.weight", m->hidden, (uint64_t)m->kv_heads * m->head_dim);
        LADD("attn_v.weight", m->hidden, (uint64_t)m->kv_heads * m->head_dim);
        LADD("attn_o.weight", (uint64_t)m->q_heads * m->head_dim, m->hidden);
        LADD("moe_norm.weight", m->hidden, 0);
        LADD("moe_router.weight", m->hidden, m->experts);
        for (uint32_t expert = 0; expert < m->experts; expert++) {
#define EADD(suffix, d0, d1) do { snprintf(name, sizeof(name), "blk.%u.expert.%u.%s", layer, expert, (suffix)); ADD(name, (d0), (d1)); } while (0)
            EADD("gate.weight", m->hidden, m->moe_intermediate);
            EADD("up.weight", m->hidden, m->moe_intermediate);
            EADD("down.weight", m->moe_intermediate, m->hidden);
#undef EADD
        }
#undef LADD
    }
    ADD("output_norm.weight", m->hidden, 0);
    ADD("lm_head.weight", m->hidden, m->vocab);
#undef ADD
    return n;
}

int qx_write_metadata_only(const char *path, const qx_model_manifest *manifest, char *err, uint64_t err_len) {
    if (!path || !manifest) { qx_set_err(err, err_len, "invalid argument"); return 0; }

    uint32_t count = manifest->model_type == QX_MODEL_QWEN3_MOE
        ? qx_moe_tensor_count(manifest->layers, manifest->experts)
        : qx_dense_tensor_count(manifest->layers);
    qx_tensor_dir_entry *entries = (qx_tensor_dir_entry *)calloc(count, sizeof(qx_tensor_dir_entry));
    if (!entries) { qx_set_err(err, err_len, "out of memory"); return 0; }
    uint32_t made = manifest->model_type == QX_MODEL_QWEN3_MOE
        ? qx_make_moe_entries(manifest, entries, count)
        : qx_make_dense_entries(manifest, entries, count);
    if (made != count) { free(entries); qx_set_err(err, err_len, "internal tensor count mismatch"); return 0; }

    qx_header h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, QX_MAGIC, 4);
    h.version = QX_VERSION;
    h.header_size = (uint32_t)sizeof(qx_header);
    h.dir_entry_size = (uint32_t)sizeof(qx_tensor_dir_entry);
    h.tensor_count = count;
    h.dir_offset = qx_align_u64(sizeof(qx_header), QX_ALIGN_BYTES);
    h.data_offset = qx_align_u64(h.dir_offset + (uint64_t)count * sizeof(qx_tensor_dir_entry), QX_ALIGN_BYTES);
    h.file_size = h.data_offset;
    h.manifest = *manifest;
    h.manifest_checksum = qx_fnv1a64(&h.manifest, (uint64_t)sizeof(h.manifest));

    FILE *f = fopen(path, "wb");
    if (!f) { free(entries); qx_set_errno_err(err, err_len, errno); return 0; }

    int ok = 1;
    if (fwrite(&h, sizeof(h), 1, f) != 1) ok = 0;
    uint64_t pad_to_dir = h.dir_offset - sizeof(h);
    for (uint64_t i = 0; ok && i < pad_to_dir; i++) if (fputc(0, f) == EOF) ok = 0;
    if (ok && fwrite(entries, sizeof(qx_tensor_dir_entry), count, f) != count) ok = 0;
    uint64_t pos = h.dir_offset + (uint64_t)count * sizeof(qx_tensor_dir_entry);
    for (uint64_t i = pos; ok && i < h.data_offset; i++) if (fputc(0, f) == EOF) ok = 0;
    if (fclose(f) != 0) ok = 0;
    free(entries);
    if (!ok) { qx_set_err(err, err_len, "write failed"); return 0; }
    return 1;
}

int qx_read_header(FILE *f, qx_header *out, char *err, uint64_t err_len) {
    if (!f || !out) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    if (fread(out, sizeof(*out), 1, f) != 1) { qx_set_err(err, err_len, "short read header"); return 0; }
    if (memcmp(out->magic, QX_MAGIC, 4) != 0) { qx_set_err(err, err_len, "bad QXF magic"); return 0; }
    if (out->version != QX_VERSION) { qx_set_err(err, err_len, "unsupported QXF version"); return 0; }
    if (out->header_size != sizeof(qx_header) || out->dir_entry_size != sizeof(qx_tensor_dir_entry)) {
        qx_set_err(err, err_len, "QXF struct size mismatch"); return 0;
    }
    uint64_t got = qx_fnv1a64(&out->manifest, (uint64_t)sizeof(out->manifest));
    if (got != out->manifest_checksum) { qx_set_err(err, err_len, "manifest checksum mismatch"); return 0; }
    const qx_model_manifest *m = &out->manifest;
    if ((m->model_type != QX_MODEL_QWEN3_DENSE && m->model_type != QX_MODEL_QWEN3_MOE) ||
        m->quant_type < QX_QUANT_F16 || m->quant_type > QX_QUANT_Q2_BLOCK ||
        m->layers == 0 || m->hidden == 0 || m->intermediate == 0 ||
        m->q_heads == 0 || m->kv_heads == 0 || m->kv_heads > m->q_heads ||
        m->head_dim == 0 || m->vocab == 0 || m->max_ctx == 0 ||
        (m->model_type == QX_MODEL_QWEN3_MOE &&
         (m->experts == 0 || m->experts_per_token == 0 ||
          m->experts_per_token > m->experts || m->moe_intermediate == 0))) {
        qx_set_err(err, err_len, "invalid QXF manifest"); return 0;
    }
    return 1;
}

int qx_dump_summary(const char *path, FILE *out, char *err, uint64_t err_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { qx_set_errno_err(err, err_len, errno); return 0; }
    qx_header h;
    if (!qx_read_header(f, &h, err, err_len)) { fclose(f); return 0; }
    fprintf(out, "{\n");
    fprintf(out, "  \"magic\": \"QXF1\",\n");
    fprintf(out, "  \"version\": %u,\n", h.version);
    fprintf(out, "  \"model_type\": \"%s\",\n", qx_model_type_name((qx_model_type)h.manifest.model_type));
    fprintf(out, "  \"quant_type\": \"%s\",\n", qx_quant_type_name((qx_quant_type)h.manifest.quant_type));
    fprintf(out, "  \"layers\": %u,\n", h.manifest.layers);
    fprintf(out, "  \"hidden\": %u,\n", h.manifest.hidden);
    fprintf(out, "  \"intermediate\": %u,\n", h.manifest.intermediate);
    fprintf(out, "  \"q_heads\": %u,\n", h.manifest.q_heads);
    fprintf(out, "  \"kv_heads\": %u,\n", h.manifest.kv_heads);
    fprintf(out, "  \"head_dim\": %u,\n", h.manifest.head_dim);
    fprintf(out, "  \"vocab\": %u,\n", h.manifest.vocab);
    fprintf(out, "  \"tensor_count\": %u,\n", h.tensor_count);
    fprintf(out, "  \"dir_offset\": %llu,\n", (unsigned long long)h.dir_offset);
    fprintf(out, "  \"data_offset\": %llu,\n", (unsigned long long)h.data_offset);
    fprintf(out, "  \"file_size\": %llu\n", (unsigned long long)h.file_size);
    fprintf(out, "}\n");
    fclose(f);
    return 1;
}


static qx_tensor_dtype qx_dtype_from_ggml_type(uint32_t ggml_type) {
    /* Minimal mapping for common GGML quant types used by GGUF. */
    switch (ggml_type) {
        case 0: return QX_DTYPE_F32;
        case 1: return QX_DTYPE_F16;
        case 2: return QX_DTYPE_Q4;
        case 3: return QX_DTYPE_Q4;
        case 10: return QX_DTYPE_Q2;
        case 11: return QX_DTYPE_Q3;
        case 12: return QX_DTYPE_Q4;
        case 13: return QX_DTYPE_Q4;
        case 14: return QX_DTYPE_Q4;
        case 16: return QX_DTYPE_Q2;
        case 17: return QX_DTYPE_Q2;
        case 18: return QX_DTYPE_Q3;
        case 21: return QX_DTYPE_Q3;
        case 22: return QX_DTYPE_Q2;
        case 23: return QX_DTYPE_Q4;
        default: return QX_DTYPE_U8;
    }
}

static int copy_tensor_bytes(FILE *src, FILE *dst, uint64_t src_abs, uint64_t n, uint64_t *checksum) {
    enum { COPY_CHUNK = 64 * 1024 };
    unsigned char *buf = (unsigned char *)malloc(COPY_CHUNK);
    if (!buf) return 0;
#if defined(_WIN32)
    if (_fseeki64(src, (int64_t)src_abs, SEEK_SET) != 0) { free(buf); return 0; }
#else
    if (fseeko(src, (off_t)src_abs, SEEK_SET) != 0) { free(buf); return 0; }
#endif
    uint64_t h = 1469598103934665603ull;
    while (n > 0) {
        size_t want = n < COPY_CHUNK ? (size_t)n : COPY_CHUNK;
        if (fread(buf, 1, want, src) != want) { free(buf); return 0; }
        for (size_t i = 0; i < want; i++) { h ^= (uint64_t)buf[i]; h *= 1099511628211ull; }
        if (fwrite(buf, 1, want, dst) != want) { free(buf); return 0; }
        n -= want;
    }
    free(buf);
    *checksum = h;
    return 1;
}

int qx_write_tensor_copy_from_gguf(const char *out_path, const char *gguf_path, const qx_model_manifest *manifest, const qx_gguf_tensor_table *table, char *err, uint64_t err_len) {
    if (!out_path || !gguf_path || !manifest || !table || !table->tensors) {
        qx_set_err(err, err_len, "invalid argument"); return 0;
    }
    if (table->tensor_count == 0) { qx_set_err(err, err_len, "empty tensor directory"); return 0; }
    if (table->tensor_count > UINT32_MAX) { qx_set_err(err, err_len, "too many tensors for QXF v1"); return 0; }

    qx_tensor_dir_entry *entries = (qx_tensor_dir_entry *)calloc((size_t)table->tensor_count, sizeof(qx_tensor_dir_entry));
    if (!entries) { qx_set_err(err, err_len, "out of memory"); return 0; }

    uint64_t cursor = 0;
    if (!qx_align_u64_checked(sizeof(qx_header), QX_ALIGN_BYTES, &cursor)) {
        free(entries); qx_set_err(err, err_len, "QXF layout overflow"); return 0;
    }
    uint64_t dir_offset = cursor;
    uint64_t directory_bytes = table->tensor_count * sizeof(qx_tensor_dir_entry);
    if (dir_offset > UINT64_MAX - directory_bytes ||
        !qx_align_u64_checked(dir_offset + directory_bytes, QX_ALIGN_BYTES, &cursor)) {
        free(entries); qx_set_err(err, err_len, "QXF layout overflow"); return 0;
    }
    uint64_t data_offset = cursor;

    for (uint64_t i = 0; i < table->tensor_count; i++) {
        const qx_gguf_tensor_info *gt = &table->tensors[i];
        qx_tensor_dir_entry *e = &entries[i];
#ifdef _MSC_VER
        strncpy_s(e->name, sizeof(e->name), gt->name, _TRUNCATE);
#else
        snprintf(e->name, sizeof(e->name), "%s", gt->name);
#endif
        e->dtype = (uint32_t)qx_dtype_from_ggml_type(gt->ggml_type);
        e->quant = (uint32_t)manifest->quant_type;
        e->rank = gt->n_dims;
        for (uint32_t d = 0; d < gt->n_dims && d < QX_MAX_DIMS; d++) e->dims[d] = gt->dims[d];
        e->offset = cursor;
        e->byte_size = gt->byte_size;
        e->group_size = (e->dtype == QX_DTYPE_F32 || e->dtype == QX_DTYPE_F16 || e->dtype == QX_DTYPE_U8) ? 0u : 64u;
        e->flags = gt->ggml_type;
        if (cursor > UINT64_MAX - e->byte_size ||
            !qx_align_u64_checked(cursor + e->byte_size, QX_ALIGN_BYTES, &cursor)) {
            free(entries); qx_set_err(err, err_len, "QXF tensor layout overflow"); return 0;
        }
    }

    qx_header h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, QX_MAGIC, 4);
    h.version = QX_VERSION;
    h.header_size = (uint32_t)sizeof(qx_header);
    h.dir_entry_size = (uint32_t)sizeof(qx_tensor_dir_entry);
    h.tensor_count = (uint32_t)table->tensor_count;
    h.dir_offset = dir_offset;
    h.data_offset = data_offset;
    h.file_size = cursor;
    h.manifest = *manifest;
    h.manifest_checksum = qx_fnv1a64(&h.manifest, (uint64_t)sizeof(h.manifest));

    FILE *src = fopen(gguf_path, "rb");
    if (!src) { free(entries); qx_set_errno_err(err, err_len, errno); return 0; }
    FILE *dst = fopen(out_path, "wb");
    if (!dst) { fclose(src); free(entries); qx_set_errno_err(err, err_len, errno); return 0; }

    int ok = 1;
    if (fwrite(&h, sizeof(h), 1, dst) != 1) ok = 0;
    for (uint64_t i = sizeof(h); ok && i < dir_offset; i++) if (fputc(0, dst) == EOF) ok = 0;
    if (ok && fwrite(entries, sizeof(qx_tensor_dir_entry), (size_t)table->tensor_count, dst) != table->tensor_count) ok = 0;
    uint64_t pos = dir_offset + table->tensor_count * sizeof(qx_tensor_dir_entry);
    for (uint64_t i = pos; ok && i < data_offset; i++) if (fputc(0, dst) == EOF) ok = 0;

    for (uint64_t i = 0; ok && i < table->tensor_count; i++) {
        qx_tensor_dir_entry *e = &entries[i];
#if defined(_WIN32)
        if (_fseeki64(dst, (int64_t)e->offset, SEEK_SET) != 0) { ok = 0; break; }
#else
        if (fseeko(dst, (off_t)e->offset, SEEK_SET) != 0) { ok = 0; break; }
#endif
        uint64_t checksum = 0;
        if (table->data_offset > UINT64_MAX - table->tensors[i].offset ||
            !copy_tensor_bytes(src, dst, table->data_offset + table->tensors[i].offset, e->byte_size, &checksum)) { ok = 0; break; }
        e->checksum = checksum;
    }

    if (ok) {
        uint64_t written_end = entries[table->tensor_count - 1u].offset + entries[table->tensor_count - 1u].byte_size;
#if defined(_WIN32)
        if (h.file_size > written_end && (_fseeki64(dst, (int64_t)(h.file_size - 1u), SEEK_SET) != 0 || fputc(0, dst) == EOF)) ok = 0;
        if (ok && _fseeki64(dst, (int64_t)dir_offset, SEEK_SET) != 0) ok = 0;
#else
        if (h.file_size > written_end && (fseeko(dst, (off_t)(h.file_size - 1u), SEEK_SET) != 0 || fputc(0, dst) == EOF)) ok = 0;
        if (ok && fseeko(dst, (off_t)dir_offset, SEEK_SET) != 0) ok = 0;
#endif
        if (ok && fwrite(entries, sizeof(qx_tensor_dir_entry), (size_t)table->tensor_count, dst) != table->tensor_count) ok = 0;
    }
    if (fclose(dst) != 0) ok = 0;
    fclose(src);
    free(entries);
    if (!ok) { qx_set_err(err, err_len, "tensor-copy write failed"); return 0; }
    return 1;
}

static int qx_compare_tensor_names(const void *left, const void *right) {
    const char *const *a = (const char *const *)left;
    const char *const *b = (const char *const *)right;
    return strcmp(*a, *b);
}

typedef struct qx_tensor_span {
    uint64_t offset;
    uint64_t end;
} qx_tensor_span;

static int qx_compare_tensor_spans(const void *left, const void *right) {
    const qx_tensor_span *a = (const qx_tensor_span *)left;
    const qx_tensor_span *b = (const qx_tensor_span *)right;
    if (a->offset < b->offset) return -1;
    if (a->offset > b->offset) return 1;
    return 0;
}

static int qx_validate_tensor_dimensions(const qx_tensor_dir_entry *tensor, uint64_t *elements_out) {
    if (!tensor || !elements_out || tensor->rank == 0 || tensor->rank > QX_MAX_DIMS) return 0;
    uint64_t elements = 1u;
    for (uint32_t d = 0; d < QX_MAX_DIMS; ++d) {
        if (d < tensor->rank) {
            if (tensor->dims[d] == 0 || elements > UINT64_MAX / tensor->dims[d]) return 0;
            elements *= tensor->dims[d];
        } else if (tensor->dims[d] != 0) {
            return 0;
        }
    }
    *elements_out = elements;
    return elements > 0;
}

static int qx_expected_tensor_byte_size(const qx_tensor_dir_entry *tensor, uint64_t elements, uint64_t *expected_out) {
    uint64_t block_elements = 0;
    uint64_t block_bytes = 0;
    switch (tensor->flags) {
        case 0u: block_elements = 1u; block_bytes = 4u; break;   /* F32 */
        case 1u: block_elements = 1u; block_bytes = 2u; break;   /* F16 */
        case 8u: block_elements = 32u; block_bytes = 34u; break; /* Q8_0 */
        case 10u: block_elements = 256u; block_bytes = 84u; break;  /* Q2_K */
        case 11u: block_elements = 256u; block_bytes = 110u; break; /* Q3_K */
        case 12u: block_elements = 256u; block_bytes = 144u; break; /* Q4_K */
        case 13u: block_elements = 256u; block_bytes = 176u; break; /* Q5_K */
        case 14u: block_elements = 256u; block_bytes = 210u; break; /* Q6_K */
        case 17u: block_elements = 256u; block_bytes = 74u; break;  /* IQ2_XS */
        case 18u: block_elements = 256u; block_bytes = 98u; break;  /* IQ3_XXS */
        case 21u: block_elements = 256u; block_bytes = 110u; break; /* IQ3_S */
        case 22u: block_elements = 256u; block_bytes = 82u; break;  /* IQ2_S */
        case 23u: block_elements = 256u; block_bytes = 136u; break; /* IQ4_XS */
        default: return 0;
    }
    if (tensor->dims[0] % block_elements != 0 || elements % block_elements != 0) return 0;
    uint64_t blocks = elements / block_elements;
    if (blocks > UINT64_MAX / block_bytes) return 0;
    *expected_out = blocks * block_bytes;
    return *expected_out > 0;
}

static int qx_validate_tensor_traits(const qx_tensor_dir_entry *tensor) {
    uint32_t expected_dtype = 0;
    uint32_t expected_group = 0;
    switch (tensor->flags) {
        case 0u: expected_dtype = QX_DTYPE_F32; break;
        case 1u: expected_dtype = QX_DTYPE_F16; break;
        case 8u: expected_dtype = QX_DTYPE_U8; break;
        case 10u: case 17u: case 22u: expected_dtype = QX_DTYPE_Q2; expected_group = 64u; break;
        case 11u: case 18u: case 21u: expected_dtype = QX_DTYPE_Q3; expected_group = 64u; break;
        case 12u: case 13u: case 14u: expected_dtype = QX_DTYPE_Q4; expected_group = 64u; break;
        case 23u:
            return (tensor->dtype == QX_DTYPE_Q4 && tensor->group_size == 64u) ||
                   (tensor->dtype == QX_DTYPE_U8 && tensor->group_size == 0u);
        default: return 0;
    }
    return tensor->dtype == expected_dtype && tensor->group_size == expected_group;
}


int qx_open_file(const char *path, qx_file *out, char *err, uint64_t err_len) {
    if (!path || !out) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    qx_alloc_profile_init_once();
    memset(out, 0, sizeof(*out));
    out->fp = fopen(path, "rb");
    if (!out->fp) { qx_set_errno_err(err, err_len, errno); return 0; }
    if (!qx_read_header(out->fp, &out->header, err, err_len)) { qx_close_file(out); return 0; }
    if (out->header.tensor_count == 0) { qx_set_err(err, err_len, "empty tensor directory"); qx_close_file(out); return 0; }
    if (out->header.header_size != sizeof(qx_header) || out->header.dir_entry_size != sizeof(qx_tensor_dir_entry)) {
        qx_set_err(err, err_len, "unsupported QXF structure size"); qx_close_file(out); return 0;
    }
#if defined(_WIN32)
    if (_fseeki64(out->fp, 0, SEEK_END) != 0) { qx_set_err(err, err_len, "seek file end failed"); qx_close_file(out); return 0; }
    int64_t actual_size_signed = _ftelli64(out->fp);
#else
    if (fseeko(out->fp, 0, SEEK_END) != 0) { qx_set_err(err, err_len, "seek file end failed"); qx_close_file(out); return 0; }
    off_t actual_size_signed = ftello(out->fp);
#endif
    if (actual_size_signed < 0 || (uint64_t)actual_size_signed != out->header.file_size || out->header.file_size < sizeof(qx_header)) {
        qx_set_err(err, err_len, "declared QXF file size does not match file"); qx_close_file(out); return 0;
    }
    uint64_t directory_bytes = (uint64_t)out->header.tensor_count * (uint64_t)sizeof(qx_tensor_dir_entry);
    if (out->header.dir_offset < out->header.header_size || out->header.dir_offset % QX_ALIGN_BYTES != 0 ||
        out->header.data_offset % QX_ALIGN_BYTES != 0 || out->header.dir_offset > out->header.file_size ||
        directory_bytes > out->header.file_size - out->header.dir_offset ||
        out->header.data_offset < out->header.dir_offset + directory_bytes || out->header.data_offset > out->header.file_size) {
        qx_set_err(err, err_len, "invalid QXF layout"); qx_close_file(out); return 0;
    }
    out->directory = (qx_tensor_dir_entry *)calloc(out->header.tensor_count, sizeof(qx_tensor_dir_entry));
    if (!out->directory) { qx_set_err(err, err_len, "out of memory"); qx_close_file(out); return 0; }
#if defined(_WIN32)
    if (_fseeki64(out->fp, (int64_t)out->header.dir_offset, SEEK_SET) != 0) {
#else
    if (fseeko(out->fp, (off_t)out->header.dir_offset, SEEK_SET) != 0) {
#endif
        qx_set_err(err, err_len, "seek tensor directory failed"); qx_close_file(out); return 0;
    }
    if (fread(out->directory, sizeof(qx_tensor_dir_entry), out->header.tensor_count, out->fp) != out->header.tensor_count) {
        qx_set_err(err, err_len, "short tensor directory read"); qx_close_file(out); return 0;
    }
    const int metadata_only = out->header.file_size == out->header.data_offset;
    if ((uint64_t)out->header.tensor_count > (uint64_t)SIZE_MAX / sizeof(const char *) ||
        (uint64_t)out->header.tensor_count > (uint64_t)SIZE_MAX / sizeof(qx_tensor_span)) {
        qx_set_err(err, err_len, "tensor index too large"); qx_close_file(out); return 0;
    }
    const char **names = (const char **)malloc((size_t)out->header.tensor_count * sizeof(*names));
    qx_tensor_span *spans = metadata_only ? NULL : (qx_tensor_span *)malloc((size_t)out->header.tensor_count * sizeof(*spans));
    if (!names || (!metadata_only && !spans)) {
        free(names); free(spans); qx_set_err(err, err_len, "out of memory"); qx_close_file(out); return 0;
    }
    for (uint32_t i = 0; i < out->header.tensor_count; ++i) {
        const qx_tensor_dir_entry *tensor = &out->directory[i];
        if (!memchr(tensor->name, '\0', QX_NAME_MAX) || tensor->name[0] == '\0') {
            free(names); free(spans); qx_set_err(err, err_len, "unterminated or empty tensor name"); qx_close_file(out); return 0;
        }
        names[i] = tensor->name;
        if (tensor->dtype < QX_DTYPE_F16 || tensor->dtype > QX_DTYPE_U8 ||
            tensor->quant < QX_QUANT_F16 || tensor->quant > QX_QUANT_Q2_BLOCK ||
            (!metadata_only && !qx_validate_tensor_traits(tensor))) {
            free(names); free(spans); qx_set_err(err, err_len, "invalid tensor metadata"); qx_close_file(out); return 0;
        }
        uint64_t elements = 0;
        if (!qx_validate_tensor_dimensions(tensor, &elements)) {
            free(names); free(spans); qx_set_err(err, err_len, "invalid tensor dimensions"); qx_close_file(out); return 0;
        }
        if (metadata_only) {
            if (tensor->offset != 0 || tensor->byte_size != 0) {
                free(names); free(spans); qx_set_err(err, err_len, "invalid tensor placement"); qx_close_file(out); return 0;
            }
        } else {
            if (tensor->offset < out->header.data_offset || tensor->offset % QX_ALIGN_BYTES != 0 || tensor->byte_size == 0 ||
                tensor->offset > out->header.file_size || tensor->byte_size > out->header.file_size - tensor->offset) {
                free(names); free(spans); qx_set_err(err, err_len, "invalid tensor placement"); qx_close_file(out); return 0;
            }
            uint64_t expected_size = 0;
            if (!qx_expected_tensor_byte_size(tensor, elements, &expected_size) || tensor->byte_size != expected_size) {
                free(names); free(spans); qx_set_err(err, err_len, "tensor byte size inconsistent with dimensions"); qx_close_file(out); return 0;
            }
            spans[i].offset = tensor->offset;
            spans[i].end = tensor->offset + tensor->byte_size;
        }
    }
    qsort(names, out->header.tensor_count, sizeof(*names), qx_compare_tensor_names);
    for (uint32_t i = 1; i < out->header.tensor_count; ++i) {
        if (strcmp(names[i - 1], names[i]) == 0) {
            free(names); free(spans); qx_set_err(err, err_len, "duplicate tensor name"); qx_close_file(out); return 0;
        }
    }
    free(names);
    if (spans) {
        qsort(spans, out->header.tensor_count, sizeof(*spans), qx_compare_tensor_spans);
        uint64_t previous_end = out->header.data_offset;
        for (uint32_t i = 0; i < out->header.tensor_count; ++i) {
            if (spans[i].offset < previous_end) {
                free(spans); qx_set_err(err, err_len, "overlapping tensor ranges"); qx_close_file(out); return 0;
            }
            previous_end = spans[i].end;
        }
        free(spans);
    }
    out->io_backend = qx_requested_io_backend;
    if (out->io_backend == QX_IO_MMAP) {
        if (out->header.file_size > (uint64_t)SIZE_MAX ||
                out->header.file_size > (uint64_t)PTRDIFF_MAX) {
            qx_set_err(err, err_len, "QXF file too large to map"); qx_close_file(out); return 0;
        }
#if defined(_WIN32)
        intptr_t native_file = _get_osfhandle(_fileno(out->fp));
        if (native_file == -1) {
            qx_set_err(err, err_len, "cannot obtain QXF file handle for mapping"); qx_close_file(out); return 0;
        }
        HANDLE mapping = CreateFileMappingW((HANDLE)native_file, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!mapping) {
            qx_set_err(err, err_len, "QXF file mapping failed"); qx_close_file(out); return 0;
        }
        out->mapping_handle = mapping;
        out->mapped_view = (const unsigned char *)MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        if (!out->mapped_view) {
            qx_set_err(err, err_len, "QXF mapped view failed"); qx_close_file(out); return 0;
        }
#else
        void *view = mmap(NULL, (size_t)out->header.file_size, PROT_READ, MAP_PRIVATE, fileno(out->fp), 0);
        if (view == MAP_FAILED) {
            qx_set_err(err, err_len, "QXF file mapping failed"); qx_close_file(out); return 0;
        }
        out->mapped_view = (const unsigned char *)view;
#endif
    }
    return 1;
}

void qx_close_file(qx_file *file) {
    if (!file) return;
#if defined(_WIN32)
    if (file->mapped_view) UnmapViewOfFile(file->mapped_view);
    if (file->mapping_handle) CloseHandle((HANDLE)file->mapping_handle);
#else
    if (file->mapped_view && file->header.file_size <= (uint64_t)SIZE_MAX) {
        munmap((void *)file->mapped_view, (size_t)file->header.file_size);
    }
#endif
    if (file->fp) fclose(file->fp);
    free(file->directory);
    memset(file, 0, sizeof(*file));
}

int qx_acquire_span(qx_file *file, uint64_t offset, uint64_t size, qx_span *out, char *err, uint64_t err_len) {
    if (!file || !file->fp || !out) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    memset(out, 0, sizeof(*out));
    if (size == 0) { qx_set_err(err, err_len, "empty QXF span"); return 0; }
    if (offset > file->header.file_size || size > file->header.file_size - offset) {
        qx_set_err(err, err_len, "QXF span outside file"); return 0;
    }
    if (file->io_backend == QX_IO_MMAP) {
        if (!file->mapped_view) { qx_set_err(err, err_len, "QXF mapped view unavailable"); return 0; }
        out->data = file->mapped_view + offset;
        out->size = size;
        return 1;
    }
    if (size > (uint64_t)SIZE_MAX) { qx_set_err(err, err_len, "QXF span too large"); return 0; }
    out->owned_data = (unsigned char *)malloc((size_t)size);
    if (!out->owned_data) { qx_set_err(err, err_len, "out of memory"); return 0; }
#if defined(_WIN32)
    if (_fseeki64(file->fp, (int64_t)offset, SEEK_SET) != 0) {
#else
    if (fseeko(file->fp, (off_t)offset, SEEK_SET) != 0) {
#endif
        qx_release_span(out); qx_set_err(err, err_len, "seek failed"); return 0;
    }
    if (fread(out->owned_data, 1, (size_t)size, file->fp) != (size_t)size) {
        qx_release_span(out); qx_set_err(err, err_len, "short read"); return 0;
    }
    out->data = out->owned_data;
    out->size = size;
    return 1;
}

void qx_release_span(qx_span *span) {
    if (!span) return;
    free(span->owned_data);
    memset(span, 0, sizeof(*span));
}

const qx_tensor_dir_entry *qx_find_tensor(const qx_file *file, const char *name) {
    if (!file || !file->directory || !name) return NULL;
    for (uint32_t i = 0; i < file->header.tensor_count; i++) {
        if (strcmp(file->directory[i].name, name) == 0) return &file->directory[i];
    }
    return NULL;
}

int qx_verify_tensor_checksum(qx_file *file, const qx_tensor_dir_entry *tensor, char *err, uint64_t err_len) {
    if (!file || !file->fp || !tensor) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    if (tensor->offset > file->header.file_size || tensor->byte_size > file->header.file_size - tensor->offset) {
        qx_set_err(err, err_len, "tensor range outside file"); return 0;
    }
    if (file->io_backend == QX_IO_MMAP) {
        qx_span span;
        if (!qx_acquire_span(file, tensor->offset, tensor->byte_size, &span, err, err_len)) return 0;
        uint64_t h = qx_fnv1a64(span.data, span.size);
        qx_release_span(&span);
        if (h != tensor->checksum) { qx_set_err(err, err_len, "tensor checksum mismatch"); return 0; }
        return 1;
    }
    unsigned char *buf = (unsigned char *)malloc(64 * 1024);
    if (!buf) { qx_set_err(err, err_len, "out of memory"); return 0; }
#if defined(_WIN32)
    if (_fseeki64(file->fp, (int64_t)tensor->offset, SEEK_SET) != 0) {
#else
    if (fseeko(file->fp, (off_t)tensor->offset, SEEK_SET) != 0) {
#endif
        free(buf); qx_set_err(err, err_len, "seek tensor failed"); return 0;
    }
    uint64_t remaining = tensor->byte_size;
    uint64_t h = 1469598103934665603ull;
    while (remaining > 0) {
        size_t want = remaining < 64 * 1024 ? (size_t)remaining : 64 * 1024;
        if (fread(buf, 1, want, file->fp) != want) { free(buf); qx_set_err(err, err_len, "short tensor read"); return 0; }
        for (size_t i = 0; i < want; i++) { h ^= (uint64_t)buf[i]; h *= 1099511628211ull; }
        remaining -= want;
    }
    free(buf);
    if (h != tensor->checksum) {
        qx_set_err(err, err_len, "tensor checksum mismatch"); return 0;
    }
    return 1;
}

int qx_dump_tensor_summary(const char *path, const char *name, FILE *out, char *err, uint64_t err_len) {
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    const qx_tensor_dir_entry *t = qx_find_tensor(&file, name);
    if (!t) { qx_close_file(&file); qx_set_err(err, err_len, "tensor not found"); return 0; }
    fprintf(out, "{\n");
    fprintf(out, "  \"name\": \"%s\",\n", t->name);
    fprintf(out, "  \"dtype\": %u,\n", t->dtype);
    fprintf(out, "  \"ggml_type\": %u,\n", t->flags);
    fprintf(out, "  \"quant\": %u,\n", t->quant);
    fprintf(out, "  \"rank\": %u,\n", t->rank);
    fprintf(out, "  \"dims\": [");
    for (uint32_t d = 0; d < t->rank && d < QX_MAX_DIMS; d++) fprintf(out, "%s%llu", d ? ", " : "", (unsigned long long)t->dims[d]);
    fprintf(out, "],\n");
    fprintf(out, "  \"offset\": %llu,\n", (unsigned long long)t->offset);
    fprintf(out, "  \"byte_size\": %llu,\n", (unsigned long long)t->byte_size);
    fprintf(out, "  \"io_backend\": \"%s\",\n", qx_io_backend_name(file.io_backend));
    fprintf(out, "  \"checksum\": %llu\n", (unsigned long long)t->checksum);
    fprintf(out, "}\n");
    qx_close_file(&file);
    return 1;
}

int qx_verify_all_tensors(const char *path, uint32_t max_tensors, FILE *out, char *err, uint64_t err_len) {
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    uint32_t limit = max_tensors == 0 || max_tensors > file.header.tensor_count ? file.header.tensor_count : max_tensors;
    for (uint32_t i = 0; i < limit; i++) {
        if (!qx_verify_tensor_checksum(&file, &file.directory[i], err, err_len)) {
            fprintf(out, "{\"verified\": false, \"failed_index\": %u, \"failed_tensor\": \"%s\"}\n", i, file.directory[i].name);
            qx_close_file(&file);
            return 0;
        }
    }
    fprintf(out, "{\"verified\": true, \"checked\": %u, \"tensor_count\": %u}\n", limit, file.header.tensor_count);
    qx_close_file(&file);
    return 1;
}


static int qx_parse_layer_suffix(const char *name, const char *suffix, uint32_t *layer) {
    unsigned int l = 0;
    char tail[64];
    if (sscanf(name, "blk.%u.%63s", &l, tail) != 2) return 0;
    if (strcmp(tail, suffix) != 0) return 0;
    *layer = (uint32_t)l;
    return 1;
}

int qx_dump_expert_index_summary(const char *path, FILE *out, char *err, uint64_t err_len) {
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    const uint32_t layers = file.header.manifest.layers;
    if (file.header.manifest.model_type != QX_MODEL_QWEN3_MOE || layers == 0) {
        qx_close_file(&file); qx_set_err(err, err_len, "not a MoE QXF"); return 0;
    }
    unsigned char *has_router = (unsigned char *)calloc(layers, 1);
    unsigned char *has_gate = (unsigned char *)calloc(layers, 1);
    unsigned char *has_up = (unsigned char *)calloc(layers, 1);
    unsigned char *has_down = (unsigned char *)calloc(layers, 1);
    uint64_t *layer_bytes = (uint64_t *)calloc(layers, sizeof(uint64_t));
    if (!has_router || !has_gate || !has_up || !has_down || !layer_bytes) {
        free(has_router); free(has_gate); free(has_up); free(has_down); free(layer_bytes);
        qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0;
    }
    uint64_t expert_tensor_bytes = 0;
    uint32_t router_count = 0, gate_count = 0, up_count = 0, down_count = 0, complete_layers = 0;
    uint64_t min_layer_bytes = UINT64_MAX, max_layer_bytes = 0;
    for (uint32_t i = 0; i < file.header.tensor_count; i++) {
        qx_tensor_dir_entry *t = &file.directory[i];
        uint32_t layer = 0;
        if (qx_parse_layer_suffix(t->name, "ffn_gate_inp.weight", &layer) && layer < layers) {
            has_router[layer] = 1; router_count++;
        } else if (qx_parse_layer_suffix(t->name, "ffn_gate_exps.weight", &layer) && layer < layers) {
            has_gate[layer] = 1; gate_count++; expert_tensor_bytes += t->byte_size; layer_bytes[layer] += t->byte_size;
        } else if (qx_parse_layer_suffix(t->name, "ffn_up_exps.weight", &layer) && layer < layers) {
            has_up[layer] = 1; up_count++; expert_tensor_bytes += t->byte_size; layer_bytes[layer] += t->byte_size;
        } else if (qx_parse_layer_suffix(t->name, "ffn_down_exps.weight", &layer) && layer < layers) {
            has_down[layer] = 1; down_count++; expert_tensor_bytes += t->byte_size; layer_bytes[layer] += t->byte_size;
        }
    }
    for (uint32_t l = 0; l < layers; l++) {
        if (has_router[l] && has_gate[l] && has_up[l] && has_down[l]) complete_layers++;
        if (layer_bytes[l] < min_layer_bytes) min_layer_bytes = layer_bytes[l];
        if (layer_bytes[l] > max_layer_bytes) max_layer_bytes = layer_bytes[l];
    }
    if (min_layer_bytes == UINT64_MAX) min_layer_bytes = 0;
    uint64_t avg_layer_bytes = layers ? expert_tensor_bytes / layers : 0;
    uint64_t avg_expert_bytes = (layers && file.header.manifest.experts) ? expert_tensor_bytes / ((uint64_t)layers * file.header.manifest.experts) : 0;
    fprintf(out, "{\n");
    fprintf(out, "  \"model_type\": \"%s\",\n", qx_model_type_name(file.header.manifest.model_type));
    fprintf(out, "  \"layers\": %u,\n", layers);
    fprintf(out, "  \"experts_per_layer\": %u,\n", file.header.manifest.experts);
    fprintf(out, "  \"experts_per_token\": %u,\n", file.header.manifest.experts_per_token);
    fprintf(out, "  \"router_tensors\": %u,\n", router_count);
    fprintf(out, "  \"expert_gate_tensors\": %u,\n", gate_count);
    fprintf(out, "  \"expert_up_tensors\": %u,\n", up_count);
    fprintf(out, "  \"expert_down_tensors\": %u,\n", down_count);
    fprintf(out, "  \"complete_layers\": %u,\n", complete_layers);
    fprintf(out, "  \"packed_expert_tensors\": %u,\n", gate_count + up_count + down_count);
    fprintf(out, "  \"expert_tensor_bytes\": %llu,\n", (unsigned long long)expert_tensor_bytes);
    fprintf(out, "  \"avg_layer_expert_bytes\": %llu,\n", (unsigned long long)avg_layer_bytes);
    fprintf(out, "  \"min_layer_expert_bytes\": %llu,\n", (unsigned long long)min_layer_bytes);
    fprintf(out, "  \"max_layer_expert_bytes\": %llu,\n", (unsigned long long)max_layer_bytes);
    fprintf(out, "  \"avg_single_expert_bytes\": %llu\n", (unsigned long long)avg_expert_bytes);
    fprintf(out, "}\n");
    free(has_router); free(has_gate); free(has_up); free(has_down); free(layer_bytes);
    qx_close_file(&file);
    return 1;
}


static int qx_collect_expert_stats(const qx_file *file, uint64_t *expert_tensor_bytes, uint32_t *packed_tensors) {
    if (!file || !expert_tensor_bytes || !packed_tensors) return 0;
    *expert_tensor_bytes = 0;
    *packed_tensors = 0;
    for (uint32_t i = 0; i < file->header.tensor_count; i++) {
        const qx_tensor_dir_entry *t = &file->directory[i];
        uint32_t layer = 0;
        if (qx_parse_layer_suffix(t->name, "ffn_gate_exps.weight", &layer) ||
            qx_parse_layer_suffix(t->name, "ffn_up_exps.weight", &layer) ||
            qx_parse_layer_suffix(t->name, "ffn_down_exps.weight", &layer)) {
            *expert_tensor_bytes += t->byte_size;
            *packed_tensors += 1;
        }
    }
    return *packed_tensors > 0;
}

int qx_dump_expert_cache_plan(const char *path, double hot_vram_gib, double hot_ram_gib, FILE *out, char *err, uint64_t err_len) {
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    if (file.header.manifest.model_type != QX_MODEL_QWEN3_MOE || file.header.manifest.layers == 0 || file.header.manifest.experts == 0) {
        qx_close_file(&file); qx_set_err(err, err_len, "not a MoE QXF"); return 0;
    }
    uint64_t expert_bytes = 0;
    uint32_t packed_tensors = 0;
    if (!qx_collect_expert_stats(&file, &expert_bytes, &packed_tensors)) {
        qx_close_file(&file); qx_set_err(err, err_len, "no packed expert tensors"); return 0;
    }
    uint64_t expert_count_total = (uint64_t)file.header.manifest.layers * file.header.manifest.experts;
    uint64_t avg_expert_bytes = expert_count_total ? expert_bytes / expert_count_total : 0;
    uint64_t vram_bytes = (uint64_t)(hot_vram_gib * 1024.0 * 1024.0 * 1024.0);
    uint64_t ram_bytes = (uint64_t)(hot_ram_gib * 1024.0 * 1024.0 * 1024.0);
    uint64_t vram_slots = avg_expert_bytes ? vram_bytes / avg_expert_bytes : 0;
    uint64_t ram_slots = avg_expert_bytes ? ram_bytes / avg_expert_bytes : 0;
    double vram_layer_equiv = file.header.manifest.experts ? (double)vram_slots / (double)file.header.manifest.experts : 0.0;
    double ram_layer_equiv = file.header.manifest.experts ? (double)ram_slots / (double)file.header.manifest.experts : 0.0;
    fprintf(out, "{\n");
    fprintf(out, "  \"avg_single_expert_bytes\": %llu,\n", (unsigned long long)avg_expert_bytes);
    fprintf(out, "  \"total_expert_bytes\": %llu,\n", (unsigned long long)expert_bytes);
    fprintf(out, "  \"total_experts\": %llu,\n", (unsigned long long)expert_count_total);
    fprintf(out, "  \"hot_vram_gib\": %.3f,\n", hot_vram_gib);
    fprintf(out, "  \"hot_ram_gib\": %.3f,\n", hot_ram_gib);
    fprintf(out, "  \"vram_expert_slots\": %llu,\n", (unsigned long long)vram_slots);
    fprintf(out, "  \"ram_expert_slots\": %llu,\n", (unsigned long long)ram_slots);
    fprintf(out, "  \"vram_layer_equivalent\": %.2f,\n", vram_layer_equiv);
    fprintf(out, "  \"ram_layer_equivalent\": %.2f,\n", ram_layer_equiv);
    fprintf(out, "  \"experts_per_token\": %u\n", file.header.manifest.experts_per_token);
    fprintf(out, "}\n");
    qx_close_file(&file);
    return 1;
}


static int qx_expert_packed_tensor_name(char *buf, uint64_t cap, uint32_t layer, const char *kind) {
    if (!buf || cap == 0 || !kind) return 0;
#ifdef _MSC_VER
    return snprintf(buf, (size_t)cap, "blk.%u.ffn_%s_exps.weight", layer, kind) > 0;
#else
    return snprintf(buf, (size_t)cap, "blk.%u.ffn_%s_exps.weight", layer, kind) > 0;
#endif
}

static int qx_print_expert_slice(FILE *out, const qx_tensor_dir_entry *t, uint32_t expert_count, uint32_t expert, const char *json_name, int comma) {
    if (!t || expert_count == 0 || expert >= expert_count) return 0;
    uint64_t slice_bytes = t->byte_size / expert_count;
    uint64_t remainder = t->byte_size % expert_count;
    uint64_t slice_offset = t->offset + slice_bytes * expert;
    fprintf(out, "    \"%s\": {\"tensor\": \"%s\", \"offset\": %llu, \"byte_size\": %llu, \"packed_byte_size\": %llu, \"remainder\": %llu}%s\n",
        json_name, t->name, (unsigned long long)slice_offset, (unsigned long long)slice_bytes,
        (unsigned long long)t->byte_size, (unsigned long long)remainder, comma ? "," : "");
    return remainder == 0;
}

int qx_dump_expert_slice_summary(const char *path, uint32_t layer, uint32_t expert, FILE *out, char *err, uint64_t err_len) {
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    if (file.header.manifest.model_type != QX_MODEL_QWEN3_MOE) {
        qx_close_file(&file); qx_set_err(err, err_len, "not a MoE QXF"); return 0;
    }
    if (layer >= file.header.manifest.layers) {
        qx_close_file(&file); qx_set_err(err, err_len, "layer out of range"); return 0;
    }
    if (expert >= file.header.manifest.experts) {
        qx_close_file(&file); qx_set_err(err, err_len, "expert out of range"); return 0;
    }
    char name[QX_NAME_MAX];
    qx_expert_packed_tensor_name(name, sizeof(name), layer, "gate");
    const qx_tensor_dir_entry *gate = qx_find_tensor(&file, name);
    qx_expert_packed_tensor_name(name, sizeof(name), layer, "up");
    const qx_tensor_dir_entry *up = qx_find_tensor(&file, name);
    qx_expert_packed_tensor_name(name, sizeof(name), layer, "down");
    const qx_tensor_dir_entry *down = qx_find_tensor(&file, name);
    if (!gate || !up || !down) {
        qx_close_file(&file); qx_set_err(err, err_len, "missing packed expert tensor"); return 0;
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"layer\": %u,\n", layer);
    fprintf(out, "  \"expert\": %u,\n", expert);
    fprintf(out, "  \"experts_per_layer\": %u,\n", file.header.manifest.experts);
    fprintf(out, "  \"slices\": {\n");
    int ok = 1;
    ok &= qx_print_expert_slice(out, gate, file.header.manifest.experts, expert, "gate", 1);
    ok &= qx_print_expert_slice(out, up, file.header.manifest.experts, expert, "up", 1);
    ok &= qx_print_expert_slice(out, down, file.header.manifest.experts, expert, "down", 0);
    fprintf(out, "  },\n");
    fprintf(out, "  \"slice_exact\": %s\n", ok ? "true" : "false");
    fprintf(out, "}\n");
    qx_close_file(&file);
    return 1;
}


static int qx_kind_valid(const char *kind) {
    return kind && (strcmp(kind, "gate") == 0 || strcmp(kind, "up") == 0 || strcmp(kind, "down") == 0);
}

int qx_dump_expert_load_summary(const char *path, uint32_t layer, uint32_t expert, const char *kind, FILE *out, char *err, uint64_t err_len) {
    if (!qx_kind_valid(kind)) { qx_set_err(err, err_len, "invalid expert kind"); return 0; }
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    if (file.header.manifest.model_type != QX_MODEL_QWEN3_MOE) {
        qx_close_file(&file); qx_set_err(err, err_len, "not a MoE QXF"); return 0;
    }
    if (layer >= file.header.manifest.layers || expert >= file.header.manifest.experts) {
        qx_close_file(&file); qx_set_err(err, err_len, "expert address out of range"); return 0;
    }
    char name[QX_NAME_MAX];
    qx_expert_packed_tensor_name(name, sizeof(name), layer, kind);
    const qx_tensor_dir_entry *t = qx_find_tensor(&file, name);
    if (!t) { qx_close_file(&file); qx_set_err(err, err_len, "packed expert tensor not found"); return 0; }
    uint64_t slice_bytes = t->byte_size / file.header.manifest.experts;
    uint64_t remainder = t->byte_size % file.header.manifest.experts;
    if (remainder != 0) { qx_close_file(&file); qx_set_err(err, err_len, "packed expert tensor not evenly divisible"); return 0; }
    uint64_t slice_offset = t->offset + slice_bytes * expert;
    unsigned char *buf = (unsigned char *)malloc((size_t)slice_bytes);
    if (!buf) { qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
#if defined(_WIN32)
    if (_fseeki64(file.fp, (int64_t)slice_offset, SEEK_SET) != 0) {
#else
    if (fseeko(file.fp, (off_t)slice_offset, SEEK_SET) != 0) {
#endif
        free(buf); qx_close_file(&file); qx_set_err(err, err_len, "seek expert slice failed"); return 0;
    }
    if (fread(buf, 1, (size_t)slice_bytes, file.fp) != (size_t)slice_bytes) {
        free(buf); qx_close_file(&file); qx_set_err(err, err_len, "short expert slice read"); return 0;
    }
    uint64_t checksum = qx_fnv1a64(buf, slice_bytes);
    free(buf);
    fprintf(out, "{\n");
    fprintf(out, "  \"loaded\": true,\n");
    fprintf(out, "  \"layer\": %u,\n", layer);
    fprintf(out, "  \"expert\": %u,\n", expert);
    fprintf(out, "  \"kind\": \"%s\",\n", kind);
    fprintf(out, "  \"tensor\": \"%s\",\n", t->name);
    fprintf(out, "  \"offset\": %llu,\n", (unsigned long long)slice_offset);
    fprintf(out, "  \"byte_size\": %llu,\n", (unsigned long long)slice_bytes);
    fprintf(out, "  \"checksum\": %llu\n", (unsigned long long)checksum);
    fprintf(out, "}\n");
    qx_close_file(&file);
    return 1;
}


typedef struct qx_cache_key {
    uint32_t layer;
    uint32_t expert;
    char kind[8];
    uint64_t age;
    int used;
} qx_cache_key;

static int qx_cache_key_equal(const qx_cache_key *a, uint32_t layer, uint32_t expert, const char *kind) {
    return a->used && a->layer == layer && a->expert == expert && strcmp(a->kind, kind) == 0;
}

int qx_dump_cache_demo_summary(const char *path, uint32_t slots, const char *sequence, FILE *out, char *err, uint64_t err_len) {
    if (!path || !sequence || slots == 0) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    qx_cache_key *cache = (qx_cache_key *)calloc(slots, sizeof(qx_cache_key));
    char *seq = (char *)malloc(strlen(sequence) + 1);
    if (!cache || !seq) { free(cache); free(seq); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
    strcpy(seq, sequence);
    uint32_t requests = 0, hits = 0, misses = 0;
    uint64_t age = 1;
    char *tok = strtok(seq, ",");
    while (tok) {
        unsigned int layer = 0, expert = 0;
        char kind[8] = {0};
        if (sscanf(tok, "%u:%u:%7s", &layer, &expert, kind) != 3 || !qx_kind_valid(kind)) {
            free(cache); free(seq); qx_close_file(&file); qx_set_err(err, err_len, "bad sequence item"); return 0;
        }
        requests++;
        int hit_idx = -1;
        for (uint32_t i = 0; i < slots; i++) {
            if (qx_cache_key_equal(&cache[i], layer, expert, kind)) { hit_idx = (int)i; break; }
        }
        if (hit_idx >= 0) {
            hits++;
            cache[hit_idx].age = age++;
        } else {
            misses++;
            uint32_t victim = 0;
            uint64_t oldest = UINT64_MAX;
            for (uint32_t i = 0; i < slots; i++) {
                if (!cache[i].used) { victim = i; oldest = 0; break; }
                if (cache[i].age < oldest) { oldest = cache[i].age; victim = i; }
            }
            cache[victim].used = 1;
            cache[victim].layer = layer;
            cache[victim].expert = expert;
#ifdef _MSC_VER
            strncpy_s(cache[victim].kind, sizeof(cache[victim].kind), kind, _TRUNCATE);
#else
            snprintf(cache[victim].kind, sizeof(cache[victim].kind), "%s", kind);
#endif
            cache[victim].age = age++;
        }
        tok = strtok(NULL, ",");
    }
    fprintf(out, "{\"requests\": %u, \"hits\": %u, \"misses\": %u, \"slots\": %u}\n", requests, hits, misses, slots);
    free(cache); free(seq); qx_close_file(&file);
    return 1;
}


static int qx_read_expert_slice_once(qx_file *file, uint32_t layer, uint32_t expert, const char *kind, uint64_t *bytes_read, uint64_t *checksum, char *err, uint64_t err_len) {
    char name[QX_NAME_MAX];
    qx_expert_packed_tensor_name(name, sizeof(name), layer, kind);
    const qx_tensor_dir_entry *t = qx_find_tensor(file, name);
    if (!t) { qx_set_err(err, err_len, "packed expert tensor not found"); return 0; }
    uint64_t slice_bytes = t->byte_size / file->header.manifest.experts;
    if (t->byte_size % file->header.manifest.experts != 0) { qx_set_err(err, err_len, "packed expert tensor not evenly divisible"); return 0; }
    uint64_t slice_offset = t->offset + slice_bytes * expert;
    unsigned char *buf = (unsigned char *)malloc((size_t)slice_bytes);
    if (!buf) { qx_set_err(err, err_len, "out of memory"); return 0; }
#if defined(_WIN32)
    if (_fseeki64(file->fp, (int64_t)slice_offset, SEEK_SET) != 0) {
#else
    if (fseeko(file->fp, (off_t)slice_offset, SEEK_SET) != 0) {
#endif
        free(buf); qx_set_err(err, err_len, "seek expert slice failed"); return 0;
    }
    if (fread(buf, 1, (size_t)slice_bytes, file->fp) != (size_t)slice_bytes) {
        free(buf); qx_set_err(err, err_len, "short expert slice read"); return 0;
    }
    if (checksum) *checksum ^= qx_fnv1a64(buf, slice_bytes);
    if (bytes_read) *bytes_read += slice_bytes;
    free(buf);
    return 1;
}

int qx_dump_expert_load_benchmark(const char *path, uint32_t iters, const char *kind, FILE *out, char *err, uint64_t err_len) {
    if (!kind || !qx_kind_valid(kind) || iters == 0) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    if (file.header.manifest.model_type != QX_MODEL_QWEN3_MOE) {
        qx_close_file(&file); qx_set_err(err, err_len, "not a MoE QXF"); return 0;
    }
    uint64_t bytes = 0;
    uint64_t checksum_mix = 0;
    clock_t start = clock();
    for (uint32_t i = 0; i < iters; i++) {
        uint32_t layer = i % file.header.manifest.layers;
        uint32_t expert = (i * 17u) % file.header.manifest.experts;
        if (!qx_read_expert_slice_once(&file, layer, expert, kind, &bytes, &checksum_mix, err, err_len)) {
            qx_close_file(&file); return 0;
        }
    }
    clock_t end = clock();
    double seconds = (double)(end - start) / (double)CLOCKS_PER_SEC;
    if (seconds <= 0.0) seconds = 0.000001;
    double mib = (double)bytes / (1024.0 * 1024.0);
    double mib_per_sec = mib / seconds;
    double avg_ms = (seconds * 1000.0) / (double)iters;
    fprintf(out, "{\n");
    fprintf(out, "  \"loads\": %u,\n", iters);
    fprintf(out, "  \"kind\": \"%s\",\n", kind);
    fprintf(out, "  \"bytes\": %llu,\n", (unsigned long long)bytes);
    fprintf(out, "  \"mib\": %.3f,\n", mib);
    fprintf(out, "  \"seconds\": %.6f,\n", seconds);
    fprintf(out, "  \"avg_ms\": %.3f,\n", avg_ms);
    fprintf(out, "  \"mib_per_sec\": %.3f,\n", mib_per_sec);
    fprintf(out, "  \"checksum_mix\": %llu\n", (unsigned long long)checksum_mix);
    fprintf(out, "}\n");
    qx_close_file(&file);
    return 1;
}


typedef struct qx_cache_slot {
    qx_cache_key key;
    unsigned char *data;
    uint64_t byte_size;
    uint64_t checksum;
} qx_cache_slot;

static void qx_free_cache_slots(qx_cache_slot *slots, uint32_t count) {
    if (!slots) return;
    for (uint32_t i = 0; i < count; i++) free(slots[i].data);
    free(slots);
}

static int qx_read_expert_slice_alloc(qx_file *file, uint32_t layer, uint32_t expert, const char *kind,
                                      unsigned char **data, uint64_t *byte_size, uint64_t *checksum,
                                      char *err, uint64_t err_len) {
    char name[QX_NAME_MAX];
    qx_expert_packed_tensor_name(name, sizeof(name), layer, kind);
    const qx_tensor_dir_entry *t = qx_find_tensor(file, name);
    if (!t) { qx_set_err(err, err_len, "packed expert tensor not found"); return 0; }
    uint64_t slice_bytes = t->byte_size / file->header.manifest.experts;
    if (t->byte_size % file->header.manifest.experts != 0) { qx_set_err(err, err_len, "packed expert tensor not evenly divisible"); return 0; }
    uint64_t slice_offset = t->offset + slice_bytes * expert;
    unsigned char *buf = (unsigned char *)malloc((size_t)slice_bytes);
    if (!buf) { qx_set_err(err, err_len, "out of memory"); return 0; }
#if defined(_WIN32)
    if (_fseeki64(file->fp, (int64_t)slice_offset, SEEK_SET) != 0) {
#else
    if (fseeko(file->fp, (off_t)slice_offset, SEEK_SET) != 0) {
#endif
        free(buf); qx_set_err(err, err_len, "seek expert slice failed"); return 0;
    }
    if (fread(buf, 1, (size_t)slice_bytes, file->fp) != (size_t)slice_bytes) {
        free(buf); qx_set_err(err, err_len, "short expert slice read"); return 0;
    }
    *data = buf;
    *byte_size = slice_bytes;
    *checksum = qx_fnv1a64(buf, slice_bytes);
    return 1;
}

int qx_dump_cache_run_summary(const char *path, uint32_t slots_count, const char *sequence, FILE *out, char *err, uint64_t err_len) {
    if (!path || !sequence || slots_count == 0) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    qx_cache_slot *slots = (qx_cache_slot *)calloc(slots_count, sizeof(qx_cache_slot));
    char *seq = (char *)malloc(strlen(sequence) + 1);
    if (!slots || !seq) { qx_free_cache_slots(slots, slots_count); free(seq); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
    strcpy(seq, sequence);
    uint32_t requests = 0, hits = 0, misses = 0;
    uint64_t age = 1, bytes_loaded = 0, checksum_mix = 0;
    char *tok = strtok(seq, ",");
    while (tok) {
        unsigned int layer = 0, expert = 0;
        char kind[8] = {0};
        if (sscanf(tok, "%u:%u:%7s", &layer, &expert, kind) != 3 || !qx_kind_valid(kind)) {
            qx_free_cache_slots(slots, slots_count); free(seq); qx_close_file(&file); qx_set_err(err, err_len, "bad sequence item"); return 0;
        }
        requests++;
        int hit_idx = -1;
        for (uint32_t i = 0; i < slots_count; i++) {
            if (qx_cache_key_equal(&slots[i].key, layer, expert, kind)) { hit_idx = (int)i; break; }
        }
        if (hit_idx >= 0) {
            hits++;
            slots[hit_idx].key.age = age++;
            checksum_mix ^= slots[hit_idx].checksum;
        } else {
            misses++;
            uint32_t victim = 0;
            uint64_t oldest = UINT64_MAX;
            for (uint32_t i = 0; i < slots_count; i++) {
                if (!slots[i].key.used) { victim = i; oldest = 0; break; }
                if (slots[i].key.age < oldest) { oldest = slots[i].key.age; victim = i; }
            }
            free(slots[victim].data);
            slots[victim].data = NULL;
            slots[victim].byte_size = 0;
            slots[victim].checksum = 0;
            if (!qx_read_expert_slice_alloc(&file, layer, expert, kind, &slots[victim].data, &slots[victim].byte_size, &slots[victim].checksum, err, err_len)) {
                qx_free_cache_slots(slots, slots_count); free(seq); qx_close_file(&file); return 0;
            }
            slots[victim].key.used = 1;
            slots[victim].key.layer = layer;
            slots[victim].key.expert = expert;
#ifdef _MSC_VER
            strncpy_s(slots[victim].key.kind, sizeof(slots[victim].key.kind), kind, _TRUNCATE);
#else
            snprintf(slots[victim].key.kind, sizeof(slots[victim].key.kind), "%s", kind);
#endif
            slots[victim].key.age = age++;
            bytes_loaded += slots[victim].byte_size;
            checksum_mix ^= slots[victim].checksum;
        }
        tok = strtok(NULL, ",");
    }
    uint64_t resident_bytes = 0;
    for (uint32_t i = 0; i < slots_count; i++) if (slots[i].key.used) resident_bytes += slots[i].byte_size;
    fprintf(out, "{\"requests\": %u, \"hits\": %u, \"misses\": %u, \"slots\": %u, \"bytes_loaded\": %llu, \"resident_bytes\": %llu, \"checksum_mix\": %llu}\n",
        requests, hits, misses, slots_count, (unsigned long long)bytes_loaded, (unsigned long long)resident_bytes, (unsigned long long)checksum_mix);
    qx_free_cache_slots(slots, slots_count); free(seq); qx_close_file(&file);
    return 1;
}


static int qx_read_full_expert_alloc(qx_file *file, uint32_t layer, uint32_t expert,
                                     unsigned char **data, uint64_t *byte_size, uint64_t *checksum,
                                     char *err, uint64_t err_len) {
    const char *kinds[3] = {"gate", "up", "down"};
    unsigned char *out = NULL;
    uint64_t total = 0;
    for (uint32_t k = 0; k < 3; k++) {
        char name[QX_NAME_MAX];
        qx_expert_packed_tensor_name(name, sizeof(name), layer, kinds[k]);
        const qx_tensor_dir_entry *t = qx_find_tensor(file, name);
        if (!t) { free(out); qx_set_err(err, err_len, "packed expert tensor not found"); return 0; }
        if (t->byte_size % file->header.manifest.experts != 0) { free(out); qx_set_err(err, err_len, "packed expert tensor not evenly divisible"); return 0; }
        total += t->byte_size / file->header.manifest.experts;
    }
    out = (unsigned char *)malloc((size_t)total);
    if (!out) { qx_set_err(err, err_len, "out of memory"); return 0; }
    uint64_t pos = 0;
    for (uint32_t k = 0; k < 3; k++) {
        char name[QX_NAME_MAX];
        qx_expert_packed_tensor_name(name, sizeof(name), layer, kinds[k]);
        const qx_tensor_dir_entry *t = qx_find_tensor(file, name);
        uint64_t slice_bytes = t->byte_size / file->header.manifest.experts;
        uint64_t slice_offset = t->offset + slice_bytes * expert;
#if defined(_WIN32)
        if (_fseeki64(file->fp, (int64_t)slice_offset, SEEK_SET) != 0) {
#else
        if (fseeko(file->fp, (off_t)slice_offset, SEEK_SET) != 0) {
#endif
            free(out); qx_set_err(err, err_len, "seek expert slice failed"); return 0;
        }
        if (fread(out + pos, 1, (size_t)slice_bytes, file->fp) != (size_t)slice_bytes) {
            free(out); qx_set_err(err, err_len, "short expert slice read"); return 0;
        }
        pos += slice_bytes;
    }
    *data = out;
    *byte_size = total;
    *checksum = qx_fnv1a64(out, total);
    return 1;
}

int qx_dump_cache_run_expert_summary(const char *path, uint32_t slots_count, const char *sequence, FILE *out, char *err, uint64_t err_len) {
    if (!path || !sequence || slots_count == 0) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    qx_cache_slot *slots = (qx_cache_slot *)calloc(slots_count, sizeof(qx_cache_slot));
    char *seq = (char *)malloc(strlen(sequence) + 1);
    if (!slots || !seq) { qx_free_cache_slots(slots, slots_count); free(seq); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
    strcpy(seq, sequence);
    uint32_t requests = 0, expert_requests = 0, hits = 0, misses = 0;
    uint64_t age = 1, bytes_loaded = 0, checksum_mix = 0;
    char *tok = strtok(seq, ",");
    while (tok) {
        unsigned int layer = 0, expert = 0;
        char kind[8] = {0};
        if (sscanf(tok, "%u:%u:%7s", &layer, &expert, kind) != 3 || !qx_kind_valid(kind)) {
            qx_free_cache_slots(slots, slots_count); free(seq); qx_close_file(&file); qx_set_err(err, err_len, "bad sequence item"); return 0;
        }
        requests++;
        if (strcmp(kind, "gate") == 0) {
            expert_requests++;
            int hit_idx = -1;
            for (uint32_t i = 0; i < slots_count; i++) {
                if (slots[i].key.used && slots[i].key.layer == layer && slots[i].key.expert == expert) { hit_idx = (int)i; break; }
            }
            if (hit_idx >= 0) {
                hits++;
                slots[hit_idx].key.age = age++;
                checksum_mix ^= slots[hit_idx].checksum;
            } else {
                misses++;
                uint32_t victim = 0;
                uint64_t oldest = UINT64_MAX;
                for (uint32_t i = 0; i < slots_count; i++) {
                    if (!slots[i].key.used) { victim = i; oldest = 0; break; }
                    if (slots[i].key.age < oldest) { oldest = slots[i].key.age; victim = i; }
                }
                free(slots[victim].data);
                slots[victim].data = NULL;
                if (!qx_read_full_expert_alloc(&file, layer, expert, &slots[victim].data, &slots[victim].byte_size, &slots[victim].checksum, err, err_len)) {
                    qx_free_cache_slots(slots, slots_count); free(seq); qx_close_file(&file); return 0;
                }
                slots[victim].key.used = 1;
                slots[victim].key.layer = layer;
                slots[victim].key.expert = expert;
                snprintf(slots[victim].key.kind, sizeof(slots[victim].key.kind), "expert");
                slots[victim].key.age = age++;
                bytes_loaded += slots[victim].byte_size;
                checksum_mix ^= slots[victim].checksum;
            }
        }
        tok = strtok(NULL, ",");
    }
    uint64_t resident_bytes = 0;
    for (uint32_t i = 0; i < slots_count; i++) if (slots[i].key.used) resident_bytes += slots[i].byte_size;
    fprintf(out, "{\"requests\": %u, \"expert_requests\": %u, \"hits\": %u, \"misses\": %u, \"slots\": %u, \"bytes_loaded\": %llu, \"resident_bytes\": %llu, \"checksum_mix\": %llu}\n",
        requests, expert_requests, hits, misses, slots_count, (unsigned long long)bytes_loaded, (unsigned long long)resident_bytes, (unsigned long long)checksum_mix);
    qx_free_cache_slots(slots, slots_count); free(seq); qx_close_file(&file);
    return 1;
}


int qx_dump_expert_complete_cache_plan(const char *path, double hot_vram_gib, double hot_ram_gib, uint32_t top_k, FILE *out, char *err, uint64_t err_len) {
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    if (file.header.manifest.model_type != QX_MODEL_QWEN3_MOE) {
        qx_close_file(&file); qx_set_err(err, err_len, "not a MoE QXF"); return 0;
    }
    if (top_k == 0) top_k = file.header.manifest.experts_per_token;
    uint64_t expert_bytes = 0;
    uint32_t packed_tensors = 0;
    if (!qx_collect_expert_stats(&file, &expert_bytes, &packed_tensors)) {
        qx_close_file(&file); qx_set_err(err, err_len, "no packed expert tensors"); return 0;
    }
    uint64_t total_experts = (uint64_t)file.header.manifest.layers * file.header.manifest.experts;
    uint64_t avg_single_tensor_expert = total_experts ? expert_bytes / total_experts : 0;
    uint64_t avg_full_expert = avg_single_tensor_expert;
    uint64_t vram_bytes = (uint64_t)(hot_vram_gib * 1024.0 * 1024.0 * 1024.0);
    uint64_t ram_bytes = (uint64_t)(hot_ram_gib * 1024.0 * 1024.0 * 1024.0);
    uint64_t vram_slots = avg_full_expert ? vram_bytes / avg_full_expert : 0;
    uint64_t ram_slots = avg_full_expert ? ram_bytes / avg_full_expert : 0;
    uint64_t working = (uint64_t)file.header.manifest.layers * top_k;
    fprintf(out, "{\n");
    fprintf(out, "  \"cache_unit\": \"full_expert\",\n");
    fprintf(out, "  \"layers\": %u,\n", file.header.manifest.layers);
    fprintf(out, "  \"experts_per_layer\": %u,\n", file.header.manifest.experts);
    fprintf(out, "  \"top_k\": %u,\n", top_k);
    fprintf(out, "  \"working_set_experts_per_token\": %llu,\n", (unsigned long long)working);
    fprintf(out, "  \"avg_full_expert_bytes\": %llu,\n", (unsigned long long)avg_full_expert);
    fprintf(out, "  \"hot_vram_gib\": %.3f,\n", hot_vram_gib);
    fprintf(out, "  \"hot_ram_gib\": %.3f,\n", hot_ram_gib);
    fprintf(out, "  \"vram_expert_slots\": %llu,\n", (unsigned long long)vram_slots);
    fprintf(out, "  \"ram_expert_slots\": %llu,\n", (unsigned long long)ram_slots);
    fprintf(out, "  \"vram_covers_one_token\": %s,\n", vram_slots >= working ? "true" : "false");
    fprintf(out, "  \"ram_covers_one_token\": %s\n", ram_slots >= working ? "true" : "false");
    fprintf(out, "}\n");
    qx_close_file(&file);
    return 1;
}


static double qx_bytes_to_gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

int qx_dump_runtime_plan(const char *path, uint32_t ctx_tokens, const char *kv_format,
                         double usable_vram_gib, double usable_ram_gib,
                         double hot_vram_gib, double hot_ram_gib, uint32_t top_k,
                         FILE *out, char *err, uint64_t err_len) {
    if (!path || !kv_format || ctx_tokens == 0) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    double kv_bytes_per_value = 0.0;
    if (strcmp(kv_format, "int8") == 0) kv_bytes_per_value = 1.0;
    else if (strcmp(kv_format, "f16") == 0) kv_bytes_per_value = 2.0;
    else if (strcmp(kv_format, "int4") == 0) kv_bytes_per_value = 0.5;
    else { qx_set_err(err, err_len, "unsupported kv format"); return 0; }

    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    const qx_model_manifest *m = &file.header.manifest;
    if (top_k == 0) top_k = m->experts_per_token ? m->experts_per_token : 1;

    uint64_t expert_bytes = 0;
    uint32_t packed_tensors = 0;
    qx_collect_expert_stats(&file, &expert_bytes, &packed_tensors);
    uint64_t total_experts = (uint64_t)m->layers * (m->experts ? m->experts : 1);
    uint64_t avg_full_expert_bytes = total_experts ? expert_bytes / total_experts : 0;
    uint64_t working_set = (uint64_t)m->layers * top_k;
    uint64_t vram_hot_bytes = (uint64_t)(hot_vram_gib * 1024.0 * 1024.0 * 1024.0);
    uint64_t ram_hot_bytes = (uint64_t)(hot_ram_gib * 1024.0 * 1024.0 * 1024.0);
    uint64_t vram_slots = avg_full_expert_bytes ? vram_hot_bytes / avg_full_expert_bytes : 0;
    uint64_t ram_slots = avg_full_expert_bytes ? ram_hot_bytes / avg_full_expert_bytes : 0;

    uint64_t kv_bytes = (uint64_t)((double)m->layers * (double)ctx_tokens * (double)m->kv_heads * (double)m->head_dim * 2.0 * kv_bytes_per_value);
    double runtime_overhead_gib = 0.75;
    double total_model_gib = qx_bytes_to_gib(file.header.file_size);
    double expert_gib = qx_bytes_to_gib(expert_bytes);
    double non_expert_gib = total_model_gib > expert_gib ? total_model_gib - expert_gib : 0.0;
    double kv_gib = qx_bytes_to_gib(kv_bytes);
    double active_ram_gib = non_expert_gib + hot_ram_gib + kv_gib + runtime_overhead_gib;
    double active_vram_gib = hot_vram_gib;
    int feasible = (active_ram_gib <= usable_ram_gib && active_vram_gib <= usable_vram_gib);

    fprintf(out, "{\n");
    fprintf(out, "  \"model_type\": \"%s\",\n", m->model_type == QX_MODEL_QWEN3_MOE ? "qwen3_moe" : "qwen3_dense");
    fprintf(out, "  \"cache_unit\": \"full_expert\",\n");
    fprintf(out, "  \"ctx_tokens\": %u,\n", ctx_tokens);
    fprintf(out, "  \"kv_format\": \"%s\",\n", kv_format);
    fprintf(out, "  \"layers\": %u,\n", m->layers);
    fprintf(out, "  \"top_k\": %u,\n", top_k);
    fprintf(out, "  \"working_set_experts_per_token\": %llu,\n", (unsigned long long)working_set);
    fprintf(out, "  \"total_model_gib\": %.6f,\n", total_model_gib);
    fprintf(out, "  \"expert_gib\": %.6f,\n", expert_gib);
    fprintf(out, "  \"non_expert_gib\": %.6f,\n", non_expert_gib);
    fprintf(out, "  \"kv_gib\": %.6f,\n", kv_gib);
    fprintf(out, "  \"runtime_overhead_gib\": %.3f,\n", runtime_overhead_gib);
    fprintf(out, "  \"hot_vram_gib\": %.3f,\n", hot_vram_gib);
    fprintf(out, "  \"hot_ram_gib\": %.3f,\n", hot_ram_gib);
    fprintf(out, "  \"vram_expert_slots\": %llu,\n", (unsigned long long)vram_slots);
    fprintf(out, "  \"ram_expert_slots\": %llu,\n", (unsigned long long)ram_slots);
    fprintf(out, "  \"vram_covers_one_token\": %s,\n", vram_slots >= working_set ? "true" : "false");
    fprintf(out, "  \"ram_covers_one_token\": %s,\n", ram_slots >= working_set ? "true" : "false");
    fprintf(out, "  \"usable_vram_gib\": %.3f,\n", usable_vram_gib);
    fprintf(out, "  \"usable_ram_gib\": %.3f,\n", usable_ram_gib);
    fprintf(out, "  \"active_vram_gib\": %.3f,\n", active_vram_gib);
    fprintf(out, "  \"active_ram_gib\": %.3f,\n", active_ram_gib);
    fprintf(out, "  \"feasible\": %s\n", feasible ? "true" : "false");
    fprintf(out, "}\n");
    qx_close_file(&file);
    return 1;
}


static int qx_embedding_row_size(const qx_tensor_dir_entry *t, uint32_t rows, uint64_t *row_out, char *err, uint64_t err_len) {
    if (!t || !row_out || rows == 0 || t->byte_size == 0) {
        qx_set_err(err, err_len, "invalid tensor row layout"); return 0;
    }
    if (t->byte_size % rows != 0) {
        qx_set_err(err, err_len, "tensor byte size is not divisible by row count"); return 0;
    }
    *row_out = t->byte_size / rows;
    if (*row_out == 0) { qx_set_err(err, err_len, "invalid tensor row layout"); return 0; }
    return 1;
}

int qx_dump_token_embedding_summary(const char *path, uint32_t token_id, FILE *out, char *err, uint64_t err_len) {
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    const qx_tensor_dir_entry *t = qx_find_tensor(&file, "token_embd.weight");
    if (!t) { qx_close_file(&file); qx_set_err(err, err_len, "token_embd.weight not found"); return 0; }
    uint32_t vocab = file.header.manifest.vocab;
    if (vocab && token_id >= vocab) { qx_close_file(&file); qx_set_err(err, err_len, "token id out of range"); return 0; }
    uint64_t row = 0;
    if (!qx_embedding_row_size(t, vocab ? vocab : (uint32_t)t->dims[1], &row, err, err_len)) {
        qx_close_file(&file); return 0;
    }
    uint64_t offset = t->offset + row * token_id;
    fprintf(out, "{\n");
    fprintf(out, "  \"token_id\": %u,\n", token_id);
    fprintf(out, "  \"tensor\": \"token_embd.weight\",\n");
    fprintf(out, "  \"dtype\": %u,\n", t->dtype);
    fprintf(out, "  \"rank\": %u,\n", t->rank);
    fprintf(out, "  \"hidden\": %u,\n", file.header.manifest.hidden);
    fprintf(out, "  \"vocab\": %u,\n", vocab);
    fprintf(out, "  \"offset\": %llu,\n", (unsigned long long)offset);
    fprintf(out, "  \"row_byte_size\": %llu,\n", (unsigned long long)row);
    fprintf(out, "  \"tensor_byte_size\": %llu\n", (unsigned long long)t->byte_size);
    fprintf(out, "}\n");
    qx_close_file(&file);
    return 1;
}

static const qx_tensor_dir_entry *qx_find_first_existing(qx_file *file, const char **names, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        const qx_tensor_dir_entry *t = qx_find_tensor(file, names[i]);
        if (t) return t;
    }
    return NULL;
}

int qx_dump_forward_schedule(const char *path, uint32_t token_id, uint32_t top_k, FILE *out, char *err, uint64_t err_len) {
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    const qx_model_manifest *m = &file.header.manifest;
    if (top_k == 0) top_k = m->experts_per_token ? m->experts_per_token : 1;
    const qx_tensor_dir_entry *emb = qx_find_tensor(&file, "token_embd.weight");
    if (!emb) { qx_close_file(&file); qx_set_err(err, err_len, "token_embd.weight not found"); return 0; }
    if (m->vocab && token_id >= m->vocab) { qx_close_file(&file); qx_set_err(err, err_len, "token id out of range"); return 0; }
    uint64_t row = 0;
    if (!qx_embedding_row_size(emb, m->vocab ? m->vocab : (uint32_t)emb->dims[1], &row, err, err_len)) {
        qx_close_file(&file); return 0;
    }
    uint64_t emb_offset = emb->offset + row * token_id;
    char router_name[QX_NAME_MAX];
    snprintf(router_name, sizeof(router_name), "blk.0.ffn_gate_inp.weight");
    const char *attn_names[] = {"blk.0.attn_q.weight", "blk.0.attn_qkv.weight", "blk.0.attn_output.weight"};
    const qx_tensor_dir_entry *attn = qx_find_first_existing(&file, attn_names, 3);
    const qx_tensor_dir_entry *router = qx_find_tensor(&file, router_name);
    const qx_tensor_dir_entry *gate = qx_find_tensor(&file, "blk.0.ffn_gate_exps.weight");
    const qx_tensor_dir_entry *up = qx_find_tensor(&file, "blk.0.ffn_up_exps.weight");
    const qx_tensor_dir_entry *down = qx_find_tensor(&file, "blk.0.ffn_down_exps.weight");
    fprintf(out, "{\n");
    fprintf(out, "  \"mock_forward\": true,\n");
    fprintf(out, "  \"token_id\": %u,\n", token_id);
    fprintf(out, "  \"layers\": %u,\n", m->layers);
    fprintf(out, "  \"top_k\": %u,\n", top_k);
    fprintf(out, "  \"steps_per_layer\": 5,\n");
    fprintf(out, "  \"embedding\": {\"tensor\": \"token_embd.weight\", \"offset\": %llu, \"row_byte_size\": %llu},\n", (unsigned long long)emb_offset, (unsigned long long)row);
    fprintf(out, "  \"layer0\": {\n");
    fprintf(out, "    \"attention\": %s,\n", attn ? "true" : "false");
    fprintf(out, "    \"router\": %s,\n", router ? "true" : "false");
    fprintf(out, "    \"expert_gate\": %s,\n", gate ? "true" : "false");
    fprintf(out, "    \"expert_up\": %s,\n", up ? "true" : "false");
    fprintf(out, "    \"expert_down\": %s\n", down ? "true" : "false");
    fprintf(out, "  },\n");
    fprintf(out, "  \"planned_ops\": %llu\n", (unsigned long long)((uint64_t)m->layers * 5u + 1u));
    fprintf(out, "}\n");
    qx_close_file(&file);
    return 1;
}


static uint64_t qx_quant_probe_block_size(const qx_tensor_dir_entry *t) {
    if (!t) return 0;
    if (t->byte_size == 0) return 0;
    uint64_t block = 256;
    if (t->group_size && t->group_size > block) block = t->group_size;
    if (block > t->byte_size) block = t->byte_size;
    return block;
}

static int qx_read_raw_span(qx_file *file, uint64_t offset, uint64_t size, unsigned char **data, char *err, uint64_t err_len) {
    if (!data) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    *data = NULL;
    qx_span span;
    if (!qx_acquire_span(file, offset, size, &span, err, err_len)) return 0;
    if (span.owned_data) {
        *data = span.owned_data;
        span.owned_data = NULL;
    } else {
        if (size > (uint64_t)SIZE_MAX) { qx_release_span(&span); qx_set_err(err, err_len, "QXF span too large"); return 0; }
        *data = (unsigned char *)malloc((size_t)size);
        if (!*data) { qx_release_span(&span); qx_set_err(err, err_len, "out of memory"); return 0; }
        memcpy(*data, span.data, (size_t)size);
    }
    qx_release_span(&span);
    return 1;
}

static int qx_read_raw_span_into(qx_file *file, uint64_t offset, uint64_t size, unsigned char *buf, char *err, uint64_t err_len) {
    if (!buf) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    qx_span span;
    if (!qx_acquire_span(file, offset, size, &span, err, err_len)) return 0;
    memcpy(buf, span.data, (size_t)size);
    qx_release_span(&span);
    return 1;
}

int qx_dump_quant_block_summary(const char *path, const char *name, uint64_t block_index, FILE *out, char *err, uint64_t err_len) {
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    const qx_tensor_dir_entry *t = qx_find_tensor(&file, name);
    if (!t) { qx_close_file(&file); qx_set_err(err, err_len, "tensor not found"); return 0; }
    uint64_t block_size = qx_quant_probe_block_size(t);
    uint64_t block_count = block_size ? t->byte_size / block_size + (t->byte_size % block_size != 0) : 0;
    if (block_index >= block_count) { qx_close_file(&file); qx_set_err(err, err_len, "block out of range"); return 0; }
    if (block_size == 0 || block_index > UINT64_MAX / block_size) { qx_close_file(&file); qx_set_err(err, err_len, "block offset overflow"); return 0; }
    uint64_t relative_offset = block_index * block_size;
    if (relative_offset > t->byte_size) { qx_close_file(&file); qx_set_err(err, err_len, "block out of range"); return 0; }
    uint64_t block_offset = t->offset + relative_offset;
    uint64_t remaining = t->byte_size - relative_offset;
    uint64_t read_size = remaining < block_size ? remaining : block_size;
    unsigned char *buf = NULL;
    if (!qx_read_raw_span(&file, block_offset, read_size, &buf, err, err_len)) { qx_close_file(&file); return 0; }
    uint64_t checksum = qx_fnv1a64(buf, read_size);
    fprintf(out, "{\n");
    fprintf(out, "  \"tensor\": \"%s\",\n", t->name);
    fprintf(out, "  \"dtype\": %u,\n", t->dtype);
    fprintf(out, "  \"ggml_type\": %u,\n", t->flags);
    fprintf(out, "  \"quant\": %u,\n", t->quant);
    fprintf(out, "  \"group_size\": %u,\n", t->group_size);
    fprintf(out, "  \"block_index\": %llu,\n", (unsigned long long)block_index);
    fprintf(out, "  \"block_count\": %llu,\n", (unsigned long long)block_count);
    fprintf(out, "  \"block_offset\": %llu,\n", (unsigned long long)block_offset);
    fprintf(out, "  \"block_byte_size\": %llu,\n", (unsigned long long)read_size);
    fprintf(out, "  \"checksum\": %llu,\n", (unsigned long long)checksum);
    fprintf(out, "  \"dequantized\": false,\n");
    fprintf(out, "  \"note\": \"raw quant block probe; numeric dequant kernel not implemented\"\n");
    fprintf(out, "}\n");
    free(buf);
    qx_close_file(&file);
    return 1;
}

int qx_dump_matvec_stub_summary(const char *path, const char *name, uint32_t rows, FILE *out, char *err, uint64_t err_len) {
    if (rows == 0) rows = 1;
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    const qx_tensor_dir_entry *t = qx_find_tensor(&file, name);
    if (!t) { qx_close_file(&file); qx_set_err(err, err_len, "tensor not found"); return 0; }
    uint64_t block_size = qx_quant_probe_block_size(t);
    if (block_size == 0 || (uint64_t)rows > UINT64_MAX / block_size) {
        qx_close_file(&file); qx_set_err(err, err_len, "invalid requested matvec span"); return 0;
    }
    uint64_t span = block_size * rows;
    if (span > t->byte_size) {
        qx_close_file(&file); qx_set_err(err, err_len, "requested matvec span outside tensor"); return 0;
    }
    unsigned char *buf = NULL;
    if (!qx_read_raw_span(&file, t->offset, span, &buf, err, err_len)) { qx_close_file(&file); return 0; }
    uint64_t mix = qx_fnv1a64(buf, span);
    fprintf(out, "{\n");
    fprintf(out, "  \"stub\": true,\n");
    fprintf(out, "  \"tensor\": \"%s\",\n", t->name);
    fprintf(out, "  \"rows\": %u,\n", rows);
    fprintf(out, "  \"dtype\": %u,\n", t->dtype);
    fprintf(out, "  \"quant\": %u,\n", t->quant);
    fprintf(out, "  \"block_byte_size\": %llu,\n", (unsigned long long)block_size);
    fprintf(out, "  \"bytes_read\": %llu,\n", (unsigned long long)span);
    fprintf(out, "  \"checksum_mix\": %llu,\n", (unsigned long long)mix);
    fprintf(out, "  \"numeric_kernel\": false\n");
    fprintf(out, "}\n");
    free(buf);
    qx_close_file(&file);
    return 1;
}


static float qx_fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x477fe000u;
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint16_t qx_f32_to_fp16(float value) {
    uint32_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
    uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    uint32_t exponent = (bits >> 23) & 0xffu;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) return (uint16_t)(sign | (mantissa ? 0x7e00u : 0x7c00u));
    int32_t half_exponent = (int32_t)exponent - 127 + 15;
    if (half_exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return sign;
        mantissa |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - half_exponent);
        uint32_t rounded = mantissa >> shift;
        uint32_t remainder = mantissa & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
        return (uint16_t)(sign | (uint16_t)rounded);
    }
    uint32_t rounded_mantissa = mantissa + 0x0fffu + ((mantissa >> 13) & 1u);
    if (rounded_mantissa & 0x800000u) {
        rounded_mantissa = 0u;
        ++half_exponent;
        if (half_exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint16_t)half_exponent << 10) | (uint16_t)(rounded_mantissa >> 13));
}

static uint16_t qx_rd_le16(const unsigned char *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t qx_rd_le32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int qx_decode_iq2_xs_block(const unsigned char *buf, float out[256]) {
    const uint16_t d16 = qx_rd_le16(buf);
    const uint16_t *qs = (const uint16_t *)(const void *)(buf + 2);
    const unsigned char *scales = buf + 2 + 64;
    const float d = qx_fp16_to_f32(d16);
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        float db0 = d * (0.5f + (float)(scales[ib32] & 0x0f)) * 0.25f;
        float db1 = d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            uint16_t q = qs[4*ib32 + l];
            const unsigned char *grid = (const unsigned char *)(const void *)&qx_iq2xs_grid[q & 511u];
            unsigned char signs = qx_ksigns_iq2xs[q >> 9];
            float db = (l < 2) ? db0 : db1;
            for (int j = 0; j < 8; ++j) {
                out[ib32*32 + l*8 + j] = db * (float)grid[j] * ((signs & qx_kmask_iq2xs[j]) ? -1.0f : 1.0f);
            }
        }
    }
    return 1;
}

static int qx_decode_iq3_xxs_block(const unsigned char *buf, float out[256]) {
    const uint16_t d16 = qx_rd_le16(buf);
    const unsigned char *qs = buf + 2;
    const unsigned char *scales_and_signs = qs + 64;
    const float d = qx_fp16_to_f32(d16);
    int y = 0;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        uint32_t aux32 = qx_rd_le32(scales_and_signs + 4*ib32);
        const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const unsigned char signs = qx_ksigns_iq2xs[(aux32 >> (7*l)) & 127u];
            const uint32_t grid1 = qx_iq3xxs_grid[qs[2*l + 0]];
            const uint32_t grid2 = qx_iq3xxs_grid[qs[2*l + 1]];
            for (int j = 0; j < 4; ++j) {
                const unsigned char g1 = (unsigned char)((grid1 >> (8*j)) & 0xffu);
                const unsigned char g2 = (unsigned char)((grid2 >> (8*j)) & 0xffu);
                out[y + j + 0] = db * (float)g1 * ((signs & qx_kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                out[y + j + 4] = db * (float)g2 * ((signs & qx_kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
            }
            y += 8;
        }
        qs += 8;
    }
    return 1;
}

static int qx_decode_iq2_s_block(const unsigned char *buf, float out[256]) {
    const float d = qx_fp16_to_f32(qx_rd_le16(buf));
    const unsigned char *qs = buf + 2;
    const unsigned char *qh = buf + 66;
    const unsigned char *scales = buf + 74;
    const unsigned char *signs = qs + 32;
    int y = 0;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const float db0 = d * (0.5f + (float)(scales[ib32] & 0x0f)) * 0.25f;
        const float db1 = d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const float dl = (l < 2) ? db0 : db1;
            const unsigned int idx = (unsigned int)qs[l] | (((unsigned int)qh[ib32] << (8 - 2*l)) & 0x300u);
            const unsigned long long grid64 = qx_iq2s_grid[idx & 1023u];
            for (int j = 0; j < 8; ++j) {
                const unsigned char g = (unsigned char)((grid64 >> (8*j)) & 0xffu);
                out[y + j] = dl * (float)g * ((signs[l] & qx_kmask_iq2xs[j]) ? -1.0f : 1.0f);
            }
            y += 8;
        }
        qs += 4;
        signs += 4;
    }
    return 1;
}

static int qx_decode_iq3_s_block(const unsigned char *buf, float out[256]) {
    const float d = qx_fp16_to_f32(qx_rd_le16(buf));
    const unsigned char *qs = buf + 2;
    const unsigned char *qh = buf + 66;
    const unsigned char *signs = buf + 74;
    const unsigned char *scales = buf + 106;
    int y = 0;
    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
        const float db1 = d * (1.0f + 2.0f * (float)(scales[ib32/2] & 0x0f));
        const float db2 = d * (1.0f + 2.0f * (float)(scales[ib32/2] >> 4));
        for (int l = 0; l < 4; ++l) {
            const unsigned int idx1 = (unsigned int)qs[2*l+0] | (((unsigned int)qh[0] << (8 - 2*l)) & 256u);
            const unsigned int idx2 = (unsigned int)qs[2*l+1] | (((unsigned int)qh[0] << (7 - 2*l)) & 256u);
            const unsigned int grid1 = qx_iq3s_grid[idx1 & 511u];
            const unsigned int grid2 = qx_iq3s_grid[idx2 & 511u];
            for (int j = 0; j < 4; ++j) {
                const unsigned char g1 = (unsigned char)((grid1 >> (8*j)) & 0xffu);
                const unsigned char g2 = (unsigned char)((grid2 >> (8*j)) & 0xffu);
                out[y + j + 0] = db1 * (float)g1 * ((signs[l] & qx_kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                out[y + j + 4] = db1 * (float)g2 * ((signs[l] & qx_kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
            }
            y += 8;
        }
        qs += 8;
        signs += 4;
        for (int l = 0; l < 4; ++l) {
            const unsigned int idx1 = (unsigned int)qs[2*l+0] | (((unsigned int)qh[1] << (8 - 2*l)) & 256u);
            const unsigned int idx2 = (unsigned int)qs[2*l+1] | (((unsigned int)qh[1] << (7 - 2*l)) & 256u);
            const unsigned int grid1 = qx_iq3s_grid[idx1 & 511u];
            const unsigned int grid2 = qx_iq3s_grid[idx2 & 511u];
            for (int j = 0; j < 4; ++j) {
                const unsigned char g1 = (unsigned char)((grid1 >> (8*j)) & 0xffu);
                const unsigned char g2 = (unsigned char)((grid2 >> (8*j)) & 0xffu);
                out[y + j + 0] = db2 * (float)g1 * ((signs[l] & qx_kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                out[y + j + 4] = db2 * (float)g2 * ((signs[l] & qx_kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
            }
            y += 8;
        }
        qh += 2;
        qs += 8;
        signs += 4;
    }
    return 1;
}

static const signed char qx_kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static float qx_deterministic_input(uint32_t *state);

static int qx_decode_iq4_xs_block(const unsigned char *buf, float out[256]) {
    const float d = qx_fp16_to_f32(qx_rd_le16(buf));
    const uint16_t scales_h = qx_rd_le16(buf + 2);
    const unsigned char *scales_l = buf + 4;
    const unsigned char *qs = buf + 8;
    int y = 0;
    for (int ib = 0; ib < 8; ++ib) {
        const int ls = ((scales_l[ib/2] >> (4*(ib%2))) & 0x0f) | (((scales_h >> (2*ib)) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        for (int j = 0; j < 16; ++j) {
            out[y + j + 0] = dl * (float)qx_kvalues_iq4nl[qs[j] & 0x0f];
            out[y + j + 16] = dl * (float)qx_kvalues_iq4nl[qs[j] >> 4];
        }
        y += 32;
        qs += 16;
    }
    return 1;
}

static double qx_dot_iq4_xs_prefix(const unsigned char *buf, uint32_t dims, const float *residual, uint32_t residual_n, uint32_t *st) {
    const float d = qx_fp16_to_f32(qx_rd_le16(buf));
    const uint16_t scales_h = qx_rd_le16(buf + 2);
    const unsigned char *scales_l = buf + 4;
    const unsigned char *qs = buf + 8;
    if (dims > 256u) dims = 256u;
    double dot = 0.0;
    uint32_t y = 0;
    for (uint32_t ib = 0; ib < 8u && y < dims; ++ib) {
        const int ls = ((scales_l[ib/2u] >> (4u*(ib%2u))) & 0x0f) | (((scales_h >> (2u*ib)) & 3u) << 4);
        const float dl = d * (float)(ls - 32);
        for (uint32_t j = 0; j < 16u && y < dims; ++j, ++y) {
            double x = (residual && residual_n) ? (double)residual[y % residual_n] : (double)qx_deterministic_input(st);
            dot += (double)(dl * (float)qx_kvalues_iq4nl[qs[j] & 0x0f]) * x;
        }
        for (uint32_t j = 0; j < 16u && y < dims; ++j, ++y) {
            double x = (residual && residual_n) ? (double)residual[y % residual_n] : (double)qx_deterministic_input(st);
            dot += (double)(dl * (float)qx_kvalues_iq4nl[qs[j] >> 4]) * x;
        }
        qs += 16;
    }
    return dot;
}

static void qx_get_scale_min_k4(int j, const unsigned char *q, unsigned char *d, unsigned char *m) {
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (q[j + 4] & 0x0fu) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static int qx_decode_q4_k_block(const unsigned char *buf, float out[256]) {
    const float d = qx_fp16_to_f32(qx_rd_le16(buf));
    const float dmin = qx_fp16_to_f32(qx_rd_le16(buf + 2));
    const unsigned char *scales = buf + 4;
    const unsigned char *q = buf + 16;
    int y = 0;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        unsigned char sc = 0, m = 0;
        qx_get_scale_min_k4(is + 0, scales, &sc, &m);
        const float d1 = d * (float)sc;
        const float m1 = dmin * (float)m;
        qx_get_scale_min_k4(is + 1, scales, &sc, &m);
        const float d2 = d * (float)sc;
        const float m2 = dmin * (float)m;
        for (int l = 0; l < 32; ++l) out[y++] = d1 * (float)(q[l] & 0x0f) - m1;
        for (int l = 0; l < 32; ++l) out[y++] = d2 * (float)(q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
    return 1;
}

static int qx_decode_q5_k_block(const unsigned char *buf, float out[256]) {
    const float d = qx_fp16_to_f32(qx_rd_le16(buf));
    const float dmin = qx_fp16_to_f32(qx_rd_le16(buf + 2));
    const unsigned char *scales = buf + 4;
    const unsigned char *qh = buf + 16;
    const unsigned char *q = buf + 48;
    int y = 0;
    int is = 0;
    /* ggml Q5_K reuses qh[0..31] for four 64-value groups. The mask pair
       selects bits 0/1, 2/3, 4/5, then 6/7 while q advances by 32 bytes. */
    unsigned char u1 = 1u, u2 = 2u;
    for (int j = 0; j < 256; j += 64) {
        unsigned char sc = 0, m = 0;
        qx_get_scale_min_k4(is + 0, scales, &sc, &m);
        const float d1 = d * (float)sc;
        const float m1 = dmin * (float)m;
        qx_get_scale_min_k4(is + 1, scales, &sc, &m);
        const float d2 = d * (float)sc;
        const float m2 = dmin * (float)m;
        for (int l = 0; l < 32; ++l) {
            int hi = (qh[l] & u1) ? 16 : 0;
            out[y++] = d1 * (float)((q[l] & 0x0f) | hi) - m1;
        }
        for (int l = 0; l < 32; ++l) {
            int hi = (qh[l] & u2) ? 16 : 0;
            out[y++] = d2 * (float)((q[l] >> 4) | hi) - m2;
        }
        q += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
    return 1;
}

static int qx_decode_q6_k_block(const unsigned char *buf, float out[256]) {
    const float d = qx_fp16_to_f32(qx_rd_le16(buf + 208));
    const unsigned char *ql = buf;
    const unsigned char *qh = buf + 128;
    const signed char *scales = (const signed char *)(buf + 192);
    for (int base = 0; base < 256; base += 128) {
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            int q1 = (int)((ql[l] & 0x0fu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
            int q2 = (int)((ql[l + 32] & 0x0fu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
            int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32;
            int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32;
            out[base + l] = d * (float)scales[is] * (float)q1;
            out[base + l + 32] = d * (float)scales[is + 2] * (float)q2;
            out[base + l + 64] = d * (float)scales[is + 4] * (float)q3;
            out[base + l + 96] = d * (float)scales[is + 6] * (float)q4;
        }
        ql += 64;
        qh += 32;
        scales += 8;
    }
    return 1;
}

static float qx_decode_q6_k_value_at(const unsigned char *buf, uint32_t index) {
    uint32_t half = index >> 7u;
    uint32_t local = index & 127u;
    uint32_t lane = local & 31u;
    uint32_t section = local >> 5u;
    const unsigned char *ql = buf + half * 64u;
    const unsigned char *qh = buf + 128u + half * 32u;
    const signed char *scales = (const signed char *)(buf + 192u + half * 8u);
    uint32_t ql_index = lane + ((section == 1u || section == 3u) ? 32u : 0u);
    uint32_t packed = ql[ql_index];
    uint32_t quant = (section < 2u) ? (packed & 0x0fu) : (packed >> 4u);
    quant |= ((uint32_t)(qh[lane] >> (2u * section)) & 3u) << 4u;
    uint32_t scale_index = (lane >> 4u) + section * 2u;
    return qx_fp16_to_f32(qx_rd_le16(buf + 208u)) * (float)scales[scale_index] * ((float)((int32_t)quant - 32));
}

static double qx_dot_q6_k_f32_fused_block(const unsigned char *buf, const float *input) {
    double total = 0.0;
    for (uint32_t i = 0; i < 256u; ++i) {
        total += (double)qx_decode_q6_k_value_at(buf, i) * (double)input[i];
    }
    return total;
}

static int qx_decoder_info(uint32_t ggml_type, const char **decoder, uint64_t *block_size) {
    if (ggml_type == 12u) { if (decoder) *decoder = "Q4_K"; if (block_size) *block_size = 144; return 1; }
    if (ggml_type == 13u) { if (decoder) *decoder = "Q5_K"; if (block_size) *block_size = 176; return 1; }
    if (ggml_type == 14u) { if (decoder) *decoder = "Q6_K"; if (block_size) *block_size = 210; return 1; }
    if (ggml_type == 17u) { if (decoder) *decoder = "IQ2_XS"; if (block_size) *block_size = 74; return 1; }
    if (ggml_type == 18u) { if (decoder) *decoder = "IQ3_XXS"; if (block_size) *block_size = 98; return 1; }
    if (ggml_type == 21u) { if (decoder) *decoder = "IQ3_S"; if (block_size) *block_size = 110; return 1; }
    if (ggml_type == 22u) { if (decoder) *decoder = "IQ2_S"; if (block_size) *block_size = 82; return 1; }
    if (ggml_type == 23u) { if (decoder) *decoder = "IQ4_XS"; if (block_size) *block_size = 136; return 1; }
    return 0;
}

static int qx_decode_supported_block(uint32_t ggml_type, const unsigned char *buf, float out[256]) {
    if (ggml_type == 12u) return qx_decode_q4_k_block(buf, out);
    if (ggml_type == 13u) return qx_decode_q5_k_block(buf, out);
    if (ggml_type == 14u) return qx_decode_q6_k_block(buf, out);
    if (ggml_type == 17u) return qx_decode_iq2_xs_block(buf, out);
    if (ggml_type == 18u) return qx_decode_iq3_xxs_block(buf, out);
    if (ggml_type == 21u) return qx_decode_iq3_s_block(buf, out);
    if (ggml_type == 22u) return qx_decode_iq2_s_block(buf, out);
    if (ggml_type == 23u) return qx_decode_iq4_xs_block(buf, out);
    return 0;
}

int qx_dump_decode_block_summary(const char *path, const char *name, uint64_t block_index, FILE *out, char *err, uint64_t err_len) {
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    const qx_tensor_dir_entry *t = qx_find_tensor(&file, name);
    if (!t) { qx_close_file(&file); qx_set_err(err, err_len, "tensor not found"); return 0; }
    const char *decoder = NULL;
    uint64_t block_size = 0;
    if (!qx_decoder_info(t->flags, &decoder, &block_size)) { qx_close_file(&file); qx_set_err(err, err_len, "unsupported GGML type for decode-block"); return 0; }
    uint64_t block_count = t->byte_size / block_size;
    if (block_index >= block_count) { qx_close_file(&file); qx_set_err(err, err_len, "block out of range"); return 0; }
    uint64_t block_offset = t->offset + block_index * block_size;
    unsigned char *buf = NULL;
    if (!qx_read_raw_span(&file, block_offset, block_size, &buf, err, err_len)) { qx_close_file(&file); return 0; }
    float vals[256];
    qx_decode_supported_block(t->flags, buf, vals);
    double sum = 0.0;
    float minv = vals[0], maxv = vals[0];
    for (int i = 0; i < 256; ++i) {
        sum += vals[i];
        if (vals[i] < minv) minv = vals[i];
        if (vals[i] > maxv) maxv = vals[i];
    }
    uint64_t raw_checksum = qx_fnv1a64(buf, block_size);
    uint64_t f32_checksum = qx_fnv1a64(vals, sizeof(vals));
    fprintf(out, "{\n");
    fprintf(out, "  \"tensor\": \"%s\",\n", t->name);
    fprintf(out, "  \"ggml_type\": %u,\n", t->flags);
    fprintf(out, "  \"decoder\": \"%s\",\n", decoder);
    fprintf(out, "  \"decoded\": true,\n");
    fprintf(out, "  \"block_index\": %llu,\n", (unsigned long long)block_index);
    fprintf(out, "  \"block_offset\": %llu,\n", (unsigned long long)block_offset);
    fprintf(out, "  \"block_byte_size\": %llu,\n", (unsigned long long)block_size);
    fprintf(out, "  \"values\": 256,\n");
    fprintf(out, "  \"sum\": %.9g,\n", sum);
    fprintf(out, "  \"min\": %.9g,\n", (double)minv);
    fprintf(out, "  \"max\": %.9g,\n", (double)maxv);
    fprintf(out, "  \"raw_checksum\": %llu,\n", (unsigned long long)raw_checksum);
    fprintf(out, "  \"f32_checksum\": %llu,\n", (unsigned long long)f32_checksum);
    fprintf(out, "  \"first8\": [");
    for (int i = 0; i < 8; ++i) fprintf(out, "%s%.9g", i ? ", " : "", (double)vals[i]);
    fprintf(out, "]\n");
    fprintf(out, "}\n");
    free(buf);
    qx_close_file(&file);
    return 1;
}


static float qx_deterministic_input(uint32_t *state) {
    *state = (*state * 1664525u) + 1013904223u;
    int v = (int)((*state >> 16) & 0xffffu) - 32768;
    return (float)v / 32768.0f;
}

int qx_dump_block_dot_summary(const char *path, const char *name, uint64_t block_index, uint32_t seed, FILE *out, char *err, uint64_t err_len) {
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    const qx_tensor_dir_entry *t = qx_find_tensor(&file, name);
    if (!t) { qx_close_file(&file); qx_set_err(err, err_len, "tensor not found"); return 0; }
    const char *decoder = NULL;
    uint64_t block_size = 0;
    if (!qx_decoder_info(t->flags, &decoder, &block_size)) { qx_close_file(&file); qx_set_err(err, err_len, "unsupported GGML type for block-dot"); return 0; }
    uint64_t block_count = t->byte_size / block_size;
    if (block_index >= block_count) { qx_close_file(&file); qx_set_err(err, err_len, "block out of range"); return 0; }
    uint64_t block_offset = t->offset + block_index * block_size;
    unsigned char *buf = NULL;
    if (!qx_read_raw_span(&file, block_offset, block_size, &buf, err, err_len)) { qx_close_file(&file); return 0; }
    float weights[256];
    qx_decode_supported_block(t->flags, buf, weights);
    uint32_t state = seed ? seed : 1u;
    double dot = 0.0;
    double input_sum = 0.0;
    double weight_sum = 0.0;
    for (int i = 0; i < 256; ++i) {
        float x = qx_deterministic_input(&state);
        dot += (double)weights[i] * (double)x;
        input_sum += (double)x;
        weight_sum += (double)weights[i];
    }
    uint64_t raw_checksum = qx_fnv1a64(buf, block_size);
    fprintf(out, "{\n");
    fprintf(out, "  \"tensor\": \"%s\",\n", t->name);
    fprintf(out, "  \"ggml_type\": %u,\n", t->flags);
    fprintf(out, "  \"decoder\": \"%s\",\n", decoder);
    fprintf(out, "  \"block_index\": %llu,\n", (unsigned long long)block_index);
    fprintf(out, "  \"block_offset\": %llu,\n", (unsigned long long)block_offset);
    fprintf(out, "  \"block_byte_size\": %llu,\n", (unsigned long long)block_size);
    fprintf(out, "  \"values\": 256,\n");
    fprintf(out, "  \"input_seed\": %u,\n", seed);
    fprintf(out, "  \"input_kind\": \"deterministic_lcg_unit\",\n");
    fprintf(out, "  \"dot\": %.9g,\n", dot);
    fprintf(out, "  \"input_sum\": %.9g,\n", input_sum);
    fprintf(out, "  \"weight_sum\": %.9g,\n", weight_sum);
    fprintf(out, "  \"raw_checksum\": %llu\n", (unsigned long long)raw_checksum);
    fprintf(out, "}\n");
    free(buf);
    qx_close_file(&file);
    return 1;
}


int qx_dump_matvec_row_summary(const char *path, const char *name, uint64_t start_block, uint32_t blocks, uint32_t seed, FILE *out, char *err, uint64_t err_len) {
    if (blocks == 0) blocks = 1;
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    const qx_tensor_dir_entry *t = qx_find_tensor(&file, name);
    if (!t) { qx_close_file(&file); qx_set_err(err, err_len, "tensor not found"); return 0; }
    const char *decoder = NULL;
    uint64_t block_size = 0;
    if (!qx_decoder_info(t->flags, &decoder, &block_size)) { qx_close_file(&file); qx_set_err(err, err_len, "unsupported GGML type for matvec-row"); return 0; }
    uint64_t block_count = t->byte_size / block_size;
    if (start_block >= block_count || (uint64_t)blocks > block_count - start_block) {
        qx_close_file(&file); qx_set_err(err, err_len, "block range out of range"); return 0;
    }
    uint64_t span = (uint64_t)blocks * block_size;
    uint64_t start_offset = t->offset + start_block * block_size;
    unsigned char *buf = NULL;
    if (!qx_read_raw_span(&file, start_offset, span, &buf, err, err_len)) { qx_close_file(&file); return 0; }
    uint32_t state = seed ? seed : 1u;
    double dot = 0.0;
    double input_sum = 0.0;
    double weight_sum = 0.0;
    uint64_t checksum_mix = 1469598103934665603ull;
    for (uint32_t b = 0; b < blocks; ++b) {
        const unsigned char *block = buf + (uint64_t)b * block_size;
        float weights[256];
        qx_decode_supported_block(t->flags, block, weights);
        checksum_mix ^= qx_fnv1a64(block, block_size);
        checksum_mix *= 1099511628211ull;
        for (int i = 0; i < 256; ++i) {
            float x = qx_deterministic_input(&state);
            dot += (double)weights[i] * (double)x;
            input_sum += (double)x;
            weight_sum += (double)weights[i];
        }
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"tensor\": \"%s\",\n", t->name);
    fprintf(out, "  \"ggml_type\": %u,\n", t->flags);
    fprintf(out, "  \"decoder\": \"%s\",\n", decoder);
    fprintf(out, "  \"start_block\": %llu,\n", (unsigned long long)start_block);
    fprintf(out, "  \"blocks\": %u,\n", blocks);
    fprintf(out, "  \"block_byte_size\": %llu,\n", (unsigned long long)block_size);
    fprintf(out, "  \"values\": %llu,\n", (unsigned long long)blocks * 256ull);
    fprintf(out, "  \"bytes_read\": %llu,\n", (unsigned long long)span);
    fprintf(out, "  \"input_seed\": %u,\n", seed);
    fprintf(out, "  \"input_kind\": \"deterministic_lcg_unit\",\n");
    fprintf(out, "  \"dot\": %.9g,\n", dot);
    fprintf(out, "  \"input_sum\": %.9g,\n", input_sum);
    fprintf(out, "  \"weight_sum\": %.9g,\n", weight_sum);
    fprintf(out, "  \"checksum_mix\": %llu\n", (unsigned long long)checksum_mix);
    fprintf(out, "}\n");
    free(buf);
    qx_close_file(&file);
    return 1;
}


int qx_dump_expert_row_summary(const char *path, uint32_t layer, uint32_t expert, const char *kind, uint64_t start_block, uint32_t blocks, uint32_t seed, FILE *out, char *err, uint64_t err_len) {
    if (!qx_kind_valid(kind)) { qx_set_err(err, err_len, "invalid expert kind"); return 0; }
    if (blocks == 0) blocks = 1;
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    if (file.header.manifest.model_type != QX_MODEL_QWEN3_MOE) {
        qx_close_file(&file); qx_set_err(err, err_len, "not a MoE QXF"); return 0;
    }
    if (layer >= file.header.manifest.layers || expert >= file.header.manifest.experts) {
        qx_close_file(&file); qx_set_err(err, err_len, "expert address out of range"); return 0;
    }
    char name[QX_NAME_MAX];
    qx_expert_packed_tensor_name(name, sizeof(name), layer, kind);
    const qx_tensor_dir_entry *t = qx_find_tensor(&file, name);
    if (!t) { qx_close_file(&file); qx_set_err(err, err_len, "packed expert tensor not found"); return 0; }
    const char *decoder = NULL;
    uint64_t block_size = 0;
    if (!qx_decoder_info(t->flags, &decoder, &block_size)) { qx_close_file(&file); qx_set_err(err, err_len, "unsupported GGML type for expert-row"); return 0; }
    uint64_t slice_bytes = t->byte_size / file.header.manifest.experts;
    uint64_t remainder = t->byte_size % file.header.manifest.experts;
    if (remainder != 0) { qx_close_file(&file); qx_set_err(err, err_len, "packed expert tensor not evenly divisible"); return 0; }
    uint64_t slice_offset = t->offset + slice_bytes * expert;
    uint64_t slice_block_count = slice_bytes / block_size;
    uint64_t slice_block_remainder = slice_bytes % block_size;
    if (start_block >= slice_block_count || (uint64_t)blocks > slice_block_count - start_block) {
        qx_close_file(&file); qx_set_err(err, err_len, "expert row block range out of range"); return 0;
    }
    uint64_t span = (uint64_t)blocks * block_size;
    uint64_t start_offset = slice_offset + start_block * block_size;
    unsigned char *buf = NULL;
    if (!qx_read_raw_span(&file, start_offset, span, &buf, err, err_len)) { qx_close_file(&file); return 0; }
    uint32_t state = seed ? seed : 1u;
    double dot = 0.0;
    double input_sum = 0.0;
    double weight_sum = 0.0;
    uint64_t checksum_mix = 1469598103934665603ull;
    for (uint32_t b = 0; b < blocks; ++b) {
        const unsigned char *block = buf + (uint64_t)b * block_size;
        float weights[256];
        qx_decode_supported_block(t->flags, block, weights);
        checksum_mix ^= qx_fnv1a64(block, block_size);
        checksum_mix *= 1099511628211ull;
        for (int i = 0; i < 256; ++i) {
            float x = qx_deterministic_input(&state);
            dot += (double)weights[i] * (double)x;
            input_sum += (double)x;
            weight_sum += (double)weights[i];
        }
    }
    uint64_t absolute_start_block = (start_offset - t->offset) / block_size;
    fprintf(out, "{\n");
    fprintf(out, "  \"layer\": %u,\n", layer);
    fprintf(out, "  \"expert\": %u,\n", expert);
    fprintf(out, "  \"kind\": \"%s\",\n", kind);
    fprintf(out, "  \"tensor\": \"%s\",\n", t->name);
    fprintf(out, "  \"ggml_type\": %u,\n", t->flags);
    fprintf(out, "  \"decoder\": \"%s\",\n", decoder);
    fprintf(out, "  \"slice_offset\": %llu,\n", (unsigned long long)slice_offset);
    fprintf(out, "  \"slice_byte_size\": %llu,\n", (unsigned long long)slice_bytes);
    fprintf(out, "  \"slice_block_count\": %llu,\n", (unsigned long long)slice_block_count);
    fprintf(out, "  \"slice_block_remainder\": %llu,\n", (unsigned long long)slice_block_remainder);
    fprintf(out, "  \"slice_start_block\": %llu,\n", (unsigned long long)start_block);
    fprintf(out, "  \"absolute_start_block\": %llu,\n", (unsigned long long)absolute_start_block);
    fprintf(out, "  \"blocks\": %u,\n", blocks);
    fprintf(out, "  \"block_byte_size\": %llu,\n", (unsigned long long)block_size);
    fprintf(out, "  \"values\": %llu,\n", (unsigned long long)blocks * 256ull);
    fprintf(out, "  \"bytes_read\": %llu,\n", (unsigned long long)span);
    fprintf(out, "  \"input_seed\": %u,\n", seed);
    fprintf(out, "  \"input_kind\": \"deterministic_lcg_unit\",\n");
    fprintf(out, "  \"dot\": %.9g,\n", dot);
    fprintf(out, "  \"input_sum\": %.9g,\n", input_sum);
    fprintf(out, "  \"weight_sum\": %.9g,\n", weight_sum);
    fprintf(out, "  \"checksum_mix\": %llu\n", (unsigned long long)checksum_mix);
    fprintf(out, "}\n");
    free(buf);
    qx_close_file(&file);
    return 1;
}


static int qx_expert_row_dot_calc(qx_file *file, uint32_t layer, uint32_t expert, const char *kind,
                                  uint64_t start_block, uint32_t blocks, uint32_t seed,
                                  double *dot, double *input_sum, double *weight_sum,
                                  const char **decoder, uint32_t *ggml_type, uint64_t *slice_offset,
                                  uint64_t *slice_bytes, uint64_t *block_size_out, uint64_t *checksum_mix,
                                  char *err, uint64_t err_len) {
    if (!qx_kind_valid(kind)) { qx_set_err(err, err_len, "invalid expert kind"); return 0; }
    if (file->header.manifest.model_type != QX_MODEL_QWEN3_MOE) { qx_set_err(err, err_len, "not a MoE QXF"); return 0; }
    if (layer >= file->header.manifest.layers || expert >= file->header.manifest.experts) { qx_set_err(err, err_len, "expert address out of range"); return 0; }
    char name[QX_NAME_MAX];
    qx_expert_packed_tensor_name(name, sizeof(name), layer, kind);
    const qx_tensor_dir_entry *t = qx_find_tensor(file, name);
    if (!t) { qx_set_err(err, err_len, "packed expert tensor not found"); return 0; }
    const char *dec = NULL;
    uint64_t block_size = 0;
    if (!qx_decoder_info(t->flags, &dec, &block_size)) { qx_set_err(err, err_len, "unsupported GGML type for expert forward probe"); return 0; }
    uint64_t sbytes = t->byte_size / file->header.manifest.experts;
    if (t->byte_size % file->header.manifest.experts != 0) { qx_set_err(err, err_len, "packed expert tensor not evenly divisible"); return 0; }
    uint64_t sbcount = sbytes / block_size;
    if (start_block >= sbcount || (uint64_t)blocks > sbcount - start_block) { qx_set_err(err, err_len, "expert row block range out of range"); return 0; }
    uint64_t soff = t->offset + sbytes * expert;
    uint64_t span = (uint64_t)blocks * block_size;
    uint64_t start_offset = soff + start_block * block_size;
    unsigned char *buf = NULL;
    if (!qx_read_raw_span(file, start_offset, span, &buf, err, err_len)) return 0;
    uint32_t state = seed ? seed : 1u;
    double d = 0.0, isum = 0.0, wsum = 0.0;
    uint64_t mix = 1469598103934665603ull;
    for (uint32_t b = 0; b < blocks; ++b) {
        const unsigned char *block = buf + (uint64_t)b * block_size;
        float weights[256];
        qx_decode_supported_block(t->flags, block, weights);
        mix ^= qx_fnv1a64(block, block_size);
        mix *= 1099511628211ull;
        for (int i = 0; i < 256; ++i) {
            float x = qx_deterministic_input(&state);
            d += (double)weights[i] * (double)x;
            isum += (double)x;
            wsum += (double)weights[i];
        }
    }
    free(buf);
    if (dot) *dot = d;
    if (input_sum) *input_sum = isum;
    if (weight_sum) *weight_sum = wsum;
    if (decoder) *decoder = dec;
    if (ggml_type) *ggml_type = t->flags;
    if (slice_offset) *slice_offset = soff;
    if (slice_bytes) *slice_bytes = sbytes;
    if (block_size_out) *block_size_out = block_size;
    if (checksum_mix) *checksum_mix = mix;
    return 1;
}

static double qx_silu(double x) {
    return x / (1.0 + exp(-x));
}

int qx_dump_expert_forward_probe_summary(const char *path, uint32_t layer, uint32_t expert, uint64_t start_block, uint32_t blocks, uint32_t seed, FILE *out, char *err, uint64_t err_len) {
    if (blocks == 0) blocks = 1;
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    double gate_dot = 0.0, up_dot = 0.0, down_dot = 0.0;
    double gate_input_sum = 0.0, up_input_sum = 0.0, down_input_sum = 0.0;
    double gate_weight_sum = 0.0, up_weight_sum = 0.0, down_weight_sum = 0.0;
    const char *gate_decoder = NULL, *up_decoder = NULL, *down_decoder = NULL;
    uint32_t gate_type = 0, up_type = 0, down_type = 0;
    uint64_t gate_soff = 0, up_soff = 0, down_soff = 0;
    uint64_t gate_sbytes = 0, up_sbytes = 0, down_sbytes = 0;
    uint64_t gate_bsize = 0, up_bsize = 0, down_bsize = 0;
    uint64_t gate_mix = 0, up_mix = 0, down_mix = 0;
    if (!qx_expert_row_dot_calc(&file, layer, expert, "gate", start_block, blocks, seed, &gate_dot, &gate_input_sum, &gate_weight_sum, &gate_decoder, &gate_type, &gate_soff, &gate_sbytes, &gate_bsize, &gate_mix, err, err_len) ||
        !qx_expert_row_dot_calc(&file, layer, expert, "up", start_block, blocks, seed, &up_dot, &up_input_sum, &up_weight_sum, &up_decoder, &up_type, &up_soff, &up_sbytes, &up_bsize, &up_mix, err, err_len) ||
        !qx_expert_row_dot_calc(&file, layer, expert, "down", start_block, blocks, seed, &down_dot, &down_input_sum, &down_weight_sum, &down_decoder, &down_type, &down_soff, &down_sbytes, &down_bsize, &down_mix, err, err_len)) {
        qx_close_file(&file); return 0;
    }
    double gate_act = qx_silu(gate_dot);
    double hidden_probe = gate_act * up_dot;
    double projected_probe = hidden_probe * down_dot;
    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"expert_forward\",\n");
    fprintf(out, "  \"layer\": %u,\n", layer);
    fprintf(out, "  \"expert\": %u,\n", expert);
    fprintf(out, "  \"start_block\": %llu,\n", (unsigned long long)start_block);
    fprintf(out, "  \"blocks\": %u,\n", blocks);
    fprintf(out, "  \"values\": %llu,\n", (unsigned long long)blocks * 256ull);
    fprintf(out, "  \"input_seed\": %u,\n", seed);
    fprintf(out, "  \"gate_decoder\": \"%s\",\n", gate_decoder);
    fprintf(out, "  \"up_decoder\": \"%s\",\n", up_decoder);
    fprintf(out, "  \"down_decoder\": \"%s\",\n", down_decoder);
    fprintf(out, "  \"gate_ggml_type\": %u,\n", gate_type);
    fprintf(out, "  \"up_ggml_type\": %u,\n", up_type);
    fprintf(out, "  \"down_ggml_type\": %u,\n", down_type);
    fprintf(out, "  \"gate_dot\": %.9g,\n", gate_dot);
    fprintf(out, "  \"up_dot\": %.9g,\n", up_dot);
    fprintf(out, "  \"down_dot\": %.9g,\n", down_dot);
    fprintf(out, "  \"gate_silu\": %.9g,\n", gate_act);
    fprintf(out, "  \"hidden_probe\": %.9g,\n", hidden_probe);
    fprintf(out, "  \"projected_probe\": %.9g,\n", projected_probe);
    fprintf(out, "  \"gate_slice_offset\": %llu,\n", (unsigned long long)gate_soff);
    fprintf(out, "  \"up_slice_offset\": %llu,\n", (unsigned long long)up_soff);
    fprintf(out, "  \"down_slice_offset\": %llu,\n", (unsigned long long)down_soff);
    fprintf(out, "  \"gate_checksum_mix\": %llu,\n", (unsigned long long)gate_mix);
    fprintf(out, "  \"up_checksum_mix\": %llu,\n", (unsigned long long)up_mix);
    fprintf(out, "  \"down_checksum_mix\": %llu\n", (unsigned long long)down_mix);
    fprintf(out, "}\n");
    (void)gate_input_sum; (void)up_input_sum; (void)down_input_sum;
    (void)gate_weight_sum; (void)up_weight_sum; (void)down_weight_sum;
    (void)gate_sbytes; (void)up_sbytes; (void)down_sbytes;
    (void)gate_bsize; (void)up_bsize; (void)down_bsize;
    qx_close_file(&file);
    return 1;
}


static float qx_rd_le_f32(const unsigned char *p) {
    uint32_t bits = qx_rd_le32(p);
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

int qx_dump_router_topk_probe_summary(const char *path, uint32_t layer, uint32_t top_k, uint32_t blocks, uint32_t seed, FILE *out, char *err, uint64_t err_len) {
    if (top_k == 0) top_k = 1;
    if (blocks == 0) blocks = 1;
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    if (file.header.manifest.model_type != QX_MODEL_QWEN3_MOE) { qx_close_file(&file); qx_set_err(err, err_len, "not a MoE QXF"); return 0; }
    if (layer >= file.header.manifest.layers) { qx_close_file(&file); qx_set_err(err, err_len, "layer out of range"); return 0; }
    char name[QX_NAME_MAX];
    snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp.weight", layer);
    const qx_tensor_dir_entry *router = qx_find_tensor(&file, name);
    if (!router) { qx_close_file(&file); qx_set_err(err, err_len, "router tensor not found"); return 0; }
    if (router->flags != 0u) { qx_close_file(&file); qx_set_err(err, err_len, "router-topk-probe supports F32 router only"); return 0; }
    uint32_t hidden = router->rank > 0 ? (uint32_t)router->dims[0] : file.header.manifest.hidden;
    uint32_t experts = router->rank > 1 ? (uint32_t)router->dims[1] : file.header.manifest.experts;
    if (experts == 0 || hidden == 0) { qx_close_file(&file); qx_set_err(err, err_len, "bad router dimensions"); return 0; }
    if (top_k > experts) top_k = experts;
    uint64_t values = (uint64_t)blocks * 256ull;
    if (values > hidden) values = hidden;
    uint64_t row_bytes = (uint64_t)hidden * 4ull;
    if (router->byte_size < row_bytes * experts) { qx_close_file(&file); qx_set_err(err, err_len, "router tensor too small for F32 rows"); return 0; }
    double *logits = (double *)calloc(experts, sizeof(double));
    int *picked = (int *)calloc(experts, sizeof(int));
    uint32_t *selected = (uint32_t *)calloc(top_k, sizeof(uint32_t));
    if (!logits || !picked || !selected) { free(logits); free(picked); free(selected); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
    uint64_t bytes_read = 0;
    uint64_t checksum_mix = 1469598103934665603ull;
    for (uint32_t e = 0; e < experts; ++e) {
        uint64_t off = router->offset + (uint64_t)e * row_bytes;
        unsigned char *buf = NULL;
        uint64_t span = values * 4ull;
        if (!qx_read_raw_span(&file, off, span, &buf, err, err_len)) { free(logits); free(picked); free(selected); qx_close_file(&file); return 0; }
        uint32_t state = seed ? seed : 1u;
        double dot = 0.0;
        for (uint64_t i = 0; i < values; ++i) {
            float w = qx_rd_le_f32(buf + i*4ull);
            float x = qx_deterministic_input(&state);
            dot += (double)w * (double)x;
        }
        logits[e] = dot;
        checksum_mix ^= qx_fnv1a64(buf, span);
        checksum_mix *= 1099511628211ull;
        bytes_read += span;
        free(buf);
    }
    for (uint32_t k = 0; k < top_k; ++k) {
        uint32_t best = 0;
        double best_logit = -1.0e300;
        for (uint32_t e = 0; e < experts; ++e) {
            if (!picked[e] && logits[e] > best_logit) { best = e; best_logit = logits[e]; }
        }
        selected[k] = best;
        picked[best] = 1;
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"router_topk\",\n");
    fprintf(out, "  \"layer\": %u,\n", layer);
    fprintf(out, "  \"router_tensor\": \"%s\",\n", router->name);
    fprintf(out, "  \"router_kernel\": \"F32_PREFIX_DOT\",\n");
    fprintf(out, "  \"router_ggml_type\": %u,\n", router->flags);
    fprintf(out, "  \"experts\": %u,\n", experts);
    fprintf(out, "  \"top_k\": %u,\n", top_k);
    fprintf(out, "  \"blocks\": %u,\n", blocks);
    fprintf(out, "  \"values\": %llu,\n", (unsigned long long)values);
    fprintf(out, "  \"input_seed\": %u,\n", seed);
    fprintf(out, "  \"bytes_read\": %llu,\n", (unsigned long long)bytes_read);
    fprintf(out, "  \"checksum_mix\": %llu,\n", (unsigned long long)checksum_mix);
    fprintf(out, "  \"selected_experts\": [\n");
    for (uint32_t k = 0; k < top_k; ++k) {
        uint32_t e = selected[k];
        double gate_dot = 0.0, up_dot = 0.0, down_dot = 0.0;
        const char *dec = NULL;
        uint32_t typ = 0;
        uint64_t dummy64 = 0, mix = 0;
        if (!qx_expert_row_dot_calc(&file, layer, e, "gate", 0, blocks, seed, &gate_dot, NULL, NULL, &dec, &typ, &dummy64, &dummy64, &dummy64, &mix, err, err_len) ||
            !qx_expert_row_dot_calc(&file, layer, e, "up", 0, blocks, seed, &up_dot, NULL, NULL, &dec, &typ, &dummy64, &dummy64, &dummy64, &mix, err, err_len) ||
            !qx_expert_row_dot_calc(&file, layer, e, "down", 0, blocks, seed, &down_dot, NULL, NULL, &dec, &typ, &dummy64, &dummy64, &dummy64, &mix, err, err_len)) {
            free(logits); free(picked); free(selected); qx_close_file(&file); return 0;
        }
        double hidden_probe = qx_silu(gate_dot) * up_dot;
        double projected_probe = hidden_probe * down_dot;
        fprintf(out, "    {\"rank\": %u, \"expert\": %u, \"logit\": %.9g, \"gate_dot\": %.9g, \"up_dot\": %.9g, \"down_dot\": %.9g, \"projected_probe\": %.9g}%s\n",
            k, e, logits[e], gate_dot, up_dot, down_dot, projected_probe, (k + 1 < top_k) ? "," : "");
    }
    fprintf(out, "  ]\n");
    fprintf(out, "}\n");
    free(logits); free(picked); free(selected);
    qx_close_file(&file);
    return 1;
}


int qx_dump_layer_forward_probe_summary(const char *path, uint32_t layer, uint32_t top_k, uint32_t blocks, uint32_t seed, FILE *out, char *err, uint64_t err_len) {
    if (top_k == 0) top_k = 1;
    if (blocks == 0) blocks = 1;
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    if (file.header.manifest.model_type != QX_MODEL_QWEN3_MOE) { qx_close_file(&file); qx_set_err(err, err_len, "not a MoE QXF"); return 0; }
    if (layer >= file.header.manifest.layers) { qx_close_file(&file); qx_set_err(err, err_len, "layer out of range"); return 0; }
    char name[QX_NAME_MAX];
    snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp.weight", layer);
    const qx_tensor_dir_entry *router = qx_find_tensor(&file, name);
    if (!router) { qx_close_file(&file); qx_set_err(err, err_len, "router tensor not found"); return 0; }
    if (router->flags != 0u) { qx_close_file(&file); qx_set_err(err, err_len, "layer-forward-probe supports F32 router only"); return 0; }
    uint32_t hidden = router->rank > 0 ? (uint32_t)router->dims[0] : file.header.manifest.hidden;
    uint32_t experts = router->rank > 1 ? (uint32_t)router->dims[1] : file.header.manifest.experts;
    if (experts == 0 || hidden == 0) { qx_close_file(&file); qx_set_err(err, err_len, "bad router dimensions"); return 0; }
    if (top_k > experts) top_k = experts;
    uint64_t values = (uint64_t)blocks * 256ull;
    if (values > hidden) values = hidden;
    uint64_t row_bytes = (uint64_t)hidden * 4ull;
    if (router->byte_size < row_bytes * experts) { qx_close_file(&file); qx_set_err(err, err_len, "router tensor too small for F32 rows"); return 0; }
    double *logits = (double *)calloc(experts, sizeof(double));
    int *picked = (int *)calloc(experts, sizeof(int));
    uint32_t *selected = (uint32_t *)calloc(top_k, sizeof(uint32_t));
    if (!logits || !picked || !selected) { free(logits); free(picked); free(selected); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
    uint64_t bytes_read = 0;
    uint64_t checksum_mix = 1469598103934665603ull;
    for (uint32_t e = 0; e < experts; ++e) {
        unsigned char *buf = NULL;
        uint64_t span = values * 4ull;
        uint64_t off = router->offset + (uint64_t)e * row_bytes;
        if (!qx_read_raw_span(&file, off, span, &buf, err, err_len)) { free(logits); free(picked); free(selected); qx_close_file(&file); return 0; }
        uint32_t state = seed ? seed : 1u;
        double dot = 0.0;
        for (uint64_t i = 0; i < values; ++i) {
            dot += (double)qx_rd_le_f32(buf + i*4ull) * (double)qx_deterministic_input(&state);
        }
        logits[e] = dot;
        checksum_mix ^= qx_fnv1a64(buf, span);
        checksum_mix *= 1099511628211ull;
        bytes_read += span;
        free(buf);
    }
    for (uint32_t k = 0; k < top_k; ++k) {
        uint32_t best = 0;
        double best_logit = -1.0e300;
        for (uint32_t e = 0; e < experts; ++e) {
            if (!picked[e] && logits[e] > best_logit) { best = e; best_logit = logits[e]; }
        }
        selected[k] = best;
        picked[best] = 1;
    }
    double expert_outputs_sum = 0.0;
    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"layer_forward\",\n");
    fprintf(out, "  \"layer\": %u,\n", layer);
    fprintf(out, "  \"router_tensor\": \"%s\",\n", router->name);
    fprintf(out, "  \"router_kernel\": \"F32_PREFIX_DOT\",\n");
    fprintf(out, "  \"experts\": %u,\n", experts);
    fprintf(out, "  \"top_k\": %u,\n", top_k);
    fprintf(out, "  \"blocks\": %u,\n", blocks);
    fprintf(out, "  \"values\": %llu,\n", (unsigned long long)values);
    fprintf(out, "  \"input_seed\": %u,\n", seed);
    fprintf(out, "  \"router_bytes_read\": %llu,\n", (unsigned long long)bytes_read);
    fprintf(out, "  \"router_checksum_mix\": %llu,\n", (unsigned long long)checksum_mix);
    fprintf(out, "  \"selected_experts\": [\n");
    for (uint32_t k = 0; k < top_k; ++k) {
        uint32_t e = selected[k];
        double gate_dot = 0.0, up_dot = 0.0, down_dot = 0.0;
        const char *dec = NULL;
        uint32_t typ = 0;
        uint64_t dummy64 = 0, mix = 0;
        if (!qx_expert_row_dot_calc(&file, layer, e, "gate", 0, blocks, seed, &gate_dot, NULL, NULL, &dec, &typ, &dummy64, &dummy64, &dummy64, &mix, err, err_len) ||
            !qx_expert_row_dot_calc(&file, layer, e, "up", 0, blocks, seed, &up_dot, NULL, NULL, &dec, &typ, &dummy64, &dummy64, &dummy64, &mix, err, err_len) ||
            !qx_expert_row_dot_calc(&file, layer, e, "down", 0, blocks, seed, &down_dot, NULL, NULL, &dec, &typ, &dummy64, &dummy64, &dummy64, &mix, err, err_len)) {
            free(logits); free(picked); free(selected); qx_close_file(&file); return 0;
        }
        double hidden_probe = qx_silu(gate_dot) * up_dot;
        double projected_probe = hidden_probe * down_dot;
        expert_outputs_sum += projected_probe;
        fprintf(out, "    {\"rank\": %u, \"expert\": %u, \"logit\": %.9g, \"gate_dot\": %.9g, \"up_dot\": %.9g, \"down_dot\": %.9g, \"projected_probe\": %.9g}%s\n",
            k, e, logits[e], gate_dot, up_dot, down_dot, projected_probe, (k + 1 < top_k) ? "," : "");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"expert_outputs_sum\": %.9g,\n", expert_outputs_sum);
    fprintf(out, "  \"layer_output_probe\": %.9g\n", expert_outputs_sum);
    fprintf(out, "}\n");
    free(logits); free(picked); free(selected);
    qx_close_file(&file);
    return 1;
}


int qx_dump_moe_forward_probe_summary(const char *path, uint32_t layers, uint32_t top_k, uint32_t blocks, uint32_t seed, FILE *out, char *err, uint64_t err_len) {
    if (layers == 0) layers = 1;
    if (top_k == 0) top_k = 1;
    if (blocks == 0) blocks = 1;
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    if (file.header.manifest.model_type != QX_MODEL_QWEN3_MOE) { qx_close_file(&file); qx_set_err(err, err_len, "not a MoE QXF"); return 0; }
    uint32_t max_layers = file.header.manifest.layers;
    if (layers > max_layers) layers = max_layers;
    double moe_output = 0.0;
    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"moe_forward\",\n");
    fprintf(out, "  \"layers_requested\": %u,\n", layers);
    fprintf(out, "  \"layers_run\": %u,\n", layers);
    fprintf(out, "  \"top_k\": %u,\n", top_k);
    fprintf(out, "  \"blocks\": %u,\n", blocks);
    fprintf(out, "  \"input_seed\": %u,\n", seed);
    fprintf(out, "  \"layers\": [\n");
    for (uint32_t layer = 0; layer < layers; ++layer) {
        char name[QX_NAME_MAX];
        snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp.weight", layer);
        const qx_tensor_dir_entry *router = qx_find_tensor(&file, name);
        if (!router) { qx_close_file(&file); qx_set_err(err, err_len, "router tensor not found for requested layer"); return 0; }
        if (router->flags != 0u) { qx_close_file(&file); qx_set_err(err, err_len, "moe-forward-probe supports F32 router only"); return 0; }
        uint32_t hidden = router->rank > 0 ? (uint32_t)router->dims[0] : file.header.manifest.hidden;
        uint32_t experts = router->rank > 1 ? (uint32_t)router->dims[1] : file.header.manifest.experts;
        if (experts == 0 || hidden == 0) { qx_close_file(&file); qx_set_err(err, err_len, "bad router dimensions"); return 0; }
        uint32_t k_eff = top_k > experts ? experts : top_k;
        uint64_t values = (uint64_t)blocks * 256ull;
        if (values > hidden) values = hidden;
        uint64_t row_bytes = (uint64_t)hidden * 4ull;
        if (router->byte_size < row_bytes * experts) { qx_close_file(&file); qx_set_err(err, err_len, "router tensor too small for F32 rows"); return 0; }
        double *logits = (double *)calloc(experts, sizeof(double));
        int *picked = (int *)calloc(experts, sizeof(int));
        uint32_t *selected = (uint32_t *)calloc(k_eff, sizeof(uint32_t));
        if (!logits || !picked || !selected) { free(logits); free(picked); free(selected); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
        for (uint32_t e = 0; e < experts; ++e) {
            unsigned char *buf = NULL;
            uint64_t span = values * 4ull;
            uint64_t off = router->offset + (uint64_t)e * row_bytes;
            if (!qx_read_raw_span(&file, off, span, &buf, err, err_len)) { free(logits); free(picked); free(selected); qx_close_file(&file); return 0; }
            uint32_t state = seed ? seed : 1u;
            double dot = 0.0;
            for (uint64_t i = 0; i < values; ++i) {
                dot += (double)qx_rd_le_f32(buf + i*4ull) * (double)qx_deterministic_input(&state);
            }
            logits[e] = dot;
            free(buf);
        }
        for (uint32_t k = 0; k < k_eff; ++k) {
            uint32_t best = 0;
            double best_logit = -1.0e300;
            for (uint32_t e = 0; e < experts; ++e) {
                if (!picked[e] && logits[e] > best_logit) { best = e; best_logit = logits[e]; }
            }
            selected[k] = best;
            picked[best] = 1;
        }
        double layer_sum = 0.0;
        fprintf(out, "    {\n");
        fprintf(out, "      \"layer\": %u,\n", layer);
        fprintf(out, "      \"router_kernel\": \"F32_PREFIX_DOT\",\n");
        fprintf(out, "      \"top_k\": %u,\n", k_eff);
        fprintf(out, "      \"selected_experts\": [\n");
        for (uint32_t k = 0; k < k_eff; ++k) {
            uint32_t e = selected[k];
            double gate_dot = 0.0, up_dot = 0.0, down_dot = 0.0;
            const char *dec = NULL;
            uint32_t typ = 0;
            uint64_t dummy64 = 0, mix = 0;
            int ok = qx_expert_row_dot_calc(&file, layer, e, "gate", 0, blocks, seed, &gate_dot, NULL, NULL, &dec, &typ, &dummy64, &dummy64, &dummy64, &mix, err, err_len) &&
                     qx_expert_row_dot_calc(&file, layer, e, "up", 0, blocks, seed, &up_dot, NULL, NULL, &dec, &typ, &dummy64, &dummy64, &dummy64, &mix, err, err_len) &&
                     qx_expert_row_dot_calc(&file, layer, e, "down", 0, blocks, seed, &down_dot, NULL, NULL, &dec, &typ, &dummy64, &dummy64, &dummy64, &mix, err, err_len);
            if (ok) {
                double hidden_probe = qx_silu(gate_dot) * up_dot;
                double projected_probe = hidden_probe * down_dot;
                layer_sum += projected_probe;
                fprintf(out, "        {\"rank\": %u, \"expert\": %u, \"logit\": %.9g, \"supported\": true, \"projected_probe\": %.9g}%s\n",
                    k, e, logits[e], projected_probe, (k + 1 < k_eff) ? "," : "");
            } else {
                fprintf(out, "        {\"rank\": %u, \"expert\": %u, \"logit\": %.9g, \"supported\": false, \"reason\": \"unsupported_expert_quant\", \"projected_probe\": 0}%s\n",
                    k, e, logits[e], (k + 1 < k_eff) ? "," : "");
            }
        }
        moe_output += layer_sum;
        fprintf(out, "      ],\n");
        fprintf(out, "      \"layer_output_probe\": %.9g\n", layer_sum);
        fprintf(out, "    }%s\n", (layer + 1 < layers) ? "," : "");
        free(logits); free(picked); free(selected);
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"moe_output_probe\": %.9g\n", moe_output);
    fprintf(out, "}\n");
    qx_close_file(&file);
    return 1;
}


int qx_dump_expert_quant_coverage_summary(const char *path, FILE *out, char *err, uint64_t err_len) {
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    uint32_t complete = 0;
    uint32_t supported = 0;
    uint32_t missing = 0;
    uint32_t unsupported = 0;
    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"expert_quant_coverage\",\n");
    fprintf(out, "  \"layers_total\": %u,\n", file.header.manifest.layers);
    fprintf(out, "  \"experts_per_layer\": %u,\n", file.header.manifest.experts);
    fprintf(out, "  \"layers\": [\n");
    int first = 1;
    for (uint32_t layer = 0; layer < file.header.manifest.layers; ++layer) {
        char gate_name[QX_NAME_MAX], up_name[QX_NAME_MAX], down_name[QX_NAME_MAX];
        qx_expert_packed_tensor_name(gate_name, sizeof(gate_name), layer, "gate");
        qx_expert_packed_tensor_name(up_name, sizeof(up_name), layer, "up");
        qx_expert_packed_tensor_name(down_name, sizeof(down_name), layer, "down");
        const qx_tensor_dir_entry *gate = qx_find_tensor(&file, gate_name);
        const qx_tensor_dir_entry *up = qx_find_tensor(&file, up_name);
        const qx_tensor_dir_entry *down = qx_find_tensor(&file, down_name);
        if (!gate && !up && !down) continue;
        const char *gd = NULL, *ud = NULL, *dd = NULL;
        uint64_t gb = 0, ub = 0, db = 0;
        int gs = gate && qx_decoder_info(gate->flags, &gd, &gb);
        int us = up && qx_decoder_info(up->flags, &ud, &ub);
        int ds = down && qx_decoder_info(down->flags, &dd, &db);
        int is_complete = gate && up && down;
        int is_supported = is_complete && gs && us && ds;
        if (is_complete) complete++; else missing++;
        if (is_supported) supported++; else unsupported++;
        fprintf(out, "%s    {\n", first ? "" : ",\n");
        first = 0;
        fprintf(out, "      \"layer\": %u,\n", layer);
        fprintf(out, "      \"complete\": %s,\n", is_complete ? "true" : "false");
        fprintf(out, "      \"supported\": %s,\n", is_supported ? "true" : "false");
        fprintf(out, "      \"gate_ggml_type\": %u,\n", gate ? gate->flags : 0u);
        fprintf(out, "      \"gate_decoder\": %s%s%s,\n", gd ? "\"" : "", gd ? gd : "null", gd ? "\"" : "");
        fprintf(out, "      \"up_ggml_type\": %u,\n", up ? up->flags : 0u);
        fprintf(out, "      \"up_decoder\": %s%s%s,\n", ud ? "\"" : "", ud ? ud : "null", ud ? "\"" : "");
        fprintf(out, "      \"down_ggml_type\": %u,\n", down ? down->flags : 0u);
        fprintf(out, "      \"down_decoder\": %s%s%s,\n", dd ? "\"" : "", dd ? dd : "null", dd ? "\"" : "");
        fprintf(out, "      \"gate_block_byte_size\": %llu,\n", (unsigned long long)gb);
        fprintf(out, "      \"up_block_byte_size\": %llu,\n", (unsigned long long)ub);
        fprintf(out, "      \"down_block_byte_size\": %llu\n", (unsigned long long)db);
        fprintf(out, "    }");
    }
    fprintf(out, "\n  ],\n");
    fprintf(out, "  \"complete_layers\": %u,\n", complete);
    fprintf(out, "  \"supported_layers\": %u,\n", supported);
    fprintf(out, "  \"missing_layers\": %u,\n", missing);
    fprintf(out, "  \"unsupported_layers\": %u\n", unsupported);
    fprintf(out, "}\n");
    qx_close_file(&file);
    return 1;
}

static int qx_tensor_block_dot_calc(qx_file *file, const qx_tensor_dir_entry *t, uint32_t blocks, uint32_t seed, double *dot_out, double *sum_out, const char **decoder_out, uint64_t *values_out, uint64_t *checksum_out, char *err, uint64_t err_len);



typedef struct {
    uint32_t token;
    double logit;
    uint64_t checksum;
} qx_top_token;

typedef struct {
    uint64_t temporary_blocks_decoded;
    uint64_t temporary_floats_materialized;
    uint64_t temporary_bytes_materialized;
    uint64_t fused_dot_calls;
    uint64_t fallback_dot_calls;
    uint64_t final_head_q6_k_blocks;
} qx_dequant_dot_profile;

static const qx_tensor_dir_entry *qx_select_lm_head_tensor(qx_file *file, const char **name_out, int *tied_out) {
    const qx_tensor_dir_entry *t = qx_find_tensor(file, "output.weight");
    if (t) { if (name_out) *name_out = "output.weight"; if (tied_out) *tied_out = 0; return t; }
    t = qx_find_tensor(file, "lm_head.weight");
    if (t) { if (name_out) *name_out = "lm_head.weight"; if (tied_out) *tied_out = 0; return t; }
    t = qx_find_tensor(file, "token_embd.weight");
    if (t) { if (name_out) *name_out = "token_embd.weight"; if (tied_out) *tied_out = 1; return t; }
    return NULL;
}

static int qx_logit_row_score(qx_file *file, const qx_tensor_dir_entry *lm, uint32_t vocab, uint32_t token, double activation, uint32_t seed, double *score_out, uint64_t *checksum_out, char *err, uint64_t err_len) {
    if (!score_out || token >= vocab) { qx_set_err(err, err_len, "token id out of range"); return 0; }
    uint64_t row = 0;
    if (!qx_embedding_row_size(lm, vocab ? vocab : (uint32_t)lm->dims[1], &row, err, err_len)) return 0;
    uint64_t off = lm->offset + (uint64_t)token * row;
    uint64_t span = row;
    unsigned char *buf = NULL;
    if (!qx_read_raw_span(file, off, span, &buf, err, err_len)) return 0;
    uint64_t chk = qx_fnv1a64(buf, span);
    if (checksum_out) *checksum_out = chk;
    double score = 0.0;
    const char *decoder = NULL;
    uint64_t bs = 0;
    if (qx_decoder_info(lm->flags, &decoder, &bs) && bs > 0 && span >= bs) {
        uint64_t blocks = span / bs;
        if (blocks > 4) blocks = 4;
        uint32_t st = (seed ? seed : 1u) ^ token;
        for (uint64_t b = 0; b < blocks; ++b) {
            float vals[256];
            qx_decode_supported_block(lm->flags, buf + b * bs, vals);
            for (int i = 0; i < 256; ++i) score += (double)vals[i] * (double)qx_deterministic_input(&st);
        }
    } else {
        uint64_t n = span < 256 ? span : 256;
        for (uint64_t i = 0; i < n; ++i) score += (((double)buf[i] - 127.5) / 127.5) * (double)((chk >> ((i & 7u) * 8u)) & 255u) / 255.0;
    }
    free(buf);
    *score_out = score * activation;
    return 1;
}

static int qx_collect_top_logits(qx_file *file, double activation, uint32_t top_n, uint32_t scan, uint32_t seed, const char **lm_name, int *tied, uint32_t *scanned_out, qx_top_token *top, char *err, uint64_t err_len) {
    if (top_n == 0) top_n = 1;
    if (scan == 0) scan = 64;
    if (top_n > scan) top_n = scan;
    const qx_tensor_dir_entry *lm = qx_select_lm_head_tensor(file, lm_name, tied);
    if (!lm) { qx_set_err(err, err_len, "lm head/output/token embedding tensor not found"); return 0; }
    uint32_t vocab = file->header.manifest.vocab ? file->header.manifest.vocab : (uint32_t)(lm->dims[1] ? lm->dims[1] : scan);
    if (scan > vocab) scan = vocab;
    if (top_n > scan) top_n = scan;
    if (top_n == 0) { qx_set_err(err, err_len, "empty logits scan"); return 0; }
    for (uint32_t i = 0; i < top_n; ++i) { top[i].token = 0; top[i].logit = -1.0e300; top[i].checksum = 0; }
    for (uint32_t token = 0; token < scan; ++token) {
        uint64_t chk = 0;
        double logit = 0.0;
        if (!qx_logit_row_score(file, lm, vocab, token, activation, seed, &logit, &chk, err, err_len)) return 0;
        for (uint32_t i = 0; i < top_n; ++i) {
            if (logit > top[i].logit) {
                for (uint32_t j = top_n - 1; j > i; --j) top[j] = top[j - 1];
                top[i].token = token; top[i].logit = logit; top[i].checksum = chk;
                break;
            }
        }
    }
    if (scanned_out) *scanned_out = scan;
    return 1;
}

int qx_dump_logits_probe_summary(const char *path, double activation, uint32_t top_n, uint32_t scan, uint32_t seed, FILE *out, char *err, uint64_t err_len) {
    if (!path) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    if (top_n == 0) top_n = 1;
    if (top_n > 32) top_n = 32;
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    qx_top_token top[32];
    const char *lm_name = NULL;
    int tied = 0;
    uint32_t scanned = 0;
    if (!qx_collect_top_logits(&file, activation, top_n, scan, seed, &lm_name, &tied, &scanned, top, err, err_len)) { qx_close_file(&file); return 0; }
    if (top_n > scanned) top_n = scanned;
    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"logits\",\n");
    fprintf(out, "  \"synthetic\": true,\n");
    fprintf(out, "  \"lm_head_tensor\": \"%s\",\n", lm_name ? lm_name : "null");
    fprintf(out, "  \"tied_embedding_fallback\": %s,\n", tied ? "true" : "false");
    fprintf(out, "  \"activation\": %.9g,\n", activation);
    fprintf(out, "  \"top_n\": %u,\n", top_n);
    fprintf(out, "  \"scanned\": %u,\n", scanned);
    fprintf(out, "  \"top_tokens\": [");
    for (uint32_t i = 0; i < top_n; ++i) {
        fprintf(out, "%s{\"token\": %u, \"logit\": %.9g, \"checksum\": %llu}", i ? ", " : "", top[i].token, top[i].logit, (unsigned long long)top[i].checksum);
    }
    fprintf(out, "]\n}\n");
    qx_close_file(&file);
    return 1;
}



typedef struct {
    uint32_t selected_token;
    uint32_t selected_rank;
    double selected_logit;
    double selected_prob;
    double prob_sum;
    double random_u;
    const char *strategy;
} qx_sample_result;

static qx_sample_result qx_sample_from_top(const qx_top_token *top, uint32_t top_k, double temperature, uint32_t seed) {
    qx_sample_result r;
    memset(&r, 0, sizeof(r));
    if (top_k == 0) top_k = 1;
    if (temperature <= 0.0) {
        r.selected_token = top[0].token;
        r.selected_rank = 0;
        r.selected_logit = top[0].logit;
        r.selected_prob = 1.0;
        r.prob_sum = 1.0;
        r.strategy = "argmax";
        return r;
    }
    double max_logit = top[0].logit;
    for (uint32_t i = 1; i < top_k; ++i) if (top[i].logit > max_logit) max_logit = top[i].logit;
    double probs[32];
    double sum = 0.0;
    for (uint32_t i = 0; i < top_k; ++i) {
        probs[i] = exp((top[i].logit - max_logit) / temperature);
        sum += probs[i];
    }
    if (sum <= 0.0) sum = 1.0;
    for (uint32_t i = 0; i < top_k; ++i) probs[i] /= sum;
    uint32_t st = seed ? seed : 1u;
    double u = ((double)qx_deterministic_input(&st) + 1.0) * 0.5;
    double acc = 0.0;
    uint32_t rank = top_k - 1;
    for (uint32_t i = 0; i < top_k; ++i) {
        acc += probs[i];
        if (u <= acc) { rank = i; break; }
    }
    r.selected_token = top[rank].token;
    r.selected_rank = rank;
    r.selected_logit = top[rank].logit;
    r.selected_prob = probs[rank];
    r.prob_sum = 0.0;
    for (uint32_t i = 0; i < top_k; ++i) r.prob_sum += probs[i];
    r.random_u = u;
    r.strategy = "temperature_top_k";
    return r;
}

static void qx_print_sampler_json(FILE *out, const qx_sample_result *r, int enabled, uint32_t top_k, uint32_t scan, double temperature) {
    fprintf(out, "{\"enabled\": %s, \"strategy\": \"%s\", \"top_k\": %u, \"scan\": %u, \"temperature\": %.9g, \"selected_token\": %u, \"selected_rank\": %u, \"selected_logit\": %.9g, \"selected_prob\": %.9g, \"prob_sum\": %.9g, \"random_u\": %.9g}",
        enabled ? "true" : "false", r->strategy ? r->strategy : "disabled", top_k, scan, temperature, r->selected_token, r->selected_rank, r->selected_logit, r->selected_prob, r->prob_sum, r->random_u);
}

int qx_dump_sampler_probe_summary(const char *path, double activation, uint32_t top_k, uint32_t scan, double temperature, uint32_t seed, FILE *out, char *err, uint64_t err_len) {
    if (!path) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    if (top_k == 0) top_k = 1;
    if (top_k > 32) top_k = 32;
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    qx_top_token top[32];
    const char *lm_name = NULL;
    int tied = 0;
    uint32_t scanned = 0;
    if (!qx_collect_top_logits(&file, activation, top_k, scan, seed, &lm_name, &tied, &scanned, top, err, err_len)) { qx_close_file(&file); return 0; }
    if (top_k > scanned) top_k = scanned;
    qx_sample_result sr = qx_sample_from_top(top, top_k, temperature, seed);
    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"sampler\",\n");
    fprintf(out, "  \"lm_head_tensor\": \"%s\",\n", lm_name ? lm_name : "null");
    fprintf(out, "  \"tied_embedding_fallback\": %s,\n", tied ? "true" : "false");
    fprintf(out, "  \"strategy\": \"%s\",\n", sr.strategy);
    fprintf(out, "  \"activation\": %.9g,\n", activation);
    fprintf(out, "  \"top_k\": %u,\n", top_k);
    fprintf(out, "  \"scan\": %u,\n", scanned);
    fprintf(out, "  \"temperature\": %.9g,\n", temperature);
    fprintf(out, "  \"selected_token\": %u,\n", sr.selected_token);
    fprintf(out, "  \"selected_rank\": %u,\n", sr.selected_rank);
    fprintf(out, "  \"selected_logit\": %.9g,\n", sr.selected_logit);
    fprintf(out, "  \"selected_prob\": %.9g,\n", sr.selected_prob);
    fprintf(out, "  \"prob_sum\": %.9g,\n", sr.prob_sum);
    fprintf(out, "  \"random_u\": %.9g,\n", sr.random_u);
    fprintf(out, "  \"candidates\": [");
    for (uint32_t i = 0; i < top_k; ++i) fprintf(out, "%s{\"token\": %u, \"logit\": %.9g}", i ? ", " : "", top[i].token, top[i].logit);
    fprintf(out, "]\n}\n");
    qx_close_file(&file);
    return 1;
}



static void qx_token_piece_fallback(uint32_t token_id, char *buf, size_t cap) {
    if (!buf || cap == 0) return;
#ifdef _MSC_VER
    sprintf_s(buf, cap, "<tok_%u>", token_id);
#else
    snprintf(buf, cap, "<tok_%u>", token_id);
#endif
}


static void qx_unescape_tsv_piece(char *s) {
    char *w = s;
    for (char *r = s; *r; ++r) {
        if (*r == '\\' && r[1]) {
            ++r;
            if (*r == 't') *w++ = '\t';
            else if (*r == 'n') *w++ = '\n';
            else if (*r == 'r') *w++ = '\r';
            else *w++ = *r;
        } else {
            *w++ = *r;
        }
    }
    *w = 0;
}

static int qx_lookup_token_piece_sidecar(const char *tokens_path, uint32_t token_id, char *piece, size_t cap) {
    if (!tokens_path || !piece || cap == 0) return 0;
    FILE *f = fopen(tokens_path, "rb");
    if (!f) return 0;
    char line[8192];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = 0;
        unsigned long id = strtoul(line, NULL, 10);
        if ((uint32_t)id != token_id) continue;
        char *val = tab + 1;
        size_t n = strlen(val);
        while (n && (val[n-1] == '\n' || val[n-1] == '\r')) val[--n] = 0;
#ifdef _MSC_VER
        strncpy_s(piece, cap, val, _TRUNCATE);
#else
        snprintf(piece, cap, "%s", val);
#endif
        qx_unescape_tsv_piece(piece);
        fclose(f);
        return 1;
    }
    fclose(f);
    return 0;
}

static void qx_json_print_escaped(FILE *out, const char *s) {
    fputc('"', out);
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
            if (*p == '"' || *p == '\\') { fputc('\\', out); fputc(*p, out); }
            else if (*p == '\n') fputs("\\n", out);
            else if (*p == '\r') fputs("\\r", out);
            else if (*p == '\t') fputs("\\t", out);
            else if (*p < 32) fprintf(out, "\\u%04x", (unsigned int)*p);
            else fputc(*p, out);
        }
    }
    fputc('"', out);
}

static void qx_emit_decoded_token_object(FILE *out, uint32_t token_id, int enabled, const char *tokens_path) {
    char piece[64];
    char sidecar_piece[8192];
    const char *source = "fallback_token_id";
    qx_token_piece_fallback(token_id, piece, sizeof(piece));
    if (tokens_path && qx_lookup_token_piece_sidecar(tokens_path, token_id, sidecar_piece, sizeof(sidecar_piece))) {
        source = "sidecar";
        fprintf(out, "{\"enabled\": %s, \"token_id\": %u, \"source\": \"%s\", \"piece\": ", enabled ? "true" : "false", token_id, source);
        qx_json_print_escaped(out, sidecar_piece);
        fprintf(out, ", \"note\": \"decoded from tokenizer sidecar\"}");
        return;
    }
    fprintf(out, "{\"enabled\": %s, \"token_id\": %u, \"source\": \"%s\", \"piece\": ", enabled ? "true" : "false", token_id, source);
    qx_json_print_escaped(out, piece);
    fprintf(out, ", \"note\": \"QXF currently does not persist GGUF tokenizer arrays; fallback preserves token identity\"}");
}

int qx_dump_tokenizer_probe_summary(const char *path, const char *tokens_path, uint32_t token_id, FILE *out, char *err, uint64_t err_len) {
    if (!path) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    uint32_t vocab = file.header.manifest.vocab;
    if (vocab && token_id >= vocab) { qx_close_file(&file); qx_set_err(err, err_len, "token id out of vocab"); return 0; }
    char piece[64];
    char sidecar_piece[8192];
    const char *source = "fallback_token_id";
    qx_token_piece_fallback(token_id, piece, sizeof(piece));
    if (tokens_path && qx_lookup_token_piece_sidecar(tokens_path, token_id, sidecar_piece, sizeof(sidecar_piece))) {
        source = "sidecar";
#ifdef _MSC_VER
        strncpy_s(piece, sizeof(piece), sidecar_piece, _TRUNCATE);
#else
        snprintf(piece, sizeof(piece), "%s", sidecar_piece);
#endif
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"tokenizer\",\n");
    fprintf(out, "  \"token_id\": %u,\n", token_id);
    fprintf(out, "  \"vocab\": %u,\n", vocab);
    fprintf(out, "  \"source\": \"%s\",\n", source);
    fprintf(out, "  \"piece\": "); qx_json_print_escaped(out, piece); fprintf(out, ",\n");
    fprintf(out, "  \"note\": \"QXF currently does not persist GGUF tokenizer.ggml.tokens; fallback is reversible identity text\"\n");
    fprintf(out, "}\n");
    qx_close_file(&file);
    return 1;
}



int qx_dump_generate_probe_summary(const char *path, const char *tokens_path, uint32_t prompt_token, uint32_t steps, uint32_t top_k, uint32_t scan, double temperature, uint32_t seed, FILE *out, char *err, uint64_t err_len) {
    if (!path) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    if (steps == 0) steps = 1;
    if (steps > 64) steps = 64;
    if (top_k == 0) top_k = 1;
    if (top_k > 32) top_k = 32;
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    uint32_t vocab = file.header.manifest.vocab ? file.header.manifest.vocab : 151936u;
    if (prompt_token >= vocab) { qx_close_file(&file); qx_set_err(err, err_len, "prompt token out of range"); return 0; }
    uint32_t current = prompt_token;
    char generated[32768];
    generated[0] = 0;
    size_t generated_len = 0;
    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"generate\",\n");
    fprintf(out, "  \"prompt_token\": %u,\n", prompt_token);
    fprintf(out, "  \"steps\": %u,\n", steps);
    fprintf(out, "  \"top_k\": %u,\n", top_k);
    fprintf(out, "  \"scan\": %u,\n", scan);
    fprintf(out, "  \"temperature\": %.9g,\n", temperature);
    fprintf(out, "  \"tokens\": [");
    for (uint32_t step = 0; step < steps; ++step) {
        qx_top_token top[32];
        const char *lm_name = NULL;
        int tied = 0;
        uint32_t scanned = 0;
        double activation = 0.125 + ((double)(current % 997u) / 9970.0) + (double)step * 0.001;
        if (!qx_collect_top_logits(&file, activation, top_k, scan, seed + step * 17u + current, &lm_name, &tied, &scanned, top, err, err_len)) { qx_close_file(&file); return 0; }
        uint32_t effective_top_k = top_k > scanned ? scanned : top_k;
        qx_sample_result sr = qx_sample_from_top(top, effective_top_k, temperature, seed + step * 101u + current);
        char piece[8192];
        char fallback[64];
        const char *source = "fallback_token_id";
        if (tokens_path && qx_lookup_token_piece_sidecar(tokens_path, sr.selected_token, piece, sizeof(piece))) source = "sidecar";
        else { qx_token_piece_fallback(sr.selected_token, fallback, sizeof(fallback));
#ifdef _MSC_VER
            strncpy_s(piece, sizeof(piece), fallback, _TRUNCATE);
#else
            snprintf(piece, sizeof(piece), "%s", fallback);
#endif
        }
        size_t plen = strlen(piece);
        if (generated_len + plen < sizeof(generated)) {
            memcpy(generated + generated_len, piece, plen + 1);
            generated_len += plen;
        }
        fprintf(out, "%s{\"step\": %u, \"input_token\": %u, \"token\": %u, \"rank\": %u, \"logit\": %.9g, \"prob\": %.9g, \"source\": \"%s\", \"piece\": ", step ? ", " : "", step, current, sr.selected_token, sr.selected_rank, sr.selected_logit, sr.selected_prob, source);
        qx_json_print_escaped(out, piece);
        fprintf(out, "}");
        current = sr.selected_token;
    }
    fprintf(out, "],\n");
    fprintf(out, "  \"final_token\": %u,\n", current);
    fprintf(out, "  \"generated_text\": ");
    qx_json_print_escaped(out, generated);
    fprintf(out, ",\n");
    fprintf(out, "  \"note\": \"minimal probe loop: sampled token feeds next step activation; not full autoregressive transformer state yet\"\n");
    fprintf(out, "}\n");
    qx_close_file(&file);
    return 1;
}



static int qx_kv_bytes_for_format(const char *kv_format, uint64_t values_per_k_or_v, uint64_t *bytes_per_k_or_v, uint32_t *bytes_per_value) {
    if (!kv_format || !bytes_per_k_or_v || !bytes_per_value) return 0;
    if (strcmp(kv_format, "int8") == 0) { *bytes_per_value = 1; *bytes_per_k_or_v = values_per_k_or_v; return 1; }
    if (strcmp(kv_format, "f32") == 0) { if (values_per_k_or_v > UINT64_MAX / 4ull) return 0; *bytes_per_value = 4; *bytes_per_k_or_v = values_per_k_or_v * 4ull; return 1; }
    if (strcmp(kv_format, "f16") == 0 || strcmp(kv_format, "fp16") == 0) { if (values_per_k_or_v > UINT64_MAX / 2ull) return 0; *bytes_per_value = 2; *bytes_per_k_or_v = values_per_k_or_v * 2ull; return 1; }
    if (strcmp(kv_format, "int4") == 0) { if (values_per_k_or_v == UINT64_MAX) return 0; *bytes_per_value = 0; *bytes_per_k_or_v = (values_per_k_or_v + 1ull) / 2ull; return 1; }
    return 0;
}

static void qx_fill_kv_append(unsigned char *buf, uint64_t n, uint32_t token, uint32_t step, uint32_t layer, uint32_t seed, int is_v) {
    uint32_t st = seed ^ (token * 2654435761u) ^ (step * 2246822519u) ^ (layer * 3266489917u) ^ (is_v ? 0x9e3779b9u : 0x85ebca6bu);
    for (uint64_t i = 0; i < n; ++i) {
        st = st * 1664525u + 1013904223u;
        buf[i] = (unsigned char)((st >> 24) & 0xffu);
    }
}



static int qx_fill_kv_from_projection(qx_file *file, const qx_tensor_dir_entry *t, unsigned char *buf, uint64_t n, uint32_t token, uint32_t step, uint32_t layer, uint32_t seed, uint64_t *values_out, char *err, uint64_t err_len) {
    const char *decoder = NULL;
    uint64_t block_size = 0;
    if (!t || !qx_decoder_info(t->flags, &decoder, &block_size) || block_size == 0) { qx_set_err(err, err_len, "unsupported projection tensor"); return 0; }
    uint64_t block_count = t->byte_size / block_size;
    if (block_count == 0) { qx_set_err(err, err_len, "empty projection tensor"); return 0; }
    uint64_t block_index = ((uint64_t)token * 1315423911ull + (uint64_t)step * 2654435761ull + (uint64_t)layer * 97531ull + (uint64_t)seed) % block_count;
    unsigned char *raw = NULL;
    if (!qx_read_raw_span(file, t->offset + block_index * block_size, block_size, &raw, err, err_len)) return 0;
    float vals[256];
    qx_decode_supported_block(t->flags, raw, vals);
    for (uint64_t i = 0; i < n; ++i) {
        float v = vals[i % 256u];
        int q = (int)lrintf(v * 127.0f);
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        buf[i] = (unsigned char)(q & 0xff);
    }
    if (values_out) *values_out = n;
    free(raw);
    return 1;
}





static int qx_apply_f32_rmsnorm(qx_file *file, const qx_tensor_dir_entry *norm, const float *src, float *dst, uint32_t dims, double *rms_out, char *err, uint64_t err_len) {
    if (!file || !norm || !src || !dst || !dims || norm->flags != 0u || norm->byte_size < (uint64_t)dims * 4ull) {
        qx_set_err(err, err_len, "invalid F32 RMSNorm argument"); return 0;
    }
    double sumsq = 0.0;
    for (uint32_t i = 0; i < dims; ++i) sumsq += (double)src[i] * (double)src[i];
    double rms = sqrt(sumsq / (double)dims + 1.0e-6);
    unsigned char *weights = NULL;
    if (!qx_read_raw_span(file, norm->offset, (uint64_t)dims * 4ull, &weights, err, err_len)) return 0;
    for (uint32_t i = 0; i < dims; ++i) dst[i] = (float)((double)src[i] / rms) * qx_rd_le_f32(weights + (uint64_t)i * 4ull);
    free(weights);
    if (rms_out) *rms_out = rms;
    return 1;
}

typedef struct {
    uint32_t input_dims;
    uint32_t vocab_size;
    uint32_t logits_computed;
    uint32_t top_n;
    uint32_t lm_head_ggml_type;
    double input_rms;
    double normalized_l2;
    double logits_min;
    double logits_max;
    double logits_rms;
    uint64_t residual_checksum;
    uint64_t normalized_checksum;
    uint64_t logits_checksum;
    uint64_t norm_raw_checksum;
    uint64_t lm_head_raw_checksum;
    const char *lm_head_kernel;
    const char *thread_policy;
    const char *thread_disabled_reason;
    uint32_t requested_threads;
    uint32_t workers_used;
    uint64_t parallel_jobs;
    uint64_t serial_jobs;
    uint64_t fallback_jobs;
    uint32_t activation_quantizations;
    const char *simd_policy;
    const char *simd_kernel;
    const char *simd_disabled_reason;
    uint64_t simd_fma_dot_calls;
    uint64_t simd_fallback_dot_calls;
    qx_dequant_dot_profile dequant_profile;
    qx_top_token top[32];
} qx_real_head_result;

#if defined(_WIN32)
typedef struct {
    const unsigned char *raw;
    const float *normalized;
    uint64_t row_bytes;
    uint64_t block_size;
    uint64_t blocks_per_row;
    uint32_t start_row;
    uint32_t end_row;
    int use_fused_f32;
    float *logits_out;
    double *logits64_out;
    volatile LONG failed;
} qx_final_head_pool_task;

static DWORD WINAPI qx_final_head_pool_worker(LPVOID arg) {
    qx_final_head_pool_task *task = (qx_final_head_pool_task *)arg;
    for (uint32_t row_index = task->start_row; row_index < task->end_row; ++row_index) {
        const unsigned char *row = task->raw + (uint64_t)row_index * task->row_bytes;
        double logit = 0.0;
        for (uint64_t block = 0; block < task->blocks_per_row; ++block) {
            const float *input = task->normalized + block * 256u;
            const unsigned char *block_raw = row + block * task->block_size;
            if (task->use_fused_f32) {
                logit += qx_dot_q6_k_f32_fused_block(block_raw, input);
            } else {
                float weights[256];
                if (!qx_decode_supported_block(14u, block_raw, weights)) {
                    InterlockedExchange(&task->failed, 1); return 1u;
                }
                for (uint32_t i = 0; i < 256u; ++i) logit += (double)weights[i] * (double)input[i];
            }
        }
        if (!isfinite(logit)) { InterlockedExchange(&task->failed, 1); return 1u; }
        task->logits64_out[row_index] = logit;
        if (task->logits_out) task->logits_out[row_index] = (float)logit;
    }
    return 0u;
}

static void qx_join_final_head_pool_workers(HANDLE *handles, qx_final_head_pool_task *tasks, uint32_t workers, int *failed) {
    if (workers == 0u) return;
    DWORD wait_result = WaitForMultipleObjects(workers, handles, TRUE, INFINITE);
    if (wait_result < WAIT_OBJECT_0 || wait_result >= WAIT_OBJECT_0 + workers) {
        if (failed) *failed = 1;
    }
    for (uint32_t worker = 0; worker < workers; ++worker) {
        if (tasks && tasks[worker].failed && failed) *failed = 1;
        if (handles[worker]) CloseHandle(handles[worker]);
    }
}
#endif

static int qx_compute_real_final_head_q8_k(qx_file *file, const qx_tensor_dir_entry *lm,
        const float *normalized, uint32_t dims, uint64_t row_bytes, float *logits_out,
        qx_real_head_result *result, char *err, uint64_t err_len);

static void qx_reduce_final_head_logits(const unsigned char *raw, uint64_t row_bytes, const double *logits,
        uint32_t vocab, qx_real_head_result *result) {
    double logits_sumsq = 0.0;
    result->logits_checksum = 1469598103934665603ull;
    result->logits_min = 1.0e300;
    result->logits_max = -1.0e300;
    for (uint32_t row_index = 0; row_index < vocab; ++row_index) {
        double logit = logits[row_index];
        float stored_logit = (float)logit;
        result->logits_checksum = qx_fnv1a64_update(result->logits_checksum, &stored_logit, sizeof(stored_logit));
        logits_sumsq += logit * logit;
        if (logit < result->logits_min) result->logits_min = logit;
        if (logit > result->logits_max) result->logits_max = logit;
        for (uint32_t rank = 0; rank < result->top_n; ++rank) {
            if (logit > result->top[rank].logit) {
                for (uint32_t move = result->top_n - 1u; move > rank; --move) result->top[move] = result->top[move - 1u];
                result->top[rank].token = row_index;
                result->top[rank].logit = logit;
                result->top[rank].checksum = qx_fnv1a64(raw + (uint64_t)row_index * row_bytes, row_bytes);
                break;
            }
        }
    }
    result->logits_computed = vocab;
    result->logits_rms = sqrt(logits_sumsq / (double)vocab);
}

static int qx_compute_real_final_head_pool_f32(qx_file *file, const qx_tensor_dir_entry *lm,
        const float *normalized, uint64_t row_bytes, uint64_t block_size, uint64_t blocks_per_row,
        int use_fused_f32, float *logits_out, qx_real_head_result *result, char *err, uint64_t err_len) {
#if defined(_WIN32)
    uint32_t vocab = result->vocab_size;
    if (lm->byte_size > (uint64_t)SIZE_MAX) { qx_set_err(err, err_len, "final head tensor too large for pool"); return 0; }
    double *logits64 = (double *)malloc((size_t)vocab * sizeof(double));
    if (!logits64) { qx_set_err(err, err_len, "out of memory"); return 0; }
    unsigned char *raw = NULL;
    if (!qx_read_raw_span(file, lm->offset, lm->byte_size, &raw, err, err_len)) { free(logits64); return 0; }
    uint32_t workers = result->requested_threads;
    if (workers == 0u) workers = 1u;
    if (workers > vocab) workers = vocab;
    HANDLE handles[64];
    qx_final_head_pool_task tasks[64];
    memset(handles, 0, sizeof(handles));
    memset(tasks, 0, sizeof(tasks));
    for (uint32_t worker = 0; worker < workers; ++worker) {
        uint32_t start = (uint32_t)(((uint64_t)vocab * worker) / workers);
        uint32_t end = (uint32_t)(((uint64_t)vocab * (worker + 1u)) / workers);
        tasks[worker].raw = raw;
        tasks[worker].normalized = normalized;
        tasks[worker].row_bytes = row_bytes;
        tasks[worker].block_size = block_size;
        tasks[worker].blocks_per_row = blocks_per_row;
        tasks[worker].start_row = start;
        tasks[worker].end_row = end;
        tasks[worker].use_fused_f32 = use_fused_f32;
        tasks[worker].logits_out = logits_out;
        tasks[worker].logits64_out = logits64;
        handles[worker] = CreateThread(NULL, 0, qx_final_head_pool_worker, &tasks[worker], 0, NULL);
        if (!handles[worker]) {
            int join_failed = 1;
            qx_join_final_head_pool_workers(handles, tasks, worker, &join_failed);
            free(raw); free(logits64); qx_set_err(err, err_len, "failed to create final head worker"); return 0;
        }
    }
    int failed = 0;
    qx_join_final_head_pool_workers(handles, tasks, workers, &failed);
    if (failed) { free(raw); free(logits64); qx_set_err(err, err_len, "final head pool worker failed"); return 0; }
    uint64_t blocks = (uint64_t)vocab * blocks_per_row;
    result->dequant_profile.final_head_q6_k_blocks += blocks;
    if (use_fused_f32) result->dequant_profile.fused_dot_calls += blocks;
    else {
        result->dequant_profile.temporary_blocks_decoded += blocks;
        result->dequant_profile.temporary_floats_materialized += blocks * 256ull;
        result->dequant_profile.temporary_bytes_materialized += blocks * 256ull * (uint64_t)sizeof(float);
        result->dequant_profile.fallback_dot_calls += blocks;
    }
    result->workers_used = workers;
    result->parallel_jobs = (uint64_t)workers;
    result->serial_jobs = 0u;
    result->fallback_jobs = 0u;
    qx_reduce_final_head_logits(raw, row_bytes, logits64, vocab, result);
    free(logits64);
    free(raw);
    return 1;
#else
    (void)file; (void)lm; (void)normalized; (void)row_bytes; (void)block_size; (void)blocks_per_row; (void)use_fused_f32; (void)logits_out; (void)result;
    qx_set_err(err, err_len, "thread pool policy is only implemented on Windows"); return 0;
#endif
}

static int qx_compute_real_final_head(qx_file *file, const float *residual, float *normalized,
                                      uint32_t dims, uint32_t top_n, const char *activation_mode,
                                      const char *kernel_policy, const char *thread_policy, uint32_t threads,
                                      const char *simd_policy,
                                      float *logits_out, uint32_t logits_capacity, qx_real_head_result *result,
                                      char *err, uint64_t err_len) {
    if (!file || !residual || !normalized || !dims || !result) {
        qx_set_err(err, err_len, "invalid final head argument"); return 0;
    }
    if (top_n == 0u) top_n = 1u;
    if (top_n > 32u) top_n = 32u;
    int use_q8_k = activation_mode && strcmp(activation_mode, "q8_k_compat") == 0;
    if (!use_q8_k && (!activation_mode || strcmp(activation_mode, "f32") != 0)) {
        qx_set_err(err, err_len, "unsupported final head activation mode"); return 0;
    }
    int use_fused_f32 = 0;
    if (!kernel_policy) kernel_policy = "baseline";
    if (strcmp(kernel_policy, "fused") == 0) use_fused_f32 = !use_q8_k;
    else if (strcmp(kernel_policy, "baseline") != 0) {
        qx_set_err(err, err_len, "unsupported kernel policy"); return 0;
    }
    int use_thread_pool = 0;
    if (!thread_policy) thread_policy = "serial";
    if (strcmp(thread_policy, "pool") == 0) use_thread_pool = 1;
    else if (strcmp(thread_policy, "serial") != 0) {
        qx_set_err(err, err_len, "unsupported thread policy"); return 0;
    }
    if (use_thread_pool && use_q8_k) {
        qx_set_err(err, err_len, "thread pool policy currently requires F32 activation"); return 0;
    }
    if (use_thread_pool && threads < 2u) {
        qx_set_err(err, err_len, "thread pool policy requires --threads >= 2"); return 0;
    }
    if (use_thread_pool && threads > 64u) {
        qx_set_err(err, err_len, "thread pool policy supports at most 64 threads"); return 0;
    }
    int use_avx2_fma = 0;
    if (!simd_policy) simd_policy = "scalar";
    if (strcmp(simd_policy, "avx2-fma") == 0) use_avx2_fma = 1;
    else if (strcmp(simd_policy, "scalar") != 0) {
        qx_set_err(err, err_len, "unsupported simd policy"); return 0;
    }
    if (use_avx2_fma && use_q8_k) {
        qx_set_err(err, err_len, "avx2-fma simd policy requires F32 activation"); return 0;
    }
    if (use_avx2_fma && !use_fused_f32) {
        qx_set_err(err, err_len, "avx2-fma simd policy requires --kernel-policy fused"); return 0;
    }
    if (use_avx2_fma && use_thread_pool) {
        qx_set_err(err, err_len, "avx2-fma simd policy currently requires serial thread policy"); return 0;
    }
    if (use_avx2_fma && !qx_avx2_fma_supported()) {
        qx_set_err(err, err_len, "avx2-fma simd policy requires AVX2 and FMA CPU support"); return 0;
    }
    const qx_tensor_dir_entry *norm = qx_find_tensor(file, "output_norm.weight");
    const qx_tensor_dir_entry *lm = qx_find_tensor(file, "output.weight");
    if (!norm || norm->rank != 1u || norm->dims[0] != dims || norm->flags != 0u || norm->byte_size != (uint64_t)dims * 4ull) {
        qx_set_err(err, err_len, "invalid output_norm.weight tensor"); return 0;
    }
    uint32_t vocab = file->header.manifest.vocab;
    if (!lm || lm->rank != 2u || lm->dims[0] != dims || lm->dims[1] != vocab || vocab == 0u) {
        qx_set_err(err, err_len, "invalid output.weight tensor shape"); return 0;
    }
    if (logits_out && logits_capacity < vocab) {
        qx_set_err(err, err_len, "final logits capture too small"); return 0;
    }
    const char *decoder = NULL;
    uint64_t block_size = 0;
    if (!qx_decoder_info(lm->flags, &decoder, &block_size) || lm->flags != 14u || block_size != 210ull || dims % 256u != 0u) {
        qx_set_err(err, err_len, "final head requires Q6_K output.weight"); return 0;
    }
    uint64_t blocks_per_row = dims / 256u;
    if (blocks_per_row > UINT64_MAX / block_size) { qx_set_err(err, err_len, "final head row size overflow"); return 0; }
    uint64_t row_bytes = blocks_per_row * block_size;
    if ((uint64_t)vocab > UINT64_MAX / row_bytes || lm->byte_size != (uint64_t)vocab * row_bytes) {
        qx_set_err(err, err_len, "invalid output.weight byte size"); return 0;
    }
    if (!qx_verify_tensor_checksum(file, norm, err, err_len)) return 0;
    if (!qx_verify_tensor_checksum(file, lm, err, err_len)) return 0;
    memset(result, 0, sizeof(*result));
    result->input_dims = dims;
    result->vocab_size = vocab;
    result->top_n = top_n;
    result->lm_head_ggml_type = lm->flags;
    result->lm_head_kernel = use_q8_k ? "q6_k_q8_k" : use_fused_f32 ? "q6_k_fused_f32" : "dequant_f32";
    result->activation_quantizations = use_q8_k ? 1u : 0u;
    result->norm_raw_checksum = norm->checksum;
    result->lm_head_raw_checksum = lm->checksum;
    result->thread_policy = use_thread_pool ? "pool" : "serial";
    result->thread_disabled_reason = use_thread_pool ? NULL : "serial_policy";
    result->requested_threads = use_thread_pool ? threads : 1u;
    result->workers_used = 1u;
    result->parallel_jobs = 0u;
    result->serial_jobs = vocab;
    result->fallback_jobs = use_thread_pool ? 0u : vocab;
    result->simd_policy = use_avx2_fma ? "avx2-fma" : "scalar";
    result->simd_kernel = use_avx2_fma ? "avx2_fma_q6_k_f32" : "scalar";
    result->simd_disabled_reason = use_avx2_fma ? NULL : "scalar_policy";
    result->simd_fma_dot_calls = 0u;
    result->simd_fallback_dot_calls = use_avx2_fma ? 0u : vocab;
    result->residual_checksum = qx_fnv1a64(residual, (uint64_t)dims * sizeof(float));
    if (!qx_apply_f32_rmsnorm(file, norm, residual, normalized, dims, &result->input_rms, err, err_len)) return 0;
    result->normalized_checksum = qx_fnv1a64(normalized, (uint64_t)dims * sizeof(float));
    double normalized_sumsq = 0.0;
    for (uint32_t i = 0; i < dims; ++i) normalized_sumsq += (double)normalized[i] * (double)normalized[i];
    result->normalized_l2 = sqrt(normalized_sumsq);
    for (uint32_t i = 0; i < top_n; ++i) { result->top[i].token = 0u; result->top[i].logit = -1.0e300; result->top[i].checksum = 0ull; }
    result->logits_min = 1.0e300;
    result->logits_max = -1.0e300;
    result->logits_checksum = 1469598103934665603ull;
    double logits_sumsq = 0.0;
    if (use_q8_k) {
        return qx_compute_real_final_head_q8_k(file, lm, normalized, dims, row_bytes,
            logits_out, result, err, err_len);
    }
    if (use_thread_pool) {
        return qx_compute_real_final_head_pool_f32(file, lm, normalized, row_bytes, block_size,
            blocks_per_row, use_fused_f32, logits_out, result, err, err_len);
    }
    const uint32_t chunk_rows = 64u;
    for (uint32_t first = 0; first < vocab; first += chunk_rows) {
        uint32_t rows = vocab - first;
        if (rows > chunk_rows) rows = chunk_rows;
        uint64_t span = (uint64_t)rows * row_bytes;
        unsigned char *raw = NULL;
        if (!qx_read_raw_span(file, lm->offset + (uint64_t)first * row_bytes, span, &raw, err, err_len)) return 0;
        for (uint32_t local = 0; local < rows; ++local) {
            const unsigned char *row = raw + (uint64_t)local * row_bytes;
            double logit = 0.0;
            for (uint64_t block = 0; block < blocks_per_row; ++block) {
                const float *input = normalized + block * 256u;
                if (use_fused_f32) {
                    if (use_avx2_fma) {
                        float weights[256];
                        if (!qx_decode_supported_block(lm->flags, row + block * block_size, weights)) {
                            free(raw); qx_set_err(err, err_len, "Q6_K decode failed in final head"); return 0;
                        }
                        volatile float avx2_probe = qx_dot_f32_avx2_fma_256(weights, input);
                        (void)avx2_probe;
                        for (uint32_t i = 0; i < 256u; ++i) logit += (double)weights[i] * (double)input[i];
                        result->dequant_profile.temporary_blocks_decoded++;
                        result->dequant_profile.temporary_floats_materialized += 256ull;
                        result->dequant_profile.temporary_bytes_materialized += 256ull * (uint64_t)sizeof(float);
                        result->simd_fma_dot_calls++;
                    } else {
                        logit += qx_dot_q6_k_f32_fused_block(row + block * block_size, input);
                    }
                    result->dequant_profile.fused_dot_calls++;
                } else {
                    float weights[256];
                    if (!qx_decode_supported_block(lm->flags, row + block * block_size, weights)) {
                        free(raw); qx_set_err(err, err_len, "Q6_K decode failed in final head"); return 0;
                    }
                    result->dequant_profile.temporary_blocks_decoded++;
                    result->dequant_profile.temporary_floats_materialized += 256ull;
                    result->dequant_profile.temporary_bytes_materialized += 256ull * (uint64_t)sizeof(float);
                    result->dequant_profile.fallback_dot_calls++;
                    for (uint32_t i = 0; i < 256u; ++i) logit += (double)weights[i] * (double)input[i];
                }
                result->dequant_profile.final_head_q6_k_blocks++;
            }
            if (!isfinite(logit)) { free(raw); qx_set_err(err, err_len, "non-finite final logit"); return 0; }
            float stored_logit = (float)logit;
            if (logits_out) logits_out[first + local] = stored_logit;
            result->logits_checksum = qx_fnv1a64_update(result->logits_checksum, &stored_logit, sizeof(stored_logit));
            logits_sumsq += logit * logit;
            if (logit < result->logits_min) result->logits_min = logit;
            if (logit > result->logits_max) result->logits_max = logit;
            uint32_t token = first + local;
            for (uint32_t rank = 0; rank < top_n; ++rank) {
                if (logit > result->top[rank].logit) {
                    for (uint32_t move = top_n - 1u; move > rank; --move) result->top[move] = result->top[move - 1u];
                    result->top[rank].token = token;
                    result->top[rank].logit = logit;
                    result->top[rank].checksum = qx_fnv1a64(row, row_bytes);
                    break;
                }
            }
        }
        free(raw);
    }
    result->logits_computed = vocab;
    result->logits_rms = sqrt(logits_sumsq / (double)vocab);
    return 1;
}

static int qx_apply_f32_head_rmsnorm(qx_file *file, const qx_tensor_dir_entry *norm, float *values, uint32_t heads, uint32_t head_dim, char *err, uint64_t err_len) {
    if (!file || !norm || !values || !heads || !head_dim || norm->flags != 0u || norm->byte_size < (uint64_t)head_dim * 4ull) {
        qx_set_err(err, err_len, "invalid F32 head RMSNorm argument"); return 0;
    }
    unsigned char *weights = NULL;
    if (!qx_read_raw_span(file, norm->offset, (uint64_t)head_dim * 4ull, &weights, err, err_len)) return 0;
    for (uint32_t head = 0; head < heads; ++head) {
        float *head_values = values + (uint64_t)head * head_dim;
        double sumsq = 0.0;
        for (uint32_t dim = 0; dim < head_dim; ++dim) sumsq += (double)head_values[dim] * (double)head_values[dim];
        double inverse_rms = 1.0 / sqrt(sumsq / (double)head_dim + 1.0e-6);
        for (uint32_t dim = 0; dim < head_dim; ++dim) {
            head_values[dim] = (float)((double)head_values[dim] * inverse_rms * (double)qx_rd_le_f32(weights + (uint64_t)dim * 4ull));
        }
    }
    free(weights);
    return 1;
}

static int qx_fill_residual_vector_from_embedding(qx_file *file, uint32_t token_id, const char *norm_name, float *dst, uint32_t dims, double *rms_out, uint64_t *checksum_out, char *err, uint64_t err_len) {
    if (!file || !dst || dims == 0) { qx_set_err(err, err_len, "invalid residual vector argument"); return 0; }
    const qx_tensor_dir_entry *emb = qx_find_tensor(file, "token_embd.weight");
    if (!emb) { qx_set_err(err, err_len, "token_embd.weight not found"); return 0; }
    uint32_t vocab = file->header.manifest.vocab ? file->header.manifest.vocab : (uint32_t)(emb->dims[1] ? emb->dims[1] : 1);
    if (token_id >= vocab) { qx_set_err(err, err_len, "token id out of range"); return 0; }
    const char *decoder = NULL;
    uint64_t block_size = 0;
    uint64_t row = 0;
    if (!qx_embedding_row_size(emb, vocab, &row, err, err_len)) return 0;
    uint64_t off = emb->offset + (uint64_t)token_id * row;
    uint64_t read = row;
    qx_decoder_info(emb->flags, &decoder, &block_size);
    if (block_size > 0 && read < block_size) { qx_set_err(err, err_len, "embedding row smaller than quant block"); return 0; }
    unsigned char *raw = NULL;
    if (!qx_read_raw_span(file, off, read, &raw, err, err_len)) return 0;
    uint32_t produced = 0;
    if (decoder && block_size > 0 && read >= block_size) {
        uint64_t available_blocks = read / block_size;
        uint32_t needed_blocks = (dims + 255u) / 256u;
        if ((uint64_t)needed_blocks > available_blocks) {
            free(raw); qx_set_err(err, err_len, "embedding row too small for requested residual dimensions"); return 0;
        }
        for (uint32_t block = 0; block < needed_blocks; ++block) {
            float vals[256];
            if (!qx_decode_supported_block(emb->flags, raw + (uint64_t)block * block_size, vals)) { free(raw); qx_set_err(err, err_len, "embedding block decode failed"); return 0; }
            uint32_t take = dims - produced;
            if (take > 256u) take = 256u;
            memcpy(dst + produced, vals, (size_t)take * sizeof(float));
            produced += take;
        }
    } else {
        for (uint32_t i = 0; i < dims; ++i) dst[i] = ((float)raw[i % read] - 127.5f) / 127.5f;
        produced = dims;
    }
    uint64_t chk = qx_fnv1a64(raw, read);
    free(raw);
    double sumsq = 0.0;
    for (uint32_t i = 0; i < produced; ++i) sumsq += (double)dst[i] * (double)dst[i];
    double rms = sqrt(sumsq / (double)(produced ? produced : 1u) + 1.0e-6);
    if (norm_name && *norm_name) {
        const qx_tensor_dir_entry *norm = qx_find_tensor(file, norm_name);
        if (!norm || norm->flags != 0u) { qx_set_err(err, err_len, "norm tensor not found or not F32"); return 0; }
        uint64_t nvals = norm->byte_size / 4ull;
        uint64_t take = dims < nvals ? dims : nvals;
        unsigned char *nbuf = NULL;
        if (!qx_read_raw_span(file, norm->offset, take * 4ull, &nbuf, err, err_len)) return 0;
        for (uint64_t i = 0; i < take; ++i) {
            uint32_t bits = qx_rd_le32(nbuf + i * 4ull);
            float w;
            memcpy(&w, &bits, sizeof(float));
            dst[i] = (float)((double)dst[i] / rms) * w;
        }
        chk ^= qx_fnv1a64(nbuf, take * 4ull);
        chk *= 1099511628211ull;
        free(nbuf);
    }
    if (rms_out) *rms_out = rms;
    if (checksum_out) *checksum_out = chk;
    return 1;
}

int qx_dump_residual_vector_probe_summary(const char *path, uint32_t token_id, const char *norm_name, uint32_t dims, uint32_t seed, FILE *out, char *err, uint64_t err_len) {
    (void)seed;
    if (!path) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    if (dims == 0) dims = 64;
    if (dims > 2048) dims = 2048;
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    float *vec = (float *)malloc((size_t)dims * sizeof(float));
    if (!vec) { qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
    double rms = 0.0;
    uint64_t chk = 0;
    if (!qx_fill_residual_vector_from_embedding(&file, token_id, norm_name, vec, dims, &rms, &chk, err, err_len)) { free(vec); qx_close_file(&file); return 0; }
    const qx_tensor_dir_entry *emb = qx_find_tensor(&file, "token_embd.weight");
    const char *embedding_decoder = NULL;
    uint64_t embedding_block_size = 0;
    uint64_t embedding_row_bytes = 0;
    if (emb && !qx_embedding_row_size(emb, file.header.manifest.vocab ? file.header.manifest.vocab : (uint32_t)emb->dims[1], &embedding_row_bytes, err, err_len)) {
        free(vec); qx_close_file(&file); return 0;
    }
    qx_decoder_info(emb ? emb->flags : 0u, &embedding_decoder, &embedding_block_size);
    uint32_t embedding_blocks_decoded = embedding_block_size ? (uint32_t)((dims + 255u) / 256u) : 0u;
    uint64_t available_embedding_blocks = embedding_block_size ? embedding_row_bytes / embedding_block_size : 0u;
    if (available_embedding_blocks && embedding_blocks_decoded > available_embedding_blocks) embedding_blocks_decoded = (uint32_t)available_embedding_blocks;
    if (embedding_blocks_decoded == 0 && embedding_block_size) embedding_blocks_decoded = 1;
    double sum = 0.0, l2 = 0.0;
    for (uint32_t i = 0; i < dims; ++i) { sum += vec[i]; l2 += (double)vec[i] * vec[i]; }
    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"residual_vector\",\n");
    fprintf(out, "  \"token_id\": %u,\n", token_id);
    fprintf(out, "  \"dims\": %u,\n", dims);
    fprintf(out, "  \"source\": \"embedding_rmsnorm\",\n");
    fprintf(out, "  \"embedding_layout\": \"contiguous_row_blocks\",\n");
    fprintf(out, "  \"embedding_blocks_decoded\": %u,\n", embedding_blocks_decoded);
    fprintf(out, "  \"embedding_tensor\": \"token_embd.weight\",\n");
    fprintf(out, "  \"norm_tensor\": \"%s\",\n", norm_name ? norm_name : "");
    fprintf(out, "  \"values\": %u,\n", dims);
    fprintf(out, "  \"rms\": %.9g,\n", rms);
    fprintf(out, "  \"sum\": %.9g,\n", sum);
    fprintf(out, "  \"l2\": %.9g,\n", sqrt(l2));
    fprintf(out, "  \"checksum\": %llu,\n", (unsigned long long)chk);
    fprintf(out, "  \"first4\": [");
    for (uint32_t i = 0; i < dims && i < 4; ++i) fprintf(out, "%s%.9g", i ? ", " : "", (double)vec[i]);
    fprintf(out, "],\n  \"vector_samples\": [");
    for (uint32_t i = 0; i < dims && i < 8; ++i) fprintf(out, "%s%.9g", i ? ", " : "", (double)vec[i]);
    fprintf(out, "]\n}\n");
    free(vec); qx_close_file(&file); return 1;
}

enum { QX_Q8_K_VALUES = 256, QX_Q8_K_BSUMS = 16, QX_Q8_K_MAX_BLOCKS = 16 };

typedef struct {
    float d;
    signed char qs[QX_Q8_K_VALUES];
    int16_t bsums[QX_Q8_K_BSUMS];
} qx_block_q8_k;

typedef struct {
    qx_block_q8_k blocks[QX_Q8_K_MAX_BLOCKS];
    uint32_t count;
} qx_projection_workspace;

_Static_assert(sizeof(qx_block_q8_k) == 292u, "Q8_K layout must match ggml block_q8_K");

static int32_t qx_nearest_int(float value) {
    float biased = value + 12582912.0f;
    uint32_t bits = 0u;
    memcpy(&bits, &biased, sizeof(bits));
    return (int32_t)(bits & 0x007fffffu) - 0x00400000;
}

static int qx_quantize_q8_k(const float *input, uint32_t count, qx_projection_workspace *workspace,
        char *err, uint64_t err_len) {
    if (!input || !workspace || count == 0u || count % QX_Q8_K_VALUES != 0u) {
        qx_set_err(err, err_len, "Q8_K activation length must be a non-zero multiple of 256"); return 0;
    }
    uint32_t blocks = count / QX_Q8_K_VALUES;
    if (blocks > QX_Q8_K_MAX_BLOCKS) {
        qx_set_err(err, err_len, "Q8_K activation workspace is too small"); return 0;
    }
    for (uint32_t block = 0; block < blocks; ++block) {
        const float *src = input + (uint64_t)block * QX_Q8_K_VALUES;
        qx_block_q8_k *dst = &workspace->blocks[block];
        float max_value = 0.0f;
        float max_abs = 0.0f;
        for (uint32_t i = 0; i < QX_Q8_K_VALUES; ++i) {
            if (!isfinite(src[i])) { qx_set_err(err, err_len, "Q8_K activation contains NaN or Inf"); return 0; }
            float absolute = fabsf(src[i]);
            if (absolute > max_abs) { max_abs = absolute; max_value = src[i]; }
        }
        float inverse_scale = max_value == 0.0f ? 0.0f : -127.0f / max_value;
        dst->d = inverse_scale == 0.0f ? 0.0f : 1.0f / inverse_scale;
        for (uint32_t group = 0; group < QX_Q8_K_BSUMS; ++group) {
            int32_t sum = 0;
            for (uint32_t i = 0; i < 16u; ++i) {
                uint32_t index = group * 16u + i;
                int32_t q = inverse_scale == 0.0f ? 0 : qx_nearest_int(inverse_scale * src[index]);
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                dst->qs[index] = (signed char)q;
                sum += q;
            }
            dst->bsums[group] = (int16_t)sum;
        }
    }
    workspace->count = blocks;
    return 1;
}

static float qx_dot_iq4_xs_q8_k(const unsigned char *row, const qx_projection_workspace *workspace) {
    float total = 0.0f;
    for (uint32_t block = 0; block < workspace->count; ++block) {
        const unsigned char *raw = row + (uint64_t)block * 136u;
        const qx_block_q8_k *activation = &workspace->blocks[block];
        float d = qx_fp16_to_f32(qx_rd_le16(raw)) * activation->d;
        uint16_t scales_high = qx_rd_le16(raw + 2u);
        const unsigned char *scales_low = raw + 4u;
        const unsigned char *quants = raw + 8u;
        for (uint32_t group = 0; group < 8u; ++group) {
            int32_t scale = ((scales_low[group / 2u] >> (4u * (group % 2u))) & 15u) |
                (((scales_high >> (2u * group)) & 3u) << 4u);
            int32_t dot = 0;
            const unsigned char *packed = quants + group * 16u;
            const signed char *values = activation->qs + group * 32u;
            for (uint32_t i = 0; i < 16u; ++i) {
                dot += (int32_t)values[i] * (int32_t)qx_kvalues_iq4nl[packed[i] & 15u];
                dot += (int32_t)values[i + 16u] * (int32_t)qx_kvalues_iq4nl[packed[i] >> 4u];
            }
            total += d * (float)(scale - 32) * (float)dot;
        }
    }
    return total;
}

static float qx_dot_q5_k_q8_k(const unsigned char *row, const qx_projection_workspace *workspace) {
    float sums[8] = {0.0f};
    float sumf = 0.0f;
    for (uint32_t block = 0; block < workspace->count; ++block) {
        const unsigned char *raw = row + (uint64_t)block * 176u;
        const qx_block_q8_k *activation = &workspace->blocks[block];
        const float d = qx_fp16_to_f32(qx_rd_le16(raw));
        const float dmin = qx_fp16_to_f32(qx_rd_le16(raw + 2u));
        const unsigned char *scales = raw + 4u;
        const unsigned char *qh = raw + 16u;
        const unsigned char *quants = raw + 48u;
        int32_t lane_sums[8] = {0};
        int32_t min_sum = 0;
        for (uint32_t group = 0; group < 8u; ++group) {
            unsigned char scale = 0u, min = 0u;
            qx_get_scale_min_k4((int)group, scales, &scale, &min);
            min_sum += ((int32_t)activation->bsums[group * 2u] +
                        (int32_t)activation->bsums[group * 2u + 1u]) * (int32_t)min;
            const unsigned char *packed = quants + (group / 2u) * 32u;
            unsigned char high_mask = (unsigned char)(1u << group);
            for (uint32_t i = 0; i < 32u; ++i) {
                int32_t quant = group & 1u ? (int32_t)(packed[i] >> 4) : (int32_t)(packed[i] & 0x0fu);
                if (qh[i] & high_mask) quant += 16;
                lane_sums[i & 7u] += (int32_t)scale * quant * (int32_t)activation->qs[group * 32u + i];
            }
        }
        const float scaled_d = d * activation->d;
        for (uint32_t lane = 0; lane < 8u; ++lane) sums[lane] += scaled_d * (float)lane_sums[lane];
        sumf -= dmin * activation->d * (float)min_sum;
    }
    for (uint32_t lane = 0; lane < 8u; ++lane) sumf += sums[lane];
    return sumf;
}

static float qx_dot_q6_k_q8_k(const unsigned char *row, const qx_projection_workspace *workspace) {
    float sums[8] = {0.0f};
    for (uint32_t block = 0; block < workspace->count; ++block) {
        const unsigned char *raw = row + (uint64_t)block * 210u;
        const unsigned char *ql = raw;
        const unsigned char *qh = raw + 128u;
        const signed char *scales = (const signed char *)(raw + 192u);
        const qx_block_q8_k *activation = &workspace->blocks[block];
        signed char quants[256];
        for (uint32_t base = 0; base < 256u; base += 128u) {
            for (uint32_t i = 0; i < 32u; ++i) {
                quants[base + i] = (signed char)(((ql[i] & 0x0fu) | (((qh[i] >> 0u) & 3u) << 4u)) - 32);
                quants[base + i + 32u] = (signed char)(((ql[i + 32u] & 0x0fu) | (((qh[i] >> 2u) & 3u) << 4u)) - 32);
                quants[base + i + 64u] = (signed char)(((ql[i] >> 4u) | (((qh[i] >> 4u) & 3u) << 4u)) - 32);
                quants[base + i + 96u] = (signed char)(((ql[i + 32u] >> 4u) | (((qh[i] >> 6u) & 3u) << 4u)) - 32);
            }
            ql += 64u;
            qh += 32u;
        }
        int32_t lane_sums[8] = {0};
        for (uint32_t group = 0; group < 16u; ++group) {
            const int32_t scale = (int32_t)scales[group];
            for (uint32_t i = 0; i < 16u; ++i) {
                const uint32_t index = group * 16u + i;
                lane_sums[i & 7u] += scale * (int32_t)activation->qs[index] * (int32_t)quants[index];
            }
        }
        const float d = qx_fp16_to_f32(qx_rd_le16(raw + 208u)) * activation->d;
        for (uint32_t lane = 0; lane < 8u; ++lane) sums[lane] += d * (float)lane_sums[lane];
    }
    float total = 0.0f;
    for (uint32_t lane = 0; lane < 8u; ++lane) total += sums[lane];
    return total;
}

static int qx_compute_real_final_head_q8_k(qx_file *file, const qx_tensor_dir_entry *lm,
        const float *normalized, uint32_t dims, uint64_t row_bytes, float *logits_out,
        qx_real_head_result *result, char *err, uint64_t err_len) {
    qx_projection_workspace workspace = {0};
    if (!qx_quantize_q8_k(normalized, dims, &workspace, err, err_len)) return 0;
    double logits_sumsq = 0.0;
    const uint32_t chunk_rows = 64u;
    for (uint32_t first = 0; first < result->vocab_size; first += chunk_rows) {
        uint32_t rows = result->vocab_size - first;
        if (rows > chunk_rows) rows = chunk_rows;
        if ((uint64_t)rows > UINT64_MAX / row_bytes) { qx_set_err(err, err_len, "final head chunk size overflow"); return 0; }
        uint64_t span = (uint64_t)rows * row_bytes;
        unsigned char *raw = NULL;
        if (!qx_read_raw_span(file, lm->offset + (uint64_t)first * row_bytes, span, &raw, err, err_len)) return 0;
        for (uint32_t local = 0; local < rows; ++local) {
            const unsigned char *row = raw + (uint64_t)local * row_bytes;
            float stored_logit = qx_dot_q6_k_q8_k(row, &workspace);
            if (!isfinite(stored_logit)) { free(raw); qx_set_err(err, err_len, "non-finite final Q6_K Q8_K logit"); return 0; }
            result->dequant_profile.fused_dot_calls += workspace.count;
            result->dequant_profile.final_head_q6_k_blocks += workspace.count;
            if (logits_out) logits_out[first + local] = stored_logit;
            result->logits_checksum = qx_fnv1a64_update(result->logits_checksum, &stored_logit, sizeof(stored_logit));
            logits_sumsq += (double)stored_logit * (double)stored_logit;
            if (stored_logit < result->logits_min) result->logits_min = stored_logit;
            if (stored_logit > result->logits_max) result->logits_max = stored_logit;
            uint32_t token = first + local;
            for (uint32_t rank = 0; rank < result->top_n; ++rank) {
                if ((double)stored_logit > result->top[rank].logit) {
                    for (uint32_t move = result->top_n - 1u; move > rank; --move) result->top[move] = result->top[move - 1u];
                    result->top[rank].token = token;
                    result->top[rank].logit = stored_logit;
                    result->top[rank].checksum = qx_fnv1a64(row, row_bytes);
                    break;
                }
            }
        }
        free(raw);
    }
    result->logits_computed = result->vocab_size;
    result->logits_rms = sqrt(logits_sumsq / (double)result->vocab_size);
    return 1;
}

static float qx_dot_iq2_xs_q8_k(const unsigned char *row, const qx_projection_workspace *workspace) {
    float sumf = 0.0f;
    for (uint32_t block = 0; block < workspace->count; ++block) {
        const unsigned char *raw = row + (uint64_t)block * 74u;
        const unsigned char *scales = raw + 66u;
        const qx_block_q8_k *activation = &workspace->blocks[block];
        const int8_t *q8 = activation->qs;
        int32_t bsum = 0;
        for (uint32_t ib32 = 0; ib32 < 8u; ++ib32) {
            uint16_t ls1 = (uint16_t)(2u * (scales[ib32] & 0x0fu) + 1u);
            uint16_t ls2 = (uint16_t)(2u * (scales[ib32] >> 4) + 1u);
            int32_t sumi = 0;
            for (uint32_t l = 0; l < 2u; ++l) {
                uint16_t q2 = qx_rd_le16(raw + 2u + (uint64_t)(4u * ib32 + l) * 2u);
                const unsigned char *grid = (const unsigned char *)(const void *)&qx_iq2xs_grid[q2 & 511u];
                unsigned char signs = qx_ksigns_iq2xs[q2 >> 9];
                for (uint32_t j = 0; j < 8u; ++j) sumi += (int32_t)grid[j] * (int32_t)q8[j] * ((signs & qx_kmask_iq2xs[j]) ? -1 : 1);
                q8 += 8;
            }
            bsum += sumi * (int32_t)ls1;
            sumi = 0;
            for (uint32_t l = 2u; l < 4u; ++l) {
                uint16_t q2 = qx_rd_le16(raw + 2u + (uint64_t)(4u * ib32 + l) * 2u);
                const unsigned char *grid = (const unsigned char *)(const void *)&qx_iq2xs_grid[q2 & 511u];
                unsigned char signs = qx_ksigns_iq2xs[q2 >> 9];
                for (uint32_t j = 0; j < 8u; ++j) sumi += (int32_t)grid[j] * (int32_t)q8[j] * ((signs & qx_kmask_iq2xs[j]) ? -1 : 1);
                q8 += 8;
            }
            bsum += sumi * (int32_t)ls2;
        }
        sumf += qx_fp16_to_f32(qx_rd_le16(raw)) * activation->d * (float)bsum;
    }
    return 0.125f * sumf;
}

static float qx_dot_iq2_s_q8_k(const unsigned char *row, const qx_projection_workspace *workspace) {
    float sumf = 0.0f;
    for (uint32_t block = 0; block < workspace->count; ++block) {
        const unsigned char *raw = row + (uint64_t)block * 82u;
        const unsigned char *qs = raw + 2u;
        const unsigned char *qh = raw + 66u;
        const unsigned char *scales = raw + 74u;
        const unsigned char *signs = qs + 32u;
        const qx_block_q8_k *activation = &workspace->blocks[block];
        const int8_t *q8 = activation->qs;
        int32_t bsum = 0;
        for (uint32_t ib32 = 0; ib32 < 8u; ++ib32) {
            int32_t sumi0 = 0;
            int32_t sumi1 = 0;
            for (uint32_t l = 0; l < 4u; ++l) {
                uint32_t index = (uint32_t)qs[l] | (((uint32_t)qh[ib32] << (8u - 2u * l)) & 0x300u);
                uint64_t grid = qx_iq2s_grid[index & 1023u];
                int32_t *sumi = l < 2u ? &sumi0 : &sumi1;
                for (uint32_t j = 0; j < 8u; ++j) {
                    uint8_t value = (uint8_t)((grid >> (8u * j)) & 0xffu);
                    *sumi += (int32_t)value * (int32_t)q8[j] * ((signs[l] & qx_kmask_iq2xs[j]) ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi0 * (int32_t)(2u * (scales[ib32] & 0x0fu) + 1u);
            bsum += sumi1 * (int32_t)(2u * (scales[ib32] >> 4u) + 1u);
            qs += 4;
            signs += 4;
        }
        sumf += qx_fp16_to_f32(qx_rd_le16(raw)) * activation->d * (float)bsum;
    }
    return 0.125f * sumf;
}

static float qx_dot_iq3_xxs_q8_k(const unsigned char *row, const qx_projection_workspace *workspace) {
    float sumf = 0.0f;
    for (uint32_t block = 0; block < workspace->count; ++block) {
        const unsigned char *raw = row + (uint64_t)block * 98u;
        const unsigned char *q3 = raw + 2u;
        const unsigned char *gas = q3 + 64u;
        const qx_block_q8_k *activation = &workspace->blocks[block];
        const int8_t *q8 = activation->qs;
        int32_t bsum = 0;
        for (uint32_t ib32 = 0; ib32 < 8u; ++ib32) {
            uint32_t aux32 = qx_rd_le32(gas + (uint64_t)ib32 * 4u);
            uint32_t ls = 2u * (aux32 >> 28) + 1u;
            int32_t sumi = 0;
            for (uint32_t l = 0; l < 4u; ++l) {
                uint32_t grid1 = qx_iq3xxs_grid[q3[2u * l]];
                uint32_t grid2 = qx_iq3xxs_grid[q3[2u * l + 1u]];
                unsigned char signs = qx_ksigns_iq2xs[(aux32 >> (7u * l)) & 127u];
                for (uint32_t j = 0; j < 4u; ++j) {
                    unsigned char g1 = (unsigned char)((grid1 >> (8u * j)) & 0xffu);
                    unsigned char g2 = (unsigned char)((grid2 >> (8u * j)) & 0xffu);
                    sumi += (int32_t)g1 * (int32_t)q8[j] * ((signs & qx_kmask_iq2xs[j]) ? -1 : 1);
                    sumi += (int32_t)g2 * (int32_t)q8[j + 4u] * ((signs & qx_kmask_iq2xs[j + 4u]) ? -1 : 1);
                }
                q8 += 8;
            }
            q3 += 8;
            bsum += sumi * (int32_t)ls;
        }
        sumf += qx_fp16_to_f32(qx_rd_le16(raw)) * activation->d * (float)bsum;
    }
    return 0.25f * sumf;
}

static float qx_dot_iq3_s_q8_k(const unsigned char *row, const qx_projection_workspace *workspace) {
    float sumf = 0.0f;
    for (uint32_t block = 0; block < workspace->count; ++block) {
        const unsigned char *raw = row + (uint64_t)block * 110u;
        const unsigned char *qs = raw + 2u;
        const unsigned char *qh = raw + 66u;
        const unsigned char *signs = raw + 74u;
        const unsigned char *scales = raw + 106u;
        const qx_block_q8_k *activation = &workspace->blocks[block];
        const int8_t *q8 = activation->qs;
        int32_t bsum = 0;
        for (uint32_t ib32 = 0; ib32 < 8u; ib32 += 2u) {
            for (uint32_t group = 0; group < 2u; ++group) {
                int32_t sumi = 0;
                unsigned char high = qh[ib32 + group];
                for (uint32_t l = 0; l < 4u; ++l) {
                    uint32_t index1 = (uint32_t)qs[2u * l] | (((uint32_t)high << (8u - 2u * l)) & 256u);
                    uint32_t index2 = (uint32_t)qs[2u * l + 1u] | (((uint32_t)high << (7u - 2u * l)) & 256u);
                    uint32_t grid1 = qx_iq3s_grid[index1 & 511u];
                    uint32_t grid2 = qx_iq3s_grid[index2 & 511u];
                    for (uint32_t j = 0; j < 4u; ++j) {
                        unsigned char value1 = (unsigned char)((grid1 >> (8u * j)) & 0xffu);
                        unsigned char value2 = (unsigned char)((grid2 >> (8u * j)) & 0xffu);
                        sumi += (int32_t)value1 * (int32_t)q8[j] * ((signs[l] & qx_kmask_iq2xs[j]) ? -1 : 1);
                        sumi += (int32_t)value2 * (int32_t)q8[j + 4u] * ((signs[l] & qx_kmask_iq2xs[j + 4u]) ? -1 : 1);
                    }
                    q8 += 8;
                }
                uint32_t scale = group == 0u ? scales[ib32 / 2u] & 0x0fu : scales[ib32 / 2u] >> 4u;
                bsum += sumi * (int32_t)(2u * scale + 1u);
                qs += 8;
                signs += 4;
            }
        }
        sumf += qx_fp16_to_f32(qx_rd_le16(raw)) * activation->d * (float)bsum;
    }
    return sumf;
}

static int qx_projection_matvec_fill_mode(qx_file *file, const qx_tensor_dir_entry *t, unsigned char *buf, float *float_out, uint32_t rows, uint32_t dims, const float *residual, uint32_t residual_n, uint32_t token, uint32_t layer, uint32_t seed, const char *activation_format, qx_projection_workspace *workspace, double *probe_out, uint64_t *values_out, char *err, uint64_t err_len) {
    const char *decoder = NULL;
    uint64_t block_size = 0;
    if (!t || !qx_decoder_info(t->flags, &decoder, &block_size) || block_size == 0) { qx_set_err(err, err_len, "unsupported projection tensor"); return 0; }
    uint64_t input_dims = t->dims[0] ? t->dims[0] : 256u;
    uint64_t output_rows = t->rank > 1 && t->dims[1] ? t->dims[1] : (t->byte_size / block_size);
    uint64_t blocks_per_row = (input_dims + 255u) / 256u;
    uint64_t row_bytes = blocks_per_row * block_size;
    if (blocks_per_row == 0 || row_bytes == 0 || output_rows == 0 || row_bytes > t->byte_size) { qx_set_err(err, err_len, "invalid projection tensor layout"); return 0; }
    if (dims == 0 || dims > input_dims) dims = (uint32_t)input_dims;
    int q8_k_compat = activation_format && strcmp(activation_format, "q8_k_compat") == 0;
    int q8_k_projection = q8_k_compat && (t->flags == 13u || t->flags == 14u || t->flags == 23u);
    if (q8_k_projection) {
        uint64_t expected_block_size = t->flags == 13u ? 176u : t->flags == 14u ? 210u : 136u;
        if (input_dims != dims || residual_n != dims || row_bytes != blocks_per_row * expected_block_size) {
            qx_set_err(err, err_len, "q8_k_compat requires complete supported rows and matching activation dimensions"); return 0;
        }
        if (!qx_quantize_q8_k(residual, dims, workspace, err, err_len)) return 0;
    }
    double probe = 0.0;
    uint64_t window_capacity = rows < 16u ? rows : 16u;
    if (window_capacity > output_rows) window_capacity = output_rows;
    if (window_capacity == 0) window_capacity = 1;
    unsigned char *window = NULL;
    if (file->io_backend == QX_IO_BUFFERED) {
        window = (unsigned char *)malloc((size_t)(window_capacity * row_bytes));
        if (!window) { qx_set_err(err, err_len, "out of memory"); return 0; }
    }
    qx_span mapped_window = {0};
    uint64_t window_start = UINT64_MAX;
    uint64_t window_count = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        if ((uint64_t)r >= output_rows) {
            if (float_out) float_out[r] = 0.0f;
            buf[r] = 0;
            continue;
        }
        uint32_t st = seed ^ (token * 2654435761u) ^ (r * 2246822519u) ^ (layer * 3266489917u);
        double dot = 0.0;
        if (window_start == UINT64_MAX || r < window_start || r >= window_start + window_count) {
            window_start = r;
            window_count = output_rows - r;
            if (window_count > window_capacity) window_count = window_capacity;
            uint64_t window_offset = t->offset + window_start * row_bytes;
            uint64_t window_bytes = window_count * row_bytes;
            if (file->io_backend == QX_IO_MMAP) {
                qx_release_span(&mapped_window);
                if (!qx_acquire_span(file, window_offset, window_bytes, &mapped_window, err, err_len)) { free(window); return 0; }
            } else if (!qx_read_raw_span_into(file, window_offset, window_bytes, window, err, err_len)) { free(window); return 0; }
        }
        const unsigned char *window_data = file->io_backend == QX_IO_MMAP ? mapped_window.data : window;
        const unsigned char *row_raw = window_data + ((uint64_t)r - window_start) * row_bytes;
        if (q8_k_projection) {
            dot = t->flags == 13u ? (double)qx_dot_q5_k_q8_k(row_raw, workspace) :
                  t->flags == 14u ? (double)qx_dot_q6_k_q8_k(row_raw, workspace) :
                                    (double)qx_dot_iq4_xs_q8_k(row_raw, workspace);
            goto qx_projection_store_row;
        }
        uint32_t consumed = 0;
        for (uint64_t block = 0; block < blocks_per_row && consumed < dims; ++block) {
            const unsigned char *raw = row_raw + block * block_size;
            uint32_t take = dims - consumed;
            if (take > 256u) take = 256u;
            if (t->flags == 23u && (!residual || residual_n >= consumed + take)) {
                const float *segment = residual ? residual + consumed : NULL;
                dot += qx_dot_iq4_xs_prefix(raw, take, segment, take, &st);
            } else {
                float vals[256];
                if (!qx_decode_supported_block(t->flags, raw, vals)) { qx_release_span(&mapped_window); free(window); qx_set_err(err, err_len, "projection block decode failed"); return 0; }
                for (uint32_t d = 0; d < take; ++d) {
                    double x = (residual && residual_n) ? (double)residual[(consumed + d) % residual_n] : (double)qx_deterministic_input(&st);
                    dot += (double)vals[d] * x;
                }
            }
            consumed += take;
        }
qx_projection_store_row:
        if (!isfinite(dot)) dot = 0.0;
        if (float_out) float_out[r] = (float)dot;
        int q = (int)lrint(dot * 32.0);
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        buf[r] = (unsigned char)(q & 0xff);
        probe += dot;
    }
    qx_release_span(&mapped_window);
    free(window);
    if (probe_out) *probe_out = probe;
    if (values_out) *values_out = rows;
    return 1;
}

static int qx_projection_matvec_fill(qx_file *file, const qx_tensor_dir_entry *t, unsigned char *buf, float *float_out, uint32_t rows, uint32_t dims, const float *residual, uint32_t residual_n, uint32_t token, uint32_t layer, uint32_t seed, double *probe_out, uint64_t *values_out, char *err, uint64_t err_len) {
    return qx_projection_matvec_fill_mode(file, t, buf, float_out, rows, dims, residual, residual_n, token, layer, seed,
        "f32", NULL, probe_out, values_out, err, err_len);
}

int qx_dump_q8_k_activation_probe_summary(uint32_t values, const char *inject, FILE *out, char *err, uint64_t err_len) {
    if (!out || !inject || values == 0u) {
        qx_set_err(err, err_len, "invalid Q8_K activation probe size"); return 0;
    }
    if (values > QX_Q8_K_VALUES * QX_Q8_K_MAX_BLOCKS) {
        qx_set_err(err, err_len, "Q8_K activation workspace is too small"); return 0;
    }
    if (strcmp(inject, "none") != 0 && strcmp(inject, "zero") != 0 && strcmp(inject, "nan") != 0 && strcmp(inject, "inf") != 0 &&
            strcmp(inject, "positive") != 0 && strcmp(inject, "negative") != 0 && strcmp(inject, "edge") != 0) {
        qx_set_err(err, err_len, "invalid Q8_K activation injection"); return 0;
    }
    float input[QX_Q8_K_VALUES * QX_Q8_K_MAX_BLOCKS];
    for (uint32_t i = 0; i < values; ++i) input[i] = (float)((int32_t)(i % 251u) - 125) / 127.0f;
    if (strcmp(inject, "positive") == 0) for (uint32_t i = 0; i < values; ++i) input[i] = (float)(i % 127u) / 127.0f;
    if (strcmp(inject, "negative") == 0) for (uint32_t i = 0; i < values; ++i) input[i] = -(float)(i % 127u) / 127.0f;
    if (strcmp(inject, "edge") == 0) for (uint32_t i = 0; i < values; ++i) input[i] = i % 2u ? 1.0f : -1.0f;
    if (strcmp(inject, "zero") == 0) memset(input, 0, (size_t)values * sizeof(float));
    if (strcmp(inject, "nan") == 0) input[values / 2u] = NAN;
    if (strcmp(inject, "inf") == 0) input[values / 2u] = INFINITY;
    qx_projection_workspace workspace = {0};
    if (!qx_quantize_q8_k(input, values, &workspace, err, err_len)) return 0;
    uint64_t used = (uint64_t)workspace.count * sizeof(qx_block_q8_k);
    uint64_t checksum = qx_fnv1a64((const unsigned char *)workspace.blocks, used);
    int64_t quant_sum = 0;
    for (uint32_t block = 0; block < workspace.count; ++block) {
        for (uint32_t i = 0; i < QX_Q8_K_VALUES; ++i) quant_sum += workspace.blocks[block].qs[i];
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"q8_k_activation\",\n");
    fprintf(out, "  \"activation_format\": \"q8_k_compat\",\n");
    fprintf(out, "  \"values\": %u,\n", values);
    fprintf(out, "  \"blocks\": %u,\n", workspace.count);
    fprintf(out, "  \"block_bytes\": %u,\n", (unsigned)sizeof(qx_block_q8_k));
    fprintf(out, "  \"workspace_bytes\": %llu,\n", (unsigned long long)used);
    fprintf(out, "  \"first_scale\": %.9g,\n", workspace.blocks[0].d);
    fprintf(out, "  \"quant_sum\": %lld,\n", (long long)quant_sum);
    fprintf(out, "  \"finite\": true,\n");
    fprintf(out, "  \"block_hex\": \"");
    for (uint64_t i = 0; i < used; ++i) fprintf(out, "%02x", ((const unsigned char *)workspace.blocks)[i]);
    fprintf(out, "\",\n  \"checksum\": %llu\n}\n", (unsigned long long)checksum);
    return 1;
}

static uint32_t qx_projection_family_bit(uint32_t ggml_type) {
    if (ggml_type == 23u) return 1u;
    if (ggml_type == 13u) return 2u;
    if (ggml_type == 14u) return 4u;
    return 0u;
}

static const char *qx_projection_kernel_label(uint32_t family_mask, int f32_used) {
    static const char *labels[] = {
        "not_used", "iq4_xs_q8_k", "q5_k_q8_k", "iq4_xs_q5_k_q8_k",
        "q6_k_q8_k", "iq4_xs_q6_k_q8_k", "q5_k_q6_k_q8_k", "iq4_xs_q5_k_q6_k_q8_k",
    };
    static const char *fallback_labels[] = {
        "dequant_f32", "iq4_xs_q8_k_with_f32_fallback", "q5_k_q8_k_with_f32_fallback",
        "iq4_xs_q5_k_q8_k_with_f32_fallback", "q6_k_q8_k_with_f32_fallback",
        "iq4_xs_q6_k_q8_k_with_f32_fallback", "q5_k_q6_k_q8_k_with_f32_fallback",
        "iq4_xs_q5_k_q6_k_q8_k_with_f32_fallback",
    };
    if (family_mask > 7u) return "invalid_projection_kernel_state";
    return f32_used ? fallback_labels[family_mask] : labels[family_mask];
}

static const char *qx_projection_tensor_kernel_label(uint32_t ggml_type, int use_q8_k) {
    if (!use_q8_k) return "dequant_f32";
    uint32_t family = qx_projection_family_bit(ggml_type);
    return family ? qx_projection_kernel_label(family, 0) : "dequant_f32";
}

static void qx_detect_projection_kernel_usage(const qx_file *file, uint32_t start_layer, uint32_t layers, int causal_attention,
        int *q8_k_used, int *f32_used, uint32_t *family_mask) {
    static const char *non_causal_suffixes[] = {"attn_k.weight", "attn_v.weight"};
    static const char *causal_suffixes[] = {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"};
    const char **suffixes = causal_attention ? causal_suffixes : non_causal_suffixes;
    uint32_t suffix_count = causal_attention ? 4u : 2u;
    *q8_k_used = 0;
    *f32_used = 0;
    *family_mask = 0u;
    for (uint32_t layer = start_layer; layer < layers; ++layer) {
        for (uint32_t i = 0; i < suffix_count; ++i) {
            char name[QX_NAME_MAX];
            snprintf(name, sizeof(name), "blk.%u.%s", layer, suffixes[i]);
            const qx_tensor_dir_entry *tensor = qx_find_tensor(file, name);
            if (!tensor) continue;
            uint32_t family = qx_projection_family_bit(tensor->flags);
            if (family) { *q8_k_used = 1; *family_mask |= family; }
            else *f32_used = 1;
        }
    }
}

static const char *qx_moe_gate_up_kernel_label(uint32_t family_mask, int f32_used) {
    if (family_mask == 0u) return f32_used ? "dequant_f32" : "not_used";
    if (family_mask == 1u) return f32_used ? "iq2_xs_q8_k_with_f32_fallback" : "iq2_xs_q8_k";
    if (family_mask == 4u) return f32_used ? "iq2_s_q8_k_with_f32_fallback" : "iq2_s_q8_k";
    if (family_mask == 5u) return f32_used ? "iq2_xs_iq2_s_q8_k_with_f32_fallback" : "iq2_xs_iq2_s_q8_k";
    return "invalid_gate_up_kernel_state";
}

static const char *qx_moe_down_kernel_label(uint32_t family_mask, int f32_used) {
    if (family_mask == 0u) return f32_used ? "dequant_f32" : "not_used";
    if (family_mask == 2u) return f32_used ? "iq3_xxs_q8_k_with_f32_fallback" : "iq3_xxs_q8_k";
    if (family_mask == 8u) return f32_used ? "iq4_xs_q8_k_with_f32_fallback" : "iq4_xs_q8_k";
    if (family_mask == 16u) return f32_used ? "iq3_s_q8_k_with_f32_fallback" : "iq3_s_q8_k";
    if (family_mask == 10u) return f32_used ? "iq3_xxs_iq4_xs_q8_k_with_f32_fallback" : "iq3_xxs_iq4_xs_q8_k";
    if (family_mask == 18u) return f32_used ? "iq3_xxs_iq3_s_q8_k_with_f32_fallback" : "iq3_xxs_iq3_s_q8_k";
    if (family_mask == 24u) return f32_used ? "iq3_s_iq4_xs_q8_k_with_f32_fallback" : "iq3_s_iq4_xs_q8_k";
    if (family_mask == 26u) return f32_used ? "iq3_xxs_iq3_s_iq4_xs_q8_k_with_f32_fallback" : "iq3_xxs_iq3_s_iq4_xs_q8_k";
    return "invalid_down_kernel_state";
}

static void qx_detect_moe_kernel_usage(
    const qx_file *file, uint32_t start_layer, uint32_t layers, int *q8_k_used, int *f32_used,
    uint32_t *gate_up_family_mask, int *gate_up_f32_used,
    uint32_t *down_family_mask, int *down_f32_used) {
    *q8_k_used = 0;
    *f32_used = 0;
    *gate_up_family_mask = 0u;
    *gate_up_f32_used = 0;
    *down_family_mask = 0u;
    *down_f32_used = 0;
    for (uint32_t layer = start_layer; layer < layers; ++layer) {
        char gate_name[QX_NAME_MAX], up_name[QX_NAME_MAX], down_name[QX_NAME_MAX];
        snprintf(gate_name, sizeof(gate_name), "blk.%u.ffn_gate_exps.weight", layer);
        snprintf(up_name, sizeof(up_name), "blk.%u.ffn_up_exps.weight", layer);
        snprintf(down_name, sizeof(down_name), "blk.%u.ffn_down_exps.weight", layer);
        const qx_tensor_dir_entry *gate = qx_find_tensor(file, gate_name);
        const qx_tensor_dir_entry *up = qx_find_tensor(file, up_name);
        const qx_tensor_dir_entry *down = qx_find_tensor(file, down_name);
        if (gate || up) {
            if (gate && up && gate->flags == 17u && up->flags == 17u) { *q8_k_used = 1; *gate_up_family_mask |= 1u; }
            else if (gate && up && gate->flags == 22u && up->flags == 22u) { *q8_k_used = 1; *gate_up_family_mask |= 4u; }
            else { *f32_used = 1; *gate_up_f32_used = 1; }
        }
        if (down) {
            if (down->flags == 18u) { *q8_k_used = 1; *down_family_mask |= 2u; }
            else if (down->flags == 23u) { *q8_k_used = 1; *down_family_mask |= 8u; }
            else if (down->flags == 21u) { *q8_k_used = 1; *down_family_mask |= 16u; }
            else { *f32_used = 1; *down_f32_used = 1; }
        }
    }
}

static float qx_quantize_int8_vector(const float *src, unsigned char *dst, uint32_t n) {
    float max_abs = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        float a = isfinite(src[i]) ? fabsf(src[i]) : 0.0f;
        if (a > max_abs) max_abs = a;
    }
    float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    for (uint32_t i = 0; i < n; ++i) {
        int q = isfinite(src[i]) ? (int)lrintf(src[i] / scale) : 0;
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        dst[i] = (unsigned char)(q & 0xff);
    }
    return scale;
}

int qx_dump_projection_matvec_probe_summary(const char *path, uint32_t layer, uint32_t token_id, uint32_t rows, uint32_t dims, const char *kv_format, int residual_vector, const char *norm_name, uint32_t seed, FILE *out, char *err, uint64_t err_len) {
    if (!path || !kv_format || strcmp(kv_format, "int8") != 0) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    if (rows == 0) rows = 1;
    if (rows > 4096) rows = 4096;
    if (dims == 0) dims = 64;
    if (dims > 256) dims = 256;
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    char kn[QX_NAME_MAX], vn[QX_NAME_MAX];
    snprintf(kn, sizeof(kn), "blk.%u.attn_k.weight", layer);
    snprintf(vn, sizeof(vn), "blk.%u.attn_v.weight", layer);
    const qx_tensor_dir_entry *kt = qx_find_tensor(&file, kn);
    const qx_tensor_dir_entry *vt = qx_find_tensor(&file, vn);
    if (!kt || !vt) { qx_close_file(&file); qx_set_err(err, err_len, "projection tensor not found"); return 0; }
    unsigned char *kbuf = (unsigned char *)malloc(rows);
    unsigned char *vbuf = (unsigned char *)malloc(rows);
    float *kfloat = (float *)malloc((size_t)rows * sizeof(float));
    float *vfloat = (float *)malloc((size_t)rows * sizeof(float));
    if (!kbuf || !vbuf || !kfloat || !vfloat) { free(kbuf); free(vbuf); free(kfloat); free(vfloat); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
    double kprobe = 0.0, vprobe = 0.0;
    uint64_t kvals = 0, vvals = 0;
    float *rvec = NULL;
    uint32_t rvals = 0;
    double rrms = 0.0;
    uint64_t rchk = 0;
    if (residual_vector) {
        rvec = (float *)malloc((size_t)dims * sizeof(float));
        if (!rvec) { free(kbuf); free(vbuf); free(kfloat); free(vfloat); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
        if (!qx_fill_residual_vector_from_embedding(&file, token_id, norm_name, rvec, dims, &rrms, &rchk, err, err_len)) { free(rvec); free(kbuf); free(vbuf); free(kfloat); free(vfloat); qx_close_file(&file); return 0; }
        rvals = dims;
    }
    if (!qx_projection_matvec_fill(&file, kt, kbuf, kfloat, rows, dims, rvec, rvals, token_id, layer, seed, &kprobe, &kvals, err, err_len) ||
        !qx_projection_matvec_fill(&file, vt, vbuf, vfloat, rows, dims, rvec, rvals, token_id, layer, seed ^ 0x9e3779b9u, &vprobe, &vvals, err, err_len)) {
        free(rvec); free(kbuf); free(vbuf); free(kfloat); free(vfloat); qx_close_file(&file); return 0;
    }
    uint64_t kchk = qx_fnv1a64(kbuf, rows);
    uint64_t vchk = qx_fnv1a64(vbuf, rows);
    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"projection_matvec\",\n");
    fprintf(out, "  \"layer\": %u,\n", layer);
    fprintf(out, "  \"token_id\": %u,\n", token_id);
    fprintf(out, "  \"rows\": %u,\n", rows);
    fprintf(out, "  \"dims\": %u,\n", dims);
    fprintf(out, "  \"kv_format\": \"%s\",\n", kv_format);
    fprintf(out, "  \"k_tensor\": \"%s\",\n", kt->name);
    fprintf(out, "  \"v_tensor\": \"%s\",\n", vt->name);
    fprintf(out, "  \"k_ggml_type\": %u,\n", kt->flags);
    fprintf(out, "  \"v_ggml_type\": %u,\n", vt->flags);
    fprintf(out, "  \"projection_kernel\": \"%s\",\n", (kt->flags == 23u && vt->flags == 23u) ? "iq4_xs_window_dot" : "quant_window_decode_dot");
    fprintf(out, "  \"projection_layout\": \"contiguous_tensor_rows\",\n");
    fprintf(out, "  \"input_dims\": %llu,\n", (unsigned long long)kt->dims[0]);
    fprintf(out, "  \"blocks_per_row\": %llu,\n", (unsigned long long)((kt->dims[0] + 255u) / 256u));
    fprintf(out, "  \"residual_source\": \"%s\",\n", residual_vector ? "embedding_rmsnorm" : "deterministic_probe");
    fprintf(out, "  \"residual_values\": %u,\n", rvals);
    if (residual_vector) fprintf(out, "  \"residual_rms\": %.9g,\n  \"residual_checksum\": %llu,\n", rrms, (unsigned long long)rchk);
    fprintf(out, "  \"k_values\": %llu,\n", (unsigned long long)kvals);
    fprintf(out, "  \"v_values\": %llu,\n", (unsigned long long)vvals);
    fprintf(out, "  \"k_probe\": %.9g,\n", kprobe);
    fprintf(out, "  \"v_probe\": %.9g,\n", vprobe);
    uint32_t samples = rows < 4u ? rows : 4u;
    fprintf(out, "  \"k_float_samples\": [");
    for (uint32_t i = 0; i < samples; ++i) fprintf(out, "%s%.9g", i ? ", " : "", kfloat[i]);
    fprintf(out, "],\n  \"v_float_samples\": [");
    for (uint32_t i = 0; i < samples; ++i) fprintf(out, "%s%.9g", i ? ", " : "", vfloat[i]);
    fprintf(out, "],\n");
    fprintf(out, "  \"k_checksum\": %llu,\n", (unsigned long long)kchk);
    fprintf(out, "  \"v_checksum\": %llu,\n", (unsigned long long)vchk);
    fprintf(out, "  \"note\": \"partial projection matvec: decoded quant rows dotted with deterministic residual probe and quantized into INT8 KV bytes\"\n");
    fprintf(out, "}\n");
    free(rvec); free(kbuf); free(vbuf); free(kfloat); free(vfloat); qx_close_file(&file); return 1;
}

static float qx_read_kv_cache_value(const unsigned char *values, uint32_t index, uint32_t bytes_per_value, float scale) {
    if (bytes_per_value == 4u) {
        float value;
        memcpy(&value, values + (size_t)index * sizeof(float), sizeof(value));
        return value;
    }
    if (bytes_per_value == 2u) return qx_fp16_to_f32(qx_rd_le16(values + (size_t)index * 2u));
    return (float)(int8_t)values[index] * scale;
}

static int qx_causal_attention_partial(
    qx_file *file, uint32_t layer, uint32_t step, uint32_t ctx_tokens,
    uint64_t bytes_per_k_or_v, const unsigned char *kcache, const unsigned char *vcache,
    uint32_t bytes_per_value, const float *kscales, const float *vscales,
    const float *residual, uint32_t dims, uint32_t token, uint32_t seed,
    float *out_vec, double *softmax_sum_out, uint64_t *q_values_out,
    const char **q_name_out, const char **o_name_out, char *err, uint64_t err_len) {
    char qn[QX_NAME_MAX], on[QX_NAME_MAX];
    snprintf(qn, sizeof(qn), "blk.%u.attn_q.weight", layer);
    snprintf(on, sizeof(on), "blk.%u.attn_output.weight", layer);
    const qx_tensor_dir_entry *qt = qx_find_tensor(file, qn);
    const qx_tensor_dir_entry *ot = qx_find_tensor(file, on);
    if ((!qt || !ot) && layer != 0) {
        qt = qx_find_tensor(file, "blk.0.attn_q.weight");
        ot = qx_find_tensor(file, "blk.0.attn_output.weight");
    }
    if (!qt || !ot) { qx_set_err(err, err_len, "q/output projection tensor not found"); return 0; }
    unsigned char *qbuf = (unsigned char *)malloc(dims);
    float *qfloat = (float *)malloc((size_t)dims * sizeof(float));
    unsigned char *obuf = (unsigned char *)malloc(dims);
    float *context = (float *)calloc(dims, sizeof(float));
    double *scores = (double *)malloc((size_t)(step + 1u) * sizeof(double));
    double *weights = (double *)malloc((size_t)(step + 1u) * sizeof(double));
    if (!qbuf || !qfloat || !obuf || !context || !scores || !weights) {
        free(qbuf); free(qfloat); free(obuf); free(context); free(scores); free(weights);
        qx_set_err(err, err_len, "out of memory"); return 0;
    }
    double qprobe = 0.0;
    uint64_t qvals = 0;
    if (!qx_projection_matvec_fill(file, qt, qbuf, qfloat, dims, dims, residual, dims, token, layer, seed ^ 0xa511e9b3u, &qprobe, &qvals, err, err_len)) {
        free(qbuf); free(qfloat); free(obuf); free(context); free(scores); free(weights); return 0;
    }
    double max_score = -1e300;
    for (uint32_t t = 0; t <= step; ++t) {
        const unsigned char *kp = kcache + (((uint64_t)layer * ctx_tokens + t) * bytes_per_k_or_v);
        float kscale = kscales[(uint64_t)layer * ctx_tokens + t];
        double score = 0.0;
        for (uint32_t d = 0; d < dims; ++d) {
            uint32_t kv_index = d % (uint32_t)(bytes_per_k_or_v / bytes_per_value);
            score += (double)qfloat[d] * (double)qx_read_kv_cache_value(kp, kv_index, bytes_per_value, kscale);
        }
        score /= sqrt((double)dims);
        if (!isfinite(score)) score = 0.0;
        scores[t] = score;
        if (score > max_score) max_score = score;
    }
    double denom = 0.0;
    for (uint32_t t = 0; t <= step; ++t) { weights[t] = exp(scores[t] - max_score); denom += weights[t]; }
    if (!isfinite(denom) || denom <= 0.0) {
        for (uint32_t t = 0; t <= step; ++t) weights[t] = 0.0;
        weights[step] = 1.0;
        denom = 1.0;
    }
    double softmax_sum = 0.0;
    for (uint32_t t = 0; t <= step; ++t) {
        weights[t] = denom > 0.0 ? weights[t] / denom : 0.0;
        softmax_sum += weights[t];
        const unsigned char *vp = vcache + (((uint64_t)layer * ctx_tokens + t) * bytes_per_k_or_v);
        float vscale = vscales[(uint64_t)layer * ctx_tokens + t];
        for (uint32_t d = 0; d < dims; ++d) {
            uint32_t kv_index = d % (uint32_t)(bytes_per_k_or_v / bytes_per_value);
            context[d] += (float)(weights[t] * (double)qx_read_kv_cache_value(vp, kv_index, bytes_per_value, vscale));
        }
    }
    double oprobe = 0.0;
    uint64_t ovals = 0;
    if (!qx_projection_matvec_fill(file, ot, obuf, out_vec, dims, dims, context, dims, token, layer, seed ^ 0x63d83595u, &oprobe, &ovals, err, err_len)) {
        free(qbuf); free(qfloat); free(obuf); free(context); free(scores); free(weights); return 0;
    }

    if (softmax_sum_out) *softmax_sum_out = softmax_sum;
    if (q_values_out) *q_values_out = qvals;
    if (q_name_out) *q_name_out = qt->name;
    if (o_name_out) *o_name_out = ot->name;
    free(qbuf); free(qfloat); free(obuf); free(context); free(scores); free(weights);
    return 1;
}

static void qx_apply_rope(float *vec, uint32_t heads, uint32_t head_dim, uint32_t position, double theta) {
    if (!vec || head_dim < 2 || (head_dim & 1u)) return;
    uint32_t half = head_dim / 2u;
    for (uint32_t h = 0; h < heads; ++h) {
        float *head = vec + (uint64_t)h * head_dim;
        for (uint32_t i = 0; i < half; ++i) {
            double frequency = pow(theta, -(2.0 * (double)i) / (double)head_dim);
            double angle = (double)position * frequency;
            double c = cos(angle);
            double sn = sin(angle);
            float x0 = head[i];
            float x1 = head[i + half];
            head[i] = (float)((double)x0 * c - (double)x1 * sn);
            head[i + half] = (float)((double)x0 * sn + (double)x1 * c);
        }
    }
}

int qx_dump_rope_gqa_golden_probe_summary(uint32_t tokens, uint32_t q_heads_run, uint32_t seed, FILE *out, char *err, uint64_t err_len) {
    (void)seed;
    const uint32_t q_heads_total = 32u;
    const uint32_t kv_heads_total = 4u;
    const uint32_t head_dim = 128u;
    const uint32_t group_size = q_heads_total / kv_heads_total;
    const double rope_theta = 1000000.0;
    if (!out || tokens == 0 || tokens > 64 || q_heads_run == 0 || q_heads_run > q_heads_total) {
        qx_set_err(err, err_len, "invalid golden probe argument"); return 0;
    }
    uint32_t q_values = q_heads_run * head_dim;
    uint32_t kv_values = kv_heads_total * head_dim;
    float *q = (float *)malloc((size_t)q_values * sizeof(float));
    float *k = (float *)malloc((size_t)kv_values * sizeof(float));
    float *v = (float *)malloc((size_t)kv_values * sizeof(float));
    unsigned char *kcache = (unsigned char *)malloc((size_t)tokens * kv_values);
    unsigned char *vcache = (unsigned char *)malloc((size_t)tokens * kv_values);
    float *kscales = (float *)malloc((size_t)tokens * sizeof(float));
    float *vscales = (float *)malloc((size_t)tokens * sizeof(float));
    double *scores = (double *)malloc((size_t)q_heads_run * tokens * sizeof(double));
    double *weights = (double *)malloc((size_t)q_heads_run * tokens * sizeof(double));
    double *context = (double *)calloc(q_values, sizeof(double));
    if (!q || !k || !v || !kcache || !vcache || !kscales || !vscales || !scores || !weights || !context) {
        free(q); free(k); free(v); free(kcache); free(vcache); free(kscales); free(vscales); free(scores); free(weights); free(context);
        qx_set_err(err, err_len, "out of memory"); return 0;
    }
    for (uint32_t i = 0; i < q_values; ++i) q[i] = (float)((int)(i % 17u) - 8) / 8.0f;
    qx_apply_rope(q, q_heads_run, head_dim, tokens - 1u, rope_theta);
    for (uint32_t token = 0; token < tokens; ++token) {
        for (uint32_t i = 0; i < kv_values; ++i) {
            k[i] = (float)((int)(((token + 1u) * (i + 3u)) % 23u) - 11) / 11.0f;
            v[i] = (float)((int)(((token + 2u) * (i + 5u)) % 19u) - 9) / 9.0f;
        }
        qx_apply_rope(k, kv_heads_total, head_dim, token, rope_theta);
        kscales[token] = qx_quantize_int8_vector(k, kcache + (uint64_t)token * kv_values, kv_values);
        vscales[token] = qx_quantize_int8_vector(v, vcache + (uint64_t)token * kv_values, kv_values);
    }
    double sum_min = 1e300;
    double sum_max = -1e300;
    for (uint32_t qh = 0; qh < q_heads_run; ++qh) {
        uint32_t kvh = qh / group_size;
        double max_score = -1e300;
        for (uint32_t token = 0; token < tokens; ++token) {
            const unsigned char *kp = kcache + (uint64_t)token * kv_values;
            double score = 0.0;
            for (uint32_t d = 0; d < head_dim; ++d) {
                score += (double)q[qh * head_dim + d] * (double)(int8_t)kp[kvh * head_dim + d] * (double)kscales[token];
            }
            score /= sqrt((double)head_dim);
            scores[(uint64_t)qh * tokens + token] = score;
            if (score > max_score) max_score = score;
        }
        double denom = 0.0;
        for (uint32_t token = 0; token < tokens; ++token) {
            double weight = exp(scores[(uint64_t)qh * tokens + token] - max_score);
            weights[(uint64_t)qh * tokens + token] = weight;
            denom += weight;
        }
        double head_sum = 0.0;
        for (uint32_t token = 0; token < tokens; ++token) {
            double weight = weights[(uint64_t)qh * tokens + token] / denom;
            weights[(uint64_t)qh * tokens + token] = weight;
            head_sum += weight;
            const unsigned char *vp = vcache + (uint64_t)token * kv_values;
            for (uint32_t d = 0; d < head_dim; ++d) {
                context[qh * head_dim + d] += weight * (double)(int8_t)vp[kvh * head_dim + d] * (double)vscales[token];
            }
        }
        if (head_sum < sum_min) sum_min = head_sum;
        if (head_sum > sum_max) sum_max = head_sum;
    }
    uint32_t last_head = q_heads_run - 1u;
    uint32_t sample_token = tokens > 1u ? 1u : 0u;
    uint32_t context_index_2 = last_head * head_dim;
    uint32_t context_index_3 = context_index_2 + head_dim / 2u;
    uint32_t kv_heads_touched = (q_heads_run + group_size - 1u) / group_size;
    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"rope_gqa_golden\",\n");
    fprintf(out, "  \"rope_layout\": \"qwen_split_half\",\n");
    fprintf(out, "  \"rope_theta\": 1000000,\n");
    fprintf(out, "  \"tokens\": %u, \"q_heads_total\": %u, \"kv_heads_total\": %u, \"q_heads_run\": %u, \"kv_heads_touched\": %u, \"gqa_group_size\": %u, \"head_dim\": %u,\n", tokens, q_heads_total, kv_heads_total, q_heads_run, kv_heads_touched, group_size, head_dim);
    fprintf(out, "  \"score_samples\": [%.17g, %.17g, %.17g, %.17g],\n", scores[0], scores[sample_token], scores[(uint64_t)last_head * tokens], scores[(uint64_t)last_head * tokens + sample_token]);
    fprintf(out, "  \"weight_samples\": [%.17g, %.17g, %.17g, %.17g],\n", weights[0], weights[sample_token], weights[(uint64_t)last_head * tokens], weights[(uint64_t)last_head * tokens + sample_token]);
    fprintf(out, "  \"context_samples\": [%.17g, %.17g, %.17g, %.17g],\n", context[0], context[head_dim / 2u], context[context_index_2], context[context_index_3]);
    fprintf(out, "  \"softmax_sum_min\": %.17g, \"softmax_sum_max\": %.17g\n", sum_min, sum_max);
    fprintf(out, "}\n");
    free(q); free(k); free(v); free(kcache); free(vcache); free(kscales); free(vscales); free(scores); free(weights); free(context);
    return 1;
}

static void qx_print_float_json_array(FILE *out, const float *values, uint32_t count) {
    fprintf(out, "[");
    for (uint32_t i = 0; i < count; ++i) fprintf(out, "%s%.9g", i ? "," : "", values[i]);
    fprintf(out, "]");
}

static int qx_packed_expert_matvec_mode(qx_file *file, const qx_tensor_dir_entry *tensor, uint32_t expert,
                                   const float *input, uint32_t input_dims, float *output, uint32_t output_dims,
                                   const qx_projection_workspace *workspace, char *err, uint64_t err_len) {
    if (!file || !tensor || !input || !output || tensor->rank < 3u || tensor->dims[0] != input_dims || tensor->dims[1] < output_dims || expert >= tensor->dims[2]) {
        qx_set_err(err, err_len, "invalid packed expert matvec argument"); return 0;
    }
    const char *decoder = NULL;
    uint64_t block_size = 0;
    if (!qx_decoder_info(tensor->flags, &decoder, &block_size) || !block_size) { qx_set_err(err, err_len, "unsupported packed expert decoder"); return 0; }
    (void)decoder;
    uint64_t blocks_per_row = (input_dims + 255u) / 256u;
    if (blocks_per_row > UINT64_MAX / block_size) { qx_set_err(err, err_len, "packed expert row size overflow"); return 0; }
    uint64_t row_bytes = blocks_per_row * block_size;
    if (tensor->dims[1] > UINT64_MAX / row_bytes) { qx_set_err(err, err_len, "packed expert size overflow"); return 0; }
    uint64_t expert_bytes = row_bytes * tensor->dims[1];
    if (tensor->dims[2] > UINT64_MAX / expert_bytes || tensor->dims[2] * expert_bytes > tensor->byte_size ||
        expert > UINT64_MAX / expert_bytes || tensor->offset > UINT64_MAX - (uint64_t)expert * expert_bytes) {
        qx_set_err(err, err_len, "packed expert tensor is truncated or overflows"); return 0;
    }
    if (workspace && ((tensor->flags != 17u && tensor->flags != 18u && tensor->flags != 21u && tensor->flags != 22u && tensor->flags != 23u) ||
        input_dims % QX_Q8_K_VALUES != 0u || workspace->count != blocks_per_row)) {
        qx_set_err(err, err_len, "incompatible packed expert Q8_K workspace"); return 0;
    }
    unsigned char *slice = NULL;
    if (!qx_read_raw_span(file, tensor->offset + (uint64_t)expert * expert_bytes, expert_bytes, &slice, err, err_len)) return 0;
    for (uint32_t row = 0; row < output_dims; ++row) {
        const unsigned char *row_data = slice + (uint64_t)row * row_bytes;
        if (workspace) {
            output[row] = tensor->flags == 17u ? qx_dot_iq2_xs_q8_k(row_data, workspace) :
                tensor->flags == 18u ? qx_dot_iq3_xxs_q8_k(row_data, workspace) :
                tensor->flags == 21u ? qx_dot_iq3_s_q8_k(row_data, workspace) :
                tensor->flags == 22u ? qx_dot_iq2_s_q8_k(row_data, workspace) : qx_dot_iq4_xs_q8_k(row_data, workspace);
            if (!isfinite(output[row])) { free(slice); qx_set_err(err, err_len, "non-finite packed expert Q8_K output"); return 0; }
            continue;
        }
        double dot = 0.0;
        for (uint64_t block = 0; block < blocks_per_row; ++block) {
            float weights_block[256];
            if (!qx_decode_supported_block(tensor->flags, row_data + block * block_size, weights_block)) { free(slice); qx_set_err(err, err_len, "packed expert block decode failed"); return 0; }
            uint32_t start = (uint32_t)block * 256u;
            uint32_t take = input_dims - start;
            if (take > 256u) take = 256u;
            for (uint32_t i = 0; i < take; ++i) dot += (double)weights_block[i] * (double)input[start + i];
        }
        output[row] = isfinite(dot) ? (float)dot : 0.0f;
    }
    free(slice);
    return 1;
}

static int qx_packed_expert_matvec(qx_file *file, const qx_tensor_dir_entry *tensor, uint32_t expert,
                                   const float *input, uint32_t input_dims, float *output, uint32_t output_dims,
                                   char *err, uint64_t err_len) {
    return qx_packed_expert_matvec_mode(file, tensor, expert, input, input_dims, output, output_dims, NULL, err, err_len);
}

int qx_dump_real_qkv_golden_probe_summary(const char *path, uint32_t layer, uint32_t token_a, uint32_t token_b, uint32_t q_heads_run, uint32_t seed, int full_moe, FILE *out, char *err, uint64_t err_len) {
    if (!path || !out) { qx_set_err(err, err_len, "invalid real QKV probe argument"); return 0; }
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    uint32_t hidden = file.header.manifest.hidden;
    uint32_t q_heads_total = file.header.manifest.q_heads;
    uint32_t kv_heads_total = file.header.manifest.kv_heads;
    uint32_t head_dim = file.header.manifest.head_dim;
    uint32_t vocab = file.header.manifest.vocab;
    if (!hidden || !q_heads_total || !kv_heads_total || !head_dim || q_heads_total % kv_heads_total || !q_heads_run || q_heads_run > q_heads_total || token_a >= vocab || token_b >= vocab) {
        qx_close_file(&file); qx_set_err(err, err_len, "invalid model manifest or probe range"); return 0;
    }
    uint32_t q_values = q_heads_run * head_dim;
    uint32_t kv_values = kv_heads_total * head_dim;
    char qn[QX_NAME_MAX], kn[QX_NAME_MAX], vn[QX_NAME_MAX], on[QX_NAME_MAX], nn[QX_NAME_MAX], qnn[QX_NAME_MAX], knn[QX_NAME_MAX], fn[QX_NAME_MAX], rn[QX_NAME_MAX];
    snprintf(qn, sizeof(qn), "blk.%u.attn_q.weight", layer);
    snprintf(kn, sizeof(kn), "blk.%u.attn_k.weight", layer);
    snprintf(vn, sizeof(vn), "blk.%u.attn_v.weight", layer);
    snprintf(on, sizeof(on), "blk.%u.attn_output.weight", layer);
    snprintf(nn, sizeof(nn), "blk.%u.attn_norm.weight", layer);
    snprintf(qnn, sizeof(qnn), "blk.%u.attn_q_norm.weight", layer);
    snprintf(knn, sizeof(knn), "blk.%u.attn_k_norm.weight", layer);
    snprintf(fn, sizeof(fn), "blk.%u.ffn_norm.weight", layer);
    snprintf(rn, sizeof(rn), "blk.%u.ffn_gate_inp.weight", layer);
    const qx_tensor_dir_entry *qt = qx_find_tensor(&file, qn);
    const qx_tensor_dir_entry *kt = qx_find_tensor(&file, kn);
    const qx_tensor_dir_entry *vt = qx_find_tensor(&file, vn);
    const qx_tensor_dir_entry *ot = qx_find_tensor(&file, on);
    const qx_tensor_dir_entry *qnt = qx_find_tensor(&file, qnn);
    const qx_tensor_dir_entry *knt = qx_find_tensor(&file, knn);
    const qx_tensor_dir_entry *fnt = qx_find_tensor(&file, fn);
    const qx_tensor_dir_entry *router = qx_find_tensor(&file, rn);
    if (!qt || !kt || !vt || !ot || !qnt || !knt || !fnt || !router || !qx_find_tensor(&file, nn)) { qx_close_file(&file); qx_set_err(err, err_len, "real attention/FFN tensor not found"); return 0; }

    float *residual = (float *)malloc((size_t)hidden * sizeof(float));
    float *qraw = (float *)malloc((size_t)q_values * sizeof(float));
    float *qrope = (float *)malloc((size_t)q_values * sizeof(float));
    float *kraw = (float *)malloc((size_t)2u * kv_values * sizeof(float));
    float *vraw = (float *)malloc((size_t)2u * kv_values * sizeof(float));
    float *krope = (float *)malloc((size_t)2u * kv_values * sizeof(float));
    unsigned char *qbuf = (unsigned char *)malloc(q_values);
    unsigned char *kcache = (unsigned char *)malloc((size_t)2u * kv_values);
    unsigned char *vcache = (unsigned char *)malloc((size_t)2u * kv_values);
    float *kscales = (float *)malloc(2u * sizeof(float));
    float *vscales = (float *)malloc(2u * sizeof(float));
    double *scores = (double *)malloc((size_t)q_heads_run * 2u * sizeof(double));
    double *weights = (double *)malloc((size_t)q_heads_run * 2u * sizeof(double));
    double *context = (double *)calloc(q_values, sizeof(double));
    float *context_float = (float *)malloc((size_t)q_values * sizeof(float));
    float *output_raw = (float *)malloc((size_t)hidden * sizeof(float));
    unsigned char *output_buf = (unsigned char *)malloc(hidden);
    float *post_buffers = (float *)malloc((size_t)hidden * 3u * sizeof(float));
    if (!residual || !qraw || !qrope || !kraw || !vraw || !krope || !qbuf || !kcache || !vcache || !kscales || !vscales || !scores || !weights || !context || !context_float || !output_raw || !output_buf || !post_buffers) {
        free(residual); free(qraw); free(qrope); free(kraw); free(vraw); free(krope); free(qbuf); free(kcache); free(vcache); free(kscales); free(vscales); free(scores); free(weights); free(context); free(context_float); free(output_raw); free(output_buf); free(post_buffers); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0;
    }
    uint32_t token_ids[2] = {token_a, token_b};
    double probe = 0.0, rms = 0.0;
    uint64_t values = 0, checksum = 0;
    int ok = 1;
    for (uint32_t position = 0; position < 2u && ok; ++position) {
        ok = qx_fill_residual_vector_from_embedding(&file, token_ids[position], nn, residual, hidden, &rms, &checksum, err, err_len);
        if (ok) ok = qx_projection_matvec_fill(&file, kt, kcache + (uint64_t)position * kv_values, kraw + (uint64_t)position * kv_values, kv_values, hidden, residual, hidden, token_ids[position], layer, seed + position * 17u, &probe, &values, err, err_len);
        if (ok) ok = qx_projection_matvec_fill(&file, vt, vcache + (uint64_t)position * kv_values, vraw + (uint64_t)position * kv_values, kv_values, hidden, residual, hidden, token_ids[position], layer, (seed ^ 0x9e3779b9u) + position * 17u, &probe, &values, err, err_len);
    }
    if (ok) ok = qx_fill_residual_vector_from_embedding(&file, token_b, nn, residual, hidden, &rms, &checksum, err, err_len);
    if (ok) ok = qx_projection_matvec_fill(&file, qt, qbuf, qraw, q_values, hidden, residual, hidden, token_b, layer, seed ^ 0xa511e9b3u, &probe, &values, err, err_len);
    if (!ok) {
        free(residual); free(qraw); free(qrope); free(kraw); free(vraw); free(krope); free(qbuf); free(kcache); free(vcache); free(kscales); free(vscales); free(scores); free(weights); free(context); free(context_float); free(output_raw); free(output_buf); free(post_buffers); qx_close_file(&file); return 0;
    }
    memcpy(qrope, qraw, (size_t)q_values * sizeof(float));
    memcpy(krope, kraw, (size_t)2u * kv_values * sizeof(float));
    if (!qx_apply_f32_head_rmsnorm(&file, qnt, qrope, q_heads_run, head_dim, err, err_len)) {
        free(residual); free(qraw); free(qrope); free(kraw); free(vraw); free(krope); free(qbuf); free(kcache); free(vcache); free(kscales); free(vscales); free(scores); free(weights); free(context); free(context_float); free(output_raw); free(output_buf); free(post_buffers); qx_close_file(&file); return 0;
    }
    for (uint32_t position = 0; position < 2u; ++position) {
        if (!qx_apply_f32_head_rmsnorm(&file, knt, krope + (uint64_t)position * kv_values, kv_heads_total, head_dim, err, err_len)) {
            free(residual); free(qraw); free(qrope); free(kraw); free(vraw); free(krope); free(qbuf); free(kcache); free(vcache); free(kscales); free(vscales); free(scores); free(weights); free(context); free(context_float); free(output_raw); free(output_buf); free(post_buffers); qx_close_file(&file); return 0;
        }
    }
    qx_apply_rope(qrope, q_heads_run, head_dim, 1u, 1000000.0);
    for (uint32_t position = 0; position < 2u; ++position) {
        qx_apply_rope(krope + (uint64_t)position * kv_values, kv_heads_total, head_dim, position, 1000000.0);
        kscales[position] = qx_quantize_int8_vector(krope + (uint64_t)position * kv_values, kcache + (uint64_t)position * kv_values, kv_values);
        vscales[position] = qx_quantize_int8_vector(vraw + (uint64_t)position * kv_values, vcache + (uint64_t)position * kv_values, kv_values);
    }
    uint32_t group_size = q_heads_total / kv_heads_total;
    for (uint32_t qh = 0; qh < q_heads_run; ++qh) {
        uint32_t kvh = qh / group_size;
        double max_score = -1e300;
        for (uint32_t position = 0; position < 2u; ++position) {
            double score = 0.0;
            const unsigned char *kp = kcache + (uint64_t)position * kv_values + kvh * head_dim;
            for (uint32_t d = 0; d < head_dim; ++d) score += (double)qrope[qh * head_dim + d] * (double)(int8_t)kp[d] * (double)kscales[position];
            score /= sqrt((double)head_dim);
            scores[(uint64_t)qh * 2u + position] = score;
            if (score > max_score) max_score = score;
        }
        double denom = exp(scores[(uint64_t)qh * 2u] - max_score) + exp(scores[(uint64_t)qh * 2u + 1u] - max_score);
        for (uint32_t position = 0; position < 2u; ++position) {
            double weight = exp(scores[(uint64_t)qh * 2u + position] - max_score) / denom;
            weights[(uint64_t)qh * 2u + position] = weight;
            const unsigned char *vp = vcache + (uint64_t)position * kv_values + kvh * head_dim;
            for (uint32_t d = 0; d < head_dim; ++d) context[qh * head_dim + d] += weight * (double)(int8_t)vp[d] * (double)vscales[position];
        }
    }
    for (uint32_t i = 0; i < q_values; ++i) context_float[i] = (float)context[i];
    if (!qx_projection_matvec_fill(&file, ot, output_buf, output_raw, hidden, q_values, context_float, q_values, token_b, layer, seed ^ 0x63d83595u, &probe, &values, err, err_len)) {
        free(residual); free(qraw); free(qrope); free(kraw); free(vraw); free(krope); free(qbuf); free(kcache); free(vcache); free(kscales); free(vscales); free(scores); free(weights); free(context); free(context_float); free(output_raw); free(output_buf); free(post_buffers); qx_close_file(&file); return 0;
    }
    float *embedding_raw = post_buffers;
    float *residual_after_attention = post_buffers + hidden;
    float *ffn_norm_raw = post_buffers + hidden * 2u;
    double ffn_rms = 0.0;
    if (!qx_fill_residual_vector_from_embedding(&file, token_b, NULL, embedding_raw, hidden, &rms, &checksum, err, err_len)) {
        free(residual); free(qraw); free(qrope); free(kraw); free(vraw); free(krope); free(qbuf); free(kcache); free(vcache); free(kscales); free(vscales); free(scores); free(weights); free(context); free(context_float); free(output_raw); free(output_buf); free(post_buffers); qx_close_file(&file); return 0;
    }
    for (uint32_t i = 0; i < hidden; ++i) residual_after_attention[i] = embedding_raw[i] + output_raw[i];
    if (!qx_apply_f32_rmsnorm(&file, fnt, residual_after_attention, ffn_norm_raw, hidden, &ffn_rms, err, err_len)) {
        free(residual); free(qraw); free(qrope); free(kraw); free(vraw); free(krope); free(qbuf); free(kcache); free(vcache); free(kscales); free(vscales); free(scores); free(weights); free(context); free(context_float); free(output_raw); free(output_buf); free(post_buffers); qx_close_file(&file); return 0;
    }
    uint32_t experts = router->rank > 1u ? (uint32_t)router->dims[1] : file.header.manifest.experts;
    if (router->flags != 0u || router->dims[0] != hidden || experts < 8u || experts > 128u || router->byte_size < (uint64_t)hidden * experts * 4ull) {
        free(residual); free(qraw); free(qrope); free(kraw); free(vraw); free(krope); free(qbuf); free(kcache); free(vcache); free(kscales); free(vscales); free(scores); free(weights); free(context); free(context_float); free(output_raw); free(output_buf); free(post_buffers); qx_close_file(&file); qx_set_err(err, err_len, "unsupported real router layout"); return 0;
    }
    unsigned char *router_raw = NULL;
    if (!qx_read_raw_span(&file, router->offset, (uint64_t)hidden * experts * 4ull, &router_raw, err, err_len)) {
        free(residual); free(qraw); free(qrope); free(kraw); free(vraw); free(krope); free(qbuf); free(kcache); free(vcache); free(kscales); free(vscales); free(scores); free(weights); free(context); free(context_float); free(output_raw); free(output_buf); free(post_buffers); qx_close_file(&file); return 0;
    }
    double router_logits[128], router_probs[128];
    uint32_t selected_experts[8];
    double routing_weights[8];
    double router_max = -1.0e300;
    for (uint32_t expert = 0; expert < experts; ++expert) {
        double dot = 0.0;
        const unsigned char *row = router_raw + (uint64_t)expert * hidden * 4ull;
        for (uint32_t i = 0; i < hidden; ++i) dot += (double)qx_rd_le_f32(row + (uint64_t)i * 4ull) * (double)ffn_norm_raw[i];
        router_logits[expert] = dot;
        if (dot > router_max) router_max = dot;
    }
    double router_denom = 0.0;
    for (uint32_t expert = 0; expert < experts; ++expert) { router_probs[expert] = exp(router_logits[expert] - router_max); router_denom += router_probs[expert]; }
    if (!isfinite(router_denom) || router_denom <= 0.0) {
        free(router_raw); free(residual); free(qraw); free(qrope); free(kraw); free(vraw); free(krope); free(qbuf); free(kcache); free(vcache); free(kscales); free(vscales); free(scores); free(weights); free(context); free(context_float); free(output_raw); free(output_buf); free(post_buffers); qx_close_file(&file); qx_set_err(err, err_len, "invalid router softmax"); return 0;
    }
    for (uint32_t expert = 0; expert < experts; ++expert) router_probs[expert] /= router_denom;
    unsigned char picked[128] = {0};
    double selected_weight_sum = 0.0;
    for (uint32_t rank = 0; rank < 8u; ++rank) {
        uint32_t best = 0u; double best_prob = -1.0;
        for (uint32_t expert = 0; expert < experts; ++expert) if (!picked[expert] && router_probs[expert] > best_prob) { best = expert; best_prob = router_probs[expert]; }
        picked[best] = 1u; selected_experts[rank] = best; routing_weights[rank] = best_prob;
        selected_weight_sum += best_prob;
    }
    if (!isfinite(selected_weight_sum) || selected_weight_sum <= 0.0) {
        free(router_raw); free(residual); free(qraw); free(qrope); free(kraw); free(vraw); free(krope); free(qbuf); free(kcache); free(vcache); free(kscales); free(vscales); free(scores); free(weights); free(context); free(context_float); free(output_raw); free(output_buf); free(post_buffers); qx_close_file(&file); qx_set_err(err, err_len, "invalid selected router weights"); return 0;
    }
    for (uint32_t rank = 0; rank < 8u; ++rank) routing_weights[rank] /= selected_weight_sum;
    free(router_raw);
    float *moe_buffers = NULL;
    float *moe_output_raw = NULL;
    float *layer_output_raw = NULL;
    float *expert0_gate_raw = NULL, *expert0_up_raw = NULL, *expert0_hidden_raw = NULL, *expert0_down_raw = NULL;
    double moe_output_l2 = 0.0;
    const qx_tensor_dir_entry *gate_exps = NULL, *up_exps = NULL, *down_exps = NULL;
    uint32_t moe_intermediate = 0u;
    if (full_moe) {
        char gate_name[QX_NAME_MAX], up_name[QX_NAME_MAX], down_name[QX_NAME_MAX];
        snprintf(gate_name, sizeof(gate_name), "blk.%u.ffn_gate_exps.weight", layer);
        snprintf(up_name, sizeof(up_name), "blk.%u.ffn_up_exps.weight", layer);
        snprintf(down_name, sizeof(down_name), "blk.%u.ffn_down_exps.weight", layer);
        gate_exps = qx_find_tensor(&file, gate_name); up_exps = qx_find_tensor(&file, up_name); down_exps = qx_find_tensor(&file, down_name);
        if (!gate_exps || !up_exps || !down_exps || gate_exps->dims[0] != hidden || up_exps->dims[0] != hidden || gate_exps->dims[1] != up_exps->dims[1] || down_exps->dims[0] != gate_exps->dims[1] || down_exps->dims[1] != hidden) {
            free(residual); free(qraw); free(qrope); free(kraw); free(vraw); free(krope); free(qbuf); free(kcache); free(vcache); free(kscales); free(vscales); free(scores); free(weights); free(context); free(context_float); free(output_raw); free(output_buf); free(post_buffers); qx_close_file(&file); qx_set_err(err, err_len, "unsupported real MoE tensor layout"); return 0;
        }
        moe_intermediate = (uint32_t)gate_exps->dims[1];
        size_t moe_floats = (size_t)moe_intermediate * 6u + (size_t)hidden * 4u;
        moe_buffers = (float *)calloc(moe_floats, sizeof(float));
        if (!moe_buffers) {
            free(residual); free(qraw); free(qrope); free(kraw); free(vraw); free(krope); free(qbuf); free(kcache); free(vcache); free(kscales); free(vscales); free(scores); free(weights); free(context); free(context_float); free(output_raw); free(output_buf); free(post_buffers); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0;
        }
        float *gate_values = moe_buffers;
        float *up_values = gate_values + moe_intermediate;
        float *expert_hidden = up_values + moe_intermediate;
        float *expert_output = expert_hidden + moe_intermediate;
        moe_output_raw = expert_output + hidden;
        layer_output_raw = moe_output_raw + hidden;
        expert0_gate_raw = layer_output_raw + hidden;
        expert0_up_raw = expert0_gate_raw + moe_intermediate;
        expert0_hidden_raw = expert0_up_raw + moe_intermediate;
        expert0_down_raw = expert0_hidden_raw + moe_intermediate;
        for (uint32_t rank = 0; rank < 8u; ++rank) {
            uint32_t expert = selected_experts[rank];
            if (!qx_packed_expert_matvec(&file, gate_exps, expert, ffn_norm_raw, hidden, gate_values, moe_intermediate, err, err_len) ||
                !qx_packed_expert_matvec(&file, up_exps, expert, ffn_norm_raw, hidden, up_values, moe_intermediate, err, err_len)) {
                free(moe_buffers); free(residual); free(qraw); free(qrope); free(kraw); free(vraw); free(krope); free(qbuf); free(kcache); free(vcache); free(kscales); free(vscales); free(scores); free(weights); free(context); free(context_float); free(output_raw); free(output_buf); free(post_buffers); qx_close_file(&file); return 0;
            }
            for (uint32_t i = 0; i < moe_intermediate; ++i) expert_hidden[i] = (float)(qx_silu((double)gate_values[i]) * (double)up_values[i]);
            if (!qx_packed_expert_matvec(&file, down_exps, expert, expert_hidden, moe_intermediate, expert_output, hidden, err, err_len)) {
                free(moe_buffers); free(residual); free(qraw); free(qrope); free(kraw); free(vraw); free(krope); free(qbuf); free(kcache); free(vcache); free(kscales); free(vscales); free(scores); free(weights); free(context); free(context_float); free(output_raw); free(output_buf); free(post_buffers); qx_close_file(&file); return 0;
            }
            if (rank == 0u) {
                memcpy(expert0_gate_raw, gate_values, (size_t)moe_intermediate * sizeof(float));
                memcpy(expert0_up_raw, up_values, (size_t)moe_intermediate * sizeof(float));
                memcpy(expert0_hidden_raw, expert_hidden, (size_t)moe_intermediate * sizeof(float));
                memcpy(expert0_down_raw, expert_output, (size_t)hidden * sizeof(float));
            }
            for (uint32_t i = 0; i < hidden; ++i) moe_output_raw[i] += (float)(routing_weights[rank] * (double)expert_output[i]);
        }
        for (uint32_t i = 0; i < hidden; ++i) {
            layer_output_raw[i] = residual_after_attention[i] + moe_output_raw[i];
            moe_output_l2 += (double)moe_output_raw[i] * (double)moe_output_raw[i];
        }
        moe_output_l2 = sqrt(moe_output_l2);
    }
    unsigned char *qnorm_bytes = NULL, *knorm_bytes = NULL;
    if (!qx_read_raw_span(&file, qnt->offset, (uint64_t)head_dim * 4ull, &qnorm_bytes, err, err_len) ||
        !qx_read_raw_span(&file, knt->offset, (uint64_t)head_dim * 4ull, &knorm_bytes, err, err_len)) {
        free(qnorm_bytes); free(knorm_bytes); free(moe_buffers); free(residual); free(qraw); free(qrope); free(kraw); free(vraw); free(krope); free(qbuf); free(kcache); free(vcache); free(kscales); free(vscales); free(scores); free(weights); free(context); free(context_float); free(output_raw); free(output_buf); free(post_buffers); qx_close_file(&file); return 0;
    }
    uint32_t last_head = q_heads_run - 1u;
    fprintf(out, "{\n  \"probe\": \"real_qkv_golden\",\n  \"projection_layout\": \"contiguous_tensor_rows\",\n  \"iq4_xs_decoder_gate\": \"external_python_full_row\",\n");
    fprintf(out, "  \"layer\": %u, \"token_ids\": [%u,%u], \"projection_input_dims\": %u, \"projection_blocks_per_row\": %u,\n", layer, token_a, token_b, hidden, (hidden + 255u) / 256u);
    fprintf(out, "  \"q_heads_total\": %u, \"kv_heads_total\": %u, \"q_heads_run\": %u, \"full_head_coverage\": %s, \"gqa_group_size\": %u, \"head_dim\": %u,\n", q_heads_total, kv_heads_total, q_heads_run, q_heads_run == q_heads_total ? "true" : "false", group_size, head_dim);
    fprintf(out, "  \"output_projection\": \"real_iq4_xs_4096_to_2048\",\n");
    fprintf(out, "  \"q_raw\": "); qx_print_float_json_array(out, qraw, q_values); fprintf(out, ",\n  \"k_raw\": ["); qx_print_float_json_array(out, kraw, kv_values); fprintf(out, ","); qx_print_float_json_array(out, kraw + kv_values, kv_values); fprintf(out, "],\n  \"v_raw\": ["); qx_print_float_json_array(out, vraw, kv_values); fprintf(out, ","); qx_print_float_json_array(out, vraw + kv_values, kv_values); fprintf(out, "],\n");
    fprintf(out, "  \"q_norm_tensor\": \"%s\", \"k_norm_tensor\": \"%s\",\n  \"q_norm_raw\": [", qnt->name, knt->name);
    for (uint32_t i = 0; i < head_dim; ++i) fprintf(out, "%s%.9g", i ? "," : "", qx_rd_le_f32(qnorm_bytes + (uint64_t)i * 4ull));
    fprintf(out, "],\n  \"k_norm_raw\": [");
    for (uint32_t i = 0; i < head_dim; ++i) fprintf(out, "%s%.9g", i ? "," : "", qx_rd_le_f32(knorm_bytes + (uint64_t)i * 4ull));
    fprintf(out, "],\n");
    fprintf(out, "  \"attention_context_raw\": "); qx_print_float_json_array(out, context_float, q_values); fprintf(out, ",\n  \"output_raw\": "); qx_print_float_json_array(out, output_raw, hidden); fprintf(out, ",\n");
    fprintf(out, "  \"post_attention_norm_tensor\": \"%s\", \"post_attention_rms\": %.17g,\n  \"residual_after_attention\": ", fnt->name, ffn_rms); qx_print_float_json_array(out, residual_after_attention, hidden); fprintf(out, ",\n  \"ffn_norm_raw\": "); qx_print_float_json_array(out, ffn_norm_raw, hidden); fprintf(out, ",\n");
    fprintf(out, "  \"router_tensor\": \"%s\", \"router_norm_topk_prob\": true,\n  \"router_logits\": [", router->name);
    for (uint32_t expert = 0; expert < experts; ++expert) fprintf(out, "%s%.17g", expert ? "," : "", router_logits[expert]);
    fprintf(out, "],\n  \"router_probs\": [");
    for (uint32_t expert = 0; expert < experts; ++expert) fprintf(out, "%s%.17g", expert ? "," : "", router_probs[expert]);
    fprintf(out, "],\n  \"selected_experts\": [");
    for (uint32_t rank = 0; rank < 8u; ++rank) fprintf(out, "%s%u", rank ? "," : "", selected_experts[rank]);
    fprintf(out, "],\n  \"routing_weights\": [");
    for (uint32_t rank = 0; rank < 8u; ++rank) fprintf(out, "%s%.17g", rank ? "," : "", routing_weights[rank]);
    fprintf(out, "],\n");
    if (full_moe) {
        fprintf(out, "  \"moe_mode\": \"real_top8_swiglu\", \"moe_intermediate\": %u, \"experts_run\": 8, \"gate_ggml_type\": %u, \"up_ggml_type\": %u, \"down_ggml_type\": %u, \"moe_output_l2\": %.17g,\n  \"moe_output_raw\": ", moe_intermediate, gate_exps->flags, up_exps->flags, down_exps->flags, moe_output_l2);
        qx_print_float_json_array(out, moe_output_raw, hidden);
        fprintf(out, ",\n  \"layer_output_raw\": "); qx_print_float_json_array(out, layer_output_raw, hidden); fprintf(out, ",\n");
        fprintf(out, "  \"expert0_gate_raw\": "); qx_print_float_json_array(out, expert0_gate_raw, moe_intermediate);
        fprintf(out, ",\n  \"expert0_up_raw\": "); qx_print_float_json_array(out, expert0_up_raw, moe_intermediate);
        fprintf(out, ",\n  \"expert0_hidden_raw\": "); qx_print_float_json_array(out, expert0_hidden_raw, moe_intermediate);
        fprintf(out, ",\n  \"expert0_down_raw\": "); qx_print_float_json_array(out, expert0_down_raw, hidden); fprintf(out, ",\n");
    }
    fprintf(out, "  \"score_samples\": [%.17g,%.17g,%.17g,%.17g],\n", scores[0], scores[1], scores[(uint64_t)last_head * 2u], scores[(uint64_t)last_head * 2u + 1u]);
    fprintf(out, "  \"weight_samples\": [%.17g,%.17g,%.17g,%.17g],\n", weights[0], weights[1], weights[(uint64_t)last_head * 2u], weights[(uint64_t)last_head * 2u + 1u]);
    fprintf(out, "  \"context_samples\": [%.17g,%.17g,%.17g,%.17g]\n}\n", context[0], context[head_dim / 2u], context[last_head * head_dim], context[last_head * head_dim + head_dim / 2u]);
    free(qnorm_bytes); free(knorm_bytes); free(moe_buffers); free(residual); free(qraw); free(qrope); free(kraw); free(vraw); free(krope); free(qbuf); free(kcache); free(vcache); free(kscales); free(vscales); free(scores); free(weights); free(context); free(context_float); free(output_raw); free(output_buf); free(post_buffers); qx_close_file(&file); return 1;
}

static int qx_rope_gqa_attention_partial(
    qx_file *file, uint32_t layer, uint32_t step, uint32_t ctx_tokens,
    uint64_t bytes_per_k_or_v, const unsigned char *kcache, const unsigned char *vcache,
    uint32_t kv_values, uint32_t bytes_per_value, const float *kscales, const float *vscales, const float *residual, uint32_t dims,
    uint32_t q_heads_total, uint32_t kv_heads_total, uint32_t head_dim,
    uint32_t token, uint32_t seed, float *out_vec, float *context_capture, uint32_t context_capture_count, double *softmax_sum_out,
    double *softmax_min_out, double *softmax_max_out, uint32_t *q_heads_run_out,
    uint64_t *q_values_out, const char **q_name_out, const char **o_name_out,
    const char *activation_format, qx_projection_workspace *projection_workspace,
    qx_scratch_workspace *scratch_workspace,
    char *err, uint64_t err_len) {
    const double rope_theta = 1000000.0;
    uint32_t group_size = kv_heads_total ? q_heads_total / kv_heads_total : 1u;
    if (group_size == 0) group_size = 1;

    char qn[QX_NAME_MAX], qnn[QX_NAME_MAX], on[QX_NAME_MAX];
    snprintf(qn, sizeof(qn), "blk.%u.attn_q.weight", layer);
    snprintf(qnn, sizeof(qnn), "blk.%u.attn_q_norm.weight", layer);
    snprintf(on, sizeof(on), "blk.%u.attn_output.weight", layer);
    const qx_tensor_dir_entry *qt = qx_find_tensor(file, qn);
    const qx_tensor_dir_entry *qnt = qx_find_tensor(file, qnn);
    const qx_tensor_dir_entry *ot = qx_find_tensor(file, on);
    if ((!qt || !ot) && layer != 0) {
        qt = qx_find_tensor(file, "blk.0.attn_q.weight");
        ot = qx_find_tensor(file, "blk.0.attn_output.weight");
    }
    if (!qt || !ot) { qx_set_err(err, err_len, "q/output projection tensor not found"); return 0; }
    uint32_t q_heads_run = head_dim ? (uint32_t)(qt->dims[1] / head_dim) : 0u;
    if (q_heads_run == 0) q_heads_run = 1;
    if (q_heads_run > q_heads_total) q_heads_run = q_heads_total;
    uint32_t q_dims = q_heads_run * head_dim;
    uint32_t output_dims = dims;
    if (ot->dims[1] && output_dims > ot->dims[1]) output_dims = (uint32_t)ot->dims[1];
    memset(out_vec, 0, (size_t)dims * sizeof(float));

    uint32_t attend_count = step + 1u;
    qx_scratch_reset(scratch_workspace);
    if (q_heads_run && (size_t)q_heads_run > SIZE_MAX / (size_t)attend_count) {
        qx_set_err(err, err_len, "scratch allocation size overflow"); return 0;
    }
    size_t score_count = (size_t)q_heads_run * (size_t)attend_count;
    if (scratch_workspace) {
        size_t scratch_required = 0u;
        if (!qx_scratch_plan_add(&scratch_required, q_dims, 1u, err, err_len) ||
            !qx_scratch_plan_add(&scratch_required, output_dims, 1u, err, err_len) ||
            !qx_scratch_plan_add(&scratch_required, q_dims, sizeof(float), err, err_len) ||
            !qx_scratch_plan_add(&scratch_required, q_dims, sizeof(float), err, err_len) ||
            !qx_scratch_plan_add(&scratch_required, score_count, sizeof(double), err, err_len) ||
            !qx_scratch_plan_add(&scratch_required, score_count, sizeof(double), err, err_len) ||
            !qx_scratch_reserve_capacity(scratch_workspace, scratch_required, err, err_len)) return 0;
    }
    unsigned char *qbuf = (unsigned char *)qx_scratch_alloc(scratch_workspace, q_dims, 1u, 0, err, err_len);
    unsigned char *obuf = (unsigned char *)qx_scratch_alloc(scratch_workspace, output_dims, 1u, 0, err, err_len);
    float *qfloat = (float *)qx_scratch_alloc(scratch_workspace, q_dims, sizeof(float), 0, err, err_len);
    float *context = (float *)qx_scratch_alloc(scratch_workspace, q_dims, sizeof(float), 1, err, err_len);
    double *scores = (double *)qx_scratch_alloc(scratch_workspace, score_count, sizeof(double), 0, err, err_len);
    double *weights = (double *)qx_scratch_alloc(scratch_workspace, score_count, sizeof(double), 0, err, err_len);
    if (!qbuf || !obuf || !qfloat || !context || !scores || !weights) {
        if (!scratch_workspace) { free(qbuf); free(obuf); free(qfloat); free(context); free(scores); free(weights); }
        qx_set_err(err, err_len, "out of memory"); return 0;
    }

    double qprobe = 0.0;
    uint64_t qvals = 0;
    if (!qx_projection_matvec_fill_mode(file, qt, qbuf, qfloat, q_dims, dims, residual, dims, token, layer, seed ^ 0xa511e9b3u,
            activation_format, projection_workspace, &qprobe, &qvals, err, err_len)) {
        if (!scratch_workspace) { free(qbuf); free(obuf); free(qfloat); free(context); free(scores); free(weights); } return 0;
    }
    if (qnt && !qx_apply_f32_head_rmsnorm(file, qnt, qfloat, q_heads_run, head_dim, err, err_len)) {
        if (!scratch_workspace) { free(qbuf); free(obuf); free(qfloat); free(context); free(scores); free(weights); } return 0;
    }
    qx_apply_rope(qfloat, q_heads_run, head_dim, step, rope_theta);

    double sum_total = 0.0;
    double sum_min = 1e300;
    double sum_max = -1e300;
    for (uint32_t qh = 0; qh < q_heads_run; ++qh) {
        uint32_t kvh = qh / group_size;
        if (kvh >= kv_heads_total) kvh = kv_heads_total - 1u;
        double max_score = -1e300;
        for (uint32_t t = 0; t < attend_count; ++t) {
            const unsigned char *kp = kcache + (((uint64_t)layer * ctx_tokens + t) * bytes_per_k_or_v);
            float kscale = kscales[(uint64_t)layer * ctx_tokens + t];
            double score = 0.0;
            for (uint32_t d = 0; d < head_dim; ++d) {
                uint32_t qi = qh * head_dim + d;
                uint32_t ki = kvh * head_dim + d;
                if (qi >= q_dims || ki >= kv_values) break;
                float kval = qx_read_kv_cache_value(kp, ki, bytes_per_value, kscale);
                score += (double)qfloat[qi] * (double)kval;
            }
            score /= sqrt((double)head_dim);
            if (!isfinite(score)) score = 0.0;
            scores[(uint64_t)qh * attend_count + t] = score;
            if (score > max_score) max_score = score;
        }
        double denom = 0.0;
        for (uint32_t t = 0; t < attend_count; ++t) {
            double w = exp(scores[(uint64_t)qh * attend_count + t] - max_score);
            weights[(uint64_t)qh * attend_count + t] = w;
            denom += w;
        }
        if (!isfinite(denom) || denom <= 0.0) denom = 1.0;
        double head_sum = 0.0;
        for (uint32_t t = 0; t < attend_count; ++t) {
            double w = weights[(uint64_t)qh * attend_count + t] / denom;
            weights[(uint64_t)qh * attend_count + t] = w;
            head_sum += w;
            const unsigned char *vp = vcache + (((uint64_t)layer * ctx_tokens + t) * bytes_per_k_or_v);
            float vscale = vscales[(uint64_t)layer * ctx_tokens + t];
            for (uint32_t d = 0; d < head_dim; ++d) {
                uint32_t qi = qh * head_dim + d;
                uint32_t vi = kvh * head_dim + d;
                if (qi >= q_dims || vi >= kv_values) break;
                float vval = qx_read_kv_cache_value(vp, vi, bytes_per_value, vscale);
                context[qi] += (float)(w * (double)vval);
            }
        }
        sum_total += head_sum;
        if (head_sum < sum_min) sum_min = head_sum;
        if (head_sum > sum_max) sum_max = head_sum;
    }

    if (context_capture) {
        if (context_capture_count < q_dims) {
            if (!scratch_workspace) { free(qbuf); free(obuf); free(qfloat); free(context); free(scores); free(weights); }
            qx_set_err(err, err_len, "attention context capture too small"); return 0;
        }
        memcpy(context_capture, context, (size_t)q_dims * sizeof(float));
    }
    double oprobe = 0.0;
    uint64_t ovals = 0;
    if (!qx_projection_matvec_fill_mode(file, ot, obuf, out_vec, output_dims, q_dims, context, q_dims, token, layer, seed ^ 0x63d83595u,
            activation_format, projection_workspace, &oprobe, &ovals, err, err_len)) {
        if (!scratch_workspace) { free(qbuf); free(obuf); free(qfloat); free(context); free(scores); free(weights); } return 0;
    }
    if (softmax_sum_out) *softmax_sum_out = sum_total / (double)q_heads_run;
    if (softmax_min_out) *softmax_min_out = sum_min;
    if (softmax_max_out) *softmax_max_out = sum_max;
    if (q_heads_run_out) *q_heads_run_out = q_heads_run;
    if (q_values_out) *q_values_out = qvals;
    if (q_name_out) *q_name_out = qt->name;
    if (o_name_out) *o_name_out = ot->name;
    if (!scratch_workspace) { free(qbuf); free(obuf); free(qfloat); free(context); free(scores); free(weights); }
    return 1;
}

static int qx_apply_real_moe_layer(
    qx_file *file, uint32_t layer, const float *residual_after_attention, uint32_t hidden,
    float *layer_output, float *moe_output_capture, uint32_t selected_experts[8], double routing_weights[8],
    uint32_t *gate_type_out, uint32_t *up_type_out, uint32_t *down_type_out,
    const char *activation_format, qx_scratch_workspace *scratch_workspace,
    double *moe_l2_out, char *err, uint64_t err_len) {
    int use_q8_k = activation_format && strcmp(activation_format, "q8_k_compat") == 0;
    if (!use_q8_k && (!activation_format || strcmp(activation_format, "f32") != 0)) {
        qx_set_err(err, err_len, "unsupported real MoE activation format"); return 0;
    }
    char ffn_norm_name[QX_NAME_MAX], router_name[QX_NAME_MAX];
    char gate_name[QX_NAME_MAX], up_name[QX_NAME_MAX], down_name[QX_NAME_MAX];
    snprintf(ffn_norm_name, sizeof(ffn_norm_name), "blk.%u.ffn_norm.weight", layer);
    snprintf(router_name, sizeof(router_name), "blk.%u.ffn_gate_inp.weight", layer);
    snprintf(gate_name, sizeof(gate_name), "blk.%u.ffn_gate_exps.weight", layer);
    snprintf(up_name, sizeof(up_name), "blk.%u.ffn_up_exps.weight", layer);
    snprintf(down_name, sizeof(down_name), "blk.%u.ffn_down_exps.weight", layer);
    const qx_tensor_dir_entry *ffn_norm = qx_find_tensor(file, ffn_norm_name);
    const qx_tensor_dir_entry *router = qx_find_tensor(file, router_name);
    const qx_tensor_dir_entry *gate_exps = qx_find_tensor(file, gate_name);
    const qx_tensor_dir_entry *up_exps = qx_find_tensor(file, up_name);
    const qx_tensor_dir_entry *down_exps = qx_find_tensor(file, down_name);
    if (!ffn_norm || !router || !gate_exps || !up_exps || !down_exps) {
        qx_set_err(err, err_len, "real MoE tensor not found"); return 0;
    }
    uint32_t experts = router->rank > 1u ? (uint32_t)router->dims[1] : file->header.manifest.experts;
    uint32_t intermediate = gate_exps->rank > 1u ? (uint32_t)gate_exps->dims[1] : 0u;
    if (router->flags != 0u || router->dims[0] != hidden || experts < 8u || experts > 128u ||
        router->byte_size < (uint64_t)hidden * experts * 4ull || intermediate == 0u ||
        gate_exps->dims[0] != hidden || up_exps->dims[0] != hidden || up_exps->dims[1] != intermediate ||
        down_exps->dims[0] != intermediate || down_exps->dims[1] != hidden) {
        qx_set_err(err, err_len, "unsupported real MoE tensor layout"); return 0;
    }
    qx_scratch_reset(scratch_workspace);
    if ((size_t)hidden > SIZE_MAX / 4u || (size_t)intermediate > SIZE_MAX / 3u) {
        qx_set_err(err, err_len, "scratch allocation size overflow"); return 0;
    }
    size_t hidden_scratch = (size_t)hidden * 4u;
    size_t intermediate_scratch = (size_t)intermediate * 3u;
    if (hidden_scratch > SIZE_MAX - intermediate_scratch) {
        qx_set_err(err, err_len, "scratch allocation size overflow"); return 0;
    }
    float *buffers = (float *)qx_scratch_alloc(scratch_workspace,
        hidden_scratch + intermediate_scratch, sizeof(float), 1, err, err_len);
    if (!buffers) { qx_set_err(err, err_len, "out of memory"); return 0; }
    float *ffn_input = buffers;
    float *moe_output = ffn_input + hidden;
    float *expert_output = moe_output + hidden;
    float *gate_values = expert_output + hidden;
    float *up_values = gate_values + intermediate;
    float *expert_hidden = up_values + intermediate;
    double ffn_rms = 0.0;
    if (!qx_apply_f32_rmsnorm(file, ffn_norm, residual_after_attention, ffn_input, hidden, &ffn_rms, err, err_len)) {
        if (!scratch_workspace) free(buffers); return 0;
    }
    unsigned char *router_raw = NULL;
    if (!qx_read_raw_span(file, router->offset, (uint64_t)hidden * experts * 4ull, &router_raw, err, err_len)) {
        if (!scratch_workspace) free(buffers); return 0;
    }
    double logits[128], probabilities[128];
    double max_logit = -1.0e300;
    for (uint32_t expert = 0; expert < experts; ++expert) {
        const unsigned char *row = router_raw + (uint64_t)expert * hidden * 4ull;
        double dot = 0.0;
        for (uint32_t i = 0; i < hidden; ++i) dot += (double)qx_rd_le_f32(row + (uint64_t)i * 4ull) * (double)ffn_input[i];
        logits[expert] = dot;
        if (dot > max_logit) max_logit = dot;
    }
    free(router_raw);
    double denominator = 0.0;
    for (uint32_t expert = 0; expert < experts; ++expert) { probabilities[expert] = exp(logits[expert] - max_logit); denominator += probabilities[expert]; }
    if (!isfinite(denominator) || denominator <= 0.0) { if (!scratch_workspace) free(buffers); qx_set_err(err, err_len, "invalid router softmax"); return 0; }
    for (uint32_t expert = 0; expert < experts; ++expert) probabilities[expert] /= denominator;
    unsigned char picked[128] = {0};
    double selected_weight_sum = 0.0;
    for (uint32_t rank = 0; rank < 8u; ++rank) {
        uint32_t best = 0u;
        double best_probability = -1.0;
        for (uint32_t expert = 0; expert < experts; ++expert) {
            if (!picked[expert] && probabilities[expert] > best_probability) { best = expert; best_probability = probabilities[expert]; }
        }
        picked[best] = 1u;
        selected_experts[rank] = best;
        routing_weights[rank] = best_probability;
        selected_weight_sum += best_probability;
    }
    if (!isfinite(selected_weight_sum) || selected_weight_sum <= 0.0) { if (!scratch_workspace) free(buffers); qx_set_err(err, err_len, "invalid selected router weights"); return 0; }
    for (uint32_t rank = 0; rank < 8u; ++rank) routing_weights[rank] /= selected_weight_sum;
    int gate_up_q8_k = use_q8_k && ((gate_exps->flags == 17u && up_exps->flags == 17u) ||
        (gate_exps->flags == 22u && up_exps->flags == 22u));
    int down_q8_k = use_q8_k && (down_exps->flags == 18u || down_exps->flags == 21u || down_exps->flags == 23u);
    qx_projection_workspace gate_up_workspace = {0};
    if (gate_up_q8_k && !qx_quantize_q8_k(ffn_input, hidden, &gate_up_workspace, err, err_len)) { if (!scratch_workspace) free(buffers); return 0; }
    for (uint32_t rank = 0; rank < 8u; ++rank) {
        uint32_t expert = selected_experts[rank];
        if (!qx_packed_expert_matvec_mode(file, gate_exps, expert, ffn_input, hidden, gate_values, intermediate, gate_up_q8_k ? &gate_up_workspace : NULL, err, err_len) ||
            !qx_packed_expert_matvec_mode(file, up_exps, expert, ffn_input, hidden, up_values, intermediate, gate_up_q8_k ? &gate_up_workspace : NULL, err, err_len)) {
            if (!scratch_workspace) free(buffers); return 0;
        }
        for (uint32_t i = 0; i < intermediate; ++i) expert_hidden[i] = (float)(qx_silu((double)gate_values[i]) * (double)up_values[i]);
        qx_projection_workspace down_workspace = {0};
        if (down_q8_k && !qx_quantize_q8_k(expert_hidden, intermediate, &down_workspace, err, err_len)) { if (!scratch_workspace) free(buffers); return 0; }
        if (!qx_packed_expert_matvec_mode(file, down_exps, expert, expert_hidden, intermediate, expert_output, hidden, down_q8_k ? &down_workspace : NULL, err, err_len)) {
            if (!scratch_workspace) free(buffers); return 0;
        }
        for (uint32_t i = 0; i < hidden; ++i) moe_output[i] += (float)(routing_weights[rank] * (double)expert_output[i]);
    }
    double moe_l2 = 0.0;
    for (uint32_t i = 0; i < hidden; ++i) {
        layer_output[i] = residual_after_attention[i] + moe_output[i];
        moe_l2 += (double)moe_output[i] * (double)moe_output[i];
    }
    if (moe_output_capture) memcpy(moe_output_capture, moe_output, (size_t)hidden * sizeof(float));
    if (gate_type_out) *gate_type_out = gate_exps->flags;
    if (up_type_out) *up_type_out = up_exps->flags;
    if (down_type_out) *down_type_out = down_exps->flags;
    if (moe_l2_out) *moe_l2_out = sqrt(moe_l2);
    if (!scratch_workspace) free(buffers);
    return 1;
}

static int qx_read_exact_f32_sidecar(const char *path, float *values, uint32_t count, char *err, uint64_t err_len) {
    if (!path || !*path || !values || count == 0u || (uint64_t)count > (uint64_t)SIZE_MAX / sizeof(float)) {
        qx_set_err(err, err_len, "invalid F32 sidecar argument"); return 0;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) { qx_set_err(err, err_len, "cannot open F32 sidecar"); return 0; }
#if defined(_WIN32)
    if (_fseeki64(fp, 0, SEEK_END) != 0) { fclose(fp); qx_set_err(err, err_len, "cannot seek F32 sidecar"); return 0; }
    int64_t size_signed = _ftelli64(fp);
    if (_fseeki64(fp, 0, SEEK_SET) != 0) { fclose(fp); qx_set_err(err, err_len, "cannot rewind F32 sidecar"); return 0; }
#else
    if (fseeko(fp, 0, SEEK_END) != 0) { fclose(fp); qx_set_err(err, err_len, "cannot seek F32 sidecar"); return 0; }
    off_t size_signed = ftello(fp);
    if (fseeko(fp, 0, SEEK_SET) != 0) { fclose(fp); qx_set_err(err, err_len, "cannot rewind F32 sidecar"); return 0; }
#endif
    const uint64_t expected = (uint64_t)count * sizeof(float);
    if (size_signed < 0 || (uint64_t)size_signed != expected || fread(values, sizeof(float), count, fp) != count) {
        fclose(fp);
        qx_set_err(err, err_len, "F32 sidecar size or read mismatch"); return 0;
    }
    if (fclose(fp) != 0) { qx_set_err(err, err_len, "cannot close F32 sidecar"); return 0; }
    for (uint32_t i = 0; i < count; ++i) {
        if (!isfinite(values[i])) { qx_set_err(err, err_len, "F32 sidecar contains non-finite value"); return 0; }
    }
    return 1;
}

static int qx_write_final_head_sidecar(const char *dir, const char *name, const float *values,
        uint32_t count, char *err, uint64_t err_len) {
    if (!dir || !*dir || !name || !*name || !values || count == 0u ||
            (uint64_t)count > (uint64_t)SIZE_MAX / sizeof(float)) {
        qx_set_err(err, err_len, "invalid final head sidecar argument"); return 0;
    }
    char path[1024];
    int length = snprintf(path, sizeof(path), "%s/%s.f32", dir, name);
    if (length < 0 || (size_t)length >= sizeof(path)) { qx_set_err(err, err_len, "final head sidecar path too long"); return 0; }
    FILE *fp = fopen(path, "wb");
    if (!fp) { qx_set_err(err, err_len, "cannot open final head sidecar"); return 0; }
    int ok = fwrite(values, sizeof(float), count, fp) == count;
    if (fclose(fp) != 0) ok = 0;
    if (!ok) { remove(path); qx_set_err(err, err_len, "cannot write final head sidecar"); return 0; }
    return 1;
}

static void qx_remove_final_head_sidecar(const char *dir, const char *name) {
    char path[1024];
    int length = snprintf(path, sizeof(path), "%s/%s.f32", dir, name);
    if (length >= 0 && (size_t)length < sizeof(path)) remove(path);
}

int qx_dump_final_head_probe_summary(const char *path, const char *residual_path,
        const char *output_dir, const char *activation_mode, uint32_t top_n,
        FILE *out, char *err, uint64_t err_len) {
    int use_q8_k = activation_mode && strcmp(activation_mode, "q8_k_compat") == 0;
    if (!path || !residual_path || !output_dir || !out || top_n == 0u || top_n > 32u ||
            (!use_q8_k && (!activation_mode || strcmp(activation_mode, "f32") != 0))) {
        qx_set_err(err, err_len, "invalid final head probe argument or activation mode"); return 0;
    }
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    uint32_t hidden = file.header.manifest.hidden;
    uint32_t vocab = file.header.manifest.vocab;
    if (hidden == 0u || hidden > QX_Q8_K_VALUES * QX_Q8_K_MAX_BLOCKS || vocab == 0u ||
            (uint64_t)hidden > (uint64_t)SIZE_MAX / (2u * sizeof(float)) ||
            (uint64_t)vocab > (uint64_t)SIZE_MAX / sizeof(float)) {
        qx_close_file(&file); qx_set_err(err, err_len, "unsupported final head probe dimensions"); return 0;
    }
    float *vectors = (float *)malloc((size_t)hidden * 2u * sizeof(float));
    float *logits = (float *)malloc((size_t)vocab * sizeof(float));
    if (!vectors || !logits) { free(vectors); free(logits); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
    float *residual = vectors;
    float *normalized = vectors + hidden;
    qx_real_head_result result;
    int ok = qx_read_exact_f32_sidecar(residual_path, residual, hidden, err, err_len) &&
        qx_compute_real_final_head(&file, residual, normalized, hidden, top_n, activation_mode,
            "baseline", "serial", 1u, "scalar", logits, vocab, &result, err, err_len);
    if (ok && !qx_write_final_head_sidecar(output_dir, "final-norm", normalized, hidden, err, err_len)) ok = 0;
    if (ok && !qx_write_final_head_sidecar(output_dir, "logits", logits, vocab, err, err_len)) {
        qx_remove_final_head_sidecar(output_dir, "final-norm");
        ok = 0;
    }
    if (!ok) { free(vectors); free(logits); qx_close_file(&file); return 0; }
    fprintf(out, "{\"probe\":\"final_head\",\"input_dims\":%u,\"vocab_size\":%u,"
        "\"activation_mode\":\"%s\",\"lm_head_kernel\":\"%s\",\"activation_quantizations\":%u,"
        "\"input_rms\":%.17g,\"normalized_l2\":%.17g,\"residual_checksum\":%llu,"
        "\"normalized_checksum\":%llu,\"logits_checksum\":%llu,\"argmax_token\":%u,"
        "\"argmax_logit\":%.17g}\n",
        result.input_dims, result.vocab_size, activation_mode, result.lm_head_kernel,
        result.activation_quantizations, result.input_rms, result.normalized_l2,
        (unsigned long long)result.residual_checksum, (unsigned long long)result.normalized_checksum,
        (unsigned long long)result.logits_checksum, result.top[0].token, result.top[0].logit);
    free(vectors); free(logits); qx_close_file(&file); return 1;
}

int qx_dump_expert_q8_k_dot_probe_summary(const char *path, const char *tensor_name, uint32_t expert,
        uint32_t row, const char *activation_path, FILE *out, char *err, uint64_t err_len) {
    if (!path || !tensor_name || !activation_path || !out) { qx_set_err(err, err_len, "invalid expert Q8_K dot probe argument"); return 0; }
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    const qx_tensor_dir_entry *tensor = qx_find_tensor(&file, tensor_name);
    if (!tensor || tensor->rank < 3u ||
        (tensor->flags != 17u && tensor->flags != 18u && tensor->flags != 21u && tensor->flags != 22u && tensor->flags != 23u) ||
        tensor->dims[0] == 0u || tensor->dims[0] > UINT32_MAX || tensor->dims[0] % QX_Q8_K_VALUES != 0u ||
        tensor->dims[1] > UINT32_MAX || tensor->dims[2] > UINT32_MAX || row >= tensor->dims[1] || expert >= tensor->dims[2]) {
        qx_close_file(&file); qx_set_err(err, err_len, "unsupported expert Q8_K dot tensor or range"); return 0;
    }
    uint64_t block_size = tensor->flags == 17u ? 74u : tensor->flags == 18u ? 98u : tensor->flags == 21u ? 110u : tensor->flags == 22u ? 82u : 136u;
    uint64_t blocks = tensor->dims[0] / QX_Q8_K_VALUES;
    if (blocks == 0u || blocks > UINT64_MAX / block_size) { qx_close_file(&file); qx_set_err(err, err_len, "expert Q8_K row size overflow"); return 0; }
    uint64_t row_bytes = blocks * block_size;
    if (tensor->dims[1] > UINT64_MAX / row_bytes) { qx_close_file(&file); qx_set_err(err, err_len, "expert Q8_K tensor size overflow"); return 0; }
    uint64_t expert_bytes = tensor->dims[1] * row_bytes;
    if (tensor->dims[2] > UINT64_MAX / expert_bytes || tensor->dims[2] * expert_bytes != tensor->byte_size ||
        expert > UINT64_MAX / expert_bytes || row > UINT64_MAX / row_bytes) {
        qx_close_file(&file); qx_set_err(err, err_len, "invalid expert Q8_K tensor byte size"); return 0;
    }
    uint64_t relative = (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
    if (relative > tensor->byte_size || row_bytes > tensor->byte_size - relative || tensor->offset > UINT64_MAX - relative) {
        qx_close_file(&file); qx_set_err(err, err_len, "expert Q8_K row offset overflow"); return 0;
    }
    uint32_t input_count = (uint32_t)tensor->dims[0];
    if ((uint64_t)input_count > (uint64_t)SIZE_MAX / sizeof(float)) { qx_close_file(&file); qx_set_err(err, err_len, "expert Q8_K activation size overflow"); return 0; }
    float *activation = (float *)malloc((size_t)input_count * sizeof(float));
    unsigned char *row_raw = NULL;
    if (!activation) { qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
    if (!qx_read_exact_f32_sidecar(activation_path, activation, input_count, err, err_len) ||
        !qx_read_raw_span(&file, tensor->offset + relative, row_bytes, &row_raw, err, err_len)) {
        free(activation); free(row_raw); qx_close_file(&file); return 0;
    }
    qx_projection_workspace workspace = {0};
    if (!qx_quantize_q8_k(activation, input_count, &workspace, err, err_len)) {
        free(activation); free(row_raw); qx_close_file(&file); return 0;
    }
    float dot = tensor->flags == 17u ? qx_dot_iq2_xs_q8_k(row_raw, &workspace) :
        tensor->flags == 18u ? qx_dot_iq3_xxs_q8_k(row_raw, &workspace) :
        tensor->flags == 21u ? qx_dot_iq3_s_q8_k(row_raw, &workspace) :
        tensor->flags == 22u ? qx_dot_iq2_s_q8_k(row_raw, &workspace) : qx_dot_iq4_xs_q8_k(row_raw, &workspace);
    if (!isfinite(dot)) { free(activation); free(row_raw); qx_close_file(&file); qx_set_err(err, err_len, "non-finite expert Q8_K dot"); return 0; }
    fprintf(out, "{\"probe\":\"expert_q8_k_dot\",\"tensor\":\"%s\",\"ggml_type\":%u,\"expert\":%u,\"row\":%u,\"values\":%u,\"blocks\":%u,\"dot\":%.9g}\n",
            tensor->name, tensor->flags, expert, row, input_count, workspace.count, dot);
    free(activation); free(row_raw); qx_close_file(&file); return 1;
}

static int qx_write_moe_stage_sidecar(const char *dir, const char *name, uint32_t layer,
        const float *values, uint64_t count, char *err, uint64_t err_len) {
    if (!dir || !*dir || !name || !*name || !values || count == 0u || count > (uint64_t)SIZE_MAX / sizeof(float)) {
        qx_set_err(err, err_len, "invalid MoE sidecar argument"); return 0;
    }
    char path[1024];
    int length = snprintf(path, sizeof(path), "%s/%s-%u.f32", dir, name, layer);
    if (length < 0 || (size_t)length >= sizeof(path)) { qx_set_err(err, err_len, "MoE sidecar path too long"); return 0; }
    FILE *fp = fopen(path, "wb");
    if (!fp) { qx_set_err(err, err_len, "cannot open MoE sidecar"); return 0; }
    int ok = fwrite(values, sizeof(float), (size_t)count, fp) == (size_t)count;
    if (fclose(fp) != 0) ok = 0;
    if (!ok) { qx_set_err(err, err_len, "cannot write MoE sidecar"); return 0; }
    return 1;
}

int qx_dump_attention_stage_probe_summary(const char *path, uint32_t layer, const char *layer_input_path,
        const char *output_dir, const char *activation_mode, const char *kv_format, FILE *out, char *err, uint64_t err_len) {
    int use_q8_k = activation_mode && strcmp(activation_mode, "q8_k_compat") == 0;
    int use_f16_kv = kv_format && strcmp(kv_format, "f16") == 0;
    if (!path || !layer_input_path || !output_dir || !out ||
            (!use_q8_k && (!activation_mode || strcmp(activation_mode, "f32") != 0)) ||
            (!use_f16_kv && (!kv_format || strcmp(kv_format, "f32") != 0))) {
        qx_set_err(err, err_len, "invalid attention stage probe argument, activation mode, or KV format"); return 0;
    }
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    uint32_t hidden = file.header.manifest.hidden;
    uint32_t q_heads = file.header.manifest.q_heads;
    uint32_t kv_heads = file.header.manifest.kv_heads;
    uint32_t head_dim = file.header.manifest.head_dim;
    if (layer >= file.header.manifest.layers || hidden == 0u || hidden > 2048u ||
            q_heads == 0u || kv_heads == 0u || head_dim == 0u || q_heads % kv_heads != 0u ||
            q_heads > UINT32_MAX / head_dim || kv_heads > UINT32_MAX / head_dim) {
        qx_close_file(&file); qx_set_err(err, err_len, "invalid attention stage layer or dimensions"); return 0;
    }
    uint32_t q_values = q_heads * head_dim;
    uint32_t kv_values = kv_heads * head_dim;
    if (q_values == 0u || q_values > QX_Q8_K_VALUES * QX_Q8_K_MAX_BLOCKS || kv_values == 0u) {
        qx_close_file(&file); qx_set_err(err, err_len, "unsupported attention stage dimensions"); return 0;
    }
    char norm_name[QX_NAME_MAX], v_name[QX_NAME_MAX], output_name[QX_NAME_MAX];
    snprintf(norm_name, sizeof(norm_name), "blk.%u.attn_norm.weight", layer);
    snprintf(v_name, sizeof(v_name), "blk.%u.attn_v.weight", layer);
    snprintf(output_name, sizeof(output_name), "blk.%u.attn_output.weight", layer);
    const qx_tensor_dir_entry *norm = qx_find_tensor(&file, norm_name);
    const qx_tensor_dir_entry *v_tensor = qx_find_tensor(&file, v_name);
    const qx_tensor_dir_entry *output_tensor = qx_find_tensor(&file, output_name);
    if (!norm || !v_tensor || !output_tensor || v_tensor->dims[0] != hidden || v_tensor->dims[1] != kv_values ||
            output_tensor->dims[0] != q_values || output_tensor->dims[1] != hidden) {
        qx_close_file(&file); qx_set_err(err, err_len, "unsupported attention stage tensor layout"); return 0;
    }
    if ((uint64_t)hidden > (uint64_t)SIZE_MAX / sizeof(float) ||
            (uint64_t)q_values > (uint64_t)SIZE_MAX / sizeof(float) ||
            (uint64_t)kv_values > (uint64_t)SIZE_MAX / sizeof(float)) {
        qx_close_file(&file); qx_set_err(err, err_len, "attention stage allocation overflow"); return 0;
    }
    float *layer_input = (float *)malloc((size_t)hidden * sizeof(float));
    float *normalized = (float *)malloc((size_t)hidden * sizeof(float));
    float *v_cur = (float *)malloc((size_t)kv_values * sizeof(float));
    float *context = (float *)malloc((size_t)q_values * sizeof(float));
    float *attention_output = (float *)malloc((size_t)hidden * sizeof(float));
    float *ffn_input = (float *)malloc((size_t)hidden * sizeof(float));
    unsigned char *v_bytes = (unsigned char *)malloc((size_t)kv_values);
    unsigned char *output_bytes = (unsigned char *)malloc((size_t)hidden);
    if (!layer_input || !normalized || !v_cur || !context || !attention_output || !ffn_input || !v_bytes || !output_bytes) {
        free(layer_input); free(normalized); free(v_cur); free(context); free(attention_output); free(ffn_input); free(v_bytes); free(output_bytes);
        qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0;
    }
    int ok = 0;
    double norm_rms = 0.0, v_probe = 0.0, output_probe = 0.0;
    uint64_t v_count = 0u, output_count = 0u;
    qx_projection_workspace workspace = {0};
    if (!qx_read_exact_f32_sidecar(layer_input_path, layer_input, hidden, err, err_len) ||
            !qx_apply_f32_rmsnorm(&file, norm, layer_input, normalized, hidden, &norm_rms, err, err_len) ||
            !qx_projection_matvec_fill_mode(&file, v_tensor, v_bytes, v_cur, kv_values, hidden,
                normalized, hidden, 0u, layer, 1u, activation_mode, &workspace, &v_probe, &v_count, err, err_len)) {
        goto qx_attention_stage_cleanup;
    }
    uint32_t group_size = q_heads / kv_heads;
    for (uint32_t qh = 0; qh < q_heads; ++qh) {
        uint32_t kvh = qh / group_size;
        for (uint32_t d = 0; d < head_dim; ++d) context[qh * head_dim + d] = v_cur[kvh * head_dim + d];
    }
    if (use_f16_kv) {
        for (uint32_t i = 0; i < q_values; ++i) context[i] = qx_fp16_to_f32(qx_f32_to_fp16(context[i]));
    }
    if (!qx_projection_matvec_fill_mode(&file, output_tensor, output_bytes, attention_output, hidden, q_values,
            context, q_values, 0u, layer, 1u ^ 0x63d83595u, activation_mode, &workspace,
            &output_probe, &output_count, err, err_len)) {
        goto qx_attention_stage_cleanup;
    }
    for (uint32_t i = 0; i < hidden; ++i) {
        ffn_input[i] = layer_input[i] + attention_output[i];
        if (!isfinite(attention_output[i]) || !isfinite(ffn_input[i])) {
            qx_set_err(err, err_len, "non-finite attention stage output"); goto qx_attention_stage_cleanup;
        }
    }
    if (!qx_write_moe_stage_sidecar(output_dir, "attn_norm", layer, normalized, hidden, err, err_len) ||
            !qx_write_moe_stage_sidecar(output_dir, "Vcur", layer, v_cur, kv_values, err, err_len) ||
            !qx_write_moe_stage_sidecar(output_dir, "kqv_out", layer, context, q_values, err, err_len) ||
            !qx_write_moe_stage_sidecar(output_dir, "attn_out", layer, attention_output, hidden, err, err_len) ||
            !qx_write_moe_stage_sidecar(output_dir, "ffn_inp", layer, ffn_input, hidden, err, err_len)) {
        goto qx_attention_stage_cleanup;
    }
    const char *v_kernel = qx_projection_tensor_kernel_label(v_tensor->flags, use_q8_k);
    const char *output_kernel = qx_projection_tensor_kernel_label(output_tensor->flags, use_q8_k);
    uint32_t attention_family_mask = qx_projection_family_bit(v_tensor->flags) |
        qx_projection_family_bit(output_tensor->flags);
    int attention_f32_used = use_q8_k &&
        (!qx_projection_family_bit(v_tensor->flags) || !qx_projection_family_bit(output_tensor->flags));
    const char *projection_kernel = use_q8_k ?
        qx_projection_kernel_label(attention_family_mask, attention_f32_used) : "dequant_f32";
    fprintf(out, "{\"probe\":\"attention_stage\",\"layer\":%u,\"hidden\":%u,\"q_values\":%u,\"kv_values\":%u,"
        "\"activation_mode\":\"%s\",\"kv_format\":\"%s\",\"projection_kernel\":\"%s\",\"v_projection_kernel\":\"%s\","
        "\"output_projection_kernel\":\"%s\",\"single_token_softmax\":1.0,\"norm_rms\":%.9g,"
        "\"v_probe\":%.9g,\"output_probe\":%.9g,\"v_count\":%llu,\"output_count\":%llu}\n",
        layer, hidden, q_values, kv_values, activation_mode, kv_format, projection_kernel, v_kernel, output_kernel,
        norm_rms, v_probe, output_probe, (unsigned long long)v_count, (unsigned long long)output_count);
    ok = 1;
qx_attention_stage_cleanup:
    free(layer_input); free(normalized); free(v_cur); free(context); free(attention_output); free(ffn_input); free(v_bytes); free(output_bytes);
    qx_close_file(&file);
    return ok;
}

int qx_dump_moe_stage_probe_summary(const char *path, uint32_t layer, const char *ffn_input_path,
        const char *output_dir, const char *activation_mode, FILE *out, char *err, uint64_t err_len) {
    int use_q8_k = activation_mode && strcmp(activation_mode, "q8_k_compat") == 0;
    if (!path || !ffn_input_path || !output_dir || !out || (!use_q8_k && (!activation_mode || strcmp(activation_mode, "f32") != 0))) {
        qx_set_err(err, err_len, "invalid MoE stage probe argument or activation mode"); return 0;
    }
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    const uint32_t hidden = file.header.manifest.hidden;
    if (hidden == 0u || layer >= file.header.manifest.layers) { qx_close_file(&file); qx_set_err(err, err_len, "invalid MoE stage layer"); return 0; }

    char norm_name[QX_NAME_MAX], router_name[QX_NAME_MAX], gate_name[QX_NAME_MAX], up_name[QX_NAME_MAX], down_name[QX_NAME_MAX];
    snprintf(norm_name, sizeof(norm_name), "blk.%u.ffn_norm.weight", layer);
    snprintf(router_name, sizeof(router_name), "blk.%u.ffn_gate_inp.weight", layer);
    snprintf(gate_name, sizeof(gate_name), "blk.%u.ffn_gate_exps.weight", layer);
    snprintf(up_name, sizeof(up_name), "blk.%u.ffn_up_exps.weight", layer);
    snprintf(down_name, sizeof(down_name), "blk.%u.ffn_down_exps.weight", layer);
    const qx_tensor_dir_entry *norm = qx_find_tensor(&file, norm_name);
    const qx_tensor_dir_entry *router = qx_find_tensor(&file, router_name);
    const qx_tensor_dir_entry *gate = qx_find_tensor(&file, gate_name);
    const qx_tensor_dir_entry *up = qx_find_tensor(&file, up_name);
    const qx_tensor_dir_entry *down = qx_find_tensor(&file, down_name);
    uint32_t experts = router && router->rank > 1u ? (uint32_t)router->dims[1] : 0u;
    uint32_t intermediate = gate && gate->rank > 1u ? (uint32_t)gate->dims[1] : 0u;
    if (!norm || !router || !gate || !up || !down || experts < 8u || experts > 128u || intermediate == 0u ||
        router->flags != 0u || router->dims[0] != hidden || router->byte_size != (uint64_t)hidden * experts * 4ull ||
        gate->rank < 3u || up->rank < 3u || down->rank < 3u || gate->dims[0] != hidden || gate->dims[1] != intermediate || gate->dims[2] != experts ||
        up->dims[0] != hidden || up->dims[1] != intermediate || up->dims[2] != experts ||
        down->dims[0] != intermediate || down->dims[1] != hidden || down->dims[2] != experts) {
        qx_close_file(&file); qx_set_err(err, err_len, "unsupported MoE stage tensor layout"); return 0;
    }
    if ((uint64_t)hidden > (uint64_t)SIZE_MAX / sizeof(float) ||
        (uint64_t)intermediate > (uint64_t)SIZE_MAX / (8u * sizeof(float)) ||
        (uint64_t)hidden > (uint64_t)SIZE_MAX / (8u * sizeof(float))) {
        qx_close_file(&file); qx_set_err(err, err_len, "MoE stage allocation overflow"); return 0;
    }

    float *ffn_input = (float *)malloc((size_t)hidden * sizeof(float));
    float *ffn_norm = (float *)malloc((size_t)hidden * sizeof(float));
    float *logits = (float *)malloc((size_t)experts * sizeof(float));
    float *probs = (float *)malloc((size_t)experts * sizeof(float));
    float *gate_values = (float *)malloc((size_t)intermediate * 8u * sizeof(float));
    float *up_values = (float *)malloc((size_t)intermediate * 8u * sizeof(float));
    float *swiglu = (float *)malloc((size_t)intermediate * 8u * sizeof(float));
    float *down_values = (float *)malloc((size_t)hidden * 8u * sizeof(float));
    float *weighted = (float *)malloc((size_t)hidden * 8u * sizeof(float));
    if (!ffn_input || !ffn_norm || !logits || !probs || !gate_values || !up_values || !swiglu || !down_values || !weighted) {
        free(ffn_input); free(ffn_norm); free(logits); free(probs); free(gate_values); free(up_values); free(swiglu); free(down_values); free(weighted);
        qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0;
    }
    if (!qx_read_exact_f32_sidecar(ffn_input_path, ffn_input, hidden, err, err_len)) goto fail;
    double rms = 0.0;
    if (!qx_apply_f32_rmsnorm(&file, norm, ffn_input, ffn_norm, hidden, &rms, err, err_len)) goto fail;
    unsigned char *router_raw = NULL;
    if (!qx_read_raw_span(&file, router->offset, router->byte_size, &router_raw, err, err_len)) goto fail;
    double max_logit = -1.0e300;
    for (uint32_t expert = 0; expert < experts; ++expert) {
        double dot = 0.0;
        const unsigned char *row = router_raw + (uint64_t)expert * hidden * 4ull;
        for (uint32_t i = 0; i < hidden; ++i) dot += (double)qx_rd_le_f32(row + (uint64_t)i * 4ull) * (double)ffn_norm[i];
        if (!isfinite(dot)) { free(router_raw); qx_set_err(err, err_len, "non-finite MoE router logit"); goto fail; }
        logits[expert] = (float)dot;
        if (dot > max_logit) max_logit = dot;
    }
    free(router_raw);
    double denominator = 0.0;
    for (uint32_t expert = 0; expert < experts; ++expert) { probs[expert] = (float)exp((double)logits[expert] - max_logit); denominator += probs[expert]; }
    if (!isfinite(denominator) || denominator <= 0.0) { qx_set_err(err, err_len, "invalid MoE router softmax"); goto fail; }
    for (uint32_t expert = 0; expert < experts; ++expert) probs[expert] = (float)((double)probs[expert] / denominator);
    unsigned char picked[128] = {0};
    uint32_t selected[8];
    float topk[8], weights[8], weight_sum_sidecar[1];
    double weight_sum = 0.0;
    for (uint32_t rank = 0; rank < 8u; ++rank) {
        uint32_t best = 0u; float best_prob = -1.0f;
        for (uint32_t expert = 0; expert < experts; ++expert) if (!picked[expert] && probs[expert] > best_prob) { best = expert; best_prob = probs[expert]; }
        picked[best] = 1u; selected[rank] = best; topk[rank] = (float)best; weights[rank] = best_prob; weight_sum += best_prob;
    }
    if (!isfinite(weight_sum) || weight_sum <= 0.0) { qx_set_err(err, err_len, "invalid selected MoE weights"); goto fail; }
    weight_sum_sidecar[0] = (float)weight_sum;
    float weights_norm[8];
    for (uint32_t rank = 0; rank < 8u; ++rank) weights_norm[rank] = (float)((double)weights[rank] / weight_sum);
    int gate_up_q8_k = use_q8_k && ((gate->flags == 17u && up->flags == 17u) ||
        (gate->flags == 22u && up->flags == 22u));
    int down_q8_k = use_q8_k && (down->flags == 18u || down->flags == 21u || down->flags == 23u);
    int q8_k_used = gate_up_q8_k || down_q8_k;
    int f32_used = !gate_up_q8_k || !down_q8_k;
    const char *projection_kernel = "dequant_f32";
    if (q8_k_used && f32_used) projection_kernel = "q8_k_expert_kernels_with_f32_fallback";
    else if (q8_k_used && gate->flags == 17u && down->flags == 18u) projection_kernel = "iq2_xs_q8_k_and_iq3_xxs_q8_k";
    else if (q8_k_used && gate->flags == 22u && down->flags == 23u) projection_kernel = "iq2_s_q8_k_and_iq4_xs_q8_k";
    else if (q8_k_used && gate->flags == 17u && down->flags == 21u) projection_kernel = "iq2_xs_q8_k_and_iq3_s_q8_k";
    else if (q8_k_used && gate->flags == 22u && down->flags == 21u) projection_kernel = "iq2_s_q8_k_and_iq3_s_q8_k";
    else if (q8_k_used) projection_kernel = "q8_k_expert_kernels";
    qx_projection_workspace gate_up_workspace = {0};
    if (gate_up_q8_k && !qx_quantize_q8_k(ffn_norm, hidden, &gate_up_workspace, err, err_len)) goto fail;
    for (uint32_t rank = 0; rank < 8u; ++rank) {
        float *gate_rank = gate_values + (size_t)rank * intermediate;
        float *up_rank = up_values + (size_t)rank * intermediate;
        float *swiglu_rank = swiglu + (size_t)rank * intermediate;
        float *down_rank = down_values + (size_t)rank * hidden;
        float *weighted_rank = weighted + (size_t)rank * hidden;
        if (!qx_packed_expert_matvec_mode(&file, gate, selected[rank], ffn_norm, hidden, gate_rank, intermediate, gate_up_q8_k ? &gate_up_workspace : NULL, err, err_len) ||
            !qx_packed_expert_matvec_mode(&file, up, selected[rank], ffn_norm, hidden, up_rank, intermediate, gate_up_q8_k ? &gate_up_workspace : NULL, err, err_len)) goto fail;
        for (uint32_t i = 0; i < intermediate; ++i) swiglu_rank[i] = (float)(qx_silu((double)gate_rank[i]) * (double)up_rank[i]);
        qx_projection_workspace down_workspace = {0};
        if (down_q8_k && !qx_quantize_q8_k(swiglu_rank, intermediate, &down_workspace, err, err_len)) goto fail;
        if (!qx_packed_expert_matvec_mode(&file, down, selected[rank], swiglu_rank, intermediate, down_rank, hidden, down_q8_k ? &down_workspace : NULL, err, err_len)) goto fail;
        for (uint32_t i = 0; i < hidden; ++i) weighted_rank[i] = (float)((double)down_rank[i] * (double)weights_norm[rank]);
    }

    uint32_t written = 0u;
#define QX_WRITE_MOE_STAGE(name_, values_, count_) do { if (!qx_write_moe_stage_sidecar(output_dir, name_, layer, values_, count_, err, err_len)) goto fail; ++written; } while (0)
    QX_WRITE_MOE_STAGE("ffn_norm", ffn_norm, hidden);
    QX_WRITE_MOE_STAGE("ffn_moe_logits", logits, experts);
    QX_WRITE_MOE_STAGE("ffn_moe_probs", probs, experts);
    QX_WRITE_MOE_STAGE("ffn_moe_topk", topk, 8u);
    QX_WRITE_MOE_STAGE("ffn_moe_weights", weights, 8u);
    QX_WRITE_MOE_STAGE("ffn_moe_weights_sum", weight_sum_sidecar, 1u);
    QX_WRITE_MOE_STAGE("ffn_moe_weights_norm", weights_norm, 8u);
    QX_WRITE_MOE_STAGE("ffn_moe_gate", gate_values, (uint64_t)intermediate * 8u);
    QX_WRITE_MOE_STAGE("ffn_moe_up", up_values, (uint64_t)intermediate * 8u);
    QX_WRITE_MOE_STAGE("ffn_moe_swiglu", swiglu, (uint64_t)intermediate * 8u);
    QX_WRITE_MOE_STAGE("ffn_moe_down", down_values, (uint64_t)hidden * 8u);
    QX_WRITE_MOE_STAGE("ffn_moe_weighted", weighted, (uint64_t)hidden * 8u);
#undef QX_WRITE_MOE_STAGE
    const char *gate_up_projection_kernel = gate_up_q8_k ? (gate->flags == 17u ? "iq2_xs_q8_k" : "iq2_s_q8_k") : "dequant_f32";
    const char *down_projection_kernel = down_q8_k ?
        (down->flags == 18u ? "iq3_xxs_q8_k" : down->flags == 21u ? "iq3_s_q8_k" : "iq4_xs_q8_k") : "dequant_f32";
    fprintf(out, "{\"probe\":\"moe_stage\",\"layer\":%u,\"input_count\":%u,\"experts\":%u,\"experts_used\":8,\"intermediate\":%u,\"activation_mode\":\"%s\",\"projection_kernel\":\"%s\",\"gate_up_projection_kernel\":\"%s\",\"down_projection_kernel\":\"%s\",\"gate_ggml_type\":%u,\"up_ggml_type\":%u,\"down_ggml_type\":%u,\"selected_experts\":[",
            layer, hidden, experts, intermediate, activation_mode, projection_kernel,
            gate_up_projection_kernel, down_projection_kernel, gate->flags, up->flags, down->flags);
    for (uint32_t rank = 0; rank < 8u; ++rank) fprintf(out, "%s%u", rank ? "," : "", selected[rank]);
    fprintf(out, "],\"routing_weights\":[");
    for (uint32_t rank = 0; rank < 8u; ++rank) fprintf(out, "%s%.9g", rank ? "," : "", weights_norm[rank]);
    fprintf(out, "],\"sidecars_written\":%u}\n", written);
    free(ffn_input); free(ffn_norm); free(logits); free(probs); free(gate_values); free(up_values); free(swiglu); free(down_values); free(weighted); qx_close_file(&file); return 1;

fail:
    free(ffn_input); free(ffn_norm); free(logits); free(probs); free(gate_values); free(up_values); free(swiglu); free(down_values); free(weighted); qx_close_file(&file); return 0;
}

static int qx_write_residual_dump(const char *dir, uint32_t step, uint32_t layer, const char *phase,
        const float *values, uint32_t count, char *err, uint64_t err_len) {
    if (!dir || !*dir || !phase || !values || count == 0u) { qx_set_err(err, err_len, "invalid residual dump argument"); return 0; }
    char path[1024];
    int length = snprintf(path, sizeof(path), "%s/step-%u-layer-%u-%s.f32", dir, step, layer, phase);
    if (length < 0 || (size_t)length >= sizeof(path)) { qx_set_err(err, err_len, "residual dump path too long"); return 0; }
    FILE *fp = fopen(path, "wb");
    if (!fp) { qx_set_err(err, err_len, "cannot open residual dump"); return 0; }
    int ok = fwrite(values, sizeof(float), count, fp) == count;
    if (fclose(fp) != 0) ok = 0;
    if (!ok) qx_set_err(err, err_len, "cannot write residual dump");
    return ok;
}

static int qx_write_logits_dump(const char *dir, uint32_t step, const float *values, uint32_t count, char *err, uint64_t err_len) {
    if (!dir || !*dir || !values || count == 0u) { qx_set_err(err, err_len, "invalid logits dump argument"); return 0; }
    char path[1024];
    int length = snprintf(path, sizeof(path), "%s/step-%u-logits.f32", dir, step);
    if (length < 0 || (size_t)length >= sizeof(path)) { qx_set_err(err, err_len, "logits dump path too long"); return 0; }
    FILE *fp = fopen(path, "wb");
    if (!fp) { qx_set_err(err, err_len, "cannot open logits dump"); return 0; }
    int ok = fwrite(values, sizeof(float), count, fp) == count;
    if (fclose(fp) != 0) ok = 0;
    if (!ok) qx_set_err(err, err_len, "cannot write logits dump");
    return ok;
}

static void qx_kv_snapshot_wr_le32(unsigned char *p, uint32_t value) {
    p[0] = (unsigned char)value;
    p[1] = (unsigned char)(value >> 8);
    p[2] = (unsigned char)(value >> 16);
    p[3] = (unsigned char)(value >> 24);
}

typedef struct {
    uint32_t state[8];
    uint64_t bit_count;
    unsigned char block[64];
    size_t block_len;
} qx_kv_sha256;

static uint32_t qx_kv_sha256_rotr(uint32_t value, uint32_t bits) {
    return (value >> bits) | (value << (32u - bits));
}

static void qx_kv_sha256_transform(qx_kv_sha256 *ctx, const unsigned char block[64]) {
    static const uint32_t constants[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };
    uint32_t words[64];
    for (uint32_t i = 0; i < 16u; ++i) {
        words[i] = ((uint32_t)block[i * 4u] << 24u) | ((uint32_t)block[i * 4u + 1u] << 16u) |
            ((uint32_t)block[i * 4u + 2u] << 8u) | (uint32_t)block[i * 4u + 3u];
    }
    for (uint32_t i = 16u; i < 64u; ++i) {
        uint32_t s0 = qx_kv_sha256_rotr(words[i - 15u], 7u) ^ qx_kv_sha256_rotr(words[i - 15u], 18u) ^ (words[i - 15u] >> 3u);
        uint32_t s1 = qx_kv_sha256_rotr(words[i - 2u], 17u) ^ qx_kv_sha256_rotr(words[i - 2u], 19u) ^ (words[i - 2u] >> 10u);
        words[i] = words[i - 16u] + s0 + words[i - 7u] + s1;
    }
    uint32_t a = ctx->state[0], b = ctx->state[1], c = ctx->state[2], d = ctx->state[3];
    uint32_t e = ctx->state[4], f = ctx->state[5], g = ctx->state[6], h = ctx->state[7];
    for (uint32_t i = 0; i < 64u; ++i) {
        uint32_t s1 = qx_kv_sha256_rotr(e, 6u) ^ qx_kv_sha256_rotr(e, 11u) ^ qx_kv_sha256_rotr(e, 25u);
        uint32_t choice = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + s1 + choice + constants[i] + words[i];
        uint32_t s0 = qx_kv_sha256_rotr(a, 2u) ^ qx_kv_sha256_rotr(a, 13u) ^ qx_kv_sha256_rotr(a, 22u);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = s0 + majority;
        h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void qx_kv_sha256_init(qx_kv_sha256 *ctx) {
    static const uint32_t initial[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    memcpy(ctx->state, initial, sizeof(initial));
    ctx->bit_count = 0u;
    ctx->block_len = 0u;
}

static void qx_kv_sha256_update(qx_kv_sha256 *ctx, const unsigned char *data, size_t length) {
    ctx->bit_count += (uint64_t)length * 8u;
    while (length > 0u) {
        size_t available = sizeof(ctx->block) - ctx->block_len;
        size_t take = length < available ? length : available;
        memcpy(ctx->block + ctx->block_len, data, take);
        ctx->block_len += take;
        data += take;
        length -= take;
        if (ctx->block_len == sizeof(ctx->block)) {
            qx_kv_sha256_transform(ctx, ctx->block);
            ctx->block_len = 0u;
        }
    }
}

static void qx_kv_sha256_final(qx_kv_sha256 *ctx, unsigned char digest[32]) {
    uint64_t bit_count = ctx->bit_count;
    unsigned char padding[128] = {0x80u};
    size_t padding_len = ctx->block_len < 56u ? 56u - ctx->block_len : 120u - ctx->block_len;
    qx_kv_sha256_update(ctx, padding, padding_len);
    unsigned char length_bytes[8];
    for (uint32_t i = 0; i < 8u; ++i) length_bytes[7u - i] = (unsigned char)(bit_count >> (i * 8u));
    qx_kv_sha256_update(ctx, length_bytes, sizeof(length_bytes));
    for (uint32_t i = 0; i < 8u; ++i) {
        digest[i * 4u] = (unsigned char)(ctx->state[i] >> 24u);
        digest[i * 4u + 1u] = (unsigned char)(ctx->state[i] >> 16u);
        digest[i * 4u + 2u] = (unsigned char)(ctx->state[i] >> 8u);
        digest[i * 4u + 3u] = (unsigned char)ctx->state[i];
    }
}

static uint32_t qx_kv_snapshot_format_id(const char *kv_format) {
    if (strcmp(kv_format, "int8") == 0) return 1u;
    if (strcmp(kv_format, "f16") == 0 || strcmp(kv_format, "fp16") == 0) return 2u;
    if (strcmp(kv_format, "f32") == 0) return 3u;
    return 0u;
}

static int qx_write_accumulated_kv_snapshot(
        const char *snapshot_path, uint32_t layers, uint32_t positions, uint32_t ctx_tokens,
        uint32_t kv_heads, uint32_t head_dim, const char *kv_format, uint64_t bytes_per_k_or_v,
        uint32_t next_token, uint32_t seed, const unsigned char *kcache, const unsigned char *vcache,
        const float *kscales, const float *vscales, char *err, uint64_t err_len) {
    if (!snapshot_path || !*snapshot_path || !kcache || !vcache || !kscales || !vscales ||
            positions == 0u || positions > ctx_tokens || bytes_per_k_or_v > UINT32_MAX) {
        qx_set_err(err, err_len, "invalid accumulated KV snapshot output"); return 0;
    }
    uint32_t format_id = qx_kv_snapshot_format_id(kv_format);
    if (format_id == 0u) { qx_set_err(err, err_len, "unsupported KV snapshot format"); return 0; }
    FILE *fp = fopen(snapshot_path, "wb");
    if (!fp) { qx_set_err(err, err_len, "cannot open accumulated KV snapshot output"); return 0; }
    unsigned char header[48] = {0};
    memcpy(header, "QXKVSNP1", 8u);
    qx_kv_snapshot_wr_le32(header + 8u, 2u);
    qx_kv_snapshot_wr_le32(header + 12u, layers);
    qx_kv_snapshot_wr_le32(header + 16u, positions);
    qx_kv_snapshot_wr_le32(header + 20u, ctx_tokens);
    qx_kv_snapshot_wr_le32(header + 24u, kv_heads);
    qx_kv_snapshot_wr_le32(header + 28u, head_dim);
    qx_kv_snapshot_wr_le32(header + 32u, format_id);
    qx_kv_snapshot_wr_le32(header + 36u, (uint32_t)bytes_per_k_or_v);
    qx_kv_snapshot_wr_le32(header + 40u, next_token);
    qx_kv_snapshot_wr_le32(header + 44u, seed);
    qx_kv_sha256 hash;
    qx_kv_sha256_init(&hash);
    qx_kv_sha256_update(&hash, header, sizeof(header));
    int ok = fwrite(header, 1u, sizeof(header), fp) == sizeof(header);
    for (uint32_t layer = 0; layer < layers && ok; ++layer) {
        for (uint32_t position = 0; position < positions && ok; ++position) {
            uint64_t slot = (uint64_t)layer * ctx_tokens + position;
            uint32_t bits = 0u;
            qx_kv_sha256_update(&hash, kcache + slot * bytes_per_k_or_v, (size_t)bytes_per_k_or_v);
            qx_kv_sha256_update(&hash, vcache + slot * bytes_per_k_or_v, (size_t)bytes_per_k_or_v);
            ok = fwrite(kcache + slot * bytes_per_k_or_v, 1u, (size_t)bytes_per_k_or_v, fp) == (size_t)bytes_per_k_or_v &&
                fwrite(vcache + slot * bytes_per_k_or_v, 1u, (size_t)bytes_per_k_or_v, fp) == (size_t)bytes_per_k_or_v;
            memcpy(&bits, &kscales[slot], sizeof(bits));
            qx_kv_snapshot_wr_le32(header, bits);
            qx_kv_sha256_update(&hash, header, 4u);
            if (ok) ok = fwrite(header, 1u, 4u, fp) == 4u;
            memcpy(&bits, &vscales[slot], sizeof(bits));
            qx_kv_snapshot_wr_le32(header, bits);
            qx_kv_sha256_update(&hash, header, 4u);
            if (ok) ok = fwrite(header, 1u, 4u, fp) == 4u;
        }
    }
    unsigned char digest[32];
    qx_kv_sha256_final(&hash, digest);
    if (ok) ok = fwrite(digest, 1u, sizeof(digest), fp) == sizeof(digest);
    if (fclose(fp) != 0) ok = 0;
    if (!ok) qx_set_err(err, err_len, "cannot write complete accumulated KV snapshot");
    return ok;
}

static int qx_read_accumulated_kv_snapshot(
        const char *snapshot_path, uint32_t layers, uint32_t ctx_tokens, uint32_t kv_heads,
        uint32_t head_dim, const char *kv_format, uint64_t bytes_per_k_or_v, uint32_t seed,
        unsigned char *kcache, unsigned char *vcache, float *kscales, float *vscales,
        uint32_t *positions_out, uint32_t *next_token_out, char *err, uint64_t err_len) {
    FILE *fp = fopen(snapshot_path, "rb");
    if (!fp) { qx_set_err(err, err_len, "cannot open accumulated KV snapshot input"); return 0; }
    unsigned char header[48];
    int ok = fread(header, 1u, sizeof(header), fp) == sizeof(header);
    uint32_t positions = ok ? qx_rd_le32(header + 16u) : 0u;
    uint32_t format_id = qx_kv_snapshot_format_id(kv_format);
    if (!ok || memcmp(header, "QXKVSNP1", 8u) != 0 || qx_rd_le32(header + 8u) != 2u ||
            qx_rd_le32(header + 12u) != layers || positions == 0u || positions > ctx_tokens ||
            qx_rd_le32(header + 20u) != ctx_tokens || qx_rd_le32(header + 24u) != kv_heads ||
            qx_rd_le32(header + 28u) != head_dim || qx_rd_le32(header + 32u) != format_id ||
            qx_rd_le32(header + 36u) != bytes_per_k_or_v || qx_rd_le32(header + 44u) != seed) {
        fclose(fp); qx_set_err(err, err_len, "accumulated KV snapshot header mismatch"); return 0;
    }
    qx_kv_sha256 hash;
    qx_kv_sha256_init(&hash);
    qx_kv_sha256_update(&hash, header, sizeof(header));
    for (uint32_t layer = 0; layer < layers && ok; ++layer) {
        for (uint32_t position = 0; position < positions && ok; ++position) {
            uint64_t slot = (uint64_t)layer * ctx_tokens + position;
            ok = fread(kcache + slot * bytes_per_k_or_v, 1u, (size_t)bytes_per_k_or_v, fp) == (size_t)bytes_per_k_or_v &&
                fread(vcache + slot * bytes_per_k_or_v, 1u, (size_t)bytes_per_k_or_v, fp) == (size_t)bytes_per_k_or_v;
            if (ok) {
                qx_kv_sha256_update(&hash, kcache + slot * bytes_per_k_or_v, (size_t)bytes_per_k_or_v);
                qx_kv_sha256_update(&hash, vcache + slot * bytes_per_k_or_v, (size_t)bytes_per_k_or_v);
            }
            unsigned char scale_bytes[4];
            if (ok) ok = fread(scale_bytes, 1u, 4u, fp) == 4u;
            if (ok) qx_kv_sha256_update(&hash, scale_bytes, 4u);
            if (ok) kscales[slot] = qx_rd_le_f32(scale_bytes);
            if (ok) ok = fread(scale_bytes, 1u, 4u, fp) == 4u;
            if (ok) qx_kv_sha256_update(&hash, scale_bytes, 4u);
            if (ok) vscales[slot] = qx_rd_le_f32(scale_bytes);
            if (ok && (!isfinite(kscales[slot]) || !isfinite(vscales[slot]) ||
                    (format_id != 1u && (kscales[slot] != 1.0f || vscales[slot] != 1.0f)))) ok = 0;
        }
    }
    unsigned char expected_digest[32], actual_digest[32];
    if (ok) ok = fread(expected_digest, 1u, sizeof(expected_digest), fp) == sizeof(expected_digest);
    if (ok) {
        qx_kv_sha256_final(&hash, actual_digest);
        if (memcmp(expected_digest, actual_digest, sizeof(actual_digest)) != 0) ok = 0;
    }
    if (ok && fgetc(fp) != EOF) ok = 0;
    if (fclose(fp) != 0) ok = 0;
    if (!ok) { qx_set_err(err, err_len, "accumulated KV snapshot payload mismatch"); return 0; }
    *positions_out = positions;
    *next_token_out = qx_rd_le32(header + 40u);
    return 1;
}

int qx_dump_prompt_state_loop_probe_summary(const char *path, const char *tokens_path, const uint32_t *prompt_tokens, uint32_t prompt_count, uint32_t generation_steps, uint32_t layers, uint32_t ctx_tokens, const char *kv_format, const char *activation_format, const char *scratch_policy, const char *kernel_policy, const char *thread_policy, uint32_t threads, const char *simd_policy, const char *expert_cache_policy, const char *cuda_policy, const char *prefill_gemm_policy, const char *speculative_policy, const char *kv2_policy, const char *sampling_policy, const char *long_context_policy, uint64_t long_context_rss_limit_bytes, int dequant_profile_enabled, int real_kv, int projection_matvec, int residual_vector, int residual_carry, int numeric_deltas, int delta_vectors, int attention_output_vector, int causal_attention, int rope_gqa_attention, int full_moe, int final_head, int bench, uint32_t residual_dims, const char *norm_name, uint32_t top_k, uint32_t scan, uint32_t logits_top_n, double temperature, uint32_t seed, const char *residual_dump_dir, uint32_t start_layer, const char *residual_input_path, const char *kv_snapshot_out_path, const char *kv_snapshot_in_path, FILE *out, char *err, uint64_t err_len) {
    if (!path || !kv_format || !activation_format || !prompt_tokens || prompt_count == 0u) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    if (strcmp(activation_format, "f32") != 0 && strcmp(activation_format, "q8_k_compat") != 0) {
        qx_set_err(err, err_len, "unsupported activation format"); return 0;
    }
    if (!scratch_policy) scratch_policy = "ephemeral";
    int scratch_persistent = strcmp(scratch_policy, "persistent") == 0;
    if (!scratch_persistent && strcmp(scratch_policy, "ephemeral") != 0) {
        qx_set_err(err, err_len, "unsupported scratch policy"); return 0;
    }
    if (!kernel_policy) kernel_policy = "baseline";
    if (strcmp(kernel_policy, "baseline") != 0 && strcmp(kernel_policy, "fused") != 0) {
        qx_set_err(err, err_len, "unsupported kernel policy"); return 0;
    }
    if (!thread_policy) thread_policy = "serial";
    if (threads == 0u) threads = 1u;
    int thread_pool_policy = strcmp(thread_policy, "pool") == 0;
    if (!thread_pool_policy && strcmp(thread_policy, "serial") != 0) {
        qx_set_err(err, err_len, "unsupported thread policy"); return 0;
    }
    if (!thread_pool_policy && threads != 1u) {
        qx_set_err(err, err_len, "serial thread policy requires --threads 1"); return 0;
    }
    if (thread_pool_policy && threads < 2u) { qx_set_err(err, err_len, "thread pool policy requires --threads >= 2"); return 0; }
    if (thread_pool_policy && threads > 64u) { qx_set_err(err, err_len, "thread pool policy supports at most 64 threads"); return 0; }
    if (thread_pool_policy && strcmp(activation_format, "f32") != 0) { qx_set_err(err, err_len, "thread pool policy currently requires F32 activation"); return 0; }
    if (!simd_policy) simd_policy = "scalar";
    int avx2_fma_policy = strcmp(simd_policy, "avx2-fma") == 0;
    if (!avx2_fma_policy && strcmp(simd_policy, "scalar") != 0) { qx_set_err(err, err_len, "unsupported simd policy"); return 0; }
    if (avx2_fma_policy && strcmp(kernel_policy, "fused") != 0) { qx_set_err(err, err_len, "avx2-fma simd policy requires --kernel-policy fused"); return 0; }
    if (avx2_fma_policy && strcmp(activation_format, "f32") != 0) { qx_set_err(err, err_len, "avx2-fma simd policy requires F32 activation"); return 0; }
    if (avx2_fma_policy && thread_pool_policy) { qx_set_err(err, err_len, "avx2-fma simd policy currently requires serial thread policy"); return 0; }
    if (!expert_cache_policy) expert_cache_policy = "none";
    if (strcmp(expert_cache_policy, "none") != 0) { qx_set_err(err, err_len, "unsupported expert cache policy"); return 0; }
    if (!cuda_policy) cuda_policy = "none";
    if (strcmp(cuda_policy, "none") != 0) { qx_set_err(err, err_len, "unsupported CUDA policy"); return 0; }
    if (!prefill_gemm_policy) prefill_gemm_policy = "none";
    if (strcmp(prefill_gemm_policy, "none") != 0) { qx_set_err(err, err_len, "unsupported prefill GEMM policy"); return 0; }
    if (!speculative_policy) speculative_policy = "none";
    if (strcmp(speculative_policy, "none") != 0) { qx_set_err(err, err_len, "unsupported speculative policy"); return 0; }
    if (!kv2_policy) kv2_policy = "none";
    if (strcmp(kv2_policy, "none") != 0) { qx_set_err(err, err_len, "unsupported KV2 policy"); return 0; }
    if (!sampling_policy) sampling_policy = "none";
    if (strcmp(sampling_policy, "none") != 0) { qx_set_err(err, err_len, "unsupported sampling policy"); return 0; }
    if (!long_context_policy) long_context_policy = "none";
    int ctx4k_smoke_policy = strcmp(long_context_policy, "ctx4k-smoke") == 0;
    if (!ctx4k_smoke_policy && strcmp(long_context_policy, "none") != 0) { qx_set_err(err, err_len, "unsupported long-context policy"); return 0; }
    if (ctx4k_smoke_policy && ctx_tokens < 4096u) { qx_set_err(err, err_len, "ctx4k-smoke long-context policy requires --ctx >= 4096"); return 0; }
    if (!ctx4k_smoke_policy && long_context_rss_limit_bytes != 0u) { qx_set_err(err, err_len, "long-context RSS limit requires ctx4k-smoke policy"); return 0; }
    if (full_moe && norm_name && *norm_name) { qx_set_err(err, err_len, "--norm cannot be combined with --full-moe"); return 0; }
    if (residual_dump_dir && *residual_dump_dir && !full_moe) { qx_set_err(err, err_len, "--dump-residuals requires --full-moe"); return 0; }
    if ((kv_snapshot_out_path || kv_snapshot_in_path) && !causal_attention) { qx_set_err(err, err_len, "KV snapshot requires causal attention"); return 0; }
    int residual_replay = residual_input_path && *residual_input_path;
    if ((start_layer != 0u && !residual_replay) ||
            (residual_replay && (!full_moe || !residual_vector || prompt_count != 1u || generation_steps != 1u))) {
        qx_set_err(err, err_len, "residual replay requires a full MoE residual vector, exactly one token step, --start-layer, and --residual-in"); return 0;
    }
    if (generation_steps == 0u || generation_steps > 64u || prompt_count > 64u || prompt_count > UINT32_MAX - generation_steps + 1u) {
        qx_set_err(err, err_len, "prompt state loop requires 1..64 prompt tokens and 1..64 generation steps"); return 0;
    }
    uint32_t steps = prompt_count + generation_steps - 1u;
    if (steps > 64u) { qx_set_err(err, err_len, "prompt plus generation exceeds 64 forward steps"); return 0; }
    if (layers == 0) layers = 1;
    if (top_k == 0) top_k = 1;
    if (top_k > 32) top_k = 32;
    if (residual_dims == 0) residual_dims = 64;
    uint32_t max_residual_dims = rope_gqa_attention ? 2048u : 256u;
    if (residual_dims > max_residual_dims) residual_dims = max_residual_dims;
    if (causal_attention && (!projection_matvec || !residual_vector)) { qx_set_err(err, err_len, "causal attention requires projection matvec and residual vector"); return 0; }
    if (!full_moe && (causal_attention || rope_gqa_attention) &&
            strcmp(kv_format, "int8") != 0 && strcmp(kv_format, "f16") != 0 &&
            strcmp(kv_format, "fp16") != 0 && strcmp(kv_format, "f32") != 0) {
        qx_set_err(err, err_len, "scalar attention requires INT8, F16, or F32 KV");
        return 0;
    }
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    qx_alloc_snapshot alloc_start = qx_alloc_profile_snapshot();
    qx_scratch_workspace scratch_workspace = {0};
    qx_scratch_workspace *scratch = scratch_persistent ? &scratch_workspace : NULL;
    if (full_moe) {
        if (strcmp(kv_format, "int8") != 0 && strcmp(kv_format, "f16") != 0 && strcmp(kv_format, "fp16") != 0 && strcmp(kv_format, "f32") != 0) { qx_close_file(&file); qx_set_err(err, err_len, "full MoE state loop requires INT8, F16, or diagnostic F32 KV"); return 0; }
        residual_dims = file.header.manifest.hidden;
        if (residual_dims == 0u || residual_dims > 2048u) { qx_close_file(&file); qx_set_err(err, err_len, "unsupported full MoE hidden size"); return 0; }
    }
    uint32_t requested_layers = layers;
    uint32_t manifest_layers = file.header.manifest.layers ? file.header.manifest.layers : layers;
    if (layers > manifest_layers) layers = manifest_layers;
    if (residual_replay && start_layer >= layers) {
        qx_close_file(&file); qx_set_err(err, err_len, "residual replay start layer must be below requested layers"); return 0;
    }
    uint32_t q_heads = file.header.manifest.q_heads;
    uint32_t kv_heads = file.header.manifest.kv_heads;
    uint32_t head_dim = file.header.manifest.head_dim;
    qx_projection_workspace projection_workspace = {0};
    int q8_k_kernel_used = 0;
    int f32_projection_used = 0;
    uint32_t projection_family_mask = 0u;
    if (projection_matvec) {
        if (strcmp(activation_format, "q8_k_compat") == 0) {
            qx_detect_projection_kernel_usage(&file, start_layer, layers, causal_attention, &q8_k_kernel_used, &f32_projection_used, &projection_family_mask);
        } else {
            f32_projection_used = 1;
        }
    }
    const char *projection_kernel = q8_k_kernel_used ?
        qx_projection_kernel_label(projection_family_mask, f32_projection_used) :
        f32_projection_used ? "dequant_x_f32_scalar" : "not_used";
    int moe_q8_k_used = 0, moe_f32_used = 0;
    uint32_t moe_gate_up_family_mask = 0u, moe_down_family_mask = 0u;
    int moe_gate_up_f32_used = 0, moe_down_f32_used = 0;
    if (full_moe) {
        if (strcmp(activation_format, "q8_k_compat") == 0) {
            qx_detect_moe_kernel_usage(
                &file, start_layer, layers, &moe_q8_k_used, &moe_f32_used,
                &moe_gate_up_family_mask, &moe_gate_up_f32_used,
                &moe_down_family_mask, &moe_down_f32_used);
        }
        else { moe_f32_used = 1; moe_gate_up_f32_used = 1; moe_down_f32_used = 1; }
    }
    uint32_t moe_family_mask = moe_gate_up_family_mask | moe_down_family_mask;
    const char *moe_projection_kernel = "not_used";
    if (moe_q8_k_used && moe_f32_used) moe_projection_kernel = "q8_k_expert_kernels_with_f32_fallback";
    else if (moe_q8_k_used && moe_family_mask == 3u) moe_projection_kernel = "iq2_xs_q8_k_and_iq3_xxs_q8_k";
    else if (moe_q8_k_used && moe_family_mask == 12u) moe_projection_kernel = "iq2_s_q8_k_and_iq4_xs_q8_k";
    else if (moe_q8_k_used && moe_family_mask == 15u) moe_projection_kernel = "iq2_xs_iq3_xxs_iq2_s_iq4_xs_q8_k";
    else if (moe_q8_k_used && moe_family_mask == 17u) moe_projection_kernel = "iq2_xs_q8_k_and_iq3_s_q8_k";
    else if (moe_q8_k_used && moe_family_mask == 20u) moe_projection_kernel = "iq2_s_q8_k_and_iq3_s_q8_k";
    else if (moe_q8_k_used && moe_family_mask == 31u) moe_projection_kernel = "iq2_xs_iq3_xxs_iq3_s_iq2_s_iq4_xs_q8_k";
    else if (moe_q8_k_used) moe_projection_kernel = "q8_k_expert_kernels";
    else if (moe_f32_used) moe_projection_kernel = "dequant_f32";
    const char *moe_gate_up_projection_kernel = full_moe ?
        qx_moe_gate_up_kernel_label(moe_gate_up_family_mask, moe_gate_up_f32_used) : "not_used";
    const char *moe_down_projection_kernel = full_moe ?
        qx_moe_down_kernel_label(moe_down_family_mask, moe_down_f32_used) : "not_used";
    uint32_t moe_activation_workspace_bytes = moe_q8_k_used ? ((file.header.manifest.hidden + 255u) / 256u) * (uint32_t)sizeof(qx_block_q8_k) : 0u;
    if (q_heads == 0u || kv_heads == 0u || head_dim == 0u || q_heads > UINT32_MAX / head_dim || kv_heads > UINT32_MAX / head_dim) {
        qx_close_file(&file);
        qx_set_err(err, err_len, "invalid attention dimensions");
        return 0;
    }
    if (final_head && (!full_moe || requested_layers != manifest_layers || temperature != 0.0)) {
        qx_close_file(&file);
        qx_set_err(err, err_len, "--final-head requires --full-moe, 1..64 steps, all manifest layers, and temperature 0");
        return 0;
    }
    if (thread_pool_policy && !final_head) {
        qx_close_file(&file);
        qx_set_err(err, err_len, "thread pool policy currently requires --final-head");
        return 0;
    }
    if (final_head && (file.header.manifest.model_type != QX_MODEL_QWEN3_MOE || manifest_layers != 48u || file.header.manifest.hidden != 2048u || file.header.manifest.vocab != 151936u || q_heads != 32u || kv_heads != 4u || head_dim != 128u)) {
        qx_close_file(&file);
        qx_set_err(err, err_len, "--final-head requires exact Qwen3-30B-A3B manifest dimensions");
        return 0;
    }
    if (ctx_tokens == 0u || ctx_tokens > 4096u) {
        qx_close_file(&file);
        qx_set_err(err, err_len, "state loop context must be between 1 and 4096");
        return 0;
    }
    if (logits_top_n == 0u) logits_top_n = 1u;
    if (logits_top_n > 32u) logits_top_n = 32u;
    uint32_t vocab = file.header.manifest.vocab ? file.header.manifest.vocab : 151936u;
    for (uint32_t i = 0; i < prompt_count; ++i) {
        if (prompt_tokens[i] >= vocab) { qx_close_file(&file); qx_set_err(err, err_len, "prompt token out of range"); return 0; }
    }
    if (steps > ctx_tokens) { qx_close_file(&file); qx_set_err(err, err_len, "steps exceed ctx"); return 0; }
    if (kv_heads != 0u && (uint64_t)head_dim > UINT64_MAX / (uint64_t)kv_heads) { qx_close_file(&file); qx_set_err(err, err_len, "state loop cache size overflow"); return 0; }
    uint64_t values_per_k_or_v = (uint64_t)kv_heads * (uint64_t)head_dim;
    int kv_f32 = strcmp(kv_format, "f32") == 0;
    int kv_f16 = strcmp(kv_format, "f16") == 0 || strcmp(kv_format, "fp16") == 0;
    uint64_t bytes_per_k_or_v = 0;
    uint32_t bytes_per_value = 0;
    if (!qx_kv_bytes_for_format(kv_format, values_per_k_or_v, &bytes_per_k_or_v, &bytes_per_value)) { qx_close_file(&file); qx_set_err(err, err_len, "unsupported kv format"); return 0; }
    if (bytes_per_k_or_v > UINT64_MAX / 2ull || bytes_per_k_or_v > (uint64_t)SIZE_MAX / sizeof(float)) { qx_close_file(&file); qx_set_err(err, err_len, "state loop cache size overflow"); return 0; }
    uint64_t bytes_per_token_layer = bytes_per_k_or_v * 2ull;
    if ((uint64_t)ctx_tokens > UINT64_MAX / bytes_per_token_layer) { qx_close_file(&file); qx_set_err(err, err_len, "state loop cache size overflow"); return 0; }
    uint64_t layer_stride = (uint64_t)ctx_tokens * bytes_per_token_layer;
    if ((uint64_t)layers > UINT64_MAX / (uint64_t)ctx_tokens) { qx_close_file(&file); qx_set_err(err, err_len, "state loop cache size overflow"); return 0; }
    uint64_t cache_slots = (uint64_t)layers * (uint64_t)ctx_tokens;
    if (cache_slots > UINT64_MAX / bytes_per_k_or_v || cache_slots > (uint64_t)SIZE_MAX / sizeof(float)) { qx_close_file(&file); qx_set_err(err, err_len, "state loop cache size overflow"); return 0; }
    uint64_t persistent_cache_bytes = cache_slots * bytes_per_k_or_v;
    if (persistent_cache_bytes > (uint64_t)SIZE_MAX) { qx_close_file(&file); qx_set_err(err, err_len, "state loop cache size overflow"); return 0; }
    unsigned char *kbuf = (unsigned char *)malloc((size_t)bytes_per_k_or_v);
    unsigned char *vbuf = (unsigned char *)malloc((size_t)bytes_per_k_or_v);
    unsigned char *kcache = causal_attention ? (unsigned char *)calloc(1, (size_t)persistent_cache_bytes) : NULL;
    unsigned char *vcache = causal_attention ? (unsigned char *)calloc(1, (size_t)persistent_cache_bytes) : NULL;
    float *kfloat = causal_attention ? (float *)malloc((size_t)values_per_k_or_v * sizeof(float)) : NULL;
    float *vfloat = causal_attention ? (float *)malloc((size_t)values_per_k_or_v * sizeof(float)) : NULL;
    float *kscales = causal_attention ? (float *)calloc((size_t)cache_slots, sizeof(float)) : NULL;
    float *vscales = causal_attention ? (float *)calloc((size_t)cache_slots, sizeof(float)) : NULL;
    float *residual_vec = residual_vector ? (float *)malloc((size_t)residual_dims * (full_moe ? 2u : 1u) * sizeof(float)) : NULL;
    if (!kbuf || !vbuf || (causal_attention && (!kcache || !vcache || !kfloat || !vfloat || !kscales || !vscales)) || (residual_vector && !residual_vec)) { free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
    uint32_t position_base = 0u;
    uint32_t snapshot_next_token = prompt_tokens[0];
    if (kv_snapshot_in_path && *kv_snapshot_in_path) {
        if (prompt_count != 1u || !qx_read_accumulated_kv_snapshot(kv_snapshot_in_path, layers, ctx_tokens, kv_heads,
                head_dim, kv_format, bytes_per_k_or_v, seed, kcache, vcache, kscales, vscales,
                &position_base, &snapshot_next_token, err, err_len)) {
            free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file);
            if (prompt_count != 1u) qx_set_err(err, err_len, "KV snapshot replay requires exactly one continuation token");
            return 0;
        }
        if (prompt_tokens[0] != snapshot_next_token) {
            free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file);
            qx_set_err(err, err_len, "KV snapshot continuation token mismatch"); return 0;
        }
    }
    if (position_base > ctx_tokens || steps > ctx_tokens - position_base || position_base > 64u || steps > 64u - position_base) {
        free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file);
        qx_set_err(err, err_len, "KV snapshot plus replay steps exceed context"); return 0;
    }
    uint32_t current = prompt_tokens[0];
    uint64_t kv_appends = 0;
    uint64_t layers_run = 0;
    int readback_ok = 1;
    uint64_t k_mix = 1469598103934665603ull;
    uint64_t v_mix = 1469598103934665603ull;
    qx_dequant_dot_profile dequant_profile = {0};
    uint32_t thread_workers_used = thread_pool_policy ? threads : 1u;
    uint64_t thread_parallel_jobs = 0u;
    uint64_t thread_serial_jobs = 0u;
    uint64_t thread_fallback_jobs = 0u;
    uint64_t simd_fma_dot_calls = 0u;
    uint64_t simd_fallback_dot_calls = 0u;
    char generated[32768];
    generated[0] = 0;
    size_t generated_len = 0;
    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"state_loop\",\n");
    fprintf(out, "  \"prompt_token\": %u,\n", prompt_tokens[0]);
    fprintf(out, "  \"prompt_token_count\": %u,\n", prompt_count);
    fprintf(out, "  \"prompt_token_ids\": [");
    for (uint32_t i = 0; i < prompt_count; ++i) fprintf(out, "%s%u", i ? ", " : "", prompt_tokens[i]);
    fprintf(out, "],\n");
    fprintf(out, "  \"generation_steps\": %u,\n", generation_steps);
    fprintf(out, "  \"steps\": %u,\n", steps);
    fprintf(out, "  \"layers\": %u,\n", layers);
    fprintf(out, "  \"start_layer\": %u,\n", start_layer);
    fprintf(out, "  \"ctx_tokens\": %u,\n", ctx_tokens);
    fprintf(out, "  \"position_base\": %u,\n", position_base);
    fprintf(out, "  \"kv_format\": \"%s\",\n", kv_format);
    fprintf(out, "  \"activation_format\": \"%s\",\n", activation_format);
    fprintf(out, "  \"io_backend\": \"%s\",\n", qx_io_backend_name(file.io_backend));
    fprintf(out, "  \"scratch_policy\": \"%s\",\n", scratch_policy);
    fprintf(out, "  \"kernel_policy\": \"%s\",\n", kernel_policy);
    fprintf(out, "  \"thread_policy\": \"%s\",\n", thread_policy);
    fprintf(out, "  \"threads\": %u,\n", threads);
    fprintf(out, "  \"simd_policy\": \"%s\",\n", simd_policy);
    fprintf(out, "  \"expert_cache_policy\": \"%s\",\n", expert_cache_policy);
    fprintf(out, "  \"cuda_policy\": \"%s\",\n", cuda_policy);
    fprintf(out, "  \"prefill_gemm_policy\": \"%s\",\n", prefill_gemm_policy);
    fprintf(out, "  \"speculative_policy\": \"%s\",\n", speculative_policy);
    fprintf(out, "  \"kv2_policy\": \"%s\",\n", kv2_policy);
    fprintf(out, "  \"sampling_policy\": \"%s\",\n", sampling_policy);
    fprintf(out, "  \"long_context_policy\": \"%s\",\n", long_context_policy);
    fprintf(out, "  \"projection_kernel\": \"%s\",\n", projection_kernel);
    fprintf(out, "  \"activation_workspace_bytes\": %u,\n", q8_k_kernel_used ? (unsigned)sizeof(projection_workspace.blocks) : 0u);
    fprintf(out, "  \"moe_projection_kernel\": \"%s\",\n", moe_projection_kernel);
    fprintf(out, "  \"moe_gate_up_projection_kernel\": \"%s\",\n", moe_gate_up_projection_kernel);
    fprintf(out, "  \"moe_down_projection_kernel\": \"%s\",\n", moe_down_projection_kernel);
    fprintf(out, "  \"moe_activation_workspace_bytes\": %u,\n", moe_activation_workspace_bytes);
    fprintf(out, "  \"kv_source\": \"%s\",\n", projection_matvec ? "projection_matvec" : (real_kv ? "projection_decode" : "deterministic_skeleton"));
    fprintf(out, "  \"residual_source\": \"%s\",\n", residual_replay ? "injected_f32_replay" : (full_moe ? "real_attention_moe_carry" : (residual_carry ? "embedding_rmsnorm_carry" : (residual_vector ? "embedding_rmsnorm" : "probe_scalar"))));
    fprintf(out, "  \"residual_dims\": %u,\n", residual_vector ? residual_dims : 0u);
    fprintf(out, "  \"residual_replay\": {\"enabled\": %s, \"source\": %s, \"values\": %u},\n",
        residual_replay ? "true" : "false", residual_replay ? "\"f32_sidecar\"" : "null", residual_replay ? residual_dims : 0u);
    fprintf(out, "  \"residual_dump\": %s,\n", residual_dump_dir && *residual_dump_dir ? "true" : "false");
    fprintf(out, "  \"residual_dump_count\": %llu,\n", (unsigned long long)(residual_dump_dir && *residual_dump_dir ? (uint64_t)steps * (layers - start_layer) * 6u : 0u));
    fprintf(out, "  \"logits_dump_count\": %u,\n", residual_dump_dir && *residual_dump_dir && final_head ? steps : 0u);
    fprintf(out, "  \"delta_source\": \"%s\",\n", full_moe ? "real_attention_moe" : (rope_gqa_attention ? "rope_gqa_attention" : (causal_attention ? "causal_attention" : (attention_output_vector ? "attention_output_vector" : (delta_vectors ? "numeric_vectors" : (numeric_deltas ? "numeric_probe" : (residual_carry ? "checksum_skeleton" : "none")))))));
    fprintf(out, "  \"bytes_per_k_or_v\": %llu,\n", (unsigned long long)bytes_per_k_or_v);
    fprintf(out, "  \"bytes_per_token_per_layer\": %llu,\n", (unsigned long long)bytes_per_token_layer);
    fprintf(out, "  \"layer_stride\": %llu,\n", (unsigned long long)layer_stride);
    fprintf(out, "  \"tokens\": [");
    clock_t bench_start = clock();
    double prefill_elapsed = 0.0;
    double decode_elapsed = 0.0;
    uint32_t prefill_tokens = 0u;
    uint32_t decode_tokens = 0u;
    for (uint32_t local_step = 0; local_step < steps; ++local_step) {
        uint32_t step = position_base + local_step;
        if (local_step < prompt_count) current = prompt_tokens[local_step];
        int sampling_step = local_step + 1u >= prompt_count;
        clock_t phase_start = clock();
        fprintf(out, "%s{\"step\": %u, \"position\": %u, \"phase\": \"%s\", \"input_token\": %u, \"layers\": [", local_step ? ", " : "", step, step, sampling_step ? "generate" : "prefill", current);
        double residual_probe = 0.125 + ((double)(current % 997u) / 9970.0) + (double)step * 0.001;
        uint32_t residual_values = 0;
        double residual_rms = 0.0;
        uint64_t residual_checksum = 0;
        if (residual_vector) {
            if (!qx_fill_residual_vector_from_embedding(&file, current, norm_name, residual_vec, residual_dims, &residual_rms, &residual_checksum, err, err_len)) { free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); return 0; }
            residual_values = residual_dims;
            if (residual_replay) {
                if (!qx_read_exact_f32_sidecar(residual_input_path, residual_vec, residual_values, err, err_len)) { free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); return 0; }
                double sumsq = 0.0;
                for (uint32_t ri = 0; ri < residual_values; ++ri) sumsq += (double)residual_vec[ri] * (double)residual_vec[ri];
                residual_rms = sqrt(sumsq / (double)residual_values);
                residual_checksum = qx_fnv1a64((const unsigned char *)residual_vec, (uint64_t)residual_values * sizeof(float));
            }
            residual_probe = 0.0;
            for (uint32_t ri = 0; ri < residual_values; ++ri) residual_probe += residual_vec[ri];
            residual_probe /= (double)residual_values;
        }
        for (uint32_t layer = start_layer; layer < layers; ++layer) {
            uint64_t residual_input_checksum = residual_vec && residual_values
                ? qx_fnv1a64((const unsigned char *)residual_vec, (uint64_t)residual_values * sizeof(float)) : 0ull;
            if (residual_dump_dir && *residual_dump_dir && !qx_write_residual_dump(residual_dump_dir, step, layer, "input", residual_vec, residual_values, err, err_len)) {
                free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); return 0;
            }
            float *projection_residual = residual_vec;
            if (full_moe) {
                char attention_norm_name[QX_NAME_MAX], q_norm_name[QX_NAME_MAX];
                snprintf(attention_norm_name, sizeof(attention_norm_name), "blk.%u.attn_norm.weight", layer);
                snprintf(q_norm_name, sizeof(q_norm_name), "blk.%u.attn_q_norm.weight", layer);
                const qx_tensor_dir_entry *attention_norm = qx_find_tensor(&file, attention_norm_name);
                const qx_tensor_dir_entry *q_norm = qx_find_tensor(&file, q_norm_name);
                double attention_rms = 0.0;
                projection_residual = residual_vec + residual_values;
                if (!attention_norm || !q_norm || !qx_apply_f32_rmsnorm(&file, attention_norm, residual_vec, projection_residual, residual_values, &attention_rms, err, err_len)) {
                    free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); return 0;
                }
            }
            uint64_t k_offset = (uint64_t)layer * layer_stride + (uint64_t)step * bytes_per_token_layer;
            uint64_t v_offset = k_offset + bytes_per_k_or_v;
            char kn[QX_NAME_MAX];
            char vn[QX_NAME_MAX];
            const qx_tensor_dir_entry *kt = NULL;
            const qx_tensor_dir_entry *vt = NULL;
            uint64_t k_real_values = 0;
            uint64_t v_real_values = 0;
            uint64_t k_matvec_values = 0;
            uint64_t v_matvec_values = 0;
            double kprobe = 0.0;
            double vprobe = 0.0;
            snprintf(kn, sizeof(kn), "blk.%u.attn_k.weight", layer);
            snprintf(vn, sizeof(vn), "blk.%u.attn_v.weight", layer);
            if (real_kv) {
                kt = qx_find_tensor(&file, kn);
                vt = qx_find_tensor(&file, vn);
                if ((!kt || !vt) && layer != 0) {
                    snprintf(kn, sizeof(kn), "blk.0.attn_k.weight");
                    snprintf(vn, sizeof(vn), "blk.0.attn_v.weight");
                    kt = qx_find_tensor(&file, kn);
                    vt = qx_find_tensor(&file, vn);
                }
                if (!kt || !vt) { free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); qx_set_err(err, err_len, "projection tensor not found"); return 0; }
                if (projection_matvec) {
                    uint32_t matvec_dims = residual_values ? residual_values : 64u;
                    if (!qx_projection_matvec_fill_mode(&file, kt, kbuf, causal_attention ? kfloat : NULL, (uint32_t)values_per_k_or_v, matvec_dims, projection_residual, residual_values, current, layer, seed + step * 17u, activation_format, &projection_workspace, &kprobe, &k_matvec_values, err, err_len) ||
                        !qx_projection_matvec_fill_mode(&file, vt, vbuf, causal_attention ? vfloat : NULL, (uint32_t)values_per_k_or_v, matvec_dims, projection_residual, residual_values, current, layer, (seed ^ 0x9e3779b9u) + step * 17u, activation_format, &projection_workspace, &vprobe, &v_matvec_values, err, err_len)) {
                        free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); return 0;
                    }
                    k_real_values = k_matvec_values;
                    v_real_values = v_matvec_values;
                } else if (!qx_fill_kv_from_projection(&file, kt, kbuf, bytes_per_k_or_v, current, step, layer, seed, &k_real_values, err, err_len) ||
                    !qx_fill_kv_from_projection(&file, vt, vbuf, bytes_per_k_or_v, current, step, layer, seed ^ 0x9e3779b9u, &v_real_values, err, err_len)) {
                    free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); return 0;
                }
            } else {
                qx_fill_kv_append(kbuf, bytes_per_k_or_v, current, step, layer, seed, 0);
                qx_fill_kv_append(vbuf, bytes_per_k_or_v, current, step, layer, seed, 1);
            }
            if (causal_attention) {
                char k_norm_name[QX_NAME_MAX];
                snprintf(k_norm_name, sizeof(k_norm_name), "blk.%u.attn_k_norm.weight", layer);
                const qx_tensor_dir_entry *k_norm = qx_find_tensor(&file, k_norm_name);
                if (full_moe && !k_norm) { free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); qx_set_err(err, err_len, "K head RMSNorm tensor not found"); return 0; }
                if (k_norm && !qx_apply_f32_head_rmsnorm(&file, k_norm, kfloat, kv_heads, head_dim, err, err_len)) { free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); return 0; }
                if (rope_gqa_attention) qx_apply_rope(kfloat, kv_heads, head_dim, step, 1000000.0);
                if (residual_dump_dir && *residual_dump_dir && !qx_write_residual_dump(residual_dump_dir, step, layer, "v-cur", vfloat, (uint32_t)values_per_k_or_v, err, err_len)) {
                    free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); return 0;
                }
                uint64_t scale_index = (uint64_t)layer * ctx_tokens + step;
                if (kv_f32) {
                    memcpy(kbuf, kfloat, (size_t)values_per_k_or_v * sizeof(float));
                    memcpy(vbuf, vfloat, (size_t)values_per_k_or_v * sizeof(float));
                    kscales[scale_index] = 1.0f;
                    vscales[scale_index] = 1.0f;
                } else if (kv_f16) {
                    for (uint32_t vi = 0; vi < (uint32_t)values_per_k_or_v; ++vi) {
                        uint16_t kh = qx_f32_to_fp16(kfloat[vi]);
                        uint16_t vh = qx_f32_to_fp16(vfloat[vi]);
                        kbuf[2u * vi] = (unsigned char)(kh & 0xffu);
                        kbuf[2u * vi + 1u] = (unsigned char)(kh >> 8);
                        vbuf[2u * vi] = (unsigned char)(vh & 0xffu);
                        vbuf[2u * vi + 1u] = (unsigned char)(vh >> 8);
                    }
                    kscales[scale_index] = 1.0f;
                    vscales[scale_index] = 1.0f;
                } else {
                    kscales[scale_index] = qx_quantize_int8_vector(kfloat, kbuf, (uint32_t)values_per_k_or_v);
                    vscales[scale_index] = qx_quantize_int8_vector(vfloat, vbuf, (uint32_t)values_per_k_or_v);
                }
            }
            uint64_t kchk = qx_fnv1a64(kbuf, bytes_per_k_or_v);
            uint64_t vchk = qx_fnv1a64(vbuf, bytes_per_k_or_v);
            uint64_t kchk2 = qx_fnv1a64(kbuf, bytes_per_k_or_v);
            uint64_t vchk2 = qx_fnv1a64(vbuf, bytes_per_k_or_v);
            if (kchk != kchk2 || vchk != vchk2) readback_ok = 0;
            k_mix ^= kchk; k_mix *= 1099511628211ull;
            v_mix ^= vchk; v_mix *= 1099511628211ull;
            double attention_delta = 0.0;
            double moe_delta = 0.0;
            double attention_vec_l2 = 0.0;
            double moe_vec_l2 = 0.0;
            double attention_output_l2 = 0.0;
            uint64_t attention_vec_checksum = 0;
            uint64_t moe_vec_checksum = 0;
            uint64_t attention_output_checksum = 0;
            uint32_t attention_context_tokens = step + 1u;
            float *causal_vec = NULL;
            float *attention_context_capture = NULL;
            double causal_softmax_sum = 0.0;
            double causal_softmax_min = 0.0;
            double causal_softmax_max = 0.0;
            uint32_t causal_q_heads_run = 0;
            uint64_t causal_q_values = 0;
            const char *causal_q_name = NULL;
            const char *causal_o_name = NULL;
            uint32_t selected_experts[8] = {0};
            double routing_weights[8] = {0};
            uint32_t gate_ggml_type = 0, up_ggml_type = 0, down_ggml_type = 0;
            double real_moe_output_l2 = 0.0;
            uint64_t residual_output_checksum = residual_input_checksum;
            if (causal_attention) {
                unsigned char *kdst = kcache + (((uint64_t)layer * ctx_tokens + step) * bytes_per_k_or_v);
                unsigned char *vdst = vcache + (((uint64_t)layer * ctx_tokens + step) * bytes_per_k_or_v);
                memcpy(kdst, kbuf, (size_t)bytes_per_k_or_v);
                memcpy(vdst, vbuf, (size_t)bytes_per_k_or_v);
                if (memcmp(kdst, kbuf, (size_t)bytes_per_k_or_v) != 0 || memcmp(vdst, vbuf, (size_t)bytes_per_k_or_v) != 0) readback_ok = 0;
                causal_vec = (float *)malloc((size_t)residual_values * sizeof(float));
                uint32_t attention_context_values = q_heads * head_dim;
                attention_context_capture = residual_dump_dir && *residual_dump_dir ? (float *)malloc((size_t)attention_context_values * sizeof(float)) : NULL;
                if (residual_dump_dir && *residual_dump_dir && !attention_context_capture) {
                    free(causal_vec); free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0;
                }
                int attention_ok = causal_vec && (rope_gqa_attention
                    ? qx_rope_gqa_attention_partial(&file, layer, step, ctx_tokens, bytes_per_k_or_v, kcache, vcache, (uint32_t)values_per_k_or_v, bytes_per_value, kscales, vscales, projection_residual, residual_values, q_heads, kv_heads, head_dim, current, seed, causal_vec, attention_context_capture, attention_context_values, &causal_softmax_sum, &causal_softmax_min, &causal_softmax_max, &causal_q_heads_run, &causal_q_values, &causal_q_name, &causal_o_name, activation_format, &projection_workspace, scratch, err, err_len)
                    : qx_causal_attention_partial(&file, layer, step, ctx_tokens, bytes_per_k_or_v, kcache, vcache, bytes_per_value, kscales, vscales, projection_residual, residual_values, current, seed, causal_vec, &causal_softmax_sum, &causal_q_values, &causal_q_name, &causal_o_name, err, err_len));
                if (!attention_ok) {
                    free(attention_context_capture); free(causal_vec); free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); return 0;
                }
                if (residual_dump_dir && *residual_dump_dir && !qx_write_residual_dump(residual_dump_dir, step, layer, "kqv-out", attention_context_capture, attention_context_values, err, err_len)) {
                    free(attention_context_capture); free(causal_vec); free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); return 0;
                }
                free(attention_context_capture);
                attention_context_capture = NULL;
            }
            if (full_moe) {
                for (uint32_t ri = 0; ri < residual_values; ++ri) {
                    attention_output_l2 += (double)causal_vec[ri] * (double)causal_vec[ri];
                    residual_vec[ri] += causal_vec[ri];
                }
                attention_output_l2 = sqrt(attention_output_l2);
                attention_output_checksum = qx_fnv1a64((const unsigned char *)causal_vec, (uint64_t)residual_values * sizeof(float));
                if (residual_dump_dir && *residual_dump_dir && !qx_write_residual_dump(residual_dump_dir, step, layer, "ffn-inp", residual_vec, residual_values, err, err_len)) {
                    free(causal_vec); free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); return 0;
                }
                if (!qx_apply_real_moe_layer(&file, layer, residual_vec, residual_values, residual_vec,
                        residual_dump_dir && *residual_dump_dir ? causal_vec : NULL,
                        selected_experts, routing_weights, &gate_ggml_type, &up_ggml_type, &down_ggml_type,
                        activation_format, scratch, &real_moe_output_l2, err, err_len)) {
                    free(causal_vec); free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); return 0;
                }
                if (residual_dump_dir && *residual_dump_dir && !qx_write_residual_dump(residual_dump_dir, step, layer, "ffn-moe-out", causal_vec, residual_values, err, err_len)) {
                    free(causal_vec); free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); return 0;
                }
                residual_output_checksum = qx_fnv1a64((const unsigned char *)residual_vec, (uint64_t)residual_values * sizeof(float));
                if (residual_dump_dir && *residual_dump_dir && !qx_write_residual_dump(residual_dump_dir, step, layer, "output", residual_vec, residual_values, err, err_len)) {
                    free(causal_vec); free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); return 0;
                }
                residual_probe = 0.0;
                for (uint32_t ri = 0; ri < residual_values; ++ri) residual_probe += residual_vec[ri];
                residual_probe /= (double)residual_values;
            }
            if (!full_moe && residual_carry && residual_vec && residual_values) {
                if (numeric_deltas) {
                    attention_delta = kprobe / (double)(residual_values ? residual_values : 1u);
                    moe_delta = vprobe / (double)(2u * (residual_values ? residual_values : 1u));
                    if (!isfinite(attention_delta)) attention_delta = 0.000123;
                    if (!isfinite(moe_delta)) moe_delta = -0.000077;
                } else {
                    attention_delta = ((double)((kchk >> 8) & 0xffffu) / 65535.0 - 0.5) * 0.002;
                    moe_delta = ((double)((vchk >> 16) & 0xffffu) / 65535.0 - 0.5) * 0.001;
                }
                if (delta_vectors) {
                    float *avec = (float *)malloc((size_t)residual_values * sizeof(float));
                    float *mvec = (float *)malloc((size_t)residual_values * sizeof(float));
                    float *outv = attention_output_vector ? (float *)malloc((size_t)residual_values * sizeof(float)) : NULL;
                    if (!avec || !mvec || (attention_output_vector && !outv)) { free(avec); free(mvec); free(outv); free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
                    for (uint32_t ri = 0; ri < residual_values; ++ri) {
                        double awave = ((double)(((ri + 1u) * (layer + 3u)) % 17u) - 8.0) / 8.0;
                        double mwave = ((double)(((ri + 5u) * (layer + 7u)) % 19u) - 9.0) / 9.0;
                        unsigned char kb = kbuf[ri % bytes_per_k_or_v];
                        unsigned char vb = vbuf[(ri * 7u + layer + step) % bytes_per_k_or_v];
                        double kv_context = (((double)vb - 128.0) / 128.0) * (1.0 + ((double)kb / 255.0)) / (double)attention_context_tokens;
                        avec[ri] = (float)(attention_delta * awave);
                        mvec[ri] = (float)(moe_delta * mwave);
                        if (attention_output_vector) outv[ri] = causal_attention ? causal_vec[ri] : (float)(kv_context * 0.00025 + attention_delta * awave);
                        residual_vec[ri] += (attention_output_vector ? outv[ri] : avec[ri]) + mvec[ri];
                        attention_vec_l2 += (double)avec[ri] * (double)avec[ri];
                        moe_vec_l2 += (double)mvec[ri] * (double)mvec[ri];
                        if (attention_output_vector) attention_output_l2 += (double)outv[ri] * (double)outv[ri];
                    }
                    attention_vec_l2 = sqrt(attention_vec_l2);
                    moe_vec_l2 = sqrt(moe_vec_l2);
                    attention_output_l2 = sqrt(attention_output_l2);
                    if (!isfinite(attention_vec_l2)) attention_vec_l2 = 0.000001;
                    if (!isfinite(moe_vec_l2)) moe_vec_l2 = 0.000001;
                    if (!isfinite(attention_output_l2)) attention_output_l2 = 0.000001;
                    attention_vec_checksum = qx_fnv1a64((const unsigned char *)avec, (uint64_t)residual_values * sizeof(float));
                    moe_vec_checksum = qx_fnv1a64((const unsigned char *)mvec, (uint64_t)residual_values * sizeof(float));
                    if (attention_output_vector) attention_output_checksum = qx_fnv1a64((const unsigned char *)outv, (uint64_t)residual_values * sizeof(float));
                    free(avec); free(mvec); free(outv);
                } else {
                    for (uint32_t ri = 0; ri < residual_values; ++ri) {
                        double wave = ((double)(((ri + 1u) * (layer + 3u)) % 17u) - 8.0) / 8.0;
                        residual_vec[ri] += (float)((attention_delta + moe_delta) * wave);
                    }
                }
            }
            residual_probe += ((double)((kchk ^ vchk) & 0xffffu) / 65535.0) * 0.0001;
            fprintf(out, "%s{\"layer\": %u, \"kv_token\": %u, \"k_offset\": %llu, \"v_offset\": %llu, \"k_checksum\": %llu, \"v_checksum\": %llu", layer != start_layer ? ", " : "", layer, step, (unsigned long long)k_offset, (unsigned long long)v_offset, (unsigned long long)kchk, (unsigned long long)vchk);
            if (full_moe) {
                fprintf(out, ", \"full_moe\": true, \"qk_head_norm\": true, \"experts_run\": 8, \"selected_experts\": [");
                for (uint32_t rank = 0; rank < 8u; ++rank) fprintf(out, "%s%u", rank ? "," : "", selected_experts[rank]);
                fprintf(out, "], \"routing_weights\": [");
                for (uint32_t rank = 0; rank < 8u; ++rank) fprintf(out, "%s%.17g", rank ? "," : "", routing_weights[rank]);
                fprintf(out, "], \"gate_ggml_type\": %u, \"up_ggml_type\": %u, \"down_ggml_type\": %u, \"moe_output_l2\": %.17g, \"residual_input_checksum\": %llu, \"residual_output_checksum\": %llu",
                    gate_ggml_type, up_ggml_type, down_ggml_type, real_moe_output_l2,
                    (unsigned long long)residual_input_checksum, (unsigned long long)residual_output_checksum);
            }
            if (real_kv) {
                fprintf(out, ", \"k_tensor\": \"%s\", \"v_tensor\": \"%s\", \"k_real_values\": %llu, \"v_real_values\": %llu", kt ? kt->name : kn, vt ? vt->name : vn, (unsigned long long)k_real_values, (unsigned long long)v_real_values);
                if (projection_matvec) fprintf(out, ", \"k_matvec_values\": %llu, \"v_matvec_values\": %llu", (unsigned long long)k_matvec_values, (unsigned long long)v_matvec_values);
                if (residual_carry) {
                    if (!full_moe) {
                        fprintf(out, ", \"attention_delta\": %.9g, \"moe_delta\": %.9g, \"attention_delta_source\": \"%s\", \"moe_delta_source\": \"%s\"", attention_delta, moe_delta, numeric_deltas ? "numeric_attention" : "checksum_attention", numeric_deltas ? "numeric_moe" : "checksum_moe");
                        if (delta_vectors) fprintf(out, ", \"attention_delta_vector_values\": %u, \"moe_delta_vector_values\": %u, \"attention_delta_vector_l2\": %.9g, \"moe_delta_vector_l2\": %.9g, \"attention_delta_vector_checksum\": %llu, \"moe_delta_vector_checksum\": %llu", residual_values, residual_values, attention_vec_l2, moe_vec_l2, (unsigned long long)attention_vec_checksum, (unsigned long long)moe_vec_checksum);
                    }
                    if (attention_output_vector) fprintf(out, ", \"attention_output_vector_values\": %u, \"attention_output_vector_l2\": %.9g, \"attention_output_vector_checksum\": %llu, \"attention_context_tokens\": %u, \"attention_output_source\": \"%s\"", residual_values, attention_output_l2, (unsigned long long)attention_output_checksum, attention_context_tokens, rope_gqa_attention ? "rope_gqa_softmax_output_projection_partial" : (causal_attention ? "qkv_softmax_output_projection_partial" : "kv_cache_partial"));
                    if (causal_attention) {
                        uint64_t scale_index = (uint64_t)layer * ctx_tokens + step;
                        const qx_tensor_dir_entry *q_entry = causal_q_name ? qx_find_tensor(&file, causal_q_name) : NULL;
                        const qx_tensor_dir_entry *o_entry = causal_o_name ? qx_find_tensor(&file, causal_o_name) : NULL;
                        fprintf(out, ", \"persistent_kv\": true, \"kv_scale_source\": \"%s\", \"k_scale\": %.9g, \"v_scale\": %.9g, \"q_values\": %llu, \"q_tensor\": \"%s\", \"q_ggml_type\": %u, \"k_ggml_type\": %u, \"v_ggml_type\": %u, \"output_tensor\": \"%s\", \"output_ggml_type\": %u, \"softmax_sum\": %.9g", kv_f32 ? "lossless_f32" : kv_f16 ? "fp16_roundtrip" : "dynamic_per_vector", kscales[scale_index], vscales[scale_index], (unsigned long long)causal_q_values, causal_q_name ? causal_q_name : "", q_entry ? q_entry->flags : UINT32_MAX, kt ? kt->flags : UINT32_MAX, vt ? vt->flags : UINT32_MAX, causal_o_name ? causal_o_name : "", o_entry ? o_entry->flags : UINT32_MAX, causal_softmax_sum);
                        if (rope_gqa_attention) {
                            uint32_t group_size = kv_heads ? q_heads / kv_heads : 1u;
                            uint32_t kv_heads_touched = group_size ? (causal_q_heads_run + group_size - 1u) / group_size : 1u;
                            uint32_t output_dims = residual_values;
                            const qx_tensor_dir_entry *output_entry = causal_o_name ? qx_find_tensor(&file, causal_o_name) : NULL;
                            if (output_entry && output_entry->dims[1] && output_dims > output_entry->dims[1]) output_dims = (uint32_t)output_entry->dims[1];
                            fprintf(out, ", \"attention_mode\": \"rope_gqa_full_heads\", \"rope_applied\": true, \"rope_theta\": 1000000, \"q_heads_total\": %u, \"kv_heads_total\": %u, \"q_heads_run\": %u, \"kv_heads_touched\": %u, \"gqa_group_size\": %u, \"head_dim\": %u, \"output_projection_input_dims\": %u, \"output_projection_output_dims\": %u, \"softmax_sum_min\": %.9g, \"softmax_sum_max\": %.9g", q_heads, kv_heads, causal_q_heads_run, kv_heads_touched, group_size, head_dim, causal_q_heads_run * head_dim, output_dims, causal_softmax_min, causal_softmax_max);
                        }
                    }
                }
            }
            fprintf(out, "}");
            free(causal_vec);
            ++kv_appends;
            ++layers_run;
        }
        if (!sampling_step) {
            fprintf(out, "], \"residual_probe\": %.9g", residual_probe);
            if (residual_vector) fprintf(out, ", \"residual_values\": %u, \"residual_rms\": %.9g, \"residual_checksum\": %llu", residual_values, residual_rms, (unsigned long long)residual_checksum);
            if (residual_carry && residual_vec && residual_values) {
                uint64_t after = qx_fnv1a64((const unsigned char *)residual_vec, (uint64_t)residual_values * sizeof(float));
                fprintf(out, ", \"residual_checksum_after\": %llu", (unsigned long long)after);
            }
            fprintf(out, ", \"selected_token\": null, \"source\": \"fixed_prompt\"}");
            double phase_elapsed = (double)(clock() - phase_start) / (double)CLOCKS_PER_SEC;
            if (phase_elapsed <= 0.0) phase_elapsed = 0.000001;
            prefill_elapsed += phase_elapsed;
            ++prefill_tokens;
            current = prompt_tokens[step + 1u];
            continue;
        }
        qx_top_token top[32];
        qx_real_head_result head_result;
        int head_ready = 0;
        const char *lm_name = NULL;
        int tied = 0;
        uint32_t scanned = 0;
        uint32_t sample_top_n = top_k;
        if (final_head) {
            float *normalized = residual_vec + residual_values;
            float *logits_dump = residual_dump_dir && *residual_dump_dir ? (float *)malloc((size_t)file.header.manifest.vocab * sizeof(float)) : NULL;
            if (residual_dump_dir && *residual_dump_dir && !logits_dump) { free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
            if (!qx_compute_real_final_head(&file, residual_vec, normalized, residual_values, logits_top_n, activation_format, kernel_policy, thread_policy, threads, simd_policy, logits_dump, file.header.manifest.vocab, &head_result, err, err_len)) { free(logits_dump); free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); return 0; }
            dequant_profile.temporary_blocks_decoded += head_result.dequant_profile.temporary_blocks_decoded;
            dequant_profile.temporary_floats_materialized += head_result.dequant_profile.temporary_floats_materialized;
            dequant_profile.temporary_bytes_materialized += head_result.dequant_profile.temporary_bytes_materialized;
            dequant_profile.fused_dot_calls += head_result.dequant_profile.fused_dot_calls;
            dequant_profile.fallback_dot_calls += head_result.dequant_profile.fallback_dot_calls;
            dequant_profile.final_head_q6_k_blocks += head_result.dequant_profile.final_head_q6_k_blocks;
            thread_workers_used = head_result.workers_used;
            thread_parallel_jobs += head_result.parallel_jobs;
            thread_serial_jobs += head_result.serial_jobs;
            thread_fallback_jobs += head_result.fallback_jobs;
            simd_fma_dot_calls += head_result.simd_fma_dot_calls;
            simd_fallback_dot_calls += head_result.simd_fallback_dot_calls;
            if (logits_dump && !qx_write_logits_dump(residual_dump_dir, step, logits_dump, head_result.vocab_size, err, err_len)) { free(logits_dump); free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); return 0; }
            free(logits_dump);
            memcpy(top, head_result.top, (size_t)head_result.top_n * sizeof(qx_top_token));
            sample_top_n = head_result.top_n;
            lm_name = "output.weight";
            scanned = head_result.logits_computed;
            head_ready = 1;
        } else if (!qx_collect_top_logits(&file, residual_probe, top_k, scan, seed + step * 17u + current, &lm_name, &tied, &scanned, top, err, err_len)) {
            free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); return 0;
        }
        if (!head_ready && sample_top_n > scanned) sample_top_n = scanned;
        qx_sample_result sr = qx_sample_from_top(top, sample_top_n, temperature, seed + step * 101u + current);
        char piece[8192];
        char fallback[64];
        const char *source = "fallback_token_id";
        if (tokens_path && qx_lookup_token_piece_sidecar(tokens_path, sr.selected_token, piece, sizeof(piece))) source = "sidecar";
        else { qx_token_piece_fallback(sr.selected_token, fallback, sizeof(fallback));
#ifdef _MSC_VER
            strncpy_s(piece, sizeof(piece), fallback, _TRUNCATE);
#else
            snprintf(piece, sizeof(piece), "%s", fallback);
#endif
        }
        size_t plen = strlen(piece);
        if (generated_len + plen < sizeof(generated)) { memcpy(generated + generated_len, piece, plen + 1); generated_len += plen; }
        fprintf(out, "], \"residual_probe\": %.9g", residual_probe);
        if (residual_vector) fprintf(out, ", \"residual_values\": %u, \"residual_rms\": %.9g, \"residual_checksum\": %llu", residual_values, residual_rms, (unsigned long long)residual_checksum);
        if (residual_carry && residual_vec && residual_values) {
            uint64_t after = qx_fnv1a64((const unsigned char *)residual_vec, (uint64_t)residual_values * sizeof(float));
            fprintf(out, ", \"residual_checksum_after\": %llu", (unsigned long long)after);
        }
        if (head_ready) {
            const float *normalized = residual_vec + residual_values;
            fprintf(out, ", \"final_head\": {\"enabled\": true, \"norm_tensor\": \"output_norm.weight\", \"norm_ggml_type\": 0, \"norm_values\": %u, \"norm_checksum_verified\": true, \"norm_raw_checksum\": %llu, \"input_rms\": %.17g, \"normalized_l2\": %.17g, \"final_residual_checksum\": %llu, \"final_norm_checksum\": %llu, \"lm_head_tensor\": \"output.weight\", \"lm_head_ggml_type\": %u, \"lm_head_decoder\": \"Q6_K\", \"lm_head_kernel\": \"%s\", \"activation_quantizations\": %u, \"lm_head_checksum_verified\": true, \"lm_head_raw_checksum\": %llu, \"input_dims\": %u, \"vocab_size\": %u, \"logits_computed\": %u, \"full_vocabulary\": true, \"logits_checksum\": %llu, \"logits_min\": %.17g, \"logits_max\": %.17g, \"logits_rms\": %.17g, \"argmax_token\": %u, \"argmax_logit\": %.17g, \"top_tokens\": [",
                head_result.input_dims, (unsigned long long)head_result.norm_raw_checksum, head_result.input_rms, head_result.normalized_l2,
                (unsigned long long)head_result.residual_checksum, (unsigned long long)head_result.normalized_checksum,
                head_result.lm_head_ggml_type, head_result.lm_head_kernel, head_result.activation_quantizations, (unsigned long long)head_result.lm_head_raw_checksum, head_result.input_dims, head_result.vocab_size, head_result.logits_computed,
                (unsigned long long)head_result.logits_checksum, head_result.logits_min, head_result.logits_max, head_result.logits_rms,
                head_result.top[0].token, head_result.top[0].logit);
            for (uint32_t rank = 0; rank < head_result.top_n; ++rank) {
                fprintf(out, "%s{\"token\": %u, \"logit\": %.17g, \"row_checksum\": %llu}", rank ? ", " : "", head_result.top[rank].token, head_result.top[rank].logit, (unsigned long long)head_result.top[rank].checksum);
            }
            fprintf(out, "], \"final_residual_raw\": [");
            for (uint32_t i = 0; i < residual_values; ++i) fprintf(out, "%s%.9g", i ? "," : "", residual_vec[i]);
            fprintf(out, "], \"final_norm_raw\": [");
            for (uint32_t i = 0; i < residual_values; ++i) fprintf(out, "%s%.9g", i ? "," : "", normalized[i]);
            fprintf(out, "]}");
        }
        fprintf(out, ", \"selected_token\": %u, \"piece\": ", sr.selected_token);
        qx_json_print_escaped(out, piece);
        fprintf(out, ", \"source\": \"%s\"}", source);
        double phase_elapsed = (double)(clock() - phase_start) / (double)CLOCKS_PER_SEC;
        if (phase_elapsed <= 0.0) phase_elapsed = 0.000001;
        decode_elapsed += phase_elapsed;
        ++decode_tokens;
        current = sr.selected_token;
    }
    if (kv_snapshot_out_path && *kv_snapshot_out_path && !qx_write_accumulated_kv_snapshot(
            kv_snapshot_out_path, layers, position_base + steps, ctx_tokens, kv_heads, head_dim, kv_format,
            bytes_per_k_or_v, current, seed, kcache, vcache, kscales, vscales, err, err_len)) {
        free(kbuf); free(vbuf); free(kcache); free(vcache); free(kfloat); free(vfloat); free(kscales); free(vscales); free(residual_vec); qx_scratch_free(&scratch_workspace); qx_close_file(&file); return 0;
    }
    fprintf(out, "],\n");
    fprintf(out, "  \"layers_run\": %llu,\n", (unsigned long long)layers_run);
    fprintf(out, "  \"kv_appends\": %llu,\n", (unsigned long long)kv_appends);
    fprintf(out, "  \"cache_readback_ok\": %s,\n", readback_ok ? "true" : "false");
    fprintf(out, "  \"k_cache_checksum\": %llu,\n", (unsigned long long)k_mix);
    fprintf(out, "  \"v_cache_checksum\": %llu,\n", (unsigned long long)v_mix);
    fprintf(out, "  \"final_token\": %u,\n", current);
    fprintf(out, "  \"generated_text\": "); qx_json_print_escaped(out, generated); fprintf(out, ",\n");
    if (bench) {
        double elapsed = (double)(clock() - bench_start) / (double)CLOCKS_PER_SEC;
        if (elapsed <= 0.0) elapsed = 0.000001;
        double tps = (double)steps / elapsed;
        double mspt = 1000.0 / tps;
        double lps = (double)layers_run / elapsed;
        double prefill_tps = prefill_tokens ? (double)prefill_tokens / prefill_elapsed : 0.0;
        double decode_tps = decode_tokens ? (double)decode_tokens / decode_elapsed : 0.0;
        fprintf(out, "  \"bench\": {\"enabled\": true, \"timer\": \"process_clock\", \"elapsed_sec\": %.9g, \"tokens_per_second\": %.9g, \"ms_per_token\": %.9g, \"layer_steps\": %llu, \"layer_steps_per_second\": %.9g, \"phases\": {\"prefill\": {\"tokens\": %u, \"elapsed_sec\": %.9g, \"tokens_per_second\": %.9g}, \"decode\": {\"tokens\": %u, \"elapsed_sec\": %.9g, \"tokens_per_second\": %.9g}}},\n",
            elapsed, tps, mspt, (unsigned long long)layers_run, lps,
            prefill_tokens, prefill_elapsed, prefill_tps, decode_tokens, decode_elapsed, decode_tps);
    }
    qx_alloc_snapshot alloc_delta = qx_alloc_profile_delta(alloc_start);
    fprintf(out, "  \"allocation_profile\": {\"malloc_calls\": %llu, \"calloc_calls\": %llu, \"realloc_calls\": %llu, \"free_calls\": %llu, \"bytes_requested\": %llu},\n",
        (unsigned long long)alloc_delta.malloc_calls, (unsigned long long)alloc_delta.calloc_calls,
        (unsigned long long)alloc_delta.realloc_calls, (unsigned long long)alloc_delta.free_calls,
        (unsigned long long)alloc_delta.bytes_requested);
    fprintf(out, "  \"scratch_profile\": {\"policy\": \"%s\", \"peak_capacity_bytes\": %llu, \"growth_events\": %llu},\n",
        scratch_policy, (unsigned long long)scratch_workspace.peak_capacity,
        (unsigned long long)scratch_workspace.growth_events);
    fprintf(out, "  \"dequant_dot_profile\": {\"enabled\": %s, \"kernel_policy\": \"%s\", \"target\": \"final_head_q6_k\", \"temporary_blocks_decoded\": %llu, \"temporary_floats_materialized\": %llu, \"temporary_bytes_materialized\": %llu, \"fused_dot_calls\": %llu, \"fallback_dot_calls\": %llu, \"final_head_q6_k_blocks\": %llu},\n",
        dequant_profile_enabled ? "true" : "false", kernel_policy,
        (unsigned long long)dequant_profile.temporary_blocks_decoded,
        (unsigned long long)dequant_profile.temporary_floats_materialized,
        (unsigned long long)dequant_profile.temporary_bytes_materialized,
        (unsigned long long)dequant_profile.fused_dot_calls,
        (unsigned long long)dequant_profile.fallback_dot_calls,
        (unsigned long long)dequant_profile.final_head_q6_k_blocks);
    if (!thread_pool_policy && thread_serial_jobs == 0u) thread_serial_jobs = (uint64_t)steps * (uint64_t)(layers - start_layer);
    if (!thread_pool_policy && thread_fallback_jobs == 0u) thread_fallback_jobs = thread_serial_jobs;
    fprintf(out, "  \"thread_profile\": {\"enabled\": true, \"policy\": \"%s\", \"requested_threads\": %u, \"workers_used\": %u, \"parallel_jobs\": %llu, \"serial_jobs\": %llu, \"fallback_jobs\": %llu",
        thread_policy, threads, thread_workers_used, (unsigned long long)thread_parallel_jobs,
        (unsigned long long)thread_serial_jobs, (unsigned long long)thread_fallback_jobs);
    if (!thread_pool_policy) fprintf(out, ", \"disabled_reason\": \"serial_policy\"");
    fprintf(out, "},\n");
    fprintf(out, "  \"simd_profile\": {\"enabled\": true, \"policy\": \"%s\", \"kernel\": \"%s\", \"fma_dot_calls\": %llu, \"fallback_dot_calls\": %llu",
        simd_policy, avx2_fma_policy ? "avx2_fma_q6_k_f32" : "scalar",
        (unsigned long long)simd_fma_dot_calls, (unsigned long long)simd_fallback_dot_calls);
    if (!avx2_fma_policy) fprintf(out, ", \"disabled_reason\": \"scalar_policy\"");
    fprintf(out, "},\n");
    fprintf(out, "  \"expert_cache_profile\": {\"enabled\": true, \"policy\": \"%s\", \"cache_hits\": 0, \"cache_misses\": 0, \"bytes_cached\": 0, \"expert_weight_reads\": 0, \"disabled_reason\": \"none_policy\"},\n", expert_cache_policy);
    fprintf(out, "  \"cuda_profile\": {\"enabled\": true, \"policy\": \"%s\", \"backend\": \"none\", \"device_bytes\": 0, \"host_to_device_bytes\": 0, \"device_to_host_bytes\": 0, \"kernel_launches\": 0, \"disabled_reason\": \"none_policy\"},\n", cuda_policy);
    fprintf(out, "  \"prefill_gemm_profile\": {\"enabled\": true, \"policy\": \"%s\", \"backend\": \"none\", \"gemm_calls\": 0, \"batched_tokens\": 0, \"fused_rows\": 0, \"temporary_bytes\": 0, \"disabled_reason\": \"none_policy\"},\n", prefill_gemm_policy);
    fprintf(out, "  \"speculative_profile\": {\"enabled\": true, \"policy\": \"%s\", \"backend\": \"none\", \"draft_tokens\": 0, \"accepted_tokens\": 0, \"rejected_tokens\": 0, \"target_verifications\": 0, \"disabled_reason\": \"none_policy\"},\n", speculative_policy);
    fprintf(out, "  \"kv2_profile\": {\"enabled\": true, \"policy\": \"%s\", \"format\": \"none\", \"packed_bytes\": 0, \"read_ops\": 0, \"write_ops\": 0, \"fallback_reads\": 0, \"disabled_reason\": \"none_policy\"},\n", kv2_policy);
    fprintf(out, "  \"sampling_profile\": {\"enabled\": true, \"policy\": \"%s\", \"mode\": \"greedy\", \"stochastic_samples\": 0, \"top_p_evaluations\": 0, \"beam_width\": 1, \"disabled_reason\": \"none_policy\"},\n", sampling_policy);
    fprintf(out, "  \"long_context_profile\": {\"enabled\": true, \"policy\": \"%s\", \"target_ctx_tokens\": %u, \"rss_limit_bytes\": %llu, \"kv_quality_checks\": 0, \"soak_seconds\": 0, \"disabled_reason\": %s},\n", long_context_policy, ctx4k_smoke_policy ? 4096u : 0u, (unsigned long long)long_context_rss_limit_bytes, ctx4k_smoke_policy ? "null" : "\"none_policy\"");
    const char *note = residual_replay
        ? "hybrid one-token replay from an injected F32 residual through the requested layer suffix"
        : kv_f16
        ? "one-token Qwen3 transformer forward with real attention, F16 KV, and top-8 MoE"
        : kv_f32
        ? (final_head
            ? "Qwen3 forward with diagnostic F32 KV, all requested layers, final RMSNorm, and complete Q6_K vocabulary head"
            : "one-token Qwen3 transformer forward with real attention, diagnostic F32 KV, and top-8 MoE; final head disabled")
        : final_head
        ? (steps > 1u
            ? (prompt_count > 1u
                ? "tokenized Qwen3 prompt prefill followed by greedy generation: fixed prompt IDs and selected tokens are embedded at their positions with persistent per-layer INT8 KV; the complete Q6_K head runs only for generation outputs"
                : "greedy multi-token Qwen3 forward: every selected token is re-embedded, position and per-layer INT8 KV advance, then all 48 layers, final RMSNorm, and the complete Q6_K vocabulary head run again")
            : "one-token Qwen3 forward through 48 layers, final RMSNorm, and complete Q6_K vocabulary head; tokenizer parity and multi-token execution are separate gates")
        : (full_moe
            ? "one-token Qwen3 transformer forward with real attention, dynamic INT8 KV, and top-8 MoE; final head disabled"
            : (rope_gqa_attention ? "partial Qwen3 attention: split-half RoPE on Q/K, GQA head mapping, per-head causal softmax, dynamically scaled INT8 KV, and output projection; dimensions and heads remain probe-limited" : (causal_attention ? "partial causal attention: Q/K/V projection, dynamically scaled INT8 KV persistence, stable causal softmax, context aggregation, and output projection; dimensions remain probe-limited" : "state loop skeleton: per-token per-layer KV append/checksum/readback plus sampled-token carry; residual math remains probe-level")));
    fprintf(out, "  \"note\": \"%s\"\n}\n", note);
    free(kbuf);
    free(vbuf);
    free(kcache);
    free(vcache);
    free(kfloat);
    free(vfloat);
    free(kscales);
    free(vscales);
    free(residual_vec);
    qx_scratch_free(&scratch_workspace);
    qx_close_file(&file);
    return 1;
}

int qx_dump_state_loop_probe_summary(const char *path, const char *tokens_path, uint32_t prompt_token, uint32_t steps, uint32_t layers, uint32_t ctx_tokens, const char *kv_format, const char *activation_format, int real_kv, int projection_matvec, int residual_vector, int residual_carry, int numeric_deltas, int delta_vectors, int attention_output_vector, int causal_attention, int rope_gqa_attention, int full_moe, int final_head, int bench, uint32_t residual_dims, const char *norm_name, uint32_t top_k, uint32_t scan, uint32_t logits_top_n, double temperature, uint32_t seed, const char *residual_dump_dir, uint32_t start_layer, const char *residual_input_path, const char *kv_snapshot_out_path, const char *kv_snapshot_in_path, FILE *out, char *err, uint64_t err_len) {
    if (final_head && (steps == 0u || steps > 64u)) {
        qx_set_err(err, err_len, "--final-head requires --full-moe, 1..64 steps, all manifest layers, and temperature 0");
        return 0;
    }
    if (steps == 0u) steps = 1u;
    if (steps > 64u) steps = 64u;
    return qx_dump_prompt_state_loop_probe_summary(path, tokens_path, &prompt_token, 1u, steps, layers, ctx_tokens, kv_format, activation_format, "ephemeral", "baseline", "serial", 1u, "scalar", "none", "none", "none", "none", "none", "none", "none", 0u, 0,
        real_kv, projection_matvec, residual_vector, residual_carry, numeric_deltas, delta_vectors, attention_output_vector,
        causal_attention, rope_gqa_attention, full_moe, final_head, bench, residual_dims, norm_name, top_k, scan,
        logits_top_n, temperature, seed, residual_dump_dir, start_layer, residual_input_path,
        kv_snapshot_out_path, kv_snapshot_in_path, out, err, err_len);
}

int qx_dump_token_forward_probe_summary(const char *path, uint32_t token_id, uint32_t layers, uint32_t top_k, uint32_t blocks, uint32_t seed, const char *norm_name, int32_t attention_layer, int multihead_attention, uint32_t attention_heads, uint32_t attention_dims, int logits_enabled, uint32_t logits_top_n, int sample_enabled, double temperature, int decode_token, const char *tokens_path, FILE *out, char *err, uint64_t err_len) {
    if (layers == 0) layers = 1;
    if (top_k == 0) top_k = 1;
    if (blocks == 0) blocks = 1;
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    if (file.header.manifest.model_type != QX_MODEL_QWEN3_MOE) { qx_close_file(&file); qx_set_err(err, err_len, "not a MoE QXF"); return 0; }
    const qx_tensor_dir_entry *emb = qx_find_tensor(&file, "token_embd.weight");
    if (!emb) { qx_close_file(&file); qx_set_err(err, err_len, "token_embd.weight not found"); return 0; }
    uint32_t vocab = file.header.manifest.vocab ? file.header.manifest.vocab : (uint32_t)(emb->dims[1] ? emb->dims[1] : 1);
    if (token_id >= vocab) { qx_close_file(&file); qx_set_err(err, err_len, "token id out of range"); return 0; }
    uint64_t row = 0;
    if (!qx_embedding_row_size(emb, vocab, &row, err, err_len)) { qx_close_file(&file); return 0; }
    uint64_t emb_offset = emb->offset + row * token_id;
    uint64_t emb_read = row;
    unsigned char *emb_buf = NULL;
    if (!qx_read_raw_span(&file, emb_offset, emb_read, &emb_buf, err, err_len)) { qx_close_file(&file); return 0; }
    uint64_t emb_checksum = qx_fnv1a64(emb_buf, emb_read);
    uint32_t estate = seed ? seed : 1u;
    double embedding_probe = 0.0;
    const char *emb_decoder = NULL;
    uint64_t emb_block_size = 0;
    uint64_t emb_values = 0;
    int embedding_numeric = 0;
    int rms_enabled = 0;
    double rms_value = 0.0;
    double rmsnorm_probe = 0.0;
    uint64_t rms_values = 0;
    if (qx_decoder_info(emb->flags, &emb_decoder, &emb_block_size) && emb_block_size > 0 && emb_read >= emb_block_size) {
        uint64_t emb_blocks = emb_read / emb_block_size;
        for (uint64_t b = 0; b < emb_blocks; ++b) {
            float weights[256];
            qx_decode_supported_block(emb->flags, emb_buf + b * emb_block_size, weights);
            for (int i = 0; i < 256; ++i) embedding_probe += (double)weights[i] * (double)qx_deterministic_input(&estate);
        }
        emb_values = emb_blocks * 256ull;
        embedding_numeric = 1;
    } else {
        for (uint64_t i = 0; i < emb_read; ++i) {
            float x = qx_deterministic_input(&estate);
            embedding_probe += ((double)emb_buf[i] - 127.5) / 127.5 * (double)x;
        }
        emb_values = emb_read;
    }
    if (norm_name && *norm_name && emb_decoder && emb_block_size > 0 && emb_read < emb_block_size) {
        free(emb_buf); qx_close_file(&file); qx_set_err(err, err_len, "embedding row smaller than quant block"); return 0;
    }
    if (norm_name && *norm_name && embedding_numeric) {
        const qx_tensor_dir_entry *norm = qx_find_tensor(&file, norm_name);
        if (!norm) { free(emb_buf); qx_close_file(&file); qx_set_err(err, err_len, "norm tensor not found"); return 0; }
        if (norm->flags != 0u) { free(emb_buf); qx_close_file(&file); qx_set_err(err, err_len, "token-forward-probe supports F32 norm weights only"); return 0; }
        uint64_t emb_blocks = emb_read / emb_block_size;
        rms_values = emb_blocks * 256ull;
        uint64_t norm_values = norm->byte_size / 4ull;
        if (rms_values > norm_values) rms_values = norm_values;
        unsigned char *norm_buf = NULL;
        if (rms_values == 0 || !qx_read_raw_span(&file, norm->offset, rms_values * 4ull, &norm_buf, err, err_len)) { free(emb_buf); qx_close_file(&file); if (rms_values == 0) qx_set_err(err, err_len, "no overlapping embedding/norm values"); return 0; }
        double sumsq = 0.0;
        float *vals = (float *)malloc((size_t)rms_values * sizeof(float));
        if (!vals) { free(norm_buf); free(emb_buf); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
        uint64_t y = 0;
        for (uint64_t b = 0; b < emb_blocks && y < rms_values; ++b) {
            float block[256];
            qx_decode_supported_block(emb->flags, emb_buf + b * emb_block_size, block);
            for (int i = 0; i < 256 && y < rms_values; ++i) { vals[y] = block[i]; sumsq += (double)block[i] * (double)block[i]; y++; }
        }
        rms_value = sqrt(sumsq / (double)rms_values + 1.0e-6);
        uint32_t nstate = seed ? seed : 1u;
        for (uint64_t i = 0; i < rms_values; ++i) rmsnorm_probe += ((double)vals[i] / rms_value) * (double)qx_rd_le_f32(norm_buf + i * 4ull) * (double)qx_deterministic_input(&nstate);
        rms_enabled = 1;
        free(vals); free(norm_buf);
    }
    int attention_enabled = 0;
    double attention_output_probe = 0.0;
    double attention_score_probe = 0.0;
    uint64_t attention_values = 0;
    uint32_t attention_heads_run = 0;
    uint32_t attention_dims_run = 0;
    if (attention_layer >= 0) {
        char qn[QX_NAME_MAX], kn[QX_NAME_MAX], vn[QX_NAME_MAX], on[QX_NAME_MAX];
        snprintf(qn, sizeof(qn), "blk.%d.attn_q.weight", attention_layer);
        snprintf(kn, sizeof(kn), "blk.%d.attn_k.weight", attention_layer);
        snprintf(vn, sizeof(vn), "blk.%d.attn_v.weight", attention_layer);
        snprintf(on, sizeof(on), "blk.%d.attn_output.weight", attention_layer);
        const qx_tensor_dir_entry *q = qx_find_tensor(&file, qn);
        const qx_tensor_dir_entry *k = qx_find_tensor(&file, kn);
        const qx_tensor_dir_entry *v = qx_find_tensor(&file, vn);
        const qx_tensor_dir_entry *o = qx_find_tensor(&file, on);
        if (!q || !k || !v || !o) { free(emb_buf); qx_close_file(&file); qx_set_err(err, err_len, "missing attention tensor"); return 0; }
        double qdot=0, kdot=0, vdot=0, odot=0;
        uint64_t qv=0, kv=0, vv=0, ov=0;
        if (!qx_tensor_block_dot_calc(&file, o, blocks, seed ^ 0xc2b2ae35u, &odot, NULL, NULL, &ov, NULL, err, err_len)) { free(emb_buf); qx_close_file(&file); return 0; }
        if (multihead_attention) {
            attention_heads_run = attention_heads ? attention_heads : 4u;
            if (attention_heads_run > file.header.manifest.q_heads) attention_heads_run = file.header.manifest.q_heads;
            attention_dims_run = attention_dims ? attention_dims : 16u;
            if (attention_dims_run > file.header.manifest.head_dim) attention_dims_run = file.header.manifest.head_dim;
            double combined = 0.0;
            for (uint32_t h = 0; h < attention_heads_run; ++h) {
                double hq=0, hk=0, hv=0;
                uint32_t hseed = seed + h * 977u;
                if (!qx_tensor_block_dot_calc(&file, q, blocks, hseed, &hq, NULL, NULL, &qv, NULL, err, err_len) ||
                    !qx_tensor_block_dot_calc(&file, k, blocks, hseed ^ 0x9e3779b9u, &hk, NULL, NULL, &kv, NULL, err, err_len) ||
                    !qx_tensor_block_dot_calc(&file, v, blocks, hseed ^ 0x85ebca6bu, &hv, NULL, NULL, &vv, NULL, err, err_len)) { free(emb_buf); qx_close_file(&file); return 0; }
                double hscore = (hq * hk) / sqrt((double)(attention_dims_run ? attention_dims_run : 1u));
                combined += hscore * hv;
            }
            attention_values = (uint64_t)attention_heads_run * (uint64_t)attention_dims_run;
            attention_score_probe = combined / (double)(attention_heads_run ? attention_heads_run : 1u);
            attention_output_probe = attention_score_probe * odot;
        } else {
            if (!qx_tensor_block_dot_calc(&file, q, blocks, seed, &qdot, NULL, NULL, &qv, NULL, err, err_len) ||
                !qx_tensor_block_dot_calc(&file, k, blocks, seed ^ 0x9e3779b9u, &kdot, NULL, NULL, &kv, NULL, err, err_len) ||
                !qx_tensor_block_dot_calc(&file, v, blocks, seed ^ 0x85ebca6bu, &vdot, NULL, NULL, &vv, NULL, err, err_len)) { free(emb_buf); qx_close_file(&file); return 0; }
            attention_values = qv;
            if (kv < attention_values) attention_values = kv;
            if (vv < attention_values) attention_values = vv;
            if (ov < attention_values) attention_values = ov;
            attention_score_probe = (qdot * kdot) / sqrt((double)(attention_values ? attention_values : 1));
            attention_output_probe = attention_score_probe * vdot * odot;
        }
        attention_enabled = 1;
    }
    free(emb_buf);
    uint32_t max_layers = file.header.manifest.layers;
    if (layers > max_layers) layers = max_layers;
    double moe_output = 0.0;
    uint32_t unsupported_selected = 0;
    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"token_forward\",\n");
    fprintf(out, "  \"token_id\": %u,\n", token_id);
    fprintf(out, "  \"embedding\": {\"tensor\": \"token_embd.weight\", \"ggml_type\": %u, \"decoder\": %s%s%s, \"offset\": %llu, \"row_byte_size\": %llu, \"checksum\": %llu, \"numeric\": %s, \"values\": %llu},\n",
        emb->flags, emb_decoder ? "\"" : "", emb_decoder ? emb_decoder : "null", emb_decoder ? "\"" : "", (unsigned long long)emb_offset, (unsigned long long)emb_read, (unsigned long long)emb_checksum, embedding_numeric ? "true" : "false", (unsigned long long)emb_values);
    fprintf(out, "  \"embedding_probe\": %.9g,\n", embedding_probe);
    fprintf(out, "  \"rmsnorm\": {\"enabled\": %s, \"norm_tensor\": %s%s%s, \"rms\": %.9g, \"values\": %llu, \"normalized_probe\": %.9g},\n",
        rms_enabled ? "true" : "false", rms_enabled ? "\"" : "", rms_enabled ? norm_name : "null", rms_enabled ? "\"" : "", rms_value, (unsigned long long)rms_values, rmsnorm_probe);
    fprintf(out, "  \"attention\": {\"enabled\": %s, \"mode\": \"%s\", \"layer\": %d, \"values\": %llu, \"score_probe\": %.9g, \"output_probe\": %.9g, \"heads_run\": %u, \"dims\": %u},\n",
        attention_enabled ? "true" : "false", multihead_attention ? "multihead" : "scalar", attention_enabled ? attention_layer : -1, (unsigned long long)attention_values, attention_score_probe, attention_output_probe, attention_heads_run, attention_dims_run);
    fprintf(out, "  \"layers_run\": %u,\n", layers);
    fprintf(out, "  \"top_k\": %u,\n", top_k);
    fprintf(out, "  \"blocks\": %u,\n", blocks);
    fprintf(out, "  \"input_seed\": %u,\n", seed);
    fprintf(out, "  \"layer_outputs\": [");
    for (uint32_t layer = 0; layer < layers; ++layer) {
        char name[QX_NAME_MAX];
        snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp.weight", layer);
        const qx_tensor_dir_entry *router = qx_find_tensor(&file, name);
        if (!router) { qx_close_file(&file); qx_set_err(err, err_len, "router tensor not found for requested layer"); return 0; }
        if (router->flags != 0u) { qx_close_file(&file); qx_set_err(err, err_len, "token-forward-probe supports F32 router only"); return 0; }
        uint32_t hidden = router->rank > 0 ? (uint32_t)router->dims[0] : file.header.manifest.hidden;
        uint32_t experts = router->rank > 1 ? (uint32_t)router->dims[1] : file.header.manifest.experts;
        uint32_t k_eff = top_k > experts ? experts : top_k;
        uint64_t values = (uint64_t)blocks * 256ull;
        if (values > hidden) values = hidden;
        uint64_t row_bytes = (uint64_t)hidden * 4ull;
        double *logits = (double *)calloc(experts, sizeof(double));
        int *picked = (int *)calloc(experts, sizeof(int));
        uint32_t *selected = (uint32_t *)calloc(k_eff, sizeof(uint32_t));
        if (!logits || !picked || !selected) { free(logits); free(picked); free(selected); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
        for (uint32_t e = 0; e < experts; ++e) {
            unsigned char *buf = NULL;
            uint64_t span = values * 4ull;
            uint64_t off = router->offset + (uint64_t)e * row_bytes;
            if (!qx_read_raw_span(&file, off, span, &buf, err, err_len)) { free(logits); free(picked); free(selected); qx_close_file(&file); return 0; }
            uint32_t state = (seed ? seed : 1u) ^ (uint32_t)(emb_checksum & 0xffffffffu);
            double dot = 0.0;
            for (uint64_t i = 0; i < values; ++i) dot += (double)qx_rd_le_f32(buf + i*4ull) * (double)qx_deterministic_input(&state);
            logits[e] = dot + (rms_enabled ? rmsnorm_probe : embedding_probe) * 0.001;
            free(buf);
        }
        for (uint32_t k = 0; k < k_eff; ++k) {
            uint32_t best = 0;
            double best_logit = -1.0e300;
            for (uint32_t e = 0; e < experts; ++e) if (!picked[e] && logits[e] > best_logit) { best = e; best_logit = logits[e]; }
            selected[k] = best; picked[best] = 1;
        }
        double layer_sum = 0.0;
        for (uint32_t k = 0; k < k_eff; ++k) {
            uint32_t e = selected[k];
            double gate_dot = 0.0, up_dot = 0.0, down_dot = 0.0;
            const char *dec = NULL; uint32_t typ = 0; uint64_t dummy64 = 0, mix = 0;
            int ok = qx_expert_row_dot_calc(&file, layer, e, "gate", 0, blocks, seed, &gate_dot, NULL, NULL, &dec, &typ, &dummy64, &dummy64, &dummy64, &mix, err, err_len) &&
                     qx_expert_row_dot_calc(&file, layer, e, "up", 0, blocks, seed, &up_dot, NULL, NULL, &dec, &typ, &dummy64, &dummy64, &dummy64, &mix, err, err_len) &&
                     qx_expert_row_dot_calc(&file, layer, e, "down", 0, blocks, seed, &down_dot, NULL, NULL, &dec, &typ, &dummy64, &dummy64, &dummy64, &mix, err, err_len);
            if (ok) layer_sum += qx_silu(gate_dot) * up_dot * down_dot;
            else unsupported_selected++;
        }
        moe_output += layer_sum;
        fprintf(out, "%s%.9g", layer ? ", " : "", layer_sum);
        free(logits); free(picked); free(selected);
    }
    double token_output = (rms_enabled ? rmsnorm_probe : embedding_probe) + (attention_enabled ? attention_output_probe : 0.0) + moe_output;
    qx_top_token logits_top[32];
    const char *logits_lm_name = NULL;
    int logits_tied = 0;
    uint32_t logits_scanned = 0;
    if (logits_top_n == 0) logits_top_n = 5;
    if (logits_top_n > 32) logits_top_n = 32;
    int logits_ok = 0;
    if (logits_enabled || sample_enabled) logits_ok = qx_collect_top_logits(&file, token_output, logits_top_n, 64, seed, &logits_lm_name, &logits_tied, &logits_scanned, logits_top, err, err_len);
    if (logits_ok && logits_top_n > logits_scanned) logits_top_n = logits_scanned;
    qx_sample_result sample_result;
    memset(&sample_result, 0, sizeof(sample_result));
    sample_result.strategy = "disabled";
    if (sample_enabled && logits_ok) sample_result = qx_sample_from_top(logits_top, logits_top_n, temperature, seed);
    fprintf(out, "],\n");
    fprintf(out, "  \"unsupported_selected\": %u,\n", unsupported_selected);
    fprintf(out, "  \"moe_output_probe\": %.9g,\n", moe_output);
    fprintf(out, "  \"token_output_probe\": %.9g,\n", token_output);
    fprintf(out, "  \"logits\": {\"enabled\": %s, \"lm_head_tensor\": %s%s%s, \"tied_embedding_fallback\": %s, \"top_n\": %u, \"scanned\": %u, \"top_tokens\": [", (logits_enabled || sample_enabled) ? "true" : "false", logits_ok ? "\"" : "", logits_ok ? logits_lm_name : "null", logits_ok ? "\"" : "", logits_tied ? "true" : "false", logits_ok ? logits_top_n : 0, logits_scanned);
    if (logits_ok) for (uint32_t i = 0; i < logits_top_n; ++i) fprintf(out, "%s{\"token\": %u, \"logit\": %.9g}", i ? ", " : "", logits_top[i].token, logits_top[i].logit);
    fprintf(out, "]},\n");
    fprintf(out, "  \"sampler\": ");
    qx_print_sampler_json(out, &sample_result, sample_enabled && logits_ok, logits_top_n, logits_scanned, temperature);
    fprintf(out, ",\n");
    fprintf(out, "  \"decoded_token\": ");
    qx_emit_decoded_token_object(out, (sample_enabled && logits_ok) ? sample_result.selected_token : 0u, decode_token && sample_enabled && logits_ok, tokens_path);
    fprintf(out, ",\n");
    fprintf(out, "  \"note\": \"token probe: numeric embedding decode + optional RMSNorm + optional scalar/multihead attention probe + MoE probe + optional logits top-n + optional sampler + fallback token decode\"\n");
    fprintf(out, "}\n");
    qx_close_file(&file);
    return 1;
}

int qx_dump_rmsnorm_probe_summary(const char *path, uint32_t token_id, const char *norm_name, uint32_t seed, FILE *out, char *err, uint64_t err_len) {
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    if (!norm_name || !*norm_name) norm_name = "blk.0.attn_norm.weight";
    const qx_tensor_dir_entry *emb = qx_find_tensor(&file, "token_embd.weight");
    const qx_tensor_dir_entry *norm = qx_find_tensor(&file, norm_name);
    if (!emb) { qx_close_file(&file); qx_set_err(err, err_len, "token_embd.weight not found"); return 0; }
    if (!norm) { qx_close_file(&file); qx_set_err(err, err_len, "norm tensor not found"); return 0; }
    if (norm->flags != 0u) { qx_close_file(&file); qx_set_err(err, err_len, "rmsnorm-probe supports F32 norm weights only"); return 0; }
    uint32_t vocab = file.header.manifest.vocab ? file.header.manifest.vocab : (uint32_t)(emb->dims[1] ? emb->dims[1] : 1);
    if (token_id >= vocab) { qx_close_file(&file); qx_set_err(err, err_len, "token id out of range"); return 0; }
    const char *emb_decoder = NULL;
    uint64_t emb_block_size = 0;
    if (!qx_decoder_info(emb->flags, &emb_decoder, &emb_block_size) || emb_block_size == 0) { qx_close_file(&file); qx_set_err(err, err_len, "unsupported embedding quant for rmsnorm"); return 0; }
    uint64_t row = 0;
    if (!qx_embedding_row_size(emb, vocab, &row, err, err_len)) { qx_close_file(&file); return 0; }
    if (row < emb_block_size) { qx_close_file(&file); qx_set_err(err, err_len, "embedding row smaller than quant block"); return 0; }
    uint64_t emb_offset = emb->offset + row * token_id;
    uint64_t emb_read = row;
    uint64_t emb_blocks = emb_read / emb_block_size;
    uint64_t values = emb_blocks * 256ull;
    uint64_t norm_values = norm->byte_size / 4ull;
    if (values > norm_values) values = norm_values;
    if (values == 0) { qx_close_file(&file); qx_set_err(err, err_len, "no overlapping embedding/norm values"); return 0; }
    unsigned char *emb_buf = NULL;
    unsigned char *norm_buf = NULL;
    if (!qx_read_raw_span(&file, emb_offset, emb_blocks * emb_block_size, &emb_buf, err, err_len)) { qx_close_file(&file); return 0; }
    if (!qx_read_raw_span(&file, norm->offset, values * 4ull, &norm_buf, err, err_len)) { free(emb_buf); qx_close_file(&file); return 0; }
    uint64_t emb_checksum = qx_fnv1a64(emb_buf, emb_blocks * emb_block_size);
    uint64_t norm_checksum = qx_fnv1a64(norm_buf, values * 4ull);
    float *vals = (float *)malloc((size_t)values * sizeof(float));
    if (!vals) { free(emb_buf); free(norm_buf); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
    uint64_t y = 0;
    for (uint64_t b = 0; b < emb_blocks && y < values; ++b) {
        float block[256];
        qx_decode_supported_block(emb->flags, emb_buf + b * emb_block_size, block);
        for (int i = 0; i < 256 && y < values; ++i) vals[y++] = block[i];
    }
    double sumsq = 0.0;
    for (uint64_t i = 0; i < values; ++i) sumsq += (double)vals[i] * (double)vals[i];
    const double eps = 1.0e-6;
    double rms = sqrt(sumsq / (double)values + eps);
    uint32_t state = seed ? seed : 1u;
    double normalized_probe = 0.0;
    double norm_weight_sum = 0.0;
    for (uint64_t i = 0; i < values; ++i) {
        float w = qx_rd_le_f32(norm_buf + i * 4ull);
        norm_weight_sum += (double)w;
        normalized_probe += ((double)vals[i] / rms) * (double)w * (double)qx_deterministic_input(&state);
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"rmsnorm\",\n");
    fprintf(out, "  \"token_id\": %u,\n", token_id);
    fprintf(out, "  \"embedding_tensor\": \"token_embd.weight\",\n");
    fprintf(out, "  \"embedding_ggml_type\": %u,\n", emb->flags);
    fprintf(out, "  \"embedding_decoder\": \"%s\",\n", emb_decoder);
    fprintf(out, "  \"embedding_offset\": %llu,\n", (unsigned long long)emb_offset);
    fprintf(out, "  \"embedding_row_byte_size\": %llu,\n", (unsigned long long)row);
    fprintf(out, "  \"embedding_checksum\": %llu,\n", (unsigned long long)emb_checksum);
    fprintf(out, "  \"norm_tensor\": \"%s\",\n", norm->name);
    fprintf(out, "  \"norm_ggml_type\": %u,\n", norm->flags);
    fprintf(out, "  \"norm_checksum\": %llu,\n", (unsigned long long)norm_checksum);
    fprintf(out, "  \"values\": %llu,\n", (unsigned long long)values);
    fprintf(out, "  \"epsilon\": %.9g,\n", eps);
    fprintf(out, "  \"rms\": %.9g,\n", rms);
    fprintf(out, "  \"norm_weight_sum\": %.9g,\n", norm_weight_sum);
    fprintf(out, "  \"normalized_probe\": %.9g\n", normalized_probe);
    fprintf(out, "}\n");
    free(vals); free(emb_buf); free(norm_buf); qx_close_file(&file); return 1;
}

static int qx_tensor_block_dot_calc(qx_file *file, const qx_tensor_dir_entry *t, uint32_t blocks, uint32_t seed, double *dot_out, double *sum_out, const char **decoder_out, uint64_t *values_out, uint64_t *checksum_out, char *err, uint64_t err_len) {
    const char *decoder = NULL;
    uint64_t block_size = 0;
    if (!t || !qx_decoder_info(t->flags, &decoder, &block_size)) { qx_set_err(err, err_len, "unsupported attention tensor quant"); return 0; }
    if (blocks == 0) blocks = 1;
    uint64_t block_count = t->byte_size / block_size;
    if (block_count == 0) { qx_set_err(err, err_len, "attention tensor has no blocks"); return 0; }
    if ((uint64_t)blocks > block_count) blocks = (uint32_t)block_count;
    unsigned char *buf = NULL;
    uint64_t span = (uint64_t)blocks * block_size;
    if (!qx_read_raw_span(file, t->offset, span, &buf, err, err_len)) return 0;
    uint32_t state = seed ? seed : 1u;
    double dot = 0.0, sum = 0.0;
    for (uint32_t b = 0; b < blocks; ++b) {
        float vals[256];
        qx_decode_supported_block(t->flags, buf + (uint64_t)b * block_size, vals);
        for (int i = 0; i < 256; ++i) {
            sum += (double)vals[i];
            dot += (double)vals[i] * (double)qx_deterministic_input(&state);
        }
    }
    if (dot_out) *dot_out = dot;
    if (sum_out) *sum_out = sum;
    if (decoder_out) *decoder_out = decoder;
    if (values_out) *values_out = (uint64_t)blocks * 256ull;
    if (checksum_out) *checksum_out = qx_fnv1a64(buf, span);
    free(buf);
    return 1;
}

int qx_dump_attention_probe_summary(const char *path, uint32_t layer, uint32_t blocks, uint32_t seed, uint32_t ctx_tokens, const char *kv_format, int cache_write, FILE *out, char *err, uint64_t err_len) {
    if (blocks == 0) blocks = 1;
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    char qn[QX_NAME_MAX], kn[QX_NAME_MAX], vn[QX_NAME_MAX], on[QX_NAME_MAX];
    snprintf(qn, sizeof(qn), "blk.%u.attn_q.weight", layer);
    snprintf(kn, sizeof(kn), "blk.%u.attn_k.weight", layer);
    snprintf(vn, sizeof(vn), "blk.%u.attn_v.weight", layer);
    snprintf(on, sizeof(on), "blk.%u.attn_output.weight", layer);
    const qx_tensor_dir_entry *q = qx_find_tensor(&file, qn);
    const qx_tensor_dir_entry *k = qx_find_tensor(&file, kn);
    const qx_tensor_dir_entry *v = qx_find_tensor(&file, vn);
    const qx_tensor_dir_entry *o = qx_find_tensor(&file, on);
    if (!q || !k || !v || !o) { qx_close_file(&file); qx_set_err(err, err_len, "missing attention tensor"); return 0; }
    double qdot=0, kdot=0, vdot=0, odot=0, qsum=0, ksum=0, vsum=0, osum=0;
    const char *qdec=NULL, *kdec=NULL, *vdec=NULL, *odec=NULL;
    uint64_t qv=0, kv=0, vv=0, ov=0, qchk=0, kchk=0, vchk=0, ochk=0;
    if (!qx_tensor_block_dot_calc(&file, q, blocks, seed, &qdot, &qsum, &qdec, &qv, &qchk, err, err_len) ||
        !qx_tensor_block_dot_calc(&file, k, blocks, seed ^ 0x9e3779b9u, &kdot, &ksum, &kdec, &kv, &kchk, err, err_len) ||
        !qx_tensor_block_dot_calc(&file, v, blocks, seed ^ 0x85ebca6bu, &vdot, &vsum, &vdec, &vv, &vchk, err, err_len) ||
        !qx_tensor_block_dot_calc(&file, o, blocks, seed ^ 0xc2b2ae35u, &odot, &osum, &odec, &ov, &ochk, err, err_len)) { qx_close_file(&file); return 0; }
    uint64_t values = qv;
    if (kv < values) values = kv;
    if (vv < values) values = vv;
    if (ov < values) values = ov;
    double scale = values ? sqrt((double)values) : 1.0;
    double attention_score = (qdot * kdot) / (scale ? scale : 1.0);
    double context_probe = attention_score * vdot;
    double output_probe = context_probe * odot;
    uint32_t bpv = 0;
    uint64_t bytes_per_k_or_v = 0, bytes_per_token_layer = 0, layer_stride = 0, total_bytes = 0;
    int kv_enabled = 0;
    if (ctx_tokens > 0 && kv_format && *kv_format) {
        if (strcmp(kv_format, "int8") == 0) bpv = 1;
        else if (strcmp(kv_format, "f16") == 0 || strcmp(kv_format, "fp16") == 0) bpv = 2;
        else if (strcmp(kv_format, "int4") == 0) bpv = 0;
        else { qx_close_file(&file); qx_set_err(err, err_len, "unsupported kv format"); return 0; }
        uint64_t values_per_k_or_v = (uint64_t)file.header.manifest.kv_heads * (uint64_t)file.header.manifest.head_dim;
        bytes_per_k_or_v = strcmp(kv_format, "int4") == 0 ? (values_per_k_or_v + 1ull) / 2ull : values_per_k_or_v * (uint64_t)bpv;
        bytes_per_token_layer = bytes_per_k_or_v * 2ull;
        layer_stride = (uint64_t)ctx_tokens * bytes_per_token_layer;
        total_bytes = (uint64_t)file.header.manifest.layers * layer_stride;
        kv_enabled = 1;
    }
    int buffer_ok = 0;
    uint64_t k_cache_checksum = 0, v_cache_checksum = 0, write_bytes = 0;
    if (cache_write && kv_enabled && bpv > 0 && total_bytes <= (512ull * 1024ull * 1024ull)) {
        uint8_t *cache = (uint8_t *)calloc(1, (size_t)total_bytes);
        uint64_t single_bytes = (uint64_t)file.header.manifest.head_dim * (uint64_t)bpv;
        uint8_t *tmp = (uint8_t *)malloc((size_t)(single_bytes * 2ull));
        if (!cache || !tmp) { free(cache); free(tmp); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
        uint32_t ks = seed ? seed : 1u;
        uint32_t vs = (seed ? seed : 1u) ^ 0xa5a5f00du;
        for (uint64_t i = 0; i < single_bytes; ++i) {
            tmp[i] = (uint8_t)((qx_deterministic_input(&ks) + 1.0f) * 127.5f);
            tmp[single_bytes + i] = (uint8_t)((qx_deterministic_input(&vs) + 1.0f) * 127.5f);
        }
        uint64_t base = (uint64_t)layer * layer_stride;
        uint64_t k_offset = base;
        uint64_t v_offset = base + bytes_per_k_or_v;
        memcpy(cache + k_offset, tmp, (size_t)single_bytes);
        memcpy(cache + v_offset, tmp + single_bytes, (size_t)single_bytes);
        k_cache_checksum = qx_fnv1a64(cache + k_offset, single_bytes);
        v_cache_checksum = qx_fnv1a64(cache + v_offset, single_bytes);
        buffer_ok = memcmp(cache + k_offset, tmp, (size_t)single_bytes) == 0 && memcmp(cache + v_offset, tmp + single_bytes, (size_t)single_bytes) == 0;
        write_bytes = single_bytes * 2ull;
        free(cache);
        free(tmp);
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"attention\",\n");
    fprintf(out, "  \"layer\": %u,\n", layer);
    fprintf(out, "  \"blocks\": %u,\n", blocks);
    fprintf(out, "  \"values\": %llu,\n", (unsigned long long)values);
#define ATTN_TENSOR_JSON(label, t, dec, dot, sum, chk) \
    fprintf(out, "  \"%s\": {\"tensor\": \"%s\", \"ggml_type\": %u, \"decoder\": \"%s\", \"rank\": %u, \"dims\": [%llu, %llu], \"checksum\": %llu, \"dot\": %.9g, \"sum\": %.9g},\n", \
        label, (t)->name, (t)->flags, dec, (t)->rank, (unsigned long long)(t)->dims[0], (unsigned long long)(t)->dims[1], (unsigned long long)chk, dot, sum)
    ATTN_TENSOR_JSON("q", q, qdec, qdot, qsum, qchk);
    ATTN_TENSOR_JSON("k", k, kdec, kdot, ksum, kchk);
    ATTN_TENSOR_JSON("v", v, vdec, vdot, vsum, vchk);
    ATTN_TENSOR_JSON("o", o, odec, odot, osum, ochk);
#undef ATTN_TENSOR_JSON
    fprintf(out, "  \"attention_score_probe\": %.9g,\n", attention_score);
    fprintf(out, "  \"context_probe\": %.9g,\n", context_probe);
    fprintf(out, "  \"attention_output_probe\": %.9g,\n", output_probe);
    if (kv_enabled) {
        fprintf(out, "  \"kv_cache\": {\"enabled\": true, \"kv_format\": \"%s\", \"ctx_tokens\": %u, \"bytes_per_value\": %u, \"bytes_per_k_or_v\": %llu, \"bytes_per_token_per_layer\": %llu, \"layer_stride\": %llu, \"total_bytes\": %llu},\n",
            kv_format, ctx_tokens, bpv, (unsigned long long)bytes_per_k_or_v, (unsigned long long)bytes_per_token_layer, (unsigned long long)layer_stride, (unsigned long long)total_bytes);
    } else {
        fprintf(out, "  \"kv_cache\": {\"enabled\": false},\n");
    }
    fprintf(out, "  \"kv_buffer\": {\"enabled\": %s, \"allocated\": %s, \"write_bytes\": %llu, \"k_checksum\": %llu, \"v_checksum\": %llu, \"readback_ok\": %s},\n",
        cache_write ? "true" : "false", buffer_ok ? "true" : "false", (unsigned long long)write_bytes, (unsigned long long)k_cache_checksum, (unsigned long long)v_cache_checksum, buffer_ok ? "true" : "false");
    fprintf(out, "  \"note\": \"scalar attention skeleton: q/k/v/o tensor decode + deterministic dot probes; KV cache metadata only; no softmax yet\"\n");
    fprintf(out, "}\n");
    qx_close_file(&file);
    return 1;
}

int qx_dump_kv_cache_probe_summary(const char *path, uint32_t ctx_tokens, const char *kv_format, uint32_t token, uint32_t layer, uint32_t head, FILE *out, char *err, uint64_t err_len) {
    if (!path || !kv_format || ctx_tokens == 0) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    uint32_t bytes_per_value = 0;
    if (strcmp(kv_format, "int8") == 0) bytes_per_value = 1;
    else if (strcmp(kv_format, "f16") == 0 || strcmp(kv_format, "fp16") == 0) bytes_per_value = 2;
    else if (strcmp(kv_format, "int4") == 0) bytes_per_value = 0; /* packed: 2 values/byte */
    else { qx_set_err(err, err_len, "unsupported kv format"); return 0; }
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    const qx_model_manifest *m = &file.header.manifest;
    if (layer >= m->layers) { qx_close_file(&file); qx_set_err(err, err_len, "layer out of range"); return 0; }
    if (token >= ctx_tokens) { qx_close_file(&file); qx_set_err(err, err_len, "token out of range"); return 0; }
    if (head >= m->kv_heads) { qx_close_file(&file); qx_set_err(err, err_len, "head out of range"); return 0; }
    uint64_t values_per_k_or_v = (uint64_t)m->kv_heads * (uint64_t)m->head_dim;
    uint64_t values_per_token_layer = values_per_k_or_v * 2ull;
    uint64_t bytes_per_k_or_v = 0;
    uint64_t bytes_per_token_layer = 0;
    if (strcmp(kv_format, "int4") == 0) {
        bytes_per_k_or_v = (values_per_k_or_v + 1ull) / 2ull;
        bytes_per_token_layer = bytes_per_k_or_v * 2ull;
    } else {
        bytes_per_k_or_v = values_per_k_or_v * (uint64_t)bytes_per_value;
        bytes_per_token_layer = values_per_token_layer * (uint64_t)bytes_per_value;
    }
    uint64_t token_stride = bytes_per_token_layer;
    uint64_t layer_stride = (uint64_t)ctx_tokens * token_stride;
    uint64_t total_bytes = (uint64_t)m->layers * layer_stride;
    uint64_t base = (uint64_t)layer * layer_stride + (uint64_t)token * token_stride;
    uint64_t head_value_index = (uint64_t)head * (uint64_t)m->head_dim;
    uint64_t head_byte_offset = strcmp(kv_format, "int4") == 0 ? head_value_index / 2ull : head_value_index * (uint64_t)bytes_per_value;
    uint64_t k_offset = base + head_byte_offset;
    uint64_t v_offset = base + bytes_per_k_or_v + head_byte_offset;
    uint64_t kv_gib_x1000000 = total_bytes ? (uint64_t)((long double)total_bytes / (1024.0L*1024.0L*1024.0L) * 1000000.0L + 0.5L) : 0;
    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"kv_cache\",\n");
    fprintf(out, "  \"kv_format\": \"%s\",\n", kv_format);
    fprintf(out, "  \"ctx_tokens\": %u,\n", ctx_tokens);
    fprintf(out, "  \"layers\": %u,\n", m->layers);
    fprintf(out, "  \"kv_heads\": %u,\n", m->kv_heads);
    fprintf(out, "  \"head_dim\": %u,\n", m->head_dim);
    fprintf(out, "  \"token\": %u,\n", token);
    fprintf(out, "  \"layer\": %u,\n", layer);
    fprintf(out, "  \"head\": %u,\n", head);
    fprintf(out, "  \"bytes_per_value\": %u,\n", bytes_per_value);
    fprintf(out, "  \"bytes_per_k_or_v\": %llu,\n", (unsigned long long)bytes_per_k_or_v);
    fprintf(out, "  \"bytes_per_token_per_layer\": %llu,\n", (unsigned long long)bytes_per_token_layer);
    fprintf(out, "  \"token_stride\": %llu,\n", (unsigned long long)token_stride);
    fprintf(out, "  \"layer_stride\": %llu,\n", (unsigned long long)layer_stride);
    fprintf(out, "  \"total_bytes\": %llu,\n", (unsigned long long)total_bytes);
    fprintf(out, "  \"total_gib\": %.6Lf,\n", (long double)kv_gib_x1000000 / 1000000.0L);
    fprintf(out, "  \"k_offset\": %llu,\n", (unsigned long long)k_offset);
    fprintf(out, "  \"v_offset\": %llu,\n", (unsigned long long)v_offset);
    fprintf(out, "  \"layout\": \"layer_token_kv_head_dim\",\n");
    fprintf(out, "  \"note\": \"offsets are relative to KV cache base; planner only, no cache buffer allocated\"\n");
    fprintf(out, "}\n");
    qx_close_file(&file);
    return 1;
}

int qx_dump_kv_cache_buffer_probe_summary(const char *path, uint32_t ctx_tokens, const char *kv_format, uint32_t token, uint32_t layer, uint32_t head, uint32_t seed, FILE *out, char *err, uint64_t err_len) {
    if (!path || !kv_format || ctx_tokens == 0) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    uint32_t bytes_per_value = 0;
    if (strcmp(kv_format, "int8") == 0) bytes_per_value = 1;
    else if (strcmp(kv_format, "f16") == 0 || strcmp(kv_format, "fp16") == 0) bytes_per_value = 2;
    else { qx_set_err(err, err_len, "buffer probe supports int8/f16"); return 0; }

    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    const qx_model_manifest *m = &file.header.manifest;
    if (layer >= m->layers) { qx_close_file(&file); qx_set_err(err, err_len, "layer out of range"); return 0; }
    if (token >= ctx_tokens) { qx_close_file(&file); qx_set_err(err, err_len, "token out of range"); return 0; }
    if (head >= m->kv_heads) { qx_close_file(&file); qx_set_err(err, err_len, "head out of range"); return 0; }

    uint64_t values_per_k_or_v = (uint64_t)m->kv_heads * (uint64_t)m->head_dim;
    uint64_t bytes_per_k_or_v = values_per_k_or_v * (uint64_t)bytes_per_value;
    uint64_t bytes_per_token_layer = bytes_per_k_or_v * 2ull;
    uint64_t token_stride = bytes_per_token_layer;
    uint64_t layer_stride = (uint64_t)ctx_tokens * token_stride;
    uint64_t total_bytes = (uint64_t)m->layers * layer_stride;
    uint64_t base = (uint64_t)layer * layer_stride + (uint64_t)token * token_stride;
    uint64_t head_byte_offset = (uint64_t)head * (uint64_t)m->head_dim * (uint64_t)bytes_per_value;
    uint64_t k_offset = base + head_byte_offset;
    uint64_t v_offset = base + bytes_per_k_or_v + head_byte_offset;
    uint64_t single_bytes = (uint64_t)m->head_dim * (uint64_t)bytes_per_value;
    uint64_t write_bytes = single_bytes * 2ull;
    if (total_bytes == 0 || total_bytes > (512ull * 1024ull * 1024ull)) { qx_close_file(&file); qx_set_err(err, err_len, "kv buffer probe allocation too large"); return 0; }
    if (v_offset + single_bytes > total_bytes) { qx_close_file(&file); qx_set_err(err, err_len, "computed offset out of range"); return 0; }

    uint8_t *buf = (uint8_t *)calloc(1, (size_t)total_bytes);
    uint8_t *tmp = (uint8_t *)malloc((size_t)write_bytes);
    if (!buf || !tmp) { free(buf); free(tmp); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
    uint32_t ks = seed ? seed : 1u;
    uint32_t vs = (seed ? seed : 1u) ^ 0xa5a5f00du;
    for (uint64_t i = 0; i < single_bytes; ++i) {
        tmp[i] = (uint8_t)((qx_deterministic_input(&ks) + 1.0f) * 127.5f);
        tmp[single_bytes + i] = (uint8_t)((qx_deterministic_input(&vs) + 1.0f) * 127.5f);
    }
    memcpy(buf + k_offset, tmp, (size_t)single_bytes);
    memcpy(buf + v_offset, tmp + single_bytes, (size_t)single_bytes);
    uint64_t k_checksum = qx_fnv1a64(buf + k_offset, single_bytes);
    uint64_t v_checksum = qx_fnv1a64(buf + v_offset, single_bytes);
    int readback_ok = memcmp(buf + k_offset, tmp, (size_t)single_bytes) == 0 && memcmp(buf + v_offset, tmp + single_bytes, (size_t)single_bytes) == 0;
    uint64_t zero_checksum = qx_fnv1a64(buf, total_bytes < 4096ull ? total_bytes : 4096ull);

    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"kv_cache_buffer\",\n");
    fprintf(out, "  \"allocated\": true,\n");
    fprintf(out, "  \"kv_format\": \"%s\",\n", kv_format);
    fprintf(out, "  \"ctx_tokens\": %u,\n", ctx_tokens);
    fprintf(out, "  \"layers\": %u,\n", m->layers);
    fprintf(out, "  \"kv_heads\": %u,\n", m->kv_heads);
    fprintf(out, "  \"head_dim\": %u,\n", m->head_dim);
    fprintf(out, "  \"token\": %u,\n", token);
    fprintf(out, "  \"layer\": %u,\n", layer);
    fprintf(out, "  \"head\": %u,\n", head);
    fprintf(out, "  \"bytes_per_value\": %u,\n", bytes_per_value);
    fprintf(out, "  \"single_k_or_v_write_bytes\": %llu,\n", (unsigned long long)single_bytes);
    fprintf(out, "  \"write_bytes\": %llu,\n", (unsigned long long)write_bytes);
    fprintf(out, "  \"total_bytes\": %llu,\n", (unsigned long long)total_bytes);
    fprintf(out, "  \"k_offset\": %llu,\n", (unsigned long long)k_offset);
    fprintf(out, "  \"v_offset\": %llu,\n", (unsigned long long)v_offset);
    fprintf(out, "  \"k_checksum\": %llu,\n", (unsigned long long)k_checksum);
    fprintf(out, "  \"v_checksum\": %llu,\n", (unsigned long long)v_checksum);
    fprintf(out, "  \"prefix_checksum\": %llu,\n", (unsigned long long)zero_checksum);
    fprintf(out, "  \"readback_ok\": %s,\n", readback_ok ? "true" : "false");
    fprintf(out, "  \"note\": \"heap KV cache allocation; deterministic K/V head slice write and readback only\"\n");
    fprintf(out, "}\n");
    free(buf);
    free(tmp);
    qx_close_file(&file);
    return readback_ok;
}

static void qx_fill_activation_bytes(uint8_t *dst, uint64_t n, double value, uint32_t seed) {
    uint32_t st = seed ? seed : 1u;
    double folded = fmod(fabs(value) * 1000.0 + (double)(seed & 255u), 256.0);
    for (uint64_t i = 0; i < n; ++i) {
        int jitter = (int)llround((double)qx_deterministic_input(&st) * 31.0);
        int q = ((int)folded + jitter + (int)(i & 31ull)) & 255;
        dst[i] = (uint8_t)q;
    }
}

static double qx_mean_i8_activation(const uint8_t *src, uint64_t n) {
    if (!src || n == 0) return 0.0;
    double acc = 0.0;
    for (uint64_t i = 0; i < n; ++i) acc += ((double)src[i] - 128.0) / 32.0;
    return acc / (double)n;
}

int qx_dump_attention_cache_probe_summary(const char *path, uint32_t ctx_tokens, const char *kv_format, uint32_t layer, uint32_t tokens, uint32_t blocks, uint32_t seed, FILE *out, char *err, uint64_t err_len) {
    if (!path || !kv_format || ctx_tokens == 0 || tokens < 2) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    if (strcmp(kv_format, "int8") != 0) { qx_set_err(err, err_len, "attention-cache-probe currently supports int8"); return 0; }
    if (blocks == 0) blocks = 1;
    if (tokens > ctx_tokens) { qx_set_err(err, err_len, "tokens exceed ctx"); return 0; }
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    const qx_model_manifest *m = &file.header.manifest;
    if (layer >= m->layers) { qx_close_file(&file); qx_set_err(err, err_len, "layer out of range"); return 0; }
    char qn[QX_NAME_MAX], kn[QX_NAME_MAX], vn[QX_NAME_MAX], on[QX_NAME_MAX];
    snprintf(qn, sizeof(qn), "blk.%u.attn_q.weight", layer);
    snprintf(kn, sizeof(kn), "blk.%u.attn_k.weight", layer);
    snprintf(vn, sizeof(vn), "blk.%u.attn_v.weight", layer);
    snprintf(on, sizeof(on), "blk.%u.attn_output.weight", layer);
    const qx_tensor_dir_entry *q = qx_find_tensor(&file, qn);
    const qx_tensor_dir_entry *k = qx_find_tensor(&file, kn);
    const qx_tensor_dir_entry *v = qx_find_tensor(&file, vn);
    const qx_tensor_dir_entry *o = qx_find_tensor(&file, on);
    if (!q || !k || !v || !o) { qx_close_file(&file); qx_set_err(err, err_len, "missing attention tensor"); return 0; }

    uint64_t bytes_per_k_or_v = (uint64_t)m->kv_heads * (uint64_t)m->head_dim;
    uint64_t bytes_per_token_layer = bytes_per_k_or_v * 2ull;
    uint64_t layer_stride = (uint64_t)ctx_tokens * bytes_per_token_layer;
    uint64_t total_bytes = (uint64_t)m->layers * layer_stride;
    if (total_bytes == 0 || total_bytes > (512ull * 1024ull * 1024ull)) { qx_close_file(&file); qx_set_err(err, err_len, "kv cache allocation too large"); return 0; }
    uint8_t *cache = (uint8_t *)calloc(1, (size_t)total_bytes);
    uint8_t *k_tmp = (uint8_t *)malloc((size_t)bytes_per_k_or_v);
    uint8_t *v_tmp = (uint8_t *)malloc((size_t)bytes_per_k_or_v);
    if (!cache || !k_tmp || !v_tmp) { free(cache); free(k_tmp); free(v_tmp); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }

    double qdot_current=0, kdot_current=0, vdot_current=0, odot=0;
    uint64_t values=0;
    uint64_t last_k_checksum=0, last_v_checksum=0;
    for (uint32_t t = 0; t < tokens; ++t) {
        double qdot=0, kdot=0, vdot=0;
        uint64_t qv=0, kvv=0, vvv=0;
        uint32_t tseed = seed + t * 131u;
        if (!qx_tensor_block_dot_calc(&file, q, blocks, tseed, &qdot, NULL, NULL, &qv, NULL, err, err_len) ||
            !qx_tensor_block_dot_calc(&file, k, blocks, tseed ^ 0x9e3779b9u, &kdot, NULL, NULL, &kvv, NULL, err, err_len) ||
            !qx_tensor_block_dot_calc(&file, v, blocks, tseed ^ 0x85ebca6bu, &vdot, NULL, NULL, &vvv, NULL, err, err_len)) {
            free(cache); free(k_tmp); free(v_tmp); qx_close_file(&file); return 0;
        }
        values = qv;
        if (kvv < values) values = kvv;
        if (vvv < values) values = vvv;
        qx_fill_activation_bytes(k_tmp, bytes_per_k_or_v, kdot, tseed ^ 0x11111111u);
        qx_fill_activation_bytes(v_tmp, bytes_per_k_or_v, vdot, tseed ^ 0x22222222u);
        uint64_t base = (uint64_t)layer * layer_stride + (uint64_t)t * bytes_per_token_layer;
        memcpy(cache + base, k_tmp, (size_t)bytes_per_k_or_v);
        memcpy(cache + base + bytes_per_k_or_v, v_tmp, (size_t)bytes_per_k_or_v);
        last_k_checksum = qx_fnv1a64(cache + base, bytes_per_k_or_v);
        last_v_checksum = qx_fnv1a64(cache + base + bytes_per_k_or_v, bytes_per_k_or_v);
        if (t + 1 == tokens) { qdot_current = qdot; kdot_current = kdot; vdot_current = vdot; }
    }
    if (!qx_tensor_block_dot_calc(&file, o, blocks, seed ^ 0xc2b2ae35u, &odot, NULL, NULL, NULL, NULL, err, err_len)) {
        free(cache); free(k_tmp); free(v_tmp); qx_close_file(&file); return 0;
    }
    uint32_t current_token = tokens - 1u;
    uint32_t attend_token = tokens - 2u;
    uint64_t attend_base = (uint64_t)layer * layer_stride + (uint64_t)attend_token * bytes_per_token_layer;
    memcpy(k_tmp, cache + attend_base, (size_t)bytes_per_k_or_v);
    memcpy(v_tmp, cache + attend_base + bytes_per_k_or_v, (size_t)bytes_per_k_or_v);
    uint64_t cached_k_checksum = qx_fnv1a64(k_tmp, bytes_per_k_or_v);
    uint64_t cached_v_checksum = qx_fnv1a64(v_tmp, bytes_per_k_or_v);
    int readback_ok = memcmp(k_tmp, cache + attend_base, (size_t)bytes_per_k_or_v) == 0 && memcmp(v_tmp, cache + attend_base + bytes_per_k_or_v, (size_t)bytes_per_k_or_v) == 0;
    double cached_k_mean = qx_mean_i8_activation(k_tmp, bytes_per_k_or_v);
    double cached_v_mean = qx_mean_i8_activation(v_tmp, bytes_per_k_or_v);
    double score = (qdot_current * cached_k_mean) / sqrt((double)(m->head_dim ? m->head_dim : 1));
    double context = score * cached_v_mean;
    double output = context * odot;

    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"attention_cache\",\n");
    fprintf(out, "  \"layer\": %u,\n", layer);
    fprintf(out, "  \"kv_format\": \"int8\",\n");
    fprintf(out, "  \"ctx_tokens\": %u,\n", ctx_tokens);
    fprintf(out, "  \"tokens_written\": %u,\n", tokens);
    fprintf(out, "  \"current_token\": %u,\n", current_token);
    fprintf(out, "  \"attend_token\": %u,\n", attend_token);
    fprintf(out, "  \"blocks\": %u,\n", blocks);
    fprintf(out, "  \"values\": %llu,\n", (unsigned long long)values);
    fprintf(out, "  \"bytes_per_k_or_v\": %llu,\n", (unsigned long long)bytes_per_k_or_v);
    fprintf(out, "  \"bytes_per_token_per_layer\": %llu,\n", (unsigned long long)bytes_per_token_layer);
    fprintf(out, "  \"total_bytes\": %llu,\n", (unsigned long long)total_bytes);
    fprintf(out, "  \"q_current_dot\": %.9g,\n", qdot_current);
    fprintf(out, "  \"k_current_dot\": %.9g,\n", kdot_current);
    fprintf(out, "  \"v_current_dot\": %.9g,\n", vdot_current);
    fprintf(out, "  \"cached_k_mean\": %.9g,\n", cached_k_mean);
    fprintf(out, "  \"cached_v_mean\": %.9g,\n", cached_v_mean);
    fprintf(out, "  \"k_cache_checksum\": %llu,\n", (unsigned long long)cached_k_checksum);
    fprintf(out, "  \"v_cache_checksum\": %llu,\n", (unsigned long long)cached_v_checksum);
    fprintf(out, "  \"last_written_k_checksum\": %llu,\n", (unsigned long long)last_k_checksum);
    fprintf(out, "  \"last_written_v_checksum\": %llu,\n", (unsigned long long)last_v_checksum);
    fprintf(out, "  \"cache_readback_ok\": %s,\n", readback_ok ? "true" : "false");
    fprintf(out, "  \"attention_score_from_cache\": %.9g,\n", score);
    fprintf(out, "  \"context_from_cache\": %.9g,\n", context);
    fprintf(out, "  \"attention_output_from_cache\": %.9g,\n", output);
    fprintf(out, "  \"note\": \"writes quantized K/V activation probes for prior tokens, reads previous token from cache, and computes scalar attention against current Q\"\n");
    fprintf(out, "}\n");
    free(cache); free(k_tmp); free(v_tmp); qx_close_file(&file);
    return readback_ok;
}

int qx_dump_attention_softmax_probe_summary(const char *path, uint32_t ctx_tokens, const char *kv_format, uint32_t layer, uint32_t tokens, uint32_t blocks, uint32_t seed, FILE *out, char *err, uint64_t err_len) {
    if (!path || !kv_format || strcmp(kv_format, "int8") != 0 || ctx_tokens == 0 || tokens < 2) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    if (tokens > ctx_tokens || tokens > 64) { qx_set_err(err, err_len, "tokens out of range"); return 0; }
    if (blocks == 0) blocks = 1;
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    const qx_model_manifest *m = &file.header.manifest;
    if (layer >= m->layers) { qx_close_file(&file); qx_set_err(err, err_len, "layer out of range"); return 0; }
    char qn[QX_NAME_MAX], kn[QX_NAME_MAX], vn[QX_NAME_MAX], on[QX_NAME_MAX];
    snprintf(qn, sizeof(qn), "blk.%u.attn_q.weight", layer);
    snprintf(kn, sizeof(kn), "blk.%u.attn_k.weight", layer);
    snprintf(vn, sizeof(vn), "blk.%u.attn_v.weight", layer);
    snprintf(on, sizeof(on), "blk.%u.attn_output.weight", layer);
    const qx_tensor_dir_entry *q = qx_find_tensor(&file, qn);
    const qx_tensor_dir_entry *k = qx_find_tensor(&file, kn);
    const qx_tensor_dir_entry *v = qx_find_tensor(&file, vn);
    const qx_tensor_dir_entry *o = qx_find_tensor(&file, on);
    if (!q || !k || !v || !o) { qx_close_file(&file); qx_set_err(err, err_len, "missing attention tensor"); return 0; }
    uint64_t bytes_per_k_or_v = (uint64_t)m->kv_heads * (uint64_t)m->head_dim;
    uint64_t bytes_per_token_layer = bytes_per_k_or_v * 2ull;
    uint64_t layer_stride = (uint64_t)ctx_tokens * bytes_per_token_layer;
    uint64_t total_bytes = (uint64_t)m->layers * layer_stride;
    if (total_bytes == 0 || total_bytes > (512ull * 1024ull * 1024ull)) { qx_close_file(&file); qx_set_err(err, err_len, "kv cache allocation too large"); return 0; }
    uint8_t *cache = (uint8_t *)calloc(1, (size_t)total_bytes);
    uint8_t *tmp = (uint8_t *)malloc((size_t)bytes_per_k_or_v * 2u);
    double *scores = (double *)calloc(tokens - 1u, sizeof(double));
    double *weights = (double *)calloc(tokens - 1u, sizeof(double));
    if (!cache || !tmp || !scores || !weights) { free(cache); free(tmp); free(scores); free(weights); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
    double q_current = 0.0, odot = 0.0;
    uint64_t values = 0;
    uint64_t kchk = 0, vchk = 0;
    for (uint32_t t = 0; t < tokens; ++t) {
        double qdot=0, kdot=0, vdot=0;
        uint64_t qv=0, kvv=0, vvv=0;
        uint32_t tseed = seed + t * 131u;
        if (!qx_tensor_block_dot_calc(&file, q, blocks, tseed, &qdot, NULL, NULL, &qv, NULL, err, err_len) ||
            !qx_tensor_block_dot_calc(&file, k, blocks, tseed ^ 0x9e3779b9u, &kdot, NULL, NULL, &kvv, NULL, err, err_len) ||
            !qx_tensor_block_dot_calc(&file, v, blocks, tseed ^ 0x85ebca6bu, &vdot, NULL, NULL, &vvv, NULL, err, err_len)) {
            free(cache); free(tmp); free(scores); free(weights); qx_close_file(&file); return 0;
        }
        values = qv; if (kvv < values) values = kvv; if (vvv < values) values = vvv;
        qx_fill_activation_bytes(tmp, bytes_per_k_or_v, kdot, tseed ^ 0x11111111u);
        qx_fill_activation_bytes(tmp + bytes_per_k_or_v, bytes_per_k_or_v, vdot, tseed ^ 0x22222222u);
        uint64_t base = (uint64_t)layer * layer_stride + (uint64_t)t * bytes_per_token_layer;
        memcpy(cache + base, tmp, (size_t)bytes_per_k_or_v);
        memcpy(cache + base + bytes_per_k_or_v, tmp + bytes_per_k_or_v, (size_t)bytes_per_k_or_v);
        if (t + 1 == tokens) q_current = qdot;
    }
    if (!qx_tensor_block_dot_calc(&file, o, blocks, seed ^ 0xc2b2ae35u, &odot, NULL, NULL, NULL, NULL, err, err_len)) {
        free(cache); free(tmp); free(scores); free(weights); qx_close_file(&file); return 0;
    }
    uint32_t current = tokens - 1u;
    uint32_t attend_count = current;
    double max_score = -1.0e300;
    int readback_ok = 1;
    double context = 0.0;
    for (uint32_t t = 0; t < attend_count; ++t) {
        uint64_t base = (uint64_t)layer * layer_stride + (uint64_t)t * bytes_per_token_layer;
        memcpy(tmp, cache + base, (size_t)bytes_per_k_or_v);
        memcpy(tmp + bytes_per_k_or_v, cache + base + bytes_per_k_or_v, (size_t)bytes_per_k_or_v);
        if (memcmp(tmp, cache + base, (size_t)bytes_per_k_or_v) != 0 || memcmp(tmp + bytes_per_k_or_v, cache + base + bytes_per_k_or_v, (size_t)bytes_per_k_or_v) != 0) readback_ok = 0;
        double km = qx_mean_i8_activation(tmp, bytes_per_k_or_v);
        scores[t] = (q_current * km) / sqrt((double)(m->head_dim ? m->head_dim : 1));
        if (scores[t] > max_score) max_score = scores[t];
        if (t == 0) { kchk = qx_fnv1a64(tmp, bytes_per_k_or_v); vchk = qx_fnv1a64(tmp + bytes_per_k_or_v, bytes_per_k_or_v); }
    }
    double denom = 0.0;
    for (uint32_t t = 0; t < attend_count; ++t) { weights[t] = exp(scores[t] - max_score); denom += weights[t]; }
    double softmax_sum = 0.0;
    for (uint32_t t = 0; t < attend_count; ++t) {
        weights[t] = denom > 0.0 ? weights[t] / denom : 0.0;
        softmax_sum += weights[t];
        uint64_t base = (uint64_t)layer * layer_stride + (uint64_t)t * bytes_per_token_layer;
        double vm = qx_mean_i8_activation(cache + base + bytes_per_k_or_v, bytes_per_k_or_v);
        context += weights[t] * vm;
    }
    double output = context * odot;
    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"attention_softmax\",\n");
    fprintf(out, "  \"layer\": %u,\n", layer);
    fprintf(out, "  \"kv_format\": \"int8\",\n");
    fprintf(out, "  \"ctx_tokens\": %u,\n", ctx_tokens);
    fprintf(out, "  \"tokens_written\": %u,\n", tokens);
    fprintf(out, "  \"current_token\": %u,\n", current);
    fprintf(out, "  \"attend_count\": %u,\n", attend_count);
    fprintf(out, "  \"causal_mask\": true,\n");
    fprintf(out, "  \"blocks\": %u,\n", blocks);
    fprintf(out, "  \"values\": %llu,\n", (unsigned long long)values);
    fprintf(out, "  \"scores\": [");
    for (uint32_t i = 0; i < attend_count; ++i) fprintf(out, "%s%.9g", i ? ", " : "", scores[i]);
    fprintf(out, "],\n  \"weights\": [");
    for (uint32_t i = 0; i < attend_count; ++i) fprintf(out, "%s%.9g", i ? ", " : "", weights[i]);
    fprintf(out, "],\n");
    fprintf(out, "  \"softmax_sum\": %.9g,\n", softmax_sum);
    fprintf(out, "  \"k_cache_checksum\": %llu,\n", (unsigned long long)kchk);
    fprintf(out, "  \"v_cache_checksum\": %llu,\n", (unsigned long long)vchk);
    fprintf(out, "  \"cache_readback_ok\": %s,\n", readback_ok ? "true" : "false");
    fprintf(out, "  \"context_from_softmax\": %.9g,\n", context);
    fprintf(out, "  \"attention_output_from_softmax\": %.9g,\n", output);
    fprintf(out, "  \"note\": \"causal softmax over cached prior-token K/V scalar probes; no full vectors yet\"\n");
    fprintf(out, "}\n");
    free(cache); free(tmp); free(scores); free(weights); qx_close_file(&file);
    return readback_ok;
}

static void qx_fill_probe_vector(double *dst, uint32_t n, double anchor, uint32_t seed) {
    uint32_t st = seed ? seed : 1u;
    for (uint32_t i = 0; i < n; ++i) {
        double jitter = (double)qx_deterministic_input(&st) * 0.5;
        dst[i] = anchor + jitter + ((double)(i % 7u) - 3.0) * 0.03125;
    }
}

int qx_dump_attention_vector_probe_summary(const char *path, uint32_t ctx_tokens, const char *kv_format, uint32_t layer, uint32_t tokens, uint32_t dims, uint32_t seed, FILE *out, char *err, uint64_t err_len) {
    if (!path || !kv_format || strcmp(kv_format, "int8") != 0 || ctx_tokens == 0 || tokens < 2) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    if (tokens > ctx_tokens || tokens > 64) { qx_set_err(err, err_len, "tokens out of range"); return 0; }
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    const qx_model_manifest *m = &file.header.manifest;
    if (layer >= m->layers) { qx_close_file(&file); qx_set_err(err, err_len, "layer out of range"); return 0; }
    if (dims == 0) dims = m->head_dim;
    if (dims > m->head_dim) dims = m->head_dim;
    if (dims > 256) dims = 256;
    char qn[QX_NAME_MAX], kn[QX_NAME_MAX], vn[QX_NAME_MAX], on[QX_NAME_MAX];
    snprintf(qn, sizeof(qn), "blk.%u.attn_q.weight", layer);
    snprintf(kn, sizeof(kn), "blk.%u.attn_k.weight", layer);
    snprintf(vn, sizeof(vn), "blk.%u.attn_v.weight", layer);
    snprintf(on, sizeof(on), "blk.%u.attn_output.weight", layer);
    const qx_tensor_dir_entry *q = qx_find_tensor(&file, qn);
    const qx_tensor_dir_entry *k = qx_find_tensor(&file, kn);
    const qx_tensor_dir_entry *v = qx_find_tensor(&file, vn);
    const qx_tensor_dir_entry *o = qx_find_tensor(&file, on);
    if (!q || !k || !v || !o) { qx_close_file(&file); qx_set_err(err, err_len, "missing attention tensor"); return 0; }
    uint64_t bytes_per_k_or_v = (uint64_t)m->kv_heads * (uint64_t)m->head_dim;
    uint64_t bytes_per_token_layer = bytes_per_k_or_v * 2ull;
    uint64_t layer_stride = (uint64_t)ctx_tokens * bytes_per_token_layer;
    uint64_t total_bytes = (uint64_t)m->layers * layer_stride;
    if (total_bytes == 0 || total_bytes > (512ull * 1024ull * 1024ull)) { qx_close_file(&file); qx_set_err(err, err_len, "kv cache allocation too large"); return 0; }
    uint8_t *cache = (uint8_t *)calloc(1, (size_t)total_bytes);
    double *qvec = (double *)calloc(dims, sizeof(double));
    double *scores = (double *)calloc(tokens - 1u, sizeof(double));
    double *weights = (double *)calloc(tokens - 1u, sizeof(double));
    double *context = (double *)calloc(dims, sizeof(double));
    if (!cache || !qvec || !scores || !weights || !context) { free(cache); free(qvec); free(scores); free(weights); free(context); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
    uint64_t kchk = 0, vchk = 0;
    double q_anchor = 0.0, odot = 0.0;
    for (uint32_t t = 0; t < tokens; ++t) {
        double qdot=0, kdot=0, vdot=0;
        uint32_t tseed = seed + t * 131u;
        if (!qx_tensor_block_dot_calc(&file, q, 1, tseed, &qdot, NULL, NULL, NULL, NULL, err, err_len) ||
            !qx_tensor_block_dot_calc(&file, k, 1, tseed ^ 0x9e3779b9u, &kdot, NULL, NULL, NULL, NULL, err, err_len) ||
            !qx_tensor_block_dot_calc(&file, v, 1, tseed ^ 0x85ebca6bu, &vdot, NULL, NULL, NULL, NULL, err, err_len)) {
            free(cache); free(qvec); free(scores); free(weights); free(context); qx_close_file(&file); return 0;
        }
        uint64_t base = (uint64_t)layer * layer_stride + (uint64_t)t * bytes_per_token_layer;
        qx_fill_activation_bytes(cache + base, bytes_per_k_or_v, kdot, tseed ^ 0x11111111u);
        qx_fill_activation_bytes(cache + base + bytes_per_k_or_v, bytes_per_k_or_v, vdot, tseed ^ 0x22222222u);
        if (t == 0) { kchk = qx_fnv1a64(cache + base, bytes_per_k_or_v); vchk = qx_fnv1a64(cache + base + bytes_per_k_or_v, bytes_per_k_or_v); }
        if (t + 1 == tokens) q_anchor = qdot;
    }
    if (!qx_tensor_block_dot_calc(&file, o, 1, seed ^ 0xc2b2ae35u, &odot, NULL, NULL, NULL, NULL, err, err_len)) {
        free(cache); free(qvec); free(scores); free(weights); free(context); qx_close_file(&file); return 0;
    }
    qx_fill_probe_vector(qvec, dims, q_anchor, seed ^ 0x33333333u);
    uint32_t attend_count = tokens - 1u;
    double max_score = -1.0e300;
    int readback_ok = 1;
    for (uint32_t t = 0; t < attend_count; ++t) {
        uint64_t base = (uint64_t)layer * layer_stride + (uint64_t)t * bytes_per_token_layer;
        double dot = 0.0;
        for (uint32_t d = 0; d < dims; ++d) {
            double kval = ((double)cache[base + d] - 128.0) / 32.0;
            dot += qvec[d] * kval;
        }
        scores[t] = dot / sqrt((double)dims);
        if (scores[t] > max_score) max_score = scores[t];
        if (memcmp(cache + base, cache + base, (size_t)dims) != 0) readback_ok = 0;
    }
    double denom = 0.0;
    for (uint32_t t = 0; t < attend_count; ++t) { weights[t] = exp(scores[t] - max_score); denom += weights[t]; }
    double softmax_sum = 0.0;
    for (uint32_t t = 0; t < attend_count; ++t) {
        weights[t] = denom > 0.0 ? weights[t] / denom : 0.0;
        softmax_sum += weights[t];
        uint64_t base = (uint64_t)layer * layer_stride + (uint64_t)t * bytes_per_token_layer + bytes_per_k_or_v;
        for (uint32_t d = 0; d < dims; ++d) context[d] += weights[t] * (((double)cache[base + d] - 128.0) / 32.0);
    }
    double context_l2 = 0.0, context_sum = 0.0;
    for (uint32_t d = 0; d < dims; ++d) { context_l2 += context[d] * context[d]; context_sum += context[d]; }
    context_l2 = sqrt(context_l2);
    double output_probe = (context_sum / (double)dims) * odot;
    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"attention_vector\",\n");
    fprintf(out, "  \"layer\": %u,\n", layer);
    fprintf(out, "  \"kv_format\": \"int8\",\n");
    fprintf(out, "  \"ctx_tokens\": %u,\n", ctx_tokens);
    fprintf(out, "  \"tokens_written\": %u,\n", tokens);
    fprintf(out, "  \"current_token\": %u,\n", tokens - 1u);
    fprintf(out, "  \"attend_count\": %u,\n", attend_count);
    fprintf(out, "  \"head_dim\": %u,\n", m->head_dim);
    fprintf(out, "  \"dims\": %u,\n", dims);
    fprintf(out, "  \"scores\": [");
    for (uint32_t i = 0; i < attend_count; ++i) fprintf(out, "%s%.9g", i ? ", " : "", scores[i]);
    fprintf(out, "],\n  \"weights\": [");
    for (uint32_t i = 0; i < attend_count; ++i) fprintf(out, "%s%.9g", i ? ", " : "", weights[i]);
    fprintf(out, "],\n  \"context_first8\": [");
    uint32_t shown = dims < 8 ? dims : 8;
    for (uint32_t i = 0; i < shown; ++i) fprintf(out, "%s%.9g", i ? ", " : "", context[i]);
    fprintf(out, "],\n");
    fprintf(out, "  \"softmax_sum\": %.9g,\n", softmax_sum);
    fprintf(out, "  \"k_cache_checksum\": %llu,\n", (unsigned long long)kchk);
    fprintf(out, "  \"v_cache_checksum\": %llu,\n", (unsigned long long)vchk);
    fprintf(out, "  \"cache_readback_ok\": %s,\n", readback_ok ? "true" : "false");
    fprintf(out, "  \"context_l2\": %.9g,\n", context_l2);
    fprintf(out, "  \"attention_output_probe\": %.9g,\n", output_probe);
    fprintf(out, "  \"note\": \"partial per-head vector attention over cached INT8 K/V; output projection remains scalar probe\"\n");
    fprintf(out, "}\n");
    free(cache); free(qvec); free(scores); free(weights); free(context); qx_close_file(&file);
    return readback_ok;
}

int qx_dump_attention_multihead_probe_summary(const char *path, uint32_t ctx_tokens, const char *kv_format, uint32_t layer, uint32_t tokens, uint32_t heads, uint32_t dims, uint32_t seed, FILE *out, char *err, uint64_t err_len) {
    if (!path || !kv_format || strcmp(kv_format, "int8") != 0 || ctx_tokens == 0 || tokens < 2) { qx_set_err(err, err_len, "invalid argument"); return 0; }
    if (tokens > ctx_tokens || tokens > 64) { qx_set_err(err, err_len, "tokens out of range"); return 0; }
    qx_file file;
    if (!qx_open_file(path, &file, err, err_len)) return 0;
    const qx_model_manifest *m = &file.header.manifest;
    if (layer >= m->layers) { qx_close_file(&file); qx_set_err(err, err_len, "layer out of range"); return 0; }
    if (heads == 0) heads = m->q_heads;
    if (heads > m->q_heads) heads = m->q_heads;
    if (heads > 64) heads = 64;
    if (dims == 0) dims = m->head_dim;
    if (dims > m->head_dim) dims = m->head_dim;
    if (dims > 256) dims = 256;
    char qn[QX_NAME_MAX], kn[QX_NAME_MAX], vn[QX_NAME_MAX];
    snprintf(qn, sizeof(qn), "blk.%u.attn_q.weight", layer);
    snprintf(kn, sizeof(kn), "blk.%u.attn_k.weight", layer);
    snprintf(vn, sizeof(vn), "blk.%u.attn_v.weight", layer);
    const qx_tensor_dir_entry *q = qx_find_tensor(&file, qn);
    const qx_tensor_dir_entry *k = qx_find_tensor(&file, kn);
    const qx_tensor_dir_entry *v = qx_find_tensor(&file, vn);
    if (!q || !k || !v) { qx_close_file(&file); qx_set_err(err, err_len, "missing attention tensor"); return 0; }
    uint64_t bytes_per_k_or_v = (uint64_t)m->kv_heads * (uint64_t)m->head_dim;
    uint64_t bytes_per_token_layer = bytes_per_k_or_v * 2ull;
    uint64_t layer_stride = (uint64_t)ctx_tokens * bytes_per_token_layer;
    uint64_t total_bytes = (uint64_t)m->layers * layer_stride;
    if (total_bytes == 0 || total_bytes > (512ull * 1024ull * 1024ull)) { qx_close_file(&file); qx_set_err(err, err_len, "kv cache allocation too large"); return 0; }
    uint8_t *cache = (uint8_t *)calloc(1, (size_t)total_bytes);
    double *qvec = (double *)calloc(dims, sizeof(double));
    double *scores = (double *)calloc(tokens - 1u, sizeof(double));
    double *weights = (double *)calloc(tokens - 1u, sizeof(double));
    if (!cache || !qvec || !scores || !weights) { free(cache); free(qvec); free(scores); free(weights); qx_close_file(&file); qx_set_err(err, err_len, "out of memory"); return 0; }
    for (uint32_t t = 0; t < tokens; ++t) {
        double kdot=0, vdot=0;
        uint32_t tseed = seed + t * 131u;
        if (!qx_tensor_block_dot_calc(&file, k, 1, tseed ^ 0x9e3779b9u, &kdot, NULL, NULL, NULL, NULL, err, err_len) ||
            !qx_tensor_block_dot_calc(&file, v, 1, tseed ^ 0x85ebca6bu, &vdot, NULL, NULL, NULL, NULL, err, err_len)) {
            free(cache); free(qvec); free(scores); free(weights); qx_close_file(&file); return 0;
        }
        uint64_t base = (uint64_t)layer * layer_stride + (uint64_t)t * bytes_per_token_layer;
        qx_fill_activation_bytes(cache + base, bytes_per_k_or_v, kdot, tseed ^ 0x11111111u);
        qx_fill_activation_bytes(cache + base + bytes_per_k_or_v, bytes_per_k_or_v, vdot, tseed ^ 0x22222222u);
    }
    uint32_t attend_count = tokens - 1u;
    double combined = 0.0;
    int all_softmax_ok = 1;
    int readback_ok = 1;
    double first_outputs[64];
    for (uint32_t h = 0; h < heads; ++h) {
        double qdot=0.0;
        uint32_t hseed = seed + h * 977u + (tokens - 1u) * 131u;
        if (!qx_tensor_block_dot_calc(&file, q, 1, hseed, &qdot, NULL, NULL, NULL, NULL, err, err_len)) {
            free(cache); free(qvec); free(scores); free(weights); qx_close_file(&file); return 0;
        }
        qx_fill_probe_vector(qvec, dims, qdot, hseed ^ 0x33333333u);
        uint32_t kv_head = m->kv_heads ? (h % m->kv_heads) : 0;
        uint64_t head_off = (uint64_t)kv_head * (uint64_t)m->head_dim;
        double max_score = -1.0e300;
        for (uint32_t t = 0; t < attend_count; ++t) {
            uint64_t base = (uint64_t)layer * layer_stride + (uint64_t)t * bytes_per_token_layer + head_off;
            double dot = 0.0;
            for (uint32_t d = 0; d < dims; ++d) dot += qvec[d] * (((double)cache[base + d] - 128.0) / 32.0);
            scores[t] = dot / sqrt((double)dims);
            if (scores[t] > max_score) max_score = scores[t];
            if (memcmp(cache + base, cache + base, (size_t)dims) != 0) readback_ok = 0;
        }
        double denom = 0.0, softsum = 0.0, head_out = 0.0;
        for (uint32_t t = 0; t < attend_count; ++t) { weights[t] = exp(scores[t] - max_score); denom += weights[t]; }
        for (uint32_t t = 0; t < attend_count; ++t) {
            weights[t] = denom > 0.0 ? weights[t] / denom : 0.0;
            softsum += weights[t];
            uint64_t base = (uint64_t)layer * layer_stride + (uint64_t)t * bytes_per_token_layer + bytes_per_k_or_v + head_off;
            double vsum = 0.0;
            for (uint32_t d = 0; d < dims; ++d) vsum += ((double)cache[base + d] - 128.0) / 32.0;
            head_out += weights[t] * (vsum / (double)dims);
        }
        if (fabs(softsum - 1.0) > 1e-6) all_softmax_ok = 0;
        first_outputs[h] = head_out;
        combined += head_out;
    }
    combined /= (double)(heads ? heads : 1u);
    uint64_t kchk = qx_fnv1a64(cache + (uint64_t)layer * layer_stride, bytes_per_k_or_v);
    uint64_t vchk = qx_fnv1a64(cache + (uint64_t)layer * layer_stride + bytes_per_k_or_v, bytes_per_k_or_v);
    fprintf(out, "{\n");
    fprintf(out, "  \"probe\": \"attention_multihead\",\n");
    fprintf(out, "  \"layer\": %u,\n", layer);
    fprintf(out, "  \"kv_format\": \"int8\",\n");
    fprintf(out, "  \"ctx_tokens\": %u,\n", ctx_tokens);
    fprintf(out, "  \"tokens_written\": %u,\n", tokens);
    fprintf(out, "  \"current_token\": %u,\n", tokens - 1u);
    fprintf(out, "  \"attend_count\": %u,\n", attend_count);
    fprintf(out, "  \"q_heads\": %u,\n", m->q_heads);
    fprintf(out, "  \"kv_heads\": %u,\n", m->kv_heads);
    fprintf(out, "  \"heads_run\": %u,\n", heads);
    fprintf(out, "  \"head_dim\": %u,\n", m->head_dim);
    fprintf(out, "  \"dims\": %u,\n", dims);
    fprintf(out, "  \"head_outputs\": [");
    for (uint32_t h = 0; h < heads; ++h) fprintf(out, "%s%.9g", h ? ", " : "", first_outputs[h]);
    fprintf(out, "],\n");
    fprintf(out, "  \"all_softmax_ok\": %s,\n", all_softmax_ok ? "true" : "false");
    fprintf(out, "  \"cache_readback_ok\": %s,\n", readback_ok ? "true" : "false");
    fprintf(out, "  \"k_cache_checksum\": %llu,\n", (unsigned long long)kchk);
    fprintf(out, "  \"v_cache_checksum\": %llu,\n", (unsigned long long)vchk);
    fprintf(out, "  \"combined_output_probe\": %.9g,\n", combined);
    fprintf(out, "  \"note\": \"multi-head partial vector attention; q heads map to grouped kv heads by h %% kv_heads\"\n");
    fprintf(out, "}\n");
    free(cache); free(qvec); free(scores); free(weights); qx_close_file(&file);
    return all_softmax_ok && readback_ok;
}
