#include "qx_gguf.h"

#include "qx_format.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

static void set_err(char *err, uint64_t err_len, const char *msg) {
    if (!err || err_len == 0) return;
#ifdef _MSC_VER
    strncpy_s(err, (size_t)err_len, msg, _TRUNCATE);
#else
    snprintf(err, (size_t)err_len, "%s", msg);
#endif
}

static void set_errno_err(char *err, uint64_t err_len, int code) {
    if (!err || err_len == 0) return;
#ifdef _MSC_VER
    strerror_s(err, (size_t)err_len, code);
#else
    snprintf(err, (size_t)err_len, "%s", strerror(code));
#endif
}

const char *qx_gguf_value_type_name(uint32_t t) {
    switch (t) {
        case QX_GGUF_UINT8: return "uint8";
        case QX_GGUF_INT8: return "int8";
        case QX_GGUF_UINT16: return "uint16";
        case QX_GGUF_INT16: return "int16";
        case QX_GGUF_UINT32: return "uint32";
        case QX_GGUF_INT32: return "int32";
        case QX_GGUF_FLOAT32: return "float32";
        case QX_GGUF_BOOL: return "bool";
        case QX_GGUF_STRING: return "string";
        case QX_GGUF_ARRAY: return "array";
        case QX_GGUF_UINT64: return "uint64";
        case QX_GGUF_INT64: return "int64";
        case QX_GGUF_FLOAT64: return "float64";
        default: return "unknown";
    }
}

static int rd(FILE *f, void *dst, size_t n) {
    return fread(dst, 1, n, f) == n;
}

static int rd_u32(FILE *f, uint32_t *v) { return rd(f, v, sizeof(*v)); }
static int rd_u64(FILE *f, uint64_t *v) { return rd(f, v, sizeof(*v)); }

static uint64_t file_size_of(FILE *f) {
#if defined(_WIN32)
    int64_t cur = _ftelli64(f);
    if (_fseeki64(f, 0, SEEK_END) != 0) return 0;
    int64_t end = _ftelli64(f);
    _fseeki64(f, cur, SEEK_SET);
    return end < 0 ? 0 : (uint64_t)end;
#else
    off_t cur = ftello(f);
    if (fseeko(f, 0, SEEK_END) != 0) return 0;
    off_t end = ftello(f);
    fseeko(f, cur, SEEK_SET);
    return end < 0 ? 0 : (uint64_t)end;
#endif
}

static int skip_bytes(FILE *f, uint64_t n) {
#if defined(_WIN32)
    return _fseeki64(f, (int64_t)n, SEEK_CUR) == 0;
#else
    return fseeko(f, (off_t)n, SEEK_CUR) == 0;
#endif
}

static int read_string(FILE *f, char *dst, uint64_t cap, uint64_t *len_out) {
    uint64_t len = 0;
    if (!rd_u64(f, &len)) return 0;
    if (len_out) *len_out = len;
    if (dst && cap > 0) {
        uint64_t keep = len < cap - 1 ? len : cap - 1;
        if (keep && !rd(f, dst, (size_t)keep)) return 0;
        dst[keep] = 0;
        if (len > keep && !skip_bytes(f, len - keep)) return 0;
    } else {
        if (!skip_bytes(f, len)) return 0;
    }
    return 1;
}

static uint64_t scalar_size(uint32_t type) {
    switch (type) {
        case QX_GGUF_UINT8: return 1;
        case QX_GGUF_INT8: return 1;
        case QX_GGUF_UINT16: return 2;
        case QX_GGUF_INT16: return 2;
        case QX_GGUF_UINT32: return 4;
        case QX_GGUF_INT32: return 4;
        case QX_GGUF_FLOAT32: return 4;
        case QX_GGUF_BOOL: return 1;
        case QX_GGUF_UINT64: return 8;
        case QX_GGUF_INT64: return 8;
        case QX_GGUF_FLOAT64: return 8;
        default: return 0;
    }
}

static int skip_value(FILE *f, uint32_t type) {
    if (type == QX_GGUF_STRING) return read_string(f, NULL, 0, NULL);
    if (type == QX_GGUF_ARRAY) {
        uint32_t elem_type = 0;
        uint64_t count = 0;
        if (!rd_u32(f, &elem_type) || !rd_u64(f, &count)) return 0;
        if (elem_type == QX_GGUF_STRING) {
            for (uint64_t i = 0; i < count; i++) if (!read_string(f, NULL, 0, NULL)) return 0;
            return 1;
        }
        uint64_t sz = scalar_size(elem_type);
        if (sz == 0) return 0;
        return skip_bytes(f, sz * count);
    }
    uint64_t sz = scalar_size(type);
    if (sz == 0) return 0;
    return skip_bytes(f, sz);
}

static int read_u64_value(FILE *f, uint32_t type, uint64_t *out) {
    uint8_t u8; uint16_t u16; uint32_t u32; uint64_t u64;
    int8_t i8; int16_t i16; int32_t i32; int64_t i64;
    switch (type) {
        case QX_GGUF_UINT8: if (!rd(f, &u8, 1)) return 0; *out = u8; return 1;
        case QX_GGUF_INT8: if (!rd(f, &i8, 1)) return 0; *out = (uint64_t)i8; return 1;
        case QX_GGUF_UINT16: if (!rd(f, &u16, 2)) return 0; *out = u16; return 1;
        case QX_GGUF_INT16: if (!rd(f, &i16, 2)) return 0; *out = (uint64_t)i16; return 1;
        case QX_GGUF_UINT32: if (!rd(f, &u32, 4)) return 0; *out = u32; return 1;
        case QX_GGUF_INT32: if (!rd(f, &i32, 4)) return 0; *out = (uint64_t)i32; return 1;
        case QX_GGUF_UINT64: if (!rd(f, &u64, 8)) return 0; *out = u64; return 1;
        case QX_GGUF_INT64: if (!rd(f, &i64, 8)) return 0; *out = (uint64_t)i64; return 1;
        default: return skip_value(f, type) ? 2 : 0;
    }
}

static void apply_u64_meta(qx_gguf_summary *s, const char *key, uint64_t value) {
    const char *prefix = "qwen3.";
    const char *moe_prefix = "qwen3moe.";
    if (strcmp(key, "general.alignment") == 0) s->alignment = (uint32_t)value;
    else if (strncmp(key, prefix, strlen(prefix)) == 0 || strncmp(key, moe_prefix, strlen(moe_prefix)) == 0) {
        const char *k = strncmp(key, prefix, strlen(prefix)) == 0 ? key + strlen(prefix) : key + strlen(moe_prefix);
        if (strcmp(k, "block_count") == 0) s->qwen3_block_count = value;
        else if (strcmp(k, "expert_count") == 0) s->qwen3_expert_count = value;
        else if (strcmp(k, "expert_used_count") == 0) s->qwen3_expert_used_count = value;
        else if (strcmp(k, "embedding_length") == 0) s->qwen3_embedding_length = value;
        else if (strcmp(k, "feed_forward_length") == 0) s->qwen3_feed_forward_length = value;
        else if (strcmp(k, "attention.head_count") == 0) s->qwen3_attention_head_count = value;
        else if (strcmp(k, "attention.head_count_kv") == 0) s->qwen3_attention_head_count_kv = value;
    }
}

static int apply_string_meta(FILE *f, qx_gguf_summary *s, const char *key) {
    char tmp[160];
    if (!read_string(f, tmp, sizeof(tmp), NULL)) return 0;
    if (strcmp(key, "general.architecture") == 0) {
#ifdef _MSC_VER
        strncpy_s(s->architecture, sizeof(s->architecture), tmp, _TRUNCATE);
#else
        snprintf(s->architecture, sizeof(s->architecture), "%s", tmp);
#endif
    } else if (strcmp(key, "general.name") == 0) {
#ifdef _MSC_VER
        strncpy_s(s->name, sizeof(s->name), tmp, _TRUNCATE);
#else
        snprintf(s->name, sizeof(s->name), "%s", tmp);
#endif
    }
    return 1;
}

int qx_gguf_inspect(const char *path, qx_gguf_summary *out, char *err, uint64_t err_len) {
    if (!path || !out) { set_err(err, err_len, "invalid argument"); return 0; }
    memset(out, 0, sizeof(*out));
    out->alignment = 32;

    FILE *f = fopen(path, "rb");
    if (!f) { set_errno_err(err, err_len, errno); return 0; }
    char magic[4];
    if (!rd(f, magic, 4) || memcmp(magic, QX_GGUF_MAGIC, 4) != 0) {
        fclose(f); set_err(err, err_len, "bad GGUF magic"); return 0;
    }
    if (!rd_u32(f, &out->version) || !rd_u64(f, &out->tensor_count) || !rd_u64(f, &out->metadata_kv_count)) {
        fclose(f); set_err(err, err_len, "short GGUF header"); return 0;
    }
    if (out->version < 2 || out->version > 3) {
        fclose(f); set_err(err, err_len, "unsupported GGUF version"); return 0;
    }

    for (uint64_t i = 0; i < out->metadata_kv_count; i++) {
        char key[160];
        uint32_t type = 0;
        if (!read_string(f, key, sizeof(key), NULL) || !rd_u32(f, &type)) {
            fclose(f); set_err(err, err_len, "bad GGUF metadata KV"); return 0;
        }
        if (type == QX_GGUF_STRING && (strcmp(key, "general.architecture") == 0 || strcmp(key, "general.name") == 0)) {
            if (!apply_string_meta(f, out, key)) { fclose(f); set_err(err, err_len, "bad GGUF string metadata"); return 0; }
        } else {
            uint64_t num = 0;
            int r = read_u64_value(f, type, &num);
            if (r == 0) { fclose(f); set_err(err, err_len, "unsupported GGUF metadata value"); return 0; }
            if (r == 1) apply_u64_meta(out, key, num);
        }
    }

#if defined(_WIN32)
    out->tensor_info_offset = (uint64_t)_ftelli64(f);
#else
    out->tensor_info_offset = (uint64_t)ftello(f);
#endif

    for (uint64_t i = 0; i < out->tensor_count; i++) {
        qx_gguf_tensor_info t;
        memset(&t, 0, sizeof(t));
        if (!read_string(f, t.name, sizeof(t.name), NULL) || !rd_u32(f, &t.n_dims)) {
            fclose(f); set_err(err, err_len, "bad GGUF tensor info"); return 0;
        }
        if (t.n_dims > QX_GGUF_MAX_DIMS) { fclose(f); set_err(err, err_len, "GGUF tensor rank too high"); return 0; }
        for (uint32_t d = 0; d < t.n_dims; d++) if (!rd_u64(f, &t.dims[d])) { fclose(f); set_err(err, err_len, "bad GGUF tensor dims"); return 0; }
        if (!rd_u32(f, &t.ggml_type) || !rd_u64(f, &t.offset)) { fclose(f); set_err(err, err_len, "bad GGUF tensor tail"); return 0; }
        if (i < 8) out->first_tensors[out->first_tensor_count++] = t;
    }
#if defined(_WIN32)
    uint64_t pos = (uint64_t)_ftelli64(f);
#else
    uint64_t pos = (uint64_t)ftello(f);
#endif
    out->data_offset = qx_align_u64(pos, out->alignment ? out->alignment : 32);
    fclose(f);
    return 1;
}

int qx_gguf_dump_summary(const char *path, FILE *out, char *err, uint64_t err_len) {
    qx_gguf_summary s;
    if (!qx_gguf_inspect(path, &s, err, err_len)) return 0;
    fprintf(out, "{\n");
    fprintf(out, "  \"magic\": \"GGUF\",\n");
    fprintf(out, "  \"version\": %u,\n", s.version);
    fprintf(out, "  \"tensor_count\": %llu,\n", (unsigned long long)s.tensor_count);
    fprintf(out, "  \"metadata_kv_count\": %llu,\n", (unsigned long long)s.metadata_kv_count);
    fprintf(out, "  \"alignment\": %u,\n", s.alignment);
    fprintf(out, "  \"architecture\": \"%s\",\n", s.architecture);
    fprintf(out, "  \"name\": \"%s\",\n", s.name);
    fprintf(out, "  \"qwen3_block_count\": %llu,\n", (unsigned long long)s.qwen3_block_count);
    fprintf(out, "  \"qwen3_expert_count\": %llu,\n", (unsigned long long)s.qwen3_expert_count);
    fprintf(out, "  \"qwen3_expert_used_count\": %llu,\n", (unsigned long long)s.qwen3_expert_used_count);
    fprintf(out, "  \"qwen3_embedding_length\": %llu,\n", (unsigned long long)s.qwen3_embedding_length);
    fprintf(out, "  \"qwen3_feed_forward_length\": %llu,\n", (unsigned long long)s.qwen3_feed_forward_length);
    fprintf(out, "  \"qwen3_attention_head_count\": %llu,\n", (unsigned long long)s.qwen3_attention_head_count);
    fprintf(out, "  \"qwen3_attention_head_count_kv\": %llu,\n", (unsigned long long)s.qwen3_attention_head_count_kv);
    fprintf(out, "  \"tensor_info_offset\": %llu,\n", (unsigned long long)s.tensor_info_offset);
    fprintf(out, "  \"data_offset\": %llu,\n", (unsigned long long)s.data_offset);
    fprintf(out, "  \"first_tensors\": [\n");
    for (uint32_t i = 0; i < s.first_tensor_count; i++) {
        const qx_gguf_tensor_info *t = &s.first_tensors[i];
        fprintf(out, "    {\"name\": \"%s\", \"rank\": %u, \"type\": %u, \"offset\": %llu, \"dims\": [",
                t->name, t->n_dims, t->ggml_type, (unsigned long long)t->offset);
        for (uint32_t d = 0; d < t->n_dims; d++) fprintf(out, "%s%llu", d ? ", " : "", (unsigned long long)t->dims[d]);
        fprintf(out, "]}%s\n", i + 1 == s.first_tensor_count ? "" : ",");
    }
    fprintf(out, "  ]\n");
    fprintf(out, "}\n");
    return 1;
}


typedef struct offset_idx {
    uint64_t offset;
    uint64_t index;
} offset_idx;

static int cmp_offset_idx(const void *a, const void *b) {
    const offset_idx *aa = (const offset_idx *)a;
    const offset_idx *bb = (const offset_idx *)b;
    if (aa->offset < bb->offset) return -1;
    if (aa->offset > bb->offset) return 1;
    return 0;
}

int qx_gguf_load_tensor_table(const char *path, qx_gguf_summary *summary, qx_gguf_tensor_table *table, char *err, uint64_t err_len) {
    if (!path || !table) { set_err(err, err_len, "invalid argument"); return 0; }
    memset(table, 0, sizeof(*table));
    qx_gguf_summary local;
    qx_gguf_summary *s = summary ? summary : &local;
    if (!qx_gguf_inspect(path, s, err, err_len)) return 0;

    FILE *f = fopen(path, "rb");
    if (!f) { set_errno_err(err, err_len, errno); return 0; }
    table->file_size = file_size_of(f);
    table->data_offset = s->data_offset;
    table->tensor_count = s->tensor_count;
    if (table->file_size < table->data_offset) { fclose(f); set_err(err, err_len, "GGUF data offset past EOF"); return 0; }

    table->tensors = (qx_gguf_tensor_info *)calloc((size_t)table->tensor_count, sizeof(qx_gguf_tensor_info));
    offset_idx *order = (offset_idx *)calloc((size_t)table->tensor_count, sizeof(offset_idx));
    if (!table->tensors || !order) {
        free(table->tensors); free(order); fclose(f); memset(table, 0, sizeof(*table));
        set_err(err, err_len, "out of memory"); return 0;
    }

#if defined(_WIN32)
    if (_fseeki64(f, (int64_t)s->tensor_info_offset, SEEK_SET) != 0) {
#else
    if (fseeko(f, (off_t)s->tensor_info_offset, SEEK_SET) != 0) {
#endif
        qx_gguf_free_tensor_table(table); free(order); fclose(f); set_err(err, err_len, "seek tensor table failed"); return 0;
    }

    for (uint64_t i = 0; i < table->tensor_count; i++) {
        qx_gguf_tensor_info *t = &table->tensors[i];
        if (!read_string(f, t->name, sizeof(t->name), NULL) || !rd_u32(f, &t->n_dims)) {
            qx_gguf_free_tensor_table(table); free(order); fclose(f); set_err(err, err_len, "bad GGUF tensor info"); return 0;
        }
        if (t->n_dims > QX_GGUF_MAX_DIMS) {
            qx_gguf_free_tensor_table(table); free(order); fclose(f); set_err(err, err_len, "GGUF tensor rank too high"); return 0;
        }
        for (uint32_t d = 0; d < t->n_dims; d++) if (!rd_u64(f, &t->dims[d])) {
            qx_gguf_free_tensor_table(table); free(order); fclose(f); set_err(err, err_len, "bad GGUF tensor dims"); return 0;
        }
        if (!rd_u32(f, &t->ggml_type) || !rd_u64(f, &t->offset)) {
            qx_gguf_free_tensor_table(table); free(order); fclose(f); set_err(err, err_len, "bad GGUF tensor tail"); return 0;
        }
        if (table->data_offset + t->offset > table->file_size) {
            qx_gguf_free_tensor_table(table); free(order); fclose(f); set_err(err, err_len, "GGUF tensor offset past EOF"); return 0;
        }
        order[i].offset = t->offset;
        order[i].index = i;
    }
    qsort(order, (size_t)table->tensor_count, sizeof(offset_idx), cmp_offset_idx);
    for (uint64_t i = 0; i < table->tensor_count; i++) {
        uint64_t idx = order[i].index;
        uint64_t cur = table->tensors[idx].offset;
        uint64_t next = (i + 1 < table->tensor_count) ? order[i + 1].offset : (table->file_size - table->data_offset);
        if (next < cur) { qx_gguf_free_tensor_table(table); free(order); fclose(f); set_err(err, err_len, "GGUF tensor offsets not monotonic"); return 0; }
        table->tensors[idx].byte_size = next - cur;
    }
    free(order);
    fclose(f);
    return 1;
}

void qx_gguf_free_tensor_table(qx_gguf_tensor_table *table) {
    if (!table) return;
    free(table->tensors);
    memset(table, 0, sizeof(*table));
}

static void write_tsv_escaped(FILE *out, const char *s) {
    for (const unsigned char *p = (const unsigned char *)s; p && *p; ++p) {
        if (*p == '\\') fputs("\\\\", out);
        else if (*p == '\t') fputs("\\t", out);
        else if (*p == '\n') fputs("\\n", out);
        else if (*p == '\r') fputs("\\r", out);
        else fputc(*p, out);
    }
}

int qx_gguf_export_tokenizer_tokens(const char *gguf_path, const char *out_path, uint64_t *count_out, char *err, uint64_t err_len) {
    if (!gguf_path || !out_path) { set_err(err, err_len, "invalid argument"); return 0; }
    if (count_out) *count_out = 0;
    FILE *f = fopen(gguf_path, "rb");
    if (!f) { set_errno_err(err, err_len, errno); return 0; }
    char magic[4];
    uint32_t version = 0;
    uint64_t tensor_count = 0, kv_count = 0;
    if (!rd(f, magic, 4) || memcmp(magic, QX_GGUF_MAGIC, 4) != 0 || !rd_u32(f, &version) || !rd_u64(f, &tensor_count) || !rd_u64(f, &kv_count)) {
        fclose(f); set_err(err, err_len, "bad GGUF header"); return 0;
    }
    (void)tensor_count;
    for (uint64_t i = 0; i < kv_count; ++i) {
        char key[160];
        uint32_t type = 0;
        if (!read_string(f, key, sizeof(key), NULL) || !rd_u32(f, &type)) { fclose(f); set_err(err, err_len, "bad GGUF metadata KV"); return 0; }
        if (strcmp(key, "tokenizer.ggml.tokens") == 0 && type == QX_GGUF_ARRAY) {
            uint32_t elem_type = 0;
            uint64_t count = 0;
            if (!rd_u32(f, &elem_type) || !rd_u64(f, &count) || elem_type != QX_GGUF_STRING) { fclose(f); set_err(err, err_len, "tokenizer tokens metadata is not string array"); return 0; }
            FILE *out = fopen(out_path, "wb");
            if (!out) { fclose(f); set_errno_err(err, err_len, errno); return 0; }
            fprintf(out, "# qx-tokenizer-v1\n");
            char *tmp = (char *)malloc(65536);
            if (!tmp) { fclose(out); fclose(f); set_err(err, err_len, "out of memory"); return 0; }
            for (uint64_t j = 0; j < count; ++j) {
                uint64_t slen = 0;
                if (!read_string(f, tmp, 65536, &slen)) { free(tmp); fclose(out); fclose(f); set_err(err, err_len, "bad tokenizer token string"); return 0; }
                fprintf(out, "%llu\t", (unsigned long long)j);
                write_tsv_escaped(out, tmp);
                fputc('\n', out);
            }
            free(tmp);
            fclose(out);
            fclose(f);
            if (count_out) *count_out = count;
            return 1;
        }
        if (!skip_value(f, type)) { fclose(f); set_err(err, err_len, "unsupported GGUF metadata value"); return 0; }
    }
    fclose(f);
    set_err(err, err_len, "tokenizer.ggml.tokens not found");
    return 0;
}
