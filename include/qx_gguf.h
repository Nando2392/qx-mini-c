#ifndef QX_GGUF_H
#define QX_GGUF_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QX_GGUF_MAGIC "GGUF"
#define QX_GGUF_NAME_MAX 160u
#define QX_GGUF_MAX_DIMS 4u

typedef enum qx_gguf_type {
    QX_GGUF_UINT8 = 0,
    QX_GGUF_INT8 = 1,
    QX_GGUF_UINT16 = 2,
    QX_GGUF_INT16 = 3,
    QX_GGUF_UINT32 = 4,
    QX_GGUF_INT32 = 5,
    QX_GGUF_FLOAT32 = 6,
    QX_GGUF_BOOL = 7,
    QX_GGUF_STRING = 8,
    QX_GGUF_ARRAY = 9,
    QX_GGUF_UINT64 = 10,
    QX_GGUF_INT64 = 11,
    QX_GGUF_FLOAT64 = 12
} qx_gguf_type;

typedef struct qx_gguf_tensor_info {
    char name[QX_GGUF_NAME_MAX];
    uint32_t n_dims;
    uint64_t dims[QX_GGUF_MAX_DIMS];
    uint32_t ggml_type;
    uint64_t offset;
    uint64_t byte_size;
} qx_gguf_tensor_info;

typedef struct qx_gguf_tensor_table {
    uint64_t tensor_count;
    uint64_t data_offset;
    uint64_t file_size;
    qx_gguf_tensor_info *tensors;
} qx_gguf_tensor_table;

typedef struct qx_gguf_summary {
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
    uint32_t alignment;
    uint64_t tensor_info_offset;
    uint64_t data_offset;
    char architecture[64];
    char name[128];
    uint64_t qwen3_block_count;
    uint64_t qwen3_expert_count;
    uint64_t qwen3_expert_used_count;
    uint64_t qwen3_embedding_length;
    uint64_t qwen3_feed_forward_length;
    uint64_t qwen3_attention_head_count;
    uint64_t qwen3_attention_head_count_kv;
    uint32_t first_tensor_count;
    qx_gguf_tensor_info first_tensors[8];
} qx_gguf_summary;

const char *qx_gguf_value_type_name(uint32_t t);
int qx_gguf_inspect(const char *path, qx_gguf_summary *out, char *err, uint64_t err_len);
int qx_gguf_dump_summary(const char *path, FILE *out, char *err, uint64_t err_len);
int qx_gguf_load_tensor_table(const char *path, qx_gguf_summary *summary, qx_gguf_tensor_table *table, char *err, uint64_t err_len);
void qx_gguf_free_tensor_table(qx_gguf_tensor_table *table);
int qx_gguf_export_tokenizer_tokens(const char *gguf_path, const char *out_path, uint64_t *count_out, char *err, uint64_t err_len);

#ifdef __cplusplus
}
#endif

#endif
