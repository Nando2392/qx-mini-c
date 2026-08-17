#ifndef QX_TOKENIZER_H
#define QX_TOKENIZER_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QX_TOKENIZER_MAX_ITEMS 1000000u
#define QX_TOKENIZER_MAX_PAYLOAD (256u * 1024u * 1024u)
#define QX_TOKENIZER_MAX_INPUT 4096u

typedef struct qx_token_entry {
    const unsigned char *text;
    uint32_t length;
    uint32_t type;
} qx_token_entry;

typedef struct qx_merge_entry {
    const unsigned char *left;
    const unsigned char *right;
    uint32_t left_length;
    uint32_t right_length;
    uint32_t rank;
} qx_merge_entry;

typedef struct qx_tokenizer {
    unsigned char *storage;
    uint32_t storage_size;
    const unsigned char *model;
    const unsigned char *pre;
    uint32_t model_length;
    uint32_t pre_length;
    uint32_t vocab_count;
    uint32_t merge_count;
    int32_t bos_token_id;
    int32_t eos_token_id;
    uint32_t flags;
    uint64_t payload_checksum;
    qx_token_entry *tokens;
    qx_merge_entry *merges;
} qx_tokenizer;

int qx_tokenizer_load(const char *path, qx_tokenizer *out, char *err, uint64_t err_len);
void qx_tokenizer_free(qx_tokenizer *tokenizer);
int qx_tokenizer_dump_summary(const char *path, FILE *out, char *err, uint64_t err_len);
int qx_tokenizer_encode(
    const qx_tokenizer *tokenizer,
    const unsigned char *input,
    uint32_t input_length,
    int parse_special,
    uint32_t *token_ids,
    uint32_t token_capacity,
    uint32_t *token_count,
    char *err,
    uint64_t err_len);
int qx_tokenizer_decode(
    const qx_tokenizer *tokenizer,
    const uint32_t *token_ids,
    uint32_t token_count,
    int render_special,
    unsigned char *output,
    uint32_t output_capacity,
    uint32_t *output_length,
    char *err,
    uint64_t err_len);

#ifdef __cplusplus
}
#endif

#endif
