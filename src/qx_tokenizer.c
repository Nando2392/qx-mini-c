#include "qx_tokenizer.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#define QX_TOKENIZER_HEADER_SIZE 48u

static int qx_tok_utf8_next(const unsigned char *text, uint32_t length, uint32_t offset, uint32_t *codepoint, uint32_t *width);
static int qx_tok_valid_utf8(const unsigned char *text, uint32_t length);

static void qx_tok_set_err(char *err, uint64_t err_len, const char *message) {
    if (!err || err_len == 0) return;
#ifdef _MSC_VER
    strncpy_s(err, (size_t)err_len, message, _TRUNCATE);
#else
    snprintf(err, (size_t)err_len, "%s", message);
#endif
}

static void qx_tok_set_errno(char *err, uint64_t err_len, int code) {
    if (!err || err_len == 0) return;
#ifdef _MSC_VER
    strerror_s(err, (size_t)err_len, code);
#else
    snprintf(err, (size_t)err_len, "%s", strerror(code));
#endif
}

static uint32_t qx_tok_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int32_t qx_tok_i32(const unsigned char *p) {
    uint32_t value = qx_tok_u32(p);
    int32_t result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

static uint64_t qx_tok_u64(const unsigned char *p) {
    return (uint64_t)qx_tok_u32(p) | ((uint64_t)qx_tok_u32(p + 4) << 32);
}

static uint64_t qx_tok_fnv1a64(const unsigned char *data, uint64_t length) {
    uint64_t value = 1469598103934665603ull;
    for (uint64_t i = 0; i < length; ++i) {
        value ^= data[i];
        value *= 1099511628211ull;
    }
    return value;
}

static int qx_tok_take(const unsigned char **cursor, const unsigned char *end, uint32_t length, const unsigned char **value) {
    if ((uint64_t)(end - *cursor) < (uint64_t)length) return 0;
    if (value) *value = *cursor;
    *cursor += length;
    return 1;
}

void qx_tokenizer_free(qx_tokenizer *tokenizer) {
    if (!tokenizer) return;
    free(tokenizer->tokens);
    free(tokenizer->merges);
    free(tokenizer->storage);
    memset(tokenizer, 0, sizeof(*tokenizer));
}

int qx_tokenizer_load(const char *path, qx_tokenizer *out, char *err, uint64_t err_len) {
    if (!path || !out) { qx_tok_set_err(err, err_len, "invalid tokenizer argument"); return 0; }
    memset(out, 0, sizeof(*out));
    FILE *file = fopen(path, "rb");
    if (!file) { qx_tok_set_errno(err, err_len, errno); return 0; }
#if defined(_WIN32)
    if (_fseeki64(file, 0, SEEK_END) != 0) { fclose(file); qx_tok_set_err(err, err_len, "tokenizer seek failed"); return 0; }
    int64_t signed_size = _ftelli64(file);
    if (signed_size < 0 || _fseeki64(file, 0, SEEK_SET) != 0) { fclose(file); qx_tok_set_err(err, err_len, "tokenizer size failed"); return 0; }
#else
    if (fseeko(file, 0, SEEK_END) != 0) { fclose(file); qx_tok_set_err(err, err_len, "tokenizer seek failed"); return 0; }
    off_t signed_size = ftello(file);
    if (signed_size < 0 || fseeko(file, 0, SEEK_SET) != 0) { fclose(file); qx_tok_set_err(err, err_len, "tokenizer size failed"); return 0; }
#endif
    uint64_t file_size = (uint64_t)signed_size;
    if (file_size < QX_TOKENIZER_HEADER_SIZE || file_size > (uint64_t)QX_TOKENIZER_HEADER_SIZE + QX_TOKENIZER_MAX_PAYLOAD) {
        fclose(file); qx_tok_set_err(err, err_len, "tokenizer size outside limits"); return 0;
    }
    unsigned char header[QX_TOKENIZER_HEADER_SIZE];
    if (fread(header, 1, sizeof(header), file) != sizeof(header)) { fclose(file); qx_tok_set_err(err, err_len, "short tokenizer header"); return 0; }
    if (memcmp(header, "QXT2", 4) != 0) { fclose(file); qx_tok_set_err(err, err_len, "bad tokenizer magic"); return 0; }
    uint32_t version = qx_tok_u32(header + 4);
    uint32_t vocab_count = qx_tok_u32(header + 8);
    uint32_t merge_count = qx_tok_u32(header + 12);
    uint32_t model_length = qx_tok_u32(header + 16);
    uint32_t pre_length = qx_tok_u32(header + 20);
    int32_t bos_token_id = qx_tok_i32(header + 24);
    int32_t eos_token_id = qx_tok_i32(header + 28);
    uint32_t flags = qx_tok_u32(header + 32);
    uint32_t payload_size = qx_tok_u32(header + 36);
    uint64_t stored_checksum = qx_tok_u64(header + 40);
    if (version != 2u) { fclose(file); qx_tok_set_err(err, err_len, "unsupported tokenizer version"); return 0; }
    if (vocab_count == 0u || vocab_count > QX_TOKENIZER_MAX_ITEMS || merge_count > QX_TOKENIZER_MAX_ITEMS) {
        fclose(file); qx_tok_set_err(err, err_len, "tokenizer item count outside limits"); return 0;
    }
    if (model_length == 0u || model_length > 64u || pre_length == 0u || pre_length > 64u || (flags & ~3u) != 0u) {
        fclose(file); qx_tok_set_err(err, err_len, "invalid tokenizer metadata"); return 0;
    }
    if ((uint64_t)payload_size != file_size - QX_TOKENIZER_HEADER_SIZE) {
        fclose(file); qx_tok_set_err(err, err_len, "tokenizer size mismatch"); return 0;
    }
    if (bos_token_id < -1 || eos_token_id < -1 || bos_token_id >= (int32_t)vocab_count || eos_token_id >= (int32_t)vocab_count) {
        fclose(file); qx_tok_set_err(err, err_len, "special token id out of range"); return 0;
    }
    unsigned char *storage = (unsigned char *)malloc(payload_size ? payload_size : 1u);
    qx_token_entry *tokens = (qx_token_entry *)calloc(vocab_count, sizeof(qx_token_entry));
    qx_merge_entry *merges = merge_count ? (qx_merge_entry *)calloc(merge_count, sizeof(qx_merge_entry)) : NULL;
    if (!storage || !tokens || (merge_count && !merges)) {
        free(storage); free(tokens); free(merges); fclose(file); qx_tok_set_err(err, err_len, "out of memory"); return 0;
    }
    if (payload_size && fread(storage, 1, payload_size, file) != payload_size) {
        free(storage); free(tokens); free(merges); fclose(file); qx_tok_set_err(err, err_len, "short tokenizer payload"); return 0;
    }
    fclose(file);
    if (qx_tok_fnv1a64(storage, payload_size) != stored_checksum) {
        free(storage); free(tokens); free(merges); qx_tok_set_err(err, err_len, "tokenizer checksum mismatch"); return 0;
    }
    const unsigned char *cursor = storage;
    const unsigned char *end = storage + payload_size;
    const unsigned char *model = NULL;
    const unsigned char *pre = NULL;
    if (!qx_tok_take(&cursor, end, model_length, &model) || !qx_tok_take(&cursor, end, pre_length, &pre)) {
        free(storage); free(tokens); free(merges); qx_tok_set_err(err, err_len, "tokenizer metadata exceeds payload"); return 0;
    }
    if (model_length != 4u || memcmp(model, "gpt2", 4) != 0 || pre_length != 5u || memcmp(pre, "qwen2", 5) != 0) {
        free(storage); free(tokens); free(merges); qx_tok_set_err(err, err_len, "unsupported tokenizer model or pre-tokenizer"); return 0;
    }
    for (uint32_t i = 0; i < vocab_count; ++i) {
        if ((uint64_t)(end - cursor) < 8u) { free(storage); free(tokens); free(merges); qx_tok_set_err(err, err_len, "truncated tokenizer token table"); return 0; }
        uint32_t length = qx_tok_u32(cursor);
        uint32_t type = qx_tok_u32(cursor + 4);
        cursor += 8;
        if (length > (1u << 20) || type < 1u || type > 6u || !qx_tok_take(&cursor, end, length, &tokens[i].text)) {
            free(storage); free(tokens); free(merges); qx_tok_set_err(err, err_len, "invalid tokenizer token entry"); return 0;
        }
        if (memchr(tokens[i].text, 0, length) != NULL) { free(storage); free(tokens); free(merges); qx_tok_set_err(err, err_len, "NUL in tokenizer token"); return 0; }
        if (!qx_tok_valid_utf8(tokens[i].text, length)) { free(storage); free(tokens); free(merges); qx_tok_set_err(err, err_len, "invalid UTF-8 in tokenizer token"); return 0; }
        tokens[i].length = length;
        tokens[i].type = type;
    }
    for (uint32_t i = 0; i < merge_count; ++i) {
        if ((uint64_t)(end - cursor) < 8u) { free(storage); free(tokens); free(merges); qx_tok_set_err(err, err_len, "truncated tokenizer merge table"); return 0; }
        uint32_t left_length = qx_tok_u32(cursor);
        uint32_t right_length = qx_tok_u32(cursor + 4);
        cursor += 8;
        if (left_length == 0u || right_length == 0u || left_length > (1u << 20) || right_length > (1u << 20) ||
            !qx_tok_take(&cursor, end, left_length, &merges[i].left) || !qx_tok_take(&cursor, end, right_length, &merges[i].right)) {
            free(storage); free(tokens); free(merges); qx_tok_set_err(err, err_len, "invalid tokenizer merge entry"); return 0;
        }
        if (memchr(merges[i].left, 0, left_length) != NULL || memchr(merges[i].right, 0, right_length) != NULL) {
            free(storage); free(tokens); free(merges); qx_tok_set_err(err, err_len, "NUL in tokenizer merge"); return 0;
        }
        if (!qx_tok_valid_utf8(merges[i].left, left_length) || !qx_tok_valid_utf8(merges[i].right, right_length)) {
            free(storage); free(tokens); free(merges); qx_tok_set_err(err, err_len, "invalid UTF-8 in tokenizer merge"); return 0;
        }
        merges[i].left_length = left_length;
        merges[i].right_length = right_length;
        merges[i].rank = i;
    }
    if (cursor != end) { free(storage); free(tokens); free(merges); qx_tok_set_err(err, err_len, "trailing tokenizer payload bytes"); return 0; }
    out->storage = storage;
    out->storage_size = payload_size;
    out->model = model;
    out->pre = pre;
    out->model_length = model_length;
    out->pre_length = pre_length;
    out->vocab_count = vocab_count;
    out->merge_count = merge_count;
    out->bos_token_id = bos_token_id;
    out->eos_token_id = eos_token_id;
    out->flags = flags;
    out->payload_checksum = stored_checksum;
    out->tokens = tokens;
    out->merges = merges;
    return 1;
}

int qx_tokenizer_dump_summary(const char *path, FILE *out, char *err, uint64_t err_len) {
    if (!out) { qx_tok_set_err(err, err_len, "invalid tokenizer output"); return 0; }
    qx_tokenizer tokenizer;
    if (!qx_tokenizer_load(path, &tokenizer, err, err_len)) return 0;
    fprintf(out, "{\n");
    fprintf(out, "  \"format\": \"QXT2\",\n");
    fprintf(out, "  \"version\": 2,\n");
    fprintf(out, "  \"model\": \"gpt2\",\n");
    fprintf(out, "  \"pre\": \"qwen2\",\n");
    fprintf(out, "  \"vocab_count\": %u,\n", tokenizer.vocab_count);
    fprintf(out, "  \"merge_count\": %u,\n", tokenizer.merge_count);
    fprintf(out, "  \"bos_token_id\": %d,\n", tokenizer.bos_token_id);
    fprintf(out, "  \"eos_token_id\": %d,\n", tokenizer.eos_token_id);
    fprintf(out, "  \"add_bos\": %s,\n", (tokenizer.flags & 1u) ? "true" : "false");
    fprintf(out, "  \"add_eos\": %s,\n", (tokenizer.flags & 2u) ? "true" : "false");
    fprintf(out, "  \"checksum_verified\": true\n");
    fprintf(out, "}\n");
    qx_tokenizer_free(&tokenizer);
    return 1;
}

typedef struct qx_tok_index {
    uint32_t *slots;
    uint32_t mask;
} qx_tok_index;

typedef struct qx_tok_symbol {
    unsigned char *text;
    uint32_t length;
} qx_tok_symbol;

enum qx_tok_class {
    QX_TOK_LETTER,
    QX_TOK_NUMBER,
    QX_TOK_SPACE,
    QX_TOK_NEWLINE,
    QX_TOK_OTHER,
    QX_TOK_UNSUPPORTED
};

static uint64_t qx_tok_hash_bytes(const unsigned char *data, uint32_t length) {
    return qx_tok_fnv1a64(data, length);
}

static uint64_t qx_tok_hash_pair(
    const unsigned char *left,
    uint32_t left_length,
    const unsigned char *right,
    uint32_t right_length) {
    uint64_t value = qx_tok_fnv1a64(left, left_length);
    value ^= 0u;
    value *= 1099511628211ull;
    for (uint32_t i = 0; i < right_length; ++i) {
        value ^= right[i];
        value *= 1099511628211ull;
    }
    return value;
}

static int qx_tok_index_init(qx_tok_index *index, uint32_t count, char *err, uint64_t err_len) {
    uint32_t size = 1u;
    while (size < count * 2u) {
        if (size > (1u << 30)) { qx_tok_set_err(err, err_len, "tokenizer hash size overflow"); return 0; }
        size <<= 1;
    }
    index->slots = (uint32_t *)calloc(size, sizeof(uint32_t));
    if (!index->slots) { qx_tok_set_err(err, err_len, "out of memory"); return 0; }
    index->mask = size - 1u;
    return 1;
}

static void qx_tok_index_free(qx_tok_index *index) {
    free(index->slots);
    index->slots = NULL;
    index->mask = 0u;
}

static int qx_tok_build_token_index(const qx_tokenizer *tokenizer, qx_tok_index *index, char *err, uint64_t err_len) {
    if (!qx_tok_index_init(index, tokenizer->vocab_count, err, err_len)) return 0;
    for (uint32_t id = 0; id < tokenizer->vocab_count; ++id) {
        uint32_t slot = (uint32_t)qx_tok_hash_bytes(tokenizer->tokens[id].text, tokenizer->tokens[id].length) & index->mask;
        while (index->slots[slot] != 0u) {
            uint32_t prior = index->slots[slot] - 1u;
            if (tokenizer->tokens[prior].length == tokenizer->tokens[id].length &&
                memcmp(tokenizer->tokens[prior].text, tokenizer->tokens[id].text, tokenizer->tokens[id].length) == 0) {
                qx_tok_index_free(index); qx_tok_set_err(err, err_len, "duplicate tokenizer token"); return 0;
            }
            slot = (slot + 1u) & index->mask;
        }
        index->slots[slot] = id + 1u;
    }
    return 1;
}

static int qx_tok_build_merge_index(const qx_tokenizer *tokenizer, qx_tok_index *index, char *err, uint64_t err_len) {
    if (!qx_tok_index_init(index, tokenizer->merge_count ? tokenizer->merge_count : 1u, err, err_len)) return 0;
    for (uint32_t rank = 0; rank < tokenizer->merge_count; ++rank) {
        const qx_merge_entry *merge = &tokenizer->merges[rank];
        uint32_t slot = (uint32_t)qx_tok_hash_pair(merge->left, merge->left_length, merge->right, merge->right_length) & index->mask;
        while (index->slots[slot] != 0u) {
            uint32_t prior_rank = index->slots[slot] - 1u;
            const qx_merge_entry *prior = &tokenizer->merges[prior_rank];
            if (prior->left_length == merge->left_length && prior->right_length == merge->right_length &&
                memcmp(prior->left, merge->left, merge->left_length) == 0 &&
                memcmp(prior->right, merge->right, merge->right_length) == 0) {
                qx_tok_index_free(index); qx_tok_set_err(err, err_len, "duplicate tokenizer merge"); return 0;
            }
            slot = (slot + 1u) & index->mask;
        }
        index->slots[slot] = rank + 1u;
    }
    return 1;
}

static int qx_tok_find_token(const qx_tokenizer *tokenizer, const qx_tok_index *index, const unsigned char *text, uint32_t length, uint32_t *token_id) {
    uint32_t slot = (uint32_t)qx_tok_hash_bytes(text, length) & index->mask;
    while (index->slots[slot] != 0u) {
        uint32_t id = index->slots[slot] - 1u;
        if (tokenizer->tokens[id].length == length && memcmp(tokenizer->tokens[id].text, text, length) == 0) {
            *token_id = id;
            return 1;
        }
        slot = (slot + 1u) & index->mask;
    }
    return 0;
}

static int qx_tok_find_merge(
    const qx_tokenizer *tokenizer,
    const qx_tok_index *index,
    const qx_tok_symbol *left,
    const qx_tok_symbol *right,
    uint32_t *rank) {
    uint32_t slot = (uint32_t)qx_tok_hash_pair(left->text, left->length, right->text, right->length) & index->mask;
    while (index->slots[slot] != 0u) {
        uint32_t candidate_rank = index->slots[slot] - 1u;
        const qx_merge_entry *merge = &tokenizer->merges[candidate_rank];
        if (merge->left_length == left->length && merge->right_length == right->length &&
            memcmp(merge->left, left->text, left->length) == 0 && memcmp(merge->right, right->text, right->length) == 0) {
            *rank = candidate_rank;
            return 1;
        }
        slot = (slot + 1u) & index->mask;
    }
    return 0;
}

static int qx_tok_utf8_next(const unsigned char *text, uint32_t length, uint32_t offset, uint32_t *codepoint, uint32_t *width) {
    if (offset >= length) return 0;
    unsigned char a = text[offset];
    if (a < 0x80u) { *codepoint = a; *width = 1u; return 1; }
    if (a >= 0xC2u && a <= 0xDFu && length - offset >= 2u && (text[offset + 1] & 0xC0u) == 0x80u) {
        *codepoint = ((uint32_t)(a & 0x1Fu) << 6) | (uint32_t)(text[offset + 1] & 0x3Fu); *width = 2u; return 1;
    }
    if (a >= 0xE0u && a <= 0xEFu && length - offset >= 3u && (text[offset + 1] & 0xC0u) == 0x80u && (text[offset + 2] & 0xC0u) == 0x80u) {
        uint32_t cp = ((uint32_t)(a & 0x0Fu) << 12) | ((uint32_t)(text[offset + 1] & 0x3Fu) << 6) | (uint32_t)(text[offset + 2] & 0x3Fu);
        if (cp < 0x800u || (cp >= 0xD800u && cp <= 0xDFFFu)) return 0;
        *codepoint = cp; *width = 3u; return 1;
    }
    if (a >= 0xF0u && a <= 0xF4u && length - offset >= 4u && (text[offset + 1] & 0xC0u) == 0x80u &&
        (text[offset + 2] & 0xC0u) == 0x80u && (text[offset + 3] & 0xC0u) == 0x80u) {
        uint32_t cp = ((uint32_t)(a & 7u) << 18) | ((uint32_t)(text[offset + 1] & 0x3Fu) << 12) |
            ((uint32_t)(text[offset + 2] & 0x3Fu) << 6) | (uint32_t)(text[offset + 3] & 0x3Fu);
        if (cp < 0x10000u || cp > 0x10FFFFu) return 0;
        *codepoint = cp; *width = 4u; return 1;
    }
    return 0;
}

static int qx_tok_valid_utf8(const unsigned char *text, uint32_t length) {
    uint32_t offset = 0u;
    while (offset < length) {
        uint32_t codepoint = 0u;
        uint32_t width = 0u;
        if (!qx_tok_utf8_next(text, length, offset, &codepoint, &width)) return 0;
        offset += width;
    }
    return 1;
}

static int qx_tok_append(
    unsigned char *output,
    uint32_t output_capacity,
    uint32_t *offset,
    const unsigned char *data,
    uint32_t length) {
    if (*offset > output_capacity || length > output_capacity - *offset) return 0;
    if (length != 0u) memcpy(output + *offset, data, length);
    *offset += length;
    return 1;
}

int qx_tokenizer_render_qwen3_chat(
    const qx_chat_message *messages,
    uint32_t message_count,
    int add_generation_prompt,
    unsigned char *output,
    uint32_t output_capacity,
    uint32_t *output_length,
    char *err,
    uint64_t err_len) {
    static const unsigned char start[] = "<|im_start|>";
    static const unsigned char end[] = "<|im_end|>\n";
    static const unsigned char assistant[] = "<|im_start|>assistant\n";
    if (!messages || !output || !output_length || message_count == 0u || message_count > QX_CHAT_MAX_MESSAGES) {
        qx_tok_set_err(err, err_len, "invalid chat message array"); return 0;
    }
    if (output_capacity > QX_CHAT_MAX_OUTPUT) output_capacity = QX_CHAT_MAX_OUTPUT;
    uint32_t offset = 0u;
    for (uint32_t i = 0u; i < message_count; ++i) {
        const qx_chat_message *message = &messages[i];
        uint32_t role_length = 0u;
        if (!message->role || !message->content) {
            qx_tok_set_err(err, err_len, "invalid chat message"); return 0;
        }
        if (strcmp(message->role, "system") == 0) role_length = 6u;
        else if (strcmp(message->role, "user") == 0) role_length = 4u;
        else if (strcmp(message->role, "assistant") == 0) role_length = 9u;
        else { qx_tok_set_err(err, err_len, "unsupported chat role"); return 0; }
        if (!qx_tok_valid_utf8(message->content, message->content_length)) {
            qx_tok_set_err(err, err_len, "chat content is not valid UTF-8"); return 0;
        }
        if (!qx_tok_append(output, output_capacity, &offset, start, (uint32_t)sizeof(start) - 1u) ||
            !qx_tok_append(output, output_capacity, &offset, (const unsigned char *)message->role, role_length) ||
            !qx_tok_append(output, output_capacity, &offset, (const unsigned char *)"\n", 1u) ||
            !qx_tok_append(output, output_capacity, &offset, message->content, message->content_length) ||
            !qx_tok_append(output, output_capacity, &offset, end, (uint32_t)sizeof(end) - 1u)) {
            qx_tok_set_err(err, err_len, "rendered chat exceeds output limit"); return 0;
        }
    }
    if (add_generation_prompt &&
        !qx_tok_append(output, output_capacity, &offset, assistant, (uint32_t)sizeof(assistant) - 1u)) {
        qx_tok_set_err(err, err_len, "rendered chat exceeds output limit"); return 0;
    }
    *output_length = offset;
    return 1;
}

static enum qx_tok_class qx_tok_classify(uint32_t cp) {
    if (cp == '\r' || cp == '\n') return QX_TOK_NEWLINE;
    if (cp == ' ' || cp == '\t' || cp == '\v' || cp == '\f') return QX_TOK_SPACE;
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z') ||
        (cp >= 0x00C0u && cp <= 0x02AFu) || (cp >= 0x0370u && cp <= 0x052Fu) ||
        (cp >= 0x0620u && cp <= 0x065Fu) ||
        (cp >= 0x066Eu && cp <= 0x06D3u) ||
        (cp >= 0x06D5u && cp <= 0x06DCu) ||
        (cp >= 0x06DFu && cp <= 0x06E8u) ||
        (cp >= 0x06EAu && cp <= 0x06EDu) ||
        (cp >= 0x3040u && cp <= 0x30FFu) || (cp >= 0x3400u && cp <= 0x4DBFu) ||
        (cp >= 0x4E00u && cp <= 0x9FFFu) || (cp >= 0xAC00u && cp <= 0xD7AFu) ||
        (cp >= 0xF900u && cp <= 0xFAFFu)) return QX_TOK_LETTER;
    if ((cp >= '0' && cp <= '9') || (cp >= 0x0660u && cp <= 0x0669u) ||
        (cp >= 0x06F0u && cp <= 0x06F9u) || (cp >= 0x0966u && cp <= 0x096Fu) ||
        (cp >= 0xFF10u && cp <= 0xFF19u)) return QX_TOK_NUMBER;
    if ((cp >= 0x21u && cp <= 0x7Eu) || (cp >= 0x00A1u && cp <= 0x00BFu) ||
        (cp >= 0x0300u && cp <= 0x036Fu) || (cp >= 0x1AB0u && cp <= 0x1AFFu) ||
        (cp >= 0x1DC0u && cp <= 0x1DFFu) || (cp >= 0x20D0u && cp <= 0x20FFu) ||
        (cp >= 0xFE20u && cp <= 0xFE2Fu) ||
        (cp >= 0x2000u && cp <= 0x206Fu) || (cp >= 0x20A0u && cp <= 0x20CFu) ||
        (cp >= 0x0600u && cp <= 0x061Fu) || (cp >= 0x066Au && cp <= 0x066Du) ||
        cp == 0x06D4u || cp == 0x06DDu || cp == 0x06DEu ||
        (cp >= 0x2190u && cp <= 0x2BFFu) || (cp >= 0x3000u && cp <= 0x303Fu) ||
        (cp >= 0xFF01u && cp <= 0xFF0Fu) || (cp >= 0xFF1Au && cp <= 0xFF20u) ||
        (cp >= 0xFF3Bu && cp <= 0xFF40u) || (cp >= 0xFF5Bu && cp <= 0xFF65u) ||
        (cp >= 0x1F000u && cp <= 0x1FAFFu)) return QX_TOK_OTHER;
    return QX_TOK_UNSUPPORTED;
}

static int qx_tok_char_at(const unsigned char *text, uint32_t length, uint32_t offset, enum qx_tok_class *kind, uint32_t *width) {
    uint32_t cp = 0u;
    if (!qx_tok_utf8_next(text, length, offset, &cp, width)) return 0;
    *kind = qx_tok_classify(cp);
    return *kind != QX_TOK_UNSUPPORTED;
}

static int qx_tok_ascii_contract(const unsigned char *text, uint32_t length, uint32_t offset, uint32_t *matched) {
    static const char *suffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
    if (offset >= length || text[offset] != '\'') return 0;
    for (uint32_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); ++i) {
        uint32_t n = (uint32_t)strlen(suffixes[i]);
        if (length - offset < n + 1u) continue;
        int same = 1;
        for (uint32_t j = 0; j < n; ++j) {
            unsigned char c = text[offset + 1u + j];
            if (c >= 'A' && c <= 'Z') c = (unsigned char)(c + ('a' - 'A'));
            if (c != (unsigned char)suffixes[i][j]) { same = 0; break; }
        }
        if (same) { *matched = n + 1u; return 1; }
    }
    return 0;
}

static int qx_tok_pretoken_length(const unsigned char *text, uint32_t length, uint32_t offset, uint32_t *chunk_length, char *err, uint64_t err_len) {
    enum qx_tok_class first;
    uint32_t first_width = 0u;
    if (!qx_tok_char_at(text, length, offset, &first, &first_width)) { qx_tok_set_err(err, err_len, "input contains unsupported Unicode class"); return 0; }
    uint32_t contraction = 0u;
    if (qx_tok_ascii_contract(text, length, offset, &contraction)) { *chunk_length = contraction; return 1; }
    uint32_t cursor = offset;
    if (first == QX_TOK_LETTER) {
        do {
            cursor += first_width;
            if (cursor >= length || !qx_tok_char_at(text, length, cursor, &first, &first_width)) break;
        } while (first == QX_TOK_LETTER);
        *chunk_length = cursor - offset; return 1;
    }
    if (first != QX_TOK_NEWLINE && first != QX_TOK_LETTER && first != QX_TOK_NUMBER && offset + first_width < length) {
        enum qx_tok_class next;
        uint32_t next_width;
        if (!qx_tok_char_at(text, length, offset + first_width, &next, &next_width)) { qx_tok_set_err(err, err_len, "input contains unsupported Unicode class"); return 0; }
        if (next == QX_TOK_LETTER) {
            cursor = offset + first_width;
            do {
                cursor += next_width;
                if (cursor >= length || !qx_tok_char_at(text, length, cursor, &next, &next_width)) break;
            } while (next == QX_TOK_LETTER);
            *chunk_length = cursor - offset; return 1;
        }
    }
    if (first == QX_TOK_NUMBER) { *chunk_length = first_width; return 1; }
    cursor = offset;
    if (first == QX_TOK_SPACE && text[offset] == ' ' && cursor + first_width < length) {
        enum qx_tok_class next;
        uint32_t next_width;
        if (!qx_tok_char_at(text, length, cursor + first_width, &next, &next_width)) { qx_tok_set_err(err, err_len, "input contains unsupported Unicode class"); return 0; }
        if (next == QX_TOK_OTHER) { cursor += first_width; first = next; first_width = next_width; }
    }
    if (first == QX_TOK_OTHER) {
        do {
            cursor += first_width;
            if (cursor >= length || !qx_tok_char_at(text, length, cursor, &first, &first_width)) break;
        } while (first == QX_TOK_OTHER);
        while (cursor < length && qx_tok_char_at(text, length, cursor, &first, &first_width) && first == QX_TOK_NEWLINE) cursor += first_width;
        *chunk_length = cursor - offset; return 1;
    }
    if (first == QX_TOK_SPACE || first == QX_TOK_NEWLINE) {
        uint32_t run_end = offset;
        uint32_t last_newline_end = offset;
        uint32_t last_unit_start = offset;
        uint32_t units = 0u;
        enum qx_tok_class kind = first;
        uint32_t width = first_width;
        while (run_end < length && (kind == QX_TOK_SPACE || kind == QX_TOK_NEWLINE)) {
            last_unit_start = run_end;
            run_end += width;
            ++units;
            if (kind == QX_TOK_NEWLINE) last_newline_end = run_end;
            if (run_end >= length || !qx_tok_char_at(text, length, run_end, &kind, &width)) break;
        }
        if (last_newline_end > offset) { *chunk_length = last_newline_end - offset; return 1; }
        if (run_end == length) { *chunk_length = run_end - offset; return 1; }
        if (units > 1u) { *chunk_length = last_unit_start - offset; return 1; }
        *chunk_length = run_end - offset; return 1;
    }
    qx_tok_set_err(err, err_len, "pre-tokenizer made no progress");
    return 0;
}

static uint32_t qx_tok_byte_codepoint(unsigned char byte) {
    if ((byte >= 33u && byte <= 126u) || (byte >= 161u && byte <= 172u) || (byte >= 174u)) return byte;
    uint32_t rank = 0u;
    for (uint32_t value = 0u; value < byte; ++value) {
        if (!((value >= 33u && value <= 126u) || (value >= 161u && value <= 172u) || value >= 174u)) ++rank;
    }
    return 256u + rank;
}

static uint32_t qx_tok_write_utf8(uint32_t cp, unsigned char output[2]) {
    if (cp < 0x80u) { output[0] = (unsigned char)cp; return 1u; }
    output[0] = (unsigned char)(0xC0u | (cp >> 6));
    output[1] = (unsigned char)(0x80u | (cp & 0x3Fu));
    return 2u;
}

static int qx_tok_codepoint_byte(uint32_t cp, unsigned char *byte) {
    for (uint32_t value = 0u; value <= 255u; ++value) {
        if (qx_tok_byte_codepoint((unsigned char)value) == cp) { *byte = (unsigned char)value; return 1; }
    }
    return 0;
}

static void qx_tok_symbols_free(qx_tok_symbol *symbols, uint32_t count) {
    if (!symbols) return;
    for (uint32_t i = 0; i < count; ++i) free(symbols[i].text);
    free(symbols);
}

static int qx_tok_bpe_chunk(
    const qx_tokenizer *tokenizer,
    const qx_tok_index *token_index,
    const qx_tok_index *merge_index,
    const unsigned char *chunk,
    uint32_t chunk_length,
    uint32_t *token_ids,
    uint32_t token_capacity,
    uint32_t *token_count,
    char *err,
    uint64_t err_len) {
    qx_tok_symbol *symbols = (qx_tok_symbol *)calloc(chunk_length ? chunk_length : 1u, sizeof(qx_tok_symbol));
    if (!symbols) { qx_tok_set_err(err, err_len, "out of memory"); return 0; }
    uint32_t count = chunk_length;
    for (uint32_t i = 0; i < chunk_length; ++i) {
        unsigned char encoded[2];
        uint32_t encoded_length = qx_tok_write_utf8(qx_tok_byte_codepoint(chunk[i]), encoded);
        symbols[i].text = (unsigned char *)malloc(encoded_length);
        if (!symbols[i].text) { qx_tok_symbols_free(symbols, count); qx_tok_set_err(err, err_len, "out of memory"); return 0; }
        memcpy(symbols[i].text, encoded, encoded_length);
        symbols[i].length = encoded_length;
    }
    uint64_t work = 0u;
    uint64_t work_limit = ((uint64_t)chunk_length * (uint64_t)(chunk_length ? chunk_length - 1u : 0u)) / 2u;
    for (;;) {
        uint32_t best_rank = UINT32_MAX;
        uint32_t best_index = UINT32_MAX;
        for (uint32_t i = 0; i + 1u < count; ++i) {
            uint32_t rank;
            if (qx_tok_find_merge(tokenizer, merge_index, &symbols[i], &symbols[i + 1u], &rank) && rank < best_rank) {
                best_rank = rank;
                best_index = i;
            }
            if (++work > work_limit) { qx_tok_symbols_free(symbols, count); qx_tok_set_err(err, err_len, "BPE work limit exceeded"); return 0; }
        }
        if (best_index == UINT32_MAX) break;
        uint32_t combined_length = symbols[best_index].length + symbols[best_index + 1u].length;
        unsigned char *combined = (unsigned char *)malloc(combined_length);
        if (!combined) { qx_tok_symbols_free(symbols, count); qx_tok_set_err(err, err_len, "out of memory"); return 0; }
        memcpy(combined, symbols[best_index].text, symbols[best_index].length);
        memcpy(combined + symbols[best_index].length, symbols[best_index + 1u].text, symbols[best_index + 1u].length);
        free(symbols[best_index].text);
        free(symbols[best_index + 1u].text);
        symbols[best_index].text = combined;
        symbols[best_index].length = combined_length;
        memmove(&symbols[best_index + 1u], &symbols[best_index + 2u], (count - best_index - 2u) * sizeof(qx_tok_symbol));
        --count;
    }
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t id;
        if (!qx_tok_find_token(tokenizer, token_index, symbols[i].text, symbols[i].length, &id)) {
            qx_tok_symbols_free(symbols, count); qx_tok_set_err(err, err_len, "BPE symbol missing from vocabulary"); return 0;
        }
        if (*token_count >= token_capacity) { qx_tok_symbols_free(symbols, count); qx_tok_set_err(err, err_len, "token output capacity exceeded"); return 0; }
        token_ids[(*token_count)++] = id;
    }
    qx_tok_symbols_free(symbols, count);
    return 1;
}

static int qx_tok_match_special(const qx_tokenizer *tokenizer, const unsigned char *input, uint32_t remaining, uint32_t *token_id, uint32_t *matched) {
    *matched = 0u;
    for (uint32_t id = 0; id < tokenizer->vocab_count; ++id) {
        const qx_token_entry *token = &tokenizer->tokens[id];
        if ((token->type == 3u || token->type == 4u) && token->length <= remaining && token->length > *matched &&
            memcmp(token->text, input, token->length) == 0) {
            *token_id = id;
            *matched = token->length;
        }
    }
    return *matched != 0u;
}

static uint32_t qx_tok_next_special_offset(
    const qx_tokenizer *tokenizer,
    const unsigned char *input,
    uint32_t input_length,
    uint32_t offset) {
    uint32_t earliest = input_length;
    for (uint32_t id = 0; id < tokenizer->vocab_count; ++id) {
        const qx_token_entry *token = &tokenizer->tokens[id];
        if ((token->type != 3u && token->type != 4u) || token->length == 0u || token->length > input_length - offset) continue;
        for (uint32_t candidate = offset; candidate + token->length <= earliest; ++candidate) {
            if (memcmp(input + candidate, token->text, token->length) == 0) {
                earliest = candidate;
                break;
            }
        }
    }
    return earliest;
}

int qx_tokenizer_encode(
    const qx_tokenizer *tokenizer,
    const unsigned char *input,
    uint32_t input_length,
    int parse_special,
    uint32_t *token_ids,
    uint32_t token_capacity,
    uint32_t *token_count,
    char *err,
    uint64_t err_len) {
    if (!tokenizer || (!input && input_length) || !token_ids || !token_count) { qx_tok_set_err(err, err_len, "invalid encode argument"); return 0; }
    if (input_length > QX_TOKENIZER_MAX_INPUT) { qx_tok_set_err(err, err_len, "input exceeds 4096-byte limit"); return 0; }
    uint32_t scan = 0u;
    while (scan < input_length) {
        uint32_t cp, width;
        if (!qx_tok_utf8_next(input, input_length, scan, &cp, &width)) { qx_tok_set_err(err, err_len, "input is not valid UTF-8"); return 0; }
        scan += width;
    }
    qx_tok_index token_index = {0};
    qx_tok_index merge_index = {0};
    if (!qx_tok_build_token_index(tokenizer, &token_index, err, err_len) ||
        !qx_tok_build_merge_index(tokenizer, &merge_index, err, err_len)) {
        qx_tok_index_free(&token_index); qx_tok_index_free(&merge_index); return 0;
    }
    *token_count = 0u;
    if ((tokenizer->flags & 1u) != 0u) {
        if (tokenizer->bos_token_id < 0 || *token_count >= token_capacity) { qx_tok_set_err(err, err_len, "invalid BOS configuration"); goto fail; }
        token_ids[(*token_count)++] = (uint32_t)tokenizer->bos_token_id;
    }
    uint32_t offset = 0u;
    while (offset < input_length) {
        uint32_t special_id = 0u;
        uint32_t special_length = 0u;
        if (parse_special && qx_tok_match_special(tokenizer, input + offset, input_length - offset, &special_id, &special_length)) {
            if (*token_count >= token_capacity) { qx_tok_set_err(err, err_len, "token output capacity exceeded"); goto fail; }
            token_ids[(*token_count)++] = special_id;
            offset += special_length;
            continue;
        }
        uint32_t chunk_length = 0u;
        uint32_t segment_end = parse_special ? qx_tok_next_special_offset(tokenizer, input, input_length, offset + 1u) : input_length;
        if (!qx_tok_pretoken_length(input, segment_end, offset, &chunk_length, err, err_len) || chunk_length == 0u) goto fail;
        if (!qx_tok_bpe_chunk(tokenizer, &token_index, &merge_index, input + offset, chunk_length, token_ids, token_capacity, token_count, err, err_len)) goto fail;
        offset += chunk_length;
    }
    if ((tokenizer->flags & 2u) != 0u) {
        if (tokenizer->eos_token_id < 0 || *token_count >= token_capacity) { qx_tok_set_err(err, err_len, "invalid EOS configuration"); goto fail; }
        token_ids[(*token_count)++] = (uint32_t)tokenizer->eos_token_id;
    }
    qx_tok_index_free(&token_index);
    qx_tok_index_free(&merge_index);
    return 1;
fail:
    qx_tok_index_free(&token_index);
    qx_tok_index_free(&merge_index);
    return 0;
}

int qx_tokenizer_decode(
    const qx_tokenizer *tokenizer,
    const uint32_t *token_ids,
    uint32_t token_count,
    int render_special,
    unsigned char *output,
    uint32_t output_capacity,
    uint32_t *output_length,
    char *err,
    uint64_t err_len) {
    if (!tokenizer || (!token_ids && token_count) || !output || !output_length) { qx_tok_set_err(err, err_len, "invalid decode argument"); return 0; }
    *output_length = 0u;
    for (uint32_t i = 0; i < token_count; ++i) {
        uint32_t id = token_ids[i];
        if (id >= tokenizer->vocab_count) { qx_tok_set_err(err, err_len, "token id out of range"); return 0; }
        const qx_token_entry *token = &tokenizer->tokens[id];
        if (token->type == 3u || token->type == 4u) {
            if (!render_special) { qx_tok_set_err(err, err_len, "special token requires --special"); return 0; }
            if (token->length > output_capacity - *output_length) { qx_tok_set_err(err, err_len, "decoded output capacity exceeded"); return 0; }
            memcpy(output + *output_length, token->text, token->length);
            *output_length += token->length;
            continue;
        }
        uint32_t cursor = 0u;
        while (cursor < token->length) {
            uint32_t cp, width;
            unsigned char byte;
            if (!qx_tok_utf8_next(token->text, token->length, cursor, &cp, &width) || !qx_tok_codepoint_byte(cp, &byte)) {
                qx_tok_set_err(err, err_len, "token piece is not valid GPT-2 byte encoding"); return 0;
            }
            if (*output_length >= output_capacity) { qx_tok_set_err(err, err_len, "decoded output capacity exceeded"); return 0; }
            output[(*output_length)++] = byte;
            cursor += width;
        }
    }
    if (!qx_tok_valid_utf8(output, *output_length)) { qx_tok_set_err(err, err_len, "decoded output is not valid UTF-8"); return 0; }
    return 1;
}
