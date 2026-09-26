#include "qx_format.h"
#include "qx_tokenizer.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define DRIVER_MAX_PROMPT_BYTES 4096u
#define DRIVER_MAX_PROMPT_TOKENS 16u

static void usage(const char *program) {
    fprintf(stderr,
        "usage: %s --model QXF --tokenizer QXT --prompt-file FILE "
        "--activation f32|q8_k_compat --expert-cache-policy none|resident-packed "
        "--expert-cache-budget-bytes N --dump-dir DIR --ctx N --generate 2 "
        "--io-backend buffered\n", program);
}

static int parse_u64_strict(const char *text, uint64_t *value) {
    char *end = NULL;
    unsigned long long parsed;
    if (text == NULL || *text == '\0' || *text == '-' || *text == '+') return 0;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') return 0;
    *value = (uint64_t)parsed;
    return 1;
}

static int parse_u32_strict(const char *text, uint32_t *value) {
    uint64_t parsed = 0u;
    if (!parse_u64_strict(text, &parsed) || parsed > UINT32_MAX) return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static int read_prompt(const char *path, unsigned char **bytes_out, uint32_t *length_out,
        char *err, size_t err_len) {
    FILE *file = fopen(path, "rb");
    long length;
    unsigned char *bytes;
    if (file == NULL) { snprintf(err, err_len, "could not open prompt file"); return 0; }
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
            fseek(file, 0, SEEK_SET) != 0) {
        fclose(file); snprintf(err, err_len, "could not size prompt file"); return 0;
    }
    if ((unsigned long)length > DRIVER_MAX_PROMPT_BYTES) {
        fclose(file); snprintf(err, err_len, "prompt file exceeds 4096 bytes"); return 0;
    }
    bytes = (unsigned char *)malloc(length == 0 ? 1u : (size_t)length);
    if (bytes == NULL) { fclose(file); snprintf(err, err_len, "out of memory"); return 0; }
    if (length != 0 && fread(bytes, 1u, (size_t)length, file) != (size_t)length) {
        free(bytes); fclose(file); snprintf(err, err_len, "could not read prompt file"); return 0;
    }
    if (fclose(file) != 0) { free(bytes); snprintf(err, err_len, "could not close prompt file"); return 0; }
    *bytes_out = bytes;
    *length_out = (uint32_t)length;
    return 1;
}

int main(int argc, char **argv) {
    const char *model = NULL, *tokenizer_path = NULL, *prompt_path = NULL;
    const char *activation = NULL, *policy = NULL, *dump_dir = NULL, *io_backend = NULL;
    uint64_t budget = UINT64_MAX;
    uint32_t ctx = 0u, generation = 0u;
    unsigned char *prompt = NULL;
    uint32_t prompt_length = 0u, prompt_count = 0u;
    uint32_t prompt_tokens[DRIVER_MAX_PROMPT_TOKENS + 1u];
    qx_tokenizer tokenizer;
    char err[512] = {0};
    DWORD dump_attributes;
    int i, ok;

    for (i = 1; i < argc; ++i) {
        const char **target = NULL;
        if (strcmp(argv[i], "--model") == 0) target = &model;
        else if (strcmp(argv[i], "--tokenizer") == 0) target = &tokenizer_path;
        else if (strcmp(argv[i], "--prompt-file") == 0) target = &prompt_path;
        else if (strcmp(argv[i], "--activation") == 0) target = &activation;
        else if (strcmp(argv[i], "--expert-cache-policy") == 0) target = &policy;
        else if (strcmp(argv[i], "--dump-dir") == 0) target = &dump_dir;
        else if (strcmp(argv[i], "--io-backend") == 0) target = &io_backend;
        else if (strcmp(argv[i], "--expert-cache-budget-bytes") == 0) {
            if (++i >= argc || !parse_u64_strict(argv[i], &budget)) {
                fprintf(stderr, "expert-cache-acceptance-driver: invalid expert cache budget\n"); return 2;
            }
            continue;
        } else if (strcmp(argv[i], "--ctx") == 0) {
            if (++i >= argc || !parse_u32_strict(argv[i], &ctx)) {
                fprintf(stderr, "expert-cache-acceptance-driver: invalid ctx\n"); return 2;
            }
            continue;
        } else if (strcmp(argv[i], "--generate") == 0) {
            if (++i >= argc || !parse_u32_strict(argv[i], &generation)) {
                fprintf(stderr, "expert-cache-acceptance-driver: invalid generation count\n"); return 2;
            }
            continue;
        } else {
            usage(argv[0]); return 2;
        }
        if (++i >= argc || argv[i][0] == '\0' || *target != NULL) { usage(argv[0]); return 2; }
        *target = argv[i];
    }

    if (!model || !tokenizer_path || !prompt_path || !activation || !policy || !dump_dir ||
            !io_backend || budget == UINT64_MAX || ctx == 0u || generation == 0u) {
        usage(argv[0]); return 2;
    }
    if (strcmp(activation, "f32") != 0 && strcmp(activation, "q8_k_compat") != 0) {
        fprintf(stderr, "expert-cache-acceptance-driver: unsupported activation\n"); return 2;
    }
    if (strcmp(io_backend, "buffered") != 0) {
        fprintf(stderr, "expert-cache-acceptance-driver: only buffered I/O is supported\n"); return 2;
    }
    if (strcmp(policy, "none") == 0) {
        if (budget != 0u) { fprintf(stderr, "expert-cache-acceptance-driver: none policy requires zero budget\n"); return 2; }
    } else if (strcmp(policy, "resident-packed") == 0) {
        if (budget == 0u) { fprintf(stderr, "expert-cache-acceptance-driver: resident-packed policy requires positive budget\n"); return 2; }
    } else {
        fprintf(stderr, "expert-cache-acceptance-driver: unsupported expert cache policy\n"); return 2;
    }
    if (ctx > 16u) { fprintf(stderr, "expert-cache-acceptance-driver: ctx must be at most 16\n"); return 2; }
    if (generation != 2u) { fprintf(stderr, "expert-cache-acceptance-driver: generation count must be exactly 2\n"); return 2; }
    dump_attributes = GetFileAttributesA(dump_dir);
    if (dump_attributes == INVALID_FILE_ATTRIBUTES || !(dump_attributes & FILE_ATTRIBUTE_DIRECTORY)) {
        fprintf(stderr, "expert-cache-acceptance-driver: dump directory must already exist\n"); return 2;
    }

    if (!read_prompt(prompt_path, &prompt, &prompt_length, err, sizeof(err))) {
        fprintf(stderr, "expert-cache-acceptance-driver: %s\n", err); return 1;
    }
    memset(&tokenizer, 0, sizeof(tokenizer));
    if (!qx_tokenizer_load(tokenizer_path, &tokenizer, err, sizeof(err))) {
        free(prompt); fprintf(stderr, "expert-cache-acceptance-driver: %s\n", err); return 1;
    }
    if (tokenizer.vocab_count != 151936u || tokenizer.payload_checksum != 6140965799433681264ull) {
        qx_tokenizer_free(&tokenizer); free(prompt);
        fprintf(stderr, "expert-cache-acceptance-driver: tokenizer is not canonical Qwen3-30B-A3B\n"); return 1;
    }
    ok = qx_tokenizer_encode(&tokenizer, prompt, prompt_length, 0, prompt_tokens,
        DRIVER_MAX_PROMPT_TOKENS + 1u, &prompt_count, err, sizeof(err));
    qx_tokenizer_free(&tokenizer);
    free(prompt);
    if (!ok) { fprintf(stderr, "expert-cache-acceptance-driver: %s\n", err); return 1; }
    if (prompt_count == 0u) { fprintf(stderr, "expert-cache-acceptance-driver: prompt produced no tokens\n"); return 1; }
    if (prompt_count > DRIVER_MAX_PROMPT_TOKENS) {
        fprintf(stderr, "expert-cache-acceptance-driver: canonical prompt exceeds 16 tokens\n"); return 2;
    }
    if (prompt_count + generation > ctx) {
        fprintf(stderr, "expert-cache-acceptance-driver: prompt plus generation exceeds ctx\n"); return 2;
    }

    if (!qx_dump_prompt_state_loop_probe_summary(model, NULL, prompt_tokens, prompt_count,
            generation, 48u, ctx, "int8", activation, "ephemeral", "baseline", "serial", 1u,
            "scalar", policy, budget, "none", "none", "none", "none", "none", "none",
            0u, 0u, 0u, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2048u, NULL, 8u,
            151936u, 5u, 0.0, 7u, dump_dir, 0u, NULL, NULL, NULL, stdout, err, sizeof(err))) {
        fprintf(stderr, "expert-cache-acceptance-driver: %s\n", err); return 1;
    }
    return 0;
}
