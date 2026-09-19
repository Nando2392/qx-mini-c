#include "qx_format.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int parse_u32(const char *text, uint32_t *value) {
    char *end = NULL;
    unsigned long parsed;

    errno = 0;
    parsed = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed > UINT32_MAX) return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static int parse_i32(const char *text, int32_t *value) {
    char *end = NULL;
    long parsed;

    errno = 0;
    parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < INT32_MIN || parsed > INT32_MAX) return 0;
    *value = (int32_t)parsed;
    return 1;
}

int main(int argc, char **argv) {
    static const uint32_t prompt_tokens[] = {9707u, 0u};
    qx_native_generation_result result;
    char err[512] = {0};
    int32_t eos_token_id;
    uint32_t max_tokens;
    uint32_t ctx_tokens;

    if (argc != 5) {
        fprintf(stderr, "usage: %s MODEL EOS_TOKEN_ID MAX_TOKENS CTX_TOKENS\n", argv[0]);
        return 2;
    }
    if (!parse_i32(argv[2], &eos_token_id) ||
        !parse_u32(argv[3], &max_tokens) ||
        !parse_u32(argv[4], &ctx_tokens)) {
        fputs("invalid numeric argument\n", stderr);
        return 2;
    }

    if (!qx_run_native_generation(
            argv[1],
            prompt_tokens,
            (uint32_t)(sizeof(prompt_tokens) / sizeof(prompt_tokens[0])),
            max_tokens,
            ctx_tokens,
            eos_token_id,
            &result,
            err,
            sizeof(err))) {
        fprintf(stderr, "native generation failed: %s\n", err);
        return 1;
    }

    fputs("{\"token_ids\":[", stdout);
    for (uint32_t i = 0; i < result.token_count; ++i) {
        printf("%s%u", i == 0u ? "" : ",", result.token_ids[i]);
    }
    printf(
        "],\"token_count\":%u,\"stopped_on_eos\":%s,"
        "\"timing\":{\"prefill_seconds\":%.17g,\"decode_seconds\":%.17g}}\n",
        result.token_count,
        result.stopped_on_eos ? "true" : "false",
        result.prefill_seconds,
        result.decode_seconds);
    return 0;
}
