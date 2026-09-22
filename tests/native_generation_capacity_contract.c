#include "qx_format.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOKEN_CANARY 0xA17EC0DEu
#define CHECKSUM_CANARY UINT64_C(0xA17EC0DEA17EC0DE)

typedef struct guarded_tokens {
    uint32_t before;
    uint32_t values[2];
    uint32_t after;
} guarded_tokens;

typedef struct guarded_checksums {
    uint64_t before;
    uint64_t values[2];
    uint64_t after;
} guarded_checksums;

typedef struct guarded_tokens_128 {
    uint32_t before;
    uint32_t values[128];
    uint32_t after;
} guarded_tokens_128;

typedef struct guarded_checksums_128 {
    uint64_t before;
    uint64_t values[128];
    uint64_t after;
} guarded_checksums_128;

static void init_contract(
        guarded_tokens *tokens,
        guarded_checksums *checksums,
        qx_native_generation_buffer_result *result,
        qx_native_generation_buffer_profile *profile,
        qx_native_generation_options *options) {
    memset(tokens, 0x5a, sizeof(*tokens));
    tokens->before = TOKEN_CANARY;
    tokens->after = TOKEN_CANARY;
    memset(checksums, 0xa5, sizeof(*checksums));
    checksums->before = CHECKSUM_CANARY;
    checksums->after = CHECKSUM_CANARY;

    memset(result, 0, sizeof(*result));
    result->struct_size = (uint32_t)sizeof(*result);
    result->version = QX_NATIVE_GENERATION_BUFFER_RESULT_VERSION;
    result->token_ids = tokens->values;
    result->token_capacity = 2u;

    memset(profile, 0, sizeof(*profile));
    profile->struct_size = (uint32_t)sizeof(*profile);
    profile->version = QX_NATIVE_GENERATION_BUFFER_PROFILE_VERSION;
    profile->full_logits_checksums = checksums->values;
    profile->full_logits_checksums_capacity = 2u;
    qx_native_generation_options_init(options);
}

static int canaries_intact(const guarded_tokens *tokens, const guarded_checksums *checksums) {
    return tokens->before == TOKEN_CANARY && tokens->after == TOKEN_CANARY &&
        checksums->before == CHECKSUM_CANARY && checksums->after == CHECKSUM_CANARY;
}

static int expect_failure(unsigned test_case, const char *needle) {
    static const uint32_t prompt[] = {9707u, 0u};
    guarded_tokens tokens;
    guarded_checksums checksums;
    qx_native_generation_buffer_result result;
    qx_native_generation_buffer_profile profile;
    qx_native_generation_options options;
    const char *path = "definitely-missing-model.qxf";
    const uint32_t *prompt_ptr = prompt;
    uint32_t prompt_count = 2u;
    uint32_t max_tokens = 2u;
    uint32_t ctx_tokens = 3u;
    const qx_native_generation_options *options_ptr = &options;
    qx_native_generation_buffer_result *result_ptr = &result;
    qx_native_generation_buffer_profile *profile_ptr = &profile;
    char err[256] = {0};

    init_contract(&tokens, &checksums, &result, &profile, &options);
    switch (test_case) {
        case 0u: options_ptr = NULL; break;
        case 1u: result_ptr = NULL; break;
        case 2u: options.version += 1u; break;
        case 3u: result.version += 1u; break;
        case 4u: profile.version += 1u; break;
        case 5u: result.token_ids = NULL; break;
        case 6u: profile.full_logits_checksums = NULL; break;
        case 7u: result.token_capacity = 1u; break;
        case 8u: profile.full_logits_checksums_capacity = 1u; break;
        case 9u: max_tokens = 4097u; break;
        case 10u: ctx_tokens = 4097u; break;
        case 11u: prompt_count = UINT32_MAX; break;
        case 12u: path = NULL; break;
        case 13u: prompt_ptr = NULL; break;
        case 14u: profile_ptr = NULL; result.token_capacity = 1u; break;
        default: return 0;
    }

    int ok = qx_run_native_generation_into_with_options(
        path, prompt_ptr, prompt_count, max_tokens, ctx_tokens, -1,
        options_ptr, result_ptr, profile_ptr, err, sizeof(err));
    return !ok && strstr(err, needle) != NULL && canaries_intact(&tokens, &checksums);
}

static int expect_128_token_admission_before_model_io(void) {
    static const uint32_t prompt[] = {9707u};
    guarded_tokens_128 tokens;
    guarded_checksums_128 checksums;
    qx_native_generation_buffer_result result;
    qx_native_generation_buffer_profile profile;
    qx_native_generation_options options;
    char err[256] = {0};

    memset(&tokens, 0x5a, sizeof(tokens));
    tokens.before = TOKEN_CANARY;
    tokens.after = TOKEN_CANARY;
    memset(&checksums, 0xa5, sizeof(checksums));
    checksums.before = CHECKSUM_CANARY;
    checksums.after = CHECKSUM_CANARY;
    memset(&result, 0, sizeof(result));
    result.struct_size = (uint32_t)sizeof(result);
    result.version = QX_NATIVE_GENERATION_BUFFER_RESULT_VERSION;
    result.token_ids = tokens.values;
    result.token_capacity = 128u;
    memset(&profile, 0, sizeof(profile));
    profile.struct_size = (uint32_t)sizeof(profile);
    profile.version = QX_NATIVE_GENERATION_BUFFER_PROFILE_VERSION;
    profile.full_logits_checksums = checksums.values;
    profile.full_logits_checksums_capacity = 128u;
    qx_native_generation_options_init(&options);

    if (qx_run_native_generation_into_with_options(
            "definitely-missing-model.qxf", prompt, 1u, 128u, 128u, -1,
            &options, &result, &profile, err, sizeof(err))) return 0;
    return (strstr(err, "open") != NULL || strstr(err, "No such") != NULL ||
            strstr(err, "cannot find") != NULL) &&
        result.token_count == 0u && result.executed_forward_steps == 0u &&
        result.first_eos_output_index == UINT32_MAX && profile.sampled_steps == 0u &&
        tokens.before == TOKEN_CANARY && tokens.after == TOKEN_CANARY &&
        checksums.before == CHECKSUM_CANARY && checksums.after == CHECKSUM_CANARY;
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

static int run_real(const char *model, int32_t eos_token_id) {
    static const uint32_t prompt[] = {9707u, 0u};
    guarded_tokens tokens;
    guarded_checksums checksums;
    qx_native_generation_buffer_result result;
    qx_native_generation_buffer_profile profile;
    qx_native_generation_options options;
    char err[512] = {0};

    init_contract(&tokens, &checksums, &result, &profile, &options);
    result.token_capacity = 2u;
    profile.full_logits_checksums_capacity = 2u;
    if (!qx_run_native_generation_into_with_options(
            model, prompt, 2u, 2u, 3u, eos_token_id, &options,
            &result, &profile, err, sizeof(err))) {
        fprintf(stderr, "native generation capacity failed: %s\n", err);
        return 1;
    }
    if (!canaries_intact(&tokens, &checksums)) {
        fputs("native generation capacity overwrote a buffer canary\n", stderr);
        return 1;
    }

    fputs("{\"token_ids\":[", stdout);
    for (uint32_t i = 0; i < result.token_count; ++i) {
        printf("%s%u", i == 0u ? "" : ",", result.token_ids[i]);
    }
    printf(
        "],\"token_count\":%u,\"stopped_on_eos\":%s,"
        "\"first_eos_output_index\":",
        result.token_count, result.stopped_on_eos ? "true" : "false");
    if (result.first_eos_output_index == UINT32_MAX) fputs("null", stdout);
    else printf("%u", result.first_eos_output_index);
    printf(
        ",\"executed_forward_steps\":%u,\"prompt_forward_steps\":%u,"
        "\"generated_input_forward_steps\":%u,\"sampled_steps\":%u}\n",
        result.executed_forward_steps, result.prompt_forward_steps,
        result.generated_input_forward_steps, profile.sampled_steps);
    return 0;
}

int main(int argc, char **argv) {
    static const char *needles[] = {
        "invalid native generation argument",
        "invalid native generation argument",
        "options version",
        "buffer result version",
        "buffer profile version",
        "output capacity",
        "checksum capacity",
        "output capacity",
        "checksum capacity",
        "1..4096",
        "context limit",
        "context limit",
        "invalid native generation argument",
        "invalid native generation argument",
        "output capacity",
    };

    _Static_assert(QX_NATIVE_GENERATION_CAPACITY_MAX == 4096u, "capacity contract must remain 4096");
    if (argc == 1) {
        for (unsigned i = 0u; i < (unsigned)(sizeof(needles) / sizeof(needles[0])); ++i) {
            if (!expect_failure(i, needles[i])) {
                fprintf(stderr, "native generation capacity preflight case %u failed\n", i);
                return (int)i + 2;
            }
        }
        if (!expect_128_token_admission_before_model_io()) {
            fputs("native generation capacity 128-token admission failed\n", stderr);
            return 17;
        }
        puts("native generation capacity contract: pass");
        return 0;
    }
    if (argc == 3) {
        int32_t eos_token_id;
        if (!parse_i32(argv[2], &eos_token_id)) {
            fputs("invalid EOS token id\n", stderr);
            return 2;
        }
        return run_real(argv[1], eos_token_id);
    }
    fprintf(stderr, "usage: %s [MODEL EOS_TOKEN_ID]\n", argv[0]);
    return 2;
}
