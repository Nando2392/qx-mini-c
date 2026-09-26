#include "qx_format.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

_Static_assert(sizeof(qx_native_generation_options) == 64u, "options ABI remains 64 bytes");
_Static_assert(offsetof(qx_native_generation_options, cuda_policy) == 28u, "legacy CUDA slot remains at 28");
_Static_assert(offsetof(qx_native_generation_options, expert_cache_policy) == 32u, "v3 cache policy offset");
_Static_assert(offsetof(qx_native_generation_options, expert_cache_budget_bytes) == 40u, "v3 cache budget offset");
_Static_assert(offsetof(qx_native_generation_options, reserved_v3) == 48u, "v3 reserved suffix offset");
_Static_assert(sizeof(qx_native_generation_profile) == 656u, "fixed v1 size remains frozen");
_Static_assert(sizeof(qx_native_generation_profile_v2) == 864u, "fixed v2 size remains frozen");
_Static_assert(offsetof(qx_native_generation_profile_v2, requested_cuda_policy) == 656u, "fixed v2 extension offset");
_Static_assert(sizeof(qx_native_generation_profile_v3) == 952u, "fixed v3 ABI size");
_Static_assert(offsetof(qx_native_generation_profile_v3, requested_expert_cache_policy) == 864u, "fixed v3 extension offset");
_Static_assert(QX_NATIVE_GENERATION_BUFFER_PROFILE_V1_SIZE == 160u, "buffer v1 size remains frozen");
_Static_assert(sizeof(qx_native_generation_buffer_profile) == 368u, "buffer v2 size remains frozen");
_Static_assert(offsetof(qx_native_generation_buffer_profile, requested_cuda_policy) == 160u, "buffer v2 extension offset");
_Static_assert(sizeof(qx_native_generation_buffer_profile_v3) == 456u, "buffer v3 ABI size");
_Static_assert(offsetof(qx_native_generation_buffer_profile_v3, requested_expert_cache_policy) == 368u, "buffer v3 extension offset");

#ifdef QX_EXPERT_CACHE_V3_HEADER_LAYOUT_ONLY
int main(void) { return 0; }
#else

#define BYTE_CANARY 0xA5u
#define TOKEN_CANARY UINT32_C(0xA17EC0DE)
#define CHECKSUM_CANARY UINT64_C(0xA17EC0DEA17EC0DE)
#define CHECK(condition, code) do { if (!(condition)) return (code); } while (0)

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

typedef struct call_state {
    uint32_t prompt[2];
    qx_native_generation_options options;
    qx_native_generation_buffer_result result;
    qx_native_generation_buffer_profile_v3 profile;
    guarded_tokens tokens;
    guarded_checksums checksums;
} call_state;

static int bytes_are_zero(const void *memory, size_t size) {
    const unsigned char *bytes = (const unsigned char *)memory;
    for (size_t i = 0u; i < size; ++i) if (bytes[i] != 0u) return 0;
    return 1;
}

static void init_state(call_state *state) {
    memset(state, BYTE_CANARY, sizeof(*state));
    state->prompt[0] = 9707u;
    state->prompt[1] = 0u;
    qx_native_generation_options_init(&state->options);
    memset(&state->result, 0, sizeof(state->result));
    state->result.struct_size = (uint32_t)sizeof(state->result);
    state->result.version = QX_NATIVE_GENERATION_BUFFER_RESULT_VERSION;
    state->result.token_ids = state->tokens.values;
    state->result.token_capacity = 2u;
    qx_native_generation_buffer_profile_v3_init(&state->profile);
    state->profile.full_logits_checksums = state->checksums.values;
    state->profile.full_logits_checksums_capacity = 2u;
    state->tokens.before = TOKEN_CANARY;
    state->tokens.values[0] = TOKEN_CANARY;
    state->tokens.values[1] = TOKEN_CANARY;
    state->tokens.after = TOKEN_CANARY;
    state->checksums.before = CHECKSUM_CANARY;
    state->checksums.values[0] = CHECKSUM_CANARY;
    state->checksums.values[1] = CHECKSUM_CANARY;
    state->checksums.after = CHECKSUM_CANARY;
}

static int test_initializers_and_source_compatibility(void) {
    qx_native_generation_options options;
    qx_native_generation_profile_v3 fixed;
    qx_native_generation_buffer_profile_v3 buffered;
    memset(&options, BYTE_CANARY, sizeof(options));
    memset(&fixed, BYTE_CANARY, sizeof(fixed));
    memset(&buffered, BYTE_CANARY, sizeof(buffered));
    qx_native_generation_options_init(&options);
    qx_native_generation_profile_v3_init(&fixed);
    qx_native_generation_buffer_profile_v3_init(&buffered);
    CHECK(options.struct_size == 64u && options.version == 3u, 10);
    CHECK(options.io_backend == QX_NATIVE_IO_BUFFERED && options.cuda_policy == QX_NATIVE_CUDA_NONE, 11);
    CHECK(options.expert_cache_policy == QX_NATIVE_EXPERT_CACHE_NONE && options.expert_cache_budget_bytes == 0u, 12);
    CHECK(options.reserved_alignment == 0u && options.reserved_v3[0] == 0u && options.reserved_v3[1] == 0u, 13);
    /* The legacy source spelling must remain usable for v1/v2 callers. */
    for (size_t i = 0u; i < sizeof(options.reserved) / sizeof(options.reserved[0]); ++i) CHECK(options.reserved[i] == 0u, 14);
    CHECK(fixed.struct_size == 952u && fixed.version == 3u, 15);
    CHECK(fixed.requested_expert_cache_policy == QX_NATIVE_EXPERT_CACHE_NONE && fixed.expert_cache_budget_bytes == 0u, 16);
    CHECK(buffered.struct_size == 456u && buffered.version == 3u, 17);
    CHECK(buffered.requested_expert_cache_policy == QX_NATIVE_EXPERT_CACHE_NONE && buffered.full_logits_checksums == NULL, 18);
    CHECK(bytes_are_zero((const unsigned char *)&fixed + 8u, sizeof(fixed) - 8u), 19);
    CHECK(bytes_are_zero((const unsigned char *)&buffered + 8u, sizeof(buffered) - 8u), 20);
    return 0;
}

static int expect_nonwriting_rejection(unsigned test_case, const char *needle) {
    call_state state, before;
    char err[256] = {0};
    init_state(&state);
    switch (test_case) {
        case 0u: state.options.struct_size -= 1u; break;
        case 1u: state.options.version += 1u; break;
        case 2u: state.options.reserved_alignment = 1u; break;
        case 3u: state.options.reserved_v3[1] = 1u; break;
        case 4u: state.options.expert_cache_policy = (qx_native_expert_cache_policy)99; break;
        case 5u: state.options.expert_cache_policy = QX_NATIVE_EXPERT_CACHE_RESIDENT_PACKED; break;
        case 6u: state.options.expert_cache_budget_bytes = 4096u; break;
        case 7u:
            state.options.expert_cache_policy = QX_NATIVE_EXPERT_CACHE_RESIDENT_PACKED;
            state.options.expert_cache_budget_bytes = 4096u;
            state.options.io_backend = QX_NATIVE_IO_MMAP;
            break;
        case 8u: state.profile.struct_size -= 1u; break;
        case 9u: state.profile.version += 1u; break;
        default: return 0;
    }
    before = state;
    CHECK(!qx_run_native_generation_into_with_options_v3(
        "v3-abi-definitely-missing.qxf", state.prompt, 2u, 2u, 3u, -1,
        &state.options, &state.result, &state.profile, err, sizeof(err)), 30 + (int)test_case);
    CHECK(strstr(err, needle) != NULL && strstr(err, "open") == NULL, 50 + (int)test_case);
    CHECK(memcmp(&state, &before, sizeof(state)) == 0, 70 + (int)test_case);
    return 0;
}

static int test_malformed_calls_are_nonwriting(void) {
    static const char *needles[] = {
        "options size", "options version", "reserved", "reserved", "expert cache policy",
        "positive budget", "budget requires", "requires buffered I/O", "profile v3 size", "profile v3 version"
    };
    for (unsigned i = 0u; i < (unsigned)(sizeof(needles) / sizeof(needles[0])); ++i) {
        int rc = expect_nonwriting_rejection(i, needles[i]);
        if (rc != 0) return rc;
    }
    return 0;
}

static int test_legacy_reserved_semantics_and_v2_cuda_control(void) {
    call_state state, before;
    char err[256] = {0};

    init_state(&state);
    state.options.version = 1u;
    state.options.cuda_policy = QX_NATIVE_CUDA_FINAL_HEAD_F32;
    before = state;
    CHECK(!qx_run_native_generation_into_with_options_v3(
        "v3-abi-definitely-missing.qxf", state.prompt, 2u, 2u, 3u, -1,
        &state.options, &state.result, &state.profile, err, sizeof(err)), 100);
    CHECK(strstr(err, "reserved") != NULL && memcmp(&state, &before, sizeof(state)) == 0, 101);

    init_state(&state);
    state.options.version = 2u;
    state.options.reserved[7] = 1u;
    before = state;
    memset(err, 0, sizeof(err));
    CHECK(!qx_run_native_generation_into_with_options_v3(
        "v3-abi-definitely-missing.qxf", state.prompt, 2u, 2u, 3u, -1,
        &state.options, &state.result, &state.profile, err, sizeof(err)), 102);
    CHECK(strstr(err, "reserved") != NULL && memcmp(&state, &before, sizeof(state)) == 0, 103);

    init_state(&state);
    state.options.version = 2u;
    state.options.cuda_policy = QX_NATIVE_CUDA_FINAL_HEAD_F32;
    memset(err, 0, sizeof(err));
    CHECK(!qx_run_native_generation_into_with_options_v3(
        "v3-abi-definitely-missing.qxf", state.prompt, 2u, 2u, 3u, -1,
        &state.options, &state.result, &state.profile, err, sizeof(err)), 104);
    CHECK(strstr(err, "CUDA final-head backend unavailable") != NULL, 105);
    CHECK(state.profile.requested_cuda_policy == QX_NATIVE_CUDA_FINAL_HEAD_F32, 106);
    CHECK(state.profile.requested_expert_cache_policy == QX_NATIVE_EXPERT_CACHE_NONE &&
          state.profile.effective_expert_cache_policy == QX_NATIVE_EXPERT_CACHE_NONE &&
          state.profile.expert_cache_budget_bytes == 0u, 107);
    return 0;
}

static int test_valid_resident_configuration_reaches_io(void) {
    call_state state;
    char err[256] = {0};
    init_state(&state);
    state.options.expert_cache_policy = QX_NATIVE_EXPERT_CACHE_RESIDENT_PACKED;
    state.options.expert_cache_budget_bytes = 4096u;
    CHECK(!qx_run_native_generation_into_with_options_v3(
        "v3-abi-definitely-missing.qxf", state.prompt, 2u, 2u, 3u, -1,
        &state.options, &state.result, &state.profile, err, sizeof(err)), 120);
    CHECK(strstr(err, "open") != NULL || strstr(err, "No such") != NULL || strstr(err, "cannot find") != NULL, 121);
    CHECK(state.tokens.before == TOKEN_CANARY && state.tokens.after == TOKEN_CANARY, 122);
    CHECK(state.checksums.before == CHECKSUM_CANARY && state.checksums.after == CHECKSUM_CANARY, 123);
    CHECK(state.tokens.values[0] == 0u && state.tokens.values[1] == 0u, 124);
    CHECK(state.checksums.values[0] == 0u && state.checksums.values[1] == 0u, 125);
    CHECK(state.result.token_ids == state.tokens.values && state.result.token_capacity == 2u, 126);
    CHECK(state.profile.full_logits_checksums == state.checksums.values && state.profile.full_logits_checksums_capacity == 2u, 127);
    CHECK(state.profile.requested_expert_cache_policy == QX_NATIVE_EXPERT_CACHE_RESIDENT_PACKED &&
          state.profile.effective_expert_cache_policy == QX_NATIVE_EXPERT_CACHE_RESIDENT_PACKED &&
          state.profile.expert_cache_budget_bytes == 4096u, 128);
    return 0;
}

int main(void) {
    int rc = test_initializers_and_source_compatibility();
    if (rc == 0) rc = test_malformed_calls_are_nonwriting();
    if (rc == 0) rc = test_legacy_reserved_semantics_and_v2_cuda_control();
    if (rc == 0) rc = test_valid_resident_configuration_reaches_io();
    if (rc != 0) {
        fprintf(stderr, "native expert cache v3 ABI contract failed: %d\n", rc);
        return rc;
    }
    puts("native expert cache v3 ABI contract: pass");
    return 0;
}
#endif
