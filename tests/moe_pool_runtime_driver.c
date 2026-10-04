#include "qx_format.h"
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *api_names[] = {"fixed-v1", "fixed-v2", "fixed-v3", "buffer-v2", "buffer-v3"};

static int invoke(unsigned api, const char *path, const qx_native_generation_options *options,
                  uint32_t *tokens, char *err, uint64_t err_len) {
    uint32_t prompt = 42u;
    qx_native_generation_result result = {0};
    qx_native_generation_profile p1 = {0};
    qx_native_generation_profile_v2 p2;
    qx_native_generation_profile_v3 p3;
    qx_native_generation_buffer_result buffer = {0};
    qx_native_generation_buffer_profile b2 = {0};
    qx_native_generation_buffer_profile_v3 b3;
    qx_native_generation_profile_v2_init(&p2);
    qx_native_generation_profile_v3_init(&p3);
    buffer.struct_size = (uint32_t)sizeof(buffer);
    buffer.version = QX_NATIVE_GENERATION_BUFFER_RESULT_VERSION;
    buffer.token_ids = tokens;
    buffer.token_capacity = 1u;
    b2.struct_size = (uint32_t)sizeof(b2);
    b2.version = QX_NATIVE_GENERATION_BUFFER_PROFILE_VERSION;
    qx_native_generation_buffer_profile_v3_init(&b3);
    uint64_t checksum = 0u;
    b2.full_logits_checksums = &checksum;
    b2.full_logits_checksums_capacity = 1u;
    b3.full_logits_checksums = &checksum;
    b3.full_logits_checksums_capacity = 1u;
    switch (api) {
        case 0u: return qx_run_native_generation_with_options(path, &prompt, 1u, 1u, 2u, -1, options, &result, &p1, err, err_len);
        case 1u: return qx_run_native_generation_with_options_v2(path, &prompt, 1u, 1u, 2u, -1, options, &result, &p2, err, err_len);
        case 2u: return qx_run_native_generation_with_options_v3(path, &prompt, 1u, 1u, 2u, -1, options, &result, &p3, err, err_len);
        case 3u: return qx_run_native_generation_into_with_options(path, &prompt, 1u, 1u, 2u, -1, options, &buffer, &b2, err, err_len);
        case 4u: return qx_run_native_generation_into_with_options_v3(path, &prompt, 1u, 1u, 2u, -1, options, &buffer, &b3, err, err_len);
        default: return 1;
    }
}

static int check_case(unsigned api, uint32_t version, unsigned test_case, const char *io_error) {
    const char *path = "missing-moe-model.qxf";
    qx_native_generation_options options;
    uint32_t token = UINT32_C(0xA17EC0DE);
    char err[512] = {0};
    const char *expected = io_error;
    qx_native_generation_options_init(&options);
    options.version = version;
    options.thread_policy = QX_NATIVE_THREAD_MOE_POOL;
    options.thread_count = 2u;
    switch (test_case) {
        case 0u: break;
        case 1u: options.thread_count = 64u; break;
        case 2u: options.thread_count = 0u; expected = "2..64"; break;
        case 3u: options.thread_count = 1u; expected = "2..64"; break;
        case 4u: options.thread_count = 65u; expected = "2..64"; break;
        case 5u: options.thread_count = UINT32_MAX; expected = "2..64"; break;
        case 6u: options.io_backend = QX_NATIVE_IO_MMAP; expected = "buffered I/O"; break;
        case 7u:
            options.cuda_policy = QX_NATIVE_CUDA_FINAL_HEAD_F32;
            expected = version == 1u ? "reserved" : "CUDA";
            break;
        case 8u: options.thread_policy = (qx_native_thread_policy)99; expected = "thread policy"; break;
        case 9u:
            options.expert_cache_policy = QX_NATIVE_EXPERT_CACHE_RESIDENT_PACKED;
            options.expert_cache_budget_bytes = 4096u;
            expected = version < 3u ? "reserved" :
                ((api == 2u || api == 4u) ? io_error : "cannot represent expert cache provenance");
            break;
        default: return 1;
    }
    int ok = invoke(api, path, &options, &token, err, sizeof(err));
    int accepted = expected == io_error;
    /* Legacy buffer-v2 clears output before checking the thread enum. Other
     * malformed options above fail in the shared validator before clearing. */
    int cleared = (accepted && api >= 3u) || (api == 3u && test_case == 8u);
    if (ok || !err[0] || (accepted ? strcmp(err, io_error) != 0 : strstr(err, expected) == NULL)
            || (!accepted && strcmp(err, io_error) == 0)
            || token != (cleared ? 0u : UINT32_C(0xA17EC0DE))) {
        fprintf(stderr, "%s options-v%u case=%u ok=%d expected=%s error=%s token=%u\n",
                api_names[api], version, test_case, ok, expected, err, token);
        return 1;
    }
    printf("%s options-v%u case=%u: %s (%s)\n", api_names[api], version, test_case,
           accepted ? "policy accepted; missing input, no model execution" : "policy rejected before input I/O", err);
    return 0;
}

int main(int argc, char **argv) {
    unsigned api = argc == 2 ? (unsigned)strtoul(argv[1], NULL, 10) : 0u;
    if (api >= (unsigned)(sizeof(api_names) / sizeof(api_names[0]))) return 2;
    /* Compare the host's fopen diagnostic, not an English substring such as "open". */
    FILE *missing = fopen("missing-moe-model.qxf", "rb");
    if (missing) { fclose(missing); return 2; }
    if (errno != ENOENT) return 2;
    char io_error[512];
    snprintf(io_error, sizeof(io_error), "%s", strerror(errno));
    for (uint32_t version = 1u; version <= QX_NATIVE_GENERATION_OPTIONS_VERSION; ++version) {
        for (unsigned test_case = 0u; test_case < 10u; ++test_case) {
            if (check_case(api, version, test_case, io_error)) return 1;
        }
    }
    printf("%s: MoE pool API preflight contract pass; no model executed\n", api_names[api]);
    return 0;
}
