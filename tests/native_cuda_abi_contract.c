#ifdef QX_ABI_HEADER_LAYOUT_ONLY
#include "qx_format.h"
#include <stddef.h>

_Static_assert(sizeof(qx_native_generation_options) == 64u, "options ABI size");
_Static_assert(offsetof(qx_native_generation_options, cuda_policy) == 28u, "options CUDA/legacy slot");
_Static_assert(sizeof(qx_native_generation_profile) == 656u, "fixed profile v1 ABI size");
_Static_assert(offsetof(qx_native_generation_profile, sampled_steps) == 48u, "fixed profile sampled_steps");
_Static_assert(offsetof(qx_native_generation_profile, scratch_peak_capacity_bytes) == 56u, "fixed profile scratch metrics");
_Static_assert(offsetof(qx_native_generation_profile, full_logits_checksums) == 144u, "fixed profile checksums");
_Static_assert(sizeof(qx_native_generation_buffer_profile) == 368u, "buffer profile v2 ABI size");
_Static_assert(offsetof(qx_native_generation_buffer_profile, full_logits_checksums) == 144u, "buffer profile v1 checksum pointer");
_Static_assert(offsetof(qx_native_generation_buffer_profile, full_logits_checksums_capacity) == 152u, "buffer profile v1 checksum capacity");
_Static_assert(offsetof(qx_native_generation_buffer_profile, requested_cuda_policy) == 160u, "buffer profile v2 CUDA policy");
_Static_assert(offsetof(qx_native_generation_buffer_profile, cuda_device_name) == 168u, "buffer profile v2 device name");
_Static_assert(offsetof(qx_native_generation_buffer_profile, cuda_resident_weight_bytes) == 304u, "buffer profile v2 counters");

int main(void) { return 0; }
#else
/*
 * Old-header ABI driver.  These declarations are frozen copies of the public
 * v1 ABI; deliberately do not include qx_format.h or derive sizes from it.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define V1_OPTIONS_SIZE 64u
#define V1_FIXED_PROFILE_SIZE 656u
#define V1_BUFFER_PROFILE_SIZE 160u
#define V2_BUFFER_PROFILE_SIZE 368u
#define MAX_TOKENS 64u
#define BYTE_CANARY 0xA5u
#define WORD_CANARY UINT64_C(0xA17EC0DEA17EC0DE)

typedef enum old_io_policy { OLD_IO_BUFFERED = 0, OLD_IO_MMAP = 1 } old_io_policy;
typedef enum old_scratch_policy { OLD_SCRATCH_EPHEMERAL = 0, OLD_SCRATCH_PERSISTENT = 1 } old_scratch_policy;
typedef enum old_kernel_policy { OLD_KERNEL_BASELINE = 0, OLD_KERNEL_FUSED_FINAL_HEAD = 1 } old_kernel_policy;
typedef enum old_thread_policy { OLD_THREAD_SERIAL = 0, OLD_THREAD_POOL = 1 } old_thread_policy;

typedef struct old_options_v1 {
    uint32_t struct_size;
    uint32_t version;
    old_io_policy io_backend;
    old_scratch_policy scratch_policy;
    old_kernel_policy kernel_policy;
    old_thread_policy thread_policy;
    uint32_t thread_count;
    uint32_t reserved[9];
} old_options_v1;

typedef struct old_fixed_profile_v1 {
    uint32_t struct_size;
    uint32_t version;
    old_io_policy requested_io_backend;
    old_io_policy effective_io_backend;
    old_scratch_policy requested_scratch_policy;
    old_scratch_policy effective_scratch_policy;
    old_kernel_policy requested_kernel_policy;
    old_kernel_policy effective_kernel_policy;
    old_thread_policy requested_thread_policy;
    old_thread_policy effective_thread_policy;
    uint32_t requested_thread_count;
    uint32_t effective_thread_count;
    uint32_t sampled_steps;
    uint32_t workers_used;
    uint64_t scratch_peak_capacity_bytes;
    uint64_t scratch_growth_events;
    uint64_t temporary_blocks_decoded;
    uint64_t temporary_floats_materialized;
    uint64_t temporary_bytes_materialized;
    uint64_t fused_final_head_dot_calls;
    uint64_t baseline_final_head_dot_calls;
    uint64_t final_head_q6_k_blocks;
    uint64_t final_head_parallel_jobs;
    uint64_t final_head_serial_jobs;
    uint64_t final_head_fallback_jobs;
    uint64_t full_logits_checksums[MAX_TOKENS];
} old_fixed_profile_v1;

typedef struct old_result {
    uint32_t token_ids[MAX_TOKENS];
    uint32_t token_count;
    int stopped_on_eos;
    double prefill_seconds;
    double decode_seconds;
} old_result;

typedef struct old_buffer_result {
    uint32_t struct_size;
    uint32_t version;
    uint32_t *token_ids;
    uint32_t token_capacity;
    uint32_t token_count;
    int stopped_on_eos;
    uint32_t first_eos_output_index;
    uint32_t executed_forward_steps;
    uint32_t prompt_forward_steps;
    uint32_t generated_input_forward_steps;
    double prefill_seconds;
    double decode_seconds;
} old_buffer_result;

typedef struct old_buffer_profile_v1 {
    uint32_t struct_size;
    uint32_t version;
    old_io_policy requested_io_backend;
    old_io_policy effective_io_backend;
    old_scratch_policy requested_scratch_policy;
    old_scratch_policy effective_scratch_policy;
    old_kernel_policy requested_kernel_policy;
    old_kernel_policy effective_kernel_policy;
    old_thread_policy requested_thread_policy;
    old_thread_policy effective_thread_policy;
    uint32_t requested_thread_count;
    uint32_t effective_thread_count;
    uint32_t sampled_steps;
    uint32_t workers_used;
    uint64_t scratch_peak_capacity_bytes;
    uint64_t scratch_growth_events;
    uint64_t temporary_blocks_decoded;
    uint64_t temporary_floats_materialized;
    uint64_t temporary_bytes_materialized;
    uint64_t fused_final_head_dot_calls;
    uint64_t baseline_final_head_dot_calls;
    uint64_t final_head_q6_k_blocks;
    uint64_t final_head_parallel_jobs;
    uint64_t final_head_serial_jobs;
    uint64_t final_head_fallback_jobs;
    uint64_t *full_logits_checksums;
    uint32_t full_logits_checksums_capacity;
} old_buffer_profile_v1;

int qx_run_native_generation_with_options(
    const char *, const uint32_t *, uint32_t, uint32_t, uint32_t, int32_t,
    const old_options_v1 *, old_result *, old_fixed_profile_v1 *, char *, uint64_t);
int qx_run_native_generation_into_with_options(
    const char *, const uint32_t *, uint32_t, uint32_t, uint32_t, int32_t,
    const old_options_v1 *, old_buffer_result *, old_buffer_profile_v1 *, char *, uint64_t);

_Static_assert(sizeof(old_options_v1) == V1_OPTIONS_SIZE, "frozen v1 options size");
_Static_assert(offsetof(old_options_v1, reserved) == 28u, "frozen v1 reserved slot");
_Static_assert(sizeof(old_fixed_profile_v1) == V1_FIXED_PROFILE_SIZE, "frozen v1 fixed profile size");
_Static_assert(offsetof(old_fixed_profile_v1, sampled_steps) == 48u, "frozen v1 sampled_steps");
_Static_assert(offsetof(old_fixed_profile_v1, scratch_peak_capacity_bytes) == 56u, "frozen v1 scratch metrics");
_Static_assert(offsetof(old_fixed_profile_v1, full_logits_checksums) == 144u, "frozen v1 checksums");
_Static_assert(sizeof(old_buffer_profile_v1) == V1_BUFFER_PROFILE_SIZE, "frozen v1 buffer profile size");
_Static_assert(offsetof(old_buffer_profile_v1, full_logits_checksums) == 144u, "frozen v1 checksum pointer");
_Static_assert(offsetof(old_buffer_profile_v1, full_logits_checksums_capacity) == 152u, "frozen v1 checksum capacity");

typedef struct guarded_fixed_profile {
    old_fixed_profile_v1 value;
    unsigned char trailing[256];
} guarded_fixed_profile;

typedef struct guarded_buffer_profile {
    old_buffer_profile_v1 value;
    unsigned char trailing[V2_BUFFER_PROFILE_SIZE - V1_BUFFER_PROFILE_SIZE + 32u];
} guarded_buffer_profile;

static void init_options(old_options_v1 *options) {
    memset(options, 0, sizeof(*options));
    options->struct_size = V1_OPTIONS_SIZE;
    options->version = 1u;
    options->io_backend = OLD_IO_BUFFERED;
    options->scratch_policy = OLD_SCRATCH_EPHEMERAL;
    options->kernel_policy = OLD_KERNEL_BASELINE;
    options->thread_policy = OLD_THREAD_SERIAL;
    options->thread_count = 1u;
}

static int bytes_are(const unsigned char *bytes, size_t count, unsigned char expected) {
    for (size_t i = 0; i < count; ++i) if (bytes[i] != expected) return 0;
    return 1;
}

static int reached_model_io(const char *err) {
    return err[0] != '\0' && strstr(err, "options") == NULL && strstr(err, "reserved") == NULL &&
        strstr(err, "profile") == NULL && strstr(err, "argument") == NULL;
}

static int test_fixed_v1_and_options(void) {
    static const uint32_t prompt[] = {9707u};
    old_options_v1 options;
    old_result result;
    guarded_fixed_profile guarded;
    char err[256] = {0};

    init_options(&options);
    memset(&result, 0x5a, sizeof(result));
    memset(&guarded, BYTE_CANARY, sizeof(guarded));
    if (qx_run_native_generation_with_options("abi-missing-model.qxf", prompt, 1u, 1u, 1u, -1,
            &options, &result, &guarded.value, err, sizeof(err)) != 0) return 10;
    if (!reached_model_io(err)) return 11;
    if (guarded.value.struct_size != V1_FIXED_PROFILE_SIZE || guarded.value.version != 1u) return 12;
    if (guarded.value.requested_io_backend != OLD_IO_BUFFERED || guarded.value.sampled_steps != 0u ||
            guarded.value.full_logits_checksums[0] != 0u) return 13;
    if (!bytes_are(guarded.trailing, sizeof(guarded.trailing), BYTE_CANARY)) return 14;

    init_options(&options);
    options.reserved[0] = 1u; /* legacy offset 28, called cuda_policy in the v2 declaration */
    memset(&guarded, BYTE_CANARY, sizeof(guarded));
    memset(err, 0, sizeof(err));
    if (qx_run_native_generation_with_options("abi-missing-model.qxf", prompt, 1u, 1u, 1u, -1,
            &options, &result, &guarded.value, err, sizeof(err)) != 0) return 15;
    if (strstr(err, "reserved") == NULL || strstr(err, "open") != NULL) return 16;
    if (!bytes_are(guarded.trailing, sizeof(guarded.trailing), BYTE_CANARY)) return 17;
    return 0;
}

static void init_buffer_call(old_options_v1 *options, old_buffer_result *result,
        guarded_buffer_profile *profile, uint32_t *tokens, uint64_t *checksums) {
    init_options(options);
    memset(result, 0, sizeof(*result));
    result->struct_size = (uint32_t)sizeof(*result);
    result->version = 1u;
    result->token_ids = tokens;
    result->token_capacity = 1u;
    memset(profile, BYTE_CANARY, sizeof(*profile));
    memset(&profile->value, 0, sizeof(profile->value));
    profile->value.struct_size = V1_BUFFER_PROFILE_SIZE;
    profile->value.version = 1u;
    profile->value.full_logits_checksums = checksums;
    profile->value.full_logits_checksums_capacity = 1u;
}

static int test_buffer_v1_prefix_and_clear(void) {
    static const uint32_t prompt[] = {9707u};
    old_options_v1 options;
    old_buffer_result result;
    guarded_buffer_profile profile;
    uint32_t token = UINT32_C(0xA17EC0DE);
    uint64_t checksum = WORD_CANARY;
    char err[256] = {0};

    init_buffer_call(&options, &result, &profile, &token, &checksum);
    if (qx_run_native_generation_into_with_options("abi-missing-model.qxf", prompt, 1u, 1u, 1u, -1,
            &options, &result, &profile.value, err, sizeof(err)) != 0) return 20;
    if (!reached_model_io(err)) return 21;
    if (profile.value.struct_size != V1_BUFFER_PROFILE_SIZE || profile.value.version != 1u) return 22;
    if (profile.value.full_logits_checksums != &checksum || profile.value.full_logits_checksums_capacity != 1u) return 23;
    if (result.token_ids != &token || result.token_capacity != 1u) return 24;
    if (token != 0u || checksum != 0u || profile.value.sampled_steps != 0u) return 25;
    if (!bytes_are(profile.trailing, sizeof(profile.trailing), BYTE_CANARY)) return 26;
    return 0;
}

static int test_truncated_v2_rejection_is_nonwriting(void) {
    static const uint32_t prompt[] = {9707u};
    old_options_v1 options;
    old_buffer_result result;
    old_buffer_result result_before;
    guarded_buffer_profile profile;
    guarded_buffer_profile profile_before;
    uint32_t token = UINT32_C(0xA17EC0DE);
    uint64_t checksum = WORD_CANARY;
    char err[256] = {0};

    init_buffer_call(&options, &result, &profile, &token, &checksum);
    profile.value.version = 2u;
    profile.value.struct_size = V1_BUFFER_PROFILE_SIZE; /* v2 requires its complete 368-byte extent */
    result_before = result;
    profile_before = profile;
    if (qx_run_native_generation_into_with_options("abi-missing-model.qxf", prompt, 1u, 1u, 1u, -1,
            &options, &result, &profile.value, err, sizeof(err)) != 0) return 30;
    if (strstr(err, "profile") == NULL || (strstr(err, "size") == NULL && strstr(err, "version") == NULL)) return 31;
    if (memcmp(&result, &result_before, sizeof(result)) != 0) return 32;
    if (memcmp(&profile, &profile_before, sizeof(profile)) != 0) return 33;
    if (token != UINT32_C(0xA17EC0DE) || checksum != WORD_CANARY) return 34;
    return 0;
}

int main(void) {
    int rc = test_fixed_v1_and_options();
    if (rc == 0) rc = test_buffer_v1_prefix_and_clear();
    if (rc == 0) rc = test_truncated_v2_rejection_is_nonwriting();
    if (rc != 0) {
        fprintf(stderr, "native CUDA ABI contract failed: %d\n", rc);
        return rc;
    }
    puts("native CUDA ABI contract: pass");
    return 0;
}
#endif
