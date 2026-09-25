#include "qx_format.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define BYTE_CANARY 0xA5u

_Static_assert(sizeof(qx_native_generation_profile) == 656u, "legacy fixed profile ABI size");
_Static_assert(sizeof(qx_native_generation_profile_v2) == 864u, "sized fixed profile v2 ABI size");
_Static_assert(offsetof(qx_native_generation_profile_v2, full_logits_checksums) == 144u, "v2 legacy checksum prefix");
_Static_assert(offsetof(qx_native_generation_profile_v2, requested_cuda_policy) == 656u, "v2 CUDA extension offset");
_Static_assert(offsetof(qx_native_generation_profile_v2, cuda_device_name) == 664u, "v2 CUDA device offset");
_Static_assert(offsetof(qx_native_generation_profile_v2, cuda_resident_weight_bytes) == 800u, "v2 CUDA counters offset");

typedef struct guarded_v1 { qx_native_generation_profile value; unsigned char trailing[32]; } guarded_v1;
typedef struct guarded_v2 { qx_native_generation_profile_v2 value; unsigned char trailing[32]; } guarded_v2;

static int bytes_are(const unsigned char *bytes, size_t count, unsigned char expected) {
    for (size_t i = 0; i < count; ++i) if (bytes[i] != expected) return 0;
    return 1;
}

static void init_cuda_options(qx_native_generation_options *options) {
    qx_native_generation_options_init(options);
    options->cuda_policy = QX_NATIVE_CUDA_FINAL_HEAD_F32;
}

static int test_initializer(void) {
    qx_native_generation_profile_v2 profile;
    memset(&profile, BYTE_CANARY, sizeof(profile));
    qx_native_generation_profile_v2_init(&profile);
    if (profile.struct_size != sizeof(profile) || profile.version != QX_NATIVE_GENERATION_PROFILE_V2_VERSION) return 10;
    if (profile.requested_cuda_policy != QX_NATIVE_CUDA_NONE || profile.full_logits_checksums[0] != 0u) return 11;
    return 0;
}

static int test_legacy_fixed_cuda_contract(void) {
    static const uint32_t prompt[] = {9707u};
    qx_native_generation_options options;
    qx_native_generation_result result;
    guarded_v1 profile;
    char err[256] = {0};
    init_cuda_options(&options);
    memset(&result, BYTE_CANARY, sizeof(result));
    memset(&profile, BYTE_CANARY, sizeof(profile));
    if (qx_run_native_generation_with_options("fixed-v2-missing.qxf", prompt, 1u, 1u, 1u, -1,
            &options, &result, &profile.value, err, sizeof(err)) != 0) return 20;
    if (strcmp(err, "legacy fixed profile cannot represent CUDA provenance; use qx_run_native_generation_with_options_v2 or pass NULL") != 0) return 21;
    if (!bytes_are(profile.trailing, sizeof(profile.trailing), BYTE_CANARY)) return 22;

    memset(err, 0, sizeof(err));
    if (qx_run_native_generation_with_options("fixed-v2-missing.qxf", prompt, 1u, 1u, 1u, -1,
            &options, &result, NULL, err, sizeof(err)) != 0) return 23;
    if (strstr(err, "CUDA final-head backend unavailable") == NULL || strstr(err, "open") != NULL) return 24;
    return 0;
}

static int test_v2_malformed_is_nonwriting(void) {
    static const uint32_t prompt[] = {9707u};
    qx_native_generation_options options;
    qx_native_generation_result result, result_before;
    guarded_v2 profile, profile_before;
    char err[256] = {0};
    qx_native_generation_options_init(&options);
    memset(&result, BYTE_CANARY, sizeof(result));
    memset(&profile, BYTE_CANARY, sizeof(profile));
    profile.value.struct_size = (uint32_t)sizeof(profile.value) - 1u;
    profile.value.version = QX_NATIVE_GENERATION_PROFILE_V2_VERSION;
    result_before = result; profile_before = profile;
    if (qx_run_native_generation_with_options_v2("fixed-v2-missing.qxf", prompt, 1u, 1u, 1u, -1,
            &options, &result, &profile.value, err, sizeof(err)) != 0) return 30;
    if (strstr(err, "fixed profile v2 size") == NULL) return 31;
    if (memcmp(&result, &result_before, sizeof(result)) != 0 || memcmp(&profile, &profile_before, sizeof(profile)) != 0) return 32;

    profile.value.struct_size = (uint32_t)sizeof(profile.value);
    profile.value.version = QX_NATIVE_GENERATION_PROFILE_V2_VERSION + 1u;
    result_before = result; profile_before = profile;
    memset(err, 0, sizeof(err));
    if (qx_run_native_generation_with_options_v2("fixed-v2-missing.qxf", prompt, 1u, 1u, 1u, -1,
            &options, &result, &profile.value, err, sizeof(err)) != 0) return 33;
    if (strstr(err, "fixed profile v2 version") == NULL) return 34;
    if (memcmp(&result, &result_before, sizeof(result)) != 0 || memcmp(&profile, &profile_before, sizeof(profile)) != 0) return 35;
    return 0;
}

static int test_v2_cpu_legacy_compatible_output(void) {
    static const uint32_t prompt[] = {9707u};
    qx_native_generation_options options;
    qx_native_generation_result result;
    guarded_v2 profile;
    char err[256] = {0};
    qx_native_generation_options_init(&options);
    memset(&result, BYTE_CANARY, sizeof(result));
    memset(&profile, BYTE_CANARY, sizeof(profile));
    qx_native_generation_profile_v2_init(&profile.value);
    memset(profile.trailing, BYTE_CANARY, sizeof(profile.trailing));
    if (qx_run_native_generation_with_options_v2("fixed-v2-missing.qxf", prompt, 1u, 1u, 1u, -1,
            &options, &result, &profile.value, err, sizeof(err)) != 0) return 40;
    if (err[0] == '\0' || strstr(err, "profile") != NULL || strstr(err, "argument") != NULL) return 41;
    if (result.token_count != 0u || result.token_ids[0] != 0u) return 42;
    if (profile.value.struct_size != sizeof(profile.value) || profile.value.version != QX_NATIVE_GENERATION_PROFILE_V2_VERSION) return 43;
    if (profile.value.requested_cuda_policy != QX_NATIVE_CUDA_NONE || profile.value.effective_cuda_policy != QX_NATIVE_CUDA_NONE) return 44;
    if (profile.value.sampled_steps != 0u || profile.value.full_logits_checksums[0] != 0u) return 45;
    if (!bytes_are(profile.trailing, sizeof(profile.trailing), BYTE_CANARY)) return 46;
    return 0;
}

int main(void) {
    int rc = test_initializer();
    if (rc == 0) rc = test_legacy_fixed_cuda_contract();
    if (rc == 0) rc = test_v2_malformed_is_nonwriting();
    if (rc == 0) rc = test_v2_cpu_legacy_compatible_output();
    if (rc != 0) { fprintf(stderr, "native fixed CUDA v2 contract failed: %d\n", rc); return rc; }
    puts("native fixed CUDA v2 contract: pass");
    return 0;
}