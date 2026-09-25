#include "qx_format.h"

#include <stdio.h>
#include <string.h>

static int fail(const char *message, const char *detail) {
    fprintf(stderr, "native_cuda_fixed_v2_gpu_driver: %s%s%s\n",
        message, detail && detail[0] ? ": " : "", detail && detail[0] ? detail : "");
    return 1;
}

int main(int argc, char **argv) {
    static const uint32_t prompt[] = {9707u};
    qx_native_generation_options options;
    qx_native_generation_result result;
    qx_native_generation_profile_v2 profile;
    qx_native_generation_result legacy_result;
    char err[512] = {0};

    if (argc != 2) return fail("usage: driver MODEL.qxf", NULL);
    qx_native_generation_options_init(&options);
    options.cuda_policy = QX_NATIVE_CUDA_FINAL_HEAD_F32;
    qx_native_generation_profile_v2_init(&profile);

    if (!qx_run_native_generation_with_options_v2(argv[1], prompt, 1u, 2u, 8u, -1,
            &options, &result, &profile, err, sizeof(err)))
        return fail("fixed-v2 CUDA generation failed", err);
    if (result.token_count != 2u) return fail("fixed-v2 did not produce two outputs", NULL);
    if (profile.requested_cuda_policy != QX_NATIVE_CUDA_FINAL_HEAD_F32 ||
            profile.effective_cuda_policy != QX_NATIVE_CUDA_FINAL_HEAD_F32)
        return fail("fixed-v2 did not preserve explicit CUDA policy", NULL);
    if (profile.cuda_device_name[0] == '\0' || profile.cuda_compute_capability_major == 0u)
        return fail("fixed-v2 did not report an explicit CUDA device", NULL);
    if (profile.cuda_weight_uploads != 1u || profile.cuda_kernel_launches != 2u ||
            profile.cuda_cpu_fallbacks != 0u)
        return fail("fixed-v2 CUDA counters violated the upload/launch/fallback contract", NULL);

    memset(err, 0, sizeof(err));
    if (!qx_run_native_generation_with_options(argv[1], prompt, 1u, 1u, 8u, -1,
            &options, &legacy_result, NULL, err, sizeof(err)))
        return fail("legacy profile-NULL CUDA generation failed", err);
    if (legacy_result.token_count != 1u) return fail("legacy profile-NULL CUDA output missing", NULL);

    printf("{\"status\":\"pass\",\"fixed_v2_outputs\":[%u,%u],"
           "\"legacy_profile_null_output\":%u,\"requested_cuda_policy\":%d,"
           "\"effective_cuda_policy\":%d,\"device\":\"%s\","
           "\"compute_capability\":\"%u.%u\",\"weight_uploads\":%llu,"
           "\"kernel_launches\":%llu,\"cpu_fallbacks\":%llu}\n",
        result.token_ids[0], result.token_ids[1], legacy_result.token_ids[0],
        (int)profile.requested_cuda_policy, (int)profile.effective_cuda_policy,
        profile.cuda_device_name, profile.cuda_compute_capability_major,
        profile.cuda_compute_capability_minor,
        (unsigned long long)profile.cuda_weight_uploads,
        (unsigned long long)profile.cuda_kernel_launches,
        (unsigned long long)profile.cuda_cpu_fallbacks);
    return 0;
}