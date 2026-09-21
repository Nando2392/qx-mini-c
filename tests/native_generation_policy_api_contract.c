#include "qx_format.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int expect_preflight_failure(qx_native_generation_options *options, const char *needle) {
    static const uint32_t prompt[] = {9707u};
    qx_native_generation_result result;
    qx_native_generation_profile profile;
    char err[256] = {0};
    int ok = qx_run_native_generation_with_options(
        "definitely-missing-model.qxf", prompt, 1u, 1u, 1u, -1,
        options, &result, &profile, err, sizeof(err));
    return !ok && strstr(err, needle) != NULL && strstr(err, "open") == NULL;
}

static int verify_global_backend(const char *model, qx_io_backend expected) {
    qx_file file;
    char err[256] = {0};
    if (!qx_open_file(model, &file, err, sizeof(err))) {
        fprintf(stderr, "[ISSUE82-BACKEND] probe-open-error=\"%s\" expected=%s\n",
            err, qx_io_backend_name(expected));
        return 0;
    }
    qx_io_backend actual = file.io_backend;
    int matched = actual == expected;
    qx_close_file(&file);
    if (!matched) {
        fprintf(stderr, "[ISSUE82-BACKEND] expected=%s actual=%s\n",
            qx_io_backend_name(expected), qx_io_backend_name(actual));
    }
    return matched;
}

static int run_preflight_contract(void) {
    qx_native_generation_options options;
    qx_native_generation_options_init(&options);
    if (options.struct_size != sizeof(options) || options.version != QX_NATIVE_GENERATION_OPTIONS_VERSION) return 1;
    if (options.io_backend != QX_NATIVE_IO_BUFFERED ||
        options.scratch_policy != QX_NATIVE_SCRATCH_EPHEMERAL ||
        options.kernel_policy != QX_NATIVE_KERNEL_BASELINE ||
        options.thread_policy != QX_NATIVE_THREAD_SERIAL || options.thread_count != 1u) return 2;

    options.struct_size--;
    if (!expect_preflight_failure(&options, "unsupported native generation options size")) return 3;
    qx_native_generation_options_init(&options);
    options.version++;
    if (!expect_preflight_failure(&options, "unsupported native generation options version")) return 4;
    qx_native_generation_options_init(&options);
    options.io_backend = (qx_native_io_policy)99;
    if (!expect_preflight_failure(&options, "unsupported native generation I/O policy")) return 5;
    qx_native_generation_options_init(&options);
    options.scratch_policy = (qx_native_scratch_policy)-1;
    if (!expect_preflight_failure(&options, "unsupported native generation scratch policy")) return 6;
    qx_native_generation_options_init(&options);
    options.kernel_policy = (qx_native_kernel_policy)99;
    if (!expect_preflight_failure(&options, "unsupported native generation kernel policy")) return 7;
    qx_native_generation_options_init(&options);
    options.thread_policy = (qx_native_thread_policy)-1;
    if (!expect_preflight_failure(&options, "unsupported native generation thread policy")) return 8;
    qx_native_generation_options_init(&options);
    options.thread_count = 2u;
    if (!expect_preflight_failure(&options, "serial native generation policy requires one thread")) return 9;
    qx_native_generation_options_init(&options);
    options.thread_policy = QX_NATIVE_THREAD_POOL;
    options.thread_count = 1u;
    if (!expect_preflight_failure(&options, "pool native generation policy requires 2..64 threads")) return 10;
    options.thread_count = 65u;
    if (!expect_preflight_failure(&options, "pool native generation policy requires 2..64 threads")) return 11;
    qx_native_generation_options_init(&options);
    options.reserved[0] = 1u;
    if (!expect_preflight_failure(&options, "reserved native generation option fields must be zero")) return 12;

    puts("native generation policy API contract: pass");
    return 0;
}

static int run_real_model_contract(const char *model) {
    static const uint32_t prompt[] = {9707u};
    qx_native_generation_options options;
    qx_native_generation_result result;
    qx_native_generation_profile profile;
    qx_file model_file;
    char err[512] = {0};

    if (!qx_set_io_backend("buffered", err, sizeof(err))) return 20;
    qx_native_generation_options_init(&options);
    options.io_backend = QX_NATIVE_IO_MMAP;
    if (qx_run_native_generation_with_options(
            "definitely-missing-model.qxf", prompt, 1u, 1u, 1u, -1,
            &options, &result, &profile, err, sizeof(err))) return 21;
    if (result.token_count != 0u || profile.struct_size != sizeof(profile) ||
            profile.version != QX_NATIVE_GENERATION_PROFILE_VERSION || profile.sampled_steps != 0u ||
            profile.full_logits_checksums[0] != 0u) return 22;
    int error_reported = err[0] != '\0';
    int backend_restored = verify_global_backend(model, QX_IO_BUFFERED);
    if (!error_reported || !backend_restored) {
        fprintf(stderr,
            "[ISSUE82-MISSING-MODEL] error=\"%s\" error_reported=%d backend_restored=%d expected_backend=%s\n",
            err, error_reported, backend_restored, qx_io_backend_name(QX_IO_BUFFERED));
        return 23;
    }

    qx_native_generation_options_init(&options);
    options.io_backend = QX_NATIVE_IO_MMAP;
    options.scratch_policy = QX_NATIVE_SCRATCH_PERSISTENT;
    options.kernel_policy = QX_NATIVE_KERNEL_FUSED_FINAL_HEAD;
    options.thread_policy = QX_NATIVE_THREAD_POOL;
    options.thread_count = 2u;
    if (!qx_run_native_generation_with_options(
            model, prompt, 1u, 1u, 1u, -1, &options, &result, &profile, err, sizeof(err))) {
        fprintf(stderr, "optimized native generation failed: %s\n", err);
        return 24;
    }
    if (!verify_global_backend(model, QX_IO_BUFFERED)) return 25;
    if (!qx_open_file(model, &model_file, err, sizeof(err))) return 26;
    uint64_t blocks = (uint64_t)result.token_count * model_file.header.manifest.vocab *
        (model_file.header.manifest.hidden / 256u);
    uint64_t jobs = (uint64_t)result.token_count * options.thread_count;
    qx_close_file(&model_file);

    if (result.token_count != 1u || profile.struct_size != sizeof(profile) ||
            profile.version != QX_NATIVE_GENERATION_PROFILE_VERSION ||
            profile.requested_io_backend != QX_NATIVE_IO_MMAP || profile.effective_io_backend != QX_NATIVE_IO_MMAP ||
            profile.requested_scratch_policy != QX_NATIVE_SCRATCH_PERSISTENT || profile.effective_scratch_policy != QX_NATIVE_SCRATCH_PERSISTENT ||
            profile.requested_kernel_policy != QX_NATIVE_KERNEL_FUSED_FINAL_HEAD || profile.effective_kernel_policy != QX_NATIVE_KERNEL_FUSED_FINAL_HEAD ||
            profile.requested_thread_policy != QX_NATIVE_THREAD_POOL || profile.effective_thread_policy != QX_NATIVE_THREAD_POOL ||
            profile.requested_thread_count != 2u || profile.effective_thread_count != 2u ||
            profile.sampled_steps != result.token_count || profile.workers_used != 2u ||
            profile.full_logits_checksums[0] == 0u || profile.full_logits_checksums[1] != 0u) return 27;
    if (profile.scratch_peak_capacity_bytes == 0u || profile.scratch_growth_events == 0u ||
            profile.temporary_blocks_decoded != 0u || profile.temporary_floats_materialized != 0u ||
            profile.temporary_bytes_materialized != 0u ||
            profile.fused_final_head_dot_calls != blocks || profile.baseline_final_head_dot_calls != 0u ||
            profile.final_head_q6_k_blocks != blocks || profile.final_head_parallel_jobs != jobs ||
            profile.final_head_serial_jobs != 0u || profile.final_head_fallback_jobs != 0u) return 28;

    puts("native generation policy real-model contract: pass");
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 1) return run_preflight_contract();
    if (argc == 2) return run_real_model_contract(argv[1]);
    fprintf(stderr, "usage: %s [MODEL]\n", argv[0]);
    return 2;
}
