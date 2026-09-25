#include "qx_cuda_final_head.h"

#include <cuda_runtime_api.h>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static int fail(const char *message) {
    std::fprintf(stderr, "cuda_final_head_backend_driver: %s\n", message);
    return 1;
}

static float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t frac = h & 0x3ffu;
    uint32_t out;
    if (exp == 0u) {
        if (frac == 0u) {
            out = sign;
        } else {
            exp = 113u;
            while ((frac & 0x400u) == 0u) {
                frac <<= 1u;
                --exp;
            }
            out = sign | (exp << 23) | ((frac & 0x3ffu) << 13);
        }
    } else if (exp == 31u) {
        out = sign | 0x7f800000u | (frac << 13);
    } else {
        out = sign | ((exp + 112u) << 23) | (frac << 13);
    }
    float result;
    std::memcpy(&result, &out, sizeof(result));
    return result;
}

/* Test-only scalar decoder, deliberately separate from the CUDA implementation. */
static float reference_q6_value(const uint8_t *block, uint32_t index) {
    const uint32_t half = index / 128u;
    const uint32_t local = index % 128u;
    const uint32_t lane = local % 32u;
    const uint32_t section = local / 32u;
    const uint8_t *ql = block + half * 64u;
    const uint8_t *qh = block + 128u + half * 32u;
    const int8_t *scales = reinterpret_cast<const int8_t *>(block + 192u + half * 8u);
    const uint32_t packed_index = lane + ((section & 1u) ? 32u : 0u);
    const uint32_t nibble = section < 2u ? (ql[packed_index] & 0x0fu) : (ql[packed_index] >> 4u);
    const uint32_t high = (qh[lane] >> (section * 2u)) & 0x03u;
    const int32_t quant = (int32_t)(nibble | (high << 4u)) - 32;
    const uint32_t scale_index = section * 2u + lane / 16u;
    const uint16_t dh = (uint16_t)block[208] | ((uint16_t)block[209] << 8u);
    return half_to_float(dh) * (float)scales[scale_index] * (float)quant;
}

static int run_independent_256_value_golden(char *err, size_t err_len) {
    /* Analytic block: d=1, scales=1, ql halves 0xF0/0x01, qh=0xE4.
       Each 128-value half decodes to 32 copies of {-32,-15,15,16};
       dot(all-ones) is therefore exactly -1024. */
    std::vector<uint8_t> packed(210u, 0u);
    for (uint32_t i = 0; i < 32u; ++i) packed[i] = 0xf0u;
    for (uint32_t i = 32u; i < 64u; ++i) packed[i] = 0x01u;
    for (uint32_t i = 64u; i < 96u; ++i) packed[i] = 0xf0u;
    for (uint32_t i = 96u; i < 128u; ++i) packed[i] = 0x01u;
    for (uint32_t i = 128u; i < 192u; ++i) packed[i] = 0xe4u;
    for (uint32_t i = 192u; i < 208u; ++i) packed[i] = 1u;
    packed[208] = 0x00u;
    packed[209] = 0x3cu;

    float reference_sum = 0.0f;
    for (uint32_t i = 0; i < 256u; ++i) reference_sum += reference_q6_value(packed.data(), i);
    if (reference_sum != -1024.0f) return fail("independent 256-value Q6_K golden is internally wrong");

    std::vector<float> activation(256u, 1.0f);
    float logit = 123.0f;
    qx_cuda_final_head_context *context = nullptr;
    if (!qx_cuda_final_head_create_q6_k_f32(packed.data(), packed.size(), 256u, 1u,
                                             &context, err, (uint64_t)err_len)) return fail(err);
    if (!qx_cuda_final_head_compute_f32(context, activation.data(), 256u, &logit, 1u,
                                         err, (uint64_t)err_len)) {
        qx_cuda_final_head_destroy(&context);
        return fail(err);
    }
    if (logit != -1024.0f) {
        qx_cuda_final_head_destroy(&context);
        return fail("CUDA failed independent 256-value Q6_K analytic golden");
    }
    qx_cuda_final_head_destroy(&context);
    if (context != nullptr) return fail("golden context destroy did not clear pointer");
    return 0;
}

static int run_repeated_destroy_memory_gate(const std::vector<uint8_t> &weights,
                                            uint32_t hidden, uint32_t vocab,
                                            char *err, size_t err_len) {
    size_t free_before = 0, total = 0, free_after = 0;
    if (cudaDeviceSynchronize() != cudaSuccess || cudaMemGetInfo(&free_before, &total) != cudaSuccess)
        return fail("cudaMemGetInfo baseline failed");
    for (uint32_t cycle = 0; cycle < 12u; ++cycle) {
        qx_cuda_final_head_context *context = nullptr;
        if (!qx_cuda_final_head_create_q6_k_f32(weights.data(), weights.size(), hidden, vocab,
                                                 &context, err, (uint64_t)err_len)) return fail(err);
        qx_cuda_final_head_destroy(&context);
        if (context != nullptr) return fail("repeated destroy did not clear context");
    }
    if (cudaDeviceSynchronize() != cudaSuccess || cudaMemGetInfo(&free_after, &total) != cudaSuccess)
        return fail("cudaMemGetInfo final sample failed");
    /* CUDA may retain runtime/allocator bookkeeping; 8 MiB is a bounded tolerance,
       not a fabricated zero-leak claim. Twelve complete create/destroy cycles must
       not cause unbounded resident device growth. */
    constexpr size_t tolerance = 8u * 1024u * 1024u;
    if (free_after + tolerance < free_before) return fail("repeated destroy exceeded 8 MiB device-memory tolerance");
    return 0;
}

struct cli_options {
    const char *weights = nullptr;
    const char *activation = nullptr;
    const char *logits = nullptr;
    const char *repeat_logits = nullptr;
    uint32_t hidden = 0;
    uint32_t vocab = 0;
    uint32_t calls = 0;
    bool json = false;
};

static bool parse_u32(const char *text, uint32_t *out) {
    if (!text || !*text || *text == '-') return false;
    char *end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0' || value == 0 || value > UINT32_MAX) return false;
    *out = (uint32_t)value;
    return true;
}

static bool parse_cli(int argc, char **argv, cli_options *options) {
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (std::strcmp(arg, "--json") == 0) {
            if (options->json) return false;
            options->json = true;
            continue;
        }
        if (i + 1 >= argc) return false;
        const char *value = argv[++i];
        if (std::strcmp(arg, "--weights") == 0 && !options->weights) options->weights = value;
        else if (std::strcmp(arg, "--activation") == 0 && !options->activation) options->activation = value;
        else if (std::strcmp(arg, "--logits") == 0 && !options->logits) options->logits = value;
        else if (std::strcmp(arg, "--repeat-logits") == 0 && !options->repeat_logits) options->repeat_logits = value;
        else if (std::strcmp(arg, "--hidden") == 0 && options->hidden == 0 && parse_u32(value, &options->hidden)) {}
        else if (std::strcmp(arg, "--vocab") == 0 && options->vocab == 0 && parse_u32(value, &options->vocab)) {}
        else if (std::strcmp(arg, "--calls") == 0 && options->calls == 0 && parse_u32(value, &options->calls)) {}
        else return false;
    }
    return options->weights && options->activation && options->logits && options->repeat_logits &&
           options->hidden && options->vocab && options->calls && options->json;
}

static bool read_exact_file(const char *path, uint64_t expected, std::vector<uint8_t> *bytes) {
    constexpr uint64_t max_input_bytes = 1ull << 30;
    if (expected == 0 || expected > max_input_bytes || expected > SIZE_MAX) return false;
    FILE *file = std::fopen(path, "rb");
    if (!file) return false;
    bool ok = std::fseek(file, 0, SEEK_END) == 0;
    const long size = ok ? std::ftell(file) : -1;
    ok = ok && size >= 0 && (uint64_t)size == expected && std::fseek(file, 0, SEEK_SET) == 0;
    if (ok) {
        bytes->resize((size_t)expected);
        ok = std::fread(bytes->data(), 1, bytes->size(), file) == bytes->size();
    }
    if (std::fclose(file) != 0) ok = false;
    if (!ok) bytes->clear();
    return ok;
}

static bool write_temp(const std::string &path, const void *data, size_t size) {
    FILE *file = std::fopen(path.c_str(), "wb");
    if (!file) return false;
    const bool ok = std::fwrite(data, 1, size, file) == size && std::fflush(file) == 0;
    const bool closed = std::fclose(file) == 0;
    if (!ok || !closed) std::remove(path.c_str());
    return ok && closed;
}

static bool publish_outputs(const char *logits_path, const char *repeat_path,
                            const std::vector<float> &logits, const std::vector<float> &repeat) {
    if (std::strcmp(logits_path, repeat_path) == 0) return false;
    const std::string first_tmp = std::string(logits_path) + ".qx-tmp";
    const std::string repeat_tmp = std::string(repeat_path) + ".qx-tmp";
    std::remove(first_tmp.c_str());
    std::remove(repeat_tmp.c_str());
    const size_t bytes = logits.size() * sizeof(float);
    if (!write_temp(first_tmp, logits.data(), bytes) || !write_temp(repeat_tmp, repeat.data(), bytes)) {
        std::remove(first_tmp.c_str());
        std::remove(repeat_tmp.c_str());
        return false;
    }
    std::remove(logits_path);
    std::remove(repeat_path);
    if (std::rename(first_tmp.c_str(), logits_path) != 0) {
        std::remove(first_tmp.c_str());
        std::remove(repeat_tmp.c_str());
        return false;
    }
    if (std::rename(repeat_tmp.c_str(), repeat_path) != 0) {
        std::remove(logits_path);
        std::remove(repeat_tmp.c_str());
        return false;
    }
    return true;
}

static std::string json_string(const char *text) {
    std::string out = "\"";
    for (const unsigned char *p = reinterpret_cast<const unsigned char *>(text); *p; ++p) {
        if (*p == '\\' || *p == '"') { out.push_back('\\'); out.push_back((char)*p); }
        else if (*p < 0x20) {
            char escaped[7];
            std::snprintf(escaped, sizeof(escaped), "\\u%04x", (unsigned)*p);
            out += escaped;
        } else out.push_back((char)*p);
    }
    out.push_back('"');
    return out;
}

static int run_cli(const cli_options &options) {
    if (options.hidden % 256u != 0 || options.calls > 100000u)
        return fail("invalid dimensions or call count");
    const uint64_t blocks_per_row = options.hidden / 256u;
    if (blocks_per_row > UINT64_MAX / 210u) return fail("weight size overflow");
    const uint64_t row_bytes = blocks_per_row * 210u;
    if (options.vocab > UINT64_MAX / row_bytes) return fail("weight size overflow");
    const uint64_t weights_bytes = row_bytes * options.vocab;
    const uint64_t activation_bytes = (uint64_t)options.hidden * sizeof(float);
    const uint64_t logits_bytes = (uint64_t)options.vocab * sizeof(float);
    if (activation_bytes > SIZE_MAX || logits_bytes > SIZE_MAX) return fail("tensor size overflow");

    std::vector<uint8_t> weights_raw, activation_raw;
    if (!read_exact_file(options.weights, weights_bytes, &weights_raw)) return fail("weights file has wrong size or cannot be read");
    if (!read_exact_file(options.activation, activation_bytes, &activation_raw)) return fail("activation file has wrong size or cannot be read");
    std::vector<float> activation(options.hidden);
    std::memcpy(activation.data(), activation_raw.data(), (size_t)activation_bytes);
    for (float value : activation) if (!std::isfinite(value)) return fail("activation contains non-finite value");

    qx_cuda_final_head_device device{};
    char err[256]{};
    if (!qx_cuda_final_head_is_available(&device, err, sizeof(err))) return fail(err);
    if (device.name[0] == '\0') return fail("empty device name");

    qx_cuda_final_head_context *context = nullptr;
    if (!qx_cuda_final_head_create_q6_k_f32(weights_raw.data(), weights_raw.size(), options.hidden,
                                             options.vocab, &context, err, sizeof(err))) return fail(err);
    std::vector<float> logits(options.vocab), first(options.vocab);
    bool ok = true;
    for (uint32_t call = 0; call < options.calls; ++call) {
        if (!qx_cuda_final_head_compute_f32(context, activation.data(), options.hidden,
                                             logits.data(), options.vocab, err, sizeof(err))) {
            ok = false;
            break;
        }
        for (float value : logits) if (!std::isfinite(value)) { ok = false; std::strcpy(err, "backend produced non-finite logit"); break; }
        if (!ok) break;
        if (call == 0) first = logits;
    }
    qx_cuda_final_head_counters counters{};
    if (ok && !qx_cuda_final_head_get_counters(context, &counters, err, sizeof(err))) ok = false;
    qx_cuda_final_head_destroy(&context);
    if (!ok) return fail(err);
    if (context != nullptr) return fail("destroy did not clear context");
    if (std::memcmp(first.data(), logits.data(), (size_t)logits_bytes) != 0) return fail("repeated logits differ");
    if (!publish_outputs(options.logits, options.repeat_logits, first, logits)) return fail("failed to publish output files");

    const std::string device_name = json_string(device.name);
    std::printf("{\"schema\":\"qx.cuda-final-head-driver.v1\",\"status\":\"pass\",\"backend\":\"cuda\","
                "\"hidden\":%u,\"vocab\":%u,\"calls\":%u,"
                "\"device\":{\"name\":%s,\"compute_capability_major\":%u,\"compute_capability_minor\":%u},"
                "\"counters\":{\"resident_weight_bytes\":%llu,\"persistent_allocations\":%llu,"
                "\"weight_uploads\":%llu,\"weight_upload_bytes\":%llu,\"host_to_device_bytes\":%llu,"
                "\"device_to_host_bytes\":%llu,\"kernel_launches\":%llu,\"cpu_fallbacks\":%llu}}\n",
                options.hidden, options.vocab, options.calls, device_name.c_str(),
                device.compute_capability_major, device.compute_capability_minor,
                (unsigned long long)counters.resident_weight_bytes,
                (unsigned long long)counters.persistent_allocations,
                (unsigned long long)counters.weight_uploads,
                (unsigned long long)counters.weight_upload_bytes,
                (unsigned long long)counters.host_to_device_bytes,
                (unsigned long long)counters.device_to_host_bytes,
                (unsigned long long)counters.kernel_launches,
                (unsigned long long)counters.cpu_fallbacks);
    return 0;
}

static int run_self_tests() {
    qx_cuda_final_head_device device{};
    char err[256]{};
    if (!qx_cuda_final_head_is_available(&device, err, sizeof(err))) {
        std::fprintf(stderr, "SKIP: %s\n", err);
        return 77;
    }
    if (device.name[0] == '\0') return fail("empty device name");
    if (run_independent_256_value_golden(err, sizeof(err)) != 0) return 1;

    constexpr uint32_t hidden = 512u;
    constexpr uint32_t vocab = 513u;
    constexpr uint64_t row_bytes = (hidden / 256u) * 210u;
    std::vector<uint8_t> weights((size_t)vocab * row_bytes);
    uint32_t state = 0x12345678u;
    for (size_t i = 0; i < weights.size(); ++i) {
        state = state * 1664525u + 1013904223u;
        weights[i] = (uint8_t)(state >> 24u);
    }
    for (uint32_t row = 0; row < vocab; ++row) {
        for (uint32_t block_index = 0; block_index < hidden / 256u; ++block_index) {
            uint8_t *block = weights.data() + (size_t)row * row_bytes + block_index * 210u;
            for (uint32_t i = 0; i < 16u; ++i) block[192u + i] = (uint8_t)((int)(i % 7u) - 3);
            block[208] = 0x00u;
            block[209] = 0x3cu;
        }
    }
    std::vector<float> activation(hidden);
    for (uint32_t i = 0; i < hidden; ++i) activation[i] = std::sin((float)i * 0.03125f) * 0.125f;
    std::vector<float> expected(vocab);
    for (uint32_t row = 0; row < vocab; ++row) {
        double sum = 0.0;
        const uint8_t *row_weights = weights.data() + (size_t)row * row_bytes;
        for (uint32_t i = 0; i < hidden; ++i)
            sum += (double)reference_q6_value(row_weights + (i / 256u) * 210u, i % 256u) * (double)activation[i];
        expected[row] = (float)sum;
    }

    qx_cuda_final_head_context *context = reinterpret_cast<qx_cuda_final_head_context *>(uintptr_t{1});
    if (qx_cuda_final_head_create_q6_k_f32(weights.data(), weights.size(), 255u, vocab,
                                            &context, err, sizeof(err)) != 0 || context != nullptr)
        return fail("invalid geometry accepted or output context not cleared");
    if (qx_cuda_final_head_create_q6_k_f32(weights.data(), weights.size() - 1u, hidden, vocab,
                                            &context, err, sizeof(err)) != 0 || context != nullptr)
        return fail("invalid packed size accepted or output context not cleared");
    if (!qx_cuda_final_head_create_q6_k_f32(weights.data(), weights.size(), hidden, vocab,
                                             &context, err, sizeof(err))) return fail(err);

    qx_cuda_final_head_counters counters{};
    if (!qx_cuda_final_head_get_counters(context, &counters, err, sizeof(err))) return fail(err);
    if (counters.resident_weight_bytes != weights.size() || counters.persistent_allocations != 3u ||
        counters.weight_uploads != 1u || counters.weight_upload_bytes != weights.size() ||
        counters.host_to_device_bytes != 0u || counters.device_to_host_bytes != 0u ||
        counters.kernel_launches != 0u || counters.cpu_fallbacks != 0u) return fail("incorrect setup counters");

    std::vector<float> logits(vocab, -777.0f);
    if (qx_cuda_final_head_compute_f32(context, activation.data(), hidden, logits.data(), vocab - 1u,
                                        err, sizeof(err)) != 0) return fail("undersized output accepted");
    for (float value : logits) if (value != -777.0f) return fail("failed capacity call modified logits");
    activation[17] = std::numeric_limits<float>::quiet_NaN();
    if (qx_cuda_final_head_compute_f32(context, activation.data(), hidden, logits.data(), vocab,
                                        err, sizeof(err)) != 0) return fail("non-finite activation accepted");
    for (float value : logits) if (value != -777.0f) return fail("failed finite check modified logits");
    activation[17] = std::sin(17.0f * 0.03125f) * 0.125f;

    if (!qx_cuda_final_head_compute_f32(context, activation.data(), hidden, logits.data(), vocab,
                                         err, sizeof(err))) return fail(err);
    const std::vector<float> first = logits;
    if (!qx_cuda_final_head_compute_f32(context, activation.data(), hidden, logits.data(), vocab,
                                         err, sizeof(err))) return fail(err);
    if (std::memcmp(first.data(), logits.data(), logits.size() * sizeof(float)) != 0) return fail("non-deterministic logits");

    double squared_error = 0.0, expected_norm = 0.0, actual_norm = 0.0, dot = 0.0;
    float max_abs = 0.0f;
    for (uint32_t i = 0; i < vocab; ++i) {
        if (!std::isfinite(logits[i])) return fail("non-finite logit");
        const float error = std::fabs(logits[i] - expected[i]);
        if (error > max_abs) max_abs = error;
        squared_error += (double)error * error;
        expected_norm += (double)expected[i] * expected[i];
        actual_norm += (double)logits[i] * logits[i];
        dot += (double)expected[i] * logits[i];
    }
    const double rmse = std::sqrt(squared_error / vocab);
    const double cosine = dot / std::sqrt(expected_norm * actual_norm);
    if (max_abs > 1.0e-4f || rmse > 2.0e-5 || cosine < 0.999999) return fail("numerical parity gate failed");

    if (!qx_cuda_final_head_get_counters(context, &counters, err, sizeof(err))) return fail(err);
    if (counters.weight_uploads != 1u || counters.weight_upload_bytes != weights.size() ||
        counters.host_to_device_bytes != 2ull * hidden * sizeof(float) ||
        counters.device_to_host_bytes != 2ull * vocab * sizeof(float) ||
        counters.kernel_launches != 2u || counters.cpu_fallbacks != 0u) return fail("incorrect per-call counters");
    qx_cuda_final_head_destroy(&context);
    if (context != nullptr) return fail("destroy did not clear context");
    qx_cuda_final_head_destroy(&context);
    if (run_repeated_destroy_memory_gate(weights, hidden, vocab, err, sizeof(err)) != 0) return 1;
    std::printf("PASS device=%s cc=%u.%u max_abs=%.9g rmse=%.9g cosine=%.12g destroy_cycles=12 memory_tolerance_bytes=8388608\n",
                device.name, device.compute_capability_major, device.compute_capability_minor,
                max_abs, rmse, cosine);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 1) return run_self_tests();
    cli_options options;
    if (!parse_cli(argc, argv, &options)) return fail("expected --weights --activation --hidden --vocab --calls --logits --repeat-logits --json exactly once");
    return run_cli(options);
}
