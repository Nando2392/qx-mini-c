#include "qx_cuda_final_head.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

struct qx_cuda_final_head_context {
    unsigned char *device_weights;
    float *device_activation;
    float *device_logits;
    uint32_t hidden;
    uint32_t vocab;
    uint64_t packed_bytes;
    qx_cuda_final_head_counters counters;
};

static void qx_clear_error(char *err, uint64_t err_len) {
    if (err != nullptr && err_len > 0u) err[0] = '\0';
}

static int qx_fail(char *err, uint64_t err_len, const char *message) {
    if (err != nullptr && err_len > 0u) {
        std::snprintf(err, (size_t)err_len, "%s", message);
        err[err_len - 1u] = '\0';
    }
    return 0;
}

static int qx_cuda_fail(char *err, uint64_t err_len, const char *operation, cudaError_t status) {
    char message[384];
    std::snprintf(message, sizeof(message), "%s: %s", operation, cudaGetErrorString(status));
    return qx_fail(err, err_len, message);
}

__device__ static float qx_half_to_float(uint16_t value) {
    const uint32_t sign = (uint32_t)(value & 0x8000u) << 16u;
    uint32_t exponent = (value >> 10u) & 0x1fu;
    uint32_t fraction = value & 0x3ffu;
    uint32_t bits;
    if (exponent == 0u) {
        if (fraction == 0u) {
            bits = sign;
        } else {
            exponent = 113u;
            while ((fraction & 0x400u) == 0u) {
                fraction <<= 1u;
                --exponent;
            }
            bits = sign | (exponent << 23u) | ((fraction & 0x3ffu) << 13u);
        }
    } else if (exponent == 31u) {
        bits = sign | 0x7f800000u | (fraction << 13u);
    } else {
        bits = sign | ((exponent + 112u) << 23u) | (fraction << 13u);
    }
    return __uint_as_float(bits);
}

/* GGML Q6_K block layout: 128 ql bytes, 64 qh bytes, 16 i8 scales, f16 d. */
__device__ static float qx_q6_k_value(const unsigned char *block, uint32_t index) {
    const uint32_t half = index >> 7u;
    const uint32_t local = index & 127u;
    const uint32_t lane = local & 31u;
    const uint32_t section = local >> 5u;
    const unsigned char *ql = block + half * 64u;
    const unsigned char *qh = block + 128u + half * 32u;
    const signed char *scales = reinterpret_cast<const signed char *>(block + 192u + half * 8u);
    const uint32_t packed_index = lane + ((section & 1u) ? 32u : 0u);
    const uint32_t nibble = section < 2u ? (ql[packed_index] & 0x0fu) : (ql[packed_index] >> 4u);
    const uint32_t high = ((uint32_t)qh[lane] >> (section * 2u)) & 0x03u;
    const int32_t quant = (int32_t)(nibble | (high << 4u)) - 32;
    const uint32_t scale_index = section * 2u + lane / 16u;
    const uint16_t d_bits = (uint16_t)block[208] | ((uint16_t)block[209] << 8u);
    return qx_half_to_float(d_bits) * (float)scales[scale_index] * (float)quant;
}

__global__ static void qx_q6_k_f32_kernel(
    const unsigned char *weights,
    const float *activation,
    uint32_t hidden,
    uint64_t row_bytes,
    uint32_t vocab,
    float *logits) {
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < vocab;
         row += blockDim.x * gridDim.x) {
        const unsigned char *row_weights = weights + (uint64_t)row * row_bytes;
        double sum = 0.0;
        for (uint32_t i = 0; i < hidden; ++i) {
            const unsigned char *block = row_weights + (uint64_t)(i >> 8u) * 210u;
            sum += (double)qx_q6_k_value(block, i & 255u) * (double)activation[i];
        }
        logits[row] = (float)sum;
    }
}

extern "C" int qx_cuda_final_head_is_available(
    qx_cuda_final_head_device *device,
    char *err,
    uint64_t err_len) {
    if (device != nullptr) std::memset(device, 0, sizeof(*device));
    int count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&count);
    if (count_status != cudaSuccess || count <= 0)
        return qx_fail(err, err_len, "CUDA final-head backend unavailable");

    cudaDeviceProp properties{};
    const cudaError_t property_status = cudaGetDeviceProperties(&properties, 0);
    if (property_status != cudaSuccess)
        return qx_cuda_fail(err, err_len, "CUDA device query failed", property_status);

    cudaFuncAttributes kernel_attributes{};
    const cudaError_t kernel_status =
        cudaFuncGetAttributes(&kernel_attributes, qx_q6_k_f32_kernel);
    if (kernel_status != cudaSuccess)
        return qx_cuda_fail(
            err,
            err_len,
            "CUDA final-head kernel is incompatible with the installed driver or device",
            kernel_status);
    if (device != nullptr) {
        std::snprintf(device->name, sizeof(device->name), "%s", properties.name);
        device->name[sizeof(device->name) - 1u] = '\0';
        device->compute_capability_major = (uint32_t)properties.major;
        device->compute_capability_minor = (uint32_t)properties.minor;
    }
    qx_clear_error(err, err_len);
    return 1;
}

extern "C" void qx_cuda_final_head_destroy(qx_cuda_final_head_context **context) {
    if (context == nullptr || *context == nullptr) return;
    qx_cuda_final_head_context *ctx = *context;
    if (ctx->device_logits != nullptr) (void)cudaFree(ctx->device_logits);
    if (ctx->device_activation != nullptr) (void)cudaFree(ctx->device_activation);
    if (ctx->device_weights != nullptr) (void)cudaFree(ctx->device_weights);
    std::free(ctx);
    *context = nullptr;
}

extern "C" int qx_cuda_final_head_create_q6_k_f32(
    const void *packed_weights,
    uint64_t packed_bytes,
    uint32_t hidden,
    uint32_t vocab,
    qx_cuda_final_head_context **out_context,
    char *err,
    uint64_t err_len) {
    if (out_context == nullptr)
        return qx_fail(err, err_len, "CUDA final-head output context is null");
    *out_context = nullptr;
    if (packed_weights == nullptr || hidden == 0u || vocab == 0u || hidden % 256u != 0u)
        return qx_fail(err, err_len, "invalid Q6_K final-head geometry");

    const uint64_t blocks_per_row = hidden / 256u;
    if (blocks_per_row > UINT64_MAX / 210u)
        return qx_fail(err, err_len, "Q6_K final-head size overflow");
    const uint64_t row_bytes = blocks_per_row * 210u;
    if ((uint64_t)vocab > UINT64_MAX / row_bytes || packed_bytes != (uint64_t)vocab * row_bytes)
        return qx_fail(err, err_len, "invalid Q6_K final-head packed byte size");
    if (packed_bytes > (uint64_t)SIZE_MAX ||
        (uint64_t)hidden * sizeof(float) > (uint64_t)SIZE_MAX ||
        (uint64_t)vocab * sizeof(float) > (uint64_t)SIZE_MAX)
        return qx_fail(err, err_len, "CUDA final-head buffer size overflow");
    if (!qx_cuda_final_head_is_available(nullptr, err, err_len)) return 0;

    qx_cuda_final_head_context *ctx =
        static_cast<qx_cuda_final_head_context *>(std::calloc(1u, sizeof(*ctx)));
    if (ctx == nullptr)
        return qx_fail(err, err_len, "CUDA final-head host allocation failed");
    ctx->hidden = hidden;
    ctx->vocab = vocab;
    ctx->packed_bytes = packed_bytes;

    cudaError_t status = cudaMalloc(reinterpret_cast<void **>(&ctx->device_weights), (size_t)packed_bytes);
    if (status != cudaSuccess) {
        qx_cuda_fail(err, err_len, "CUDA weight allocation failed", status);
        qx_cuda_final_head_destroy(&ctx);
        return 0;
    }
    ++ctx->counters.persistent_allocations;

    status = cudaMalloc(reinterpret_cast<void **>(&ctx->device_activation), (size_t)hidden * sizeof(float));
    if (status != cudaSuccess) {
        qx_cuda_fail(err, err_len, "CUDA activation allocation failed", status);
        qx_cuda_final_head_destroy(&ctx);
        return 0;
    }
    ++ctx->counters.persistent_allocations;

    status = cudaMalloc(reinterpret_cast<void **>(&ctx->device_logits), (size_t)vocab * sizeof(float));
    if (status != cudaSuccess) {
        qx_cuda_fail(err, err_len, "CUDA logits allocation failed", status);
        qx_cuda_final_head_destroy(&ctx);
        return 0;
    }
    ++ctx->counters.persistent_allocations;

    status = cudaMemcpy(ctx->device_weights, packed_weights, (size_t)packed_bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        qx_cuda_fail(err, err_len, "CUDA weight upload failed", status);
        qx_cuda_final_head_destroy(&ctx);
        return 0;
    }
    ctx->counters.resident_weight_bytes = packed_bytes;
    ctx->counters.weight_uploads = 1u;
    ctx->counters.weight_upload_bytes = packed_bytes;
    *out_context = ctx;
    qx_clear_error(err, err_len);
    return 1;
}

extern "C" int qx_cuda_final_head_compute_f32(
    qx_cuda_final_head_context *context,
    const float *normalized,
    uint32_t hidden,
    float *logits,
    uint32_t logits_capacity,
    char *err,
    uint64_t err_len) {
    if (context == nullptr || normalized == nullptr || logits == nullptr)
        return qx_fail(err, err_len, "invalid CUDA final-head compute argument");
    if (hidden != context->hidden)
        return qx_fail(err, err_len, "CUDA final-head hidden size mismatch");
    if (logits_capacity < context->vocab)
        return qx_fail(err, err_len, "CUDA final-head logits capacity too small");
    for (uint32_t i = 0; i < hidden; ++i) {
        if (!std::isfinite(normalized[i]))
            return qx_fail(err, err_len, "CUDA final-head input is non-finite");
    }

    const size_t activation_bytes = (size_t)hidden * sizeof(float);
    const size_t logits_bytes = (size_t)context->vocab * sizeof(float);
    float *staging = static_cast<float *>(std::malloc(logits_bytes));
    if (staging == nullptr)
        return qx_fail(err, err_len, "CUDA final-head host logits allocation failed");

    cudaError_t status = cudaMemcpy(context->device_activation, normalized, activation_bytes,
                                    cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        std::free(staging);
        return qx_cuda_fail(err, err_len, "CUDA activation upload failed", status);
    }
    context->counters.host_to_device_bytes += activation_bytes;

    constexpr uint32_t threads = 128u;
    uint64_t block_count = ((uint64_t)context->vocab + threads - 1u) / threads;
    if (block_count > 65535u) block_count = 65535u;
    const uint64_t row_bytes = (uint64_t)(hidden / 256u) * 210u;
    qx_q6_k_f32_kernel<<<(uint32_t)block_count, threads>>>(
        context->device_weights, context->device_activation, hidden, row_bytes,
        context->vocab, context->device_logits);
    ++context->counters.kernel_launches;

    status = cudaGetLastError();
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        std::free(staging);
        return qx_cuda_fail(err, err_len, "CUDA final-head kernel failed", status);
    }

    status = cudaMemcpy(staging, context->device_logits, logits_bytes, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        std::free(staging);
        return qx_cuda_fail(err, err_len, "CUDA logits download failed", status);
    }
    context->counters.device_to_host_bytes += logits_bytes;
    for (uint32_t i = 0; i < context->vocab; ++i) {
        if (!std::isfinite(staging[i])) {
            std::free(staging);
            return qx_fail(err, err_len, "CUDA final-head result is non-finite");
        }
    }
    std::memcpy(logits, staging, logits_bytes);
    std::free(staging);
    qx_clear_error(err, err_len);
    return 1;
}

extern "C" int qx_cuda_final_head_get_counters(
    const qx_cuda_final_head_context *context,
    qx_cuda_final_head_counters *out_counters,
    char *err,
    uint64_t err_len) {
    if (context == nullptr || out_counters == nullptr)
        return qx_fail(err, err_len, "invalid CUDA final-head counter argument");
    *out_counters = context->counters;
    qx_clear_error(err, err_len);
    return 1;
}
