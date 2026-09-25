#include "qx_cuda_final_head.h"

#include <stddef.h>
#include <string.h>

struct qx_cuda_final_head_context {
    int unused;
};

static int qx_cuda_final_head_unavailable(char *err, uint64_t err_len) {
    static const char message[] = "CUDA final-head backend unavailable";
    if (err != NULL && err_len > 0u) {
        size_t count = sizeof(message) - 1u;
        if ((uint64_t)count >= err_len) {
            count = (size_t)(err_len - 1u);
        }
        memcpy(err, message, count);
        err[count] = '\0';
    }
    return 0;
}

int qx_cuda_final_head_is_available(
    qx_cuda_final_head_device *device,
    char *err,
    uint64_t err_len) {
    if (device != NULL) {
        memset(device, 0, sizeof(*device));
    }
    return qx_cuda_final_head_unavailable(err, err_len);
}

int qx_cuda_final_head_create_q6_k_f32(
    const void *packed_weights,
    uint64_t packed_bytes,
    uint32_t hidden,
    uint32_t vocab,
    qx_cuda_final_head_context **out_context,
    char *err,
    uint64_t err_len) {
    (void)packed_weights;
    (void)packed_bytes;
    (void)hidden;
    (void)vocab;
    if (out_context != NULL) {
        *out_context = NULL;
    }
    return qx_cuda_final_head_unavailable(err, err_len);
}

int qx_cuda_final_head_compute_f32(
    qx_cuda_final_head_context *context,
    const float *normalized,
    uint32_t hidden,
    float *logits,
    uint32_t logits_capacity,
    char *err,
    uint64_t err_len) {
    (void)context;
    (void)normalized;
    (void)hidden;
    (void)logits;
    (void)logits_capacity;
    return qx_cuda_final_head_unavailable(err, err_len);
}

int qx_cuda_final_head_get_counters(
    const qx_cuda_final_head_context *context,
    qx_cuda_final_head_counters *out_counters,
    char *err,
    uint64_t err_len) {
    (void)context;
    if (out_counters != NULL) {
        memset(out_counters, 0, sizeof(*out_counters));
    }
    return qx_cuda_final_head_unavailable(err, err_len);
}

void qx_cuda_final_head_destroy(qx_cuda_final_head_context **context) {
    if (context != NULL) {
        *context = NULL;
    }
}
