#ifndef QX_CUDA_FINAL_HEAD_H
#define QX_CUDA_FINAL_HEAD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qx_cuda_final_head_context qx_cuda_final_head_context;

typedef struct qx_cuda_final_head_device {
    char name[128];
    uint32_t compute_capability_major;
    uint32_t compute_capability_minor;
} qx_cuda_final_head_device;

typedef struct qx_cuda_final_head_counters {
    uint64_t resident_weight_bytes;
    uint64_t persistent_allocations;
    uint64_t weight_uploads;
    uint64_t weight_upload_bytes;
    uint64_t host_to_device_bytes;
    uint64_t device_to_host_bytes;
    uint64_t kernel_launches;
    uint64_t cpu_fallbacks;
} qx_cuda_final_head_counters;

int qx_cuda_final_head_is_available(
    qx_cuda_final_head_device *device,
    char *err,
    uint64_t err_len);

int qx_cuda_final_head_create_q6_k_f32(
    const void *packed_weights,
    uint64_t packed_bytes,
    uint32_t hidden,
    uint32_t vocab,
    qx_cuda_final_head_context **out_context,
    char *err,
    uint64_t err_len);

int qx_cuda_final_head_compute_f32(
    qx_cuda_final_head_context *context,
    const float *normalized,
    uint32_t hidden,
    float *logits,
    uint32_t logits_capacity,
    char *err,
    uint64_t err_len);

int qx_cuda_final_head_get_counters(
    const qx_cuda_final_head_context *context,
    qx_cuda_final_head_counters *out_counters,
    char *err,
    uint64_t err_len);

void qx_cuda_final_head_destroy(
    qx_cuda_final_head_context **context);

#ifdef __cplusplus
}
#endif

#endif
