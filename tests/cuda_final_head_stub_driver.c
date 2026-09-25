#include "qx_cuda_final_head.h"

#include <stdio.h>
#include <string.h>

static int expected_error(const char *err) {
    return strcmp(err, "CUDA final-head backend unavailable") == 0;
}

int main(void) {
    char err[128] = {0};
    qx_cuda_final_head_device device;
    qx_cuda_final_head_context *ctx = (qx_cuda_final_head_context *)1;
    qx_cuda_final_head_counters counters;
    float input = 0.0f;
    float output = 0.0f;
    unsigned char packed[210] = {0};
    if (qx_cuda_final_head_is_available(&device, err, sizeof(err)) != 0 || !expected_error(err)) return 1;
    err[0] = '\0';
    if (qx_cuda_final_head_create_q6_k_f32(packed, sizeof(packed), 256, 1, &ctx, err, sizeof(err)) != 0 ||
        ctx != NULL || !expected_error(err)) return 2;
    err[0] = '\0';
    if (qx_cuda_final_head_compute_f32(NULL, &input, 256, &output, 1, err, sizeof(err)) != 0 || !expected_error(err)) return 3;
    err[0] = '\0';
    if (qx_cuda_final_head_get_counters(NULL, &counters, err, sizeof(err)) != 0 || !expected_error(err)) return 4;
    qx_cuda_final_head_destroy(NULL);
    qx_cuda_final_head_destroy(&ctx);
    puts("PASS stub unavailable contract");
    return 0;
}
