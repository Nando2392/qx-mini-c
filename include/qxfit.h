#ifndef QXFIT_H
#define QXFIT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum qx_kv_format {
    QX_KV_FP16 = 0,
    QX_KV_INT8 = 1,
    QX_KV_INT4 = 2
} qx_kv_format;

typedef struct qx_model_shape {
    const char *name;
    uint32_t layers;
    uint32_t hidden;
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t vocab;
} qx_model_shape;

typedef struct qx_budget {
    double ram_gib;
    double vram_gib;
    double os_reserve_gib;
    double cuda_reserve_gib;
    double scratch_gib;
    double tokenizer_gib;
} qx_budget;

typedef struct qx_fit_request {
    qx_model_shape shape;
    qx_budget budget;
    double weight_gib;
    uint32_t ctx_tokens;
    qx_kv_format kv_format;
} qx_fit_request;

typedef struct qx_fit_plan {
    int feasible;
    double kv_gib;
    double runtime_overhead_gib;
    double total_active_gib;
    double usable_ram_gib;
    double usable_vram_gib;
    double suggested_vram_weights_gib;
    double suggested_ram_weights_gib;
    uint32_t max_ctx_tokens_int8;
    char reason[192];
} qx_fit_plan;

qx_model_shape qx_shape_qwen3_4b(void);
qx_model_shape qx_shape_qwen3_8b(void);
qx_model_shape qx_shape_qwen3_14b(void);
qx_model_shape qx_shape_qwen3_30b_a3b(void);

const char *qx_kv_format_name(qx_kv_format fmt);
int qx_parse_kv_format(const char *s, qx_kv_format *out);

double qx_kv_gib(qx_model_shape shape, uint32_t ctx_tokens, qx_kv_format fmt);
uint32_t qx_max_ctx_for_kv_gib(qx_model_shape shape, double kv_budget_gib, qx_kv_format fmt);
qx_fit_plan qx_plan_fit(qx_fit_request req);

#ifdef __cplusplus
}
#endif

#endif
