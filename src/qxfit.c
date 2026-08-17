#include "qxfit.h"

#include <stdio.h>
#include <string.h>

#define QX_GIB_BYTES (1024.0 * 1024.0 * 1024.0)

qx_model_shape qx_shape_qwen3_4b(void) {
    qx_model_shape s = {"qwen3-4b", 36, 2560, 32, 8, 128, 151936};
    return s;
}

qx_model_shape qx_shape_qwen3_8b(void) {
    qx_model_shape s = {"qwen3-8b", 36, 4096, 32, 8, 128, 151936};
    return s;
}

qx_model_shape qx_shape_qwen3_14b(void) {
    qx_model_shape s = {"qwen3-14b", 40, 5120, 40, 8, 128, 151936};
    return s;
}

qx_model_shape qx_shape_qwen3_30b_a3b(void) {
    qx_model_shape s = {"qwen3-30b-a3b", 48, 2048, 32, 4, 128, 151936};
    return s;
}

const char *qx_kv_format_name(qx_kv_format fmt) {
    switch (fmt) {
        case QX_KV_FP16: return "fp16";
        case QX_KV_INT8: return "int8";
        case QX_KV_INT4: return "int4";
        default: return "unknown";
    }
}

int qx_parse_kv_format(const char *s, qx_kv_format *out) {
    if (!s || !out) return 0;
    if (strcmp(s, "fp16") == 0) { *out = QX_KV_FP16; return 1; }
    if (strcmp(s, "int8") == 0) { *out = QX_KV_INT8; return 1; }
    if (strcmp(s, "int4") == 0) { *out = QX_KV_INT4; return 1; }
    return 0;
}

double qx_kv_gib(qx_model_shape shape, uint32_t ctx_tokens, qx_kv_format fmt) {
    double bytes_per_value;
    switch (fmt) {
        case QX_KV_FP16: bytes_per_value = 2.0; break;
        case QX_KV_INT8: bytes_per_value = 1.0; break;
        case QX_KV_INT4: bytes_per_value = 0.5; break;
        default: bytes_per_value = 2.0; break;
    }

    /* K and V, all layers, grouped KV heads. */
    double bytes_per_token = 2.0 * (double)shape.layers * (double)shape.kv_heads *
                             (double)shape.head_dim * bytes_per_value;
    return (bytes_per_token * (double)ctx_tokens) / QX_GIB_BYTES;
}

uint32_t qx_max_ctx_for_kv_gib(qx_model_shape shape, double kv_budget_gib, qx_kv_format fmt) {
    double one_token = qx_kv_gib(shape, 1, fmt);
    if (one_token <= 0.0 || kv_budget_gib <= 0.0) return 0;
    double tokens = kv_budget_gib / one_token;
    if (tokens > 4294967295.0) return 4294967295u;
    return (uint32_t)tokens;
}

qx_fit_plan qx_plan_fit(qx_fit_request req) {
    qx_fit_plan p;
    memset(&p, 0, sizeof(p));

    p.usable_ram_gib = req.budget.ram_gib - req.budget.os_reserve_gib;
    p.usable_vram_gib = req.budget.vram_gib - req.budget.cuda_reserve_gib;
    if (p.usable_ram_gib < 0.0) p.usable_ram_gib = 0.0;
    if (p.usable_vram_gib < 0.0) p.usable_vram_gib = 0.0;

    p.kv_gib = qx_kv_gib(req.shape, req.ctx_tokens, req.kv_format);
    p.runtime_overhead_gib = p.kv_gib + req.budget.scratch_gib + req.budget.tokenizer_gib;
    p.total_active_gib = req.weight_gib + p.runtime_overhead_gib;

    double vram_after_runtime = p.usable_vram_gib - p.runtime_overhead_gib;
    if (vram_after_runtime < 0.0) vram_after_runtime = 0.0;

    p.suggested_vram_weights_gib = req.weight_gib < vram_after_runtime ? req.weight_gib : vram_after_runtime;
    p.suggested_ram_weights_gib = req.weight_gib - p.suggested_vram_weights_gib;

    double combined_usable = p.usable_ram_gib + p.usable_vram_gib;
    p.max_ctx_tokens_int8 = qx_max_ctx_for_kv_gib(req.shape, p.usable_vram_gib * 0.35, QX_KV_INT8);

    if (req.weight_gib <= 0.0) {
        p.feasible = 0;
        snprintf(p.reason, sizeof(p.reason), "invalid weight size");
    } else if (p.runtime_overhead_gib >= p.usable_vram_gib && req.budget.vram_gib > 0.0) {
        p.feasible = 0;
        snprintf(p.reason, sizeof(p.reason), "runtime overhead %.2f GiB exceeds usable VRAM %.2f GiB", p.runtime_overhead_gib, p.usable_vram_gib);
    } else if (p.total_active_gib > combined_usable) {
        p.feasible = 0;
        snprintf(p.reason, sizeof(p.reason), "active footprint %.2f GiB exceeds combined usable %.2f GiB", p.total_active_gib, combined_usable);
    } else if (p.suggested_ram_weights_gib > p.usable_ram_gib) {
        p.feasible = 0;
        snprintf(p.reason, sizeof(p.reason), "RAM-side weights %.2f GiB exceed usable RAM %.2f GiB", p.suggested_ram_weights_gib, p.usable_ram_gib);
    } else {
        p.feasible = 1;
        snprintf(p.reason, sizeof(p.reason), "fit: %.2f GiB weights in VRAM, %.2f GiB weights in RAM/mmap", p.suggested_vram_weights_gib, p.suggested_ram_weights_gib);
    }

    return p;
}
