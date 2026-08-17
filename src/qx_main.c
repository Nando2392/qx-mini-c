#include "qxfit.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s --model qwen3-8b --weight-gib 3.3 --ctx 4096 --kv int8 [--ram-gib 14 --vram-gib 5]\n"
        "models: qwen3-4b, qwen3-8b, qwen3-14b, qwen3-30b-a3b\n"
        "kv: fp16, int8, int4\n", argv0);
}

static int parse_shape(const char *name, qx_model_shape *out) {
    if (strcmp(name, "qwen3-4b") == 0) { *out = qx_shape_qwen3_4b(); return 1; }
    if (strcmp(name, "qwen3-8b") == 0) { *out = qx_shape_qwen3_8b(); return 1; }
    if (strcmp(name, "qwen3-14b") == 0) { *out = qx_shape_qwen3_14b(); return 1; }
    if (strcmp(name, "qwen3-30b-a3b") == 0) { *out = qx_shape_qwen3_30b_a3b(); return 1; }
    return 0;
}

int main(int argc, char **argv) {
    const char *model = "qwen3-8b";
    double weight_gib = 3.3;
    uint32_t ctx = 4096;
    qx_kv_format kv = QX_KV_INT8;

    qx_budget budget = {0};
    budget.ram_gib = 14.0;
    budget.vram_gib = 5.0;
    budget.os_reserve_gib = 3.0;
    budget.cuda_reserve_gib = 0.8;
    budget.scratch_gib = 0.35;
    budget.tokenizer_gib = 0.12;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model = argv[++i];
        else if (strcmp(argv[i], "--weight-gib") == 0 && i + 1 < argc) weight_gib = atof(argv[++i]);
        else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) ctx = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--kv") == 0 && i + 1 < argc) {
            if (!qx_parse_kv_format(argv[++i], &kv)) { usage(argv[0]); return 2; }
        } else if (strcmp(argv[i], "--ram-gib") == 0 && i + 1 < argc) budget.ram_gib = atof(argv[++i]);
        else if (strcmp(argv[i], "--vram-gib") == 0 && i + 1 < argc) budget.vram_gib = atof(argv[++i]);
        else { usage(argv[0]); return 2; }
    }

    qx_model_shape shape;
    if (!parse_shape(model, &shape)) {
        fprintf(stderr, "unknown model: %s\n", model);
        usage(argv[0]);
        return 2;
    }

    qx_fit_request req = {0};
    req.shape = shape;
    req.budget = budget;
    req.weight_gib = weight_gib;
    req.ctx_tokens = ctx;
    req.kv_format = kv;

    qx_fit_plan p = qx_plan_fit(req);

    printf("{\n");
    printf("  \"model\": \"%s\",\n", shape.name);
    printf("  \"weight_gib\": %.3f,\n", weight_gib);
    printf("  \"ctx_tokens\": %u,\n", ctx);
    printf("  \"kv_format\": \"%s\",\n", qx_kv_format_name(kv));
    printf("  \"kv_gib\": %.3f,\n", p.kv_gib);
    printf("  \"runtime_overhead_gib\": %.3f,\n", p.runtime_overhead_gib);
    printf("  \"total_active_gib\": %.3f,\n", p.total_active_gib);
    printf("  \"usable_ram_gib\": %.3f,\n", p.usable_ram_gib);
    printf("  \"usable_vram_gib\": %.3f,\n", p.usable_vram_gib);
    printf("  \"suggested_vram_weights_gib\": %.3f,\n", p.suggested_vram_weights_gib);
    printf("  \"suggested_ram_weights_gib\": %.3f,\n", p.suggested_ram_weights_gib);
    printf("  \"max_ctx_tokens_int8_vram35pct\": %u,\n", p.max_ctx_tokens_int8);
    printf("  \"feasible\": %s,\n", p.feasible ? "true" : "false");
    printf("  \"reason\": \"%s\"\n", p.reason);
    printf("}\n");

    return p.feasible ? 0 : 1;
}
