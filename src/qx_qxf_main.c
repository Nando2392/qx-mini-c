#include "qx_format.h"
#include "qx_gguf.h"
#include "qx_tokenizer.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage:\n"
        "  %s create --model qwen3-8b --quant q3q4mix --out models/qwen3-8b.qxf\n"
        "  %s inspect --in models/qwen3-8b.qxf\n"
        "  %s inspect-gguf --in models/source.gguf\n"
        "  %s create-from-gguf --in source.gguf --model qwen3-30b-a3b --quant q2 --out model.qxf\n"
        "  %s create-from-gguf-copy --in source.gguf --model qwen3-30b-a3b --quant q2 --out model.qxf\n"
        "  %s inspect-tensor --in model.qxf --name token_embd.weight [--io-backend buffered|mmap]\n"
        "  %s verify-qxf --in model.qxf [--max 16] [--io-backend buffered|mmap]\n"
        "  %s expert-index --in model.qxf\n"
        "  %s expert-quant-coverage --in model.qxf\n"
        "  %s expert-plan --in model.qxf [--vram-gib 2.0] [--ram-gib 6.5]\n"
        "  %s expert-slice --in model.qxf --layer 0 --expert 0\n"
        "  %s expert-load --in model.qxf --layer 0 --expert 0 --kind gate\n"
        "  %s cache-demo --in model.qxf --slots 2 --sequence 0:0:gate,0:0:gate\n"
        "  %s bench-expert-load --in model.qxf --iters 128 --kind gate\n"
        "  %s cache-run --in model.qxf --slots 2 --sequence 0:0:gate,0:0:gate\n"
        "  %s cache-run-expert --in model.qxf --slots 2 --sequence 0:0:gate,0:0:up,0:0:down\n"
        "  %s expert-cache-plan-complete --in model.qxf --vram-gib 2.0 --ram-gib 6.5 --top-k 8\n"
        "  %s runtime-plan --in model.qxf --ctx 4096 --kv int8 --vram-gib 4.2 --ram-gib 11.0 --hot-vram-gib 2.0 --hot-ram-gib 6.5 --top-k 8\n"
        "  %s kv-cache-probe --in model.qxf --ctx 4096 --kv int8 --token 42 --layer 0 --head 0\n"
        "  %s kv-cache-buffer-probe --in model.qxf --ctx 64 --kv int8 --token 7 --layer 1 --head 2 --seed 7\n"
        "  %s attention-cache-probe --in model.qxf --ctx 64 --kv int8 --layer 0 --tokens 2 --blocks 2 --seed 7\n"
        "  %s attention-softmax-probe --in model.qxf --ctx 64 --kv int8 --layer 0 --tokens 4 --blocks 2 --seed 7\n"
        "  %s attention-vector-probe --in model.qxf --ctx 64 --kv int8 --layer 0 --tokens 4 --dims 16 --seed 7\n"
        "  %s attention-multihead-probe --in model.qxf --ctx 64 --kv int8 --layer 0 --tokens 4 --heads 4 --dims 16 --seed 7\n"
        "  %s logits-probe --in model.qxf --activation 0.125 --top-n 5 --scan 64 --seed 7\n"
        "  %s sampler-probe --in model.qxf --activation 0.125 --top-k 5 --scan 64 --temperature 0.7 --seed 7\n"
        "  %s tokenizer-export --gguf model.gguf --out model.tokens.tsv\n"
        "  qxqxf tokenizer-inspect --tokenizer model.qxt\n"
        "  qxqxf tokenizer-encode --tokenizer model.qxt --text-file prompt.txt [--parse-special]\n"
        "  qxqxf tokenizer-decode --tokenizer model.qxt --ids 9707,0 [--special]\n"
        "  qxqxf chat-template-render --message system:system.txt --message user:prompt.txt [--add-generation-prompt]\n"
        "  qxqxf prompt-state-loop-probe --in model.qxf --tokenizer model.qxt --text-file prompt.txt --generate 2 --layers 48 --ctx 16 --kv int8 --activation f32 --io-backend buffered|mmap --scratch-policy ephemeral|persistent --kernel-policy baseline|fused --thread-policy serial --threads 1 --temperature 0 --seed 7 --full-moe --final-head [--bench] [--dequant-profile] [--parse-special] [--top-n 5] [--kv-snapshot-out file]\n"
        "  %s tokenizer-probe --in model.qxf --token-id 42\n"
        "  %s generate-probe --in model.qxf --tokens model.tokens.tsv --prompt-token 42 --steps 3 --top-k 5 --scan 64 --temperature 0 --seed 7\n"
        "  %s residual-vector-probe --in model.qxf --token-id 42 --norm blk.0.attn_norm.weight --dims 64 --seed 7\n"
        "  %s projection-matvec-probe --in model.qxf --layer 0 --token-id 42 --rows 4 --dims 64 --kv int8 --seed 7\n"
        "  qxqxf q8-k-activation-probe --values 256 [--inject zero|nan|inf]\n"
        "  %s state-loop-probe --in model.qxf --tokens model.tokens.tsv --prompt-token 42 --steps 1 --layers 48 --ctx 16 --kv int8 --activation f32 --temperature 0 --seed 7 --full-moe [--start-layer N --residual-in layer-N.f32] [--final-head --top-n 5] [--dump-residuals dir] [--kv-snapshot-out file | --kv-snapshot-in file] [--bench]\n"
        "  %s rope-gqa-golden-probe --tokens 2 --q-heads-run 9 --seed 7\n"
        "  qxqxf real-qkv-golden-probe --in model.qxf --layer 0 --token-a 42 --token-b 43 --q-heads-run 9 --seed 7\n"
        "  qxqxf attention-stage-probe --in model.qxf --layer 1 --layer-in layer-1.f32 --out-dir sidecars --activation q8_k_compat --kv f16\n"
        "  qxqxf moe-stage-probe --in model.qxf --layer 0 --ffn-inp ffn_inp-0.f32 --out-dir sidecars\n"
        "  qxqxf final-head-probe --in model.qxf --residual l_out-47.f32 --out-dir sidecars --activation q8_k_compat --top-n 5\n"
        "  qxqxf expert-q8-k-dot-probe --in model.qxf --name blk.0.ffn_gate_exps.weight --expert 49 --row 0 --activation ffn_norm-0.f32\n"
        "  %s token-embedding --in model.qxf --token-id 42\n"
        "  %s rmsnorm-probe --in model.qxf --token-id 42 --norm blk.0.attn_norm.weight --seed 7\n"
        "  %s attention-probe --in model.qxf --layer 0 --blocks 2 --seed 7\n"
        "  %s forward-schedule --in model.qxf --token-id 42 --top-k 8\n"
        "  %s quant-block --in model.qxf --name token_embd.weight --block 0\n"
        "  %s matvec-stub --in model.qxf --name token_embd.weight --rows 2\n"
        "  %s decode-block --in model.qxf --name blk.0.ffn_gate_exps.weight --block 0\n"
        "  %s block-dot --in model.qxf --name blk.0.ffn_gate_exps.weight --block 0 --seed 7\n"
        "  %s matvec-row --in model.qxf --name blk.0.ffn_gate_exps.weight --start-block 0 --blocks 8 --seed 7\n"
        "  %s expert-row --in model.qxf --layer 0 --expert 0 --kind gate --start-block 0 --blocks 8 --seed 7\n"
        "  %s expert-forward-probe --in model.qxf --layer 0 --expert 0 --start-block 0 --blocks 8 --seed 7\n"
        "  %s router-topk-probe --in model.qxf --layer 0 --top-k 8 --blocks 8 --seed 7\n"
        "  %s layer-forward-probe --in model.qxf --layer 0 --top-k 8 --blocks 8 --seed 7\n"
        "  %s moe-forward-probe --in model.qxf --layers 2 --top-k 8 --blocks 8 --seed 7\n"
        "  %s token-forward-probe --in model.qxf --token-id 42 --layers 2 --top-k 8 --blocks 8 --seed 7\n"
        "  %s route-trace --layers 48 --experts 128 --top-k 8 --tokens 4 --seed 7\n"
        "models: qwen3-4b, qwen3-8b, qwen3-14b, qwen3-30b-a3b\n"
        "quant: f16, q4, q3, q3q4mix, q2\n", argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0);
}

static uint32_t qx_lcg_next(uint32_t *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static uint32_t qx_trace_expert(uint32_t seed, uint32_t layer, uint32_t token, uint32_t k, uint32_t experts, uint32_t reuse_pct) {
    uint32_t reuse = reuse_pct > 100u ? 100u : reuse_pct;
    uint32_t state = seed ? seed : 1u;
    state ^= layer * 747796405u;
    if (reuse < 100u) state ^= token * 2891336453u;
    uint32_t base = qx_lcg_next(&state) % experts;
    uint32_t fresh = (uint32_t)(((uint64_t)qx_lcg_next(&state) * 100ull) >> 32) >= reuse;
    uint32_t token_term = fresh ? token * 31u : 0u;
    return (base + k * 17u + token_term + layer * 7u) % experts;
}

static int route_trace(uint32_t layers, uint32_t experts, uint32_t top_k, uint32_t tokens, uint32_t seed, uint32_t reuse_pct) {
    if (layers == 0 || experts == 0 || top_k == 0 || top_k > experts || tokens == 0) return 0;
    uint64_t requests = (uint64_t)layers * tokens * top_k * 3u;
    printf("{\n");
    printf("  \"layers\": %u,\n", layers);
    printf("  \"experts\": %u,\n", experts);
    printf("  \"top_k\": %u,\n", top_k);
    printf("  \"tokens\": %u,\n", tokens);
    printf("  \"reuse_pct\": %u,\n", reuse_pct > 100u ? 100u : reuse_pct);
    printf("  \"requests\": %llu,\n", (unsigned long long)requests);
    printf("  \"sequence\": \"");
    uint64_t emitted = 0;
    const char *kinds[3] = {"gate", "up", "down"};
    for (uint32_t tok = 0; tok < tokens; tok++) {
        for (uint32_t layer = 0; layer < layers; layer++) {
            for (uint32_t k = 0; k < top_k; k++) {
                uint32_t expert = qx_trace_expert(seed, layer, tok, k, experts, reuse_pct);
                for (uint32_t kind = 0; kind < 3; kind++) {
                    if (emitted++) printf(",");
                    printf("%u:%u:%s", layer, expert, kinds[kind]);
                }
            }
        }
    }
    printf("\"\n}\n");
    return 1;
}

static char *read_text_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
#if defined(_WIN32)
    if (_fseeki64(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    __int64 n = _ftelli64(f);
    if (n < 0) { fclose(f); return NULL; }
    if (_fseeki64(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
#else
    if (fseeko(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    off_t n = ftello(f);
    if (n < 0) { fclose(f); return NULL; }
    if (fseeko(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
#endif
    char *buf = (char *)malloc((size_t)n + 1u);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); return NULL; }
    buf[(size_t)n] = 0;
    fclose(f);
    return buf;
}

static int validate_gguf_against_manifest(const qx_gguf_summary *g, const qx_model_manifest *m, char *err, size_t err_len) {
    if (m->model_type == QX_MODEL_QWEN3_MOE && g->qwen3_expert_count != 0 && g->qwen3_expert_count != m->experts) {
        snprintf(err, err_len, "expert_count mismatch: gguf=%llu manifest=%u", (unsigned long long)g->qwen3_expert_count, m->experts);
        return 0;
    }
    if (g->qwen3_block_count != 0 && g->qwen3_block_count != m->layers) {
        snprintf(err, err_len, "layer count mismatch: gguf=%llu manifest=%u", (unsigned long long)g->qwen3_block_count, m->layers);
        return 0;
    }
    if (g->qwen3_embedding_length != 0 && g->qwen3_embedding_length != m->hidden) {
        snprintf(err, err_len, "hidden mismatch: gguf=%llu manifest=%u", (unsigned long long)g->qwen3_embedding_length, m->hidden);
        return 0;
    }
    if (g->qwen3_attention_head_count != 0 && g->qwen3_attention_head_count != m->q_heads) {
        snprintf(err, err_len, "q head mismatch: gguf=%llu manifest=%u", (unsigned long long)g->qwen3_attention_head_count, m->q_heads);
        return 0;
    }
    if (g->qwen3_attention_head_count_kv != 0 && g->qwen3_attention_head_count_kv != m->kv_heads) {
        snprintf(err, err_len, "kv head mismatch: gguf=%llu manifest=%u", (unsigned long long)g->qwen3_attention_head_count_kv, m->kv_heads);
        return 0;
    }
    return 1;
}

static int qx_cli_read_prompt(const char *path, unsigned char **data, uint32_t *length, char *err, size_t err_len) {
    FILE *file = fopen(path, "rb");
    if (!file) { snprintf(err, err_len, "cannot open text file"); return 0; }
    unsigned char *buffer = (unsigned char *)malloc(QX_TOKENIZER_MAX_INPUT + 1u);
    if (!buffer) { fclose(file); snprintf(err, err_len, "out of memory"); return 0; }
    size_t count = fread(buffer, 1, QX_TOKENIZER_MAX_INPUT + 1u, file);
    if (ferror(file)) { free(buffer); fclose(file); snprintf(err, err_len, "text file read failed"); return 0; }
    int trailing = fgetc(file);
    fclose(file);
    if (count > QX_TOKENIZER_MAX_INPUT || trailing != EOF) {
        free(buffer); snprintf(err, err_len, "input exceeds 4096-byte limit"); return 0;
    }
    *data = buffer;
    *length = (uint32_t)count;
    return 1;
}

static int qx_cli_parse_ids(const char *text, uint32_t *ids, uint32_t capacity, uint32_t *count, char *err, size_t err_len) {
    if (!text || !*text) { snprintf(err, err_len, "empty token id list"); return 0; }
    const char *cursor = text;
    *count = 0u;
    while (*cursor) {
        if (*count >= capacity) { snprintf(err, err_len, "too many token ids"); return 0; }
        errno = 0;
        char *end = NULL;
        unsigned long value = strtoul(cursor, &end, 10);
        if (errno != 0 || end == cursor || value > UINT32_MAX || (*end != '\0' && *end != ',')) {
            snprintf(err, err_len, "invalid token id list"); return 0;
        }
        ids[(*count)++] = (uint32_t)value;
        if (*end == '\0') break;
        cursor = end + 1;
        if (*cursor == '\0') { snprintf(err, err_len, "invalid token id list"); return 0; }
    }
    return 1;
}

static int qx_cli_parse_u32_arg(const char *name, const char *text, uint32_t *out, char *err, size_t err_len) {
    if (!text || !*text) { snprintf(err, err_len, "invalid %s", name); return 0; }
    errno = 0;
    char *end = NULL;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno != 0 || !end || end == text || *end != '\0' || parsed > UINT32_MAX) {
        snprintf(err, err_len, "invalid %s", name); return 0;
    }
    *out = (uint32_t)parsed;
    return 1;
}

static void qx_cli_json_string(const unsigned char *text, uint32_t length) {
    putchar('"');
    for (uint32_t i = 0; i < length; ++i) {
        unsigned char c = text[i];
        if (c == '"' || c == '\\') { putchar('\\'); putchar(c); }
        else if (c == 8u) fputs("\\b", stdout);
        else if (c == 12u) fputs("\\f", stdout);
        else if (c == 10u) fputs("\\n", stdout);
        else if (c == 13u) { putchar(92); putchar('r'); }
        else if (c == 9u) fputs("\\t", stdout);
        else if (c < 0x20u) printf("\\u%04x", (unsigned)c);
        else putchar(c);
    }
    putchar('"');
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 2; }

    if (strcmp(argv[1], "q8-k-activation-probe") == 0) {
        uint32_t values = 0u;
        int values_seen = 0;
        const char *inject = "none";
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "--values") == 0 && i + 1 < argc) {
                errno = 0;
                char *end = NULL;
                unsigned long long parsed = strtoull(argv[++i], &end, 10);
                if (errno != 0 || !end || *end != '\0' || parsed > UINT32_MAX) {
                    fprintf(stderr, "q8-k-activation-probe failed: invalid --values\n"); return 2;
                }
                values = (uint32_t)parsed;
                values_seen = 1;
            }
            else if (strcmp(argv[i], "--inject") == 0 && i + 1 < argc) inject = argv[++i];
            else { usage(argv[0]); return 2; }
        }
        if (!values_seen) { fprintf(stderr, "q8-k-activation-probe failed: invalid --values\n"); return 2; }
        char err[256];
        if (!qx_dump_q8_k_activation_probe_summary(values, inject, stdout, err, sizeof(err))) {
            fprintf(stderr, "q8-k-activation-probe failed: %s\n", err); return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "tokenizer-inspect") == 0) {
        const char *tokenizer_path = NULL;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) tokenizer_path = argv[++i];
            else { usage(argv[0]); return 2; }
        }
        if (!tokenizer_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_tokenizer_dump_summary(tokenizer_path, stdout, err, sizeof(err))) {
            fprintf(stderr, "tokenizer-inspect failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "tokenizer-encode") == 0) {
        const char *tokenizer_path = NULL;
        const char *text_path = NULL;
        int parse_special = 0;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) tokenizer_path = argv[++i];
            else if (strcmp(argv[i], "--text-file") == 0 && i + 1 < argc) text_path = argv[++i];
            else if (strcmp(argv[i], "--parse-special") == 0) parse_special = 1;
            else { usage(argv[0]); return 2; }
        }
        if (!tokenizer_path || !text_path) { usage(argv[0]); return 2; }
        char err[256];
        unsigned char *input = NULL;
        uint32_t input_length = 0u;
        if (!qx_cli_read_prompt(text_path, &input, &input_length, err, sizeof(err))) {
            fprintf(stderr, "tokenizer-encode failed: %s\n", err); return 1;
        }
        qx_tokenizer tokenizer;
        if (!qx_tokenizer_load(tokenizer_path, &tokenizer, err, sizeof(err))) {
            free(input); fprintf(stderr, "tokenizer-encode failed: %s\n", err); return 1;
        }
        uint32_t ids[QX_TOKENIZER_MAX_INPUT + 2u];
        uint32_t count = 0u;
        if (!qx_tokenizer_encode(&tokenizer, input, input_length, parse_special, ids, QX_TOKENIZER_MAX_INPUT + 2u, &count, err, sizeof(err))) {
            qx_tokenizer_free(&tokenizer); free(input); fprintf(stderr, "tokenizer-encode failed: %s\n", err); return 1;
        }
        printf("{\n  \"input_bytes\": %u,\n  \"parse_special\": %s,\n  \"token_count\": %u,\n  \"token_ids\": [", input_length, parse_special ? "true" : "false", count);
        for (uint32_t i = 0; i < count; ++i) printf("%s%u", i ? ", " : "", ids[i]);
        printf("]\n}\n");
        qx_tokenizer_free(&tokenizer);
        free(input);
        return 0;
    }

    if (strcmp(argv[1], "chat-template-render") == 0) {
        qx_chat_message messages[QX_CHAT_MAX_MESSAGES];
        unsigned char *contents[QX_CHAT_MAX_MESSAGES] = {0};
        uint32_t message_count = 0u;
        int add_generation_prompt = 0;
        char err[256];
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "--message") == 0 && i + 1 < argc) {
                const char *spec = argv[++i];
                const char *separator = strchr(spec, ':');
                if (!separator || separator == spec || message_count == QX_CHAT_MAX_MESSAGES) {
                    fprintf(stderr, "chat-template-render failed: invalid --message\n"); goto chat_usage_fail;
                }
                size_t role_length = (size_t)(separator - spec);
                const char *role = NULL;
                if (role_length == 6u && memcmp(spec, "system", 6u) == 0) role = "system";
                else if (role_length == 4u && memcmp(spec, "user", 4u) == 0) role = "user";
                else if (role_length == 9u && memcmp(spec, "assistant", 9u) == 0) role = "assistant";
                else { fprintf(stderr, "chat-template-render failed: unsupported chat role\n"); goto chat_fail; }
                uint32_t content_length = 0u;
                if (!qx_cli_read_prompt(separator + 1, &contents[message_count], &content_length, err, sizeof(err))) {
                    fprintf(stderr, "chat-template-render failed: %s\n", err); goto chat_fail;
                }
                messages[message_count].role = role;
                messages[message_count].content = contents[message_count];
                messages[message_count].content_length = content_length;
                ++message_count;
            } else if (strcmp(argv[i], "--add-generation-prompt") == 0) {
                add_generation_prompt = 1;
            } else { usage(argv[0]); goto chat_usage_fail; }
        }
        if (message_count == 0u) { usage(argv[0]); goto chat_usage_fail; }
        unsigned char *output = (unsigned char *)malloc(QX_CHAT_MAX_OUTPUT);
        uint32_t output_length = 0u;
        if (!output) { fprintf(stderr, "chat-template-render failed: out of memory\n"); goto chat_fail; }
        if (!qx_tokenizer_render_qwen3_chat(messages, message_count, add_generation_prompt, output, QX_CHAT_MAX_OUTPUT, &output_length, err, sizeof(err))) {
            free(output); fprintf(stderr, "chat-template-render failed: %s\n", err); goto chat_fail;
        }
        printf("{\n  \"template\": \"qwen3-chatml-subset-v1\",\n  \"message_count\": %u,\n  \"add_generation_prompt\": %s,\n  \"utf8_bytes\": %u,\n  \"text\": ", message_count, add_generation_prompt ? "true" : "false", output_length);
        qx_cli_json_string(output, output_length);
        printf("\n}\n");
        free(output);
        for (uint32_t i = 0u; i < message_count; ++i) free(contents[i]);
        return 0;
chat_usage_fail:
        for (uint32_t i = 0u; i < message_count; ++i) free(contents[i]);
        return 2;
chat_fail:
        for (uint32_t i = 0u; i < message_count; ++i) free(contents[i]);
        return 1;
    }

    if (strcmp(argv[1], "tokenizer-decode") == 0) {
        const char *tokenizer_path = NULL;
        const char *ids_text = NULL;
        int special = 0;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) tokenizer_path = argv[++i];
            else if (strcmp(argv[i], "--ids") == 0 && i + 1 < argc) ids_text = argv[++i];
            else if (strcmp(argv[i], "--special") == 0) special = 1;
            else { usage(argv[0]); return 2; }
        }
        if (!tokenizer_path || !ids_text) { usage(argv[0]); return 2; }
        char err[256];
        uint32_t ids[QX_TOKENIZER_MAX_INPUT + 2u];
        uint32_t count = 0u;
        if (!qx_cli_parse_ids(ids_text, ids, QX_TOKENIZER_MAX_INPUT + 2u, &count, err, sizeof(err))) {
            fprintf(stderr, "tokenizer-decode failed: %s\n", err); return 1;
        }
        qx_tokenizer tokenizer;
        if (!qx_tokenizer_load(tokenizer_path, &tokenizer, err, sizeof(err))) {
            fprintf(stderr, "tokenizer-decode failed: %s\n", err); return 1;
        }
        unsigned char *output = (unsigned char *)malloc(1u << 20);
        uint32_t output_length = 0u;
        if (!output || !qx_tokenizer_decode(&tokenizer, ids, count, special, output, 1u << 20, &output_length, err, sizeof(err))) {
            free(output); qx_tokenizer_free(&tokenizer); fprintf(stderr, "tokenizer-decode failed: %s\n", output ? err : "out of memory"); return 1;
        }
        printf("{\n  \"token_count\": %u,\n  \"utf8_bytes\": %u,\n  \"text\": ", count, output_length);
        qx_cli_json_string(output, output_length);
        printf("\n}\n");
        free(output);
        qx_tokenizer_free(&tokenizer);
        return 0;
    }

    if (strcmp(argv[1], "prompt-state-loop-probe") == 0) {
        const char *in_path = NULL;
        const char *tokenizer_path = NULL;
        const char *text_path = NULL;
        const char *kv_format = "int8";
        const char *activation_format = "f32";
        const char *io_backend = "buffered";
        const char *scratch_policy = "ephemeral";
        const char *kernel_policy = "baseline";
        const char *thread_policy = "serial";
        const char *simd_policy = "scalar";
        const char *expert_cache_policy = "none";
        const char *cuda_policy = "none";
        const char *kv_snapshot_out_path = NULL;
        uint32_t generation_steps = 0u;
        uint32_t layers = 48u;
        uint32_t ctx = 16u;
        uint32_t top_n = 5u;
        uint32_t seed = 7u;
        uint32_t threads = 1u;
        double temperature = 0.0;
        int parse_special = 0;
        int full_moe = 0;
        int final_head = 0;
        int bench = 0;
        int dequant_profile = 0;
        char err[256];
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) tokenizer_path = argv[++i];
            else if (strcmp(argv[i], "--text-file") == 0 && i + 1 < argc) text_path = argv[++i];
            else if (strcmp(argv[i], "--generate") == 0 && i + 1 < argc) generation_steps = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) layers = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) ctx = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--kv") == 0 && i + 1 < argc) kv_format = argv[++i];
            else if (strcmp(argv[i], "--activation") == 0 && i + 1 < argc) activation_format = argv[++i];
            else if (strcmp(argv[i], "--io-backend") == 0 && i + 1 < argc) io_backend = argv[++i];
            else if (strcmp(argv[i], "--scratch-policy") == 0 && i + 1 < argc) scratch_policy = argv[++i];
            else if (strcmp(argv[i], "--kernel-policy") == 0 && i + 1 < argc) kernel_policy = argv[++i];
            else if (strcmp(argv[i], "--thread-policy") == 0 && i + 1 < argc) thread_policy = argv[++i];
            else if (strcmp(argv[i], "--simd-policy") == 0 && i + 1 < argc) simd_policy = argv[++i];
            else if (strcmp(argv[i], "--expert-cache-policy") == 0 && i + 1 < argc) expert_cache_policy = argv[++i];
            else if (strcmp(argv[i], "--cuda-policy") == 0 && i + 1 < argc) cuda_policy = argv[++i];
            else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
                if (!qx_cli_parse_u32_arg("--threads", argv[++i], &threads, err, sizeof(err))) {
                    fprintf(stderr, "prompt-state-loop-probe failed: %s\n", err); return 2;
                }
            }
            else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) temperature = strtod(argv[++i], NULL);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--top-n") == 0 && i + 1 < argc) top_n = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--kv-snapshot-out") == 0 && i + 1 < argc) kv_snapshot_out_path = argv[++i];
            else if (strcmp(argv[i], "--parse-special") == 0) parse_special = 1;
            else if (strcmp(argv[i], "--full-moe") == 0) full_moe = 1;
            else if (strcmp(argv[i], "--final-head") == 0) final_head = 1;
            else if (strcmp(argv[i], "--bench") == 0) bench = 1;
            else if (strcmp(argv[i], "--dequant-profile") == 0) dequant_profile = 1;
            else { usage(argv[0]); return 2; }
        }
        if (!in_path || !tokenizer_path || !text_path || !full_moe || !final_head) { usage(argv[0]); return 2; }
        int thread_pool_policy = strcmp(thread_policy, "pool") == 0;
        if (!thread_pool_policy && strcmp(thread_policy, "serial") != 0) {
            fprintf(stderr, "prompt-state-loop-probe failed: unsupported thread policy\n"); return 2;
        }
        if (!thread_pool_policy && threads != 1u) {
            fprintf(stderr, "prompt-state-loop-probe failed: serial thread policy requires --threads 1\n"); return 2;
        }
        if (thread_pool_policy && threads < 2u) {
            fprintf(stderr, "prompt-state-loop-probe failed: thread pool policy requires --threads >= 2\n"); return 2;
        }
        if (thread_pool_policy && threads > 64u) {
            fprintf(stderr, "prompt-state-loop-probe failed: thread pool policy supports at most 64 threads\n"); return 2;
        }
        if (thread_pool_policy && strcmp(activation_format, "f32") != 0) {
            fprintf(stderr, "prompt-state-loop-probe failed: thread pool policy currently requires F32 activation\n"); return 2;
        }
        int avx2_fma_policy = strcmp(simd_policy, "avx2-fma") == 0;
        if (!avx2_fma_policy && strcmp(simd_policy, "scalar") != 0) {
            fprintf(stderr, "prompt-state-loop-probe failed: unsupported simd policy\n"); return 2;
        }
        if (avx2_fma_policy && strcmp(kernel_policy, "fused") != 0) {
            fprintf(stderr, "prompt-state-loop-probe failed: avx2-fma simd policy requires --kernel-policy fused\n"); return 2;
        }
        if (avx2_fma_policy && strcmp(activation_format, "f32") != 0) {
            fprintf(stderr, "prompt-state-loop-probe failed: avx2-fma simd policy requires F32 activation\n"); return 2;
        }
        if (avx2_fma_policy && thread_pool_policy) {
            fprintf(stderr, "prompt-state-loop-probe failed: avx2-fma simd policy currently requires serial thread policy\n"); return 2;
        }
        if (strcmp(expert_cache_policy, "none") != 0) {
            fprintf(stderr, "prompt-state-loop-probe failed: unsupported expert cache policy\n"); return 2;
        }
        if (strcmp(cuda_policy, "none") != 0) {
            fprintf(stderr, "prompt-state-loop-probe failed: unsupported CUDA policy\n"); return 2;
        }
        if (!qx_set_io_backend(io_backend, err, sizeof(err))) {
            fprintf(stderr, "prompt-state-loop-probe failed: %s\n", err); return 2;
        }
        unsigned char *input = NULL;
        uint32_t input_length = 0u;
        if (!qx_cli_read_prompt(text_path, &input, &input_length, err, sizeof(err))) {
            fprintf(stderr, "prompt-state-loop-probe failed: %s\n", err); return 1;
        }
        qx_tokenizer tokenizer;
        if (!qx_tokenizer_load(tokenizer_path, &tokenizer, err, sizeof(err))) {
            free(input); fprintf(stderr, "prompt-state-loop-probe failed: %s\n", err); return 1;
        }
        if (tokenizer.vocab_count != 151936u) {
            qx_tokenizer_free(&tokenizer); free(input); fprintf(stderr, "prompt-state-loop-probe failed: tokenizer vocabulary does not match Qwen3-30B-A3B\n"); return 1;
        }
        if (tokenizer.payload_checksum != 6140965799433681264ull) {
            qx_tokenizer_free(&tokenizer); free(input); fprintf(stderr, "prompt-state-loop-probe failed: tokenizer fingerprint does not match Qwen3-30B-A3B\n"); return 1;
        }
        uint32_t ids[QX_TOKENIZER_MAX_INPUT + 2u];
        uint32_t count = 0u;
        if (!qx_tokenizer_encode(&tokenizer, input, input_length, parse_special, ids, QX_TOKENIZER_MAX_INPUT + 2u, &count, err, sizeof(err)) || count == 0u) {
            qx_tokenizer_free(&tokenizer); free(input);
            fprintf(stderr, "prompt-state-loop-probe failed: %s\n", count == 0u ? "prompt produced no tokens" : err); return 1;
        }
        qx_tokenizer_free(&tokenizer);
        free(input);
        if (!qx_dump_prompt_state_loop_probe_summary(in_path, NULL, ids, count, generation_steps, layers, ctx, kv_format, activation_format, scratch_policy, kernel_policy, thread_policy, threads, simd_policy, expert_cache_policy, cuda_policy, dequant_profile,
                1, 1, 1, 1, 1, 1, 1, 1, 1, full_moe, final_head, bench, 2048u, NULL, 8u, 151936u,
                top_n, temperature, seed, NULL, 0u, NULL, kv_snapshot_out_path, NULL, stdout, err, sizeof(err))) {
            fprintf(stderr, "prompt-state-loop-probe failed: %s\n", err); return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "create") == 0) {
        const char *model = "qwen3-8b";
        const char *quant_s = "q3q4mix";
        const char *out_path = NULL;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model = argv[++i];
            else if (strcmp(argv[i], "--quant") == 0 && i + 1 < argc) quant_s = argv[++i];
            else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
            else { usage(argv[0]); return 2; }
        }
        if (!out_path) { usage(argv[0]); return 2; }

        qx_quant_type quant;
        if (!qx_parse_quant_type(quant_s, &quant)) {
            fprintf(stderr, "unknown quant: %s\n", quant_s);
            return 2;
        }
        qx_model_manifest manifest;
        if (!qx_manifest_for_model(model, quant, &manifest)) {
            fprintf(stderr, "unknown model: %s\n", model);
            return 2;
        }
        char err[256];
        if (!qx_write_metadata_only(out_path, &manifest, err, sizeof(err))) {
            fprintf(stderr, "create failed: %s\n", err);
            return 1;
        }
        printf("wrote %s\n", out_path);
        return 0;
    }

    if (strcmp(argv[1], "inspect") == 0) {
        const char *in_path = NULL;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_summary(in_path, stdout, err, sizeof(err))) {
            fprintf(stderr, "inspect failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "inspect-gguf") == 0) {
        const char *in_path = NULL;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_gguf_dump_summary(in_path, stdout, err, sizeof(err))) {
            fprintf(stderr, "inspect-gguf failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "create-from-gguf") == 0 || strcmp(argv[1], "create-from-gguf-copy") == 0) {
        int copy_tensors = strcmp(argv[1], "create-from-gguf-copy") == 0;
        const char *in_path = NULL;
        const char *out_path = NULL;
        const char *model = "qwen3-30b-a3b";
        const char *quant_s = "q2";
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
            else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model = argv[++i];
            else if (strcmp(argv[i], "--quant") == 0 && i + 1 < argc) quant_s = argv[++i];
            else { usage(argv[0]); return 2; }
        }
        if (!in_path || !out_path) { usage(argv[0]); return 2; }

        qx_quant_type quant;
        qx_model_manifest manifest;
        if (!qx_parse_quant_type(quant_s, &quant)) { fprintf(stderr, "unknown quant: %s\n", quant_s); return 2; }
        if (!qx_manifest_for_model(model, quant, &manifest)) { fprintf(stderr, "unknown model: %s\n", model); return 2; }

        qx_gguf_summary gguf;
        char err[256];
        if (!qx_gguf_inspect(in_path, &gguf, err, sizeof(err))) {
            fprintf(stderr, "inspect source GGUF failed: %s\n", err);
            return 1;
        }
        if (!validate_gguf_against_manifest(&gguf, &manifest, err, sizeof(err))) {
            fprintf(stderr, "GGUF validation failed: %s\n", err);
            return 1;
        }
        if (copy_tensors) {
            qx_gguf_tensor_table table;
            if (!qx_gguf_load_tensor_table(in_path, &gguf, &table, err, sizeof(err))) {
                fprintf(stderr, "load GGUF tensor table failed: %s\n", err);
                return 1;
            }
            int ok = qx_write_tensor_copy_from_gguf(out_path, in_path, &manifest, &table, err, sizeof(err));
            qx_gguf_free_tensor_table(&table);
            if (!ok) { fprintf(stderr, "tensor-copy create failed: %s\n", err); return 1; }
        } else {
            if (!qx_write_metadata_only(out_path, &manifest, err, sizeof(err))) {
                fprintf(stderr, "create failed: %s\n", err);
                return 1;
            }
        }
        printf("wrote %s from %s%s\n", out_path, in_path, copy_tensors ? " with tensor copy" : "");
        return 0;
    }

    if (strcmp(argv[1], "inspect-tensor") == 0) {
        const char *in_path = NULL;
        const char *name = NULL;
        const char *io_backend = "buffered";
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--name") == 0 && i + 1 < argc) name = argv[++i];
            else if (strcmp(argv[i], "--io-backend") == 0 && i + 1 < argc) io_backend = argv[++i];
            else { usage(argv[0]); return 2; }
        }
        if (!in_path || !name) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_set_io_backend(io_backend, err, sizeof(err))) {
            fprintf(stderr, "inspect-tensor failed: %s\n", err); return 2;
        }
        if (!qx_dump_tensor_summary(in_path, name, stdout, err, sizeof(err))) {
            fprintf(stderr, "inspect-tensor failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "verify-qxf") == 0) {
        const char *in_path = NULL;
        const char *io_backend = "buffered";
        uint32_t max_tensors = 0;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--max") == 0 && i + 1 < argc) max_tensors = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--io-backend") == 0 && i + 1 < argc) io_backend = argv[++i];
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_set_io_backend(io_backend, err, sizeof(err))) {
            fprintf(stderr, "verify-qxf failed: %s\n", err); return 2;
        }
        if (!qx_verify_all_tensors(in_path, max_tensors, stdout, err, sizeof(err))) {
            fprintf(stderr, "verify-qxf failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "expert-index") == 0) {
        const char *in_path = NULL;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_expert_index_summary(in_path, stdout, err, sizeof(err))) {
            fprintf(stderr, "expert-index failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "expert-quant-coverage") == 0) {
        const char *in_path = NULL;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_expert_quant_coverage_summary(in_path, stdout, err, sizeof(err))) {
            fprintf(stderr, "expert-quant-coverage failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "expert-plan") == 0) {
        const char *in_path = NULL;
        double vram_gib = 2.0;
        double ram_gib = 6.5;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--vram-gib") == 0 && i + 1 < argc) vram_gib = strtod(argv[++i], NULL);
            else if (strcmp(argv[i], "--ram-gib") == 0 && i + 1 < argc) ram_gib = strtod(argv[++i], NULL);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_expert_cache_plan(in_path, vram_gib, ram_gib, stdout, err, sizeof(err))) {
            fprintf(stderr, "expert-plan failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "expert-slice") == 0) {
        const char *in_path = NULL;
        uint32_t layer = 0;
        uint32_t expert = 0;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) layer = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--expert") == 0 && i + 1 < argc) expert = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_expert_slice_summary(in_path, layer, expert, stdout, err, sizeof(err))) {
            fprintf(stderr, "expert-slice failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "expert-load") == 0) {
        const char *in_path = NULL;
        const char *kind = "gate";
        uint32_t layer = 0;
        uint32_t expert = 0;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) layer = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--expert") == 0 && i + 1 < argc) expert = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--kind") == 0 && i + 1 < argc) kind = argv[++i];
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_expert_load_summary(in_path, layer, expert, kind, stdout, err, sizeof(err))) {
            fprintf(stderr, "expert-load failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "cache-demo") == 0) {
        const char *in_path = NULL;
        const char *sequence = NULL;
        uint32_t slots = 0;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--slots") == 0 && i + 1 < argc) slots = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--sequence") == 0 && i + 1 < argc) sequence = argv[++i];
            else { usage(argv[0]); return 2; }
        }
        if (!in_path || !sequence || slots == 0) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_cache_demo_summary(in_path, slots, sequence, stdout, err, sizeof(err))) {
            fprintf(stderr, "cache-demo failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "bench-expert-load") == 0) {
        const char *in_path = NULL;
        const char *kind = "gate";
        uint32_t iters = 128;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--kind") == 0 && i + 1 < argc) kind = argv[++i];
            else { usage(argv[0]); return 2; }
        }
        if (!in_path || iters == 0) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_expert_load_benchmark(in_path, iters, kind, stdout, err, sizeof(err))) {
            fprintf(stderr, "bench-expert-load failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "cache-run") == 0) {
        const char *in_path = NULL;
        const char *sequence = NULL;
        const char *sequence_file = NULL;
        uint32_t slots = 0;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--slots") == 0 && i + 1 < argc) slots = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--sequence") == 0 && i + 1 < argc) sequence = argv[++i];
            else if (strcmp(argv[i], "--sequence-file") == 0 && i + 1 < argc) sequence_file = argv[++i];
            else { usage(argv[0]); return 2; }
        }
        char *owned_sequence = NULL;
        if (!sequence && sequence_file) {
            owned_sequence = read_text_file(sequence_file);
            sequence = owned_sequence;
        }
        if (!in_path || !sequence || slots == 0) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_cache_run_summary(in_path, slots, sequence, stdout, err, sizeof(err))) {
            fprintf(stderr, "cache-run failed: %s\n", err);
            free(owned_sequence);
            return 1;
        }
        free(owned_sequence);
        return 0;
    }

    if (strcmp(argv[1], "cache-run-expert") == 0) {
        const char *in_path = NULL;
        const char *sequence = NULL;
        const char *sequence_file = NULL;
        uint32_t slots = 0;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--slots") == 0 && i + 1 < argc) slots = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--sequence") == 0 && i + 1 < argc) sequence = argv[++i];
            else if (strcmp(argv[i], "--sequence-file") == 0 && i + 1 < argc) sequence_file = argv[++i];
            else { usage(argv[0]); return 2; }
        }
        char *owned_sequence = NULL;
        if (!sequence && sequence_file) { owned_sequence = read_text_file(sequence_file); sequence = owned_sequence; }
        if (!in_path || !sequence || slots == 0) { free(owned_sequence); usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_cache_run_expert_summary(in_path, slots, sequence, stdout, err, sizeof(err))) {
            fprintf(stderr, "cache-run-expert failed: %s\n", err);
            free(owned_sequence);
            return 1;
        }
        free(owned_sequence);
        return 0;
    }

    if (strcmp(argv[1], "expert-cache-plan-complete") == 0) {
        const char *in_path = NULL;
        double vram_gib = 2.0;
        double ram_gib = 6.5;
        uint32_t top_k = 8;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--vram-gib") == 0 && i + 1 < argc) vram_gib = strtod(argv[++i], NULL);
            else if (strcmp(argv[i], "--ram-gib") == 0 && i + 1 < argc) ram_gib = strtod(argv[++i], NULL);
            else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_expert_complete_cache_plan(in_path, vram_gib, ram_gib, top_k, stdout, err, sizeof(err))) {
            fprintf(stderr, "expert-cache-plan-complete failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "runtime-plan") == 0) {
        const char *in_path = NULL;
        const char *kv = "int8";
        uint32_t ctx = 4096;
        uint32_t top_k = 8;
        double vram_gib = 4.2, ram_gib = 11.0, hot_vram_gib = 2.0, hot_ram_gib = 6.5;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) ctx = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--kv") == 0 && i + 1 < argc) kv = argv[++i];
            else if (strcmp(argv[i], "--vram-gib") == 0 && i + 1 < argc) vram_gib = strtod(argv[++i], NULL);
            else if (strcmp(argv[i], "--ram-gib") == 0 && i + 1 < argc) ram_gib = strtod(argv[++i], NULL);
            else if (strcmp(argv[i], "--hot-vram-gib") == 0 && i + 1 < argc) hot_vram_gib = strtod(argv[++i], NULL);
            else if (strcmp(argv[i], "--hot-ram-gib") == 0 && i + 1 < argc) hot_ram_gib = strtod(argv[++i], NULL);
            else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_runtime_plan(in_path, ctx, kv, vram_gib, ram_gib, hot_vram_gib, hot_ram_gib, top_k, stdout, err, sizeof(err))) {
            fprintf(stderr, "runtime-plan failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "kv-cache-probe") == 0) {
        const char *in_path = NULL;
        const char *kv = "int8";
        uint32_t ctx = 4096;
        uint32_t token = 0;
        uint32_t layer = 0;
        uint32_t head = 0;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) ctx = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--kv") == 0 && i + 1 < argc) kv = argv[++i];
            else if (strcmp(argv[i], "--token") == 0 && i + 1 < argc) token = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) layer = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--head") == 0 && i + 1 < argc) head = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_kv_cache_probe_summary(in_path, ctx, kv, token, layer, head, stdout, err, sizeof(err))) {
            fprintf(stderr, "kv-cache-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "kv-cache-buffer-probe") == 0) {
        const char *in_path = NULL;
        const char *kv = "int8";
        uint32_t ctx = 64;
        uint32_t token = 0;
        uint32_t layer = 0;
        uint32_t head = 0;
        uint32_t seed = 1;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) ctx = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--kv") == 0 && i + 1 < argc) kv = argv[++i];
            else if (strcmp(argv[i], "--token") == 0 && i + 1 < argc) token = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) layer = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--head") == 0 && i + 1 < argc) head = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_kv_cache_buffer_probe_summary(in_path, ctx, kv, token, layer, head, seed, stdout, err, sizeof(err))) {
            fprintf(stderr, "kv-cache-buffer-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "attention-cache-probe") == 0) {
        const char *in_path = NULL;
        const char *kv = "int8";
        uint32_t ctx = 64;
        uint32_t layer = 0;
        uint32_t tokens = 2;
        uint32_t blocks = 1;
        uint32_t seed = 1;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) ctx = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--kv") == 0 && i + 1 < argc) kv = argv[++i];
            else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) layer = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) blocks = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_attention_cache_probe_summary(in_path, ctx, kv, layer, tokens, blocks, seed, stdout, err, sizeof(err))) {
            fprintf(stderr, "attention-cache-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "attention-softmax-probe") == 0) {
        const char *in_path = NULL;
        const char *kv = "int8";
        uint32_t ctx = 64;
        uint32_t layer = 0;
        uint32_t tokens = 4;
        uint32_t blocks = 1;
        uint32_t seed = 1;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) ctx = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--kv") == 0 && i + 1 < argc) kv = argv[++i];
            else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) layer = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) blocks = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_attention_softmax_probe_summary(in_path, ctx, kv, layer, tokens, blocks, seed, stdout, err, sizeof(err))) {
            fprintf(stderr, "attention-softmax-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "attention-vector-probe") == 0) {
        const char *in_path = NULL;
        const char *kv = "int8";
        uint32_t ctx = 64;
        uint32_t layer = 0;
        uint32_t tokens = 4;
        uint32_t dims = 16;
        uint32_t seed = 1;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) ctx = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--kv") == 0 && i + 1 < argc) kv = argv[++i];
            else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) layer = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--dims") == 0 && i + 1 < argc) dims = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_attention_vector_probe_summary(in_path, ctx, kv, layer, tokens, dims, seed, stdout, err, sizeof(err))) {
            fprintf(stderr, "attention-vector-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "attention-multihead-probe") == 0) {
        const char *in_path = NULL;
        const char *kv = "int8";
        uint32_t ctx = 64;
        uint32_t layer = 0;
        uint32_t tokens = 4;
        uint32_t heads = 4;
        uint32_t dims = 16;
        uint32_t seed = 1;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) ctx = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--kv") == 0 && i + 1 < argc) kv = argv[++i];
            else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) layer = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) heads = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--dims") == 0 && i + 1 < argc) dims = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_attention_multihead_probe_summary(in_path, ctx, kv, layer, tokens, heads, dims, seed, stdout, err, sizeof(err))) {
            fprintf(stderr, "attention-multihead-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "token-embedding") == 0) {
        const char *in_path = NULL;
        uint32_t token_id = 0;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--token-id") == 0 && i + 1 < argc) token_id = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_token_embedding_summary(in_path, token_id, stdout, err, sizeof(err))) {
            fprintf(stderr, "token-embedding failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "rmsnorm-probe") == 0) {
        const char *in_path = NULL;
        const char *norm = NULL;
        uint32_t token_id = 0;
        uint32_t seed = 1;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--token-id") == 0 && i + 1 < argc) token_id = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--norm") == 0 && i + 1 < argc) norm = argv[++i];
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path || !norm) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_rmsnorm_probe_summary(in_path, token_id, norm, seed, stdout, err, sizeof(err))) {
            fprintf(stderr, "rmsnorm-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "attention-probe") == 0) {
        const char *in_path = NULL;
        const char *kv = NULL;
        uint32_t layer = 0;
        uint32_t blocks = 1;
        uint32_t seed = 1;
        uint32_t ctx = 0;
        int cache_write = 0;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) layer = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) blocks = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) ctx = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--kv") == 0 && i + 1 < argc) kv = argv[++i];
            else if (strcmp(argv[i], "--cache-write") == 0) cache_write = 1;
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_attention_probe_summary(in_path, layer, blocks, seed, ctx, kv, cache_write, stdout, err, sizeof(err))) {
            fprintf(stderr, "attention-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "forward-schedule") == 0) {
        const char *in_path = NULL;
        uint32_t token_id = 0;
        uint32_t top_k = 8;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--token-id") == 0 && i + 1 < argc) token_id = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_forward_schedule(in_path, token_id, top_k, stdout, err, sizeof(err))) {
            fprintf(stderr, "forward-schedule failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "quant-block") == 0) {
        const char *in_path = NULL;
        const char *name = NULL;
        uint64_t block = 0;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--name") == 0 && i + 1 < argc) name = argv[++i];
            else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) block = (uint64_t)strtoull(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path || !name) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_quant_block_summary(in_path, name, block, stdout, err, sizeof(err))) {
            fprintf(stderr, "quant-block failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "matvec-stub") == 0) {
        const char *in_path = NULL;
        const char *name = NULL;
        uint32_t rows = 1;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--name") == 0 && i + 1 < argc) name = argv[++i];
            else if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) rows = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path || !name) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_matvec_stub_summary(in_path, name, rows, stdout, err, sizeof(err))) {
            fprintf(stderr, "matvec-stub failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "decode-block") == 0) {
        const char *in_path = NULL;
        const char *name = NULL;
        uint64_t block = 0;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--name") == 0 && i + 1 < argc) name = argv[++i];
            else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) block = (uint64_t)strtoull(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path || !name) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_decode_block_summary(in_path, name, block, stdout, err, sizeof(err))) {
            fprintf(stderr, "decode-block failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "block-dot") == 0) {
        const char *in_path = NULL;
        const char *name = NULL;
        uint64_t block = 0;
        uint32_t seed = 1;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--name") == 0 && i + 1 < argc) name = argv[++i];
            else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) block = (uint64_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path || !name) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_block_dot_summary(in_path, name, block, seed, stdout, err, sizeof(err))) {
            fprintf(stderr, "block-dot failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "matvec-row") == 0) {
        const char *in_path = NULL;
        const char *name = NULL;
        uint64_t start_block = 0;
        uint32_t blocks = 1;
        uint32_t seed = 1;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--name") == 0 && i + 1 < argc) name = argv[++i];
            else if (strcmp(argv[i], "--start-block") == 0 && i + 1 < argc) start_block = (uint64_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) blocks = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path || !name) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_matvec_row_summary(in_path, name, start_block, blocks, seed, stdout, err, sizeof(err))) {
            fprintf(stderr, "matvec-row failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "expert-row") == 0) {
        const char *in_path = NULL;
        const char *kind = NULL;
        uint32_t layer = 0;
        uint32_t expert = 0;
        uint64_t start_block = 0;
        uint32_t blocks = 1;
        uint32_t seed = 1;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) layer = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--expert") == 0 && i + 1 < argc) expert = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--kind") == 0 && i + 1 < argc) kind = argv[++i];
            else if (strcmp(argv[i], "--start-block") == 0 && i + 1 < argc) start_block = (uint64_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) blocks = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path || !kind) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_expert_row_summary(in_path, layer, expert, kind, start_block, blocks, seed, stdout, err, sizeof(err))) {
            fprintf(stderr, "expert-row failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "expert-forward-probe") == 0) {
        const char *in_path = NULL;
        uint32_t layer = 0;
        uint32_t expert = 0;
        uint64_t start_block = 0;
        uint32_t blocks = 1;
        uint32_t seed = 1;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) layer = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--expert") == 0 && i + 1 < argc) expert = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--start-block") == 0 && i + 1 < argc) start_block = (uint64_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) blocks = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_expert_forward_probe_summary(in_path, layer, expert, start_block, blocks, seed, stdout, err, sizeof(err))) {
            fprintf(stderr, "expert-forward-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "router-topk-probe") == 0) {
        const char *in_path = NULL;
        uint32_t layer = 0;
        uint32_t top_k = 8;
        uint32_t blocks = 1;
        uint32_t seed = 1;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) layer = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) blocks = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_router_topk_probe_summary(in_path, layer, top_k, blocks, seed, stdout, err, sizeof(err))) {
            fprintf(stderr, "router-topk-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "layer-forward-probe") == 0) {
        const char *in_path = NULL;
        uint32_t layer = 0;
        uint32_t top_k = 8;
        uint32_t blocks = 1;
        uint32_t seed = 1;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) layer = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) blocks = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_layer_forward_probe_summary(in_path, layer, top_k, blocks, seed, stdout, err, sizeof(err))) {
            fprintf(stderr, "layer-forward-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "moe-forward-probe") == 0) {
        const char *in_path = NULL;
        uint32_t layers = 1;
        uint32_t top_k = 8;
        uint32_t blocks = 1;
        uint32_t seed = 1;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) layers = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) blocks = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_moe_forward_probe_summary(in_path, layers, top_k, blocks, seed, stdout, err, sizeof(err))) {
            fprintf(stderr, "moe-forward-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "logits-probe") == 0) {
        const char *in_path = NULL;
        double activation = 1.0;
        uint32_t top_n = 5;
        uint32_t scan = 64;
        uint32_t seed = 1;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--activation") == 0 && i + 1 < argc) activation = strtod(argv[++i], NULL);
            else if (strcmp(argv[i], "--top-n") == 0 && i + 1 < argc) top_n = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--scan") == 0 && i + 1 < argc) scan = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_logits_probe_summary(in_path, activation, top_n, scan, seed, stdout, err, sizeof(err))) {
            fprintf(stderr, "logits-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "sampler-probe") == 0) {
        const char *in_path = NULL;
        double activation = 1.0;
        double temperature = 0.0;
        uint32_t top_k = 5;
        uint32_t scan = 64;
        uint32_t seed = 1;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--activation") == 0 && i + 1 < argc) activation = strtod(argv[++i], NULL);
            else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--scan") == 0 && i + 1 < argc) scan = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) temperature = strtod(argv[++i], NULL);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_sampler_probe_summary(in_path, activation, top_k, scan, temperature, seed, stdout, err, sizeof(err))) {
            fprintf(stderr, "sampler-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "tokenizer-export") == 0) {
        const char *gguf_path = NULL;
        const char *out_path = NULL;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--gguf") == 0 && i + 1 < argc) gguf_path = argv[++i];
            else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
            else { usage(argv[0]); return 2; }
        }
        if (!gguf_path || !out_path) { usage(argv[0]); return 2; }
        char err[256];
        uint64_t count = 0;
        if (!qx_gguf_export_tokenizer_tokens(gguf_path, out_path, &count, err, sizeof(err))) {
            fprintf(stderr, "tokenizer-export failed: %s\n", err);
            return 1;
        }
        printf("{\n  \"exported\": true,\n  \"token_count\": %llu\n}\n", (unsigned long long)count);
        return 0;
    }

    if (strcmp(argv[1], "tokenizer-probe") == 0) {
        const char *in_path = NULL;
        const char *tokens_path = NULL;
        uint32_t token_id = 0;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens_path = argv[++i];
            else if (strcmp(argv[i], "--token-id") == 0 && i + 1 < argc) token_id = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_tokenizer_probe_summary(in_path, tokens_path, token_id, stdout, err, sizeof(err))) {
            fprintf(stderr, "tokenizer-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "generate-probe") == 0) {
        const char *in_path = NULL;
        const char *tokens_path = NULL;
        uint32_t prompt_token = 0;
        uint32_t steps = 1;
        uint32_t top_k = 5;
        uint32_t scan = 64;
        double temperature = 0.0;
        uint32_t seed = 1;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens_path = argv[++i];
            else if (strcmp(argv[i], "--prompt-token") == 0 && i + 1 < argc) prompt_token = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) steps = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--scan") == 0 && i + 1 < argc) scan = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) temperature = strtod(argv[++i], NULL);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_generate_probe_summary(in_path, tokens_path, prompt_token, steps, top_k, scan, temperature, seed, stdout, err, sizeof(err))) {
            fprintf(stderr, "generate-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "residual-vector-probe") == 0) {
        const char *in_path = NULL;
        const char *norm = NULL;
        uint32_t token_id = 0;
        uint32_t dims = 64;
        uint32_t seed = 1;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--token-id") == 0 && i + 1 < argc) token_id = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--norm") == 0 && i + 1 < argc) norm = argv[++i];
            else if (strcmp(argv[i], "--dims") == 0 && i + 1 < argc) dims = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_residual_vector_probe_summary(in_path, token_id, norm, dims, seed, stdout, err, sizeof(err))) {
            fprintf(stderr, "residual-vector-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "projection-matvec-probe") == 0) {
        const char *in_path = NULL;
        const char *kv_format = "int8";
        const char *norm = NULL;
        int residual_vector = 0;
        uint32_t layer = 0;
        uint32_t token_id = 0;
        uint32_t rows = 4;
        uint32_t dims = 64;
        uint32_t seed = 1;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) layer = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--token-id") == 0 && i + 1 < argc) token_id = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) rows = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--dims") == 0 && i + 1 < argc) dims = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--kv") == 0 && i + 1 < argc) kv_format = argv[++i];
            else if (strcmp(argv[i], "--residual-vector") == 0) residual_vector = 1;
            else if (strcmp(argv[i], "--norm") == 0 && i + 1 < argc) norm = argv[++i];
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_projection_matvec_probe_summary(in_path, layer, token_id, rows, dims, kv_format, residual_vector, norm, seed, stdout, err, sizeof(err))) {
            fprintf(stderr, "projection-matvec-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "rope-gqa-golden-probe") == 0) {
        uint32_t tokens = 2;
        uint32_t q_heads_run = 9;
        uint32_t seed = 7;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--q-heads-run") == 0 && i + 1 < argc) q_heads_run = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        char err[256];
        if (!qx_dump_rope_gqa_golden_probe_summary(tokens, q_heads_run, seed, stdout, err, sizeof(err))) {
            fprintf(stderr, "rope-gqa-golden-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "real-qkv-golden-probe") == 0) {
        const char *in_path = NULL;
        uint32_t layer = 0, token_a = 42, token_b = 43, q_heads_run = 9, seed = 7;
        int full_moe = 0;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) layer = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--token-a") == 0 && i + 1 < argc) token_a = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--token-b") == 0 && i + 1 < argc) token_b = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--q-heads-run") == 0 && i + 1 < argc) q_heads_run = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--full-moe") == 0) full_moe = 1;
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_real_qkv_golden_probe_summary(in_path, layer, token_a, token_b, q_heads_run, seed, full_moe, stdout, err, sizeof(err))) {
            fprintf(stderr, "real-qkv-golden-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "moe-stage-probe") == 0) {
        const char *in_path = NULL;
        const char *ffn_input_path = NULL;
        const char *output_dir = NULL;
        const char *activation_mode = "f32";
        uint32_t layer = 0u;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--ffn-inp") == 0 && i + 1 < argc) ffn_input_path = argv[++i];
            else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) output_dir = argv[++i];
            else if (strcmp(argv[i], "--activation") == 0 && i + 1 < argc) activation_mode = argv[++i];
            else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
                char *end = NULL;
                errno = 0;
                unsigned long parsed = strtoul(argv[++i], &end, 10);
                if (errno || !end || *end || parsed > UINT32_MAX) { usage(argv[0]); return 2; }
                layer = (uint32_t)parsed;
            } else { usage(argv[0]); return 2; }
        }
        if (!in_path || !ffn_input_path || !output_dir) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_moe_stage_probe_summary(in_path, layer, ffn_input_path, output_dir, activation_mode, stdout, err, sizeof(err))) {
            fprintf(stderr, "moe-stage-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "attention-stage-probe") == 0) {
        const char *in_path = NULL;
        const char *layer_input_path = NULL;
        const char *output_dir = NULL;
        const char *activation_mode = "f32";
        const char *kv_format = "f32";
        uint32_t layer = 0u;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--layer-in") == 0 && i + 1 < argc) layer_input_path = argv[++i];
            else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) output_dir = argv[++i];
            else if (strcmp(argv[i], "--activation") == 0 && i + 1 < argc) activation_mode = argv[++i];
            else if (strcmp(argv[i], "--kv") == 0 && i + 1 < argc) kv_format = argv[++i];
            else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
                char *end = NULL;
                errno = 0;
                unsigned long parsed = strtoul(argv[++i], &end, 10);
                if (errno || !end || *end || parsed > UINT32_MAX) { usage(argv[0]); return 2; }
                layer = (uint32_t)parsed;
            } else { usage(argv[0]); return 2; }
        }
        if (!in_path || !layer_input_path || !output_dir) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_attention_stage_probe_summary(in_path, layer, layer_input_path, output_dir, activation_mode, kv_format, stdout, err, sizeof(err))) {
            fprintf(stderr, "attention-stage-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "final-head-probe") == 0) {
        const char *in_path = NULL;
        const char *residual_path = NULL;
        const char *output_dir = NULL;
        const char *activation_mode = "f32";
        uint32_t top_n = 5u;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--residual") == 0 && i + 1 < argc) residual_path = argv[++i];
            else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) output_dir = argv[++i];
            else if (strcmp(argv[i], "--activation") == 0 && i + 1 < argc) activation_mode = argv[++i];
            else if (strcmp(argv[i], "--top-n") == 0 && i + 1 < argc) {
                char *end = NULL;
                errno = 0;
                unsigned long parsed = strtoul(argv[++i], &end, 10);
                if (errno || !end || *end || parsed == 0u || parsed > 32u) { usage(argv[0]); return 2; }
                top_n = (uint32_t)parsed;
            } else { usage(argv[0]); return 2; }
        }
        if (!in_path || !residual_path || !output_dir) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_final_head_probe_summary(in_path, residual_path, output_dir, activation_mode, top_n, stdout, err, sizeof(err))) {
            fprintf(stderr, "final-head-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "expert-q8-k-dot-probe") == 0) {
        const char *in_path = NULL;
        const char *tensor_name = NULL;
        const char *activation_path = NULL;
        uint32_t expert = UINT32_MAX, row = UINT32_MAX;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--name") == 0 && i + 1 < argc) tensor_name = argv[++i];
            else if (strcmp(argv[i], "--activation") == 0 && i + 1 < argc) activation_path = argv[++i];
            else if ((strcmp(argv[i], "--expert") == 0 || strcmp(argv[i], "--row") == 0) && i + 1 < argc) {
                int is_expert = strcmp(argv[i], "--expert") == 0;
                char *end = NULL;
                errno = 0;
                unsigned long parsed = strtoul(argv[++i], &end, 10);
                if (errno || !end || *end || parsed > UINT32_MAX) { usage(argv[0]); return 2; }
                if (is_expert) expert = (uint32_t)parsed; else row = (uint32_t)parsed;
            } else { usage(argv[0]); return 2; }
        }
        if (!in_path || !tensor_name || !activation_path || expert == UINT32_MAX || row == UINT32_MAX) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_expert_q8_k_dot_probe_summary(in_path, tensor_name, expert, row, activation_path, stdout, err, sizeof(err))) {
            fprintf(stderr, "expert-q8-k-dot-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "state-loop-probe") == 0) {
        const char *in_path = NULL;
        const char *tokens_path = NULL;
        const char *kv_format = "int8";
        const char *activation_format = "f32";
        int real_kv = 0;
        int projection_matvec = 0;
        int residual_vector = 0;
        int residual_carry = 0;
        int numeric_deltas = 0;
        int delta_vectors = 0;
        int attention_output_vector = 0;
        int causal_attention = 0;
        int rope_gqa_attention = 0;
        int full_moe = 0;
        int final_head = 0;
        int bench = 0;
        const char *residual_dump_dir = NULL;
        const char *residual_input_path = NULL;
        const char *kv_snapshot_out_path = NULL;
        const char *kv_snapshot_in_path = NULL;
        int start_layer_set = 0;
        uint32_t start_layer = 0u;
        const char *norm = NULL;
        uint32_t residual_dims = 64;
        uint32_t prompt_token = 0;
        uint32_t steps = 1;
        uint32_t layers = 1;
        uint32_t ctx = 16;
        uint32_t top_k = 5;
        uint32_t scan = 64;
        uint32_t logits_top_n = 5;
        double temperature = 0.0;
        uint32_t seed = 1;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens_path = argv[++i];
            else if (strcmp(argv[i], "--prompt-token") == 0 && i + 1 < argc) prompt_token = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) steps = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) layers = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) ctx = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--kv") == 0 && i + 1 < argc) kv_format = argv[++i];
            else if (strcmp(argv[i], "--activation") == 0 && i + 1 < argc) activation_format = argv[++i];
            else if (strcmp(argv[i], "--real-kv") == 0) real_kv = 1;
            else if (strcmp(argv[i], "--projection-matvec") == 0) { projection_matvec = 1; real_kv = 1; }
            else if (strcmp(argv[i], "--residual-vector") == 0) { residual_vector = 1; projection_matvec = 1; real_kv = 1; }
            else if (strcmp(argv[i], "--residual-carry") == 0) { residual_carry = 1; residual_vector = 1; projection_matvec = 1; real_kv = 1; }
            else if (strcmp(argv[i], "--numeric-deltas") == 0) { numeric_deltas = 1; residual_carry = 1; residual_vector = 1; projection_matvec = 1; real_kv = 1; }
            else if (strcmp(argv[i], "--delta-vectors") == 0) { delta_vectors = 1; numeric_deltas = 1; residual_carry = 1; residual_vector = 1; projection_matvec = 1; real_kv = 1; }
            else if (strcmp(argv[i], "--attention-output-vector") == 0) { attention_output_vector = 1; delta_vectors = 1; numeric_deltas = 1; residual_carry = 1; residual_vector = 1; projection_matvec = 1; real_kv = 1; }
            else if (strcmp(argv[i], "--causal-attention") == 0) { causal_attention = 1; attention_output_vector = 1; delta_vectors = 1; numeric_deltas = 1; residual_carry = 1; residual_vector = 1; projection_matvec = 1; real_kv = 1; }
            else if (strcmp(argv[i], "--rope-gqa-attention") == 0) { rope_gqa_attention = 1; causal_attention = 1; attention_output_vector = 1; delta_vectors = 1; numeric_deltas = 1; residual_carry = 1; residual_vector = 1; projection_matvec = 1; real_kv = 1; }
            else if (strcmp(argv[i], "--full-moe") == 0) { full_moe = 1; rope_gqa_attention = 1; causal_attention = 1; attention_output_vector = 1; delta_vectors = 1; numeric_deltas = 1; residual_carry = 1; residual_vector = 1; projection_matvec = 1; real_kv = 1; residual_dims = 2048; }
            else if (strcmp(argv[i], "--final-head") == 0) final_head = 1;
            else if (strcmp(argv[i], "--bench") == 0) bench = 1;
            else if (strcmp(argv[i], "--dump-residuals") == 0 && i + 1 < argc) residual_dump_dir = argv[++i];
            else if (strcmp(argv[i], "--residual-in") == 0 && i + 1 < argc) residual_input_path = argv[++i];
            else if (strcmp(argv[i], "--kv-snapshot-out") == 0 && i + 1 < argc) kv_snapshot_out_path = argv[++i];
            else if (strcmp(argv[i], "--kv-snapshot-in") == 0 && i + 1 < argc) kv_snapshot_in_path = argv[++i];
            else if (strcmp(argv[i], "--start-layer") == 0 && i + 1 < argc) {
                char *end = NULL;
                errno = 0;
                unsigned long parsed = strtoul(argv[++i], &end, 10);
                if (errno || !end || *end || parsed > UINT32_MAX) { usage(argv[0]); return 2; }
                start_layer = (uint32_t)parsed;
                start_layer_set = 1;
            }
            else if (strcmp(argv[i], "--residual-dims") == 0 && i + 1 < argc) residual_dims = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--norm") == 0 && i + 1 < argc) norm = argv[++i];
            else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--scan") == 0 && i + 1 < argc) scan = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--top-n") == 0 && i + 1 < argc) logits_top_n = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) temperature = strtod(argv[++i], NULL);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!in_path || start_layer_set != (residual_input_path != NULL) || (kv_snapshot_out_path && kv_snapshot_in_path)) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_state_loop_probe_summary(in_path, tokens_path, prompt_token, steps, layers, ctx, kv_format, activation_format, real_kv, projection_matvec, residual_vector, residual_carry, numeric_deltas, delta_vectors, attention_output_vector, causal_attention, rope_gqa_attention, full_moe, final_head, bench, residual_dims, norm, top_k, scan, logits_top_n, temperature, seed, residual_dump_dir, start_layer, residual_input_path, kv_snapshot_out_path, kv_snapshot_in_path, stdout, err, sizeof(err))) {
            fprintf(stderr, "state-loop-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "token-forward-probe") == 0) {
        const char *in_path = NULL;
        const char *norm = NULL;
        const char *tokens_path = NULL;
        int32_t attention_layer = -1;
        int multihead_attention = 0;
        uint32_t attention_heads = 4;
        uint32_t attention_dims = 16;
        int logits_enabled = 0;
        uint32_t logits_top_n = 5;
        int sample_enabled = 0;
        double temperature = 0.0;
        int decode_token = 0;
        uint32_t token_id = 0;
        uint32_t layers = 1;
        uint32_t top_k = 8;
        uint32_t blocks = 1;
        uint32_t seed = 1;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_path = argv[++i];
            else if (strcmp(argv[i], "--token-id") == 0 && i + 1 < argc) token_id = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) layers = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) blocks = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--norm") == 0 && i + 1 < argc) norm = argv[++i];
            else if (strcmp(argv[i], "--attention-layer") == 0 && i + 1 < argc) attention_layer = (int32_t)strtol(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--multihead-attention") == 0) multihead_attention = 1;
            else if (strcmp(argv[i], "--attention-heads") == 0 && i + 1 < argc) attention_heads = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--attention-dims") == 0 && i + 1 < argc) attention_dims = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--logits") == 0) logits_enabled = 1;
            else if (strcmp(argv[i], "--top-n") == 0 && i + 1 < argc) logits_top_n = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--sample") == 0) sample_enabled = 1;
            else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) temperature = strtod(argv[++i], NULL);
            else if (strcmp(argv[i], "--sample-top-k") == 0 && i + 1 < argc) logits_top_n = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--decode-token") == 0) decode_token = 1;
            else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens_path = argv[++i];
            else { usage(argv[0]); return 2; }
        }
        if (!in_path) { usage(argv[0]); return 2; }
        char err[256];
        if (!qx_dump_token_forward_probe_summary(in_path, token_id, layers, top_k, blocks, seed, norm, attention_layer, multihead_attention, attention_heads, attention_dims, logits_enabled, logits_top_n, sample_enabled, temperature, decode_token, tokens_path, stdout, err, sizeof(err))) {
            fprintf(stderr, "token-forward-probe failed: %s\n", err);
            return 1;
        }
        return 0;
    }

    if (strcmp(argv[1], "route-trace") == 0) {
        uint32_t layers = 48, experts = 128, top_k = 8, tokens = 1, seed = 1, reuse_pct = 0;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) layers = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--experts") == 0 && i + 1 < argc) experts = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--reuse-pct") == 0 && i + 1 < argc) reuse_pct = (uint32_t)strtoul(argv[++i], NULL, 10);
            else { usage(argv[0]); return 2; }
        }
        if (!route_trace(layers, experts, top_k, tokens, seed, reuse_pct)) {
            fprintf(stderr, "route-trace failed: invalid arguments\n");
            return 1;
        }
        return 0;
    }

    usage(argv[0]);
    return 2;
}
