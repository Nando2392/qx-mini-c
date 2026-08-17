#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "llama.h"

#ifndef LLAMA_COMMIT
#define LLAMA_COMMIT "unknown"
#endif

static bool parse_u32(const char * text, uint32_t * value) {
    if (!text || !*text || !value || *text == '-') return false;
    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno || !end || *end || parsed > std::numeric_limits<uint32_t>::max()) return false;
    *value = static_cast<uint32_t>(parsed);
    return true;
}

static bool parse_tokens(const char * text, std::vector<uint32_t> * tokens) {
    if (!text || !*text || !tokens) return false;
    tokens->clear();
    const char * cursor = text;
    while (*cursor) {
        const char * comma = std::strchr(cursor, ',');
        const std::string part(cursor, comma ? static_cast<size_t>(comma - cursor) : std::strlen(cursor));
        uint32_t token = 0;
        if (part.empty() || !parse_u32(part.c_str(), &token)) return false;
        tokens->push_back(token);
        if (!comma) break;
        cursor = comma + 1;
        if (!*cursor) return false;
    }
    return !tokens->empty() && tokens->size() <= 64;
}

static bool decode_one(llama_context * ctx, uint32_t token_id, int32_t n_vocab, uint32_t * selected) {
    llama_token token = static_cast<llama_token>(token_id);
    llama_batch batch = llama_batch_get_one(&token, 1);
    if (llama_decode(ctx, batch) != 0) return false;
    float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits || !selected) return false;
    uint32_t argmax = 0;
    if (!std::isfinite(logits[0])) return false;
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (!std::isfinite(logits[i])) return false;
        if (logits[i] > logits[argmax]) argmax = static_cast<uint32_t>(i);
    }
    *selected = argmax;
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: llama_sequence_oracle <model.gguf> <tokens-csv> <generation-steps> <f16|q8_0>\n");
        return 2;
    }

    std::vector<uint32_t> prompt_tokens;
    uint32_t generation_steps = 0;
    if (!parse_tokens(argv[2], &prompt_tokens) || !parse_u32(argv[3], &generation_steps) || generation_steps == 0 || generation_steps > 64) {
        std::fprintf(stderr, "invalid prompt tokens or generation steps\n");
        return 2;
    }
    enum ggml_type kv_type = GGML_TYPE_F16;
    if (std::strcmp(argv[4], "q8_0") == 0) kv_type = GGML_TYPE_Q8_0;
    else if (std::strcmp(argv[4], "f16") != 0) {
        std::fprintf(stderr, "unsupported KV type\n");
        return 2;
    }
    if (prompt_tokens.size() > std::numeric_limits<uint32_t>::max() - generation_steps + 1u) return 2;
    const uint32_t total_decodes = static_cast<uint32_t>(prompt_tokens.size()) + generation_steps - 1u;
    if (total_decodes > 127u) {
        std::fprintf(stderr, "sequence exceeds oracle context\n");
        return 2;
    }

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (!model) {
        std::fprintf(stderr, "model load failed\n");
        llama_backend_free();
        return 3;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = vocab ? llama_vocab_n_tokens(vocab) : 0;
    bool valid = n_vocab > 0;
    for (uint32_t token : prompt_tokens) valid = valid && token < static_cast<uint32_t>(n_vocab);
    if (!valid) {
        std::fprintf(stderr, "invalid vocabulary or prompt token\n");
        llama_model_free(model);
        llama_backend_free();
        return 4;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 128;
    ctx_params.n_batch = 1;
    ctx_params.n_ubatch = 1;
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;
    ctx_params.type_k = kv_type;
    ctx_params.type_v = kv_type;
    ctx_params.offload_kqv = false;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::fprintf(stderr, "context init failed\n");
        llama_model_free(model);
        llama_backend_free();
        return 5;
    }

    uint32_t selected = 0;
    bool ok = true;
    for (uint32_t token : prompt_tokens) {
        ok = ok && decode_one(ctx, token, n_vocab, &selected);
        if (!ok) break;
    }
    std::vector<uint32_t> generated;
    if (ok) generated.push_back(selected);
    for (uint32_t step = 1; ok && step < generation_steps; ++step) {
        ok = decode_one(ctx, selected, n_vocab, &selected);
        if (ok) generated.push_back(selected);
    }

    std::printf("{\"schema\":1,\"llama_commit\":\"%s\",\"kv_type\":\"%s\",\"n_vocab\":%d,\"generation_steps\":%u,\"prompt_tokens\":[",
                LLAMA_COMMIT, argv[4], n_vocab, generation_steps);
    for (size_t i = 0; i < prompt_tokens.size(); ++i) std::printf("%s%u", i ? "," : "", prompt_tokens[i]);
    std::printf("],\"generated_tokens\":[");
    for (size_t i = 0; i < generated.size(); ++i) std::printf("%s%u", i ? "," : "", generated[i]);
    std::printf("],\"ok\":%s}\n", ok && generated.size() == generation_steps ? "true" : "false");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ok && generated.size() == generation_steps ? 0 : 6;
}
