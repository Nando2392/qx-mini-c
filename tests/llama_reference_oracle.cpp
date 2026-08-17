#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <direct.h>
#include <limits>
#include <string>
#include <vector>

#include "llama.h"
#include "llama-ext.h"

#ifndef LLAMA_COMMIT
#define LLAMA_COMMIT "unknown"
#endif

static bool parse_u32(const char * text, uint32_t * value) {
    if (!text || !*text || !value || *text == '-') return false;
    char * end = nullptr;
    errno = 0;
    unsigned long parsed = std::strtoul(text, &end, 10);
    if (errno || !end || *end || parsed > std::numeric_limits<uint32_t>::max()) return false;
    *value = static_cast<uint32_t>(parsed);
    return true;
}

static bool parse_layers(const char * text, uint32_t n_layer, std::vector<uint32_t> * layers) {
    if (!text || !layers) return false;
    const char * cursor = text;
    while (*cursor) {
        const char * comma = std::strchr(cursor, ',');
        std::string part(cursor, comma ? static_cast<size_t>(comma - cursor) : std::strlen(cursor));
        uint32_t layer = 0;
        if (!parse_u32(part.c_str(), &layer) || layer >= n_layer) return false;
        for (uint32_t existing : *layers) if (existing == layer) return false;
        layers->push_back(layer);
        if (layers->size() > 64) return false;
        if (!comma) break;
        cursor = comma + 1;
    }
    return !layers->empty();
}

static uint64_t fnv1a64(const void * data, size_t size) {
    const auto * bytes = static_cast<const unsigned char *>(data);
    uint64_t value = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < size; ++i) {
        value ^= bytes[i];
        value *= UINT64_C(1099511628211);
    }
    return value;
}

static bool write_f32(const std::string & path, const float * values, size_t count) {
    if (!values || count > std::numeric_limits<size_t>::max() / sizeof(float)) return false;
    FILE * file = std::fopen(path.c_str(), "wb");
    if (!file) return false;
    bool ok = std::fwrite(values, sizeof(float), count, file) == count;
    ok = std::fclose(file) == 0 && ok;
    return ok;
}

int main(int argc, char ** argv) {
    if (argc != 5 && argc != 6) {
        std::fprintf(stderr, "usage: llama_reference_oracle <model.gguf> <output-dir> <token-id> <layers-csv> [f16|q8_0]\n");
        return 2;
    }
    const char * kv_type_name = argc == 6 ? argv[5] : "f16";
    enum ggml_type kv_type = GGML_TYPE_F16;
    if (std::strcmp(kv_type_name, "q8_0") == 0) kv_type = GGML_TYPE_Q8_0;
    else if (std::strcmp(kv_type_name, "f16") != 0) {
        std::fprintf(stderr, "unsupported KV type\n");
        return 2;
    }

    uint32_t token_id = 0;
    if (!parse_u32(argv[3], &token_id)) {
        std::fprintf(stderr, "invalid token id\n");
        return 2;
    }
    if (_mkdir(argv[2]) != 0 && errno != EEXIST) {
        std::fprintf(stderr, "cannot create output directory\n");
        return 3;
    }

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (!model) {
        std::fprintf(stderr, "model load failed\n");
        llama_backend_free();
        return 4;
    }

    const int32_t n_embd = llama_model_n_embd(model);
    const int32_t n_layer = llama_model_n_layer(model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = vocab ? llama_vocab_n_tokens(vocab) : 0;
    std::vector<uint32_t> layers;
    if (n_embd <= 0 || n_layer <= 0 || n_vocab <= 0 || token_id >= static_cast<uint32_t>(n_vocab) ||
        !parse_layers(argv[4], static_cast<uint32_t>(n_layer), &layers)) {
        std::fprintf(stderr, "invalid model dimensions, token, or layers\n");
        llama_model_free(model);
        llama_backend_free();
        return 5;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 16;
    ctx_params.n_batch = 1;
    ctx_params.n_ubatch = 1;
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;
    ctx_params.type_k = kv_type;
    ctx_params.type_v = kv_type;
    ctx_params.embeddings = true;
    ctx_params.offload_kqv = false;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::fprintf(stderr, "context init failed\n");
        llama_model_free(model);
        llama_backend_free();
        return 6;
    }

    for (uint32_t layer : layers) llama_set_embeddings_layer_inp(ctx, layer, true);
    llama_token token = static_cast<llama_token>(token_id);
    llama_batch batch = llama_batch_get_one(&token, 1);
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "decode failed\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 7;
    }

    bool ok = true;
    std::printf("{\"schema\":1,\"llama_commit\":\"%s\",\"kv_type\":\"%s\",\"token_id\":%u,\"n_embd\":%d,\"n_layer\":%d,\"n_vocab\":%d,\"layers\":[", LLAMA_COMMIT, kv_type_name, token_id, n_embd, n_layer, n_vocab);
    for (size_t i = 0; i < layers.size(); ++i) {
        const uint32_t layer = layers[i];
        const float * residual = llama_get_embeddings_layer_inp(ctx, layer);
        std::string path = std::string(argv[2]) + "/layer-" + std::to_string(layer) + ".f32";
        const bool written = residual && write_f32(path, residual, static_cast<size_t>(n_embd));
        ok = ok && written;
        if (i) std::printf(",");
        std::printf("{\"layer\":%u,\"count\":%d,\"fnv1a64\":\"%" PRIu64 "\",\"written\":%s}",
                    layer, n_embd, residual ? fnv1a64(residual, static_cast<size_t>(n_embd) * sizeof(float)) : 0,
                    written ? "true" : "false");
    }

    float * logits = llama_get_logits_ith(ctx, -1);
    std::string logits_path = std::string(argv[2]) + "/logits.f32";
    const bool logits_written = logits && write_f32(logits_path, logits, static_cast<size_t>(n_vocab));
    ok = ok && logits_written;
    uint32_t argmax = 0;
    if (logits) {
        for (int32_t i = 1; i < n_vocab; ++i) if (logits[i] > logits[argmax]) argmax = static_cast<uint32_t>(i);
    }
    std::printf("],\"logits\":{\"count\":%d,\"fnv1a64\":\"%" PRIu64 "\",\"argmax\":%u,\"written\":%s},\"ok\":%s}\n",
                n_vocab, logits ? fnv1a64(logits, static_cast<size_t>(n_vocab) * sizeof(float)) : 0,
                argmax, logits_written ? "true" : "false", ok ? "true" : "false");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 8;
}
