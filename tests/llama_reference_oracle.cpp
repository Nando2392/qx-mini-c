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

#include "ggml-backend.h"
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

struct internal_capture_record {
    const char * name;
    std::vector<float> values;
    bool captured;
};

struct internal_capture_state {
    internal_capture_record records[6] = {
        {"ffn_inp-0", {}, false},
        {"ffn_moe_out-0", {}, false},
        {"l_out-0", {}, false},
        {"Vcur-0", {}, false},
        {"kqv_out-0", {}, false},
        {"l_out-47", {}, false},
    };
    bool failed = false;
};

static bool capture_internal_tensor(struct ggml_tensor * tensor, bool ask, void * user_data) {
    auto * state = static_cast<internal_capture_state *>(user_data);
    internal_capture_record * target = nullptr;
    for (auto & record : state->records) {
        if (std::strcmp(tensor->name, record.name) == 0) {
            target = &record;
            break;
        }
    }
    if (ask) return target != nullptr;
    if (!target || target->captured || state->failed) return true;
    if (!tensor->buffer || !ggml_is_contiguous(tensor) ||
        (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16)) {
        state->failed = true;
        return true;
    }
    const int64_t elements_signed = ggml_nelements(tensor);
    if (elements_signed <= 0 || static_cast<uint64_t>(elements_signed) > std::numeric_limits<size_t>::max()) {
        state->failed = true;
        return true;
    }
    const size_t elements = static_cast<size_t>(elements_signed);
    const size_t bytes = ggml_nbytes(tensor);
    const size_t item_size = tensor->type == GGML_TYPE_F32 ? sizeof(float) : sizeof(ggml_fp16_t);
    if (elements > std::numeric_limits<size_t>::max() / sizeof(float) ||
        elements > std::numeric_limits<size_t>::max() / item_size || bytes != elements * item_size) {
        state->failed = true;
        return true;
    }
    std::vector<unsigned char> raw(bytes);
    if (ggml_backend_buffer_is_host(tensor->buffer)) std::memcpy(raw.data(), tensor->data, bytes);
    else ggml_backend_tensor_get(tensor, raw.data(), 0, bytes);
    target->values.resize(elements);
    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(target->values.data(), raw.data(), bytes);
    } else {
        for (size_t i = 0; i < elements; ++i) {
            ggml_fp16_t value;
            std::memcpy(&value, raw.data() + i * sizeof(value), sizeof(value));
            target->values[i] = ggml_fp16_to_fp32(value);
        }
    }
    target->captured = true;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5 || argc > 7) {
        std::fprintf(stderr, "usage: llama_reference_oracle <model.gguf> <output-dir> <token-id> <layers-csv> [f16|q8_0] [internals]\n");
        return 2;
    }
    const char * kv_type_name = argc >= 6 ? argv[5] : "f16";
    const bool capture_internals = argc == 7 && std::strcmp(argv[6], "internals") == 0;
    if (argc == 7 && !capture_internals) {
        std::fprintf(stderr, "unsupported capture mode\n");
        return 2;
    }
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
    internal_capture_state capture_state;
    if (capture_internals) {
        ctx_params.cb_eval = capture_internal_tensor;
        ctx_params.cb_eval_user_data = &capture_state;
    }
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

    bool ok = !capture_state.failed;
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
    float * result_norm = llama_get_embeddings(ctx);
    std::string result_norm_path = std::string(argv[2]) + "/result_norm.f32";
    const bool result_norm_written = result_norm && write_f32(result_norm_path, result_norm, static_cast<size_t>(n_embd));
    ok = ok && result_norm_written;
    uint32_t argmax = 0;
    if (logits) {
        for (int32_t i = 1; i < n_vocab; ++i) if (logits[i] > logits[argmax]) argmax = static_cast<uint32_t>(i);
    }
    std::printf("],\"logits\":{\"count\":%d,\"fnv1a64\":\"%" PRIu64 "\",\"argmax\":%u,\"written\":%s},\"result_norm\":{\"count\":%d,\"fnv1a64\":\"%" PRIu64 "\",\"written\":%s},\"internals\":[",
                n_vocab, logits ? fnv1a64(logits, static_cast<size_t>(n_vocab) * sizeof(float)) : 0,
                argmax, logits_written ? "true" : "false", n_embd,
                result_norm ? fnv1a64(result_norm, static_cast<size_t>(n_embd) * sizeof(float)) : 0,
                result_norm_written ? "true" : "false");
    size_t internals_captured = 0;
    if (capture_internals) {
        for (size_t i = 0; i < sizeof(capture_state.records) / sizeof(capture_state.records[0]); ++i) {
            auto & record = capture_state.records[i];
            const bool valid = record.captured && !record.values.empty();
            const std::string path = std::string(argv[2]) + "/" + record.name + ".f32";
            const bool written = valid && write_f32(path, record.values.data(), record.values.size());
            ok = ok && written;
            if (written) ++internals_captured;
            if (i) std::printf(",");
            std::printf("{\"name\":\"%s\",\"count\":%zu,\"fnv1a64\":\"%" PRIu64 "\",\"written\":%s}",
                        record.name, record.values.size(),
                        valid ? fnv1a64(record.values.data(), record.values.size() * sizeof(float)) : 0,
                        written ? "true" : "false");
        }
    }
    std::printf("],\"internals_captured\":%zu,\"ok\":%s}\n", internals_captured, ok ? "true" : "false");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 8;
}
