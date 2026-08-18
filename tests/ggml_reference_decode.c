#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml-quants.h"
#include "ggml-cpu.h"

static int parse_u64(const char *text, uint64_t *value) {
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || !end || *end) return 0;
    *value = (uint64_t)parsed;
    return 1;
}

int main(int argc, char **argv) {
    _setmode(_fileno(stdout), _O_BINARY);
    if (argc == 3 && strcmp(argv[1], "q8_k_quantize") == 0) {
        FILE *input = fopen(argv[2], "rb");
        if (!input) return 3;
        float activation[QK_K];
        if (fread(activation, sizeof(float), QK_K, input) != QK_K) { fclose(input); return 3; }
        if (fgetc(input) != EOF) { fclose(input); return 3; }
        fclose(input);
        block_q8_K block;
        quantize_row_q8_K_ref(activation, &block, QK_K);
        return fwrite(&block, sizeof(block), 1, stdout) == 1 ? 0 : 5;
    }
    if (argc == 6 && (strcmp(argv[1], "iq2_xs_q8_k_dot") == 0 || strcmp(argv[1], "iq3_xxs_q8_k_dot") == 0 ||
                      strcmp(argv[1], "iq2_s_q8_k_dot") == 0 || strcmp(argv[1], "iq4_xs_q8_k_dot") == 0)) {
        uint64_t offset = 0, blocks = 0;
        if (!parse_u64(argv[3], &offset) || !parse_u64(argv[4], &blocks) || !blocks || blocks > INT32_MAX / QK_K) return 2;
        const size_t block_size = strcmp(argv[1], "iq2_xs_q8_k_dot") == 0 ? sizeof(block_iq2_xs) :
            strcmp(argv[1], "iq3_xxs_q8_k_dot") == 0 ? sizeof(block_iq3_xxs) :
            strcmp(argv[1], "iq2_s_q8_k_dot") == 0 ? sizeof(block_iq2_s) : sizeof(block_iq4_xs);
        if (blocks > SIZE_MAX / block_size || blocks > SIZE_MAX / sizeof(block_q8_K) || blocks > SIZE_MAX / (QK_K * sizeof(float))) return 2;
        FILE *weights_file = fopen(argv[2], "rb");
        FILE *activation_file = fopen(argv[5], "rb");
        if (!weights_file || !activation_file) { if (weights_file) fclose(weights_file); if (activation_file) fclose(activation_file); return 3; }
        if (_fseeki64(weights_file, (__int64)offset, SEEK_SET) != 0) { fclose(weights_file); fclose(activation_file); return 3; }
        void *weights = malloc((size_t)blocks * block_size);
        float *activation = (float *)malloc((size_t)blocks * QK_K * sizeof(float));
        block_q8_K *q8 = (block_q8_K *)malloc((size_t)blocks * sizeof(block_q8_K));
        if (!weights || !activation || !q8) { free(weights); free(activation); free(q8); fclose(weights_file); fclose(activation_file); return 4; }
        const size_t count = (size_t)blocks * QK_K;
        int ok = fread(weights, block_size, (size_t)blocks, weights_file) == (size_t)blocks &&
                 fread(activation, sizeof(float), count, activation_file) == count && fgetc(activation_file) == EOF;
        fclose(weights_file);
        fclose(activation_file);
        if (!ok) { free(weights); free(activation); free(q8); return 3; }
        quantize_row_q8_K_ref(activation, q8, (int64_t)count);
        ggml_cpu_init();
        enum ggml_type type = strcmp(argv[1], "iq2_xs_q8_k_dot") == 0 ? GGML_TYPE_IQ2_XS :
            strcmp(argv[1], "iq3_xxs_q8_k_dot") == 0 ? GGML_TYPE_IQ3_XXS :
            strcmp(argv[1], "iq2_s_q8_k_dot") == 0 ? GGML_TYPE_IQ2_S : GGML_TYPE_IQ4_XS;
        const struct ggml_type_traits_cpu *traits = ggml_get_type_traits_cpu(type);
        if (!traits || !traits->vec_dot || traits->vec_dot_type != GGML_TYPE_Q8_K) { free(weights); free(activation); free(q8); return 6; }
        float dot = 0.0f;
        traits->vec_dot((int)count, &dot, 0, weights, 0, q8, 0, 1);
        free(weights); free(activation); free(q8);
        return fwrite(&dot, sizeof(dot), 1, stdout) == 1 ? 0 : 5;
    }
    if (argc == 6 && strcmp(argv[1], "q6_k_logits") == 0) {
        uint64_t offset = 0, rows = 0;
        if (!parse_u64(argv[3], &offset) || !parse_u64(argv[4], &rows) || !rows || rows > UINT32_MAX) return 2;
        FILE *activation_file = fopen(argv[5], "rb");
        if (!activation_file) return 3;
        float activation[2048];
        if (fread(activation, sizeof(float), 2048, activation_file) != 2048) { fclose(activation_file); return 3; }
        fclose(activation_file);
        FILE *input = fopen(argv[2], "rb");
        if (!input) return 3;
        if (_fseeki64(input, (__int64)offset, SEEK_SET) != 0) { fclose(input); return 3; }
        block_q6_K raw[8];
        float weights[2048];
        for (uint64_t row = 0; row < rows; ++row) {
            if (fread(raw, sizeof(block_q6_K), 8, input) != 8) { fclose(input); return 3; }
            dequantize_row_q6_K(raw, weights, 2048);
            double dot = 0.0;
            for (uint32_t i = 0; i < 2048; ++i) dot += (double)weights[i] * (double)activation[i];
            float logit = (float)dot;
            if (fwrite(&logit, sizeof(logit), 1, stdout) != 1) { fclose(input); return 5; }
        }
        fclose(input);
        return 0;
    }
    if (argc != 5) {
        fprintf(stderr, "usage: ggml_reference_decode q8_k_quantize <activation.f32>\n"
                        "   or: ggml_reference_decode <iq2_xs|iq3_xxs|iq2_s|iq4_xs>_q8_k_dot <file> <offset> <blocks> <activation.f32>\n"
                        "   or: ggml_reference_decode <q5_k|iq2_xs|iq3_xxs> <file> <offset> <blocks>\n"
                        "   or: ggml_reference_decode q6_k_logits <file> <offset> <rows> <activation.f32>\n");
        return 2;
    }
    uint64_t offset = 0, blocks = 0;
    if (!parse_u64(argv[3], &offset) || !parse_u64(argv[4], &blocks) || !blocks) return 2;
    size_t block_size = 0;
    if (strcmp(argv[1], "q5_k") == 0) block_size = sizeof(block_q5_K);
    else if (strcmp(argv[1], "iq2_xs") == 0) block_size = sizeof(block_iq2_xs);
    else if (strcmp(argv[1], "iq3_xxs") == 0) block_size = sizeof(block_iq3_xxs);
    else return 2;
    FILE *input = fopen(argv[2], "rb");
    if (!input) return 3;
    if (_fseeki64(input, (__int64)offset, SEEK_SET) != 0) { fclose(input); return 3; }
    void *raw = malloc((size_t)blocks * block_size);
    float *decoded = (float *)malloc((size_t)blocks * 256u * sizeof(float));
    if (!raw || !decoded) { free(raw); free(decoded); fclose(input); return 4; }
    if (fread(raw, block_size, (size_t)blocks, input) != (size_t)blocks) { free(raw); free(decoded); fclose(input); return 3; }
    fclose(input);
    if (strcmp(argv[1], "q5_k") == 0) dequantize_row_q5_K((const block_q5_K *)raw, decoded, (int64_t)blocks * 256);
    else if (strcmp(argv[1], "iq2_xs") == 0) dequantize_row_iq2_xs((const block_iq2_xs *)raw, decoded, (int64_t)blocks * 256);
    else dequantize_row_iq3_xxs((const block_iq3_xxs *)raw, decoded, (int64_t)blocks * 256);
    size_t count = (size_t)blocks * 256u;
    int ok = fwrite(decoded, sizeof(float), count, stdout) == count;
    free(raw);
    free(decoded);
    return ok ? 0 : 5;
}
