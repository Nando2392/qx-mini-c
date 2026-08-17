#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml-quants.h"

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
                        "   or: ggml_reference_decode <iq2_xs|iq3_xxs> <file> <offset> <blocks>\n"
                        "   or: ggml_reference_decode q6_k_logits <file> <offset> <rows> <activation.f32>\n");
        return 2;
    }
    uint64_t offset = 0, blocks = 0;
    if (!parse_u64(argv[3], &offset) || !parse_u64(argv[4], &blocks) || !blocks) return 2;
    size_t block_size = 0;
    if (strcmp(argv[1], "iq2_xs") == 0) block_size = sizeof(block_iq2_xs);
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
    if (strcmp(argv[1], "iq2_xs") == 0) dequantize_row_iq2_xs((const block_iq2_xs *)raw, decoded, (int64_t)blocks * 256);
    else dequantize_row_iq3_xxs((const block_iq3_xxs *)raw, decoded, (int64_t)blocks * 256);
    size_t count = (size_t)blocks * 256u;
    int ok = fwrite(decoded, sizeof(float), count, stdout) == count;
    free(raw);
    free(decoded);
    return ok ? 0 : 5;
}
