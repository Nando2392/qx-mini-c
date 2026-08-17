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
    if (argc != 5) {
        fprintf(stderr, "usage: ggml_reference_decode <iq2_xs|iq3_xxs> <file> <offset> <blocks>\n");
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
