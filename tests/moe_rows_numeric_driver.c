/* Postimplementation numeric gate: private production entry, independent Q4_K oracle. */
#include "../src/qx_format.c"

#define ROWS 9u
#define EXPERTS 3u
#define REPEATS 12u
#define SENTINEL (-1234567.25f)

static unsigned scale_value(unsigned expert, unsigned row, unsigned block, unsigned group) {
    return 1u + (expert * 11u + row * 7u + block * 13u + group * 9u) % 63u;
}
static unsigned min_value(unsigned expert, unsigned row, unsigned block, unsigned group) {
    return (expert * 17u + row * 3u + block * 19u + group * 5u) % 64u;
}
static unsigned quant_value(unsigned expert, unsigned row, unsigned block, unsigned index) {
    return (expert * 3u + row * 5u + block * 7u + index * 11u + index / 32u) % 16u;
}

static void pack_block(unsigned char bytes[144], unsigned expert, unsigned row, unsigned block) {
    unsigned s[8], m[8];
    memset(bytes, 0, 144);
    /* Exactly representable binary16: d=1/8, dmin=1/16. */
    bytes[1] = 0x30; bytes[3] = 0x2c;
    for (unsigned g = 0; g < 8u; ++g) {
        s[g] = scale_value(expert, row, block, g);
        m[g] = min_value(expert, row, block, g);
    }
    for (unsigned g = 0; g < 4u; ++g) {
        bytes[4u + g] = (unsigned char)(s[g] | ((s[g + 4u] >> 4) << 6));
        bytes[8u + g] = (unsigned char)(m[g] | ((m[g + 4u] >> 4) << 6));
        bytes[12u + g] = (unsigned char)((s[g + 4u] & 15u) | ((m[g + 4u] & 15u) << 4));
    }
    for (unsigned i = 0; i < 128u; ++i) {
        unsigned index = (i / 32u) * 64u + i % 32u;
        bytes[16u + i] = (unsigned char)(quant_value(expert, row, block, index) |
            (quant_value(expert, row, block, index + 32u) << 4));
    }
}

/* No production decoder, half conversion, or packed-byte reads in the oracle. */
static float reference_dot(unsigned expert, unsigned row, const float *input, unsigned dims) {
    double sum = 0.0;
    for (unsigned i = 0; i < dims; ++i) {
        unsigned block = i / 256u, index = i % 256u, group = index / 32u;
        float weight = 0.125f * (float)scale_value(expert, row, block, group) *
            (float)quant_value(expert, row, block, index) -
            0.0625f * (float)min_value(expert, row, block, group);
        sum += (double)weight * (double)input[i];
    }
    return (float)sum;
}

static void *cache_alloc(void *context, size_t size) {
    (void)context;
    return malloc(size);
}
static void cache_free(void *context, void *ptr) {
    (void)context;
    free(ptr);
}
static int cache_read(void *context, uint64_t offset, void *dst, uint64_t size, uint64_t *actual) {
    FILE *fp = (FILE *)context;
    *actual = 0;
    if (_fseeki64(fp, (int64_t)offset, SEEK_SET) != 0) return 0;
    *actual = (uint64_t)fread(dst, 1, (size_t)size, fp);
    return *actual == size;
}
static int guarded_equal(const float *actual, const float *golden) {
    return memcmp(actual, golden, (ROWS + 2u) * sizeof(float)) == 0;
}

static int run_case(qx_row_pool *pool, unsigned dims, unsigned rows,
                    unsigned *numeric_failures, unsigned *status_failures, unsigned *serial_calls) {
    qx_file file = {0};
    qx_tensor_dir_entry tensor = {0};
    qx_expert_cache cache = {0};
    qx_expert_cache_ops ops = {0};
    float input[511], serial[ROWS + 2u], parallel[ROWS + 2u], golden[ROWS + 2u];
    unsigned blocks = (dims + 255u) / 256u;
    int result = 0;
    char err[256] = {0};
    file.fp = fopen("moe_rows_q4k.bin", "w+b");
    if (!file.fp) { perror("fixture open"); return 0; }
    /* Nonzero tensor offset and three distinct experts catch slice/address errors. */
    for (unsigned i = 0; i < 37u; ++i) if (fputc(0xa5, file.fp) == EOF) goto cleanup;
    for (unsigned e = 0; e < EXPERTS; ++e)
        for (unsigned r = 0; r < ROWS; ++r)
            for (unsigned b = 0; b < blocks; ++b) {
                unsigned char bytes[144];
                pack_block(bytes, e, r, b);
                if (fwrite(bytes, 1, sizeof(bytes), file.fp) != sizeof(bytes)) goto cleanup;
            }
    if (fflush(file.fp) != 0) goto cleanup;
    tensor.rank = 3; tensor.flags = 12u; tensor.offset = 37u;
    tensor.dims[0] = dims; tensor.dims[1] = ROWS; tensor.dims[2] = EXPERTS;
    tensor.byte_size = (uint64_t)blocks * 144u * ROWS * EXPERTS;
    file.header.file_size = tensor.offset + tensor.byte_size;
    file.io_backend = QX_IO_BUFFERED;
    ops.context = file.fp; ops.read_exact = cache_read;
    ops.alloc = cache_alloc; ops.free = cache_free;
    if (!qx_expert_cache_init(&cache, tensor.byte_size / EXPERTS * 2u, &ops, err, sizeof(err))) {
        fprintf(stderr, "cache init: %s\n", err); goto cleanup;
    }
    for (unsigned cached = 0; cached < 2u; ++cached) {
        qx_expert_cache *active_cache = cached ? &cache : NULL;
        for (unsigned repeat = 0; repeat < REPEATS; ++repeat) {
            unsigned expert = (repeat / 2u) % EXPERTS;
            for (unsigned i = 0; i < dims; ++i)
                input[i] = (float)((int)((i * 17u + repeat * 7u) % 41u) - 20) * 0.03125f;
            for (unsigned r = 0; r < ROWS + 2u; ++r) serial[r] = parallel[r] = golden[r] = SENTINEL;
            for (unsigned r = 0; r < rows; ++r) golden[r + 1u] = reference_dot(expert, r, input, dims);
            int serial_ok = qx_packed_expert_matvec_mode(&file, &tensor, expert, input, dims,
                serial + 1, rows, NULL, active_cache, NULL, err, sizeof(err));
            ++*serial_calls;
            if (!serial_ok) {
                fprintf(stderr, "serial status dims=%u rows=%u cache=%u repeat=%u: %s\n", dims, rows, cached, repeat, err);
                ++*status_failures;
            }
            int parallel_ok = qx_packed_expert_matvec_mode(&file, &tensor, expert, input, dims,
                parallel + 1, rows, NULL, active_cache, pool, err, sizeof(err));
            if (!parallel_ok) {
                if (*status_failures == 0u) fprintf(stderr,
                    "pool status dims=%u rows=%u cache=%u repeat=%u: %s\n", dims, rows, cached, repeat, err);
                ++*status_failures;
            }
            if (!guarded_equal(serial, golden) || !guarded_equal(parallel, golden) ||
                memcmp(serial + 1, parallel + 1, rows * sizeof(float)) != 0) {
                fprintf(stderr, "numeric/sentinel mismatch dims=%u rows=%u cache=%u repeat=%u\n", dims, rows, cached, repeat);
                ++*numeric_failures;
            }
        }
    }
    if (!cache.counters.hits || !cache.counters.evictions) {
        fprintf(stderr, "cache fixture did not exercise hits and evictions\n"); goto cleanup;
    }
    result = 1;
cleanup:
    qx_expert_cache_destroy(&cache);
    if (fclose(file.fp) != 0) result = 0;
    if (remove("moe_rows_q4k.bin") != 0) result = 0;
    return result;
}

int main(int argc, char **argv) {
    static const unsigned dims[] = {1u, 255u, 256u, 511u};
    static const unsigned rows[] = {1u, 3u, 5u, 7u};
    unsigned workers, numeric_failures = 0, status_failures = 0, serial_calls = 0;
    char err[256] = {0};
    qx_row_pool *pool;
    int setup_ok = 1;
    if (argc != 2 || (strcmp(argv[1], "2") && strcmp(argv[1], "3"))) return 2;
    workers = (unsigned)(argv[1][0] - '0');
    pool = qx_row_pool_create(workers, err, sizeof(err));
    if (!pool) { fprintf(stderr, "pool create: %s\n", err); return 2; }
    for (unsigned i = 0; i < sizeof(dims) / sizeof(dims[0]); ++i)
        if (!run_case(pool, dims[i], rows[i], &numeric_failures, &status_failures, &serial_calls)) setup_ok = 0;
    printf("workers=%u serial_calls=%u pool_jobs=%llu numeric_failures=%u status_failures=%u setup_ok=%d\n",
        workers, serial_calls, (unsigned long long)qx_row_pool_jobs(pool), numeric_failures, status_failures, setup_ok);
    int ok = setup_ok && !numeric_failures && !status_failures && serial_calls == 96u &&
        qx_row_pool_jobs(pool) == 96u;
    qx_row_pool_destroy(pool);
    return ok ? 0 : 1;
}
