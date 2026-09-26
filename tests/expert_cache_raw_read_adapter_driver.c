#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Include production so this isolated test can exercise the private adapter. */
#include "../src/qx_format.c"

#define CHECK(condition, code) do { if (!(condition)) return (code); } while (0)

int main(void) {
    static const unsigned char source[] = {0x11u, 0x22u, 0x33u};
    const char *path = "expert-cache-truncated-raw-read.bin";
    FILE *writer = fopen(path, "wb");
    qx_file file;
    qx_expert_cache cache;
    qx_expert_cache_ops ops;
    qx_expert_cache_borrow borrow = {0};
    qx_expert_cache_counters counters = {0};
    qx_expert_cache_key key = {0u, 0u, 4u, 7u, 0u};
    char err[128] = {0};
    int acquire_ok;

    CHECK(writer != NULL, 10);
    CHECK(fwrite(source, 1u, sizeof(source), writer) == sizeof(source), 11);
    CHECK(fclose(writer) == 0, 12);

    memset(&file, 0, sizeof(file));
    file.fp = fopen(path, "rb");
    CHECK(file.fp != NULL, 13);
    file.header.file_size = 4u; /* Advertise the requested span; physical file is truncated. */
    file.io_backend = QX_IO_BUFFERED;

    ops.context = &file;
    ops.alloc = qx_expert_cache_alloc_callback;
    ops.free = qx_expert_cache_free_callback;
    ops.read_exact = qx_expert_cache_read_callback;
    CHECK(qx_expert_cache_init(&cache, 4u, &ops, err, sizeof(err)), 14);

    acquire_ok = qx_expert_cache_acquire(&cache, key, &borrow, err, sizeof(err));
    qx_expert_cache_snapshot(&cache, &counters);

    CHECK(!acquire_ok, 20);
    CHECK(strcmp(err, "expert cache read failed") == 0, 21);
    CHECK(counters.requests == 1u && counters.misses == 1u && counters.loads == 0u, 22);
    CHECK(counters.buffered_bytes_read == 3u, 23);
    CHECK(counters.current_resident_packed_bytes == 0u, 24);
    CHECK(borrow.data == NULL && borrow.entry == NULL, 25);

    qx_expert_cache_destroy(&cache);
    CHECK(fclose(file.fp) == 0, 26);
    CHECK(remove(path) == 0, 27);
    printf("result=failure read=%llu loads=%llu resident=%llu borrow=null\n",
        (unsigned long long)counters.buffered_bytes_read,
        (unsigned long long)counters.loads,
        (unsigned long long)counters.current_resident_packed_bytes);
    return 0;
}
