#include "qx_format.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    char err[256] = {0};
    qx_file file;
    qx_span span = {0};
    if (argc != 2 && argc != 3) return 2;
    if (!qx_set_io_backend("mmap", err, sizeof(err))) return 3;
    if (!qx_open_file(argv[1], &file, err, sizeof(err))) return 4;
    if (argc == 3 && strcmp(argv[2], "bounds") == 0) {
        int accepted = qx_acquire_span(&file, file.header.file_size, 1u, &span, err, sizeof(err));
        qx_release_span(&span);
        qx_close_file(&file);
        if (accepted || strcmp(err, "QXF span outside file") != 0) return 7;
        printf("{\"rejected\":\"bounds\"}\n");
        return 0;
    }
    if (argc == 3 && strcmp(argv[2], "overflow") == 0) {
        int accepted = qx_acquire_span(&file, UINT64_MAX - 7u, 16u, &span, err, sizeof(err));
        qx_release_span(&span);
        qx_close_file(&file);
        if (accepted || strcmp(err, "QXF span outside file") != 0) return 8;
        printf("{\"rejected\":\"overflow\"}\n");
        return 0;
    }
    if (argc == 3 && strcmp(argv[2], "empty") == 0) {
        int accepted = qx_acquire_span(&file, file.header.data_offset, 0u, &span, err, sizeof(err));
        qx_release_span(&span);
        qx_close_file(&file);
        if (accepted || strcmp(err, "empty QXF span") != 0) return 9;
        printf("{\"rejected\":\"empty\"}\n");
        return 0;
    }
    if (argc == 3 && strcmp(argv[2], "cleanup") == 0) {
        int accepted = qx_acquire_span(&file, file.header.file_size, 1u, &span, err, sizeof(err));
        qx_release_span(&span);
        qx_close_file(&file);
        qx_close_file(&file);
        if (accepted || file.fp || file.directory || file.mapped_view || file.mapping_handle ||
                span.data || span.owned_data || span.size != 0u) return 10;
        printf("{\"cleanup\":true}\n");
        return 0;
    }
    const qx_tensor_dir_entry *tensor = qx_find_tensor(&file, "token_embd.weight");
    if (!tensor || !qx_acquire_span(&file, tensor->offset, tensor->byte_size, &span, err, sizeof(err))) {
        qx_close_file(&file);
        return 5;
    }
    if (!span.data || span.size != tensor->byte_size || span.owned_data != NULL) {
        qx_release_span(&span);
        qx_close_file(&file);
        return 6;
    }
    printf("{\"valid_span\":true,\"size\":%llu}\n", (unsigned long long)span.size);
    qx_release_span(&span);
    qx_close_file(&file);
    return 0;
}
