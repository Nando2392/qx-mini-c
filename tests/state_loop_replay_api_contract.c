#include "qx_format.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    char err[256] = {0};
    int ok = qx_dump_state_loop_probe_summary(
        "missing-model.qxf",
        NULL,
        42u,
        1u,
        2u,
        4u,
        "f16",
        "q8_k_compat",
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        2048u,
        NULL,
        8u,
        64u,
        5u,
        0.0,
        7u,
        NULL,
        1u,
        "missing-residual.f32",
        NULL,
        NULL,
        stdout,
        err,
        sizeof(err));
    if (ok) {
        fputs("invalid replay API arguments unexpectedly succeeded\n", stderr);
        return 1;
    }
    if (strstr(err, "residual vector") == NULL) {
        fprintf(stderr, "unexpected replay API error: %s\n", err);
        return 1;
    }
    return 0;
}
