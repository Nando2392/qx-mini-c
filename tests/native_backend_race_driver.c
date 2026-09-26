#define QX_NATIVE_BACKEND_TEST_HOOKS 1
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <windows.h>

/* Include production so this focused test can drive private deterministic hooks. */
#include "../src/qx_format.c"

#define CHECK(condition, code) do { if (!(condition)) { \
    fprintf(stderr, "check failed at line %d (code %d)\n", __LINE__, (code)); return (code); \
} } while (0)

typedef enum call_kind {
    CALL_FIXED_LEGACY = 1,
    CALL_BUFFER_LEGACY = 2,
    CALL_BUFFER_V3_RESIDENT = 3
} call_kind;

typedef struct call_state {
    call_kind kind;
    const char *path;
    int ok;
    char err[256];
    qx_native_io_policy requested;
    qx_native_io_policy effective;
} call_state;

static HANDLE g_before_open;
static HANDLE g_continue_open;
static volatile LONG g_open_seen;
static qx_io_backend g_open_backend;

static void backend_hook(int event, qx_io_backend backend) {
    if (event == QX_NATIVE_BACKEND_TEST_BEFORE_CHECK) {
        SetEvent(g_before_open);
        WaitForSingleObject(g_continue_open, INFINITE);
    } else if (event == QX_NATIVE_BACKEND_TEST_OPENED) {
        g_open_backend = backend;
        InterlockedExchange(&g_open_seen, 1);
    }
}

static DWORD WINAPI run_native_call(LPVOID opaque) {
    static const uint32_t prompt[] = {9707u};
    call_state *state = (call_state *)opaque;
    qx_native_generation_options options;
    qx_native_generation_options_init(&options);
    options.io_backend = QX_NATIVE_IO_BUFFERED;

    if (state->kind == CALL_FIXED_LEGACY) {
        qx_native_generation_result result;
        qx_native_generation_profile profile;
        memset(&result, 0, sizeof(result));
        memset(&profile, 0, sizeof(profile));
        state->ok = qx_run_native_generation_with_options(state->path, prompt, 1u, 1u, 1u, -1,
            &options, &result, &profile, state->err, sizeof(state->err));
        state->requested = profile.requested_io_backend;
        state->effective = profile.effective_io_backend;
    } else {
        uint32_t token = 0u;
        uint64_t checksum = 0u;
        qx_native_generation_buffer_result result = {0};
        result.struct_size = (uint32_t)sizeof(result);
        result.version = QX_NATIVE_GENERATION_BUFFER_RESULT_VERSION;
        result.token_ids = &token;
        result.token_capacity = 1u;
        if (state->kind == CALL_BUFFER_V3_RESIDENT) {
            qx_native_generation_buffer_profile_v3 profile;
            qx_native_generation_buffer_profile_v3_init(&profile);
            profile.full_logits_checksums = &checksum;
            profile.full_logits_checksums_capacity = 1u;
            options.expert_cache_policy = QX_NATIVE_EXPERT_CACHE_RESIDENT_PACKED;
            options.expert_cache_budget_bytes = 64u;
            state->ok = qx_run_native_generation_into_with_options_v3(state->path, prompt, 1u, 1u, 1u, -1,
                &options, &result, &profile, state->err, sizeof(state->err));
            state->requested = profile.requested_io_backend;
            state->effective = profile.effective_io_backend;
        } else {
            qx_native_generation_buffer_profile profile = {0};
            profile.struct_size = (uint32_t)sizeof(profile);
            profile.version = QX_NATIVE_GENERATION_BUFFER_PROFILE_VERSION;
            profile.full_logits_checksums = &checksum;
            profile.full_logits_checksums_capacity = 1u;
            state->ok = qx_run_native_generation_into_with_options(state->path, prompt, 1u, 1u, 1u, -1,
                &options, &result, &profile, state->err, sizeof(state->err));
            state->requested = profile.requested_io_backend;
            state->effective = profile.effective_io_backend;
        }
    }
    return 0u;
}

static int run_forced_interleaving(call_kind kind, const char *path) {
    call_state state;
    HANDLE worker;
    DWORD wait_result;
    memset(&state, 0, sizeof(state));
    state.kind = kind;
    state.path = path;
    ResetEvent(g_before_open);
    ResetEvent(g_continue_open);
    InterlockedExchange(&g_open_seen, 0);
    g_open_backend = (qx_io_backend)-1;

    CHECK(qx_set_io_backend("buffered", state.err, sizeof(state.err)), 20 + kind);
    worker = CreateThread(NULL, 0u, run_native_call, &state, 0u, NULL);
    CHECK(worker != NULL, 30 + kind);
    wait_result = WaitForSingleObject(g_before_open, 5000u);
    CHECK(wait_result == WAIT_OBJECT_0, 40 + kind);

    CHECK(qx_set_io_backend("mmap", state.err, sizeof(state.err)), 50 + kind);
    SetEvent(g_continue_open);
    wait_result = WaitForSingleObject(worker, 10000u);
    CHECK(wait_result == WAIT_OBJECT_0, 60 + kind);
    CloseHandle(worker);

    CHECK(state.ok == 0, 70 + kind); /* metadata-only model must fail after open */
    CHECK(strstr(state.err, "resident packed expert cache requires buffered I/O") == NULL, 80 + kind);
    CHECK(InterlockedCompareExchange(&g_open_seen, 0, 0) == 1, 90 + kind);
    CHECK(g_open_backend == QX_IO_BUFFERED, 100 + kind);
    CHECK(state.requested == QX_NATIVE_IO_BUFFERED, 110 + kind);
    CHECK(state.effective == QX_NATIVE_IO_BUFFERED, 120 + kind);
    CHECK(qx_requested_io_backend == QX_IO_MMAP, 130 + kind);
    return 0;
}

static int test_public_default_open_and_failure_leave_global_unchanged(const char *path) {
    qx_file file;
    char err[256] = {0};
    CHECK(qx_set_io_backend("mmap", err, sizeof(err)), 200);
    CHECK(qx_open_file(path, &file, err, sizeof(err)), 201);
    CHECK(file.io_backend == QX_IO_MMAP, 202);
    qx_close_file(&file);
    CHECK(qx_requested_io_backend == QX_IO_MMAP, 203);
    CHECK(!qx_open_file("missing-native-backend-race.qxf", &file, err, sizeof(err)), 204);
    CHECK(qx_requested_io_backend == QX_IO_MMAP, 205);
    return 0;
}

int main(void) {
    const char *path = "native-backend-race-metadata.qxf";
    qx_model_manifest manifest;
    char err[256] = {0};
    int rc;

    CHECK(qx_manifest_for_model("qwen3-30b-a3b", QX_QUANT_Q4_BLOCK, &manifest), 1);
    CHECK(qx_write_metadata_only(path, &manifest, err, sizeof(err)), 2);
    g_before_open = CreateEventW(NULL, TRUE, FALSE, NULL);
    g_continue_open = CreateEventW(NULL, TRUE, FALSE, NULL);
    CHECK(g_before_open != NULL && g_continue_open != NULL, 3);
    qx_native_backend_test_hook = backend_hook;

    rc = run_forced_interleaving(CALL_FIXED_LEGACY, path); if (rc) return rc;
    rc = run_forced_interleaving(CALL_BUFFER_LEGACY, path); if (rc) return rc;
    rc = run_forced_interleaving(CALL_BUFFER_V3_RESIDENT, path); if (rc) return rc;
    rc = test_public_default_open_and_failure_leave_global_unchanged(path); if (rc) return rc;

    qx_native_backend_test_hook = NULL;
    CloseHandle(g_before_open);
    CloseHandle(g_continue_open);
    CHECK(remove(path) == 0, 4);
    puts("native backend isolation: pass");
    return 0;
}
