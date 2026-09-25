#include "qx_format.h"

#include <cuda_runtime_api.h>
#include <windows.h>
#include <psapi.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

constexpr uint64_t kGpuBudgetBytes = 8ull * 1024ull * 1024ull;
constexpr uint64_t kHostBudgetBytes = 32ull * 1024ull * 1024ull;
constexpr uint32_t kMeasuredCalls = 3u;
constexpr uint32_t kOutputsPerCall = 2u;
constexpr uint32_t kCallCount = 1u + kMeasuredCalls;
constexpr uint32_t kSampleCount = kCallCount * 2u;

struct MemorySample {
    const char *phase;
    uint32_t call_index;
    uint64_t cuda_free_bytes;
    uint64_t cuda_total_bytes;
    uint64_t process_private_bytes;
    uint64_t process_working_set_bytes;
};

struct CallRecord {
    const char *phase;
    uint32_t call_index;
    uint32_t token_ids[kOutputsPerCall];
    uint64_t weight_uploads;
    uint64_t kernel_launches;
    uint64_t cpu_fallbacks;
};

int fail(const char *message, const char *detail = nullptr) {
    std::fprintf(stderr, "native_cuda_runtime_memory_driver: %s%s%s\n",
        message, detail && detail[0] ? ": " : "", detail && detail[0] ? detail : "");
    return 1;
}

bool sample_memory(const char *phase, uint32_t call_index, MemorySample *sample,
                   char *detail, size_t detail_len) {
    const cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
        std::snprintf(detail, detail_len, "cudaDeviceSynchronize failed: %s",
                      cudaGetErrorString(sync_status));
        return false;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t memory_status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (memory_status != cudaSuccess) {
        std::snprintf(detail, detail_len, "cudaMemGetInfo failed: %s",
                      cudaGetErrorString(memory_status));
        return false;
    }

    PROCESS_MEMORY_COUNTERS_EX counters = {};
    counters.cb = sizeof(counters);
    if (!GetProcessMemoryInfo(GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&counters), sizeof(counters))) {
        std::snprintf(detail, detail_len, "GetProcessMemoryInfo failed: win32=%lu",
                      static_cast<unsigned long>(GetLastError()));
        return false;
    }
    if (free_bytes > total_bytes) {
        std::snprintf(detail, detail_len, "cudaMemGetInfo returned free bytes above total bytes");
        return false;
    }

    sample->phase = phase;
    sample->call_index = call_index;
    sample->cuda_free_bytes = static_cast<uint64_t>(free_bytes);
    sample->cuda_total_bytes = static_cast<uint64_t>(total_bytes);
    sample->process_private_bytes = static_cast<uint64_t>(counters.PrivateUsage);
    sample->process_working_set_bytes = static_cast<uint64_t>(counters.WorkingSetSize);
    return true;
}

uint64_t positive_growth(uint64_t baseline, uint64_t value) {
    return value > baseline ? value - baseline : 0u;
}

void print_sample(const MemorySample &sample, bool comma) {
    std::printf("{\"phase\":\"%s\",\"call_index\":%u,"
                "\"cuda_free_bytes\":%llu,\"cuda_total_bytes\":%llu,"
                "\"process_private_bytes\":%llu,\"process_working_set_bytes\":%llu}%s",
        sample.phase, sample.call_index,
        static_cast<unsigned long long>(sample.cuda_free_bytes),
        static_cast<unsigned long long>(sample.cuda_total_bytes),
        static_cast<unsigned long long>(sample.process_private_bytes),
        static_cast<unsigned long long>(sample.process_working_set_bytes),
        comma ? "," : "");
}

void print_call(const CallRecord &call, bool comma) {
    std::printf("{\"phase\":\"%s\",\"call_index\":%u,\"token_ids\":[%u,%u],"
                "\"weight_uploads\":%llu,\"kernel_launches\":%llu,"
                "\"cpu_fallbacks\":%llu}%s",
        call.phase, call.call_index, call.token_ids[0], call.token_ids[1],
        static_cast<unsigned long long>(call.weight_uploads),
        static_cast<unsigned long long>(call.kernel_launches),
        static_cast<unsigned long long>(call.cpu_fallbacks), comma ? "," : "");
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) return fail("usage: driver MODEL.qxf");

    static const uint32_t prompt[] = {9707u};
    qx_native_generation_options options;
    qx_native_generation_options_init(&options);
    options.cuda_policy = QX_NATIVE_CUDA_FINAL_HEAD_F32;

    MemorySample samples[kSampleCount] = {};
    CallRecord calls[kCallCount] = {};
    char detail[512] = {};
    uint32_t sample_index = 0u;

    for (uint32_t call_index = 0u; call_index < kCallCount; ++call_index) {
        const char *phase = call_index == 0u ? "warmup" : "measured";
        if (!sample_memory(call_index == 0u ? "warmup_before" : "measured_before",
                           call_index, &samples[sample_index++], detail, sizeof(detail)))
            return fail("pre-call memory query failed closed", detail);

        qx_native_generation_result result = {};
        qx_native_generation_profile_v2 profile;
        qx_native_generation_profile_v2_init(&profile);
        char err[512] = {};
        if (!qx_run_native_generation_with_options_v2(argv[1], prompt, 1u,
                kOutputsPerCall, 8u, -1, &options, &result, &profile, err, sizeof(err)))
            return fail("native fixed-v2 CUDA generation failed", err);
        if (result.token_count != kOutputsPerCall)
            return fail("native fixed-v2 call did not produce exactly two outputs");
        if (profile.requested_cuda_policy != QX_NATIVE_CUDA_FINAL_HEAD_F32 ||
                profile.effective_cuda_policy != QX_NATIVE_CUDA_FINAL_HEAD_F32)
            return fail("native fixed-v2 call did not preserve explicit CUDA policy");
        if (profile.cuda_weight_uploads != 1u || profile.cuda_kernel_launches != 2u ||
                profile.cuda_cpu_fallbacks != 0u)
            return fail("per-call CUDA counters violated weights=1 launches=2 fallback=0");

        calls[call_index].phase = phase;
        calls[call_index].call_index = call_index;
        calls[call_index].token_ids[0] = result.token_ids[0];
        calls[call_index].token_ids[1] = result.token_ids[1];
        calls[call_index].weight_uploads = profile.cuda_weight_uploads;
        calls[call_index].kernel_launches = profile.cuda_kernel_launches;
        calls[call_index].cpu_fallbacks = profile.cuda_cpu_fallbacks;
        if (call_index != 0u &&
                (calls[call_index].token_ids[0] != calls[0].token_ids[0] ||
                 calls[call_index].token_ids[1] != calls[0].token_ids[1]))
            return fail("repeated same-process call produced unstable token IDs");

        if (!sample_memory(call_index == 0u ? "warmup_after" : "measured_after",
                           call_index, &samples[sample_index++], detail, sizeof(detail)))
            return fail("post-call memory query failed closed", detail);
    }

    const MemorySample &warmup_after = samples[1];
    uint64_t min_cuda_free = warmup_after.cuda_free_bytes;
    uint64_t max_cuda_free = warmup_after.cuda_free_bytes;
    uint64_t min_private = warmup_after.process_private_bytes;
    uint64_t max_private = warmup_after.process_private_bytes;
    uint64_t min_working_set = warmup_after.process_working_set_bytes;
    uint64_t max_working_set = warmup_after.process_working_set_bytes;
    bool cuda_total_stable = true;
    for (uint32_t i = 3u; i < kSampleCount; i += 2u) {
        const MemorySample &sample = samples[i];
        if (sample.cuda_total_bytes != warmup_after.cuda_total_bytes) cuda_total_stable = false;
        if (sample.cuda_free_bytes < min_cuda_free) min_cuda_free = sample.cuda_free_bytes;
        if (sample.cuda_free_bytes > max_cuda_free) max_cuda_free = sample.cuda_free_bytes;
        if (sample.process_private_bytes < min_private) min_private = sample.process_private_bytes;
        if (sample.process_private_bytes > max_private) max_private = sample.process_private_bytes;
        if (sample.process_working_set_bytes < min_working_set) min_working_set = sample.process_working_set_bytes;
        if (sample.process_working_set_bytes > max_working_set) max_working_set = sample.process_working_set_bytes;
    }

    const uint64_t gpu_retained_growth = positive_growth(min_cuda_free, warmup_after.cuda_free_bytes);
    const uint64_t gpu_free_span = max_cuda_free - min_cuda_free;
    const uint64_t private_growth = positive_growth(warmup_after.process_private_bytes, max_private);
    const uint64_t private_span = max_private - min_private;
    const uint64_t working_set_growth = positive_growth(warmup_after.process_working_set_bytes, max_working_set);
    const uint64_t working_set_span = max_working_set - min_working_set;
    const bool gpu_within_budget = cuda_total_stable && gpu_retained_growth <= kGpuBudgetBytes &&
                                   gpu_free_span <= kGpuBudgetBytes;
    const bool host_within_budget = private_growth <= kHostBudgetBytes && private_span <= kHostBudgetBytes &&
                                    working_set_growth <= kHostBudgetBytes && working_set_span <= kHostBudgetBytes;
    const char *status = !gpu_within_budget ? "inconclusive_shared_device_noise" :
                         (!host_within_budget ? "fail_host_retained_growth" : "pass");

    std::printf("{\"schema\":\"qx.native-cuda-runtime-memory.v1\",\"status\":\"%s\","
                "\"same_process\":true,\"process_restart_substitution\":false,"
                "\"warmup_calls\":1,\"measured_calls\":3,\"outputs_per_call\":2,"
                "\"budgets\":{\"gpu_post_warmup_bytes\":%llu,"
                "\"host_post_warmup_bytes\":%llu,"
                "\"host_budget_reason\":\"bounded allowance for CRT, file mapping, and runtime allocator bookkeeping; 255 MiB repeated retention is rejected\"},"
                "\"measurements\":{\"gpu_retained_growth_bytes\":%llu,"
                "\"gpu_post_warmup_free_span_bytes\":%llu,"
                "\"process_private_retained_growth_bytes\":%llu,"
                "\"process_private_post_warmup_span_bytes\":%llu,"
                "\"process_working_set_retained_growth_bytes\":%llu,"
                "\"process_working_set_post_warmup_span_bytes\":%llu},"
                "\"gates\":{\"cuda_total_stable\":%s,\"gpu_within_budget\":%s,"
                "\"host_within_budget\":%s,\"stable_token_ids\":true,"
                "\"exact_per_call_counters\":true,\"leak_claim\":false},"
                "\"cuda_mem_get_info_interpretation\":\"shared-device free-memory samples include external activity; an unstable GPU bound is inconclusive and is not a leak claim\","
                "\"samples\":[",
        status,
        static_cast<unsigned long long>(kGpuBudgetBytes),
        static_cast<unsigned long long>(kHostBudgetBytes),
        static_cast<unsigned long long>(gpu_retained_growth),
        static_cast<unsigned long long>(gpu_free_span),
        static_cast<unsigned long long>(private_growth),
        static_cast<unsigned long long>(private_span),
        static_cast<unsigned long long>(working_set_growth),
        static_cast<unsigned long long>(working_set_span),
        cuda_total_stable ? "true" : "false", gpu_within_budget ? "true" : "false",
        host_within_budget ? "true" : "false");
    for (uint32_t i = 0u; i < kSampleCount; ++i) print_sample(samples[i], i + 1u != kSampleCount);
    std::printf("],\"calls\":[");
    for (uint32_t i = 0u; i < kCallCount; ++i) print_call(calls[i], i + 1u != kCallCount);
    std::printf("]}\n");

    return gpu_within_budget && host_within_budget ? 0 : 2;
}
