#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace qx_fault {

enum class Point {
    none,
    allocation,
    weight_upload,
    activation_upload,
    launch_observation,
    synchronize,
    download,
    incompatible_kernel,
};

static Point point = Point::none;
static unsigned allocation_failure_ordinal = 0;
static unsigned allocation_calls = 0;
static unsigned successful_allocations = 0;
static unsigned successful_frees = 0;
static void *live_allocations[16] = {};
static unsigned live_count = 0;

static void reset_all(Point next = Point::none, unsigned ordinal = 0) {
    point = next;
    allocation_failure_ordinal = ordinal;
    allocation_calls = 0;
    successful_allocations = 0;
    successful_frees = 0;
    live_count = 0;
    std::memset(live_allocations, 0, sizeof(live_allocations));
}

static void set_compute_fault(Point next) {
    point = next;
}

static cudaError_t wrapped_cudaMalloc(void **address, size_t bytes) {
    ++allocation_calls;
    if (point == Point::allocation && allocation_calls == allocation_failure_ordinal) {
        if (address != nullptr) *address = nullptr;
        return cudaErrorMemoryAllocation;
    }
    const cudaError_t status = cudaMalloc(address, bytes);
    if (status == cudaSuccess) {
        if (live_count >= sizeof(live_allocations) / sizeof(live_allocations[0])) {
            (void)cudaFree(*address);
            *address = nullptr;
            return cudaErrorMemoryAllocation;
        }
        live_allocations[live_count++] = *address;
        ++successful_allocations;
    }
    return status;
}

static cudaError_t wrapped_cudaFree(void *address) {
    int tracked = -1;
    for (unsigned i = 0; i < live_count; ++i) {
        if (live_allocations[i] == address) {
            tracked = static_cast<int>(i);
            break;
        }
    }
    const cudaError_t status = cudaFree(address);
    if (status == cudaSuccess && tracked >= 0) {
        live_allocations[static_cast<unsigned>(tracked)] = live_allocations[live_count - 1u];
        live_allocations[live_count - 1u] = nullptr;
        --live_count;
        ++successful_frees;
    }
    return status;
}

static cudaError_t wrapped_cudaMemcpy(void *destination, const void *source, size_t bytes,
                                      cudaMemcpyKind kind) {
    if (point == Point::weight_upload && kind == cudaMemcpyHostToDevice)
        return cudaErrorInvalidValue;
    if (point == Point::activation_upload && kind == cudaMemcpyHostToDevice)
        return cudaErrorInvalidValue;
    if (point == Point::download && kind == cudaMemcpyDeviceToHost)
        return cudaErrorInvalidValue;
    return cudaMemcpy(destination, source, bytes, kind);
}

static cudaError_t wrapped_cudaGetLastError() {
    const cudaError_t observed = cudaGetLastError();
    if (observed != cudaSuccess) return observed;
    if (point == Point::launch_observation) return cudaErrorLaunchFailure;
    return cudaSuccess;
}

static cudaError_t wrapped_cudaDeviceSynchronize() {
    const cudaError_t observed = cudaDeviceSynchronize();
    if (observed != cudaSuccess) return observed;
    if (point == Point::synchronize) return cudaErrorLaunchFailure;
    return cudaSuccess;
}

template <typename Function>
static cudaError_t wrapped_cudaFuncGetAttributes(cudaFuncAttributes *attributes, Function function) {
    if (point == Point::incompatible_kernel) return cudaErrorInvalidDeviceFunction;
    return cudaFuncGetAttributes(attributes, function);
}

}  // namespace qx_fault

#define cudaMalloc qx_fault::wrapped_cudaMalloc
#define cudaMemcpy qx_fault::wrapped_cudaMemcpy
#define cudaGetLastError qx_fault::wrapped_cudaGetLastError
#define cudaDeviceSynchronize qx_fault::wrapped_cudaDeviceSynchronize
#define cudaFuncGetAttributes qx_fault::wrapped_cudaFuncGetAttributes
#define cudaFree qx_fault::wrapped_cudaFree
#include "../src/qx_cuda_final_head.cu"
#undef cudaFree
#undef cudaFuncGetAttributes
#undef cudaDeviceSynchronize
#undef cudaGetLastError
#undef cudaMemcpy
#undef cudaMalloc

namespace {

constexpr uint32_t kHidden = 256u;
constexpr uint32_t kVocab = 2u;
constexpr uint64_t kPackedBytes = 2u * 210u;
constexpr float kSentinel = -918273.5f;
static int passed = 0;

static int fail(const char *test, const char *message) {
    std::fprintf(stderr, "cuda_final_head_fault_driver: %s: %s\n", test, message);
    return 1;
}

static bool bounded_explicit_error(const char *err, size_t capacity, const char *operation) {
    return err != nullptr && capacity > 1u && err[0] != '\0' &&
           std::memchr(err, '\0', capacity) != nullptr && std::strstr(err, operation) != nullptr;
}

static std::vector<unsigned char> weights() {
    std::vector<unsigned char> result(static_cast<size_t>(kPackedBytes), 0u);
    return result;
}

static std::vector<float> activation() {
    std::vector<float> result(kHidden);
    for (uint32_t i = 0; i < kHidden; ++i) result[i] = static_cast<float>(i % 13u) * 0.03125f;
    return result;
}

static int check_balance(const char *test, unsigned expected_allocations) {
    if (qx_fault::successful_allocations != expected_allocations)
        return fail(test, "successful allocation count mismatch");
    if (qx_fault::successful_frees != expected_allocations)
        return fail(test, "successful free count mismatch");
    if (qx_fault::live_count != 0u) return fail(test, "tracked device allocation remains live");
    return 0;
}

static int test_incompatible_kernel() {
    const char *name = "incompatible_kernel_rejected";
    qx_fault::reset_all(qx_fault::Point::incompatible_kernel);
    qx_cuda_final_head_device device;
    std::memset(&device, 0xa5, sizeof(device));
    char err[128];
    std::memset(err, 0x5a, sizeof(err));
    if (qx_cuda_final_head_is_available(&device, err, sizeof(err)) != 0)
        return fail(name, "incompatible kernel was accepted");
    if (!bounded_explicit_error(err, sizeof(err), "kernel is incompatible"))
        return fail(name, "missing bounded explicit incompatibility error");
    const qx_cuda_final_head_device zero{};
    if (std::memcmp(&device, &zero, sizeof(device)) != 0)
        return fail(name, "failed availability probe published device metadata");
    ++passed;
    return 0;
}

static int test_allocation_failure(unsigned ordinal, const char *operation) {
    char name[64];
    std::snprintf(name, sizeof(name), "allocation_%u_cleanup", ordinal);
    qx_fault::reset_all(qx_fault::Point::allocation, ordinal);
    const std::vector<unsigned char> packed = weights();
    qx_cuda_final_head_context *context = reinterpret_cast<qx_cuda_final_head_context *>(uintptr_t{1});
    char err[128];
    std::memset(err, 0x5a, sizeof(err));
    if (qx_cuda_final_head_create_q6_k_f32(packed.data(), packed.size(), kHidden, kVocab,
                                           &context, err, sizeof(err)) != 0)
        return fail(name, "injected allocation failure was accepted");
    if (context != nullptr) return fail(name, "create failure did not clear caller context");
    if (!bounded_explicit_error(err, sizeof(err), operation))
        return fail(name, "missing bounded explicit allocation error");
    if (check_balance(name, ordinal - 1u) != 0) return 1;
    ++passed;
    return 0;
}

static int test_weight_upload_failure() {
    const char *name = "weight_upload_cleanup";
    qx_fault::reset_all(qx_fault::Point::weight_upload);
    const std::vector<unsigned char> packed = weights();
    qx_cuda_final_head_context *context = reinterpret_cast<qx_cuda_final_head_context *>(uintptr_t{1});
    char err[128];
    std::memset(err, 0x5a, sizeof(err));
    if (qx_cuda_final_head_create_q6_k_f32(packed.data(), packed.size(), kHidden, kVocab,
                                           &context, err, sizeof(err)) != 0)
        return fail(name, "injected weight upload failure was accepted");
    if (context != nullptr) return fail(name, "create failure did not clear caller context");
    if (!bounded_explicit_error(err, sizeof(err), "weight upload failed"))
        return fail(name, "missing bounded explicit upload error");
    if (check_balance(name, 3u) != 0) return 1;
    ++passed;
    return 0;
}

static int test_compute_failure(qx_fault::Point point, const char *name, const char *operation,
                                uint64_t expected_launches) {
    qx_fault::reset_all();
    const std::vector<unsigned char> packed = weights();
    const std::vector<float> input = activation();
    qx_cuda_final_head_context *context = nullptr;
    char err[128]{};
    if (!qx_cuda_final_head_create_q6_k_f32(packed.data(), packed.size(), kHidden, kVocab,
                                            &context, err, sizeof(err)))
        return fail(name, err);
    qx_fault::set_compute_fault(point);
    float logits[kVocab] = {kSentinel, kSentinel};
    std::memset(err, 0x5a, sizeof(err));
    if (qx_cuda_final_head_compute_f32(context, input.data(), kHidden, logits, kVocab,
                                       err, sizeof(err)) != 0) {
        qx_cuda_final_head_destroy(&context);
        return fail(name, "injected compute failure was accepted");
    }
    if (logits[0] != kSentinel || logits[1] != kSentinel) {
        qx_cuda_final_head_destroy(&context);
        return fail(name, "failed compute modified caller logits");
    }
    if (!bounded_explicit_error(err, sizeof(err), operation)) {
        qx_cuda_final_head_destroy(&context);
        return fail(name, "missing bounded explicit compute error");
    }
    qx_cuda_final_head_counters counters{};
    char counter_err[128]{};
    if (!qx_cuda_final_head_get_counters(context, &counters, counter_err, sizeof(counter_err))) {
        qx_cuda_final_head_destroy(&context);
        return fail(name, counter_err);
    }
    if (counters.cpu_fallbacks != 0u || counters.kernel_launches != expected_launches) {
        qx_cuda_final_head_destroy(&context);
        return fail(name, "counter evidence indicates fallback or wrong launch count");
    }
    qx_cuda_final_head_destroy(&context);
    if (context != nullptr) return fail(name, "destroy did not clear context");
    if (check_balance(name, 3u) != 0) return 1;
    ++passed;
    return 0;
}

static int test_success_control() {
    const char *name = "real_gpu_success_control";
    qx_fault::reset_all();
    const std::vector<unsigned char> packed = weights();
    const std::vector<float> input = activation();
    qx_cuda_final_head_context *context = nullptr;
    char err[128]{};
    if (!qx_cuda_final_head_create_q6_k_f32(packed.data(), packed.size(), kHidden, kVocab,
                                            &context, err, sizeof(err)))
        return fail(name, err);
    float logits[kVocab] = {kSentinel, kSentinel};
    if (!qx_cuda_final_head_compute_f32(context, input.data(), kHidden, logits, kVocab,
                                        err, sizeof(err))) {
        qx_cuda_final_head_destroy(&context);
        return fail(name, err);
    }
    if (logits[0] != 0.0f || logits[1] != 0.0f)
        return fail(name, "real kernel did not produce zero logits for zero-scale weights");
    qx_cuda_final_head_counters counters{};
    if (!qx_cuda_final_head_get_counters(context, &counters, err, sizeof(err)))
        return fail(name, err);
    if (counters.persistent_allocations != 3u || counters.weight_uploads != 1u ||
        counters.kernel_launches != 1u || counters.cpu_fallbacks != 0u)
        return fail(name, "success counters mismatch or silent fallback observed");
    qx_cuda_final_head_destroy(&context);
    if (check_balance(name, 3u) != 0) return 1;
    ++passed;
    return 0;
}

}  // namespace

int main() {
    qx_cuda_final_head_device device{};
    char availability_error[128]{};
    qx_fault::reset_all();
    if (!qx_cuda_final_head_is_available(&device, availability_error, sizeof(availability_error))) {
        std::fprintf(stderr, "cuda_final_head_fault_driver: real CUDA backend unavailable: %s\n",
                     availability_error);
        return 77;
    }
    if (device.name[0] == '\0') return fail("availability_control", "real device name is empty");

    if (test_incompatible_kernel() != 0) return 1;
    if (test_allocation_failure(1u, "weight allocation failed") != 0) return 1;
    if (test_allocation_failure(2u, "activation allocation failed") != 0) return 1;
    if (test_allocation_failure(3u, "logits allocation failed") != 0) return 1;
    if (test_weight_upload_failure() != 0) return 1;
    if (test_compute_failure(qx_fault::Point::activation_upload, "activation_upload_preserves_output",
                             "activation upload failed", 0u) != 0) return 1;
    if (test_compute_failure(qx_fault::Point::launch_observation, "get_last_error_preserves_output",
                             "kernel failed", 1u) != 0) return 1;
    if (test_compute_failure(qx_fault::Point::synchronize, "synchronize_preserves_output",
                             "kernel failed", 1u) != 0) return 1;
    if (test_compute_failure(qx_fault::Point::download, "download_preserves_output",
                             "logits download failed", 1u) != 0) return 1;
    if (test_success_control() != 0) return 1;

    std::printf("{\"schema\":\"qx.cuda-final-head-faults.v1\",\"status\":\"pass\","
                "\"tests_passed\":%d,\"tests\":["
                "\"incompatible_kernel_rejected\",\"allocation_1_cleanup\","
                "\"allocation_2_cleanup\",\"allocation_3_cleanup\",\"weight_upload_cleanup\","
                "\"activation_upload_preserves_output\",\"get_last_error_preserves_output\","
                "\"synchronize_preserves_output\",\"download_preserves_output\","
                "\"real_gpu_success_control\"],\"real_device\":\"%s\","
                "\"launch_injection\":\"cudaGetLastError return injection after a real kernel launch; not an actually failed device kernel\","
                "\"whole_backend_mocked\":false,\"production_source_included\":true,"
                "\"exact_alloc_free_tracking\":true,\"silent_fallbacks\":0}\n",
                passed, device.name);
    return passed == 10 ? 0 : 1;
}
