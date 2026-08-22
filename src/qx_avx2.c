#include <stdint.h>

#if defined(_MSC_VER) && defined(_M_X64)
#include <immintrin.h>
#include <intrin.h>

int qx_avx2_fma_supported(void) {
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 1);
    int osxsave = (regs[2] & (1 << 27)) != 0;
    int avx = (regs[2] & (1 << 28)) != 0;
    int fma = (regs[2] & (1 << 12)) != 0;
    if (!osxsave || !avx || !fma) return 0;
    unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6ull) != 0x6ull) return 0;
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;
}

float qx_dot_f32_avx2_fma_256(const float *weights, const float *input) {
    __m256 acc = _mm256_setzero_ps();
    for (uint32_t i = 0; i < 256u; i += 8u) {
        __m256 w = _mm256_loadu_ps(weights + i);
        __m256 x = _mm256_loadu_ps(input + i);
        acc = _mm256_fmadd_ps(w, x, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
#else
int qx_avx2_fma_supported(void) { return 0; }
float qx_dot_f32_avx2_fma_256(const float *weights, const float *input) {
    float total = 0.0f;
    for (uint32_t i = 0; i < 256u; ++i) total += weights[i] * input[i];
    return total;
}
#endif
