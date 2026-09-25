#ifndef QX_FORMAT_H
#define QX_FORMAT_H

#include <stdint.h>
#include <stdio.h>
#include "qx_gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QX_MAGIC "QXF1"
#define QX_VERSION 1u
#define QX_NAME_MAX 96u
#define QX_MAX_DIMS 4u
#define QX_ALIGN_BYTES 4096ull

typedef enum qx_model_type {
    QX_MODEL_QWEN3_DENSE = 1,
    QX_MODEL_QWEN3_MOE = 2
} qx_model_type;

typedef enum qx_quant_type {
    QX_QUANT_F16 = 1,
    QX_QUANT_Q4_BLOCK = 2,
    QX_QUANT_Q3_BLOCK = 3,
    QX_QUANT_Q3Q4_MIXED = 4,
    QX_QUANT_Q2_BLOCK = 5
} qx_quant_type;

typedef enum qx_tensor_dtype {
    QX_DTYPE_F16 = 1,
    QX_DTYPE_F32 = 2,
    QX_DTYPE_Q4 = 3,
    QX_DTYPE_Q3 = 4,
    QX_DTYPE_Q2 = 5,
    QX_DTYPE_U8 = 6
} qx_tensor_dtype;

typedef struct qx_model_manifest {
    qx_model_type model_type;
    qx_quant_type quant_type;
    uint32_t layers;
    uint32_t hidden;
    uint32_t intermediate;
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t vocab;
    uint32_t max_ctx;
    uint32_t experts;
    uint32_t experts_per_token;
    uint32_t moe_intermediate;
} qx_model_manifest;

typedef struct qx_header {
    char magic[4];
    uint32_t version;
    uint32_t header_size;
    uint32_t dir_entry_size;
    uint32_t tensor_count;
    uint64_t dir_offset;
    uint64_t data_offset;
    uint64_t file_size;
    uint64_t manifest_checksum;
    qx_model_manifest manifest;
    uint8_t reserved[160];
} qx_header;

typedef struct qx_tensor_dir_entry {
    char name[QX_NAME_MAX];
    uint32_t dtype;
    uint32_t quant;
    uint32_t rank;
    uint64_t dims[QX_MAX_DIMS];
    uint64_t offset;
    uint64_t byte_size;
    uint32_t group_size;
    uint32_t flags;
    uint64_t checksum;
    uint8_t reserved[32];
} qx_tensor_dir_entry;

typedef enum qx_io_backend {
    QX_IO_BUFFERED = 0,
    QX_IO_MMAP = 1
} qx_io_backend;

typedef struct qx_span {
    const unsigned char *data;
    unsigned char *owned_data;
    uint64_t size;
} qx_span;

typedef struct qx_file {
    FILE *fp;
    qx_header header;
    qx_tensor_dir_entry *directory;
    qx_io_backend io_backend;
    const unsigned char *mapped_view;
    void *mapping_handle;
} qx_file;

uint64_t qx_align_u64(uint64_t value, uint64_t alignment);
uint64_t qx_fnv1a64(const void *data, uint64_t len);
const char *qx_model_type_name(qx_model_type t);
const char *qx_quant_type_name(qx_quant_type t);
int qx_parse_quant_type(const char *s, qx_quant_type *out);
int qx_manifest_for_model(const char *model, qx_quant_type quant, qx_model_manifest *out);
uint32_t qx_dense_tensor_count(uint32_t layers);
uint32_t qx_moe_tensor_count(uint32_t layers, uint32_t experts);
int qx_write_metadata_only(const char *path, const qx_model_manifest *manifest, char *err, uint64_t err_len);
int qx_write_tensor_copy_from_gguf(const char *out_path, const char *gguf_path, const qx_model_manifest *manifest, const qx_gguf_tensor_table *table, char *err, uint64_t err_len);
int qx_read_header(FILE *f, qx_header *out, char *err, uint64_t err_len);
int qx_set_io_backend(const char *backend, char *err, uint64_t err_len);
const char *qx_io_backend_name(qx_io_backend backend);
int qx_dump_summary(const char *path, FILE *out, char *err, uint64_t err_len);
int qx_open_file(const char *path, qx_file *out, char *err, uint64_t err_len);
void qx_close_file(qx_file *file);
int qx_acquire_span(qx_file *file, uint64_t offset, uint64_t size, qx_span *out, char *err, uint64_t err_len);
void qx_release_span(qx_span *span);
const qx_tensor_dir_entry *qx_find_tensor(const qx_file *file, const char *name);
int qx_verify_tensor_checksum(qx_file *file, const qx_tensor_dir_entry *tensor, char *err, uint64_t err_len);
int qx_dump_tensor_summary(const char *path, const char *name, FILE *out, char *err, uint64_t err_len);
int qx_verify_all_tensors(const char *path, uint32_t max_tensors, FILE *out, char *err, uint64_t err_len);
int qx_dump_expert_index_summary(const char *path, FILE *out, char *err, uint64_t err_len);
int qx_dump_expert_cache_plan(const char *path, double hot_vram_gib, double hot_ram_gib, FILE *out, char *err, uint64_t err_len);
int qx_dump_expert_slice_summary(const char *path, uint32_t layer, uint32_t expert, FILE *out, char *err, uint64_t err_len);
int qx_dump_expert_load_summary(const char *path, uint32_t layer, uint32_t expert, const char *kind, FILE *out, char *err, uint64_t err_len);
int qx_dump_cache_demo_summary(const char *path, uint32_t slots, const char *sequence, FILE *out, char *err, uint64_t err_len);
int qx_dump_expert_load_benchmark(const char *path, uint32_t iters, const char *kind, FILE *out, char *err, uint64_t err_len);
int qx_dump_cache_run_summary(const char *path, uint32_t slots, const char *sequence, FILE *out, char *err, uint64_t err_len);
int qx_dump_cache_run_expert_summary(const char *path, uint32_t slots, const char *sequence, FILE *out, char *err, uint64_t err_len);
int qx_dump_expert_complete_cache_plan(const char *path, double hot_vram_gib, double hot_ram_gib, uint32_t top_k, FILE *out, char *err, uint64_t err_len);
int qx_dump_runtime_plan(const char *path, uint32_t ctx_tokens, const char *kv_format, double usable_vram_gib, double usable_ram_gib, double hot_vram_gib, double hot_ram_gib, uint32_t top_k, FILE *out, char *err, uint64_t err_len);
int qx_dump_token_embedding_summary(const char *path, uint32_t token_id, FILE *out, char *err, uint64_t err_len);
int qx_dump_forward_schedule(const char *path, uint32_t token_id, uint32_t top_k, FILE *out, char *err, uint64_t err_len);
int qx_dump_quant_block_summary(const char *path, const char *name, uint64_t block_index, FILE *out, char *err, uint64_t err_len);
int qx_dump_matvec_stub_summary(const char *path, const char *name, uint32_t rows, FILE *out, char *err, uint64_t err_len);
int qx_dump_decode_block_summary(const char *path, const char *name, uint64_t block_index, FILE *out, char *err, uint64_t err_len);
int qx_dump_block_dot_summary(const char *path, const char *name, uint64_t block_index, uint32_t seed, FILE *out, char *err, uint64_t err_len);
int qx_dump_matvec_row_summary(const char *path, const char *name, uint64_t start_block, uint32_t blocks, uint32_t seed, FILE *out, char *err, uint64_t err_len);
int qx_dump_expert_row_summary(const char *path, uint32_t layer, uint32_t expert, const char *kind, uint64_t start_block, uint32_t blocks, uint32_t seed, FILE *out, char *err, uint64_t err_len);
int qx_dump_expert_forward_probe_summary(const char *path, uint32_t layer, uint32_t expert, uint64_t start_block, uint32_t blocks, uint32_t seed, FILE *out, char *err, uint64_t err_len);
int qx_dump_router_topk_probe_summary(const char *path, uint32_t layer, uint32_t top_k, uint32_t blocks, uint32_t seed, FILE *out, char *err, uint64_t err_len);
int qx_dump_layer_forward_probe_summary(const char *path, uint32_t layer, uint32_t top_k, uint32_t blocks, uint32_t seed, FILE *out, char *err, uint64_t err_len);
int qx_dump_moe_forward_probe_summary(const char *path, uint32_t layers, uint32_t top_k, uint32_t blocks, uint32_t seed, FILE *out, char *err, uint64_t err_len);
int qx_dump_expert_quant_coverage_summary(const char *path, FILE *out, char *err, uint64_t err_len);
int qx_dump_token_forward_probe_summary(const char *path, uint32_t token_id, uint32_t layers, uint32_t top_k, uint32_t blocks, uint32_t seed, const char *norm_name, int32_t attention_layer, int multihead_attention, uint32_t attention_heads, uint32_t attention_dims, int logits_enabled, uint32_t logits_top_n, int sample_enabled, double temperature, int decode_token, const char *tokens_path, FILE *out, char *err, uint64_t err_len);
int qx_dump_logits_probe_summary(const char *path, double activation, uint32_t top_n, uint32_t scan, uint32_t seed, FILE *out, char *err, uint64_t err_len);
int qx_dump_sampler_probe_summary(const char *path, double activation, uint32_t top_k, uint32_t scan, double temperature, uint32_t seed, FILE *out, char *err, uint64_t err_len);
int qx_dump_tokenizer_probe_summary(const char *path, const char *tokens_path, uint32_t token_id, FILE *out, char *err, uint64_t err_len);
int qx_dump_generate_probe_summary(const char *path, const char *tokens_path, uint32_t prompt_token, uint32_t steps, uint32_t top_k, uint32_t scan, double temperature, uint32_t seed, FILE *out, char *err, uint64_t err_len);
int qx_dump_residual_vector_probe_summary(const char *path, uint32_t token_id, const char *norm_name, uint32_t dims, uint32_t seed, FILE *out, char *err, uint64_t err_len);
int qx_dump_projection_matvec_probe_summary(const char *path, uint32_t layer, uint32_t token_id, uint32_t rows, uint32_t dims, const char *kv_format, int residual_vector, const char *norm_name, uint32_t seed, FILE *out, char *err, uint64_t err_len);
int qx_dump_q8_k_activation_probe_summary(uint32_t values, const char *inject, FILE *out, char *err, uint64_t err_len);
int qx_dump_state_loop_probe_summary(const char *path, const char *tokens_path, uint32_t prompt_token, uint32_t steps, uint32_t layers, uint32_t ctx_tokens, const char *kv_format, const char *activation_format, int real_kv, int projection_matvec, int residual_vector, int residual_carry, int numeric_deltas, int delta_vectors, int attention_output_vector, int causal_attention, int rope_gqa_attention, int full_moe, int final_head, int bench, uint32_t residual_dims, const char *norm_name, uint32_t top_k, uint32_t scan, uint32_t logits_top_n, double temperature, uint32_t seed, const char *residual_dump_dir, uint32_t start_layer, const char *residual_input_path, const char *kv_snapshot_out_path, const char *kv_snapshot_in_path, FILE *out, char *err, uint64_t err_len);
int qx_dump_prompt_state_loop_probe_summary(const char *path, const char *tokens_path, const uint32_t *prompt_tokens, uint32_t prompt_count, uint32_t generation_steps, uint32_t layers, uint32_t ctx_tokens, const char *kv_format, const char *activation_format, const char *scratch_policy, const char *kernel_policy, const char *thread_policy, uint32_t threads, const char *simd_policy, const char *expert_cache_policy, const char *cuda_policy, const char *prefill_gemm_policy, const char *speculative_policy, const char *kv2_policy, const char *sampling_policy, const char *long_context_policy, uint64_t long_context_rss_limit_bytes, uint64_t long_context_kv_quality_checks, uint64_t long_context_soak_seconds, int dequant_profile, int real_kv, int projection_matvec, int residual_vector, int residual_carry, int numeric_deltas, int delta_vectors, int attention_output_vector, int causal_attention, int rope_gqa_attention, int full_moe, int final_head, int bench, uint32_t residual_dims, const char *norm_name, uint32_t top_k, uint32_t scan, uint32_t logits_top_n, double temperature, uint32_t seed, const char *residual_dump_dir, uint32_t start_layer, const char *residual_input_path, const char *kv_snapshot_out_path, const char *kv_snapshot_in_path, FILE *out, char *err, uint64_t err_len);
#define QX_NATIVE_GENERATION_MAX_TOKENS 64u
#define QX_NATIVE_GENERATION_CAPACITY_MAX 4096u
#define QX_NATIVE_GENERATION_OPTIONS_VERSION 2u
#define QX_NATIVE_GENERATION_PROFILE_VERSION 1u
#define QX_NATIVE_GENERATION_PROFILE_V2_VERSION 2u
#define QX_NATIVE_GENERATION_BUFFER_RESULT_VERSION 1u
#define QX_NATIVE_GENERATION_BUFFER_PROFILE_VERSION 2u
#define QX_NATIVE_GENERATION_OPTIONS_V1_SIZE 64u
#define QX_NATIVE_GENERATION_PROFILE_V1_SIZE 656u
#define QX_NATIVE_GENERATION_BUFFER_PROFILE_V1_SIZE 160u

typedef enum qx_native_io_policy {
    QX_NATIVE_IO_BUFFERED = 0,
    QX_NATIVE_IO_MMAP = 1
} qx_native_io_policy;

typedef enum qx_native_scratch_policy {
    QX_NATIVE_SCRATCH_EPHEMERAL = 0,
    QX_NATIVE_SCRATCH_PERSISTENT = 1
} qx_native_scratch_policy;

typedef enum qx_native_kernel_policy {
    QX_NATIVE_KERNEL_BASELINE = 0,
    QX_NATIVE_KERNEL_FUSED_FINAL_HEAD = 1
} qx_native_kernel_policy;

typedef enum qx_native_thread_policy {
    QX_NATIVE_THREAD_SERIAL = 0,
    QX_NATIVE_THREAD_POOL = 1
} qx_native_thread_policy;

typedef enum qx_native_cuda_policy {
    QX_NATIVE_CUDA_NONE = 0,
    QX_NATIVE_CUDA_FINAL_HEAD_F32 = 1
} qx_native_cuda_policy;

/* Shared version 1/version 2 policy layout.  In version 1 the word at offset
 * 28 is reserved and must be zero; version 2 names it cuda_policy.  Call
 * qx_native_generation_options_init before changing fields. */
typedef struct qx_native_generation_options {
    uint32_t struct_size;
    uint32_t version;
    qx_native_io_policy io_backend;
    qx_native_scratch_policy scratch_policy;
    qx_native_kernel_policy kernel_policy;
    qx_native_thread_policy thread_policy;
    uint32_t thread_count;
    qx_native_cuda_policy cuda_policy;
    uint32_t reserved[8];
} qx_native_generation_options;

/* Optional, separately versioned execution provenance.  Logit checksums are
 * FNV-1a64 over the complete F32 vocabulary logits for each sampled step. */
typedef struct qx_native_generation_profile {
    uint32_t struct_size;
    uint32_t version;
    qx_native_io_policy requested_io_backend;
    qx_native_io_policy effective_io_backend;
    qx_native_scratch_policy requested_scratch_policy;
    qx_native_scratch_policy effective_scratch_policy;
    qx_native_kernel_policy requested_kernel_policy;
    qx_native_kernel_policy effective_kernel_policy;
    qx_native_thread_policy requested_thread_policy;
    qx_native_thread_policy effective_thread_policy;
    uint32_t requested_thread_count;
    uint32_t effective_thread_count;
    uint32_t sampled_steps;
    uint32_t workers_used;
    uint64_t scratch_peak_capacity_bytes;
    uint64_t scratch_growth_events;
    uint64_t temporary_blocks_decoded;
    uint64_t temporary_floats_materialized;
    uint64_t temporary_bytes_materialized;
    uint64_t fused_final_head_dot_calls;
    uint64_t baseline_final_head_dot_calls;
    uint64_t final_head_q6_k_blocks;
    uint64_t final_head_parallel_jobs;
    uint64_t final_head_serial_jobs;
    uint64_t final_head_fallback_jobs;
    uint64_t full_logits_checksums[QX_NATIVE_GENERATION_MAX_TOKENS];
} qx_native_generation_profile;

/* Sized fixed-result provenance.  The first 656 bytes preserve the complete
 * legacy fixed profile layout; CUDA provenance is an additive v2 suffix. */
typedef struct qx_native_generation_profile_v2 {
    uint32_t struct_size;
    uint32_t version;
    qx_native_io_policy requested_io_backend;
    qx_native_io_policy effective_io_backend;
    qx_native_scratch_policy requested_scratch_policy;
    qx_native_scratch_policy effective_scratch_policy;
    qx_native_kernel_policy requested_kernel_policy;
    qx_native_kernel_policy effective_kernel_policy;
    qx_native_thread_policy requested_thread_policy;
    qx_native_thread_policy effective_thread_policy;
    uint32_t requested_thread_count;
    uint32_t effective_thread_count;
    uint32_t sampled_steps;
    uint32_t workers_used;
    uint64_t scratch_peak_capacity_bytes;
    uint64_t scratch_growth_events;
    uint64_t temporary_blocks_decoded;
    uint64_t temporary_floats_materialized;
    uint64_t temporary_bytes_materialized;
    uint64_t fused_final_head_dot_calls;
    uint64_t baseline_final_head_dot_calls;
    uint64_t final_head_q6_k_blocks;
    uint64_t final_head_parallel_jobs;
    uint64_t final_head_serial_jobs;
    uint64_t final_head_fallback_jobs;
    uint64_t full_logits_checksums[QX_NATIVE_GENERATION_MAX_TOKENS];
    qx_native_cuda_policy requested_cuda_policy;
    qx_native_cuda_policy effective_cuda_policy;
    char cuda_device_name[128];
    uint32_t cuda_compute_capability_major;
    uint32_t cuda_compute_capability_minor;
    uint64_t cuda_resident_weight_bytes;
    uint64_t cuda_persistent_allocations;
    uint64_t cuda_weight_uploads;
    uint64_t cuda_weight_upload_bytes;
    uint64_t cuda_host_to_device_bytes;
    uint64_t cuda_device_to_host_bytes;
    uint64_t cuda_kernel_launches;
    uint64_t cuda_cpu_fallbacks;
} qx_native_generation_profile_v2;

typedef struct qx_native_generation_result {
    uint32_t token_ids[QX_NATIVE_GENERATION_MAX_TOKENS];
    uint32_t token_count;
    int stopped_on_eos;
    double prefill_seconds;
    double decode_seconds;
} qx_native_generation_result;

/* Caller-owned storage variant.  Initialize struct_size/version and provide
 * token_ids/token_capacity before calling the corresponding entry point. */
typedef struct qx_native_generation_buffer_result {
    uint32_t struct_size;
    uint32_t version;
    uint32_t *token_ids;
    uint32_t token_capacity;
    uint32_t token_count;
    int stopped_on_eos;
    uint32_t first_eos_output_index;
    uint32_t executed_forward_steps;
    uint32_t prompt_forward_steps;
    uint32_t generated_input_forward_steps;
    double prefill_seconds;
    double decode_seconds;
} qx_native_generation_buffer_result;

typedef struct qx_native_generation_buffer_profile {
    uint32_t struct_size;
    uint32_t version;
    qx_native_io_policy requested_io_backend;
    qx_native_io_policy effective_io_backend;
    qx_native_scratch_policy requested_scratch_policy;
    qx_native_scratch_policy effective_scratch_policy;
    qx_native_kernel_policy requested_kernel_policy;
    qx_native_kernel_policy effective_kernel_policy;
    qx_native_thread_policy requested_thread_policy;
    qx_native_thread_policy effective_thread_policy;
    uint32_t requested_thread_count;
    uint32_t effective_thread_count;
    uint32_t sampled_steps;
    uint32_t workers_used;
    uint64_t scratch_peak_capacity_bytes;
    uint64_t scratch_growth_events;
    uint64_t temporary_blocks_decoded;
    uint64_t temporary_floats_materialized;
    uint64_t temporary_bytes_materialized;
    uint64_t fused_final_head_dot_calls;
    uint64_t baseline_final_head_dot_calls;
    uint64_t final_head_q6_k_blocks;
    uint64_t final_head_parallel_jobs;
    uint64_t final_head_serial_jobs;
    uint64_t final_head_fallback_jobs;
    uint64_t *full_logits_checksums;
    uint32_t full_logits_checksums_capacity;
    uint32_t v1_compatibility_padding;
    qx_native_cuda_policy requested_cuda_policy;
    qx_native_cuda_policy effective_cuda_policy;
    char cuda_device_name[128];
    uint32_t cuda_compute_capability_major;
    uint32_t cuda_compute_capability_minor;
    uint64_t cuda_resident_weight_bytes;
    uint64_t cuda_persistent_allocations;
    uint64_t cuda_weight_uploads;
    uint64_t cuda_weight_upload_bytes;
    uint64_t cuda_host_to_device_bytes;
    uint64_t cuda_device_to_host_bytes;
    uint64_t cuda_kernel_launches;
    uint64_t cuda_cpu_fallbacks;
} qx_native_generation_buffer_profile;

/* Greedy native generation with the fixed 48-layer F32/INT8-KV runtime.
 * prompt_count and max_tokens must be non-zero; prompt_count + max_tokens - 1
 * must fit both ctx_tokens and QX_NATIVE_GENERATION_MAX_TOKENS. eos_token_id < 0
 * disables EOS stopping. result is required and is cleared before validation/run. */
int qx_run_native_generation(const char *path, const uint32_t *prompt_tokens, uint32_t prompt_count, uint32_t max_tokens, uint32_t ctx_tokens, int32_t eos_token_id, qx_native_generation_result *result, char *err, uint64_t err_len);
void qx_native_generation_options_init(qx_native_generation_options *options);
void qx_native_generation_profile_v2_init(qx_native_generation_profile_v2 *profile);
int qx_run_native_generation_with_options(const char *path, const uint32_t *prompt_tokens,
    uint32_t prompt_count, uint32_t max_tokens, uint32_t ctx_tokens, int32_t eos_token_id,
    const qx_native_generation_options *options, qx_native_generation_result *result,
    qx_native_generation_profile *profile, char *err, uint64_t err_len);
int qx_run_native_generation_with_options_v2(const char *path, const uint32_t *prompt_tokens,
    uint32_t prompt_count, uint32_t max_tokens, uint32_t ctx_tokens, int32_t eos_token_id,
    const qx_native_generation_options *options, qx_native_generation_result *result,
    qx_native_generation_profile_v2 *profile, char *err, uint64_t err_len);
int qx_run_native_generation_into_with_options(const char *path, const uint32_t *prompt_tokens,
    uint32_t prompt_count, uint32_t max_tokens, uint32_t ctx_tokens, int32_t eos_token_id,
    const qx_native_generation_options *options, qx_native_generation_buffer_result *result,
    qx_native_generation_buffer_profile *profile, char *err, uint64_t err_len);
int qx_dump_rope_gqa_golden_probe_summary(uint32_t tokens, uint32_t q_heads_run, uint32_t seed, FILE *out, char *err, uint64_t err_len);
int qx_dump_real_qkv_golden_probe_summary(const char *path, uint32_t layer, uint32_t token_a, uint32_t token_b, uint32_t q_heads_run, uint32_t seed, int full_moe, FILE *out, char *err, uint64_t err_len);
int qx_dump_attention_stage_probe_summary(const char *path, uint32_t layer, const char *layer_input_path, const char *output_dir, const char *activation_mode, const char *kv_format, FILE *out, char *err, uint64_t err_len);
int qx_dump_moe_stage_probe_summary(const char *path, uint32_t layer, const char *ffn_input_path, const char *output_dir, const char *activation_mode, FILE *out, char *err, uint64_t err_len);
int qx_dump_moe_stage_probe_summary_mode(const char *path, uint32_t layer, const char *ffn_input_path, const char *output_dir, const char *activation_mode, const char *router_precision, FILE *out, char *err, uint64_t err_len);
int qx_dump_final_head_probe_summary(const char *path, const char *residual_path, const char *output_dir, const char *activation_mode, uint32_t top_n, FILE *out, char *err, uint64_t err_len);
int qx_dump_expert_q8_k_dot_probe_summary(const char *path, const char *tensor_name, uint32_t expert, uint32_t row, const char *activation_path, FILE *out, char *err, uint64_t err_len);
int qx_dump_rmsnorm_probe_summary(const char *path, uint32_t token_id, const char *norm_name, uint32_t seed, FILE *out, char *err, uint64_t err_len);
int qx_dump_attention_probe_summary(const char *path, uint32_t layer, uint32_t blocks, uint32_t seed, uint32_t ctx_tokens, const char *kv_format, int cache_write, FILE *out, char *err, uint64_t err_len);
int qx_dump_kv_cache_probe_summary(const char *path, uint32_t ctx_tokens, const char *kv_format, uint32_t token, uint32_t layer, uint32_t head, FILE *out, char *err, uint64_t err_len);
int qx_dump_kv_cache_buffer_probe_summary(const char *path, uint32_t ctx_tokens, const char *kv_format, uint32_t token, uint32_t layer, uint32_t head, uint32_t seed, FILE *out, char *err, uint64_t err_len);
int qx_dump_attention_cache_probe_summary(const char *path, uint32_t ctx_tokens, const char *kv_format, uint32_t layer, uint32_t tokens, uint32_t blocks, uint32_t seed, FILE *out, char *err, uint64_t err_len);
int qx_dump_attention_softmax_probe_summary(const char *path, uint32_t ctx_tokens, const char *kv_format, uint32_t layer, uint32_t tokens, uint32_t blocks, uint32_t seed, FILE *out, char *err, uint64_t err_len);
int qx_dump_attention_vector_probe_summary(const char *path, uint32_t ctx_tokens, const char *kv_format, uint32_t layer, uint32_t tokens, uint32_t dims, uint32_t seed, FILE *out, char *err, uint64_t err_len);
int qx_dump_attention_multihead_probe_summary(const char *path, uint32_t ctx_tokens, const char *kv_format, uint32_t layer, uint32_t tokens, uint32_t heads, uint32_t dims, uint32_t seed, FILE *out, char *err, uint64_t err_len);

#ifdef __cplusplus
}
#endif

#endif
