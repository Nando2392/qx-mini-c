from __future__ import annotations

import hashlib
import importlib.util
import math
from types import SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "q8k_perf_experiment.py"
SPEC = importlib.util.spec_from_file_location("q8k_perf_experiment", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PERF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PERF)


def native_payload(
    *, activation: str, selected: tuple[int, ...], checksum_bias: int = 0,
    io_backend: str = "buffered",
    scratch_policy: str = "ephemeral",
    kernel_policy: str = "baseline",
    thread_policy: str = "serial",
    threads: int = 1,
    simd_policy: str = "scalar",
    expert_cache_policy: str = "none",
    cuda_policy: str = "none",
    prefill_gemm_policy: str = "none",
    speculative_policy: str = "none",
    kv2_policy: str = "none",
    sampling_policy: str = "none",
    long_context_policy: str = "none",
) -> dict:
    tokens = [
        {"phase": "prefill", "input_token": 9707, "selected_token": None},
        {
            "phase": "generate",
            "input_token": 0,
            "selected_token": selected[0],
            "final_head": {
                "final_residual_checksum": 100 + checksum_bias,
                "final_norm_checksum": 200 + checksum_bias,
                "logits_checksum": 300 + checksum_bias,
                "argmax_token": selected[0],
            },
        },
        {
            "phase": "generate",
            "input_token": selected[0],
            "selected_token": selected[1],
            "final_head": {
                "final_residual_checksum": 101 + checksum_bias,
                "final_norm_checksum": 201 + checksum_bias,
                "logits_checksum": 301 + checksum_bias,
                "argmax_token": selected[1],
            },
        },
    ]
    return {
        "probe": "state_loop",
        "prompt_token_ids": [9707, 0],
        "prompt_token_count": 2,
        "generation_steps": 2,
        "ctx_tokens": 16,
        "kv_format": "int8",
        "activation_format": activation,
        "io_backend": io_backend,
        "scratch_policy": scratch_policy,
        "kernel_policy": kernel_policy,
        "thread_policy": thread_policy,
        "threads": threads,
        "simd_policy": simd_policy,
        "expert_cache_policy": expert_cache_policy,
        "cuda_policy": cuda_policy,
        "prefill_gemm_policy": prefill_gemm_policy,
        "speculative_policy": speculative_policy,
        "kv2_policy": kv2_policy,
        "sampling_policy": sampling_policy,
        "long_context_policy": long_context_policy,
        "layers": 48,
        "cache_readback_ok": True,
        "tokens": tokens,
        "final_token": selected[-1],
        "k_cache_checksum": 400 + checksum_bias,
        "v_cache_checksum": 500 + checksum_bias,
        "bench": {
            "enabled": True,
            "elapsed_sec": 3.0,
            "tokens_per_second": 1.0,
            "ms_per_token": 1000.0,
            "layer_steps": 144,
            "layer_steps_per_second": 48.0,
            "phases": {
                "prefill": {"tokens": 1, "elapsed_sec": 1.0, "tokens_per_second": 1.0},
                "decode": {"tokens": 2, "elapsed_sec": 2.0, "tokens_per_second": 1.0},
            },
        },
        "allocation_profile": {
            "malloc_calls": 10,
            "calloc_calls": 2,
            "realloc_calls": 0,
            "free_calls": 12,
            "bytes_requested": 4096,
        },
        "scratch_profile": {
            "policy": scratch_policy,
            "peak_capacity_bytes": 0 if scratch_policy == "ephemeral" else 65536,
            "growth_events": 0 if scratch_policy == "ephemeral" else 1,
        },
        "dequant_dot_profile": {
            "enabled": True,
            "kernel_policy": kernel_policy,
            "temporary_blocks_decoded": 1215488 if kernel_policy == "baseline" and activation == "f32" else 0,
            "temporary_floats_materialized": 311164928 if kernel_policy == "baseline" and activation == "f32" else 0,
            "temporary_bytes_materialized": 1244659712 if kernel_policy == "baseline" and activation == "f32" else 0,
            "fused_dot_calls": 0 if kernel_policy == "baseline" and activation == "f32" else 1215488,
            "fallback_dot_calls": 1215488 if kernel_policy == "baseline" and activation == "f32" else 0,
            "final_head_q6_k_blocks": 1215488,
        },
        "thread_profile": {
            "enabled": True,
            "policy": thread_policy,
            "requested_threads": threads,
            "workers_used": 1,
            "parallel_jobs": 0,
            "serial_jobs": 96,
            "fallback_jobs": 96,
            "disabled_reason": "serial_policy",
        },
        "simd_profile": {
            "enabled": True,
            "policy": simd_policy,
            "kernel": "avx2_fma_q6_k_f32" if simd_policy == "avx2-fma" else "scalar",
            "fma_dot_calls": 1215488 if simd_policy == "avx2-fma" else 0,
            "fallback_dot_calls": 0 if simd_policy == "avx2-fma" else 1215488,
            "disabled_reason": "scalar_policy" if simd_policy == "scalar" else None,
        },
        "expert_cache_profile": {
            "enabled": True,
            "policy": expert_cache_policy,
            "cache_hits": 0,
            "cache_misses": 0,
            "bytes_cached": 0,
            "expert_weight_reads": 0,
            "disabled_reason": "none_policy" if expert_cache_policy == "none" else None,
        },
        "cuda_profile": {
            "enabled": True,
            "policy": cuda_policy,
            "backend": "none",
            "device_bytes": 0,
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 0,
            "kernel_launches": 0,
            "disabled_reason": "none_policy" if cuda_policy == "none" else None,
        },
        "prefill_gemm_profile": {
            "enabled": True,
            "policy": prefill_gemm_policy,
            "backend": "none",
            "gemm_calls": 0,
            "batched_tokens": 0,
            "fused_rows": 0,
            "temporary_bytes": 0,
            "disabled_reason": "none_policy" if prefill_gemm_policy == "none" else None,
        },
        "speculative_profile": {
            "enabled": True,
            "policy": speculative_policy,
            "backend": "none",
            "draft_tokens": 0,
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "target_verifications": 0,
            "disabled_reason": "none_policy" if speculative_policy == "none" else None,
        },
        "kv2_profile": {
            "enabled": True,
            "policy": kv2_policy,
            "format": "none",
            "packed_bytes": 0,
            "read_ops": 0,
            "write_ops": 0,
            "fallback_reads": 0,
            "disabled_reason": "none_policy" if kv2_policy == "none" else None,
        },
        "sampling_profile": {
            "enabled": True,
            "policy": sampling_policy,
            "mode": "greedy",
            "stochastic_samples": 0,
            "top_p_evaluations": 0,
            "beam_width": 1,
            "disabled_reason": "none_policy" if sampling_policy == "none" else None,
        },
        "long_context_profile": {
            "enabled": True,
            "policy": long_context_policy,
            "target_ctx_tokens": 4096 if long_context_policy == "ctx4k-smoke" else 0,
            "rss_limit_bytes": 0,
            "kv_quality_checks": 0,
            "soak_seconds": 0,
            "disabled_reason": "none_policy" if long_context_policy == "none" else None,
        },
    }


def test_summarize_is_fail_closed_and_reports_dispersion():
    summary = PERF.summarize([1.0, 2.0, 3.0])
    assert summary == {
        "count": 3,
        "min": 1.0,
        "median": 2.0,
        "max": 3.0,
        "mad": 1.0,
        "stdev": 1.0,
        "pstdev": pytest.approx(math.sqrt(2.0 / 3.0)),
    }
    for invalid in ([], [0.0, 1.0, 2.0], [1.0, float("nan"), 2.0]):
        with pytest.raises(ValueError):
            PERF.summarize(invalid)


def test_source_state_records_dirty_diff_digest(monkeypatch):
    revision = "a" * 40
    diff = b"binary diff\x00payload"
    results = iter([
        SimpleNamespace(stdout=revision + "\n"),
        SimpleNamespace(stdout=diff),
    ])
    monkeypatch.setattr(PERF.subprocess, "run", lambda *args, **kwargs: next(results))

    assert PERF.source_state() == {
        "revision": revision,
        "working_tree_dirty": True,
        "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def test_extract_output_signature_requires_complete_final_head_evidence():
    payload = native_payload(activation="f32", selected=(358, 1184))
    signature = PERF.extract_output_signature(payload)
    assert signature["prompt_token_ids"] == [9707, 0]
    assert signature["selected_tokens"] == [358, 1184]
    assert signature["logits_checksums"] == [300, 301]
    assert signature["cache_readback_ok"] is True

    del payload["tokens"][1]["final_head"]["logits_checksum"]
    with pytest.raises(ValueError, match="logits_checksum"):
        PERF.extract_output_signature(payload)


def test_validate_native_payload_rejects_argument_or_phase_drift():
    payload = native_payload(activation="q8_k_compat", selected=(358, 1184))
    signature = PERF.validate_native_payload(
        payload,
        activation="q8_k_compat",
        io_backend="buffered",
        scratch_policy="ephemeral",
        kv="int8",
        layers=48,
        ctx=16,
        generation_steps=2,
    )
    assert signature["selected_tokens"] == [358, 1184]

    payload["ctx_tokens"] = 17
    with pytest.raises(ValueError, match="ctx_tokens"):
        PERF.validate_native_payload(
            payload,
            activation="q8_k_compat",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )

    wrong_prefill = native_payload(activation="q8_k_compat", selected=(358, 1184))
    wrong_prefill["bench"]["phases"]["prefill"]["tokens"] = 2
    with pytest.raises(ValueError, match="prefill.tokens"):
        PERF.validate_native_payload(
            wrong_prefill,
            activation="q8_k_compat",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )

    wrong_backend = native_payload(activation="q8_k_compat", selected=(358, 1184))
    wrong_backend["io_backend"] = "mmap"
    with pytest.raises(ValueError, match="io_backend"):
        PERF.validate_native_payload(
            wrong_backend,
            activation="q8_k_compat",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )

    missing_selected = native_payload(activation="q8_k_compat", selected=(358, 1184))
    missing_selected["tokens"][2]["selected_token"] = None
    with pytest.raises(ValueError, match="selected-token count"):
        PERF.validate_native_payload(
            missing_selected,
            activation="q8_k_compat",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )


def test_output_gate_requires_buffered_mmap_and_scratch_equivalence_within_each_activation():
    f32 = PERF.extract_output_signature(native_payload(activation="f32", selected=(358, 1184)))
    f32_mmap = PERF.extract_output_signature(
        native_payload(activation="f32", selected=(358, 1184), io_backend="mmap")
    )
    q8k = PERF.extract_output_signature(
        native_payload(activation="q8_k_compat", selected=(358, 1184), checksum_bias=10)
    )
    q8k_mmap = PERF.extract_output_signature(
        native_payload(activation="q8_k_compat", selected=(358, 1184), checksum_bias=10, io_backend="mmap")
    )
    cells = {}
    for policy in ("ephemeral", "persistent"):
        cells[f"{policy}:buffered:f32"] = [f32, f32]
        cells[f"{policy}:mmap:f32"] = [f32_mmap, f32_mmap]
        cells[f"{policy}:buffered:q8_k_compat"] = [q8k, q8k]
        cells[f"{policy}:mmap:q8_k_compat"] = [q8k_mmap, q8k_mmap]
    gate = PERF.validate_output_contract(cells)
    assert gate["status"] == "pass"
    assert gate["io_equivalence"] == {"f32": True, "q8_k_compat": True}
    assert gate["activation_numeric_checksums_equal"] is False

    changed = PERF.extract_output_signature(
        native_payload(activation="q8_k_compat", selected=(358, 999), checksum_bias=20)
    )
    with pytest.raises(ValueError, match="not deterministic"):
        PERF.validate_output_contract({**cells, "ephemeral:buffered:q8_k_compat": [q8k, changed]})
    with pytest.raises(ValueError, match="buffered/mmap output mismatch"):
        PERF.validate_output_contract({**cells, "ephemeral:mmap:f32": [changed]})
    with pytest.raises(ValueError, match="ephemeral/persistent output mismatch"):
        PERF.validate_output_contract({
            **cells,
            "persistent:buffered:f32": [changed],
            "persistent:mmap:f32": [changed],
        })


def test_build_inference_command_fixes_real_prompt_and_modal_arguments(tmp_path):
    command = PERF.build_inference_command(
        qx_exe=tmp_path / "qxqxf.exe",
        model=tmp_path / "model.qxf",
        tokenizer=tmp_path / "model.qxt",
        prompt_file=tmp_path / "prompt.txt",
        activation="f32",
        kv="int8",
        layers=48,
        ctx=16,
        generation_steps=2,
        seed=7,
        io_backend="mmap",
        scratch_policy="persistent",
        kernel_policy="fused",
        thread_policy="serial",
        threads=1,
    )
    assert command[1] == "prompt-state-loop-probe"
    assert "--full-moe" in command
    assert "--final-head" in command
    assert "--bench" in command
    assert command[command.index("--temperature") + 1] == "0"
    assert command[command.index("--io-backend") + 1] == "mmap"
    assert command[command.index("--scratch-policy") + 1] == "persistent"
    assert command[command.index("--kernel-policy") + 1] == "fused"
    assert command[command.index("--thread-policy") + 1] == "serial"
    assert command[command.index("--threads") + 1] == "1"
    assert command[command.index("--long-context-kv-quality-checks") + 1] == "0"
    assert command[command.index("--long-context-soak-seconds") + 1] == "0"
    assert "--dequant-profile" in command


def test_validate_native_payload_requires_dequant_dot_profile():
    payload = native_payload(activation="f32", selected=(358, 1184))
    del payload["dequant_dot_profile"]
    with pytest.raises(ValueError, match="dequant_dot_profile"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="baseline",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )


def test_validate_native_payload_requires_thread_profile():
    payload = native_payload(activation="f32", selected=(358, 1184))
    del payload["thread_profile"]
    with pytest.raises(ValueError, match="thread_profile"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="baseline",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )


def test_validate_native_payload_rejects_thread_policy_drift():
    payload = native_payload(activation="f32", selected=(358, 1184), thread_policy="serial", threads=1)
    payload["thread_profile"]["policy"] = "pool"
    with pytest.raises(ValueError, match="thread_profile.policy"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="baseline",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )


def test_validate_native_payload_rejects_thread_count_drift():
    payload = native_payload(activation="f32", selected=(358, 1184), thread_policy="serial", threads=1)
    payload["thread_profile"]["requested_threads"] = 2
    with pytest.raises(ValueError, match="thread_profile.requested_threads"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="baseline",
            thread_policy="serial",
            threads=1,
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )


def test_validate_native_payload_requires_real_pool_profile():
    payload = native_payload(activation="f32", selected=(358, 1184), thread_policy="pool", threads=4)
    with pytest.raises(ValueError, match="thread_profile.workers_used"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="baseline",
            thread_policy="pool",
            threads=4,
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )

    payload["thread_profile"] = {
        "enabled": True,
        "policy": "pool",
        "requested_threads": 4,
        "workers_used": 4,
        "parallel_jobs": 96,
        "serial_jobs": 1,
        "fallback_jobs": 0,
    }
    with pytest.raises(ValueError, match="thread_profile.serial_jobs"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="baseline",
            thread_policy="pool",
            threads=4,
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )

    payload["thread_profile"]["serial_jobs"] = 0
    signature = PERF.validate_native_payload(
        payload,
        activation="f32",
        io_backend="buffered",
        scratch_policy="ephemeral",
        kernel_policy="baseline",
        thread_policy="pool",
        threads=4,
        kv="int8",
        layers=48,
        ctx=16,
        generation_steps=2,
    )
    assert signature["selected_tokens"] == [358, 1184]


def test_compare_kernel_policy_requires_fused_to_reduce_f32_temporaries():
    baseline = PERF.compact_run(
        {"wall_elapsed_seconds": 3.0, "peak_rss_bytes": 4096, "payload": native_payload(activation="f32", selected=(358, 1184), kernel_policy="baseline")},
        activation="f32",
        io_backend="buffered",
        kv="int8",
        layers=48,
        ctx=16,
        generation_steps=2,
        scratch_policy="ephemeral",
        kernel_policy="baseline",
    )
    fused = PERF.compact_run(
        {"wall_elapsed_seconds": 3.0, "peak_rss_bytes": 4096, "payload": native_payload(activation="f32", selected=(358, 1184), kernel_policy="fused")},
        activation="f32",
        io_backend="buffered",
        kv="int8",
        layers=48,
        ctx=16,
        generation_steps=2,
        scratch_policy="ephemeral",
        kernel_policy="fused",
    )

    gate = PERF.compare_kernel_policy_effects({"baseline": [baseline, baseline, baseline], "fused": [fused, fused, fused]})
    assert gate["status"] == "pass"
    assert gate["temporary_bytes_materialized_reduced"] is True
    assert gate["fused_dot_calls_increased"] is True

    with pytest.raises(ValueError, match="did not reduce"):
        PERF.compare_kernel_policy_effects({"baseline": [baseline, baseline, baseline], "fused": [baseline, baseline, baseline]})


def test_validate_native_payload_requires_avx2_fma_profile():
    payload = native_payload(activation="f32", selected=(358, 1184), kernel_policy="fused", simd_policy="avx2-fma")
    payload["simd_profile"]["fma_dot_calls"] = 0
    with pytest.raises(ValueError, match="simd_profile.fma_dot_calls"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="fused",
            thread_policy="serial",
            threads=1,
            simd_policy="avx2-fma",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )

    payload["simd_profile"]["fma_dot_calls"] = 1215488
    signature = PERF.validate_native_payload(
        payload,
        activation="f32",
        io_backend="buffered",
        scratch_policy="ephemeral",
        kernel_policy="fused",
        thread_policy="serial",
        threads=1,
        simd_policy="avx2-fma",
        kv="int8",
        layers=48,
        ctx=16,
        generation_steps=2,
    )
    assert signature["selected_tokens"] == [358, 1184]


def test_validate_native_payload_rejects_scratch_policy_drift():
    payload = native_payload(activation="f32", selected=(358, 1184), scratch_policy="persistent")
    signature = PERF.validate_native_payload(
        payload,
        activation="f32",
        io_backend="buffered",
        scratch_policy="persistent",
        kv="int8",
        layers=48,
        ctx=16,
        generation_steps=2,
    )
    assert signature["selected_tokens"] == [358, 1184]

    with pytest.raises(ValueError, match="scratch_policy"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )


def test_validate_native_payload_requires_expert_cache_profile():
    payload = native_payload(activation="f32", selected=(358, 1184))
    del payload["expert_cache_profile"]
    with pytest.raises(ValueError, match="expert_cache_profile"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="baseline",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )


def test_validate_native_payload_rejects_fake_expert_cache_profile():
    payload = native_payload(activation="f32", selected=(358, 1184), expert_cache_policy="none")
    payload["expert_cache_profile"]["cache_hits"] = 1
    with pytest.raises(ValueError, match="expert_cache_profile.cache_hits"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="baseline",
            expert_cache_policy="none",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )


def test_validate_native_payload_requires_cuda_profile():
    payload = native_payload(activation="f32", selected=(358, 1184))
    del payload["cuda_profile"]
    with pytest.raises(ValueError, match="cuda_profile"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="baseline",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )


def test_validate_native_payload_rejects_fake_cuda_profile():
    payload = native_payload(activation="f32", selected=(358, 1184), cuda_policy="none")
    payload["cuda_profile"]["kernel_launches"] = 1
    with pytest.raises(ValueError, match="cuda_profile.kernel_launches"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="baseline",
            cuda_policy="none",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )


def test_validate_native_payload_requires_prefill_gemm_profile():
    payload = native_payload(activation="f32", selected=(358, 1184))
    del payload["prefill_gemm_profile"]
    with pytest.raises(ValueError, match="prefill_gemm_profile"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="baseline",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )


def test_validate_native_payload_rejects_fake_prefill_gemm_profile():
    payload = native_payload(activation="f32", selected=(358, 1184), prefill_gemm_policy="none")
    payload["prefill_gemm_profile"]["gemm_calls"] = 1
    with pytest.raises(ValueError, match="prefill_gemm_profile.gemm_calls"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="baseline",
            prefill_gemm_policy="none",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )


def test_validate_native_payload_requires_speculative_and_kv2_profiles():
    payload = native_payload(activation="f32", selected=(358, 1184))
    del payload["speculative_profile"]
    del payload["kv2_profile"]
    with pytest.raises(ValueError, match="speculative_profile"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="baseline",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )


def test_validate_native_payload_rejects_fake_speculative_and_kv2_profiles():
    payload = native_payload(activation="f32", selected=(358, 1184))
    payload["speculative_profile"]["draft_tokens"] = 1
    with pytest.raises(ValueError, match="speculative_profile.draft_tokens"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="baseline",
            speculative_policy="none",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )
    payload = native_payload(activation="f32", selected=(358, 1184))
    payload["kv2_profile"]["packed_bytes"] = 1
    with pytest.raises(ValueError, match="kv2_profile.packed_bytes"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="baseline",
            kv2_policy="none",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )


def test_validate_native_payload_requires_sampling_profile():
    payload = native_payload(activation="f32", selected=(358, 1184))
    del payload["sampling_profile"]
    with pytest.raises(ValueError, match="sampling_profile"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="baseline",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )


def test_validate_native_payload_rejects_fake_sampling_profile():
    payload = native_payload(activation="f32", selected=(358, 1184))
    payload["sampling_profile"]["stochastic_samples"] = 1
    with pytest.raises(ValueError, match="sampling_profile.stochastic_samples"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="baseline",
            sampling_policy="none",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )


def test_validate_native_payload_requires_long_context_profile():
    payload = native_payload(activation="f32", selected=(358, 1184))
    del payload["long_context_profile"]
    with pytest.raises(ValueError, match="long_context_profile"):
        PERF.validate_native_payload(
            payload, activation="f32", io_backend="buffered", scratch_policy="ephemeral",
            kernel_policy="baseline", kv="int8", layers=48, ctx=16, generation_steps=2,
        )


def test_validate_native_payload_rejects_fake_long_context_profile():
    payload = native_payload(activation="f32", selected=(358, 1184))
    payload["long_context_profile"]["kv_quality_checks"] = 1
    with pytest.raises(ValueError, match="long_context_profile.kv_quality_checks"):
        PERF.validate_native_payload(
            payload, activation="f32", io_backend="buffered", scratch_policy="ephemeral",
            kernel_policy="baseline", long_context_policy="none", kv="int8", layers=48, ctx=16, generation_steps=2,
        )


def test_validate_native_payload_accepts_ctx4k_smoke_profile():
    payload = native_payload(activation="f32", selected=(358, 1184), long_context_policy="ctx4k-smoke")
    payload["ctx_tokens"] = 4096

    signature = PERF.validate_native_payload(
        payload,
        activation="f32",
        io_backend="buffered",
        scratch_policy="ephemeral",
        kernel_policy="baseline",
        long_context_policy="ctx4k-smoke",
        kv="int8",
        layers=48,
        ctx=4096,
        generation_steps=2,
    )

    assert signature["selected_tokens"] == [358, 1184]


def test_validate_native_payload_rejects_ctx4k_smoke_without_target_ctx():
    payload = native_payload(activation="f32", selected=(358, 1184), long_context_policy="ctx4k-smoke")
    payload["ctx_tokens"] = 4096
    payload["long_context_profile"]["target_ctx_tokens"] = 0

    with pytest.raises(ValueError, match="long_context_profile.target_ctx_tokens"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="baseline",
            long_context_policy="ctx4k-smoke",
            kv="int8",
            layers=48,
            ctx=4096,
            generation_steps=2,
        )


def test_validate_native_payload_accepts_ctx4k_smoke_rss_limit_profile():
    payload = native_payload(activation="f32", selected=(358, 1184), long_context_policy="ctx4k-smoke")
    payload["ctx_tokens"] = 4096
    payload["long_context_profile"]["rss_limit_bytes"] = 8192

    signature = PERF.validate_native_payload(
        payload,
        activation="f32",
        io_backend="buffered",
        scratch_policy="ephemeral",
        kernel_policy="baseline",
        long_context_policy="ctx4k-smoke",
        kv="int8",
        layers=48,
        ctx=4096,
        generation_steps=2,
    )

    assert signature["selected_tokens"] == [358, 1184]


def test_compact_run_rejects_peak_rss_above_long_context_limit():
    payload = native_payload(activation="f32", selected=(358, 1184), long_context_policy="ctx4k-smoke")
    payload["ctx_tokens"] = 4096
    payload["long_context_profile"]["rss_limit_bytes"] = 100
    raw = {"wall_elapsed_seconds": 3.5, "peak_rss_bytes": 101, "payload": payload}

    with pytest.raises(ValueError, match="peak_rss_bytes exceeds long_context_profile.rss_limit_bytes"):
        PERF.compact_run(
            raw,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="baseline",
            long_context_policy="ctx4k-smoke",
            kv="int8",
            layers=48,
            ctx=4096,
            generation_steps=2,
        )


def test_compact_run_preserves_validated_long_context_profile():
    payload = native_payload(activation="f32", selected=(358, 1184), long_context_policy="ctx4k-smoke")
    payload["ctx_tokens"] = 4096
    payload["long_context_profile"]["rss_limit_bytes"] = 4096
    raw = {"wall_elapsed_seconds": 3.5, "peak_rss_bytes": 1024, "payload": payload}

    run = PERF.compact_run(
        raw,
        activation="f32",
        io_backend="buffered",
        scratch_policy="ephemeral",
        kernel_policy="baseline",
        long_context_policy="ctx4k-smoke",
        kv="int8",
        layers=48,
        ctx=4096,
        generation_steps=2,
    )

    assert run["long_context_profile"] == {
        "enabled": True,
        "policy": "ctx4k-smoke",
        "target_ctx_tokens": 4096,
        "rss_limit_bytes": 4096,
        "kv_quality_checks": 0,
        "soak_seconds": 0,
        "disabled_reason": None,
    }


def test_compact_run_preserves_default_long_context_profile():
    payload = native_payload(activation="f32", selected=(358, 1184), long_context_policy="none")
    raw = {"wall_elapsed_seconds": 3.5, "peak_rss_bytes": 1024, "payload": payload}

    run = PERF.compact_run(
        raw,
        activation="f32",
        io_backend="buffered",
        scratch_policy="ephemeral",
        kernel_policy="baseline",
        long_context_policy="none",
        kv="int8",
        layers=48,
        ctx=16,
        generation_steps=2,
    )

    assert run["long_context_profile"] == {
        "enabled": True,
        "policy": "none",
        "target_ctx_tokens": 0,
        "rss_limit_bytes": 0,
        "kv_quality_checks": 0,
        "soak_seconds": 0,
        "disabled_reason": "none_policy",
    }


def test_summarize_runs_preserves_common_long_context_profile():
    payload = native_payload(activation="f32", selected=(358, 1184), long_context_policy="ctx4k-smoke")
    payload["ctx_tokens"] = 4096
    payload["long_context_profile"]["rss_limit_bytes"] = 4096
    raw = {"wall_elapsed_seconds": 3.5, "peak_rss_bytes": 1024, "payload": payload}
    run = PERF.compact_run(
        raw,
        activation="f32",
        io_backend="buffered",
        scratch_policy="ephemeral",
        kernel_policy="baseline",
        long_context_policy="ctx4k-smoke",
        kv="int8",
        layers=48,
        ctx=4096,
        generation_steps=2,
    )

    summary = PERF.summarize_runs([run, dict(run)])

    assert summary["long_context_profile"] == run["long_context_profile"]


def test_summarize_runs_preserves_common_long_context_profile_across_many_runs():
    payload = native_payload(activation="f32", selected=(358, 1184), long_context_policy="ctx4k-smoke")
    payload["ctx_tokens"] = 4096
    raw = {"wall_elapsed_seconds": 3.5, "peak_rss_bytes": 1024, "payload": payload}
    run = PERF.compact_run(
        raw,
        activation="f32",
        io_backend="buffered",
        scratch_policy="ephemeral",
        kernel_policy="baseline",
        long_context_policy="ctx4k-smoke",
        kv="int8",
        layers=48,
        ctx=4096,
        generation_steps=2,
    )

    summary = PERF.summarize_runs([dict(run) for _ in range(8)])

    assert summary["long_context_profile"] == run["long_context_profile"]
    assert summary["total_latency_seconds"]["count"] == 8


def test_summarize_runs_rejects_mixed_long_context_profiles():
    base_payload = native_payload(activation="f32", selected=(358, 1184), long_context_policy="none")
    base_run = PERF.compact_run(
        {"wall_elapsed_seconds": 3.5, "peak_rss_bytes": 1024, "payload": base_payload},
        activation="f32",
        io_backend="buffered",
        scratch_policy="ephemeral",
        kernel_policy="baseline",
        long_context_policy="none",
        kv="int8",
        layers=48,
        ctx=16,
        generation_steps=2,
    )
    mixed_run = dict(base_run)
    mixed_run["long_context_profile"] = dict(base_run["long_context_profile"], rss_limit_bytes=1)

    with pytest.raises(ValueError, match="long_context_profile differs across measured runs"):
        PERF.summarize_runs([base_run, mixed_run])


def test_validate_native_payload_rejects_ctx4k_smoke_kv_quality_checks_until_implemented():
    payload = native_payload(activation="f32", selected=(358, 1184), long_context_policy="ctx4k-smoke")
    payload["ctx_tokens"] = 4096
    payload["long_context_profile"]["kv_quality_checks"] = 1

    with pytest.raises(ValueError, match="long_context_profile.kv_quality_checks"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="baseline",
            long_context_policy="ctx4k-smoke",
            kv="int8",
            layers=48,
            ctx=4096,
            generation_steps=2,
        )


def test_q8k_perf_experiment_rejects_nonzero_long_context_kv_quality_checks():
    parser_error = SimpleNamespace(error=lambda message: (_ for _ in ()).throw(ValueError(message)))

    with pytest.raises(ValueError, match="long-context KV quality checks are not implemented"):
        PERF.validate_long_context_cli_policy(
            parser_error,
            long_context_policy="ctx4k-smoke",
            ctx=4096,
            long_context_rss_limit_bytes=0,
            long_context_kv_quality_checks=1,
            long_context_soak_seconds=0,
        )


def test_validate_native_payload_rejects_ctx4k_smoke_soak_seconds_until_implemented():
    payload = native_payload(activation="f32", selected=(358, 1184), long_context_policy="ctx4k-smoke")
    payload["ctx_tokens"] = 4096
    payload["long_context_profile"]["soak_seconds"] = 1

    with pytest.raises(ValueError, match="long_context_profile.soak_seconds"):
        PERF.validate_native_payload(
            payload,
            activation="f32",
            io_backend="buffered",
            scratch_policy="ephemeral",
            kernel_policy="baseline",
            long_context_policy="ctx4k-smoke",
            kv="int8",
            layers=48,
            ctx=4096,
            generation_steps=2,
        )


def test_q8k_perf_experiment_rejects_nonzero_long_context_soak_seconds():
    parser_error = SimpleNamespace(error=lambda message: (_ for _ in ()).throw(ValueError(message)))

    with pytest.raises(ValueError, match="long-context soak seconds are not implemented"):
        PERF.validate_long_context_cli_policy(
            parser_error,
            long_context_policy="ctx4k-smoke",
            ctx=4096,
            long_context_rss_limit_bytes=0,
            long_context_kv_quality_checks=0,
            long_context_soak_seconds=1,
        )


def test_build_artifact_provenance_pins_source_model_and_runtime_inputs(tmp_path):
    paths = {}
    for name in ("benchmark_script", "qx_exe", "source_model_gguf", "model_qxf", "tokenizer_qxt", "prompt"):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        paths[name] = path

    artifacts = PERF.build_artifact_provenance(**paths)

    assert list(artifacts) == ["benchmark_script", "qx_exe", "source_model_gguf", "model_qxf", "tokenizer_qxt", "prompt"]
    for name, path in paths.items():
        assert artifacts[name]["path"] == str(path.resolve())
        assert artifacts[name]["size"] == len(name)
        assert len(artifacts[name]["sha256"]) == 64
