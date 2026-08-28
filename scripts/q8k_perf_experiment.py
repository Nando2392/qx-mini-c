#!/usr/bin/env python
"""Fail-closed CPU inference baseline for QXF I/O, activation and scratch modes.

Measures:
- startup/model-load wall latency
- native prefill and decode latency/tokens per second
- end-to-end wall latency and peak RSS
- allocation/free counters and scratch capacity/growth
- deterministic per-cell output signatures, buffered/mmap equivalence and
  ephemeral/persistent equivalence

Usage:
    python -m pip install -r scripts/requirements-q8k-perf.txt
    python scripts/q8k_perf_experiment.py \
      --qx-exe build/qxqxf.exe \
      --source-model models/Qwen3-30B-A3B-UD-IQ2_M.gguf \
      --model models/Qwen3-30B-A3B-UD-IQ2_M.qxf \
      --tokenizer models/Qwen3-30B-A3B.qxt \
      --prompt-file tests/fixtures/q8k_perf_prompt.txt \
      --output wiki/evidence/issue-25-persistent-scratch-baseline.json \
      --kv int8 \
      --repetitions 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil


def one_run(command: list[str]) -> dict[str, Any]:
    """Run one command, capture JSON without pipe deadlock, and sample RSS."""
    started = time.perf_counter()
    with tempfile.TemporaryFile() as stdout_file:
        process = subprocess.Popen(command, stdout=stdout_file, stderr=subprocess.PIPE, text=True)
        tracked = psutil.Process(process.pid)
        peak_rss = 0
        while process.poll() is None:
            try:
                peak_rss = max(peak_rss, tracked.memory_info().rss)
            except psutil.Error:
                pass
            time.sleep(0.005)
        stderr = process.stderr.read() if process.stderr else ""
        elapsed = time.perf_counter() - started
        if process.returncode:
            raise RuntimeError(f"benchmark failed ({process.returncode}): {stderr[-2000:]}")
        stdout_file.seek(0)
        raw = stdout_file.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("benchmark command did not emit one valid UTF-8 JSON value") from exc
    if not isinstance(payload, dict):
        raise ValueError("benchmark command JSON must be an object")
    return {"wall_elapsed_seconds": elapsed, "peak_rss_bytes": peak_rss, "payload": payload}


def summarize(values: list[float]) -> dict:
    """Return statistical summary of timing values."""
    if not values or any(isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("summary values must be finite positive numbers")
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    return {
        "count": len(values),
        "min": min(values),
        "median": median,
        "max": max(values),
        "mad": statistics.median(deviations),
        "stdev": statistics.stdev(values),
        "pstdev": statistics.pstdev(values),
    }


def summarize_non_negative(values: list[float]) -> dict:
    """Return statistical summary of counters that may be zero."""
    if not values or any(isinstance(value, bool) or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("summary values must be finite non-negative numbers")
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    return {
        "count": len(values),
        "min": min(values),
        "median": median,
        "max": max(values),
        "mad": statistics.median(deviations),
        "stdev": statistics.stdev(values),
        "pstdev": statistics.pstdev(values),
    }


def _require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_exact_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _require_positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def extract_output_signature(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract deterministic output evidence from a native runtime payload."""
    prompt_ids = payload.get("prompt_token_ids")
    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise ValueError("prompt_token_ids must be a non-empty array")
    prompt_ids = [_require_exact_int(token, "prompt_token_ids item") for token in prompt_ids]
    tokens = payload.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("tokens must be a non-empty array")
    selected: list[int] = []
    residual_checksums: list[int] = []
    norm_checksums: list[int] = []
    logits_checksums: list[int] = []
    for index, raw_token in enumerate(tokens):
        token = _require_object(raw_token, f"tokens[{index}]")
        selected_token = token.get("selected_token")
        if selected_token is None:
            continue
        selected.append(_require_exact_int(selected_token, f"tokens[{index}].selected_token"))
        head = _require_object(token.get("final_head"), f"tokens[{index}].final_head")
        residual_checksums.append(_require_exact_int(head.get("final_residual_checksum"), "final_residual_checksum"))
        norm_checksums.append(_require_exact_int(head.get("final_norm_checksum"), "final_norm_checksum"))
        logits_checksums.append(_require_exact_int(head.get("logits_checksum"), "logits_checksum"))
        if _require_exact_int(head.get("argmax_token"), "argmax_token") != selected[-1]:
            raise ValueError("argmax_token does not match selected_token")
    if not selected:
        raise ValueError("runtime payload has no selected tokens")
    if payload.get("cache_readback_ok") is not True:
        raise ValueError("cache_readback_ok must be true")
    final_token = _require_exact_int(payload.get("final_token"), "final_token")
    if final_token != selected[-1]:
        raise ValueError("final_token does not match the final selected token")
    return {
        "prompt_token_ids": prompt_ids,
        "selected_tokens": selected,
        "final_residual_checksums": residual_checksums,
        "final_norm_checksums": norm_checksums,
        "logits_checksums": logits_checksums,
        "final_token": final_token,
        "k_cache_checksum": _require_exact_int(payload.get("k_cache_checksum"), "k_cache_checksum"),
        "v_cache_checksum": _require_exact_int(payload.get("v_cache_checksum"), "v_cache_checksum"),
        "cache_readback_ok": True,
    }


def validate_native_payload(
    payload: dict[str, Any], *, activation: str, io_backend: str, scratch_policy: str,
    kv: str, layers: int, ctx: int, generation_steps: int, kernel_policy: str = "baseline",
    thread_policy: str = "serial", threads: int = 1, simd_policy: str = "scalar",
    expert_cache_policy: str = "none",
    cuda_policy: str = "none",
    prefill_gemm_policy: str = "none",
    speculative_policy: str = "none",
    kv2_policy: str = "none",
    sampling_policy: str = "none",
    long_context_policy: str = "none",
) -> dict[str, Any]:
    """Validate native timing, modality and output evidence fail-closed."""
    expected = {
        "probe": "state_loop", "activation_format": activation,
        "io_backend": io_backend,
        "scratch_policy": scratch_policy, "kernel_policy": kernel_policy,
        "thread_policy": thread_policy, "threads": threads,
        "simd_policy": simd_policy,
        "expert_cache_policy": expert_cache_policy,
        "cuda_policy": cuda_policy,
        "prefill_gemm_policy": prefill_gemm_policy,
        "speculative_policy": speculative_policy,
        "kv2_policy": kv2_policy,
        "sampling_policy": sampling_policy,
        "long_context_policy": long_context_policy,
        "kv_format": kv, "layers": layers, "ctx_tokens": ctx,
        "generation_steps": generation_steps,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{key} does not match the fixed benchmark contract")
    prompt_ids = payload.get("prompt_token_ids")
    if not isinstance(prompt_ids, list) or len(prompt_ids) < 2:
        raise ValueError("prompt_token_ids must contain at least two fixed prompt tokens")
    prompt_count = _require_exact_int(payload.get("prompt_token_count"), "prompt_token_count")
    if prompt_count != len(prompt_ids):
        raise ValueError("prompt_token_count does not match prompt_token_ids")
    expected_prefill_tokens = prompt_count - 1
    raw_tokens = payload.get("tokens")
    if not isinstance(raw_tokens, list) or len(raw_tokens) != expected_prefill_tokens + generation_steps:
        raise ValueError("runtime token evidence count does not match prompt and generation contract")
    observed_phases = [
        _require_object(raw_token, f"tokens[{index}]").get("phase")
        for index, raw_token in enumerate(raw_tokens)
    ]
    expected_phases = ["prefill"] * expected_prefill_tokens + ["generate"] * generation_steps
    if observed_phases != expected_phases:
        raise ValueError("runtime token phases do not match prefill/decode contract")
    bench = _require_object(payload.get("bench"), "bench")
    if bench.get("enabled") is not True:
        raise ValueError("native bench must be enabled")
    _require_positive_number(bench.get("elapsed_sec"), "bench.elapsed_sec")
    phases = _require_object(bench.get("phases"), "bench.phases")
    if set(phases) != {"prefill", "decode"}:
        raise ValueError("bench.phases must contain exactly prefill and decode")
    for phase_name in ("prefill", "decode"):
        phase = _require_object(phases[phase_name], f"bench.phases.{phase_name}")
        if _require_exact_int(phase.get("tokens"), f"{phase_name}.tokens") <= 0:
            raise ValueError(f"{phase_name}.tokens must be positive")
        _require_positive_number(phase.get("elapsed_sec"), f"{phase_name}.elapsed_sec")
        _require_positive_number(phase.get("tokens_per_second"), f"{phase_name}.tokens_per_second")
    if _require_exact_int(phases["prefill"].get("tokens"), "prefill.tokens") != expected_prefill_tokens:
        raise ValueError("prefill.tokens does not match fixed prompt length")
    if _require_exact_int(phases["decode"].get("tokens"), "decode.tokens") != generation_steps:
        raise ValueError("decode token count does not match generation_steps")
    observed_selected_count = sum(
        _require_object(raw_token, f"tokens[{index}]").get("selected_token") is not None
        for index, raw_token in enumerate(raw_tokens)
    )
    if observed_selected_count != generation_steps:
        raise ValueError("selected-token count does not match generation_steps")
    signature = extract_output_signature(payload)
    if len(signature["selected_tokens"]) != generation_steps:
        raise ValueError("selected-token count does not match generation_steps")
    profile = _require_object(payload.get("dequant_dot_profile"), "dequant_dot_profile")
    if profile.get("enabled") is not True:
        raise ValueError("dequant_dot_profile.enabled must be true")
    if profile.get("kernel_policy") != kernel_policy:
        raise ValueError("dequant_dot_profile.kernel_policy does not match fixed benchmark contract")
    for field in (
        "temporary_blocks_decoded", "temporary_floats_materialized", "temporary_bytes_materialized",
        "fused_dot_calls", "fallback_dot_calls", "final_head_q6_k_blocks",
    ):
        value = _require_exact_int(profile.get(field), f"dequant_dot_profile.{field}")
        if value < 0:
            raise ValueError(f"dequant_dot_profile.{field} must be non-negative")
    thread_profile = _require_object(payload.get("thread_profile"), "thread_profile")
    if thread_profile.get("enabled") is not True:
        raise ValueError("thread_profile.enabled must be true")
    if thread_profile.get("policy") != thread_policy:
        raise ValueError("thread_profile.policy does not match fixed benchmark contract")
    if _require_exact_int(thread_profile.get("requested_threads"), "thread_profile.requested_threads") != threads:
        raise ValueError("thread_profile.requested_threads does not match fixed benchmark contract")
    for field in ("workers_used", "parallel_jobs", "serial_jobs", "fallback_jobs"):
        value = _require_exact_int(thread_profile.get(field), f"thread_profile.{field}")
        if value < 0:
            raise ValueError(f"thread_profile.{field} must be non-negative")
    if thread_policy == "serial":
        if thread_profile.get("disabled_reason") != "serial_policy":
            raise ValueError("thread_profile.disabled_reason must record serial_policy")
        if _require_exact_int(thread_profile.get("workers_used"), "thread_profile.workers_used") != 1:
            raise ValueError("thread_profile.workers_used must be 1 for serial policy")
        if _require_exact_int(thread_profile.get("parallel_jobs"), "thread_profile.parallel_jobs") != 0:
            raise ValueError("thread_profile.parallel_jobs must be 0 for serial policy")
    elif thread_policy == "pool":
        if _require_exact_int(thread_profile.get("workers_used"), "thread_profile.workers_used") != threads:
            raise ValueError("thread_profile.workers_used must match --threads for pool policy")
        if _require_exact_int(thread_profile.get("parallel_jobs"), "thread_profile.parallel_jobs") <= 0:
            raise ValueError("thread_profile.parallel_jobs must be positive for pool policy")
        if _require_exact_int(thread_profile.get("serial_jobs"), "thread_profile.serial_jobs") != 0:
            raise ValueError("thread_profile.serial_jobs must be 0 for pool policy")
        if _require_exact_int(thread_profile.get("fallback_jobs"), "thread_profile.fallback_jobs") != 0:
            raise ValueError("thread_profile.fallback_jobs must be 0 for pool policy")
        if thread_profile.get("disabled_reason"):
            raise ValueError("thread_profile.disabled_reason must be absent for pool policy")
    else:
        raise ValueError("unsupported thread_policy")
    simd_profile = _require_object(payload.get("simd_profile"), "simd_profile")
    if simd_profile.get("enabled") is not True:
        raise ValueError("simd_profile.enabled must be true")
    if simd_profile.get("policy") != simd_policy:
        raise ValueError("simd_profile.policy does not match fixed benchmark contract")
    for field in ("fma_dot_calls", "fallback_dot_calls"):
        value = _require_exact_int(simd_profile.get(field), f"simd_profile.{field}")
        if value < 0:
            raise ValueError(f"simd_profile.{field} must be non-negative")
    if simd_policy == "scalar":
        if simd_profile.get("kernel") != "scalar":
            raise ValueError("simd_profile.kernel must be scalar for scalar policy")
        if simd_profile.get("disabled_reason") != "scalar_policy":
            raise ValueError("simd_profile.disabled_reason must record scalar_policy")
        if _require_exact_int(simd_profile.get("fma_dot_calls"), "simd_profile.fma_dot_calls") != 0:
            raise ValueError("simd_profile.fma_dot_calls must be 0 for scalar policy")
    elif simd_policy == "avx2-fma":
        if simd_profile.get("kernel") != "avx2_fma_q6_k_f32":
            raise ValueError("simd_profile.kernel must record avx2_fma_q6_k_f32")
        if _require_exact_int(simd_profile.get("fma_dot_calls"), "simd_profile.fma_dot_calls") <= 0:
            raise ValueError("simd_profile.fma_dot_calls must be positive for avx2-fma policy")
        if _require_exact_int(simd_profile.get("fallback_dot_calls"), "simd_profile.fallback_dot_calls") != 0:
            raise ValueError("simd_profile.fallback_dot_calls must be 0 for avx2-fma policy")
        if simd_profile.get("disabled_reason"):
            raise ValueError("simd_profile.disabled_reason must be absent for avx2-fma policy")
    else:
        raise ValueError("unsupported simd_policy")
    expert_cache_profile = _require_object(payload.get("expert_cache_profile"), "expert_cache_profile")
    if expert_cache_profile.get("enabled") is not True:
        raise ValueError("expert_cache_profile.enabled must be true")
    if expert_cache_profile.get("policy") != expert_cache_policy:
        raise ValueError("expert_cache_profile.policy does not match fixed benchmark contract")
    for field in ("cache_hits", "cache_misses", "bytes_cached", "expert_weight_reads"):
        value = _require_exact_int(expert_cache_profile.get(field), f"expert_cache_profile.{field}")
        if value < 0:
            raise ValueError(f"expert_cache_profile.{field} must be non-negative")
    if expert_cache_policy == "none":
        if expert_cache_profile.get("disabled_reason") != "none_policy":
            raise ValueError("expert_cache_profile.disabled_reason must record none_policy")
        for field in ("cache_hits", "cache_misses", "bytes_cached"):
            if _require_exact_int(expert_cache_profile.get(field), f"expert_cache_profile.{field}") != 0:
                raise ValueError(f"expert_cache_profile.{field} must be 0 for none policy")
    else:
        raise ValueError("unsupported expert_cache_policy")
    cuda_profile = _require_object(payload.get("cuda_profile"), "cuda_profile")
    if cuda_profile.get("enabled") is not True:
        raise ValueError("cuda_profile.enabled must be true")
    if cuda_profile.get("policy") != cuda_policy:
        raise ValueError("cuda_profile.policy does not match fixed benchmark contract")
    for field in ("device_bytes", "host_to_device_bytes", "device_to_host_bytes", "kernel_launches"):
        value = _require_exact_int(cuda_profile.get(field), f"cuda_profile.{field}")
        if value < 0:
            raise ValueError(f"cuda_profile.{field} must be non-negative")
    if cuda_policy == "none":
        if cuda_profile.get("backend") != "none":
            raise ValueError("cuda_profile.backend must be none for none policy")
        if cuda_profile.get("disabled_reason") != "none_policy":
            raise ValueError("cuda_profile.disabled_reason must record none_policy")
        for field in ("device_bytes", "host_to_device_bytes", "device_to_host_bytes", "kernel_launches"):
            if _require_exact_int(cuda_profile.get(field), f"cuda_profile.{field}") != 0:
                raise ValueError(f"cuda_profile.{field} must be 0 for none policy")
    else:
        raise ValueError("unsupported cuda_policy")
    prefill_gemm_profile = _require_object(payload.get("prefill_gemm_profile"), "prefill_gemm_profile")
    if prefill_gemm_profile.get("enabled") is not True:
        raise ValueError("prefill_gemm_profile.enabled must be true")
    if prefill_gemm_profile.get("policy") != prefill_gemm_policy:
        raise ValueError("prefill_gemm_profile.policy does not match fixed benchmark contract")
    for field in ("gemm_calls", "batched_tokens", "fused_rows", "temporary_bytes"):
        value = _require_exact_int(prefill_gemm_profile.get(field), f"prefill_gemm_profile.{field}")
        if value < 0:
            raise ValueError(f"prefill_gemm_profile.{field} must be non-negative")
    if prefill_gemm_policy == "none":
        if prefill_gemm_profile.get("backend") != "none":
            raise ValueError("prefill_gemm_profile.backend must be none for none policy")
        if prefill_gemm_profile.get("disabled_reason") != "none_policy":
            raise ValueError("prefill_gemm_profile.disabled_reason must record none_policy")
        for field in ("gemm_calls", "batched_tokens", "fused_rows", "temporary_bytes"):
            if _require_exact_int(prefill_gemm_profile.get(field), f"prefill_gemm_profile.{field}") != 0:
                raise ValueError(f"prefill_gemm_profile.{field} must be 0 for none policy")
    else:
        raise ValueError("unsupported prefill_gemm_policy")
    speculative_profile = _require_object(payload.get("speculative_profile"), "speculative_profile")
    if speculative_profile.get("enabled") is not True:
        raise ValueError("speculative_profile.enabled must be true")
    if speculative_profile.get("policy") != speculative_policy:
        raise ValueError("speculative_profile.policy does not match fixed benchmark contract")
    for field in ("draft_tokens", "accepted_tokens", "rejected_tokens", "target_verifications"):
        value = _require_exact_int(speculative_profile.get(field), f"speculative_profile.{field}")
        if value < 0:
            raise ValueError(f"speculative_profile.{field} must be non-negative")
    if speculative_policy == "none":
        if speculative_profile.get("backend") != "none":
            raise ValueError("speculative_profile.backend must be none for none policy")
        if speculative_profile.get("disabled_reason") != "none_policy":
            raise ValueError("speculative_profile.disabled_reason must record none_policy")
        for field in ("draft_tokens", "accepted_tokens", "rejected_tokens", "target_verifications"):
            if _require_exact_int(speculative_profile.get(field), f"speculative_profile.{field}") != 0:
                raise ValueError(f"speculative_profile.{field} must be 0 for none policy")
    else:
        raise ValueError("unsupported speculative_policy")
    kv2_profile = _require_object(payload.get("kv2_profile"), "kv2_profile")
    if kv2_profile.get("enabled") is not True:
        raise ValueError("kv2_profile.enabled must be true")
    if kv2_profile.get("policy") != kv2_policy:
        raise ValueError("kv2_profile.policy does not match fixed benchmark contract")
    for field in ("packed_bytes", "read_ops", "write_ops", "fallback_reads"):
        value = _require_exact_int(kv2_profile.get(field), f"kv2_profile.{field}")
        if value < 0:
            raise ValueError(f"kv2_profile.{field} must be non-negative")
    if kv2_policy == "none":
        if kv2_profile.get("format") != "none":
            raise ValueError("kv2_profile.format must be none for none policy")
        if kv2_profile.get("disabled_reason") != "none_policy":
            raise ValueError("kv2_profile.disabled_reason must record none_policy")
        for field in ("packed_bytes", "read_ops", "write_ops", "fallback_reads"):
            if _require_exact_int(kv2_profile.get(field), f"kv2_profile.{field}") != 0:
                raise ValueError(f"kv2_profile.{field} must be 0 for none policy")
    else:
        raise ValueError("unsupported kv2_policy")
    sampling_profile = _require_object(payload.get("sampling_profile"), "sampling_profile")
    if sampling_profile.get("enabled") is not True:
        raise ValueError("sampling_profile.enabled must be true")
    if sampling_profile.get("policy") != sampling_policy:
        raise ValueError("sampling_profile.policy does not match fixed benchmark contract")
    for field in ("stochastic_samples", "top_p_evaluations", "beam_width"):
        value = _require_exact_int(sampling_profile.get(field), f"sampling_profile.{field}")
        if value < 0:
            raise ValueError(f"sampling_profile.{field} must be non-negative")
    if sampling_policy == "none":
        if sampling_profile.get("mode") != "greedy":
            raise ValueError("sampling_profile.mode must be greedy for none policy")
        if sampling_profile.get("disabled_reason") != "none_policy":
            raise ValueError("sampling_profile.disabled_reason must record none_policy")
        for field in ("stochastic_samples", "top_p_evaluations"):
            if _require_exact_int(sampling_profile.get(field), f"sampling_profile.{field}") != 0:
                raise ValueError(f"sampling_profile.{field} must be 0 for none policy")
        if _require_exact_int(sampling_profile.get("beam_width"), "sampling_profile.beam_width") != 1:
            raise ValueError("sampling_profile.beam_width must be 1 for none policy")
    else:
        raise ValueError("unsupported sampling_policy")
    long_context_profile = _require_object(payload.get("long_context_profile"), "long_context_profile")
    if long_context_profile.get("enabled") is not True:
        raise ValueError("long_context_profile.enabled must be true")
    if long_context_profile.get("policy") != long_context_policy:
        raise ValueError("long_context_profile.policy does not match fixed benchmark contract")
    for field in ("target_ctx_tokens", "rss_limit_bytes", "kv_quality_checks", "soak_seconds"):
        value = _require_exact_int(long_context_profile.get(field), f"long_context_profile.{field}")
        if value < 0:
            raise ValueError(f"long_context_profile.{field} must be non-negative")
    if long_context_policy == "none":
        if long_context_profile.get("disabled_reason") != "none_policy":
            raise ValueError("long_context_profile.disabled_reason must record none_policy")
        for field in ("target_ctx_tokens", "rss_limit_bytes", "kv_quality_checks", "soak_seconds"):
            if _require_exact_int(long_context_profile.get(field), f"long_context_profile.{field}") != 0:
                raise ValueError(f"long_context_profile.{field} must be 0 for none policy")
    elif long_context_policy == "ctx4k-smoke":
        if ctx < 4096:
            raise ValueError("ctx4k-smoke long_context_policy requires ctx >= 4096")
        if _require_exact_int(long_context_profile.get("target_ctx_tokens"), "long_context_profile.target_ctx_tokens") != 4096:
            raise ValueError("long_context_profile.target_ctx_tokens must be 4096 for ctx4k-smoke policy")
        if long_context_profile.get("disabled_reason") is not None:
            raise ValueError("long_context_profile.disabled_reason must be null for ctx4k-smoke policy")
        for field in ("kv_quality_checks", "soak_seconds"):
            if _require_exact_int(long_context_profile.get(field), f"long_context_profile.{field}") != 0:
                raise ValueError(f"long_context_profile.{field} must be 0 for ctx4k-smoke policy")
    else:
        raise ValueError("unsupported long_context_policy")
    return signature


def validate_output_contract(signatures: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Require 2x2x2 determinism plus exact I/O and scratch-policy equality per activation."""
    expected = {
        f"{policy}:{backend}:{activation}"
        for policy in ("ephemeral", "persistent")
        for backend in ("buffered", "mmap")
        for activation in ("f32", "q8_k_compat")
    }
    if set(signatures) != expected:
        raise ValueError("output evidence must contain the complete scratch-policy buffered/mmap activation matrix")
    canonical: dict[str, dict[str, Any]] = {}
    for cell, items in signatures.items():
        if not items:
            raise ValueError(f"{cell} has no measured output signatures")
        if any(item != items[0] for item in items[1:]):
            raise ValueError(f"{cell} output is not deterministic across repetitions")
        canonical[cell] = items[0]
    for activation in ("f32", "q8_k_compat"):
        for policy in ("ephemeral", "persistent"):
            if canonical[f"{policy}:buffered:{activation}"] != canonical[f"{policy}:mmap:{activation}"]:
                raise ValueError(f"buffered/mmap output mismatch for {activation}")
        for backend in ("buffered", "mmap"):
            if canonical[f"ephemeral:{backend}:{activation}"] != canonical[f"persistent:{backend}:{activation}"]:
                raise ValueError(f"ephemeral/persistent output mismatch for {backend} {activation}")
    f32 = canonical["ephemeral:buffered:f32"]
    q8k = canonical["ephemeral:buffered:q8_k_compat"]
    if f32["prompt_token_ids"] != q8k["prompt_token_ids"]:
        raise ValueError("prompt token IDs differ across activation modes")
    selected_equal = f32["selected_tokens"] == q8k["selected_tokens"]
    checksum_fields = (
        "final_residual_checksums", "final_norm_checksums", "logits_checksums",
        "k_cache_checksum", "v_cache_checksum",
    )
    checksums_equal = all(f32[field] == q8k[field] for field in checksum_fields)
    return {
        "status": "pass",
        "io_equivalence": {"f32": True, "q8_k_compat": True},
        "scratch_policy_equivalence": {"f32": True, "q8_k_compat": True},
        "prompt_token_ids": f32["prompt_token_ids"],
        "selected_tokens_equal_across_activations": selected_equal,
        "selected_tokens_by_mode": {
            "f32": f32["selected_tokens"],
            "q8_k_compat": q8k["selected_tokens"],
        },
        "activation_numeric_checksums_equal": checksums_equal,
        "allowed_cross_activation_differences": [
            "selected-token sequence because q8_k_compat is an opt-in numerical mode without global greedy parity",
            "final residual, final norm, logits, and KV cache checksums",
        ],
    }


def build_inference_command(
    *, qx_exe: Path, model: Path, tokenizer: Path, prompt_file: Path,
    activation: str, kv: str, layers: int, ctx: int,
    generation_steps: int, seed: int, io_backend: str, scratch_policy: str, kernel_policy: str,
    thread_policy: str = "serial", threads: int = 1, simd_policy: str = "scalar",
    expert_cache_policy: str = "none",
    cuda_policy: str = "none",
    prefill_gemm_policy: str = "none",
    speculative_policy: str = "none",
    kv2_policy: str = "none",
    sampling_policy: str = "none",
    long_context_policy: str = "none",
    long_context_rss_limit_bytes: int = 0,
    long_context_kv_quality_checks: int = 0,
    long_context_soak_seconds: int = 0,
) -> list[str]:
    """Build the fixed real-prompt inference command for one A/B cell."""
    return [
        str(qx_exe), "prompt-state-loop-probe", "--in", str(model),
        "--tokenizer", str(tokenizer), "--text-file", str(prompt_file),
        "--generate", str(generation_steps), "--layers", str(layers),
        "--ctx", str(ctx), "--kv", kv, "--activation", activation,
        "--io-backend", io_backend,
        "--scratch-policy", scratch_policy,
        "--kernel-policy", kernel_policy,
        "--thread-policy", thread_policy, "--threads", str(threads),
        "--simd-policy", simd_policy,
        "--expert-cache-policy", expert_cache_policy,
        "--cuda-policy", cuda_policy,
        "--prefill-gemm-policy", prefill_gemm_policy,
        "--speculative-policy", speculative_policy,
        "--kv2-policy", kv2_policy,
        "--sampling-policy", sampling_policy,
        "--long-context-policy", long_context_policy,
        "--long-context-rss-limit-bytes", str(long_context_rss_limit_bytes),
        "--long-context-kv-quality-checks", str(long_context_kv_quality_checks),
        "--long-context-soak-seconds", str(long_context_soak_seconds),
        "--temperature", "0", "--seed", str(seed),
        "--full-moe", "--final-head", "--bench", "--dequant-profile",
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_provenance(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256_file(path)}


def build_artifact_provenance(
    *, benchmark_script: Path, qx_exe: Path, source_model_gguf: Path, model_qxf: Path,
    tokenizer_qxt: Path, prompt: Path,
) -> dict[str, dict[str, Any]]:
    return {
        "benchmark_script": artifact_provenance(benchmark_script),
        "qx_exe": artifact_provenance(qx_exe),
        "source_model_gguf": artifact_provenance(source_model_gguf),
        "model_qxf": artifact_provenance(model_qxf),
        "tokenizer_qxt": artifact_provenance(tokenizer_qxt),
        "prompt": artifact_provenance(prompt),
    }


def validate_long_context_cli_policy(
    parser: argparse.ArgumentParser, *, long_context_policy: str, ctx: int,
    long_context_rss_limit_bytes: int, long_context_kv_quality_checks: int,
    long_context_soak_seconds: int,
) -> None:
    if long_context_rss_limit_bytes < 0:
        parser.error("long-context RSS limit must be non-negative")
    if long_context_kv_quality_checks < 0:
        parser.error("long-context KV quality checks must be non-negative")
    if long_context_soak_seconds < 0:
        parser.error("long-context soak seconds must be non-negative")
    if long_context_policy == "ctx4k-smoke" and ctx < 4096:
        parser.error("ctx4k-smoke long-context policy requires ctx >= 4096")
    if long_context_policy == "none" and long_context_rss_limit_bytes != 0:
        parser.error("long-context RSS limit requires ctx4k-smoke policy")
    if long_context_kv_quality_checks != 0:
        parser.error("long-context KV quality checks are not implemented")
    if long_context_soak_seconds != 0:
        parser.error("long-context soak seconds are not implemented")


def source_state() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
        text=True, capture_output=True, check=True,
    )
    revision = result.stdout.strip()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("git revision is not a lowercase 40-character SHA-1")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, check=True,
    ).stdout
    return {
        "revision": revision,
        "working_tree_dirty": bool(diff),
        "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def compact_run(
    raw: dict[str, Any], *, activation: str, io_backend: str, kv: str, layers: int,
    ctx: int, generation_steps: int, scratch_policy: str, kernel_policy: str,
    thread_policy: str = "serial", threads: int = 1, long_context_policy: str = "none",
) -> dict[str, Any]:
    payload = _require_object(raw.get("payload"), "runtime payload")
    signature = validate_native_payload(
        payload, activation=activation, io_backend=io_backend, scratch_policy=scratch_policy,
        kernel_policy=kernel_policy, thread_policy=thread_policy, threads=threads,
        long_context_policy=long_context_policy,
        kv=kv, layers=layers, ctx=ctx, generation_steps=generation_steps,
    )
    bench = _require_object(payload["bench"], "bench")
    phases = _require_object(bench["phases"], "bench.phases")
    prefill = _require_object(phases["prefill"], "bench.phases.prefill")
    decode = _require_object(phases["decode"], "bench.phases.decode")
    allocation = _require_object(payload.get("allocation_profile"), "allocation_profile")
    scratch = _require_object(payload.get("scratch_profile"), "scratch_profile")
    dequant = _require_object(payload.get("dequant_dot_profile"), "dequant_dot_profile")
    thread_profile = _require_object(payload.get("thread_profile"), "thread_profile")
    expert_cache = _require_object(payload.get("expert_cache_profile"), "expert_cache_profile")
    cuda = _require_object(payload.get("cuda_profile"), "cuda_profile")
    prefill_gemm = _require_object(payload.get("prefill_gemm_profile"), "prefill_gemm_profile")
    speculative = _require_object(payload.get("speculative_profile"), "speculative_profile")
    kv2 = _require_object(payload.get("kv2_profile"), "kv2_profile")
    peak_rss = _require_exact_int(raw.get("peak_rss_bytes"), "peak_rss_bytes")
    if peak_rss <= 0:
        raise ValueError("peak_rss_bytes must be positive")
    long_context = _require_object(payload.get("long_context_profile"), "long_context_profile")
    rss_limit = _require_exact_int(long_context.get("rss_limit_bytes"), "long_context_profile.rss_limit_bytes")
    if rss_limit and peak_rss > rss_limit:
        raise ValueError("peak_rss_bytes exceeds long_context_profile.rss_limit_bytes")
    return {
        "total_latency_seconds": _require_positive_number(raw.get("wall_elapsed_seconds"), "wall_elapsed_seconds"),
        "native_process_seconds": _require_positive_number(bench.get("elapsed_sec"), "bench.elapsed_sec"),
        "prefill_latency_seconds": _require_positive_number(prefill.get("elapsed_sec"), "prefill.elapsed_sec"),
        "prefill_tokens_per_second": _require_positive_number(prefill.get("tokens_per_second"), "prefill.tokens_per_second"),
        "decode_latency_seconds": _require_positive_number(decode.get("elapsed_sec"), "decode.elapsed_sec"),
        "decode_tokens_per_second": _require_positive_number(decode.get("tokens_per_second"), "decode.tokens_per_second"),
        "peak_rss_bytes": peak_rss,
        "allocation_malloc_calls": _require_exact_int(allocation.get("malloc_calls"), "allocation.malloc_calls"),
        "allocation_calloc_calls": _require_exact_int(allocation.get("calloc_calls"), "allocation.calloc_calls"),
        "allocation_realloc_calls": _require_exact_int(allocation.get("realloc_calls"), "allocation.realloc_calls"),
        "allocation_free_calls": _require_exact_int(allocation.get("free_calls"), "allocation.free_calls"),
        "allocation_bytes_requested": _require_exact_int(allocation.get("bytes_requested"), "allocation.bytes_requested"),
        "scratch_peak_capacity_bytes": _require_exact_int(scratch.get("peak_capacity_bytes"), "scratch.peak_capacity_bytes"),
        "scratch_growth_events": _require_exact_int(scratch.get("growth_events"), "scratch.growth_events"),
        "allocation_profile": allocation,
        "scratch_profile": scratch,
        "dequant_dot_profile": dequant,
        "thread_profile": thread_profile,
        "expert_cache_profile": expert_cache,
        "cuda_profile": cuda,
        "prefill_gemm_profile": prefill_gemm,
        "speculative_profile": speculative,
        "kv2_profile": kv2,
        "long_context_profile": long_context,
        "dequant_temporary_blocks_decoded": _require_exact_int(dequant.get("temporary_blocks_decoded"), "dequant.temporary_blocks_decoded"),
        "dequant_temporary_floats_materialized": _require_exact_int(dequant.get("temporary_floats_materialized"), "dequant.temporary_floats_materialized"),
        "dequant_temporary_bytes_materialized": _require_exact_int(dequant.get("temporary_bytes_materialized"), "dequant.temporary_bytes_materialized"),
        "dequant_fused_dot_calls": _require_exact_int(dequant.get("fused_dot_calls"), "dequant.fused_dot_calls"),
        "dequant_fallback_dot_calls": _require_exact_int(dequant.get("fallback_dot_calls"), "dequant.fallback_dot_calls"),
        "dequant_final_head_q6_k_blocks": _require_exact_int(dequant.get("final_head_q6_k_blocks"), "dequant.final_head_q6_k_blocks"),
        "thread_workers_used": _require_exact_int(thread_profile.get("workers_used"), "thread.workers_used"),
        "thread_parallel_jobs": _require_exact_int(thread_profile.get("parallel_jobs"), "thread.parallel_jobs"),
        "thread_serial_jobs": _require_exact_int(thread_profile.get("serial_jobs"), "thread.serial_jobs"),
        "thread_fallback_jobs": _require_exact_int(thread_profile.get("fallback_jobs"), "thread.fallback_jobs"),
        "expert_cache_hits": _require_exact_int(expert_cache.get("cache_hits"), "expert_cache.cache_hits"),
        "expert_cache_misses": _require_exact_int(expert_cache.get("cache_misses"), "expert_cache.cache_misses"),
        "expert_cache_bytes_cached": _require_exact_int(expert_cache.get("bytes_cached"), "expert_cache.bytes_cached"),
        "expert_cache_weight_reads": _require_exact_int(expert_cache.get("expert_weight_reads"), "expert_cache.expert_weight_reads"),
        "cuda_device_bytes": _require_exact_int(cuda.get("device_bytes"), "cuda.device_bytes"),
        "cuda_host_to_device_bytes": _require_exact_int(cuda.get("host_to_device_bytes"), "cuda.host_to_device_bytes"),
        "cuda_device_to_host_bytes": _require_exact_int(cuda.get("device_to_host_bytes"), "cuda.device_to_host_bytes"),
        "cuda_kernel_launches": _require_exact_int(cuda.get("kernel_launches"), "cuda.kernel_launches"),
        "prefill_gemm_calls": _require_exact_int(prefill_gemm.get("gemm_calls"), "prefill_gemm.gemm_calls"),
        "prefill_gemm_batched_tokens": _require_exact_int(prefill_gemm.get("batched_tokens"), "prefill_gemm.batched_tokens"),
        "prefill_gemm_fused_rows": _require_exact_int(prefill_gemm.get("fused_rows"), "prefill_gemm.fused_rows"),
        "prefill_gemm_temporary_bytes": _require_exact_int(prefill_gemm.get("temporary_bytes"), "prefill_gemm.temporary_bytes"),
        "speculative_draft_tokens": _require_exact_int(speculative.get("draft_tokens"), "speculative.draft_tokens"),
        "speculative_accepted_tokens": _require_exact_int(speculative.get("accepted_tokens"), "speculative.accepted_tokens"),
        "speculative_rejected_tokens": _require_exact_int(speculative.get("rejected_tokens"), "speculative.rejected_tokens"),
        "speculative_target_verifications": _require_exact_int(speculative.get("target_verifications"), "speculative.target_verifications"),
        "kv2_packed_bytes": _require_exact_int(kv2.get("packed_bytes"), "kv2.packed_bytes"),
        "kv2_read_ops": _require_exact_int(kv2.get("read_ops"), "kv2.read_ops"),
        "kv2_write_ops": _require_exact_int(kv2.get("write_ops"), "kv2.write_ops"),
        "kv2_fallback_reads": _require_exact_int(kv2.get("fallback_reads"), "kv2.fallback_reads"),
        "output_signature": signature,
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    positive_fields = (
        "total_latency_seconds", "native_process_seconds",
        "prefill_latency_seconds", "prefill_tokens_per_second",
        "decode_latency_seconds", "decode_tokens_per_second", "peak_rss_bytes",
    )
    non_negative_fields = (
        "allocation_malloc_calls", "allocation_calloc_calls", "allocation_realloc_calls",
        "allocation_free_calls", "allocation_bytes_requested", "scratch_peak_capacity_bytes",
        "scratch_growth_events", "dequant_temporary_blocks_decoded", "dequant_temporary_floats_materialized",
        "dequant_temporary_bytes_materialized", "dequant_fused_dot_calls", "dequant_fallback_dot_calls",
        "dequant_final_head_q6_k_blocks", "thread_workers_used", "thread_parallel_jobs",
        "thread_serial_jobs", "thread_fallback_jobs", "expert_cache_hits", "expert_cache_misses",
        "expert_cache_bytes_cached", "expert_cache_weight_reads", "cuda_device_bytes",
        "cuda_host_to_device_bytes", "cuda_device_to_host_bytes", "cuda_kernel_launches",
        "prefill_gemm_calls", "prefill_gemm_batched_tokens", "prefill_gemm_fused_rows",
        "prefill_gemm_temporary_bytes", "speculative_draft_tokens", "speculative_accepted_tokens",
        "speculative_rejected_tokens", "speculative_target_verifications", "kv2_packed_bytes",
        "kv2_read_ops", "kv2_write_ops", "kv2_fallback_reads",
    )
    first_long_context_profile = _require_object(runs[0].get("long_context_profile"), "long_context_profile")
    for run in runs[1:]:
        profile = _require_object(run.get("long_context_profile"), "long_context_profile")
        if profile != first_long_context_profile:
            raise ValueError("long_context_profile differs across measured runs")
    summary = {field: summarize([float(run[field]) for run in runs]) for field in positive_fields}
    summary.update({field: summarize_non_negative([float(run[field]) for run in runs]) for field in non_negative_fields})
    summary["long_context_profile"] = json.loads(json.dumps(first_long_context_profile))
    return summary


def _validate_report_long_context_profile(profile: dict[str, Any]) -> None:
    if profile.get("enabled") is not True:
        raise ValueError("long_context_profile.enabled must be true")
    policy = profile.get("policy")
    if policy == "none":
        if profile.get("disabled_reason") != "none_policy":
            raise ValueError("long_context_profile.disabled_reason must record none_policy")
    elif policy == "ctx4k-smoke":
        if profile.get("disabled_reason") is not None:
            raise ValueError("long_context_profile.disabled_reason must be null for ctx4k-smoke policy")
    else:
        raise ValueError("unsupported long_context_profile.policy")
    numeric_values: dict[str, int] = {}
    for field in ("target_ctx_tokens", "rss_limit_bytes", "kv_quality_checks", "soak_seconds"):
        value = _require_exact_int(profile.get(field), f"long_context_profile.{field}")
        if value < 0:
            raise ValueError(f"long_context_profile.{field} must be non-negative")
        numeric_values[field] = value
    if policy == "none" and numeric_values["target_ctx_tokens"] != 0:
        raise ValueError("long_context_profile.target_ctx_tokens must be 0 for none policy")
    if policy == "none" and numeric_values["rss_limit_bytes"] != 0:
        raise ValueError("long_context_profile.rss_limit_bytes must be 0 for none policy")
    if numeric_values["kv_quality_checks"] != 0:
        raise ValueError("long_context_profile.kv_quality_checks must be 0")
    if policy == "ctx4k-smoke" and numeric_values["target_ctx_tokens"] != 4096:
        raise ValueError("long_context_profile.target_ctx_tokens must be 4096 for ctx4k-smoke policy")


def summarize_cells_long_context_profile(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the common long-context profile across benchmark cells, fail-closed on drift."""
    if not cells:
        raise ValueError("benchmark cells must be non-empty")
    first_cell = _require_object(cells[0], "benchmark cell")
    if "summary" not in first_cell:
        raise ValueError("cell.summary is required")
    first_summary = _require_object(first_cell.get("summary"), "cell.summary")
    if "long_context_profile" not in first_summary:
        raise ValueError("long_context_profile is required")
    first_profile = _require_object(first_summary.get("long_context_profile"), "cell.summary.long_context_profile")
    _validate_report_long_context_profile(first_profile)
    for cell in cells[1:]:
        cell_obj = _require_object(cell, "benchmark cell")
        if "summary" not in cell_obj:
            raise ValueError("cell.summary is required")
        summary = _require_object(cell_obj.get("summary"), "cell.summary")
        if "long_context_profile" not in summary:
            raise ValueError("long_context_profile is required")
        profile = _require_object(summary.get("long_context_profile"), "cell.summary.long_context_profile")
        _validate_report_long_context_profile(profile)
        if profile != first_profile:
            raise ValueError("long_context_profile differs across benchmark cells")
    return json.loads(json.dumps(first_profile))


def build_long_context_measurement_gate(cells: list[dict[str, Any]], *, ctx: int) -> dict[str, Any]:
    """Summarize the measured long-context contract for the completed benchmark report."""
    if not cells:
        raise ValueError("long_context_measurement requires benchmark cells")
    profile = summarize_cells_long_context_profile(cells)
    policy = profile.get("policy")
    target_ctx_tokens = _require_exact_int(profile.get("target_ctx_tokens"), "long_context_profile.target_ctx_tokens")
    rss_limit_bytes = _require_exact_int(profile.get("rss_limit_bytes"), "long_context_profile.rss_limit_bytes")
    kv_quality_checks = _require_exact_int(profile.get("kv_quality_checks"), "long_context_profile.kv_quality_checks")
    soak_seconds = _require_exact_int(profile.get("soak_seconds"), "long_context_profile.soak_seconds")
    if rss_limit_bytes < 0:
        raise ValueError("long_context_profile.rss_limit_bytes must be non-negative")
    if kv_quality_checks != 0:
        raise ValueError("long_context_profile.kv_quality_checks must be 0 for measurement")
    if soak_seconds != 0:
        raise ValueError("long_context_profile.soak_seconds must be 0 for measurement")
    measured_run_count = 0
    for cell in cells:
        summary = _require_object(cell.get("summary"), "cell.summary")
        if "peak_rss_bytes" not in summary:
            raise ValueError("peak_rss_bytes summary is required")
        peak_rss = _require_object(summary.get("peak_rss_bytes"), "peak_rss_bytes summary")
        if "count" not in peak_rss:
            raise ValueError("peak_rss_bytes count is required")
        count = _require_exact_int(peak_rss.get("count"), "cell.summary.peak_rss_bytes.count")
        if count <= 0:
            raise ValueError("peak_rss_bytes count must be positive")
        measured_run_count += count
    if policy == "none":
        if target_ctx_tokens != 0:
            raise ValueError("none long-context measurement must not record target ctx tokens")
        if rss_limit_bytes != 0:
            raise ValueError("none long-context measurement must not record an RSS limit")
    elif policy == "ctx4k-smoke":
        if target_ctx_tokens != 4096:
            raise ValueError("ctx4k-smoke measurement must record target_ctx_tokens=4096")
        if ctx < target_ctx_tokens:
            raise ValueError("ctx4k-smoke measurement requires ctx >= target_ctx_tokens")
    else:
        raise ValueError("unsupported long-context measurement policy")
    return {
        "status": "pass",
        "policy": policy,
        "target_ctx_tokens": target_ctx_tokens,
        "rss_limit_bytes": rss_limit_bytes,
        "rss_limit_active": rss_limit_bytes != 0,
        "measured_ctx_tokens": ctx,
        "measured_cell_count": len(cells),
        "measured_run_count": measured_run_count,
        "peak_rss_summary_present": True,
    }


def compare_kernel_policy_effects(runs_by_policy: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    expected = {"baseline", "fused"}
    if set(runs_by_policy) != expected or any(not runs_by_policy[policy] for policy in expected):
        raise ValueError("kernel policy comparison requires baseline and fused runs")
    baseline = summarize_runs(runs_by_policy["baseline"])
    fused = summarize_runs(runs_by_policy["fused"])
    baseline_temp = baseline["dequant_temporary_bytes_materialized"]["median"]
    fused_temp = fused["dequant_temporary_bytes_materialized"]["median"]
    baseline_calls = baseline["dequant_fused_dot_calls"]["median"]
    fused_calls = fused["dequant_fused_dot_calls"]["median"]
    if fused_temp >= baseline_temp:
        raise ValueError("fused kernel policy did not reduce temporary bytes materialized")
    if fused_calls <= baseline_calls:
        raise ValueError("fused kernel policy did not increase fused dot calls")
    return {
        "status": "pass",
        "temporary_bytes_materialized_reduced": True,
        "fused_dot_calls_increased": True,
        "baseline_temporary_bytes_median": baseline_temp,
        "fused_temporary_bytes_median": fused_temp,
        "baseline_fused_dot_calls_median": baseline_calls,
        "fused_fused_dot_calls_median": fused_calls,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed QXF I/O x activation CPU baseline")
    parser.add_argument("--qx-exe", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kv", choices=("f32", "int8"), default="int8")
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--ctx", type=int, default=16)
    parser.add_argument("--generate", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--long-context-policy", choices=("none", "ctx4k-smoke"), default="none")
    parser.add_argument("--long-context-rss-limit-bytes", type=int, default=0)
    parser.add_argument("--long-context-kv-quality-checks", type=int, default=0)
    parser.add_argument("--long-context-soak-seconds", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in (args.qx_exe, args.source_model, args.model, args.tokenizer, args.prompt_file):
        if not path.is_file():
            parser.error(f"missing fixture: {path}")
    if not args.prompt_file.read_bytes():
        parser.error("prompt file must not be empty")
    if args.repetitions < 3:
        parser.error("repetitions must be >= 3")
    if args.warmups < 1:
        parser.error("warmups must be >= 1")
    if args.layers != 48:
        parser.error("this final-head baseline requires exactly 48 layers")
    if args.ctx < 2 or args.generate < 1 or args.seed < 0:
        parser.error("ctx must be >= 2, generate >= 1, and seed >= 0")
    validate_long_context_cli_policy(
        parser,
        long_context_policy=args.long_context_policy,
        ctx=args.ctx,
        long_context_rss_limit_bytes=args.long_context_rss_limit_bytes,
        long_context_kv_quality_checks=args.long_context_kv_quality_checks,
        long_context_soak_seconds=args.long_context_soak_seconds,
    )
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists (use --overwrite): {args.output}")

    try:
        startup_cells: list[dict[str, Any]] = []
        for io_backend in ("buffered", "mmap"):
            startup_command = [
                str(args.qx_exe), "inspect-tensor", "--in", str(args.model),
                "--name", "token_embd.weight", "--io-backend", io_backend,
            ]
            startup_warmups = [one_run(startup_command) for _ in range(args.warmups)]
            startup_measured = [one_run(startup_command) for _ in range(args.repetitions)]
            for run in startup_warmups + startup_measured:
                payload = _require_object(run["payload"], "inspect payload")
                if payload.get("name") != "token_embd.weight" or payload.get("io_backend") != io_backend:
                    raise ValueError("startup/model-load probe drifted from the fixed QXF I/O backend")
            startup_cells.append({
                "io_backend": io_backend,
                "command": startup_command,
                "warmups": [run["wall_elapsed_seconds"] for run in startup_warmups],
                "measured": [run["wall_elapsed_seconds"] for run in startup_measured],
                "summary_seconds": summarize([run["wall_elapsed_seconds"] for run in startup_measured]),
            })

        cells: list[dict[str, Any]] = []
        signatures: dict[str, list[dict[str, Any]]] = {}
        kernel_runs: dict[str, list[dict[str, Any]]] = {"baseline": [], "fused": []}
        for scratch_policy in ("ephemeral", "persistent"):
          for io_backend in ("buffered", "mmap"):
            for activation in ("f32", "q8_k_compat"):
              for kernel_policy in ("baseline", "fused"):
                command = build_inference_command(
                    qx_exe=args.qx_exe, model=args.model, tokenizer=args.tokenizer,
                    prompt_file=args.prompt_file, activation=activation, kv=args.kv,
                    layers=args.layers, ctx=args.ctx, generation_steps=args.generate,
                    seed=args.seed, io_backend=io_backend, scratch_policy=scratch_policy,
                    kernel_policy=kernel_policy, thread_policy="serial", threads=1,
                    long_context_policy=args.long_context_policy,
                    long_context_rss_limit_bytes=args.long_context_rss_limit_bytes,
                    long_context_kv_quality_checks=args.long_context_kv_quality_checks,
                    long_context_soak_seconds=args.long_context_soak_seconds,
                )
                warmups = [
                    compact_run(one_run(command), activation=activation, io_backend=io_backend, kv=args.kv,
                                layers=args.layers, ctx=args.ctx, generation_steps=args.generate,
                                scratch_policy=scratch_policy, kernel_policy=kernel_policy,
                                thread_policy="serial", threads=1,
                                long_context_policy=args.long_context_policy)
                    for _ in range(args.warmups)
                ]
                measured = [
                    compact_run(one_run(command), activation=activation, io_backend=io_backend, kv=args.kv,
                                layers=args.layers, ctx=args.ctx, generation_steps=args.generate,
                                scratch_policy=scratch_policy, kernel_policy=kernel_policy,
                                thread_policy="serial", threads=1,
                                long_context_policy=args.long_context_policy)
                    for _ in range(args.repetitions)
                ]
                cell_key = f"{scratch_policy}:{io_backend}:{activation}"
                signatures.setdefault(cell_key, []).extend(run["output_signature"] for run in warmups + measured)
                if scratch_policy == "ephemeral" and io_backend == "buffered" and activation == "f32":
                    kernel_runs[kernel_policy].extend(measured)
                cells.append({
                    "scratch_policy": scratch_policy,
                    "io_backend": io_backend,
                    "activation_format": activation,
                    "kernel_policy": kernel_policy,
                    "command": command,
                    "warmups": warmups,
                    "measured": measured,
                    "summary": summarize_runs(measured),
                })

        output_gate = validate_output_contract(signatures)
        kernel_policy_effects = compare_kernel_policy_effects(kernel_runs)
        by_cell = {(cell["scratch_policy"], cell["io_backend"], cell["activation_format"], cell["kernel_policy"]): cell for cell in cells}
        comparisons: dict[str, Any] = {}
        for activation in ("f32", "q8_k_compat"):
            buffered_summary = by_cell[("ephemeral", "buffered", activation, "baseline")]["summary"]
            mmap_summary = by_cell[("ephemeral", "mmap", activation, "baseline")]["summary"]
            activation_comparison: dict[str, float] = {}
            for field in ("total_latency_seconds", "prefill_latency_seconds", "decode_latency_seconds"):
                activation_comparison[f"{field}_speedup_buffered_over_mmap"] = (
                    buffered_summary[field]["median"] / mmap_summary[field]["median"]
                )
            activation_comparison["peak_rss_median_delta_bytes_mmap_minus_buffered"] = (
                mmap_summary["peak_rss_bytes"]["median"] - buffered_summary["peak_rss_bytes"]["median"]
            )
            for io_backend in ("buffered", "mmap"):
                ephemeral_summary = by_cell[("ephemeral", io_backend, activation, "baseline")]["summary"]
                persistent_summary = by_cell[("persistent", io_backend, activation, "baseline")]["summary"]
                activation_comparison[f"allocation_malloc_median_delta_ephemeral_minus_persistent_{io_backend}"] = (
                    ephemeral_summary["allocation_malloc_calls"]["median"] - persistent_summary["allocation_malloc_calls"]["median"]
                )
                activation_comparison[f"allocation_free_median_delta_ephemeral_minus_persistent_{io_backend}"] = (
                    ephemeral_summary["allocation_free_calls"]["median"] - persistent_summary["allocation_free_calls"]["median"]
                )
            comparisons[activation] = activation_comparison

        report = {
            "schema": 4,
            "measurement": "persistent-scratch-2x2x2-baseline",
            "status": "pass",
            "claim_scope": "Pinned CPU, model, QXF, tokenizer, prompt, revision and arguments only; not global throughput or model/logit parity.",
            "provenance": {
                "source_state": source_state(),
                "artifacts": build_artifact_provenance(
                    benchmark_script=Path(__file__), qx_exe=args.qx_exe,
                    source_model_gguf=args.source_model,
                    model_qxf=args.model, tokenizer_qxt=args.tokenizer,
                    prompt=args.prompt_file,
                ),
                "platform": {
                    "system": platform.system(), "release": platform.release(),
                    "machine": platform.machine(), "processor": platform.processor(),
                    "python": platform.python_version(),
                },
                "fixed_arguments": {
                    "scratch_policies": ["ephemeral", "persistent"],
                    "kernel_policies": ["baseline", "fused"],
                    "thread_policy": "serial",
                    "threads": 1,
                    "activation_modes": ["f32", "q8_k_compat"],
                    "io_backends": ["buffered", "mmap"],
                    "kv": args.kv, "layers": args.layers, "ctx": args.ctx,
                    "generation_steps": args.generate, "seed": args.seed,
                    "long_context_policy": args.long_context_policy,
                    "long_context_rss_limit_bytes": args.long_context_rss_limit_bytes,
                    "long_context_kv_quality_checks": args.long_context_kv_quality_checks,
                    "long_context_soak_seconds": args.long_context_soak_seconds,
                    "temperature": 0, "warmups": args.warmups,
                    "repetitions": args.repetitions,
                },
            },
            "phase_definitions": {
                "startup_model_load": "Wall-clock qxqxf inspect process for the same QXF; reported separately and not subtracted from inference.",
                "prefill": "Native C clock() elapsed sum for fixed prompt positions before the first generation output, including per-position JSON emission.",
                "decode": "Native C clock() elapsed sum for generation positions including the complete final head and per-position JSON emission.",
                "total": "Wall-clock prompt-state-loop process including tokenizer/model open, allocation, prefill, decode, JSON emission and teardown.",
                "peak_rss": "Maximum process resident set sampled by psutil at 5 ms intervals.",
            },
            "startup_model_load": startup_cells,
            "cells": cells,
            "long_context_profile": summarize_cells_long_context_profile(cells),
            "long_context_measurement": build_long_context_measurement_gate(cells, ctx=args.ctx),
            "output_contract": output_gate,
            "kernel_policy_effects": kernel_policy_effects,
            "comparisons": comparisons,
        }
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        print(f"benchmark failed closed: {exc}", file=sys.stderr)
        return 2

    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": "pass", "output": str(args.output),
        "report_sha256": hashlib.sha256(encoded).hexdigest(),
        "selected_tokens_by_mode": output_gate["selected_tokens_by_mode"],
        "kernel_policy_effects": kernel_policy_effects,
        "comparisons": comparisons,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())