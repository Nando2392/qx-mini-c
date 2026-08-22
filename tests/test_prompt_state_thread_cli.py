from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
QXQXF = ROOT / "build" / "qxqxf.exe"


def prompt_state_command(
    threads: str, *, thread_policy: str = "serial", activation: str = "f32",
    kernel_policy: str = "baseline", simd_policy: str = "scalar", expert_cache_policy: str = "none",
    cuda_policy: str = "none",
    prefill_gemm_policy: str = "none",
    speculative_policy: str = "none",
    kv2_policy: str = "none",
    sampling_policy: str = "none",
) -> list[str]:
    return [
        str(QXQXF),
        "prompt-state-loop-probe",
        "--in",
        "missing-model.qxf",
        "--tokenizer",
        "missing-tokenizer.qxt",
        "--text-file",
        "missing-prompt.txt",
        "--generate",
        "1",
        "--layers",
        "48",
        "--ctx",
        "16",
        "--kv",
        "int8",
        "--activation",
        activation,
        "--kernel-policy",
        kernel_policy,
        "--thread-policy",
        thread_policy,
        "--threads",
        threads,
        "--simd-policy",
        simd_policy,
        "--expert-cache-policy",
        expert_cache_policy,
        "--cuda-policy",
        cuda_policy,
        "--prefill-gemm-policy",
        prefill_gemm_policy,
        "--speculative-policy",
        speculative_policy,
        "--kv2-policy",
        kv2_policy,
        "--sampling-policy",
        sampling_policy,
        "--temperature",
        "0",
        "--seed",
        "7",
        "--full-moe",
        "--final-head",
    ]


@pytest.mark.skipif(not QXQXF.exists(), reason="qxqxf.exe must be built before CLI tests")
@pytest.mark.parametrize("threads", ["abc", "1abc", "4294967296"])
def test_prompt_state_loop_probe_rejects_malformed_threads_before_file_io(threads: str):
    result = subprocess.run(prompt_state_command(threads), cwd=ROOT, capture_output=True, text=True)

    assert result.returncode == 2
    assert "invalid --threads" in result.stderr
    assert "text file read failed" not in result.stderr
    assert "failed to open" not in result.stderr


@pytest.mark.skipif(not QXQXF.exists(), reason="qxqxf.exe must be built before CLI tests")
@pytest.mark.parametrize(
    ("thread_policy", "threads", "activation", "message"),
    [
        ("pool", "1", "f32", "thread pool policy requires --threads >= 2"),
        ("pool", "65", "f32", "thread pool policy supports at most 64 threads"),
        ("pool", "2", "q8_k_compat", "thread pool policy currently requires F32 activation"),
        ("serial", "2", "f32", "serial thread policy requires --threads 1"),
    ],
)
def test_prompt_state_loop_probe_rejects_thread_policy_contract_before_file_io(
    thread_policy: str, threads: str, activation: str, message: str
):
    result = subprocess.run(
        prompt_state_command(threads, thread_policy=thread_policy, activation=activation),
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert "text file read failed" not in result.stderr
    assert "failed to open" not in result.stderr


@pytest.mark.skipif(not QXQXF.exists(), reason="qxqxf.exe must be built before CLI tests")
@pytest.mark.parametrize(
    ("kernel_policy", "activation", "thread_policy", "threads", "simd_policy", "message"),
    [
        ("baseline", "f32", "serial", "1", "avx2-fma", "avx2-fma simd policy requires --kernel-policy fused"),
        ("fused", "q8_k_compat", "serial", "1", "avx2-fma", "avx2-fma simd policy requires F32 activation"),
        ("fused", "f32", "pool", "2", "avx2-fma", "avx2-fma simd policy currently requires serial thread policy"),
        ("fused", "f32", "serial", "1", "neon", "unsupported simd policy"),
    ],
)
def test_prompt_state_loop_probe_rejects_simd_policy_contract_before_file_io(
    kernel_policy: str, activation: str, thread_policy: str, threads: str, simd_policy: str, message: str
):
    result = subprocess.run(
        prompt_state_command(
            threads,
            thread_policy=thread_policy,
            activation=activation,
            kernel_policy=kernel_policy,
            simd_policy=simd_policy,
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert "text file read failed" not in result.stderr
    assert "failed to open" not in result.stderr


@pytest.mark.skipif(not QXQXF.exists(), reason="qxqxf.exe must be built before CLI tests")
def test_prompt_state_loop_probe_rejects_expert_cache_policy_contract_before_file_io():
    result = subprocess.run(
        prompt_state_command("1", expert_cache_policy="resident"),
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unsupported expert cache policy" in result.stderr
    assert "text file read failed" not in result.stderr
    assert "failed to open" not in result.stderr


@pytest.mark.skipif(not QXQXF.exists(), reason="qxqxf.exe must be built before CLI tests")
def test_prompt_state_loop_probe_rejects_cuda_policy_contract_before_file_io():
    result = subprocess.run(
        prompt_state_command("1", cuda_policy="hybrid"),
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unsupported CUDA policy" in result.stderr
    assert "text file read failed" not in result.stderr
    assert "failed to open" not in result.stderr


@pytest.mark.skipif(not QXQXF.exists(), reason="qxqxf.exe must be built before CLI tests")
def test_prompt_state_loop_probe_rejects_prefill_gemm_policy_contract_before_file_io():
    result = subprocess.run(
        prompt_state_command("1", prefill_gemm_policy="batched"),
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unsupported prefill GEMM policy" in result.stderr
    assert "text file read failed" not in result.stderr
    assert "failed to open" not in result.stderr


@pytest.mark.skipif(not QXQXF.exists(), reason="qxqxf.exe must be built before CLI tests")
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"speculative_policy": "draft"}, "unsupported speculative policy"),
        ({"kv2_policy": "packed"}, "unsupported KV2 policy"),
    ],
)
def test_prompt_state_loop_probe_rejects_speculative_kv2_policy_contract_before_file_io(kwargs, message: str):
    result = subprocess.run(
        prompt_state_command("1", **kwargs),
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert "text file read failed" not in result.stderr
    assert "failed to open" not in result.stderr

@pytest.mark.skipif(not QXQXF.exists(), reason="qxqxf.exe must be built before CLI tests")
def test_prompt_state_loop_probe_rejects_sampling_policy_contract_before_file_io():
    result = subprocess.run(
        prompt_state_command("1", sampling_policy="top-p"),
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unsupported sampling policy" in result.stderr
    assert "text file read failed" not in result.stderr
    assert "failed to open" not in result.stderr
