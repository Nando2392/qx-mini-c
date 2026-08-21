from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
QXQXF = ROOT / "build" / "qxqxf.exe"


def prompt_state_command(threads: str, *, thread_policy: str = "serial", activation: str = "f32") -> list[str]:
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
        "--thread-policy",
        thread_policy,
        "--threads",
        threads,
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
