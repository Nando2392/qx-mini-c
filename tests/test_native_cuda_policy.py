from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
QXQXF = ROOT / "build" / "qxqxf.exe"
MISSING_MODEL = "cuda-policy-missing-model.qxf"
MISSING_TOKENIZER = "cuda-policy-missing-tokenizer.qxt"
MISSING_PROMPT = "cuda-policy-missing-prompt.txt"


@pytest.fixture(scope="module", autouse=True)
def native_executable() -> None:
    if not QXQXF.is_file():
        pytest.skip("qxqxf.exe must be built before native CUDA policy CLI tests")


def run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(QXQXF), *arguments],
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
    )


def generate_command(*extra: str) -> tuple[str, ...]:
    return (
        "generate",
        "--in",
        MISSING_MODEL,
        "--tokenizer",
        MISSING_TOKENIZER,
        "--text-file",
        MISSING_PROMPT,
        "--max-tokens",
        "1",
        "--ctx",
        "16",
        *extra,
    )


def prompt_state_command(*, activation: str, cuda_policy: str) -> tuple[str, ...]:
    return (
        "prompt-state-loop-probe",
        "--in",
        MISSING_MODEL,
        "--tokenizer",
        MISSING_TOKENIZER,
        "--text-file",
        MISSING_PROMPT,
        "--generate",
        "1",
        "--layers",
        "1",
        "--ctx",
        "16",
        "--kv",
        "int8",
        "--activation",
        activation,
        "--cuda-policy",
        cuda_policy,
        "--full-moe",
        "--final-head",
    )


def assert_rejected_before_file_io(completed: subprocess.CompletedProcess[str]) -> None:
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr
    for missing_path in (MISSING_MODEL, MISSING_TOKENIZER, MISSING_PROMPT):
        assert missing_path not in completed.stderr


def test_generate_explicit_cuda_policy_none_preserves_existing_behavior() -> None:
    control = run(*generate_command())
    explicit_none = run(*generate_command("--cuda-policy", "none"))

    assert explicit_none.returncode == control.returncode
    assert explicit_none.stdout == control.stdout
    assert explicit_none.stderr == control.stderr


def test_generate_rejects_unsupported_cuda_policy_before_file_io() -> None:
    completed = run(*generate_command("--cuda-policy", "not-a-cuda-policy"))

    assert_rejected_before_file_io(completed)
    assert "cuda" in completed.stderr.lower()


def test_generate_final_head_f32_rejects_cpu_only_backend_before_file_io() -> None:
    unsupported = run(*generate_command("--cuda-policy", "not-a-cuda-policy"))
    completed = run(*generate_command("--cuda-policy", "final-head-f32"))

    assert_rejected_before_file_io(completed)
    assert "cuda" in completed.stderr.lower()
    assert completed.stderr != unsupported.stderr


def test_prompt_state_rejects_q8_activation_for_f32_only_cuda_backend_before_file_io() -> None:
    f32_control = run(*prompt_state_command(activation="f32", cuda_policy="final-head-f32"))
    completed = run(
        *prompt_state_command(activation="q8_k_compat", cuda_policy="final-head-f32")
    )

    assert_rejected_before_file_io(completed)
    assert "activation" in completed.stderr.lower()
    assert completed.stderr != f32_control.stderr
