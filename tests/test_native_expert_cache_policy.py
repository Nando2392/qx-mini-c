from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
QXQXF = ROOT / "build" / "qxqxf.exe"
FILE_IO_ERROR = "prompt-state-loop-probe failed: cannot open text file\n"


def command(*extra: str) -> list[str]:
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
        "f32",
        "--io-backend",
        "buffered",
        "--scratch-policy",
        "ephemeral",
        "--kernel-policy",
        "baseline",
        "--thread-policy",
        "serial",
        "--threads",
        "1",
        "--simd-policy",
        "scalar",
        "--cuda-policy",
        "none",
        "--prefill-gemm-policy",
        "none",
        "--speculative-policy",
        "none",
        "--kv2-policy",
        "none",
        "--sampling-policy",
        "none",
        "--long-context-policy",
        "none",
        "--temperature",
        "0",
        "--seed",
        "7",
        "--full-moe",
        "--final-head",
        *extra,
    ]


def run(*extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command(*extra),
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
    )


@pytest.fixture(scope="module", autouse=True)
def require_native_executable() -> None:
    if not QXQXF.is_file():
        pytest.skip("qxqxf.exe must be built before native expert-cache policy tests")


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--expert-cache-policy", "none"],
    ],
    ids=["implicit-none", "explicit-none"],
)
def test_none_policy_preserves_default_path_to_file_io(arguments: list[str]) -> None:
    completed = run(*arguments)

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert completed.stderr == FILE_IO_ERROR


def test_unknown_expert_cache_policy_fails_before_file_io() -> None:
    completed = run("--expert-cache-policy", "resident")

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "prompt-state-loop-probe failed: unsupported expert cache policy\n"


@pytest.mark.parametrize(
    "budget",
    ["-1", "invalid", "1byte", "18446744073709551616"],
    ids=["negative", "non-numeric", "trailing-junk", "uint64-overflow"],
)
def test_invalid_expert_cache_budget_fails_before_file_io(budget: str) -> None:
    completed = run(
        "--expert-cache-policy",
        "resident-packed",
        "--expert-cache-budget-bytes",
        budget,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "prompt-state-loop-probe failed: invalid --expert-cache-budget-bytes\n"


@pytest.mark.parametrize(
    "budget_arguments",
    [[], ["--expert-cache-budget-bytes", "0"]],
    ids=["omitted", "explicit-zero"],
)
def test_resident_packed_rejects_zero_budget_before_file_io(
    budget_arguments: list[str],
) -> None:
    completed = run(
        "--expert-cache-policy",
        "resident-packed",
        *budget_arguments,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == (
        "prompt-state-loop-probe failed: resident-packed expert cache policy requires "
        "--expert-cache-budget-bytes > 0\n"
    )


def test_none_policy_rejects_nonzero_budget_before_file_io() -> None:
    completed = run(
        "--expert-cache-policy",
        "none",
        "--expert-cache-budget-bytes",
        "4096",
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == (
        "prompt-state-loop-probe failed: expert cache budget requires resident-packed policy\n"
    )


def test_mmap_resident_packed_fails_before_file_io() -> None:
    completed = run(
        "--io-backend",
        "mmap",
        "--expert-cache-policy",
        "resident-packed",
        "--expert-cache-budget-bytes",
        "4096",
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == (
        "prompt-state-loop-probe failed: resident-packed expert cache policy requires buffered I/O\n"
    )


def test_valid_resident_packed_configuration_reaches_file_io() -> None:
    completed = run(
        "--expert-cache-policy",
        "resident-packed",
        "--expert-cache-budget-bytes",
        "4096",
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert completed.stderr == FILE_IO_ERROR
