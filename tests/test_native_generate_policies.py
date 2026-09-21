from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
QXQXF = ROOT / "build" / "qxqxf.exe"
MODEL = ROOT / "models" / "Qwen3-30B-A3B-UD-IQ2_M.qxf"
TOKENIZER = ROOT / "models" / "Qwen3-30B-A3B.qxt"


def require_or_skip(paths: tuple[Path, ...], purpose: str) -> None:
    missing = [path for path in paths if not path.is_file()]
    if not missing:
        return
    message = f"{purpose} requires local assets: " + ", ".join(str(path) for path in missing)
    if os.environ.get("QX_REQUIRE_NATIVE_GENERATE") == "1":
        pytest.fail(message)
    pytest.skip(message)


def missing_asset_command(*extra: str) -> list[str]:
    return [
        str(QXQXF),
        "generate",
        "--in",
        "missing-model.qxf",
        "--tokenizer",
        "missing-tokenizer.qxt",
        "--text-file",
        "missing-prompt.txt",
        "--max-tokens",
        "1",
        "--ctx",
        "16",
        *extra,
    ]


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
    )


@pytest.fixture(scope="module", autouse=True)
def native_executable() -> None:
    require_or_skip((QXQXF,), "native generate policy CLI tests")


@pytest.mark.parametrize(
    ("arguments", "expected_error"),
    [
        (["--io-backend", "direct"], "unsupported native generation I/O policy"),
        (["--scratch-policy", "reused"], "unsupported native generation scratch policy"),
        (["--kernel-policy", "auto"], "unsupported native generation kernel policy"),
        (["--thread-policy", "auto"], "unsupported native generation thread policy"),
        (["--thread-policy", "serial", "--threads", "2"], "serial native generation policy requires one thread"),
        (["--thread-policy", "pool", "--threads", "1"], "pool native generation policy requires 2..64 threads"),
        (["--thread-policy", "pool", "--threads", "65"], "pool native generation policy requires 2..64 threads"),
        (["--thread-policy", "pool"], "pool native generation policy requires 2..64 threads"),
    ],
)
def test_generate_rejects_invalid_policies_before_tokenizer_or_model_io(
    arguments: list[str], expected_error: str
) -> None:
    completed = run(missing_asset_command(*arguments))

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == f"generate failed: {expected_error}\n"
    assert "missing-tokenizer" not in completed.stderr
    assert "missing-model" not in completed.stderr


@pytest.mark.parametrize("threads", ["-1", "4294967296", "2junk", "two"])
def test_generate_rejects_invalid_thread_counts_before_file_io(threads: str) -> None:
    completed = run(missing_asset_command("--thread-policy", "pool", "--threads", threads))

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "generate failed: invalid --threads\n"


def real_command(prompt: Path, *extra: str) -> list[str]:
    return [
        str(QXQXF),
        "generate",
        "--in",
        str(MODEL),
        "--tokenizer",
        str(TOKENIZER),
        "--text-file",
        str(prompt),
        "--max-tokens",
        "1",
        "--ctx",
        "2",
        *extra,
    ]


def test_generate_policy_flags_preserve_legacy_json_without_profile(tmp_path: Path) -> None:
    require_or_skip((MODEL, TOKENIZER), "native generate policy acceptance test")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Hello!", encoding="utf-8", newline="")

    completed = run(real_command(prompt, "--io-backend", "buffered", "--scratch-policy", "ephemeral"))

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert set(payload) == {
        "prompt_token_count",
        "generated_token_ids",
        "generated_text",
        "stop_reason",
        "timing",
    }


def test_generate_execution_profile_reports_requested_effective_and_actual_counters(
    tmp_path: Path,
) -> None:
    require_or_skip((MODEL, TOKENIZER), "native generate execution-profile acceptance test")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Hello!", encoding="utf-8", newline="")

    completed = run(
        real_command(
            prompt,
            "--io-backend",
            "mmap",
            "--scratch-policy",
            "persistent",
            "--kernel-policy",
            "fused",
            "--thread-policy",
            "pool",
            "--threads",
            "2",
            "--execution-profile",
        )
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert set(payload) == {
        "prompt_token_count",
        "generated_token_ids",
        "generated_text",
        "stop_reason",
        "timing",
        "execution_profile",
    }
    profile = payload["execution_profile"]
    assert set(profile) == {
        "requested_io_backend",
        "effective_io_backend",
        "requested_scratch_policy",
        "effective_scratch_policy",
        "requested_kernel_policy",
        "effective_kernel_policy",
        "requested_thread_policy",
        "effective_thread_policy",
        "requested_thread_count",
        "effective_thread_count",
        "sampled_steps",
        "workers_used",
        "scratch_peak_capacity_bytes",
        "scratch_growth_events",
        "temporary_blocks_decoded",
        "temporary_floats_materialized",
        "temporary_bytes_materialized",
        "fused_final_head_dot_calls",
        "baseline_final_head_dot_calls",
        "final_head_q6_k_blocks",
        "final_head_parallel_jobs",
        "final_head_serial_jobs",
        "final_head_fallback_jobs",
        "full_logits_checksums",
    }
    assert profile["requested_io_backend"] == profile["effective_io_backend"] == "mmap"
    assert profile["requested_scratch_policy"] == profile["effective_scratch_policy"] == "persistent"
    assert profile["requested_kernel_policy"] == profile["effective_kernel_policy"] == "fused"
    assert profile["requested_thread_policy"] == profile["effective_thread_policy"] == "pool"
    assert profile["requested_thread_count"] == profile["effective_thread_count"] == 2
    assert profile["sampled_steps"] == len(payload["generated_token_ids"]) == 1
    assert profile["workers_used"] >= 1
    assert profile["scratch_peak_capacity_bytes"] > 0
    assert profile["fused_final_head_dot_calls"] > 0
    assert profile["baseline_final_head_dot_calls"] == 0
    assert profile["final_head_parallel_jobs"] > 0
    assert profile["final_head_serial_jobs"] == 0
    assert profile["full_logits_checksums"]
    assert all(isinstance(checksum, str) and checksum.isdecimal() for checksum in profile["full_logits_checksums"])
