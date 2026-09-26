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

BASE_PAYLOAD_KEYS = {
    "prompt_token_count",
    "generated_token_ids",
    "generated_text",
    "stop_reason",
    "timing",
}


def require_or_skip(paths: tuple[Path, ...], purpose: str) -> None:
    missing = [path for path in paths if not path.is_file()]
    if not missing:
        return
    message = f"{purpose} requires local assets: " + ", ".join(str(path) for path in missing)
    if os.environ.get("QX_REQUIRE_NATIVE_GENERATE") == "1":
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture(scope="module", autouse=True)
def native_generate_assets() -> None:
    require_or_skip((QXQXF, TOKENIZER), "native expert-cache CLI profile tests")


@pytest.mark.parametrize(
    "profile_flags",
    [
        (),
        ("--capacity-profile",),
        ("--execution-profile",),
        ("--execution-profile", "--capacity-profile"),
    ],
    ids=["default", "capacity-only", "execution-only", "both"],
)
def test_generate_profile_flag_combinations_reach_model_io(
    tmp_path: Path, profile_flags: tuple[str, ...]
) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Hi", encoding="utf-8", newline="")
    missing_model = tmp_path / "missing-model.qxf"

    completed = subprocess.run(
        [
            str(QXQXF),
            "generate",
            "--in",
            str(missing_model),
            "--tokenizer",
            str(TOKENIZER),
            "--text-file",
            str(prompt),
            "--max-tokens",
            "2",
            "--ctx",
            "16",
            *profile_flags,
        ],
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
        timeout=120,
    )

    assert completed.returncode == 1
    assert "failed to open" in completed.stderr or "No such file or directory" in completed.stderr
    assert "profile checksum capacity is smaller than max_tokens" not in completed.stderr


@pytest.mark.parametrize(
    ("profile_flags", "expected_extra_keys"),
    [
        ((), set()),
        (("--capacity-profile",), {"capacity_profile"}),
        (("--execution-profile",), {"execution_profile"}),
        (
            ("--execution-profile", "--capacity-profile"),
            {"execution_profile", "capacity_profile"},
        ),
        (
            (
                "--expert-cache-policy",
                "resident-packed",
                "--expert-cache-budget-bytes",
                "616464384",
            ),
            {"expert_cache_profile"},
        ),
    ],
    ids=["default", "capacity-only", "execution-only", "both", "resident-packed"],
)
def test_generate_profile_flag_combinations_preserve_exact_payload_keys(
    tmp_path: Path,
    profile_flags: tuple[str, ...],
    expected_extra_keys: set[str],
) -> None:
    require_or_skip((MODEL, TOKENIZER), "native expert-cache CLI profile compatibility tests")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Hello!", encoding="utf-8", newline="")

    completed = subprocess.run(
        [
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
            *profile_flags,
        ],
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert set(json.loads(completed.stdout)) == BASE_PAYLOAD_KEYS | expected_extra_keys
