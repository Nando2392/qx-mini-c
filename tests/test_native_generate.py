from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from native_generate_helpers import (
    copy_qxt_with_eos,
    copy_qxt_with_token_pieces,
    write_mini_qxt,
)


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


@pytest.fixture(scope="module")
def native_executable() -> Path:
    require_or_skip((QXQXF,), "native generate CLI tests")
    return QXQXF


@pytest.fixture(scope="module")
def real_native_assets(native_executable: Path) -> tuple[Path, Path, Path]:
    require_or_skip((MODEL, TOKENIZER), "native generate real-model acceptance tests")
    return native_executable, MODEL, TOKENIZER


def generate_command(*extra: str) -> list[str]:
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
        "2",
        "--ctx",
        "16",
        *extra,
    ]


def real_generate_command(
    executable: Path,
    model: Path,
    tokenizer: Path,
    prompt: Path,
    *,
    max_tokens: int,
    ctx: int,
) -> list[str]:
    return [
        str(executable),
        "generate",
        "--in",
        str(model),
        "--tokenizer",
        str(tokenizer),
        "--text-file",
        str(prompt),
        "--max-tokens",
        str(max_tokens),
        "--ctx",
        str(ctx),
    ]


def run_generate(command: list[str]) -> dict:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
        check=True,
    )
    assert completed.stdout.endswith("\n")
    assert completed.stdout.count("\n") == 1
    return json.loads(completed.stdout)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--max-tokens", "0"),
        ("--max-tokens", "65"),
        ("--max-tokens", "-1"),
        ("--max-tokens", "4294967296"),
        ("--max-tokens", "1junk"),
        ("--max-tokens", "two"),
        ("--ctx", "0"),
        ("--ctx", "4097"),
        ("--ctx", "-1"),
        ("--ctx", "4294967296"),
        ("--ctx", "16junk"),
        ("--ctx", "sixteen"),
    ],
)
def test_generate_rejects_invalid_numeric_arguments_before_file_io(
    native_executable: Path, flag: str, value: str
):
    command = generate_command()
    command[0] = str(native_executable)
    command[command.index(flag) + 1] = value

    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, encoding="utf-8", errors="strict"
    )

    assert completed.returncode == 2
    assert f"invalid {flag}" in completed.stderr
    assert "failed to open" not in completed.stderr
    assert "text file read failed" not in completed.stderr


def test_generate_requires_all_arguments_before_file_io(native_executable: Path):
    completed = subprocess.run(
        [str(native_executable), "generate", "--max-tokens", "2", "--ctx", "16"],
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
    )

    assert completed.returncode == 2
    assert "generate requires --in, --tokenizer, --text-file, --max-tokens, and --ctx" in completed.stderr


def test_tokenizer_decode_joins_utf8_bytes_split_across_tokens(native_executable: Path, tmp_path: Path):
    # Synthetic controlled tokenizer-only fixture: "Ã" and "©" are the GPT-2
    # byte-unicode spellings of UTF-8 bytes C3 and A9, respectively.
    tokenizer = write_mini_qxt(tmp_path / "split-utf8.qxt", ["Ã", "©"])
    completed = subprocess.run(
        [str(native_executable), "tokenizer-decode", "--tokenizer", str(tokenizer), "--ids", "0,1"],
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
        check=True,
    )

    assert json.loads(completed.stdout) == {"token_count": 2, "utf8_bytes": 2, "text": "é"}


@pytest.mark.parametrize(
    ("tokens", "token_ids"),
    [
        (["Ã"], "0"),
        (["Ã", "("], "0,1"),
    ],
    ids=["incomplete", "invalid-continuation"],
)
def test_tokenizer_decode_rejects_invalid_utf8(
    native_executable: Path, tmp_path: Path, tokens: list[str], token_ids: str
):
    tokenizer = write_mini_qxt(tmp_path / "invalid-utf8.qxt", tokens)

    completed = subprocess.run(
        [str(native_executable), "tokenizer-decode", "--tokenizer", str(tokenizer), "--ids", token_ids],
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
    )

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert completed.stderr == "tokenizer-decode failed: decoded output is not valid UTF-8\n"


def test_generate_context_rejection_precedes_model_io(
    real_native_assets: tuple[Path, Path, Path], tmp_path: Path
):
    executable, _, tokenizer = real_native_assets
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Hello!", encoding="utf-8", newline="")
    command = real_generate_command(
        executable, tmp_path / "missing-model.qxf", tokenizer, prompt, max_tokens=2, ctx=2
    )

    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, encoding="utf-8", errors="strict"
    )

    assert completed.returncode == 2
    assert "prompt and generation exceed --ctx" in completed.stderr
    assert "failed to open" not in completed.stderr


def test_generate_enforces_64_forward_position_boundary_before_model_io(
    real_native_assets: tuple[Path, Path, Path], tmp_path: Path
):
    executable, _, tokenizer = real_native_assets
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Hello!", encoding="utf-8", newline="")
    missing_model = tmp_path / "missing-model.qxf"

    exact = subprocess.run(
        real_generate_command(executable, missing_model, tokenizer, prompt, max_tokens=63, ctx=64),
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
    )
    overflow = subprocess.run(
        real_generate_command(executable, missing_model, tokenizer, prompt, max_tokens=64, ctx=65),
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
    )

    assert exact.returncode == 1
    assert "No such file or directory" in exact.stderr
    assert "64 forward positions" not in exact.stderr
    assert overflow.returncode == 2
    assert "require more than 64 forward positions" in overflow.stderr
    assert "No such file or directory" not in overflow.stderr


def test_generate_real_hello_is_deterministic_qxt_decoded_and_exact_context_fit(
    real_native_assets: tuple[Path, Path, Path], tmp_path: Path
):
    executable, model, tokenizer = real_native_assets
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Hello!", encoding="utf-8", newline="")
    command = real_generate_command(executable, model, tokenizer, prompt, max_tokens=2, ctx=3)

    payload = run_generate(command)
    repeat = run_generate(command)

    assert set(payload) == {"prompt_token_count", "generated_token_ids", "generated_text", "stop_reason", "timing"}
    assert set(payload["timing"]) == {"prefill_seconds", "decode_seconds"}
    assert payload["prompt_token_count"] == 2
    assert payload["generated_token_ids"] == [358, 1184]
    assert payload["generated_token_ids"] == repeat["generated_token_ids"]
    assert payload["generated_text"] == repeat["generated_text"]
    assert payload["stop_reason"] == "max_tokens"
    assert payload["timing"]["prefill_seconds"] >= 0
    assert payload["timing"]["decode_seconds"] > 0
    assert "<token-" not in payload["generated_text"]

    decoded = subprocess.run(
        [str(executable), "tokenizer-decode", "--tokenizer", str(tokenizer), "--ids", "358,1184"],
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
        check=True,
    )
    assert payload["generated_text"] == json.loads(decoded.stdout)["text"]


@pytest.mark.parametrize("eos_token_id", [358, 1184])
def test_generate_rejects_noncanonical_eos_before_model_io(
    real_native_assets: tuple[Path, Path, Path], tmp_path: Path, eos_token_id: int
):
    executable, _, tokenizer = real_native_assets
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Hello!", encoding="utf-8", newline="")
    mismatched_tokenizer = copy_qxt_with_eos(
        tokenizer, tmp_path / f"eos-{eos_token_id}.qxt", eos_token_id
    )
    completed = subprocess.run(
        real_generate_command(
            executable,
            tmp_path / "missing-model.qxf",
            mismatched_tokenizer,
            prompt,
            max_tokens=2,
            ctx=3,
        ),
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
    )

    assert completed.returncode != 0
    assert "tokenizer EOS metadata does not match Qwen3-30B-A3B" in completed.stderr
    assert "failed to open" not in completed.stderr
    assert "No such file or directory" not in completed.stderr


def test_controlled_real_qxt_preserves_prompt_encoding_and_joins_split_utf8_tokens(
    real_native_assets: tuple[Path, Path, Path], tmp_path: Path
):
    executable, _, tokenizer = real_native_assets
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Hello!", encoding="utf-8", newline="")
    # A model-bound generate command must not use this modified tokenizer.  The
    # fixture is limited to proving that the real vocabulary can be rewritten
    # without breaking prompt tokenization and that decoding joins raw bytes
    # across token boundaries.
    split_utf8 = copy_qxt_with_token_pieces(
        tokenizer,
        tmp_path / "no-eos-split-utf8.qxt",
        {358: "Ã", 1184: "©"},
        eos_token_id=-1,
    )

    encoded = json.loads(
        subprocess.run(
            [str(executable), "tokenizer-encode", "--tokenizer", str(split_utf8), "--text-file", str(prompt)],
            cwd=ROOT,
            capture_output=True,
            encoding="utf-8",
            errors="strict",
            check=True,
        ).stdout
    )
    decoded = json.loads(
        subprocess.run(
            [str(executable), "tokenizer-decode", "--tokenizer", str(split_utf8), "--ids", "358,1184"],
            cwd=ROOT,
            capture_output=True,
            encoding="utf-8",
            errors="strict",
            check=True,
        ).stdout
    )

    assert encoded["token_ids"] == [9707, 0]
    assert decoded == {"token_count": 2, "utf8_bytes": 2, "text": "é"}
