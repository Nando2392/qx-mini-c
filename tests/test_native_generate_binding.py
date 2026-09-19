from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import pytest

from native_generate_helpers import (
    QXT_HEADER,
    copy_qxt_with_eos,
    copy_qxt_with_token_pieces,
    write_mini_qxt,
)


ROOT = Path(__file__).resolve().parents[1]
QXQXF = ROOT / "build" / "qxqxf.exe"
TOKENIZER = ROOT / "models" / "Qwen3-30B-A3B.qxt"


def require_native_binding_assets() -> None:
    missing = [path for path in (QXQXF, TOKENIZER) if not path.is_file()]
    if missing:
        pytest.skip("native tokenizer binding tests require: " + ", ".join(map(str, missing)))


def generate_with_missing_model(tokenizer: Path, prompt: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(QXQXF),
            "generate",
            "--in",
            str(prompt.parent / "missing-model.qxf"),
            "--tokenizer",
            str(tokenizer),
            "--text-file",
            str(prompt),
            "--max-tokens",
            "2",
            "--ctx",
            "16",
        ],
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
    )


def copy_qxt_with_header_i32(source: Path, destination: Path, offset: int, value: int) -> Path:
    raw = bytearray(source.read_bytes())
    assert QXT_HEADER.unpack_from(raw)[:2] == (b"QXT2", 2)
    struct.pack_into("<i", raw, offset, value)
    destination.write_bytes(raw)
    return destination


def test_generate_rejects_wrong_vocabulary_before_model_read(tmp_path: Path):
    require_native_binding_assets()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Hello!", encoding="utf-8", newline="")
    tokenizer = write_mini_qxt(tmp_path / "wrong-vocab.qxt", ["Hello!"])

    completed = generate_with_missing_model(tokenizer, prompt)

    assert completed.returncode == 1
    assert "tokenizer vocabulary does not match Qwen3-30B-A3B" in completed.stderr
    assert "missing-model.qxf" not in completed.stderr


def test_generate_rejects_reordered_vocabulary_payload_before_model_read(tmp_path: Path):
    require_native_binding_assets()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Hello!", encoding="utf-8", newline="")
    tokenizer = copy_qxt_with_token_pieces(
        TOKENIZER,
        tmp_path / "reordered-vocabulary.qxt",
        {358: "Ã", 1184: "©"},
        eos_token_id=151645,
    )

    completed = generate_with_missing_model(tokenizer, prompt)

    assert completed.returncode == 1
    assert "tokenizer fingerprint does not match Qwen3-30B-A3B" in completed.stderr
    assert "missing-model.qxf" not in completed.stderr


@pytest.mark.parametrize(
    ("name", "make_tokenizer", "expected_error"),
    [
        (
            "bos",
            lambda source, destination: copy_qxt_with_header_i32(source, destination, 24, 0),
            "tokenizer BOS metadata does not match Qwen3-30B-A3B",
        ),
        (
            "eos",
            lambda source, destination: copy_qxt_with_eos(source, destination, 358),
            "tokenizer EOS metadata does not match Qwen3-30B-A3B",
        ),
        (
            "flags",
            lambda source, destination: copy_qxt_with_header_i32(source, destination, 32, 1),
            "tokenizer flags do not match Qwen3-30B-A3B",
        ),
    ],
)
def test_generate_rejects_retargeted_tokenizer_semantics_before_model_read(
    tmp_path: Path, name: str, make_tokenizer, expected_error: str
):
    require_native_binding_assets()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Hello!", encoding="utf-8", newline="")
    tokenizer = make_tokenizer(TOKENIZER, tmp_path / f"altered-{name}.qxt")

    completed = generate_with_missing_model(tokenizer, prompt)

    assert completed.returncode == 1
    assert expected_error in completed.stderr
    assert "missing-model.qxf" not in completed.stderr
