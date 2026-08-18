import json
import os
import struct
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
QX_EXE = ROOT / "build" / "qxqxf.exe"
QXF = ROOT / "models" / "Qwen3-30B-A3B-UD-IQ2_M.qxf"
GGML_EXE = ROOT / "build" / "ggml_reference_decode.exe"
GGML_BUILD = ROOT / "tests" / "build_ggml_reference.bat"
LLAMA_CPP_DIR = Path(os.environ.get("LLAMA_CPP_DIR", ROOT.parent / "llama.cpp-k3"))


def fnv1a64(data: bytes) -> int:
    value = 1469598103934665603
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def assert_oracle_match(qx: dict, oracle_bytes: bytes, tensor: str, block: int) -> None:
    values = struct.unpack("<256f", oracle_bytes)
    assert qx["f32_checksum"] == fnv1a64(oracle_bytes), (
        f"full 256-float checksum mismatch: {tensor} block {block}"
    )
    assert max(abs(a - b) for a, b in zip(qx["first8"], values[:8])) <= 1e-8
    assert abs(qx["sum"] - sum(values)) <= 1e-6
    assert abs(qx["min"] - min(values)) <= 1e-8
    assert abs(qx["max"] - max(values)) <= 1e-8


def build_ggml_reference() -> None:
    if os.name != "nt" or not (LLAMA_CPP_DIR / "ggml" / "include" / "ggml.h").exists():
        pytest.skip("pinned Windows llama.cpp oracle checkout is not available")
    env = os.environ.copy()
    env["LLAMA_CPP_DIR"] = str(LLAMA_CPP_DIR)
    build = subprocess.run(
        ["cmd.exe", "/c", str(GGML_BUILD)], cwd=ROOT, env=env, text=True, capture_output=True
    )
    assert build.returncode == 0, build.stdout + build.stderr


def test_oracle_match_rejects_checksum_mismatch():
    oracle = struct.pack("<256f", *([0.0] * 256))
    qx = {
        "f32_checksum": fnv1a64(oracle) ^ 1,
        "first8": [0.0] * 8,
        "sum": 0.0,
        "min": 0.0,
        "max": 0.0,
    }

    with pytest.raises(AssertionError, match="full 256-float checksum mismatch"):
        assert_oracle_match(qx, oracle, "synthetic", 0)


def test_ggml_reference_helper_rejects_unsupported_and_truncated_inputs(tmp_path):
    build_ggml_reference()
    truncated_q5 = tmp_path / "truncated-q5-k.bin"
    short_activation = tmp_path / "short-activation.f32"
    truncated_q5.write_bytes(b"\x00" * 175)
    short_activation.write_bytes(struct.pack("<255f", *([0.0] * 255)))

    unsupported = subprocess.run(
        [str(GGML_EXE), "unsupported", str(truncated_q5), "0", "1"], capture_output=True
    )
    truncated = subprocess.run(
        [str(GGML_EXE), "q5_k", str(truncated_q5), "0", "1"], capture_output=True
    )
    short = subprocess.run(
        [str(GGML_EXE), "q8_k_quantize", str(short_activation)], capture_output=True
    )

    assert unsupported.returncode == 2
    assert truncated.returncode == 3
    assert short.returncode == 3
    assert unsupported.stdout == truncated.stdout == short.stdout == b""


def test_q5_k_decoder_matches_full_public_ggml_oracle_blocks():
    if os.name != "nt" or not QX_EXE.exists() or not QXF.exists():
        pytest.skip("local Windows Qwen fixtures are not available")
    build_ggml_reference()

    cases = (
        ("blk.1.attn_v.weight", 0),
        ("blk.1.attn_v.weight", 7),
        ("blk.1.attn_output.weight", 0),
        ("blk.1.attn_output.weight", 15),
    )
    for tensor, block in cases:
        qx = json.loads(
            subprocess.check_output(
                [
                    str(QX_EXE),
                    "decode-block",
                    "--in",
                    str(QXF),
                    "--name",
                    tensor,
                    "--block",
                    str(block),
                ],
                text=True,
            )
        )
        oracle = subprocess.run(
            [str(GGML_EXE), "q5_k", str(QXF), str(qx["block_offset"]), "1"],
            capture_output=True,
        )
        assert oracle.returncode == 0, oracle.stderr.decode(errors="replace")
        assert_oracle_match(qx, oracle.stdout, tensor, block)
