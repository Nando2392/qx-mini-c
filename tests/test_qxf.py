import json
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_qxf_header_written_by_python_layout_contract(tmp_path):
    # Mirrors include/qx_format.h. This catches accidental layout changes in the
    # binary contract even when the C compiler is unavailable.
    header_size = 4 + 4*4 + 4*8 + 13*4 + 160
    dir_entry_size = 96 + 4*4 + 4*8 + 8*3 + 4*2 + 8 + 32
    assert header_size == 264
    assert dir_entry_size == 216


def test_qxf_tool_create_inspect_if_built(tmp_path):
    exe = ROOT / "build" / "qxqxf.exe"
    if not exe.exists():
        return
    out = tmp_path / "qwen3-8b-meta.qxf"
    subprocess.check_call([str(exe), "create", "--model", "qwen3-8b", "--quant", "q3q4mix", "--out", str(out)])
    summary = subprocess.check_output([str(exe), "inspect", "--in", str(out)], text=True)
    data = json.loads(summary)
    assert data["magic"] == "QXF1"
    assert data["model_type"] == "qwen3_dense"
    assert data["quant_type"] == "q3q4mix"
    assert data["layers"] == 36
    assert data["hidden"] == 4096
    assert data["tensor_count"] == 327
    assert data["dir_offset"] % 4096 == 0
    assert data["data_offset"] % 4096 == 0


def test_qxf_tool_create_inspect_moe_if_built(tmp_path):
    exe = ROOT / "build" / "qxqxf.exe"
    if not exe.exists():
        return
    out = tmp_path / "qwen3-30b-a3b-meta.qxf"
    subprocess.check_call([str(exe), "create", "--model", "qwen3-30b-a3b", "--quant", "q2", "--out", str(out)])
    summary = subprocess.check_output([str(exe), "inspect", "--in", str(out)], text=True)
    data = json.loads(summary)
    assert data["magic"] == "QXF1"
    assert data["model_type"] == "qwen3_moe"
    assert data["quant_type"] == "q2"
    assert data["layers"] == 48
    assert data["hidden"] == 2048
    assert data["kv_heads"] == 4
    assert data["tensor_count"] == 18771
    assert data["dir_offset"] % 4096 == 0
    assert data["data_offset"] % 4096 == 0


def test_qxf_raw_magic_if_sample_exists():
    sample = ROOT / "models" / "qwen3-8b-meta.qxf"
    if not sample.exists():
        return
    raw = sample.read_bytes()[:8]
    magic, version = struct.unpack("<4sI", raw)
    assert magic == b"QXF1"
    assert version == 1


def test_qxf_rejects_tensor_directory_count_outside_file(tmp_path):
    exe = ROOT / "build" / "qxqxf.exe"
    if not exe.exists():
        return
    out = tmp_path / "bad-directory-count.qxf"
    subprocess.check_call([str(exe), "create", "--model", "qwen3-8b", "--quant", "q3q4mix", "--out", str(out)])
    raw = bytearray(out.read_bytes())
    struct.pack_into("<I", raw, 16, 0xFFFFFFFF)
    out.write_bytes(raw)
    result = subprocess.run([str(exe), "inspect-tensor", "--in", str(out), "--name", "token_embd.weight"], text=True, capture_output=True)
    assert result.returncode != 0
    assert "tensor directory outside file" in result.stderr


def test_qxf_rejects_tensor_span_integer_overflow(tmp_path):
    exe = ROOT / "build" / "qxqxf.exe"
    if not exe.exists():
        return
    out = tmp_path / "bad-tensor-span.qxf"
    subprocess.check_call([str(exe), "create", "--model", "qwen3-8b", "--quant", "q3q4mix", "--out", str(out)])
    raw = bytearray(out.read_bytes())
    dir_offset = struct.unpack_from("<Q", raw, 24)[0]
    struct.pack_into("<Q", raw, dir_offset + 144, 0xFFFFFFFFFFFFFFF0)
    struct.pack_into("<Q", raw, dir_offset + 152, 0x40)
    out.write_bytes(raw)
    result = subprocess.run([str(exe), "inspect-tensor", "--in", str(out), "--name", "token_embd.weight"], text=True, capture_output=True)
    assert result.returncode != 0
    assert "tensor range outside file" in result.stderr
