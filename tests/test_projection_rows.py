import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / "qxqxf.exe"


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def scale_min_k4(index, scales):
    if index < 4:
        return scales[index] & 63, scales[index + 4] & 63
    return (scales[index + 4] & 15) | ((scales[index - 4] >> 6) << 4), (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4)


def decode_q4_k(block):
    d = struct.unpack_from("<e", block, 0)[0]
    dmin = struct.unpack_from("<e", block, 2)[0]
    scales, quants = block[4:16], block[16:144]
    out, quant_offset, scale_index = [], 0, 0
    for _ in range(0, 256, 64):
        sc1, m1 = scale_min_k4(scale_index, scales)
        sc2, m2 = scale_min_k4(scale_index + 1, scales)
        chunk = quants[quant_offset : quant_offset + 32]
        out.extend(d * sc1 * (value & 15) - dmin * m1 for value in chunk)
        out.extend(d * sc2 * (value >> 4) - dmin * m2 for value in chunk)
        quant_offset += 32
        scale_index += 2
    return out


def deterministic_inputs(seed, count):
    state, out = seed & 0xFFFFFFFF, []
    for _ in range(count):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        value = ((state >> 16) & 0xFFFF) - 32768
        out.append(f32(value / 32768.0))
    return out


def expected_rows(qxf, tensor_name, token, layer, seed, rows):
    meta = json.loads(subprocess.check_output([str(EXE), "inspect-tensor", "--in", str(qxf), "--name", tensor_name], text=True))
    assert meta["dims"][0] == 256
    assert meta["ggml_type"] == 12
    raw = qxf.read_bytes()
    expected = []
    for row in range(rows):
        block = raw[meta["offset"] + row * 144 : meta["offset"] + (row + 1) * 144]
        weights = decode_q4_k(block)
        row_seed = (seed ^ ((token * 2654435761) & 0xFFFFFFFF) ^ ((row * 2246822519) & 0xFFFFFFFF) ^ ((layer * 3266489917) & 0xFFFFFFFF)) & 0xFFFFFFFF
        inputs = deterministic_inputs(row_seed, 256)
        expected.append(f32(sum(weight * value for weight, value in zip(weights, inputs))))
    return expected


def test_projection_uses_true_contiguous_tensor_rows(tmp_path):
    if not EXE.exists():
        return
    gguf, qxf = tmp_path / "mini.gguf", tmp_path / "mini.qxf"
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "make_synthetic_gguf.py"), "--out", str(gguf)])
    subprocess.check_call([str(EXE), "create-from-gguf-copy", "--in", str(gguf), "--model", "qwen3-30b-a3b", "--quant", "q2", "--out", str(qxf)])
    actual = json.loads(subprocess.check_output([str(EXE), "projection-matvec-probe", "--in", str(qxf), "--layer", "0", "--token-id", "42", "--rows", "2", "--dims", "256", "--kv", "int8", "--seed", "7"], text=True))
    assert actual["projection_layout"] == "contiguous_tensor_rows"
    assert actual["input_dims"] == 256
    assert actual["blocks_per_row"] == 1
    expected_k = expected_rows(qxf, "blk.0.attn_k.weight", 42, 0, 7, 2)
    expected_v = expected_rows(qxf, "blk.0.attn_v.weight", 42, 0, 7 ^ 0x9E3779B9, 2)
    for got, expected in zip(actual["k_float_samples"], expected_k):
        assert math.isclose(got, expected, rel_tol=2e-6, abs_tol=2e-6)
    for got, expected in zip(actual["v_float_samples"], expected_v):
        assert math.isclose(got, expected, rel_tol=2e-6, abs_tol=2e-6)


def test_real_embedding_decodes_consecutive_blocks_before_rmsnorm():
    model = ROOT / "models" / "Qwen3-30B-A3B-UD-IQ2_M.qxf"
    if not EXE.exists() or not model.exists():
        pytest.skip("real Qwen QXF is not available")
    token, dims = 42, 512
    actual = json.loads(subprocess.check_output([str(EXE), "residual-vector-probe", "--in", str(model), "--token-id", str(token), "--norm", "blk.0.attn_norm.weight", "--dims", str(dims), "--seed", "7"], text=True))
    embedding = json.loads(subprocess.check_output([str(EXE), "inspect-tensor", "--in", str(model), "--name", "token_embd.weight"], text=True))
    norm = json.loads(subprocess.check_output([str(EXE), "inspect-tensor", "--in", str(model), "--name", "blk.0.attn_norm.weight"], text=True))
    row_bytes = 8 * 144
    with model.open("rb") as handle:
        handle.seek(embedding["offset"] + token * row_bytes)
        row = handle.read(row_bytes)
        handle.seek(norm["offset"])
        norm_raw = handle.read(512 * 4)
    decoded = decode_q4_k(row[:144]) + decode_q4_k(row[144:288])
    rms = math.sqrt(sum(value * value for value in decoded) / dims + 1e-6)
    norm_values = struct.unpack("<512f", norm_raw)
    expected = [f32(decoded[index] / rms * norm_values[index]) for index in range(8)]
    assert actual["embedding_layout"] == "contiguous_row_blocks"
    assert actual["embedding_blocks_decoded"] == 2
    for got, want in zip(actual["vector_samples"], expected):
        assert math.isclose(got, want, rel_tol=3e-6, abs_tol=3e-6)
