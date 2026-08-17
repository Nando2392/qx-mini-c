import json
import math
import struct
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / "qxqxf.exe"
MODEL = ROOT / "models" / "Qwen3-30B-A3B-UD-IQ2_M.qxf"


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def rope(values, heads, head_dim, position, theta=1_000_000.0):
    out = list(values)
    half = head_dim // 2
    for head in range(heads):
        base = head * head_dim
        for index in range(half):
            angle = position * theta ** (-(2.0 * index) / head_dim)
            cosine, sine = math.cos(angle), math.sin(angle)
            left, right = out[base + index], out[base + index + half]
            out[base + index] = f32(left * cosine - right * sine)
            out[base + index + half] = f32(left * sine + right * cosine)
    return out


def quantize(values):
    scale = f32(max(abs(value) for value in values) / 127.0)
    return [max(-127, min(127, round(value / scale))) for value in values], scale


def head_rmsnorm(values, heads, head_dim, weights, epsilon=1e-6):
    out = list(values)
    for head in range(heads):
        base = head * head_dim
        mean_square = sum(float(out[base + dim]) ** 2 for dim in range(head_dim)) / head_dim
        inverse_rms = 1.0 / math.sqrt(mean_square + epsilon)
        for dim in range(head_dim):
            out[base + dim] = f32(float(out[base + dim]) * inverse_rms * float(weights[dim]))
    return out


def external_attention(payload):
    head_dim = payload["head_dim"]
    q_heads_run = payload["q_heads_run"]
    kv_heads = payload["kv_heads_total"]
    group_size = payload["gqa_group_size"]
    q = rope(head_rmsnorm(payload["q_raw"], q_heads_run, head_dim, payload["q_norm_raw"]), q_heads_run, head_dim, 1)
    keys, values, kscales, vscales = [], [], [], []
    for position in range(2):
        key = rope(head_rmsnorm(payload["k_raw"][position], kv_heads, head_dim, payload["k_norm_raw"]), kv_heads, head_dim, position)
        key_q, key_scale = quantize(key)
        value_q, value_scale = quantize(payload["v_raw"][position])
        keys.append(key_q)
        values.append(value_q)
        kscales.append(key_scale)
        vscales.append(value_scale)
    scores, weights = [], []
    context = [0.0] * (q_heads_run * head_dim)
    for q_head in range(q_heads_run):
        kv_head = q_head // group_size
        head_scores = [
            sum(q[q_head * head_dim + dim] * keys[token][kv_head * head_dim + dim] * kscales[token] for dim in range(head_dim)) / math.sqrt(head_dim)
            for token in range(2)
        ]
        maximum = max(head_scores)
        exponentials = [math.exp(value - maximum) for value in head_scores]
        denominator = sum(exponentials)
        head_weights = [value / denominator for value in exponentials]
        scores.append(head_scores)
        weights.append(head_weights)
        for token, weight in enumerate(head_weights):
            for dim in range(head_dim):
                context[q_head * head_dim + dim] += weight * values[token][kv_head * head_dim + dim] * vscales[token]
    return (
        [scores[0][0], scores[0][1], scores[8][0], scores[8][1]],
        [weights[0][0], weights[0][1], weights[8][0], weights[8][1]],
        [context[0], context[64], context[1024], context[1088]],
    )


def test_real_qwen_qkv_matches_external_rope_gqa_reference():
    if not EXE.exists() or not MODEL.exists():
        pytest.skip("real Qwen QXF is not available")
    payload = json.loads(
        subprocess.check_output(
            [str(EXE), "real-qkv-golden-probe", "--in", str(MODEL), "--layer", "0", "--token-a", "42", "--token-b", "43", "--q-heads-run", "9", "--seed", "7"],
            text=True,
        )
    )
    assert payload["probe"] == "real_qkv_golden"
    assert payload["projection_layout"] == "contiguous_tensor_rows"
    assert payload["projection_input_dims"] == 2048
    assert payload["projection_blocks_per_row"] == 8
    assert len(payload["q_raw"]) == 1152
    assert len(payload["k_raw"]) == 2 and all(len(row) == 512 for row in payload["k_raw"])
    assert len(payload["v_raw"]) == 2 and all(len(row) == 512 for row in payload["v_raw"])
    assert payload["q_norm_tensor"] == "blk.0.attn_q_norm.weight"
    assert payload["k_norm_tensor"] == "blk.0.attn_k_norm.weight"
    assert len(payload["q_norm_raw"]) == 128
    assert len(payload["k_norm_raw"]) == 128
    scores, weights, context = external_attention(payload)
    for got, expected in zip(payload["score_samples"], scores):
        assert math.isclose(got, expected, rel_tol=5e-6, abs_tol=5e-6)
    for got, expected in zip(payload["weight_samples"], weights):
        assert math.isclose(got, expected, rel_tol=5e-6, abs_tol=5e-6)
    for got, expected in zip(payload["context_samples"], context):
        assert math.isclose(got, expected, rel_tol=5e-6, abs_tol=5e-6)
