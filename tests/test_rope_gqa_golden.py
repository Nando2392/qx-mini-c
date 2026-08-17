import json
import math
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / "qxqxf.exe"


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def apply_rope(values, heads, head_dim, position, theta=1_000_000.0):
    out = list(values)
    half = head_dim // 2
    for head in range(heads):
        base = head * head_dim
        for i in range(half):
            frequency = theta ** (-(2.0 * i) / head_dim)
            angle = position * frequency
            c, s = math.cos(angle), math.sin(angle)
            x0, x1 = out[base + i], out[base + i + half]
            out[base + i] = f32(x0 * c - x1 * s)
            out[base + i + half] = f32(x0 * s + x1 * c)
    return out


def quantize_int8(values):
    maximum = max(abs(value) for value in values)
    scale = f32(maximum / 127.0 if maximum else 1.0)
    quantized = [max(-127, min(127, round(value / scale))) for value in values]
    return quantized, scale


def reference():
    q_heads_total, kv_heads_total = 32, 4
    q_heads_run, head_dim, tokens = 9, 128, 2
    group_size = q_heads_total // kv_heads_total
    q = [f32(((i % 17) - 8) / 8.0) for i in range(q_heads_run * head_dim)]
    q = apply_rope(q, q_heads_run, head_dim, tokens - 1)
    keys, values, kscales, vscales = [], [], [], []
    for token in range(tokens):
        k = [f32(((((token + 1) * (i + 3)) % 23) - 11) / 11.0) for i in range(kv_heads_total * head_dim)]
        v = [f32(((((token + 2) * (i + 5)) % 19) - 9) / 9.0) for i in range(kv_heads_total * head_dim)]
        k = apply_rope(k, kv_heads_total, head_dim, token)
        kq, ks = quantize_int8(k)
        vq, vs = quantize_int8(v)
        keys.append(kq)
        values.append(vq)
        kscales.append(ks)
        vscales.append(vs)

    scores, weights = [], []
    context = [0.0] * (q_heads_run * head_dim)
    for qh in range(q_heads_run):
        kvh = qh // group_size
        head_scores = []
        for token in range(tokens):
            score = sum(
                q[qh * head_dim + d] * keys[token][kvh * head_dim + d] * kscales[token]
                for d in range(head_dim)
            ) / math.sqrt(head_dim)
            head_scores.append(score)
        maximum = max(head_scores)
        exps = [math.exp(score - maximum) for score in head_scores]
        denom = sum(exps)
        head_weights = [value / denom for value in exps]
        scores.append(head_scores)
        weights.append(head_weights)
        for token, weight in enumerate(head_weights):
            for d in range(head_dim):
                index = qh * head_dim + d
                context[index] += weight * values[token][kvh * head_dim + d] * vscales[token]
    return scores, weights, context


def test_rope_gqa_golden_matches_independent_python_reference():
    if not EXE.exists():
        return
    actual = json.loads(
        subprocess.check_output(
            [str(EXE), "rope-gqa-golden-probe", "--tokens", "2", "--q-heads-run", "9", "--seed", "7"],
            text=True,
        )
    )
    scores, weights, context = reference()
    assert actual["probe"] == "rope_gqa_golden"
    assert actual["rope_layout"] == "qwen_split_half"
    assert actual["q_heads_total"] == 32
    assert actual["kv_heads_total"] == 4
    assert actual["q_heads_run"] == 9
    assert actual["kv_heads_touched"] == 2
    assert actual["gqa_group_size"] == 8
    assert actual["head_dim"] == 128
    expected_scores = [scores[0][0], scores[0][1], scores[8][0], scores[8][1]]
    expected_weights = [weights[0][0], weights[0][1], weights[8][0], weights[8][1]]
    expected_context = [context[0], context[64], context[1024], context[1088]]
    for got, expected in zip(actual["score_samples"], expected_scores):
        assert math.isclose(got, expected, rel_tol=2e-6, abs_tol=2e-6)
    for got, expected in zip(actual["weight_samples"], expected_weights):
        assert math.isclose(got, expected, rel_tol=2e-6, abs_tol=2e-6)
    for got, expected in zip(actual["context_samples"], expected_context):
        assert math.isclose(got, expected, rel_tol=2e-6, abs_tol=2e-6)
    assert math.isclose(actual["softmax_sum_min"], 1.0, abs_tol=1e-9)
    assert math.isclose(actual["softmax_sum_max"], 1.0, abs_tol=1e-9)
