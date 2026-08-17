#!/usr/bin/env python
"""Spike reproducible: IQ4_XS x F32 frente a IQ4_XS x Q8_K y Vcur de llama.cpp."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import struct
import subprocess
import time
from pathlib import Path

QK_K = 256
IQ4_XS_BYTES = 136
Q8_K_BYTES = 4 + QK_K + (QK_K // 16) * 2
IQ4_VALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def nearest_int(value: float) -> int:
    if not math.isfinite(value) or abs(value) > 4194303.0:
        raise ValueError("Q8_K rounding input is non-finite or out of range")
    bits = struct.unpack("<I", struct.pack("<f", f32(value + 12582912.0)))[0]
    return (bits & 0x007FFFFF) - 0x00400000


def quantize_q8_k(values: tuple[float, ...]) -> tuple[tuple[float, tuple[int, ...], tuple[int, ...]], ...]:
    if not values or len(values) % QK_K:
        raise ValueError("Q8_K input length must be a positive multiple of 256")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Q8_K input contains NaN or Inf")
    blocks = []
    for start in range(0, len(values), QK_K):
        block = values[start : start + QK_K]
        amax = 0.0
        signed_max = 0.0
        for value in block:
            absolute = abs(value)
            if absolute > amax:
                amax = absolute
                signed_max = value
        if amax == 0.0:
            quants = (0,) * QK_K
            blocks.append((0.0, quants, (0,) * (QK_K // 16)))
            continue
        inverse_scale = f32(-127.0 / signed_max)
        quants = tuple(min(127, nearest_int(f32(inverse_scale * value))) for value in block)
        if any(quant < -128 or quant > 127 for quant in quants):
            raise ValueError("Q8_K quant escaped int8 range")
        sums = tuple(sum(quants[group : group + 16]) for group in range(0, QK_K, 16))
        blocks.append((f32(1.0 / inverse_scale), quants, sums))
    return tuple(blocks)


def decode_q4_k(block: bytes) -> tuple[float, ...]:
    if len(block) != 144:
        raise ValueError("invalid Q4_K block size")
    d, dmin = struct.unpack_from("<ee", block)
    scales, quants = block[4:16], block[16:144]
    values = []
    quant_offset = 0
    scale_index = 0
    for _ in range(4):
        if scale_index < 4:
            scale_a = scales[scale_index] & 63
            min_a = scales[scale_index + 4] & 63
            scale_b = scales[scale_index + 1] & 63
            min_b = scales[scale_index + 5] & 63
        else:
            scale_a = (scales[scale_index + 4] & 15) | ((scales[scale_index - 4] >> 6) << 4)
            min_a = (scales[scale_index + 4] >> 4) | ((scales[scale_index] >> 6) << 4)
            j = scale_index + 1
            scale_b = (scales[j + 4] & 15) | ((scales[j - 4] >> 6) << 4)
            min_b = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
        chunk = quants[quant_offset : quant_offset + 32]
        values.extend(f32(d * scale_a * (value & 15) - dmin * min_a) for value in chunk)
        values.extend(f32(d * scale_b * (value >> 4) - dmin * min_b) for value in chunk)
        quant_offset += 32
        scale_index += 2
    return tuple(values)


def decode_iq4_xs(block: bytes) -> tuple[float, ...]:
    if len(block) != IQ4_XS_BYTES:
        raise ValueError("invalid IQ4_XS block size")
    d = struct.unpack_from("<e", block)[0]
    scales_high = struct.unpack_from("<H", block, 2)[0]
    scales_low, quants = block[4:8], block[8:]
    values = []
    for group in range(8):
        local_scale = ((scales_low[group // 2] >> (4 * (group % 2))) & 15) | (((scales_high >> (2 * group)) & 3) << 4)
        multiplier = f32(d * (local_scale - 32))
        chunk = quants[group * 16 : group * 16 + 16]
        values.extend(f32(multiplier * IQ4_VALUES[value & 15]) for value in chunk)
        values.extend(f32(multiplier * IQ4_VALUES[value >> 4]) for value in chunk)
    return tuple(values)


def dot_iq4_xs_f32(row: bytes, activation: tuple[float, ...]) -> float:
    if len(row) % IQ4_XS_BYTES or len(activation) != len(row) // IQ4_XS_BYTES * QK_K:
        raise ValueError("IQ4_XS/F32 dimensions do not match")
    dot = 0.0
    for block_index in range(len(row) // IQ4_XS_BYTES):
        weights = decode_iq4_xs(row[block_index * IQ4_XS_BYTES : (block_index + 1) * IQ4_XS_BYTES])
        base = block_index * QK_K
        dot += sum(float(weight) * float(value) for weight, value in zip(weights, activation[base : base + QK_K]))
    return f32(dot)


def dot_iq4_xs_q8_k(row: bytes, q8_blocks: tuple[tuple[float, tuple[int, ...], tuple[int, ...]], ...]) -> float:
    if len(row) != len(q8_blocks) * IQ4_XS_BYTES:
        raise ValueError("IQ4_XS/Q8_K dimensions do not match")
    total = 0.0
    for block_index, (q8_scale, quants8, _block_sums) in enumerate(q8_blocks):
        block = row[block_index * IQ4_XS_BYTES : (block_index + 1) * IQ4_XS_BYTES]
        weight_scale = struct.unpack_from("<e", block)[0]
        scales_high = struct.unpack_from("<H", block, 2)[0]
        scales_low, quants4 = block[4:8], block[8:]
        integer_sum = 0
        for group in range(8):
            local_scale = ((scales_low[group // 2] >> (4 * (group % 2))) & 15) | (((scales_high >> (2 * group)) & 3) << 4)
            local_scale -= 32
            chunk = quants4[group * 16 : group * 16 + 16]
            base = group * 32
            group_sum = sum(quants8[base + index] * IQ4_VALUES[value & 15] for index, value in enumerate(chunk))
            group_sum += sum(quants8[base + 16 + index] * IQ4_VALUES[value >> 4] for index, value in enumerate(chunk))
            integer_sum += group_sum * local_scale
        total = f32(total + f32(float(weight_scale) * q8_scale * integer_sum))
    return total


def inspect_tensor(exe: Path, model: Path, name: str) -> dict:
    return json.loads(subprocess.check_output([str(exe), "inspect-tensor", "--in", str(model), "--name", name], text=True))


def read_span(model: Path, offset: int, size: int) -> bytes:
    with model.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(size)
    if len(data) != size:
        raise ValueError("short model read")
    return data


def activation_for_token(exe: Path, model: Path, token: int) -> tuple[float, ...]:
    embedding = inspect_tensor(exe, model, "token_embd.weight")
    norm = inspect_tensor(exe, model, "blk.0.attn_norm.weight")
    if embedding["dims"][0] != 2048 or norm["dims"][0] != 2048:
        raise ValueError("unexpected Qwen3 hidden size")
    row = read_span(model, embedding["offset"] + token * 8 * 144, 8 * 144)
    decoded = tuple(value for index in range(8) for value in decode_q4_k(row[index * 144 : (index + 1) * 144]))
    weights = struct.unpack("<2048f", read_span(model, norm["offset"], 2048 * 4))
    rms = math.sqrt(sum(float(value) * float(value) for value in decoded) / 2048 + 1e-6)
    return tuple(f32(f32(value / rms) * weights[index]) for index, value in enumerate(decoded))


def read_f32(path: Path) -> tuple[float, ...]:
    data = path.read_bytes()
    if not data or len(data) % 4:
        raise ValueError("invalid F32 sidecar")
    values = struct.unpack(f"<{len(data) // 4}f", data)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite oracle sidecar")
    return values


def metrics(actual: tuple[float, ...], expected: tuple[float, ...]) -> dict:
    if len(actual) != len(expected) or not actual:
        raise ValueError("metric dimensions do not match")
    deltas = tuple(float(got) - float(want) for got, want in zip(actual, expected))
    index = max(range(len(deltas)), key=lambda item: abs(deltas[item]))
    actual_l2 = sum(float(value) ** 2 for value in actual)
    expected_l2 = sum(float(value) ** 2 for value in expected)
    denominator = math.sqrt(actual_l2 * expected_l2)
    return {
        "count": len(actual),
        "max_abs": abs(deltas[index]),
        "max_abs_index": index,
        "rmse": math.sqrt(sum(delta * delta for delta in deltas) / len(deltas)),
        "cosine": sum(float(got) * float(want) for got, want in zip(actual, expected)) / denominator if denominator else 0.0,
    }


def checksum(values: tuple[float, ...]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(values)}f", *values)).hexdigest()


def timed_projection(rows: tuple[bytes, ...], operation, repetitions: int) -> tuple[tuple[float, ...], list[float]]:
    timings = []
    output = ()
    for _ in range(repetitions):
        started = time.perf_counter()
        output = tuple(operation(row) for row in rows)
        timings.append(time.perf_counter() - started)
    return output, timings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qx-exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--oracle-vcur", type=Path, required=True)
    parser.add_argument("--token", type=int, default=42)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    try:
        if args.token < 0 or args.repetitions < 3 or args.repetitions > 50:
            raise ValueError("invalid token or repetitions")
        if not args.qx_exe.is_file() or not args.model.is_file() or not args.oracle_vcur.is_file():
            raise ValueError("missing QX executable, model, or oracle sidecar")
        activation = activation_for_token(args.qx_exe, args.model, args.token)
        tensor = inspect_tensor(args.qx_exe, args.model, "blk.0.attn_v.weight")
        if tensor.get("ggml_type") != 23 or tensor.get("dims", [])[:2] != [2048, 512]:
            raise ValueError("unexpected V projection contract")
        blocks_per_row = len(activation) // QK_K
        row_bytes = blocks_per_row * IQ4_XS_BYTES
        raw = read_span(args.model, tensor["offset"], tensor["dims"][1] * row_bytes)
        rows = tuple(raw[index * row_bytes : (index + 1) * row_bytes] for index in range(tensor["dims"][1]))
        oracle = read_f32(args.oracle_vcur)
        if len(oracle) != len(rows):
            raise ValueError("oracle Vcur dimensions do not match projection")
        q8_blocks = quantize_q8_k(activation)
        f32_output, f32_timings = timed_projection(rows, lambda row: dot_iq4_xs_f32(row, activation), args.repetitions)
        q8_output, q8_timings = timed_projection(rows, lambda row: dot_iq4_xs_q8_k(row, q8_blocks), args.repetitions)
        report = {
            "schema": 1,
            "experiment": "f32-vs-q8k-activation-spike",
            "token": args.token,
            "layer": 0,
            "checkpoint": "Vcur",
            "dimensions": {"input": len(activation), "output": len(rows), "qk_k": QK_K, "blocks": len(q8_blocks)},
            "temporary_bytes": {"f32": len(activation) * 4, "q8_k": len(q8_blocks) * Q8_K_BYTES},
            "comparisons": {
                "qx_f32_vs_llama": metrics(f32_output, oracle),
                "qx_q8_k_vs_llama": metrics(q8_output, oracle),
                "qx_q8_k_vs_qx_f32": metrics(q8_output, f32_output),
            },
            "timing_seconds_full_v_projection_python_spike": {
                "f32": {"median": statistics.median(f32_timings), "min": min(f32_timings), "max": max(f32_timings), "runs": f32_timings},
                "q8_k": {"median": statistics.median(q8_timings), "min": min(q8_timings), "max": max(q8_timings), "runs": q8_timings},
            },
            "checksums_sha256": {"f32": checksum(f32_output), "q8_k": checksum(q8_output), "llama": checksum(oracle)},
            "first_divergent_layer": 0,
            "first_divergent_checkpoint": "Vcur",
            "notes": [
                "Timing is a Python spike over the complete 2048x512 V projection, not QX C latency or tok/s.",
                "llama oracle sidecar comes from the pinned CPU graph; Q8_K is reimplemented from the documented mathematical contract.",
            ],
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"schema": 1, "experiment": "f32-vs-q8k-activation-spike", "error": str(exc), "passed": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
