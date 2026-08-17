#!/usr/bin/env python
"""Compare QX and llama.cpp MoE stage sidecars by stage and selected expert."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path


STAGES = (
    "ffn_norm",
    "ffn_moe_logits",
    "ffn_moe_probs",
    "ffn_moe_topk",
    "ffn_moe_weights",
    "ffn_moe_weights_sum",
    "ffn_moe_weights_norm",
    "ffn_moe_gate",
    "ffn_moe_up",
    "ffn_moe_swiglu",
    "ffn_moe_down",
    "ffn_moe_weighted",
)
EXPERT_STAGES = (
    ("gate", "ffn_moe_gate"),
    ("up", "ffn_moe_up"),
    ("swiglu", "ffn_moe_swiglu"),
    ("down", "ffn_moe_down"),
    ("weighted", "ffn_moe_weighted"),
)


def read_f32(path: Path) -> tuple[float, ...]:
    raw = path.read_bytes()
    if not raw or len(raw) % 4:
        raise ValueError(f"invalid F32 sidecar size: {path.name}")
    values = struct.unpack(f"<{len(raw) // 4}f", raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite F32 sidecar: {path.name}")
    return values


def metrics(actual: tuple[float, ...], expected: tuple[float, ...]) -> dict[str, float | int]:
    if len(actual) != len(expected) or not actual:
        raise ValueError(f"sidecar length mismatch: {len(actual)} != {len(expected)}")
    deltas = [float(got) - float(want) for got, want in zip(actual, expected)]
    index = max(range(len(deltas)), key=lambda item: abs(deltas[item]))
    actual_l2 = sum(float(value) * float(value) for value in actual)
    expected_l2 = sum(float(value) * float(value) for value in expected)
    dot = sum(float(got) * float(want) for got, want in zip(actual, expected))
    denominator = math.sqrt(actual_l2 * expected_l2)
    return {
        "count": len(actual),
        "max_abs": abs(deltas[index]),
        "rmse": math.sqrt(sum(delta * delta for delta in deltas) / len(deltas)),
        "cosine": dot / denominator if denominator else (1.0 if actual == expected else 0.0),
        "max_abs_index": index,
    }


def compare_directories(qx_dir: Path, llama_dir: Path, layer: int) -> dict:
    pairs = {
        stage: (
            read_f32(qx_dir / f"{stage}-{layer}.f32"),
            read_f32(llama_dir / f"{stage}-{layer}.f32"),
        )
        for stage in STAGES
    }
    selected_raw = pairs["ffn_moe_topk"][0]
    selected = [int(value) for value in selected_raw]
    if any(float(expert) != value or expert < 0 for expert, value in zip(selected, selected_raw)):
        raise ValueError("top-k sidecar contains invalid expert id")
    experts_used = len(selected)
    hidden = len(pairs["ffn_norm"][0])
    expert_count = len(pairs["ffn_moe_logits"][0])
    if experts_used == 0 or expert_count == 0 or len(set(selected)) != experts_used or any(expert >= expert_count for expert in selected):
        raise ValueError("invalid top-k expert selection")
    if len(pairs["ffn_moe_probs"][0]) != expert_count:
        raise ValueError("router probability shape mismatch")
    for stage in ("ffn_moe_weights", "ffn_moe_weights_norm"):
        if len(pairs[stage][0]) != experts_used:
            raise ValueError(f"{stage} shape mismatch")
    if len(pairs["ffn_moe_weights_sum"][0]) != 1:
        raise ValueError("weight sum shape mismatch")
    gate_count = len(pairs["ffn_moe_gate"][0])
    if gate_count % experts_used:
        raise ValueError("gate shape mismatch")
    intermediate = gate_count // experts_used
    for stage in ("ffn_moe_up", "ffn_moe_swiglu"):
        if len(pairs[stage][0]) != intermediate * experts_used:
            raise ValueError(f"{stage} shape mismatch")
    for stage in ("ffn_moe_down", "ffn_moe_weighted"):
        if len(pairs[stage][0]) != hidden * experts_used:
            raise ValueError(f"{stage} shape mismatch")

    report = {
        "schema": 1,
        "layer": layer,
        "hidden": hidden,
        "intermediate": intermediate,
        "expert_count": expert_count,
        "experts_used": experts_used,
        "selected_experts": selected,
        "stages": {stage: metrics(*pairs[stage]) for stage in STAGES},
        "experts": [],
        "passed": True,
    }
    for rank, expert in enumerate(selected):
        item = {"rank": rank, "expert_id": expert}
        for short_name, stage in EXPERT_STAGES:
            width = intermediate if short_name in ("gate", "up", "swiglu") else hidden
            actual, expected = pairs[stage]
            start = rank * width
            item[short_name] = metrics(actual[start : start + width], expected[start : start + width])
        report["experts"].append(item)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qx-dir", type=Path, required=True)
    parser.add_argument("--llama-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.layer < 0:
            raise ValueError("layer must be non-negative")
        report = compare_directories(args.qx_dir, args.llama_dir, args.layer)
    except (OSError, ValueError) as exc:
        print(json.dumps({"schema": 1, "passed": False, "error": str(exc)}))
        return 2
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
