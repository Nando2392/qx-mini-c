#!/usr/bin/env python
"""Compare lossless F32 residual sidecars from QX and llama.cpp oracles."""

import argparse
import json
import math
import struct
import sys
from pathlib import Path


def read_f32(path: Path) -> tuple[float, ...]:
    data = path.read_bytes()
    if not data or len(data) % 4:
        raise ValueError(f"invalid F32 sidecar size: {path.name}")
    values = struct.unpack(f"<{len(data) // 4}f", data)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite F32 sidecar: {path.name}")
    return values


def compare_values(actual: tuple[float, ...], expected: tuple[float, ...]) -> dict[str, float | int]:
    if len(actual) != len(expected):
        raise ValueError(f"residual length mismatch: {len(actual)} != {len(expected)}")
    deltas = [float(got) - float(want) for got, want in zip(actual, expected)]
    actual_l2 = sum(float(value) * float(value) for value in actual)
    expected_l2 = sum(float(value) * float(value) for value in expected)
    dot = sum(float(got) * float(want) for got, want in zip(actual, expected))
    denominator = math.sqrt(actual_l2 * expected_l2)
    return {
        "count": len(actual),
        "max_abs": max(map(abs, deltas)),
        "rmse": math.sqrt(sum(delta * delta for delta in deltas) / len(deltas)),
        "cosine": dot / denominator if denominator else (1.0 if actual == expected else 0.0),
    }


def parse_layers(text: str) -> list[int]:
    try:
        layers = [int(part) for part in text.split(",")]
    except ValueError as exc:
        raise ValueError("layers must be comma-separated non-negative integers") from exc
    if not layers or any(layer < 0 for layer in layers) or len(set(layers)) != len(layers):
        raise ValueError("layers must be unique non-negative integers")
    return layers


def compare_directories(qx_dir: Path, llama_dir: Path, layers: list[int], max_abs: float, min_cosine: float) -> dict:
    results = []
    first_divergent = None
    for layer in layers:
        metrics = compare_values(
            read_f32(qx_dir / f"step-0-layer-{layer}-input.f32"),
            read_f32(llama_dir / f"layer-{layer}.f32"),
        )
        within_tolerance = metrics["max_abs"] <= max_abs and metrics["cosine"] >= min_cosine
        if not within_tolerance and first_divergent is None:
            first_divergent = layer
        results.append({"layer": layer, **metrics, "within_tolerance": within_tolerance})
    return {
        "schema": 1,
        "max_abs_tolerance": max_abs,
        "min_cosine_tolerance": min_cosine,
        "layers": results,
        "first_divergent_layer": first_divergent,
        "passed": first_divergent is None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qx-dir", type=Path, required=True)
    parser.add_argument("--llama-dir", type=Path, required=True)
    parser.add_argument("--layers", default="0,1,24,47")
    parser.add_argument("--max-abs", type=float, default=1e-5)
    parser.add_argument("--min-cosine", type=float, default=0.999999)
    args = parser.parse_args()
    try:
        if args.max_abs < 0 or not -1 <= args.min_cosine <= 1:
            raise ValueError("invalid tolerance")
        report = compare_directories(
            args.qx_dir,
            args.llama_dir,
            parse_layers(args.layers),
            args.max_abs,
            args.min_cosine,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"schema": 1, "error": str(exc), "passed": False}))
        return 2
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
