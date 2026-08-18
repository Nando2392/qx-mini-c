#!/usr/bin/env python
"""Compare lossless F32 residual sidecars from QX and llama.cpp oracles."""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

LLAMA_PHASE_NAMES = {
    "input": "layer-{layer}.f32",
    "v-cur": "Vcur-{layer}.f32",
    "kqv-out": "kqv_out-{layer}.f32",
    "ffn-inp": "ffn_inp-{layer}.f32",
    "ffn-moe-out": "ffn_moe_out-{layer}.f32",
    "output": "l_out-{layer}.f32",
}


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
    delta_l2 = math.sqrt(sum(delta * delta for delta in deltas))
    return {
        "count": len(actual),
        "max_abs": max(map(abs, deltas)),
        "rmse": delta_l2 / math.sqrt(len(deltas)),
        "cosine": dot / denominator if denominator else (1.0 if actual == expected else 0.0),
        "delta_l2": delta_l2,
    }


def parse_layers(text: str) -> list[int]:
    try:
        layers = [int(part) for part in text.split(",")]
    except ValueError as exc:
        raise ValueError("layers must be comma-separated non-negative integers") from exc
    if (not layers or any(layer < 0 for layer in layers) or
            any(current >= following for current, following in zip(layers, layers[1:]))):
        raise ValueError("layers must be strictly increasing non-negative integers")
    return layers


def compare_directories(
    qx_dir: Path,
    llama_dir: Path,
    layers: list[int],
    max_abs: float,
    min_cosine: float,
    phase: str = "input",
    amplification_gain: float = 2.0,
    amplification_start_layer: int = 0,
) -> dict:
    results = []
    first_divergent = None
    first_material_amplification = None
    previous_delta_l2 = None
    for layer in layers:
        metrics = compare_values(
            read_f32(qx_dir / f"step-0-layer-{layer}-{phase}.f32"),
            read_f32(llama_dir / LLAMA_PHASE_NAMES[phase].format(layer=layer)),
        )
        within_tolerance = metrics["max_abs"] <= max_abs and metrics["cosine"] >= min_cosine
        if not within_tolerance and first_divergent is None:
            first_divergent = layer
        gain_from_previous = None
        if previous_delta_l2 is not None and previous_delta_l2 > 0.0:
            gain_from_previous = metrics["delta_l2"] / previous_delta_l2
            if (layer > amplification_start_layer and gain_from_previous > amplification_gain and
                    first_material_amplification is None):
                first_material_amplification = layer
        results.append({
            "layer": layer,
            **metrics,
            "gain_from_previous": gain_from_previous,
            "within_tolerance": within_tolerance,
        })
        previous_delta_l2 = metrics["delta_l2"]
    return {
        "schema": 1,
        "phase": phase,
        "max_abs_tolerance": max_abs,
        "min_cosine_tolerance": min_cosine,
        "amplification_gain_threshold": amplification_gain,
        "amplification_start_layer": amplification_start_layer,
        "layers": results,
        "first_divergent_layer": first_divergent,
        "first_material_amplification_layer": first_material_amplification,
        "passed": first_divergent is None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qx-dir", type=Path, required=True)
    parser.add_argument("--llama-dir", type=Path, required=True)
    parser.add_argument("--layers", default="0,1,24,47")
    parser.add_argument("--phase", choices=tuple(LLAMA_PHASE_NAMES), default="input")
    parser.add_argument("--max-abs", type=float, default=1e-5)
    parser.add_argument("--min-cosine", type=float, default=0.999999)
    parser.add_argument("--amplification-gain", type=float, default=2.0)
    parser.add_argument("--amplification-start-layer", type=int, default=0)
    args = parser.parse_args()
    try:
        if (args.max_abs < 0 or not -1 <= args.min_cosine <= 1 or
                not math.isfinite(args.amplification_gain) or args.amplification_gain <= 0 or
                args.amplification_start_layer < 0):
            raise ValueError("invalid tolerance")
        report = compare_directories(
            args.qx_dir,
            args.llama_dir,
            parse_layers(args.layers),
            args.max_abs,
            args.min_cosine,
            args.phase,
            args.amplification_gain,
            args.amplification_start_layer,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"schema": 1, "error": str(exc), "passed": False}))
        return 2
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
