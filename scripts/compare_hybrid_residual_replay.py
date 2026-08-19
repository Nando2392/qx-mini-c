#!/usr/bin/env python3
"""Compare modal-equivalent hybrid residual replays against llama.cpp sidecars."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Sequence


def read_exact_f32(path: Path, expected_count: int) -> tuple[float, ...]:
    raw = path.read_bytes()
    expected_bytes = expected_count * 4
    if len(raw) != expected_bytes:
        raise ValueError(f"{path}: expected {expected_bytes} bytes, found {len(raw)}")
    values = struct.unpack(f"<{expected_count}f", raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path}: contains NaN or Inf")
    return values


def metrics(actual: Sequence[float], reference: Sequence[float]) -> dict[str, float]:
    deltas = [left - right for left, right in zip(actual, reference, strict=True)]
    squared_error = sum(delta * delta for delta in deltas)
    actual_norm2 = sum(value * value for value in actual)
    reference_norm2 = sum(value * value for value in reference)
    denominator = math.sqrt(actual_norm2 * reference_norm2)
    cosine = sum(left * right for left, right in zip(actual, reference, strict=True)) / denominator if denominator else 1.0
    return {
        "max_abs": max(abs(delta) for delta in deltas),
        "rmse": math.sqrt(squared_error / len(deltas)),
        "l2": math.sqrt(squared_error),
        "cosine": cosine,
        "reference_l2": math.sqrt(reference_norm2),
    }


def validate_run_metadata(
    path: Path, start_layer: int, expected_count: int, kv_format: str
) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    replay = data.get("residual_replay")
    if not isinstance(replay, dict) or replay.get("enabled") is not True:
        raise ValueError(f"{path}: residual replay metadata is not enabled")
    if data.get("start_layer") != start_layer:
        raise ValueError(f"{path}: expected start_layer {start_layer}")
    if data.get("kv_format") != kv_format:
        raise ValueError(f"{path}: expected kv_format {kv_format}")
    if data.get("residual_source") != "injected_f32_replay":
        raise ValueError(f"{path}: expected residual_source injected_f32_replay")
    if replay.get("source") != "f32_sidecar" or replay.get("values") != expected_count:
        raise ValueError(f"{path}: residual replay source or value count is invalid")


def build_report(
    oracle_dir: Path,
    hybrid_dir: Path,
    layers: int,
    expected_count: int,
    kv_format: str,
) -> dict[str, object]:
    if layers <= 0 or expected_count <= 0:
        raise ValueError("layers and expected-count must be positive")
    final_reference = read_exact_f32(oracle_dir / f"l_out-{layers - 1}.f32", expected_count)
    baseline_dir = hybrid_dir / "start-0"
    rows: list[dict[str, object]] = []
    zero_metrics = {"max_abs": 0.0, "rmse": 0.0, "l2": 0.0, "cosine": 1.0, "reference_l2": 0.0}

    for start_layer in range(layers):
        run_dir = hybrid_dir / f"start-{start_layer}"
        validate_run_metadata(
            run_dir / "result.json", start_layer, expected_count, kv_format
        )
        final = metrics(
            read_exact_f32(run_dir / f"step-0-layer-{layers - 1}-output.f32", expected_count),
            final_reference,
        )
        if start_layer == 0:
            incoming = dict(zero_metrics)
        else:
            incoming = metrics(
                read_exact_f32(
                    baseline_dir / f"step-0-layer-{start_layer - 1}-output.f32",
                    expected_count,
                ),
                read_exact_f32(oracle_dir / f"layer-{start_layer}.f32", expected_count),
            )
        rows.append(
            {
                "start_layer": start_layer,
                "incoming_accumulated": incoming,
                "final_after_replay": final,
            }
        )

    return {
        "schema": "qx-hybrid-residual-replay-v1",
        "mode": {
            "layers": layers,
            "expected_count": expected_count,
            "kv_format": kv_format,
            "residual_source": "llama.cpp layer-N.f32",
            "kv_source": "QX recomputed; oracle KV is not injected",
            "limitation": "Residual-only injection does not replace an accumulated KV cache.",
        },
        "rows": rows,
        "largest_final_rmse": sorted(
            rows,
            key=lambda row: row["final_after_replay"]["rmse"],
            reverse=True,
        )[:10],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--hybrid-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--kv-format", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_report(
            args.oracle_dir,
            args.hybrid_dir,
            args.layers,
            args.expected_count,
            args.kv_format,
        )
        rendered = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
