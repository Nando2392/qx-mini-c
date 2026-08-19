#!/usr/bin/env python3
"""Prepare, run, and analyze scaled residual suffix replays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path
from typing import Sequence


MANIFEST_SCHEMA = "qx-scaled-residual-replay-manifest-v1"
REPORT_SCHEMA = "qx-scaled-residual-replay-v1"


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def read_exact_f32(path: Path, expected_count: int) -> tuple[float, ...]:
    raw = path.read_bytes()
    expected_bytes = expected_count * 4
    if len(raw) != expected_bytes:
        raise ValueError(f"{path}: expected {expected_bytes} bytes, found {len(raw)}")
    values = struct.unpack(f"<{expected_count}f", raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path}: contains NaN or Inf")
    return values


def write_f32(path: Path, values: Sequence[float]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(actual: Sequence[float], reference: Sequence[float]) -> dict[str, float]:
    if len(actual) != len(reference) or not actual:
        raise ValueError(f"sidecar length mismatch: {len(actual)} != {len(reference)}")
    deltas = [left - right for left, right in zip(actual, reference, strict=True)]
    squared_error = math.fsum(delta * delta for delta in deltas)
    actual_norm2 = math.fsum(value * value for value in actual)
    reference_norm2 = math.fsum(value * value for value in reference)
    denominator = math.sqrt(actual_norm2 * reference_norm2)
    dot = math.fsum(left * right for left, right in zip(actual, reference, strict=True))
    return {
        "max_abs": max(abs(delta) for delta in deltas),
        "rmse": math.sqrt(squared_error / len(deltas)),
        "l2": math.sqrt(squared_error),
        "cosine": dot / denominator if denominator else (1.0 if actual == reference else 0.0),
    }


def parse_scales(raw: str) -> list[float]:
    try:
        scales = [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("scales must be comma-separated finite numbers") from exc
    if not scales or not all(math.isfinite(scale) for scale in scales):
        raise ValueError("scales must be comma-separated finite numbers")
    if len(scales) != len(set(scales)):
        raise ValueError("scales must be unique")
    if 0.0 not in scales or not any(scale != 0.0 for scale in scales):
        raise ValueError("scale grid must contain zero and at least one non-zero scale")
    return scales


def scale_slug(scale: float) -> str:
    if scale == 0.0:
        return "scale-zero"
    sign = "neg" if scale < 0.0 else "pos"
    magnitude = format(abs(scale), ".17g").replace(".", "p").replace("+", "")
    return f"scale-{sign}-{magnitude}"


def validate_matrix(layers: int, start_layer: int, expected_count: int) -> None:
    if layers <= 0 or start_layer <= 0 or start_layer >= layers:
        raise ValueError("require 0 < start-layer < layers")
    if expected_count <= 0:
        raise ValueError("expected-count must be positive")


def validate_execution_values(prompt_token: int, ctx: int, seed: int) -> None:
    if prompt_token < 0 or ctx <= 0 or seed < 0:
        raise ValueError("prompt-token and seed must be non-negative; ctx must be positive")


def prepare_experiment(args: argparse.Namespace) -> dict[str, object]:
    validate_matrix(args.layers, args.start_layer, args.expected_count)
    validate_execution_values(args.prompt_token, args.ctx, args.seed)
    scales = parse_scales(args.scales)
    if args.experiment_dir.exists() and any(args.experiment_dir.iterdir()):
        raise ValueError(f"experiment directory is not empty: {args.experiment_dir}")
    args.experiment_dir.mkdir(parents=True, exist_ok=True)

    oracle_input_path = args.oracle_dir / f"layer-{args.start_layer}.f32"
    baseline_input_path = (
        args.baseline_dir / f"step-0-layer-{args.start_layer - 1}-output.f32"
    )
    oracle_final_path = args.oracle_dir / f"l_out-{args.layers - 1}.f32"
    oracle_input = read_exact_f32(oracle_input_path, args.expected_count)
    baseline_input = read_exact_f32(baseline_input_path, args.expected_count)
    read_exact_f32(oracle_final_path, args.expected_count)
    direction = tuple(
        float(baseline) - float(oracle)
        for baseline, oracle in zip(baseline_input, oracle_input, strict=True)
    )
    direction_metrics = metrics(baseline_input, oracle_input)
    if direction_metrics["l2"] <= 0.0:
        raise ValueError("baseline perturbation direction is zero")

    runs: list[dict[str, object]] = []
    for scale in scales:
        directory = scale_slug(scale)
        run_dir = args.experiment_dir / directory
        run_dir.mkdir()
        residual = tuple(
            f32(float(oracle) + scale * delta)
            for oracle, delta in zip(oracle_input, direction, strict=True)
        )
        effective = metrics(residual, oracle_input)
        if scale != 0.0 and effective["l2"] <= 0.0:
            raise ValueError(f"scale {scale} collapses to the zero perturbation in F32")
        residual_path = run_dir / "residual.f32"
        write_f32(residual_path, residual)
        runs.append(
            {
                "scale": scale,
                "directory": directory,
                "residual_sha256": sha256_file(residual_path),
                "effective_input_delta": effective,
            }
        )

    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "matrix": {
            "layers": args.layers,
            "start_layer": args.start_layer,
            "expected_count": args.expected_count,
            "kv_format": args.kv_format,
            "activation_format": args.activation_format,
            "prompt_token": args.prompt_token,
            "ctx": args.ctx,
            "temperature": 0.0,
            "seed": args.seed,
            "residual_formula": "oracle_layer_input + scale * (qx_baseline_layer_input - oracle_layer_input)",
        },
        "sources": {
            "oracle_input": {
                "path": str(oracle_input_path.resolve()),
                "sha256": sha256_file(oracle_input_path),
            },
            "baseline_input": {
                "path": str(baseline_input_path.resolve()),
                "sha256": sha256_file(baseline_input_path),
            },
            "oracle_final": {
                "path": str(oracle_final_path.resolve()),
                "sha256": sha256_file(oracle_final_path),
            },
        },
        "direction": direction_metrics,
        "runs": runs,
    }
    (args.experiment_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def validate_source(manifest_entry: object, expected_path: Path, label: str) -> None:
    if not isinstance(manifest_entry, dict):
        raise ValueError(f"manifest source is invalid: {label}")
    if Path(str(manifest_entry.get("path"))).resolve() != expected_path.resolve():
        raise ValueError(f"manifest source path mismatch: {label}")
    if manifest_entry.get("sha256") != sha256_file(expected_path):
        raise ValueError(f"manifest source hash mismatch: {label}")


def load_routing(
    path: Path,
    *,
    start_layer: int,
    layers: int,
    expected_count: int,
    kv_format: str,
    activation_format: str,
    prompt_token: int,
    ctx: int,
) -> dict[int, dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: result root must be a JSON object")
    replay = data.get("residual_replay")
    if data.get("start_layer") != start_layer:
        raise ValueError(f"{path}: expected start_layer {start_layer}")
    if data.get("kv_format") != kv_format:
        raise ValueError(f"{path}: expected kv_format {kv_format}")
    if data.get("activation_format") != activation_format:
        raise ValueError(f"{path}: expected activation_format {activation_format}")
    if data.get("residual_source") != "injected_f32_replay":
        raise ValueError(f"{path}: expected residual_source injected_f32_replay")
    if not isinstance(replay, dict) or replay.get("enabled") is not True:
        raise ValueError(f"{path}: residual replay metadata is not enabled")
    if replay.get("source") != "f32_sidecar" or replay.get("values") != expected_count:
        raise ValueError(f"{path}: residual replay source or value count is invalid")
    suffix_layers = layers - start_layer
    if (
        data.get("probe") != "state_loop"
        or data.get("prompt_token") != prompt_token
        or data.get("prompt_token_count") != 1
        or data.get("prompt_token_ids") != [prompt_token]
        or data.get("generation_steps") != 1
        or data.get("steps") != 1
        or data.get("layers") != layers
        or data.get("ctx_tokens") != ctx
        or data.get("residual_dump") is not True
        or data.get("residual_dump_count") != suffix_layers * 6
        or data.get("delta_source") != "real_attention_moe"
        or data.get("layers_run") != suffix_layers
        or data.get("kv_appends") != suffix_layers
        or data.get("cache_readback_ok") is not True
    ):
        raise ValueError(f"{path}: expected fixed one-token full-MoE replay matrix")
    tokens = data.get("tokens")
    if not isinstance(tokens, list) or len(tokens) != 1 or not isinstance(tokens[0], dict):
        raise ValueError(f"{path}: expected exactly one token trace")
    if (
        tokens[0].get("step") != 0
        or tokens[0].get("position") != 0
        or tokens[0].get("input_token") != prompt_token
    ):
        raise ValueError(f"{path}: token trace does not match the fixed input")
    layer_rows = tokens[0].get("layers")
    if not isinstance(layer_rows, list):
        raise ValueError(f"{path}: layer trace is missing")

    routing: dict[int, dict[str, object]] = {}
    for row in layer_rows:
        if not isinstance(row, dict) or not isinstance(row.get("layer"), int):
            raise ValueError(f"{path}: invalid layer trace row")
        layer = row["layer"]
        experts = row.get("selected_experts")
        weights = row.get("routing_weights")
        if row.get("full_moe") is not True or row.get("experts_run") != 8:
            raise ValueError(f"{path}: expected full MoE top-8 trace at layer {layer}")
        if (
            not isinstance(experts, list)
            or len(experts) != 8
            or any(not isinstance(expert, int) or expert < 0 or expert >= 128 for expert in experts)
            or len(set(experts)) != len(experts)
            or not isinstance(weights, list)
            or len(weights) != len(experts)
            or any(
                not isinstance(weight, (int, float))
                or not math.isfinite(float(weight))
                or float(weight) <= 0.0
                for weight in weights
            )
            or not math.isclose(
                math.fsum(float(weight) for weight in weights),
                1.0,
                rel_tol=1e-6,
                abs_tol=1e-7,
            )
        ):
            raise ValueError(f"{path}: invalid routing trace at layer {layer}")
        if layer in routing:
            raise ValueError(f"{path}: duplicate routing layer {layer}")
        routing[layer] = {
            "selected_experts": experts,
            "routing_weights": [float(weight) for weight in weights],
        }
    expected_layers = set(range(start_layer, layers))
    if set(routing) != expected_layers:
        raise ValueError(f"{path}: routing layer coverage mismatch")
    return routing


def load_manifest(args: argparse.Namespace) -> dict[str, object]:
    path = args.experiment_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: manifest root must be a JSON object")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{path}: unsupported manifest schema")
    matrix = manifest.get("matrix")
    sources = manifest.get("sources")
    runs = manifest.get("runs")
    if not isinstance(matrix, dict) or not isinstance(sources, dict) or not isinstance(runs, list):
        raise ValueError(f"{path}: incomplete manifest")
    start_layer = matrix.get("start_layer")
    layers = matrix.get("layers")
    expected_count = matrix.get("expected_count")
    if not all(isinstance(value, int) for value in (start_layer, layers, expected_count)):
        raise ValueError(f"{path}: invalid matrix dimensions")
    validate_matrix(layers, start_layer, expected_count)
    prompt_token = matrix.get("prompt_token")
    ctx = matrix.get("ctx")
    seed = matrix.get("seed")
    if not all(isinstance(value, int) for value in (prompt_token, ctx, seed)):
        raise ValueError(f"{path}: invalid execution matrix")
    validate_execution_values(prompt_token, ctx, seed)
    if not isinstance(matrix.get("kv_format"), str) or not matrix["kv_format"]:
        raise ValueError(f"{path}: invalid kv_format")
    if not isinstance(matrix.get("activation_format"), str) or not matrix["activation_format"]:
        raise ValueError(f"{path}: invalid activation_format")
    validate_source(
        sources.get("oracle_input"),
        args.oracle_dir / f"layer-{start_layer}.f32",
        "oracle_input",
    )
    validate_source(
        sources.get("baseline_input"),
        args.baseline_dir / f"step-0-layer-{start_layer - 1}-output.f32",
        "baseline_input",
    )
    validate_source(
        sources.get("oracle_final"),
        args.oracle_dir / f"l_out-{layers - 1}.f32",
        "oracle_final",
    )
    return manifest


def analyze_experiment(args: argparse.Namespace) -> dict[str, object]:
    manifest = load_manifest(args)
    matrix = manifest["matrix"]
    runs = manifest["runs"]
    assert isinstance(matrix, dict) and isinstance(runs, list)
    layers = int(matrix["layers"])
    start_layer = int(matrix["start_layer"])
    expected_count = int(matrix["expected_count"])
    kv_format = str(matrix["kv_format"])
    activation_format = str(matrix["activation_format"])
    prompt_token = int(matrix["prompt_token"])
    ctx = int(matrix["ctx"])
    oracle_input = read_exact_f32(
        args.oracle_dir / f"layer-{start_layer}.f32", expected_count
    )
    baseline_input = read_exact_f32(
        args.baseline_dir / f"step-0-layer-{start_layer - 1}-output.f32",
        expected_count,
    )
    oracle_final = read_exact_f32(
        args.oracle_dir / f"l_out-{layers - 1}.f32", expected_count
    )
    direction = tuple(
        float(baseline) - float(oracle)
        for baseline, oracle in zip(baseline_input, oracle_input, strict=True)
    )

    loaded: list[dict[str, object]] = []
    seen_scales: set[float] = set()
    for item in runs:
        if not isinstance(item, dict) or not isinstance(item.get("scale"), (int, float)):
            raise ValueError("manifest run is invalid")
        scale = float(item["scale"])
        if not math.isfinite(scale) or scale in seen_scales:
            raise ValueError("manifest scales must be finite and unique")
        seen_scales.add(scale)
        directory = item.get("directory")
        if not isinstance(directory, str) or Path(directory).name != directory:
            raise ValueError("manifest run directory is invalid")
        run_dir = args.experiment_dir / directory
        residual_path = run_dir / "residual.f32"
        if item.get("residual_sha256") != sha256_file(residual_path):
            raise ValueError(f"{residual_path}: residual hash mismatch")
        residual = read_exact_f32(residual_path, expected_count)
        expected_residual = tuple(
            f32(float(oracle) + scale * delta)
            for oracle, delta in zip(oracle_input, direction, strict=True)
        )
        if residual != expected_residual:
            raise ValueError(f"{residual_path}: residual formula mismatch")
        final = read_exact_f32(
            run_dir / f"step-0-layer-{layers - 1}-output.f32", expected_count
        )
        routing = load_routing(
            run_dir / "result.json",
            start_layer=start_layer,
            layers=layers,
            expected_count=expected_count,
            kv_format=kv_format,
            activation_format=activation_format,
            prompt_token=prompt_token,
            ctx=ctx,
        )
        loaded.append(
            {
                "scale": scale,
                "directory": directory,
                "residual": residual,
                "final": final,
                "routing": routing,
            }
        )
    zero = next((row for row in loaded if row["scale"] == 0.0), None)
    if zero is None:
        raise ValueError("manifest must contain scale zero")
    zero_final = zero["final"]
    zero_routing = zero["routing"]
    assert isinstance(zero_final, tuple) and isinstance(zero_routing, dict)
    direction_norm2 = math.fsum(delta * delta for delta in direction)
    if direction_norm2 <= 0.0:
        raise ValueError("baseline perturbation direction is zero")

    rows: list[dict[str, object]] = []
    order_transition_scales: list[float] = []
    membership_transition_scales: list[float] = []
    for loaded_row in loaded:
        scale = float(loaded_row["scale"])
        residual = loaded_row["residual"]
        final = loaded_row["final"]
        routing = loaded_row["routing"]
        assert isinstance(residual, tuple) and isinstance(final, tuple) and isinstance(routing, dict)
        effective_input = metrics(residual, oracle_input)
        effective_delta = tuple(
            value - oracle
            for value, oracle in zip(residual, oracle_input, strict=True)
        )
        requested_delta = tuple(scale * delta for delta in direction)
        direction_fit_metrics = metrics(effective_delta, requested_delta)
        projected_scale = (
            math.fsum(
                value * delta
                for value, delta in zip(effective_delta, direction, strict=True)
            )
            / direction_norm2
        )
        final_vs_zero = metrics(final, zero_final)
        order_transitions = [
            layer
            for layer in range(start_layer, layers)
            if routing[layer]["selected_experts"]
            != zero_routing[layer]["selected_experts"]
        ]
        membership_transitions = [
            layer
            for layer in range(start_layer, layers)
            if set(routing[layer]["selected_experts"])
            != set(zero_routing[layer]["selected_experts"])
        ]
        if order_transitions:
            order_transition_scales.append(scale)
        if membership_transitions:
            membership_transition_scales.append(scale)
        gain = None
        if effective_input["l2"] > 0.0:
            gain = final_vs_zero["l2"] / effective_input["l2"]
            if not math.isfinite(gain):
                raise ValueError(f"non-finite suffix response gain at scale {scale}")
        rows.append(
            {
                "scale": scale,
                "directory": loaded_row["directory"],
                "effective_input_delta": effective_input,
                "requested_direction_fit": {
                    "projected_scale": projected_scale,
                    "cosine": direction_fit_metrics["cosine"],
                    "error_l2": direction_fit_metrics["l2"],
                },
                "final_vs_oracle": metrics(final, oracle_final),
                "final_vs_scale_zero": final_vs_zero,
                "suffix_response_gain_l2": gain,
                "routing_order_transition_layers": order_transitions,
                "routing_order_transition_count": len(order_transitions),
                "routing_membership_transition_layers": membership_transitions,
                "routing_membership_transition_count": len(membership_transitions),
            }
        )

    return {
        "schema": REPORT_SCHEMA,
        "matrix": matrix,
        "direction": metrics(baseline_input, oracle_input),
        "rows": rows,
        "routing": {
            "reference_scale": 0.0,
            "order_transition_scales": order_transition_scales,
            "membership_transition_scales": membership_transition_scales,
        },
        "verdict": (
            "topk_membership_transition_observed"
            if membership_transition_scales
            else "topk_rank_order_transition_observed"
            if order_transition_scales
            else "no_topk_transition_observed"
        ),
        "limitation": (
            "One token at position zero; residual replay does not replace accumulated prior-token KV. "
            "Absence of a top-8 transition does not by itself prove a globally linear suffix."
        ),
    }


def execute_runs(args: argparse.Namespace, manifest: dict[str, object]) -> None:
    if not args.qxqxf.is_file() or not args.model.is_file():
        raise ValueError("qxqxf and model must be existing files")
    matrix = manifest["matrix"]
    runs = manifest["runs"]
    assert isinstance(matrix, dict) and isinstance(runs, list)
    for item in runs:
        assert isinstance(item, dict)
        run_dir = args.experiment_dir / str(item["directory"])
        command = [
            str(args.qxqxf),
            "state-loop-probe",
            "--in",
            str(args.model),
            "--prompt-token",
            str(matrix["prompt_token"]),
            "--steps",
            "1",
            "--layers",
            str(matrix["layers"]),
            "--start-layer",
            str(matrix["start_layer"]),
            "--residual-in",
            str(run_dir / "residual.f32"),
            "--ctx",
            str(matrix["ctx"]),
            "--kv",
            str(matrix["kv_format"]),
            "--activation",
            str(matrix["activation_format"]),
            "--temperature",
            "0",
            "--seed",
            str(matrix["seed"]),
            "--full-moe",
            "--dump-residuals",
            str(run_dir),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        (run_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"scale {item['scale']} replay failed with exit {completed.returncode}: "
                f"{completed.stderr.strip()}"
            )
        json.loads(completed.stdout)
        (run_dir / "result.json").write_text(completed.stdout, encoding="utf-8")


def add_common_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--scales", required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--start-layer", type=int, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--kv-format", required=True)
    parser.add_argument("--activation-format", required=True)
    parser.add_argument("--prompt-token", type=int, default=42)
    parser.add_argument("--ctx", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    add_common_prepare_arguments(prepare)
    run = subparsers.add_parser("run")
    add_common_prepare_arguments(run)
    run.add_argument("--qxqxf", type=Path, required=True)
    run.add_argument("--model", type=Path, required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--oracle-dir", type=Path, required=True)
    analyze.add_argument("--baseline-dir", type=Path, required=True)
    analyze.add_argument("--experiment-dir", type=Path, required=True)
    analyze.add_argument("--output", type=Path)
    return parser.parse_args()


def render_report(report: dict[str, object], output: Path | None) -> None:
    rendered = json.dumps(report, indent=2) + "\n"
    if output:
        output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)


def main() -> int:
    args = parse_args()
    try:
        if args.command == "prepare":
            manifest = prepare_experiment(args)
            sys.stdout.write(json.dumps(manifest, indent=2) + "\n")
        elif args.command == "run":
            manifest = prepare_experiment(args)
            execute_runs(args, manifest)
            analysis_args = argparse.Namespace(
                oracle_dir=args.oracle_dir,
                baseline_dir=args.baseline_dir,
                experiment_dir=args.experiment_dir,
            )
            render_report(analyze_experiment(analysis_args), args.experiment_dir / "report.json")
        else:
            render_report(analyze_experiment(args), args.output)
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
