#!/usr/bin/env python3
"""Run and aggregate scaled residual replay across token, activation, and KV modes."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "qx-scaled-residual-matrix-manifest-v1"
REPORT_SCHEMA = "qx-scaled-residual-matrix-v1"
CELL_REPORT_SCHEMA = "qx-scaled-residual-replay-v1"
ACTIVATION_FORMATS = {"f32", "q8_k_compat"}
KV_FORMATS = {"f16", "f32", "int8"}
ORACLE_KV_FORMATS = {"f16": "f16", "f32": "f32", "int8": "q8_0"}
VERDICTS = {
    "no_topk_transition_observed",
    "topk_rank_order_transition_observed",
    "topk_membership_transition_observed",
}
SCRIPT_DIR = Path(__file__).resolve().parent
CELL_SCRIPT = SCRIPT_DIR / "scaled_residual_replay.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_json_text(text: str, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label}: non-finite JSON number {value}")

    data = json.loads(text, parse_constant=reject_constant)
    if not isinstance(data, dict):
        raise ValueError(f"{label}: report root must be a JSON object")
    validate_finite(data, label)
    return data


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    return parse_json_text(path.read_text(encoding="utf-8"), label)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_finite(value: object, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label}: non-finite numeric value")
    if isinstance(value, dict):
        for key, child in value.items():
            validate_finite(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_finite(child, f"{label}[{index}]")


def parse_int_csv(text: str, label: str) -> list[int]:
    try:
        values = [int(part) for part in text.split(",")]
    except ValueError as exc:
        raise ValueError(f"{label} must be comma-separated integers") from exc
    if not values or any(value < 0 for value in values):
        raise ValueError(f"{label} must contain non-negative integers")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicates")
    return values


def parse_choice_csv(text: str, label: str, allowed: set[str]) -> list[str]:
    values = text.split(",")
    if not values or any(not value or value not in allowed for value in values):
        raise ValueError(f"{label} contains an unsupported value")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicates")
    return values


def parse_scales(text: str) -> list[float]:
    try:
        values = [float(part) for part in text.split(",")]
    except ValueError as exc:
        raise ValueError("scales must be comma-separated finite numbers") from exc
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("scales must be comma-separated finite numbers")
    if len(values) != len(set(values)):
        raise ValueError("scales must not contain duplicates")
    if 0.0 not in values or len(values) < 2:
        raise ValueError("scales must include zero and at least one perturbation")
    return values


def validate_dimensions(layers: int, start_layer: int, expected_count: int, ctx: int, seed: int) -> None:
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (layers, start_layer, expected_count, ctx, seed)
    ):
        raise ValueError("invalid matrix dimensions")
    if layers <= 0 or start_layer <= 0 or start_layer >= layers:
        raise ValueError("require 0 < start-layer < layers")
    if expected_count <= 0 or ctx <= 0 or seed < 0:
        raise ValueError("expected-count and ctx must be positive; seed must be non-negative")


def cell_directory(token: int, activation: str, kv_format: str) -> str:
    return f"cells/token-{token}/activation-{activation}/kv-{kv_format}"


def artifact_entry(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"required artifact is not a file: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    tokens = parse_int_csv(args.tokens, "tokens")
    activations = parse_choice_csv(args.activations, "activations", ACTIVATION_FORMATS)
    kv_formats = parse_choice_csv(args.kv_formats, "kv-formats", KV_FORMATS)
    scales = parse_scales(args.scales)
    validate_dimensions(args.layers, args.start_layer, args.expected_count, args.ctx, args.seed)
    artifacts = {
        "qxqxf": artifact_entry(args.qxqxf),
        "model": artifact_entry(args.model),
        "llama_oracle": artifact_entry(args.llama_oracle),
        "gguf": artifact_entry(args.gguf),
    }
    cells = []
    for token, activation, kv_format in itertools.product(tokens, activations, kv_formats):
        cells.append(
            {
                "prompt_token": token,
                "activation_format": activation,
                "kv_format": kv_format,
                "oracle_kv_format": ORACLE_KV_FORMATS[kv_format],
                "directory": cell_directory(token, activation, kv_format),
            }
        )
    return {
        "schema": MANIFEST_SCHEMA,
        "matrix": {
            "prompt_tokens": tokens,
            "activation_formats": activations,
            "kv_formats": kv_formats,
            "scales": scales,
            "layers": args.layers,
            "start_layer": args.start_layer,
            "expected_count": args.expected_count,
            "ctx": args.ctx,
            "temperature": 0.0,
            "seed": args.seed,
        },
        "artifacts": artifacts,
        "cells": cells,
    }


def prepare_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.experiment_dir.exists() and any(args.experiment_dir.iterdir()):
        raise ValueError(f"experiment directory is not empty: {args.experiment_dir}")
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    write_json(args.experiment_dir / "matrix-manifest.json", manifest)
    return manifest


def run_command(command: list[str], *, label: str, stderr_path: Path) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit {completed.returncode}: {completed.stderr.strip()}"
        )
    return parse_json_text(completed.stdout, label)


def validate_oracle_result(
    result: dict[str, Any], *, token: int, oracle_kv: str, matrix: dict[str, Any]
) -> None:
    start_layer = matrix["start_layer"]
    final_layer = matrix["layers"] - 1
    layer_rows = result.get("layers")
    internals = result.get("internals")
    if (
        type(result.get("schema")) is not int
        or result.get("schema") != 1
        or result.get("ok") is not True
        or type(result.get("token_id")) is not int
        or result.get("token_id") != token
        or result.get("kv_type") != oracle_kv
        or type(result.get("n_embd")) is not int
        or result.get("n_embd") != matrix["expected_count"]
        or type(result.get("n_layer")) is not int
        or result.get("n_layer") != matrix["layers"]
        or not isinstance(layer_rows, list)
        or not isinstance(internals, list)
    ):
        raise ValueError("oracle result metadata mismatch")
    layer = next(
        (
            row
            for row in layer_rows
            if isinstance(row, dict)
            and type(row.get("layer")) is int
            and row.get("layer") == start_layer
        ),
        None,
    )
    final = next(
        (
            row
            for row in internals
            if isinstance(row, dict) and row.get("name") == f"l_out-{final_layer}"
        ),
        None,
    )
    if (
        not isinstance(layer, dict)
        or type(layer.get("count")) is not int
        or layer.get("count") != matrix["expected_count"]
        or layer.get("written") is not True
        or not isinstance(final, dict)
        or type(final.get("count")) is not int
        or final.get("count") != matrix["expected_count"]
        or final.get("written") is not True
    ):
        raise ValueError("oracle result is missing required residual sidecars")


def validate_baseline_result(
    result: dict[str, Any], *, token: int, activation: str, kv_format: str, matrix: dict[str, Any]
) -> None:
    integer_fields = {
        "prompt_token": token,
        "prompt_token_count": 1,
        "generation_steps": 1,
        "steps": 1,
        "layers": matrix["layers"],
        "start_layer": 0,
        "ctx_tokens": matrix["ctx"],
        "layers_run": matrix["layers"],
        "kv_appends": matrix["layers"],
    }
    if (
        result.get("probe") != "state_loop"
        or any(
            type(result.get(field)) is not int or result.get(field) != expected
            for field, expected in integer_fields.items()
        )
        or not isinstance(result.get("prompt_token_ids"), list)
        or len(result["prompt_token_ids"]) != 1
        or type(result["prompt_token_ids"][0]) is not int
        or result["prompt_token_ids"][0] != token
        or result.get("kv_format") != kv_format
        or result.get("activation_format") != activation
        or result.get("residual_dump") is not True
        or result.get("layers_run") != matrix["layers"]
        or result.get("kv_appends") != matrix["layers"]
        or result.get("cache_readback_ok") is not True
    ):
        raise ValueError("baseline result metadata mismatch")


def validate_cell_report(
    report: dict[str, Any], *, cell: dict[str, Any], matrix: dict[str, Any]
) -> tuple[list[float], list[float], str]:
    def require_metrics(value: object, keys: set[str], label: str) -> None:
        if not isinstance(value, dict) or not keys.issubset(value):
            raise ValueError(f"cell report {label} metrics are incomplete")
        if any(
            not isinstance(value[key], (int, float))
            or isinstance(value[key], bool)
            or not math.isfinite(float(value[key]))
            for key in keys
        ):
            raise ValueError(f"cell report {label} metrics are invalid")

    if report.get("schema") != CELL_REPORT_SCHEMA:
        raise ValueError("cell report schema mismatch")
    cell_matrix = report.get("matrix")
    if not isinstance(cell_matrix, dict):
        raise ValueError("cell metadata mismatch: matrix is missing")
    expected = {
        "prompt_token": cell["prompt_token"],
        "activation_format": cell["activation_format"],
        "kv_format": cell["kv_format"],
        "layers": matrix["layers"],
        "start_layer": matrix["start_layer"],
        "expected_count": matrix["expected_count"],
        "ctx": matrix["ctx"],
        "seed": matrix["seed"],
    }
    integer_keys = {"prompt_token", "layers", "start_layer", "expected_count", "ctx", "seed"}
    if any(
        (key in integer_keys and type(cell_matrix.get(key)) is not int)
        or cell_matrix.get(key) != value
        for key, value in expected.items()
    ):
        raise ValueError("cell metadata mismatch")
    require_metrics(report.get("direction"), {"max_abs", "rmse", "l2", "cosine"}, "direction")
    if not isinstance(report.get("limitation"), str) or not report["limitation"].strip():
        raise ValueError("cell report limitation is missing")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != len(matrix["scales"]):
        raise ValueError("cell report does not contain the complete scale matrix")
    row_scales = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("scale"), (int, float))
            or isinstance(row.get("scale"), bool)
        ):
            raise ValueError("cell report contains an invalid scale row")
        scale = float(row["scale"])
        if not isinstance(row.get("directory"), str) or not row["directory"]:
            raise ValueError("cell report row metrics are incomplete")
        for key in ("effective_input_delta", "final_vs_oracle", "final_vs_scale_zero"):
            require_metrics(row.get(key), {"max_abs", "rmse", "l2", "cosine"}, "row")
        require_metrics(
            row.get("requested_direction_fit"),
            {"projected_scale", "cosine", "error_l2"},
            "row",
        )
        gain = row.get("suffix_response_gain_l2")
        if (scale == 0.0 and gain is not None) or (
            scale != 0.0
            and (
                not isinstance(gain, (int, float))
                or isinstance(gain, bool)
                or not math.isfinite(float(gain))
                or float(gain) < 0.0
            )
        ):
            raise ValueError("cell report row metrics are invalid")
        order_layers = row.get("routing_order_transition_layers")
        membership_layers = row.get("routing_membership_transition_layers")
        for layers, count_key in (
            (order_layers, "routing_order_transition_count"),
            (membership_layers, "routing_membership_transition_count"),
        ):
            count = row.get(count_key)
            if (
                not isinstance(layers, list)
                or any(
                    not isinstance(layer, int)
                    or isinstance(layer, bool)
                    or layer < matrix["start_layer"]
                    or layer >= matrix["layers"]
                    for layer in layers
                )
                or len(layers) != len(set(layers))
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count != len(layers)
            ):
                raise ValueError("cell report row routing metrics are invalid")
        if not set(membership_layers).issubset(set(order_layers)):
            raise ValueError("cell report membership transitions must also be order transitions")
        row_scales.append(scale)
    if len(row_scales) != len(set(row_scales)) or set(row_scales) != set(matrix["scales"]):
        raise ValueError("cell report does not contain the complete scale matrix")
    routing = report.get("routing")
    if (
        not isinstance(routing, dict)
        or not isinstance(routing.get("reference_scale"), (int, float))
        or isinstance(routing.get("reference_scale"), bool)
        or routing.get("reference_scale") != 0.0
    ):
        raise ValueError("cell report routing summary is invalid")
    order = routing.get("order_transition_scales")
    membership = routing.get("membership_transition_scales")
    for values in (order, membership):
        if (
            not isinstance(values, list)
            or any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                for value in values
            )
            or len(values) != len(set(float(value) for value in values))
            or not set(float(value) for value in values).issubset(set(matrix["scales"]))
        ):
            raise ValueError("cell report routing transition scales are invalid")
    order_values = [float(value) for value in order]
    membership_values = [float(value) for value in membership]
    if not set(membership_values).issubset(set(order_values)):
        raise ValueError("cell report membership scales must also be order-transition scales")
    verdict = report.get("verdict")
    if verdict not in VERDICTS:
        raise ValueError("cell report verdict is invalid")
    expected_verdict = (
        "topk_membership_transition_observed"
        if membership_values
        else "topk_rank_order_transition_observed"
        if order_values
        else "no_topk_transition_observed"
    )
    if verdict != expected_verdict:
        raise ValueError("cell report verdict is inconsistent with routing transitions")
    return order_values, membership_values, verdict


def validate_manifest(
    manifest: dict[str, Any],
    experiment_dir: Path,
    trusted_artifacts: dict[str, Path],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported matrix manifest schema")
    matrix = manifest.get("matrix")
    artifacts = manifest.get("artifacts")
    cells = manifest.get("cells")
    if not isinstance(matrix, dict) or not isinstance(artifacts, dict) or not isinstance(cells, list):
        raise ValueError("incomplete matrix manifest")
    tokens = matrix.get("prompt_tokens")
    activations = matrix.get("activation_formats")
    kv_formats = matrix.get("kv_formats")
    scales = matrix.get("scales")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in tokens
        )
        or len(tokens) != len(set(tokens))
        or not isinstance(activations, list)
        or not activations
        or any(value not in ACTIVATION_FORMATS for value in activations)
        or len(activations) != len(set(activations))
        or not isinstance(kv_formats, list)
        or not kv_formats
        or any(value not in KV_FORMATS for value in kv_formats)
        or len(kv_formats) != len(set(kv_formats))
        or not isinstance(scales, list)
        or not scales
        or any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in scales
        )
        or len(scales) != len(set(float(value) for value in scales))
        or 0.0 not in scales
    ):
        raise ValueError("invalid matrix dimensions")
    validate_dimensions(
        matrix.get("layers", 0),
        matrix.get("start_layer", 0),
        matrix.get("expected_count", 0),
        matrix.get("ctx", 0),
        matrix.get("seed", -1),
    )
    for name in ("qxqxf", "model", "llama_oracle", "gguf"):
        entry = artifacts.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"matrix artifact is missing: {name}")
        manifest_path = entry.get("path")
        trusted_path = trusted_artifacts[name].resolve()
        if not isinstance(manifest_path, str) or Path(manifest_path).resolve() != trusted_path:
            raise ValueError(f"matrix artifact path mismatch: {name}")
        if not trusted_path.is_file() or entry.get("sha256") != sha256_file(trusted_path):
            raise ValueError(f"matrix artifact hash mismatch: {name}")
    expected_keys = set(itertools.product(tokens, activations, kv_formats))
    actual_keys: set[tuple[int, str, str]] = set()
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("invalid matrix cell")
        if type(cell.get("prompt_token")) is not int:
            raise ValueError("invalid matrix cell metadata")
        key = (
            cell.get("prompt_token"),
            cell.get("activation_format"),
            cell.get("kv_format"),
        )
        if key in actual_keys:
            raise ValueError("matrix contains duplicate cells")
        actual_keys.add(key)
        if key not in expected_keys:
            raise ValueError("matrix contains an unexpected cell")
        token, activation, kv_format = key
        expected_directory = cell_directory(token, activation, kv_format)
        if (
            cell.get("directory") != expected_directory
            or cell.get("oracle_kv_format") != ORACLE_KV_FORMATS[kv_format]
        ):
            raise ValueError("cell metadata mismatch")
    if actual_keys != expected_keys:
        raise ValueError("manifest does not contain the complete matrix")
    return matrix, cells


def analyze_matrix(
    experiment_dir: Path, trusted_artifacts: dict[str, Path]
) -> dict[str, Any]:
    manifest_path = experiment_dir / "matrix-manifest.json"
    manifest = load_json_object(manifest_path, "matrix manifest")
    matrix, cells = validate_manifest(manifest, experiment_dir, trusted_artifacts)
    summaries = []
    order_count = 0
    membership_count = 0
    for cell in cells:
        token = cell["prompt_token"]
        activation = cell["activation_format"]
        kv_format = cell["kv_format"]
        expected_path = (experiment_dir / cell["directory"] / "report.json").resolve()
        report_entry = cell.get("report")
        if not isinstance(report_entry, dict):
            raise ValueError("manifest does not contain the complete matrix outputs")
        report_path = Path(str(report_entry.get("path"))).resolve()
        if report_path != expected_path or not report_path.is_file():
            raise ValueError("cell report path mismatch or missing output")
        if report_entry.get("sha256") != sha256_file(report_path):
            raise ValueError("cell report hash mismatch")
        report = load_json_object(report_path, str(report_path))
        order, membership, verdict = validate_cell_report(report, cell=cell, matrix=matrix)
        response_curve = [
            {
                "scale": float(row["scale"]),
                "final_vs_scale_zero_l2": float(row["final_vs_scale_zero"]["l2"]),
                "suffix_response_gain_l2": (
                    None
                    if row["suffix_response_gain_l2"] is None
                    else float(row["suffix_response_gain_l2"])
                ),
                "routing_order_transition_count": row["routing_order_transition_count"],
                "routing_membership_transition_count": row[
                    "routing_membership_transition_count"
                ],
            }
            for row in report["rows"]
        ]
        response_by_scale = {row["scale"]: row for row in response_curve}
        unit_response = None
        if -1.0 in response_by_scale and 1.0 in response_by_scale:
            minus_one_l2 = response_by_scale[-1.0]["final_vs_scale_zero_l2"]
            plus_one_l2 = response_by_scale[1.0]["final_vs_scale_zero_l2"]
            smaller_l2 = min(minus_one_l2, plus_one_l2)
            unit_response = {
                "minus_one_l2": minus_one_l2,
                "plus_one_l2": plus_one_l2,
                "max_to_min_l2_ratio": (
                    None
                    if smaller_l2 == 0.0
                    else max(minus_one_l2, plus_one_l2) / smaller_l2
                ),
            }
        order_count += int(bool(order))
        membership_count += int(bool(membership))
        summaries.append(
            {
                "prompt_token": token,
                "activation_format": activation,
                "kv_format": kv_format,
                "oracle_kv_format": cell["oracle_kv_format"],
                "report_sha256": report_entry["sha256"],
                "direction": report.get("direction"),
                "response_curve": response_curve,
                "unit_response": unit_response,
                "order_transition_scales": order,
                "membership_transition_scales": membership,
                "verdict": verdict,
            }
        )
    return {
        "schema": REPORT_SCHEMA,
        "matrix": matrix,
        "artifacts": manifest["artifacts"],
        "summary": {
            "cell_count": len(summaries),
            "order_transition_cell_count": order_count,
            "membership_transition_cell_count": membership_count,
        },
        "cells": summaries,
        "limitation": (
            "These are independent one-token, position-zero runs. They do not validate "
            "accumulated multi-token KV snapshot/replay or autoregressive equivalence."
        ),
    }


def execute_matrix(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    matrix = manifest["matrix"]
    artifacts = {name: Path(entry["path"]) for name, entry in manifest["artifacts"].items()}
    oracle_results: dict[tuple[int, str], dict[str, Any]] = {}
    for cell in manifest["cells"]:
        token = cell["prompt_token"]
        activation = cell["activation_format"]
        kv_format = cell["kv_format"]
        oracle_kv = cell["oracle_kv_format"]
        oracle_key = (token, kv_format)
        oracle_dir = args.experiment_dir / "sources" / f"token-{token}" / f"oracle-{kv_format}"
        if oracle_key not in oracle_results:
            oracle_dir.mkdir(parents=True)
            oracle_command = [
                str(artifacts["llama_oracle"]),
                str(artifacts["gguf"]),
                str(oracle_dir),
                str(token),
                str(matrix["start_layer"]),
                oracle_kv,
                f"internals={matrix['start_layer']}",
            ]
            oracle_result = run_command(
                oracle_command,
                label=f"oracle token={token} kv={kv_format}",
                stderr_path=oracle_dir / "stderr.txt",
            )
            validate_oracle_result(
                oracle_result, token=token, oracle_kv=oracle_kv, matrix=matrix
            )
            write_json(oracle_dir / "result.json", oracle_result)
            oracle_results[oracle_key] = {
                "path": str((oracle_dir / "result.json").resolve()),
                "sha256": sha256_file(oracle_dir / "result.json"),
                "command": oracle_command,
            }
        baseline_dir = (
            args.experiment_dir
            / "sources"
            / f"token-{token}"
            / f"activation-{activation}"
            / f"kv-{kv_format}"
            / "baseline"
        )
        baseline_dir.mkdir(parents=True)
        baseline_command = [
            str(artifacts["qxqxf"]),
            "state-loop-probe",
            "--in",
            str(artifacts["model"]),
            "--prompt-token",
            str(token),
            "--steps",
            "1",
            "--layers",
            str(matrix["layers"]),
            "--ctx",
            str(matrix["ctx"]),
            "--kv",
            kv_format,
            "--activation",
            activation,
            "--temperature",
            "0",
            "--seed",
            str(matrix["seed"]),
            "--full-moe",
            "--dump-residuals",
            str(baseline_dir),
        ]
        baseline_result = run_command(
            baseline_command,
            label=f"baseline token={token} activation={activation} kv={kv_format}",
            stderr_path=baseline_dir / "stderr.txt",
        )
        validate_baseline_result(
            baseline_result,
            token=token,
            activation=activation,
            kv_format=kv_format,
            matrix=matrix,
        )
        write_json(baseline_dir / "result.json", baseline_result)
        cell_dir = args.experiment_dir / cell["directory"]
        cell_command = [
            sys.executable,
            str(CELL_SCRIPT),
            "run",
            "--qxqxf",
            str(artifacts["qxqxf"]),
            "--model",
            str(artifacts["model"]),
            "--oracle-dir",
            str(oracle_dir),
            "--baseline-dir",
            str(baseline_dir),
            "--experiment-dir",
            str(cell_dir),
            f"--scales={','.join(format(value, '.17g') for value in matrix['scales'])}",
            "--layers",
            str(matrix["layers"]),
            "--start-layer",
            str(matrix["start_layer"]),
            "--expected-count",
            str(matrix["expected_count"]),
            "--kv-format",
            kv_format,
            "--activation-format",
            activation,
            "--prompt-token",
            str(token),
            "--ctx",
            str(matrix["ctx"]),
            "--seed",
            str(matrix["seed"]),
        ]
        cell_report = run_command(
            cell_command,
            label=f"cell token={token} activation={activation} kv={kv_format}",
            stderr_path=cell_dir.parent / f"{cell_dir.name}-stderr.txt",
        )
        validate_cell_report(cell_report, cell=cell, matrix=matrix)
        report_path = cell_dir / "report.json"
        if not report_path.is_file():
            raise ValueError("cell runner did not write report.json")
        cell["sources"] = {
            "oracle": oracle_results[oracle_key],
            "baseline": {
                "path": str((baseline_dir / "result.json").resolve()),
                "sha256": sha256_file(baseline_dir / "result.json"),
                "command": baseline_command,
            },
            "cell_command": cell_command,
        }
        cell["report"] = {
            "path": str(report_path.resolve()),
            "sha256": sha256_file(report_path),
        }
        write_json(args.experiment_dir / "matrix-manifest.json", manifest)
    report = analyze_matrix(args.experiment_dir, artifacts)
    write_json(args.experiment_dir / "matrix-report.json", report)
    return report


def add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--qxqxf", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--llama-oracle", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--kv-formats", required=True)
    parser.add_argument("--scales", required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--start-layer", type=int, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--ctx", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)


def add_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--qxqxf", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--llama-oracle", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    add_plan_arguments(plan)
    run = subparsers.add_parser("run")
    add_plan_arguments(run)
    analyze = subparsers.add_parser("analyze")
    add_artifact_arguments(analyze)
    analyze.add_argument("--experiment-dir", type=Path, required=True)
    analyze.add_argument("--output", type=Path)
    return parser.parse_args()


def render(value: object, output: Path | None = None) -> None:
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if output is not None:
        output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)


def main() -> int:
    args = parse_args()
    try:
        if args.command == "plan":
            render(prepare_plan(args))
        elif args.command == "run":
            manifest = prepare_plan(args)
            render(execute_matrix(args, manifest))
        else:
            trusted_artifacts = {
                "qxqxf": args.qxqxf,
                "model": args.model,
                "llama_oracle": args.llama_oracle,
                "gguf": args.gguf,
            }
            render(analyze_matrix(args.experiment_dir, trusted_artifacts), args.output)
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
