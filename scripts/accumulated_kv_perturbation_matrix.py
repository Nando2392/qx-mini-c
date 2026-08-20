#!/usr/bin/env python3
"""Plan and run fail-closed accumulated-KV residual perturbation matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


MANIFEST_SCHEMA = "qx-accumulated-kv-perturbation-manifest-v1"
REPORT_SCHEMA = "qx-accumulated-kv-perturbation-report-v1"
KV_FORMATS = {"f16", "int8"}
ACTIVATION_FORMATS = {"f32", "q8_k_compat"}
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def read_exact_f32(path: Path, expected_count: int) -> tuple[float, ...]:
    raw = path.read_bytes()
    if len(raw) != expected_count * 4:
        raise ValueError(
            f"{path}: expected {expected_count * 4} bytes, found {len(raw)}"
        )
    values = struct.unpack(f"<{expected_count}f", raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path}: contains NaN or Inf")
    return values


def write_f32(path: Path, values: Sequence[float]) -> None:
    _atomic_write_bytes(path, struct.pack(f"<{len(values)}f", *values))


def scaled_residual(
    baseline: Sequence[float], reference: Sequence[float], scale: float
) -> tuple[float, ...]:
    if len(baseline) != len(reference) or not math.isfinite(scale):
        raise ValueError("scaled residual inputs are invalid")
    return tuple(
        f32(base + scale * (target - base))
        for base, target in zip(baseline, reference, strict=True)
    )


def vector_metrics(actual: Sequence[float], reference: Sequence[float]) -> dict[str, float]:
    if not actual or len(actual) != len(reference):
        raise ValueError("residual vector length mismatch")
    deltas = [left - right for left, right in zip(actual, reference, strict=True)]
    squared_error = math.fsum(delta * delta for delta in deltas)
    actual_norm2 = math.fsum(value * value for value in actual)
    reference_norm2 = math.fsum(value * value for value in reference)
    denominator = math.sqrt(actual_norm2 * reference_norm2)
    dot = math.fsum(left * right for left, right in zip(actual, reference, strict=True))
    result = {
        "max_abs": max(abs(delta) for delta in deltas),
        "rmse": math.sqrt(squared_error / len(deltas)),
        "l2": math.sqrt(squared_error),
        "cosine": dot / denominator if denominator else (1.0 if actual == reference else 0.0),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("non-finite residual metric")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_object_from_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be a JSON object")
    return value


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_text(path: Path, value: str) -> None:
    _atomic_write_bytes(path, value.encode("utf-8"))


def write_json(path: Path, value: object) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def artifact_entry(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"required artifact is not a file: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def parse_choice_csv(raw: str, label: str, allowed: set[str]) -> list[str]:
    values = raw.split(",")
    if not values or any(not value or value not in allowed for value in values):
        raise ValueError(f"invalid {label}")
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {label}")
    return values


def parse_scales(raw: str) -> list[float]:
    try:
        values = [float(item) for item in raw.split(",")]
    except ValueError as exc:
        raise ValueError("invalid scales") from exc
    if any(not math.isfinite(value) for value in values):
        raise ValueError("scales must be finite")
    if len(set(values)) != len(values):
        raise ValueError("scales must be unique")
    if values.count(0.0) != 1 or not any(value < 0.0 for value in values) or not any(
        value > 0.0 for value in values
    ):
        raise ValueError("scales require one zero, one negative, and one positive value")
    return values


def validate_dimensions(args: argparse.Namespace) -> None:
    integer_fields = (
        "prompt_token",
        "prefix_steps",
        "continuation_steps",
        "layers",
        "start_layer",
        "expected_count",
        "direction_index",
        "ctx",
        "seed",
    )
    if any(type(getattr(args, field)) is not int for field in integer_fields):
        raise ValueError("experiment dimensions must be integers")
    if args.prompt_token < 0 or args.seed < 0:
        raise ValueError("prompt token and seed must be non-negative")
    if args.prefix_steps < 2:
        raise ValueError("prefix requires at least two accumulated positions")
    if args.continuation_steps != 1:
        raise ValueError("current residual replay seam supports exactly one continuation step")
    if args.layers <= 1 or not 0 < args.start_layer < args.layers:
        raise ValueError("require 0 < start-layer < layers")
    if args.expected_count <= 0:
        raise ValueError("expected-count must be positive")
    if args.direction_index < 0 or args.direction_index >= args.expected_count:
        raise ValueError("direction-index is outside the residual vector")
    if not math.isfinite(args.direction_magnitude) or args.direction_magnitude == 0.0:
        raise ValueError("direction-magnitude must be finite and non-zero")
    if args.ctx < args.prefix_steps + args.continuation_steps:
        raise ValueError("ctx is too small for prefix plus continuation")


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    validate_dimensions(args)
    if not REVISION_RE.fullmatch(args.revision):
        raise ValueError("revision must be a lowercase 40-character Git commit")
    kv_formats = parse_choice_csv(args.kv_formats, "KV formats", KV_FORMATS)
    if kv_formats != ["f16", "int8"]:
        raise ValueError("matrix requires KV formats in canonical f16,int8 order")
    if args.activation_format not in ACTIVATION_FORMATS:
        raise ValueError("unsupported activation format")
    scales = parse_scales(args.scales)
    cells = []
    for cell_ordinal, kv_format in enumerate(kv_formats):
        directory = f"cells/kv-{kv_format}"
        cells.append(
            {
                "ordinal": cell_ordinal,
                "kv_format": kv_format,
                "directory": directory,
                "runs": [
                    {
                        "ordinal": run_ordinal,
                        "scale": scale,
                        "directory": f"{directory}/runs/scale-{run_ordinal:03d}",
                    }
                    for run_ordinal, scale in enumerate(scales)
                ],
            }
        )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "revision": args.revision,
        "experiment": {
            "prompt_token": args.prompt_token,
            "prefix_steps": args.prefix_steps,
            "continuation_steps": args.continuation_steps,
            "kv_formats": kv_formats,
            "scales": scales,
            "activation_format": args.activation_format,
            "layers": args.layers,
            "start_layer": args.start_layer,
            "expected_count": args.expected_count,
            "direction_index": args.direction_index,
            "direction_magnitude": args.direction_magnitude,
            "runtime_mode": "full_moe",
            "ctx": args.ctx,
            "seed": args.seed,
        },
        "artifacts": {
            "qxqxf": artifact_entry(args.qxqxf),
            "model": artifact_entry(args.model),
        },
        "cells": cells,
    }
    if getattr(args, "tokens", None) is not None:
        manifest["artifacts"]["tokens"] = artifact_entry(args.tokens)
    return manifest


def prepare_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.experiment_dir.exists() and any(args.experiment_dir.iterdir()):
        raise ValueError(f"experiment directory is not empty: {args.experiment_dir}")
    manifest = build_manifest(args)
    write_json(args.experiment_dir / "matrix-manifest.json", manifest)
    return manifest


def json_int_equals(value: object, expected: int) -> bool:
    return type(value) is int and value == expected


def _relative_artifact(root: Path, path: Path) -> dict[str, str]:
    root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact escapes experiment directory: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"required artifact is not a file: {path}")
    return {"path": relative.as_posix(), "sha256": sha256_file(resolved)}


def _verify_relative_artifact(
    root: Path, entry: object, expected_relative: str, label: str
) -> Path:
    if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
        raise ValueError(f"invalid artifact record: {label}")
    if entry.get("path") != expected_relative:
        raise ValueError(f"artifact path mismatch: {label}")
    path = (root.resolve() / expected_relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"artifact path traversal: {label}") from exc
    if not path.is_file() or entry.get("sha256") != sha256_file(path):
        raise ValueError(f"artifact hash mismatch: {label}")
    return path


def _runtime_command(
    args: argparse.Namespace,
    *,
    kv_format: str,
    prompt_token: int,
    steps: int,
    dump_dir: Path,
    snapshot_out: Path | None = None,
    snapshot_in: Path | None = None,
    start_layer: int | None = None,
    residual_in: Path | None = None,
) -> list[str]:
    command = [
        str(args.qxqxf),
        "state-loop-probe",
        "--in",
        str(args.model),
        "--prompt-token",
        str(prompt_token),
        "--steps",
        str(steps),
        "--layers",
        str(args.layers),
        "--ctx",
        str(args.ctx),
        "--kv",
        kv_format,
        "--activation",
        args.activation_format,
        "--temperature",
        "0",
        "--seed",
        str(args.seed),
        "--dump-residuals",
        str(dump_dir),
    ]
    command.append("--full-moe")
    if getattr(args, "tokens", None) is not None:
        command.extend(["--tokens", str(args.tokens)])
    if snapshot_out is not None:
        command.extend(["--kv-snapshot-out", str(snapshot_out)])
    if snapshot_in is not None:
        command.extend(["--kv-snapshot-in", str(snapshot_in)])
    if start_layer is not None or residual_in is not None:
        if start_layer is None or residual_in is None:
            raise ValueError("start-layer and residual-in must be provided together")
        command.extend(["--start-layer", str(start_layer), "--residual-in", str(residual_in)])
    return command


def _run_json(command: list[str], result_path: Path) -> dict[str, Any]:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, text=True, capture_output=True)
    _atomic_write_text(
        result_path.parent / f"{result_path.stem}.stderr.txt", completed.stderr
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"runtime failed with exit {completed.returncode}: {completed.stderr.strip()}"
        )
    _atomic_write_text(result_path, completed.stdout)
    return load_json_object(result_path, "runtime result")


def _validate_runtime_result(
    value: dict[str, Any],
    *,
    kv_format: str,
    activation_format: str,
    prompt_token: int,
    steps: int,
    layers: int,
    position_base: int,
    start_layer: int = 0,
) -> None:
    if value.get("probe") != "state_loop":
        raise ValueError("runtime result probe mismatch")
    expected_ints = {
        "prompt_token": prompt_token,
        "steps": steps,
        "layers_run": (layers - start_layer) * steps,
        "position_base": position_base,
    }
    for key, expected in expected_ints.items():
        if not json_int_equals(value.get(key), expected):
            raise ValueError(f"runtime result {key} mismatch")
    if value.get("kv_format") != kv_format or value.get("activation_format") != activation_format:
        raise ValueError("runtime result format mismatch")
    tokens = value.get("tokens")
    if not isinstance(tokens, list) or len(tokens) != steps:
        raise ValueError("runtime result token count mismatch")
    for ordinal, token in enumerate(tokens):
        if not isinstance(token, dict):
            raise ValueError("runtime token entry must be an object")
        if not json_int_equals(token.get("position"), position_base + ordinal):
            raise ValueError("runtime token position mismatch")
        token_layers = token.get("layers")
        if not isinstance(token_layers, list) or not token_layers:
            raise ValueError("runtime token layers are incomplete")
        layer_ids = [item.get("layer") if isinstance(item, dict) else None for item in token_layers]
        if any(type(layer_id) is not int for layer_id in layer_ids):
            raise ValueError("runtime layer id must be an integer")
        if layer_ids != list(range(start_layer, layers)):
            raise ValueError("runtime layers are partial or disordered")


def _continuation_suffix(token: dict[str, Any], start_layer: int) -> list[dict[str, Any]]:
    layers = token.get("layers")
    if not isinstance(layers, list):
        raise ValueError("runtime layers are missing")
    suffix = [item for item in layers if isinstance(item, dict) and item.get("layer", -1) >= start_layer]
    if not suffix or suffix[0].get("layer") != start_layer:
        raise ValueError("runtime suffix is incomplete")
    return suffix


def _selected_experts(token: dict[str, Any], start_layer: int) -> list[list[int]]:
    result: list[list[int]] = []
    for layer in _continuation_suffix(token, start_layer):
        experts = layer.get("selected_experts")
        if not isinstance(experts, list) or any(type(item) is not int for item in experts):
            raise ValueError("runtime selected_experts is invalid")
        result.append(experts)
    return result


def _dump_path(directory: Path, layer: int) -> Path:
    candidates = [directory / f"l_out-{layer}.f32"]
    candidates.extend(sorted(directory.glob(f"step-*-layer-{layer}-output.f32")))
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"expected one residual dump for layer {layer} in {directory}")
    return matches[0]


def _build_snapshot_manifest(
    args: argparse.Namespace,
    snapshot_path: Path,
    output_path: Path,
    prompt_tokens: list[int],
) -> dict[str, Any]:
    helper = Path(__file__).resolve().with_name("kv_snapshot_replay.py")
    command = [
        sys.executable,
        str(helper),
        "build",
        "--snapshot",
        str(snapshot_path),
        "--model",
        str(args.model),
        "--binary",
        str(args.qxqxf),
        "--revision",
        args.revision,
        "--activation",
        args.activation_format,
        "--prompt-tokens",
        ",".join(str(token) for token in prompt_tokens),
        "--out",
        str(output_path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"snapshot manifest build failed: {completed.stderr.strip()}")
    return load_json_object(output_path, "snapshot manifest")


def _validate_snapshot_with_helper(
    manifest_path: Path,
    *,
    model: Path,
    binary: Path,
    geometry: dict[str, Any],
) -> None:
    helper = Path(__file__).resolve().with_name("kv_snapshot_replay.py")
    with tempfile.TemporaryDirectory(prefix="qx-kv-geometry-") as temporary:
        expected_path = Path(temporary) / "expected.json"
        write_json(expected_path, geometry)
        completed = subprocess.run(
            [
                sys.executable,
                str(helper),
                "validate",
                "--manifest",
                str(manifest_path),
                "--model",
                str(model),
                "--binary",
                str(binary),
                "--expected-geometry",
                str(expected_path),
            ],
            text=True,
            capture_output=True,
        )
    if completed.returncode != 0:
        raise ValueError(f"snapshot helper validation failed: {completed.stderr.strip()}")


def _run_cell(
    args: argparse.Namespace, root: Path, cell: dict[str, Any]
) -> dict[str, Any]:
    kv_format = str(cell["kv_format"])
    cell_dir = root / str(cell["directory"])
    baseline_dir = cell_dir / "baseline"
    capture_dir = cell_dir / "capture"
    replay_dir = cell_dir / "replay"
    snapshot_path = cell_dir / "snapshot.bin"
    baseline_path = baseline_dir / "result.json"
    capture_path = capture_dir / "result.json"
    replay_path = replay_dir / "result.json"

    baseline = _run_json(
        _runtime_command(
            args,
            kv_format=kv_format,
            prompt_token=args.prompt_token,
            steps=args.prefix_steps + 1,
            dump_dir=baseline_dir,
        ),
        baseline_path,
    )
    capture = _run_json(
        _runtime_command(
            args,
            kv_format=kv_format,
            prompt_token=args.prompt_token,
            steps=args.prefix_steps,
            dump_dir=capture_dir,
            snapshot_out=snapshot_path,
        ),
        capture_path,
    )
    _validate_runtime_result(
        baseline,
        kv_format=kv_format,
        activation_format=args.activation_format,
        prompt_token=args.prompt_token,
        steps=args.prefix_steps + 1,
        layers=args.layers,
        position_base=0,
    )
    _validate_runtime_result(
        capture,
        kv_format=kv_format,
        activation_format=args.activation_format,
        prompt_token=args.prompt_token,
        steps=args.prefix_steps,
        layers=args.layers,
        position_base=0,
    )
    baseline_tokens = baseline["tokens"]
    capture_tokens = capture["tokens"]
    if capture_tokens != baseline_tokens[: args.prefix_steps]:
        raise ValueError(f"{kv_format}: baseline/capture prefix is not exact")
    continuation_token = capture.get("final_token")
    if type(continuation_token) is not int:
        raise ValueError(f"{kv_format}: capture final_token must be an integer")
    prompt_tokens = [token.get("input_token") for token in capture_tokens]
    if any(type(token) is not int for token in prompt_tokens):
        raise ValueError(f"{kv_format}: capture prompt tokens are invalid")
    snapshot_manifest_path = cell_dir / "snapshot-manifest.json"
    snapshot_manifest = _build_snapshot_manifest(
        args, snapshot_path, snapshot_manifest_path, prompt_tokens
    )
    producer = snapshot_manifest.get("producer")
    geometry = snapshot_manifest.get("geometry")
    if not isinstance(producer, dict) or producer.get("revision") != args.revision:
        raise ValueError(f"{kv_format}: snapshot revision mismatch")
    if not isinstance(geometry, dict):
        raise ValueError(f"{kv_format}: snapshot geometry is missing")
    expected_geometry = {
        "layers": args.layers,
        "ctx_tokens": args.ctx,
        "positions": args.prefix_steps,
        "kv_format": kv_format,
        "activation_format": args.activation_format,
        "next_token": continuation_token,
    }
    for key, expected in expected_geometry.items():
        if geometry.get(key) != expected or (
            isinstance(expected, int) and type(geometry.get(key)) is not int
        ):
            raise ValueError(f"{kv_format}: snapshot geometry mismatch for {key}")

    replay = _run_json(
        _runtime_command(
            args,
            kv_format=kv_format,
            prompt_token=continuation_token,
            steps=1,
            dump_dir=replay_dir,
            snapshot_in=snapshot_path,
        ),
        replay_path,
    )
    _validate_runtime_result(
        replay,
        kv_format=kv_format,
        activation_format=args.activation_format,
        prompt_token=continuation_token,
        steps=1,
        layers=args.layers,
        position_base=args.prefix_steps,
    )
    if replay["tokens"][0] != baseline_tokens[args.prefix_steps]:
        raise ValueError(f"{kv_format}: uninterrupted/replay continuation is not exact")
    if replay.get("final_token") != baseline.get("final_token"):
        raise ValueError(f"{kv_format}: uninterrupted/replay final token mismatch")

    replay_input_path = _dump_path(replay_dir, args.start_layer - 1)
    replay_final_path = _dump_path(replay_dir, args.layers - 1)
    replay_input = read_exact_f32(replay_input_path, args.expected_count)
    read_exact_f32(replay_final_path, args.expected_count)
    reference = list(replay_input)
    reference[args.direction_index] = f32(
        reference[args.direction_index] + args.direction_magnitude
    )
    if reference[args.direction_index] == replay_input[args.direction_index]:
        raise ValueError("direction magnitude collapses in F32")
    reference_path = cell_dir / "reference-residual.f32"
    write_f32(reference_path, reference)

    completed_runs: list[dict[str, Any]] = []
    for run in cell["runs"]:
        scale = float(run["scale"])
        run_dir = root / str(run["directory"])
        residual_path = run_dir / "residual.f32"
        residual = scaled_residual(replay_input, reference, scale)
        if scale != 0.0 and residual == tuple(replay_input):
            raise ValueError(f"{kv_format}: scale {scale} collapses in F32")
        write_f32(residual_path, residual)
        result_path = run_dir / "result.json"
        result = _run_json(
            _runtime_command(
                args,
                kv_format=kv_format,
                prompt_token=continuation_token,
                steps=1,
                dump_dir=run_dir,
                snapshot_in=snapshot_path,
                start_layer=args.start_layer,
                residual_in=residual_path,
            ),
            result_path,
        )
        _validate_runtime_result(
            result,
            kv_format=kv_format,
            activation_format=args.activation_format,
            prompt_token=continuation_token,
            steps=1,
            layers=args.layers,
            position_base=args.prefix_steps,
            start_layer=args.start_layer,
        )
        final_path = _dump_path(run_dir, args.layers - 1)
        read_exact_f32(final_path, args.expected_count)
        completed_runs.append(
            {
                **run,
                "residual": _relative_artifact(root, residual_path),
                "result": _relative_artifact(root, result_path),
                "final_residual": _relative_artifact(root, final_path),
            }
        )

    zero = next(run for run in completed_runs if float(run["scale"]) == 0.0)
    zero_result_path = root / zero["result"]["path"]
    zero_result = load_json_object(zero_result_path, "zero-scale result")
    if _continuation_suffix(zero_result["tokens"][0], args.start_layer) != _continuation_suffix(
        replay["tokens"][0], args.start_layer
    ):
        raise ValueError(f"{kv_format}: zero-scale suffix is not exact")
    if zero_result.get("final_token") != replay.get("final_token"):
        raise ValueError(f"{kv_format}: zero-scale token is not exact")
    zero_final_path = root / zero["final_residual"]["path"]
    if zero_final_path.read_bytes() != replay_final_path.read_bytes():
        raise ValueError(f"{kv_format}: zero-scale final residual is not exact")

    return {
        "ordinal": cell["ordinal"],
        "kv_format": kv_format,
        "directory": cell["directory"],
        "continuation_token": continuation_token,
        "sources": {
            "baseline": _relative_artifact(root, baseline_path),
            "capture": _relative_artifact(root, capture_path),
            "replay": _relative_artifact(root, replay_path),
            "snapshot": _relative_artifact(root, snapshot_path),
            "snapshot_manifest": _relative_artifact(root, snapshot_manifest_path),
            "replay_input": _relative_artifact(root, replay_input_path),
            "replay_final": _relative_artifact(root, replay_final_path),
            "reference_residual": _relative_artifact(root, reference_path),
        },
        "runs": completed_runs,
    }


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    manifest = prepare_plan(args)
    root = args.experiment_dir.resolve()
    completed_cells = [_run_cell(args, root, cell) for cell in manifest["cells"]]
    manifest["cells"] = completed_cells
    manifest["complete"] = True
    write_json(root / "matrix-manifest.json", manifest)
    report = analyze_matrix(root, args.qxqxf, args.model, args.revision, args.tokens)
    write_json(root / "report.json", report)
    return report


def analyze_matrix(
    root: Path,
    qxqxf: Path,
    model: Path,
    revision: str,
    tokens: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    manifest = load_json_object(root / "matrix-manifest.json", "matrix manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("complete") is not True:
        raise ValueError("matrix manifest is incomplete")
    if set(manifest) != {"schema", "revision", "experiment", "artifacts", "cells", "complete"}:
        raise ValueError("matrix manifest keys are invalid")
    if manifest.get("revision") != revision or not REVISION_RE.fullmatch(revision):
        raise ValueError("matrix revision mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("matrix artifacts are missing")
    expected_artifact_keys = {"qxqxf", "model"} | ({"tokens"} if tokens is not None else set())
    if set(artifacts) != expected_artifact_keys:
        raise ValueError("matrix artifact keys are invalid")
    for key, trusted in (("qxqxf", qxqxf), ("model", model)):
        entry = artifacts.get(key)
        if not isinstance(entry, dict) or Path(str(entry.get("path"))).resolve() != trusted.resolve():
            raise ValueError(f"trusted {key} path mismatch")
        if entry.get("sha256") != sha256_file(trusted):
            raise ValueError(f"trusted {key} hash mismatch")
    token_entry = artifacts.get("tokens")
    if token_entry is not None:
        if tokens is None or not isinstance(token_entry, dict):
            raise ValueError("trusted tokens artifact is required")
        if Path(str(token_entry.get("path"))).resolve() != tokens.resolve():
            raise ValueError("trusted tokens path mismatch")
        if token_entry.get("sha256") != sha256_file(tokens):
            raise ValueError("trusted tokens hash mismatch")
    elif tokens is not None:
        raise ValueError("unexpected trusted tokens artifact")
    experiment = manifest.get("experiment")
    cells = manifest.get("cells")
    if not isinstance(experiment, dict) or not isinstance(cells, list):
        raise ValueError("matrix structure is invalid")
    expected_experiment_keys = {
        "prompt_token",
        "prefix_steps",
        "continuation_steps",
        "kv_formats",
        "scales",
        "activation_format",
        "layers",
        "start_layer",
        "expected_count",
        "direction_index",
        "direction_magnitude",
        "runtime_mode",
        "ctx",
        "seed",
    }
    if set(experiment) != expected_experiment_keys or experiment.get("runtime_mode") != "full_moe":
        raise ValueError("matrix experiment keys or runtime mode are invalid")
    integer_fields = (
        "prompt_token",
        "prefix_steps",
        "continuation_steps",
        "layers",
        "start_layer",
        "expected_count",
        "direction_index",
        "ctx",
        "seed",
    )
    if any(type(experiment.get(field)) is not int for field in integer_fields):
        raise ValueError("matrix experiment integer field is invalid")
    if experiment["prefix_steps"] < 2 or experiment["continuation_steps"] != 1:
        raise ValueError("matrix multi-token geometry is invalid")
    if not math.isfinite(experiment["direction_magnitude"]) or experiment["direction_magnitude"] == 0.0:
        raise ValueError("matrix direction magnitude is invalid")
    expected_formats = experiment.get("kv_formats")
    scales = experiment.get("scales")
    if expected_formats != ["f16", "int8"] or not isinstance(scales, list):
        raise ValueError("matrix format order mismatch")
    validated_scales = parse_scales(",".join(str(scale) for scale in scales))
    if scales != validated_scales:
        raise ValueError("matrix scales are invalid")
    if [cell.get("kv_format") if isinstance(cell, dict) else None for cell in cells] != expected_formats:
        raise ValueError("matrix cells are partial or disordered")
    expected_count = experiment.get("expected_count")
    start_layer = experiment.get("start_layer")
    layers = experiment.get("layers")
    if any(type(value) is not int for value in (expected_count, start_layer, layers)):
        raise ValueError("matrix geometry is invalid")

    report_cells: list[dict[str, Any]] = []
    for ordinal, cell in enumerate(cells):
        if not isinstance(cell, dict) or not json_int_equals(cell.get("ordinal"), ordinal):
            raise ValueError("matrix cell ordinal mismatch")
        if set(cell) != {"ordinal", "kv_format", "directory", "continuation_token", "sources", "runs"}:
            raise ValueError("matrix cell keys are invalid")
        kv_format = str(cell["kv_format"])
        cell_dir = f"cells/kv-{kv_format}"
        if cell.get("directory") != cell_dir:
            raise ValueError("matrix cell directory mismatch")
        sources = cell.get("sources")
        if not isinstance(sources, dict):
            raise ValueError("matrix cell sources are missing")
        if set(sources) != {
            "baseline",
            "capture",
            "replay",
            "snapshot",
            "snapshot_manifest",
            "replay_input",
            "replay_final",
            "reference_residual",
        }:
            raise ValueError("matrix source keys are invalid")
        expected_sources = {
            "baseline": f"{cell_dir}/baseline/result.json",
            "capture": f"{cell_dir}/capture/result.json",
            "replay": f"{cell_dir}/replay/result.json",
            "snapshot": f"{cell_dir}/snapshot.bin",
            "snapshot_manifest": f"{cell_dir}/snapshot-manifest.json",
            "reference_residual": f"{cell_dir}/reference-residual.f32",
        }
        source_paths = {
            key: _verify_relative_artifact(root, sources.get(key), relative, f"{kv_format}/{key}")
            for key, relative in expected_sources.items()
        }
        for key in ("replay_input", "replay_final"):
            entry = sources.get(key)
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError(f"invalid artifact record: {kv_format}/{key}")
            source_paths[key] = _verify_relative_artifact(
                root, entry, entry["path"], f"{kv_format}/{key}"
            )
            if not entry["path"].startswith(f"{cell_dir}/replay/"):
                raise ValueError(f"artifact path mismatch: {kv_format}/{key}")
        snapshot_manifest = load_json_object(source_paths["snapshot_manifest"], "snapshot manifest")
        geometry = snapshot_manifest.get("geometry")
        producer = snapshot_manifest.get("producer")
        if not isinstance(geometry, dict) or not isinstance(producer, dict):
            raise ValueError("snapshot provenance is invalid")
        required_geometry = {
            "layers": layers,
            "ctx_tokens": experiment.get("ctx"),
            "positions": experiment.get("prefix_steps"),
            "kv_format": kv_format,
            "activation_format": experiment.get("activation_format"),
            "next_token": cell.get("continuation_token"),
        }
        for key, expected in required_geometry.items():
            if geometry.get(key) != expected or (
                isinstance(expected, int) and type(geometry.get(key)) is not int
            ):
                raise ValueError(f"snapshot geometry mismatch: {kv_format}/{key}")
        if producer.get("revision") != revision:
            raise ValueError(f"snapshot revision mismatch: {kv_format}")
        _validate_snapshot_with_helper(
            source_paths["snapshot_manifest"],
            model=model,
            binary=qxqxf,
            geometry=geometry,
        )
        payload = snapshot_manifest.get("payload")
        if not isinstance(payload, dict) or payload.get("sha256") != sha256_file(source_paths["snapshot"]):
            raise ValueError(f"snapshot payload hash mismatch: {kv_format}")

        baseline = load_json_object(source_paths["baseline"], "baseline result")
        capture = load_json_object(source_paths["capture"], "capture result")
        replay = load_json_object(source_paths["replay"], "replay result")
        continuation_token = cell.get("continuation_token")
        if type(continuation_token) is not int:
            raise ValueError("continuation token is invalid")
        runtime_common = {
            "kv_format": kv_format,
            "activation_format": experiment.get("activation_format"),
            "layers": layers,
        }
        _validate_runtime_result(
            baseline,
            **runtime_common,
            prompt_token=experiment.get("prompt_token"),
            steps=experiment.get("prefix_steps") + 1,
            position_base=0,
        )
        _validate_runtime_result(
            capture,
            **runtime_common,
            prompt_token=experiment.get("prompt_token"),
            steps=experiment.get("prefix_steps"),
            position_base=0,
        )
        _validate_runtime_result(
            replay,
            **runtime_common,
            prompt_token=continuation_token,
            steps=1,
            position_base=experiment.get("prefix_steps"),
        )
        if capture["tokens"] != baseline["tokens"][: experiment["prefix_steps"]]:
            raise ValueError(f"{kv_format}: baseline/capture prefix is not exact")
        if replay["tokens"][0] != baseline["tokens"][experiment["prefix_steps"]]:
            raise ValueError(f"{kv_format}: uninterrupted/replay continuation is not exact")
        replay_token = replay.get("tokens", [None])[0]
        if not isinstance(replay_token, dict):
            raise ValueError("replay token is invalid")
        replay_final = read_exact_f32(source_paths["replay_final"], expected_count)
        replay_input = read_exact_f32(source_paths["replay_input"], expected_count)
        reference = read_exact_f32(source_paths["reference_residual"], expected_count)
        expected_reference = list(replay_input)
        direction_index = experiment["direction_index"]
        if direction_index < 0 or direction_index >= expected_count:
            raise ValueError("matrix direction index is invalid")
        expected_reference[direction_index] = f32(
            expected_reference[direction_index] + experiment["direction_magnitude"]
        )
        if reference != tuple(expected_reference):
            raise ValueError("reference residual does not match the declared direction")
        runs = cell.get("runs")
        if not isinstance(runs, list) or [run.get("scale") if isinstance(run, dict) else None for run in runs] != scales:
            raise ValueError("matrix runs are partial or disordered")
        zero_run: dict[str, Any] | None = None
        run_reports: list[dict[str, Any]] = []
        for run_ordinal, run in enumerate(runs):
            if not isinstance(run, dict) or not json_int_equals(run.get("ordinal"), run_ordinal):
                raise ValueError("matrix run ordinal mismatch")
            if set(run) != {"ordinal", "scale", "directory", "residual", "result", "final_residual"}:
                raise ValueError("matrix run keys are invalid")
            run_dir = f"{cell_dir}/runs/scale-{run_ordinal:03d}"
            if run.get("directory") != run_dir:
                raise ValueError("matrix run directory mismatch")
            residual_path = _verify_relative_artifact(root, run.get("residual"), f"{run_dir}/residual.f32", "run residual")
            result_path = _verify_relative_artifact(root, run.get("result"), f"{run_dir}/result.json", "run result")
            final_entry = run.get("final_residual")
            if not isinstance(final_entry, dict) or not isinstance(final_entry.get("path"), str):
                raise ValueError("run final residual record is invalid")
            final_path = _verify_relative_artifact(root, final_entry, final_entry["path"], "run final residual")
            if not final_entry["path"].startswith(f"{run_dir}/"):
                raise ValueError("run final residual path mismatch")
            residual = read_exact_f32(residual_path, expected_count)
            final = read_exact_f32(final_path, expected_count)
            result = load_json_object(result_path, "run result")
            _validate_runtime_result(
                result,
                **runtime_common,
                prompt_token=continuation_token,
                steps=1,
                position_base=experiment.get("prefix_steps"),
                start_layer=start_layer,
            )
            expected_residual = scaled_residual(replay_input, reference, float(run["scale"]))
            if residual != expected_residual:
                raise ValueError("run residual does not match the declared F32 formula")
            token = result.get("tokens", [None])[0]
            if not isinstance(token, dict):
                raise ValueError("run token is invalid")
            run_report = {
                "ordinal": run_ordinal,
                "scale": run["scale"],
                "selected_token": result.get("final_token"),
                "input_delta": vector_metrics(residual, replay_input),
                "final_delta": vector_metrics(final, replay_final),
                "routing_changed": _selected_experts(token, start_layer)
                != _selected_experts(replay_token, start_layer),
            }
            run_reports.append(run_report)
            if float(run["scale"]) == 0.0:
                zero_run = run_report
        if zero_run is None or zero_run["selected_token"] != replay.get("final_token"):
            raise ValueError(f"{kv_format}: zero-scale token control failed")
        if zero_run["input_delta"]["l2"] != 0.0 or zero_run["final_delta"]["l2"] != 0.0:
            raise ValueError(f"{kv_format}: zero-scale numeric control failed")
        if zero_run["routing_changed"] is not False:
            raise ValueError(f"{kv_format}: zero-scale routing control failed")
        report_cells.append(
            {
                "ordinal": ordinal,
                "kv_format": kv_format,
                "continuation_token": cell.get("continuation_token"),
                "control_exact": True,
                "runs": run_reports,
            }
        )
    left_cell, right_cell = report_cells
    cross_mode_runs = []
    for left_run, right_run in zip(left_cell["runs"], right_cell["runs"], strict=True):
        if left_run["ordinal"] != right_run["ordinal"] or left_run["scale"] != right_run["scale"]:
            raise ValueError("cross-mode run order mismatch")
        cross_mode_runs.append(
            {
                "ordinal": left_run["ordinal"],
                "scale": left_run["scale"],
                "selected_token_parity": left_run["selected_token"] == right_run["selected_token"],
                "routing_change_agreement": left_run["routing_changed"]
                == right_run["routing_changed"],
                "final_l2": {
                    "f16": left_run["final_delta"]["l2"],
                    "int8": right_run["final_delta"]["l2"],
                },
            }
        )
    return {
        "schema": REPORT_SCHEMA,
        "revision": revision,
        "matrix_complete": True,
        "experiment": dict(experiment),
        "provenance": {
            key: {"sha256": value["sha256"]}
            for key, value in artifacts.items()
        },
        "claims": {
            "replay_control": "exact",
            "perturbation": "mechanistic_association_only",
        },
        "cells": report_cells,
        "cross_mode": {
            "formats": ["f16", "int8"],
            "runs": cross_mode_runs,
        },
    }


def add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--qxqxf", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prompt-token", type=int, required=True)
    parser.add_argument("--prefix-steps", type=int, required=True)
    parser.add_argument("--continuation-steps", type=int, required=True)
    parser.add_argument("--kv-formats", required=True)
    parser.add_argument("--scales", required=True)
    parser.add_argument("--activation-format", required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--start-layer", type=int, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--direction-index", type=int, required=True)
    parser.add_argument("--direction-magnitude", type=float, required=True)

    parser.add_argument("--ctx", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="write a validated matrix plan")
    add_plan_arguments(plan)
    run = subparsers.add_parser("run", help="execute and analyze a complete matrix")
    add_plan_arguments(run)
    analyze = subparsers.add_parser("analyze", help="reanalyze a completed matrix")
    analyze.add_argument("--experiment-dir", type=Path, required=True)
    analyze.add_argument("--qxqxf", type=Path, required=True)
    analyze.add_argument("--model", type=Path, required=True)
    analyze.add_argument("--tokens", type=Path)
    analyze.add_argument("--revision", required=True)
    return parser.parse_args(argv)


def render(value: object) -> None:
    sys.stdout.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main() -> int:
    args = parse_args()
    try:
        if args.command == "plan":
            render(prepare_plan(args))
        elif args.command == "run":
            render(run_matrix(args))
        else:
            render(
                analyze_matrix(
                    args.experiment_dir,
                    args.qxqxf,
                    args.model,
                    args.revision,
                    args.tokens,
                )
            )
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
