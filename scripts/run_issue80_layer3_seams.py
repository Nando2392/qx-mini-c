#!/usr/bin/env python3
"""Run and analyze the Issue 80 fixed-KV layer-3 attention/MoE seam experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

MODES = ("f32", "q8_k_compat")
INTEGRATED_PHASES = ("input", "v-cur", "kqv-out", "ffn-inp", "ffn-moe-out", "output")
MOE_STAGES = (
    "ffn_norm", "ffn_moe_logits", "ffn_moe_probs", "ffn_moe_topk",
    "ffn_moe_weights", "ffn_moe_weights_sum", "ffn_moe_weights_norm",
    "ffn_moe_gate", "ffn_moe_up", "ffn_moe_swiglu", "ffn_moe_down",
    "ffn_moe_weighted",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _read_f32(path: Path, count: int | None = None) -> tuple[float, ...]:
    raw = path.read_bytes()
    if not raw or len(raw) % 4:
        raise ValueError(f"F32 sidecar count mismatch: {path}")
    values = struct.unpack(f"<{len(raw) // 4}f", raw)
    if count is not None and len(values) != count:
        raise ValueError(f"F32 sidecar count mismatch: {path}")
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"non-finite F32 sidecar: {path}")
    return values


def _metrics(left: Iterable[float], right: Iterable[float]) -> dict[str, Any]:
    a, b = tuple(left), tuple(right)
    if len(a) != len(b):
        raise ValueError("comparison count mismatch")
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    dot = math.fsum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(math.fsum(float(x) * float(x) for x in a))
    nb = math.sqrt(math.fsum(float(y) * float(y) for y in b))
    cosine = 1.0 if a == b else (dot / (na * nb) if na and nb else 0.0)
    return {
        "count": len(a), "max_abs": max(map(abs, diffs), default=0.0),
        "rmse": math.sqrt(math.fsum(d * d for d in diffs) / len(diffs)) if diffs else 0.0,
        "cosine": cosine, "byte_exact": struct.pack(f"<{len(a)}f", *a) == struct.pack(f"<{len(b)}f", *b),
    }


def _add_f32(left: Iterable[float], right: Iterable[float]) -> tuple[float, ...]:
    a, b = tuple(left), tuple(right)
    if len(a) != len(b):
        raise ValueError("F32 residual addition shape mismatch")
    return tuple(_f32(_f32(x) + _f32(y)) for x, y in zip(a, b))


def _sub_f32(left: Iterable[float], right: Iterable[float]) -> tuple[float, ...]:
    a, b = tuple(left), tuple(right)
    if len(a) != len(b):
        raise ValueError("F32 subtraction shape mismatch")
    return tuple(_f32(_f32(x) - _f32(y)) for x, y in zip(a, b))


def _ordered_weighted_sum(weighted: tuple[float, ...], hidden: int, ranks: int) -> tuple[float, ...]:
    if len(weighted) != hidden * ranks:
        raise ValueError("ffn_moe_weighted shape mismatch")
    output = []
    for column in range(hidden):
        value = _f32(0.0)
        for rank in range(ranks):
            value = _f32(value + weighted[rank * hidden + column])
        output.append(value)
    return tuple(output)


def _json_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _routing(value: Any, message: str, *, experts: int = 128) -> list[int]:
    if (
        not isinstance(value, list) or len(value) != 8
        or any(type(item) is not int or not 0 <= item < experts for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(message)
    return value


def _routing_weights(value: Any, message: str, *, sum_tolerance: float) -> list[float]:
    if (
        not isinstance(value, list) or len(value) != 8
        or any(type(item) not in (int, float) or not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in value)
        or not math.isclose(math.fsum(value), 1.0, rel_tol=0.0, abs_tol=sum_tolerance)
    ):
        raise ValueError(message)
    return [float(item) for item in value]


def _exact_int(value: Any, expected: int | None = None) -> bool:
    return type(value) is int and (expected is None or value == expected)


def _load_integrated(directory: Path, mode: str, hidden: int, vcur_count: int, kqv_count: int,
                     layers: int, continuation_token: int) -> dict[str, Any]:
    payload = _json_object(directory / "result.json")
    replay = payload.get("residual_replay")
    if not (
        payload.get("probe") == "state_loop" and _exact_int(payload.get("prompt_token"), continuation_token)
        and _exact_int(payload.get("steps"), 1) and _exact_int(payload.get("layers"), layers)
        and _exact_int(payload.get("layers_run"), layers - 3) and _exact_int(payload.get("start_layer"), 3)
        and _exact_int(payload.get("position_base"), 1) and payload.get("kv_format") == "int8"
        and payload.get("activation_format") == mode and payload.get("residual_source") == "injected_f32_replay"
        and isinstance(replay, dict) and replay.get("enabled") is True
        and replay.get("source") == "f32_sidecar" and _exact_int(replay.get("values"), hidden)
    ):
        raise ValueError(f"{mode}: integrated metadata invalid")
    tokens = payload.get("tokens")
    if not isinstance(tokens, list) or len(tokens) != 1 or not isinstance(tokens[0], dict) or not _exact_int(tokens[0].get("step"), 1) or not _exact_int(tokens[0].get("position"), 1) or not _exact_int(tokens[0].get("input_token"), continuation_token):
        raise ValueError(f"{mode}: integrated metadata invalid")
    layer_rows = tokens[0].get("layers")
    if not isinstance(layer_rows, list) or len(layer_rows) != layers - 3 or any(not isinstance(row, dict) for row in layer_rows) or any(not _exact_int(row.get("layer"), layer) for row, layer in zip(layer_rows, range(3, layers))):
        raise ValueError("integrated layers are duplicate or disordered")
    row = layer_rows[0]
    if not _exact_int(row.get("attention_context_tokens"), 2):
        raise ValueError("attention context token count must be 2")
    if row.get("full_moe") is not True:
        raise ValueError(f"{mode}: integrated metadata invalid")
    selected = _routing(row.get("selected_experts"), "integrated routing must contain ordered integer expert IDs")
    routing_weights = _routing_weights(row.get("routing_weights"), "integrated routing weights invalid", sum_tolerance=1e-12)
    expected = {f"step-1-layer-3-{phase}.f32" for phase in INTEGRATED_PHASES}
    actual = {path.name for path in directory.glob("step-1-layer-3-*.f32")}
    if actual != expected:
        raise ValueError("unexpected layer-3 sidecars")
    counts = {"input": hidden, "v-cur": vcur_count, "kqv-out": kqv_count, "ffn-inp": hidden, "ffn-moe-out": hidden, "output": hidden}
    values = {phase: _read_f32(directory / f"step-1-layer-3-{phase}.f32", counts[phase]) for phase in INTEGRATED_PHASES}
    return {"payload": payload, "selected": selected, "routing_weights": routing_weights, "values": values}


def _load_moe(directory: Path, mode: str, hidden: int) -> dict[str, Any]:
    payload = _json_object(directory / "result.json")
    if (
        payload.get("probe") != "moe_stage"
        or not _exact_int(payload.get("layer"), 3)
        or not _exact_int(payload.get("input_count"), hidden)
        or payload.get("activation_mode") != mode
        or not _exact_int(payload.get("sidecars_written"), 12)
        or payload.get("router_precision") != "integrated_double"
        or payload.get("routing_sidecar_element_type") != "f32"
        or payload.get("weighted_contribution_element_type") != "f32"
        or payload.get("weighted_contribution_formula") != "f32(integrated_double_weight*expert_f32)"
    ):
        raise ValueError(f"{mode}: standalone MoE metadata invalid")
    experts_used = payload.get("experts_used")
    experts = payload.get("experts")
    intermediate = payload.get("intermediate")
    if not _exact_int(experts_used, 8) or type(experts) is not int or not 8 <= experts <= 128 or type(intermediate) is not int or intermediate <= 0:
        raise ValueError(f"{mode}: standalone MoE metadata invalid")
    selected = _routing(payload.get("selected_experts"), "standalone routing must contain ordered integer expert IDs", experts=experts)
    routing_weights = _routing_weights(payload.get("routing_weights"), "standalone routing weights invalid", sum_tolerance=5e-7)
    expected = {f"{stage}-3.f32" for stage in MOE_STAGES}
    actual = {path.name for path in directory.glob("*-3.f32")}
    if actual != expected:
        raise ValueError("unexpected standalone MoE sidecars")
    counts = {
        "ffn_norm": hidden, "ffn_moe_logits": experts, "ffn_moe_probs": experts,
        "ffn_moe_topk": experts_used, "ffn_moe_weights": experts_used,
        "ffn_moe_weights_sum": 1, "ffn_moe_weights_norm": experts_used,
        "ffn_moe_gate": intermediate * experts_used, "ffn_moe_up": intermediate * experts_used,
        "ffn_moe_swiglu": intermediate * experts_used, "ffn_moe_down": hidden * experts_used,
        "ffn_moe_weighted": hidden * experts_used,
    }
    values = {stage: _read_f32(directory / f"{stage}-3.f32", counts[stage]) for stage in MOE_STAGES}
    topk = values["ffn_moe_topk"]
    if any(not value.is_integer() for value in topk) or [int(value) for value in topk] != selected:
        raise ValueError("standalone top-k sidecar contradicts routing metadata")
    normalized = values["ffn_moe_weights_norm"]
    if any(_f32(metadata) != sidecar for metadata, sidecar in zip(routing_weights, normalized)):
        raise ValueError("standalone routing weights contradict normalized-weight sidecar")
    if not math.isclose(math.fsum(normalized), 1.0, rel_tol=0.0, abs_tol=5e-7):
        raise ValueError("standalone routing weights do not sum to one")
    output = _ordered_weighted_sum(values["ffn_moe_weighted"], hidden, experts_used)
    return {"payload": payload, "selected": selected, "values": values, "output": output}


def analyze_artifacts(*, work: Path, snapshot: Path, residual: Path, executable: Path, model: Path,
                      expected_snapshot_sha256: str, expected_residual_sha256: str,
                      hidden: int, vcur_count: int, kqv_count: int, layers: int,
                      continuation_token: int) -> dict[str, Any]:
    work, snapshot, residual, executable, model = map(Path, (work, snapshot, residual, executable, model))
    if any(type(value) is not int or value <= 0 for value in (hidden, vcur_count, kqv_count)) or type(layers) is not int or layers <= 3 or type(continuation_token) is not int or continuation_token < 0:
        raise ValueError("invalid experiment dimensions")
    snapshot_digest, residual_digest = sha256(snapshot), sha256(residual)
    if snapshot_digest != expected_snapshot_sha256:
        raise ValueError("snapshot SHA-256 mismatch")
    if residual_digest != expected_residual_sha256:
        raise ValueError("residual SHA-256 mismatch")
    residual_values = _read_f32(residual, hidden)
    integrated = {mode: _load_integrated(work / f"integrated-{mode}", mode, hidden, vcur_count, kqv_count, layers, continuation_token) for mode in MODES}
    fixed = {mode: _load_moe(work / f"moe-fixed-f32-input-{mode}", mode, hidden) for mode in MODES}
    controls = {mode: _load_moe(work / f"moe-control-{mode}", mode, hidden) for mode in MODES}
    for mode in MODES:
        if integrated[mode]["values"]["input"] != residual_values:
            raise ValueError(f"{mode}: integrated input does not match bound residual")
        if controls[mode]["selected"] != integrated[mode]["selected"]:
            raise ValueError(f"{mode}: integrated/standalone routing mismatch")
        if struct.pack(f"<{hidden}f", *controls[mode]["output"]) != struct.pack(f"<{hidden}f", *integrated[mode]["values"]["ffn-moe-out"]):
            mismatch = _metrics(controls[mode]["output"], integrated[mode]["values"]["ffn-moe-out"])
            raise ValueError(f"{mode}: integrated/standalone MoE output is not byte-exact (max_abs={mismatch['max_abs']:.9g}, rmse={mismatch['rmse']:.9g})")
        control_layer = _add_f32(integrated[mode]["values"]["ffn-inp"], controls[mode]["output"])
        if struct.pack(f"<{hidden}f", *control_layer) != struct.pack(f"<{hidden}f", *integrated[mode]["values"]["output"]):
            mismatch = _metrics(control_layer, integrated[mode]["values"]["output"])
            raise ValueError(f"{mode}: integrated/standalone layer output is not byte-exact (max_abs={mismatch['max_abs']:.9g}, rmse={mismatch['rmse']:.9g})")
    a, b = integrated["f32"]["values"], integrated["q8_k_compat"]["values"]
    fmoe, qmoe = fixed["f32"], fixed["q8_k_compat"]
    return {
        "schema": "qx-issue80-layer3-seams-v1",
        "provenance": {"snapshot_sha256": snapshot_digest, "residual_sha256": residual_digest, "executable_sha256": sha256(executable), "model_sha256": sha256(model), "continuation_token": continuation_token},
        "attention": {
            "input": _metrics(a["input"], b["input"]),
            "vcur": _metrics(a["v-cur"], b["v-cur"]),
            "kqv_output": _metrics(a["kqv-out"], b["kqv-out"]),
            "post_attention_contribution_derived": _metrics(_sub_f32(a["ffn-inp"], a["input"]), _sub_f32(b["ffn-inp"], b["input"])),
            "ffn_input": _metrics(a["ffn-inp"], b["ffn-inp"]),
        },
        "routing": {"integrated": {mode: integrated[mode]["selected"] for mode in MODES}, "same_f32_input": {mode: fixed[mode]["selected"] for mode in MODES}},
        "moe_same_f32_input": {
            "stages": {stage: _metrics(fmoe["values"][stage], qmoe["values"][stage]) for stage in MOE_STAGES},
            "moe_output": _metrics(fmoe["output"], qmoe["output"]),
            "layer_output": _metrics(_add_f32(a["ffn-inp"], fmoe["output"]), _add_f32(a["ffn-inp"], qmoe["output"])),
        },
        "integrated_outputs": {
            "moe_output": _metrics(a["ffn-moe-out"], b["ffn-moe-out"]),
            "layer_output": _metrics(a["output"], b["output"]),
        },
        "bridge_controls": {
            mode: {"moe_output_byte_exact": True, "layer_output_byte_exact": True} for mode in MODES
        } | {"summation_semantics": "ordered_rank_f32_accumulation"},
    }


def _invoke(command: list[str], result: Path) -> None:
    completed = subprocess.run(command, text=True, capture_output=True)
    result.parent.mkdir(parents=True, exist_ok=True)
    (result.parent / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr}")
    try:
        payload = json.loads(completed.stdout, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid command JSON output") from error
    if not isinstance(payload, dict):
        raise ValueError("invalid command JSON output")
    result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    for field, minimum, maximum in (
        ("hidden", 1, 0xFFFFFFFF), ("vcur_count", 1, 0xFFFFFFFF),
        ("kqv_count", 1, 0xFFFFFFFF), ("layers", 4, 48),
        ("ctx", 2, 0xFFFFFFFF), ("seed", 0, 0xFFFFFFFF),
        ("continuation_token", 0, 0xFFFFFFFF),
    ):
        value = getattr(args, field)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{field} must be an integer in range {minimum}..{maximum}")
    out = Path(args.out)
    if out.exists():
        raise ValueError("output directory already exists")
    inputs = {name: sha256(Path(path)) for name, path in (("snapshot", args.snapshot), ("residual", args.residual), ("executable", args.qxqxf), ("model", args.model))}
    if inputs["snapshot"] != args.snapshot_sha256:
        raise ValueError("snapshot SHA-256 mismatch")
    if inputs["residual"] != args.residual_sha256:
        raise ValueError("residual SHA-256 mismatch")
    out.mkdir(parents=True, exist_ok=False)
    commands: list[list[str]] = []

    common = [str(args.qxqxf), "state-loop-probe", "--in", str(args.model), "--prompt-token", str(args.continuation_token), "--steps", "1", "--layers", str(args.layers), "--ctx", str(args.ctx), "--kv", "int8", "--temperature", "0", "--seed", str(args.seed), "--full-moe", "--kv-snapshot-in", str(args.snapshot), "--start-layer", "3", "--residual-in", str(args.residual)]
    for mode in MODES:
        directory = out / f"integrated-{mode}"
        directory.mkdir()
        command = common + ["--activation", mode, "--dump-residuals", str(directory)]
        commands.append(command); _invoke(command, directory / "result.json")
    f32_input = out / "integrated-f32" / "step-1-layer-3-ffn-inp.f32"
    ffn_inputs = {mode: out / f"integrated-{mode}" / "step-1-layer-3-ffn-inp.f32" for mode in MODES}
    expected_ffn_digests = {mode: sha256(path) for mode, path in ffn_inputs.items()}
    moe_input_provenance: list[dict[str, Any]] = []
    for mode in MODES:
        for label, ffn_input, expected_digest in (("fixed-f32-input", f32_input, expected_ffn_digests["f32"]), ("control", ffn_inputs[mode], expected_ffn_digests[mode])):
            directory = out / f"moe-{label}-{mode}"
            directory.mkdir()
            command = [str(args.qxqxf), "moe-stage-probe", "--in", str(args.model), "--layer", "3", "--ffn-inp", str(ffn_input), "--out-dir", str(directory), "--activation", mode, "--router-precision", "integrated_double"]
            before_digest = sha256(ffn_input)
            if before_digest != expected_digest:
                raise ValueError(f"{label}-{mode}: MoE FFN input hash changed before consumer")
            commands.append(command); _invoke(command, directory / "result.json")
            after_digest = sha256(ffn_input)
            if after_digest != expected_digest:
                raise ValueError(f"{label}-{mode}: MoE FFN input hash changed after consumer")
            moe_input_provenance.append({"consumer": f"moe-{label}-{mode}", "path": str(ffn_input), "sha256_before": before_digest, "sha256_after": after_digest})
    after = {name: sha256(Path(path)) for name, path in (("snapshot", args.snapshot), ("residual", args.residual), ("executable", args.qxqxf), ("model", args.model))}
    if after != inputs:
        raise ValueError("bound input hash changed during execution")
    report = analyze_artifacts(work=out, snapshot=Path(args.snapshot), residual=Path(args.residual), executable=Path(args.qxqxf), model=Path(args.model), expected_snapshot_sha256=args.snapshot_sha256, expected_residual_sha256=args.residual_sha256, hidden=args.hidden, vcur_count=args.vcur_count, kqv_count=args.kqv_count, layers=args.layers, continuation_token=args.continuation_token)
    report["provenance"]["moe_ffn_inputs"] = moe_input_provenance
    report["execution"] = {"input_sha256_before": inputs, "input_sha256_after": after, "commands": commands}
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qxqxf", required=True, type=Path); parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path); parser.add_argument("--residual", required=True, type=Path)
    parser.add_argument("--snapshot-sha256", required=True); parser.add_argument("--residual-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path); parser.add_argument("--hidden", required=True, type=int)
    parser.add_argument("--vcur-count", required=True, type=int); parser.add_argument("--kqv-count", required=True, type=int)
    parser.add_argument("--continuation-token", required=True, type=int)
    parser.add_argument("--layers", type=int, default=48); parser.add_argument("--ctx", type=int, default=4); parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
