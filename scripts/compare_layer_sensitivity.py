#!/usr/bin/env python
"""Compare layer perturbations or a same-input attention and MoE chain."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path


PROBE_STAGES = (
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
    actual_l2_sq = sum(float(value) * float(value) for value in actual)
    expected_l2_sq = sum(float(value) * float(value) for value in expected)
    delta_l2_sq = sum(delta * delta for delta in deltas)
    denominator = math.sqrt(actual_l2_sq * expected_l2_sq)
    dot = sum(float(got) * float(want) for got, want in zip(actual, expected))
    result = {
        "count": len(actual),
        "max_abs": abs(deltas[index]),
        "rmse": math.sqrt(delta_l2_sq / len(deltas)),
        "cosine": dot / denominator if denominator else (1.0 if actual == expected else 0.0),
        "delta_l2": math.sqrt(delta_l2_sq),
        "max_abs_index": index,
    }
    if not all(math.isfinite(float(value)) for value in result.values()):
        raise ValueError("non-finite comparison metric")
    return result


def subtract(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    if len(left) != len(right):
        raise ValueError("sidecar length mismatch in subtraction")
    return tuple(float(a) - float(b) for a, b in zip(left, right))


def add(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    if len(left) != len(right):
        raise ValueError("sidecar length mismatch in addition")
    return tuple(float(a) + float(b) for a, b in zip(left, right))


def add_f32(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    if len(left) != len(right):
        raise ValueError("sidecar length mismatch in F32 addition")
    return tuple(
        struct.unpack("<f", struct.pack("<f", float(a) + float(b)))[0]
        for a, b in zip(left, right)
    )


def safe_gain(numerator: float, denominator: float, label: str) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(f"invalid zero or non-finite gain denominator: {label}")
    gain = numerator / denominator
    if not math.isfinite(gain):
        raise ValueError(f"non-finite gain: {label}")
    return gain


def validate_routing_weights(values: dict[str, tuple[float, ...]], selected: list[int]) -> None:
    raw = values["ffn_moe_weights"]
    normalized = values["ffn_moe_weights_norm"]
    reported_sum = values["ffn_moe_weights_sum"][0]
    actual_sum = math.fsum(raw)
    if actual_sum <= 0.0 or not math.isclose(actual_sum, reported_sum, rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError("weight sum contradicts raw weights")
    if any(
        not math.isclose(weight, values["ffn_moe_probs"][expert], rel_tol=1e-6, abs_tol=1e-7)
        for expert, weight in zip(selected, raw)
    ):
        raise ValueError("raw weights contradict selected router probabilities")
    if any(
        not math.isclose(weight, raw_weight / actual_sum, rel_tol=1e-6, abs_tol=1e-7)
        for weight, raw_weight in zip(normalized, raw)
    ) or not math.isclose(math.fsum(normalized), 1.0, rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError("normalized weights contradict raw weights")


def load_probe(directory: Path, layer: int, hidden: int) -> dict:
    values = {stage: read_f32(directory / f"{stage}-{layer}.f32") for stage in PROBE_STAGES}
    selected_raw = values["ffn_moe_topk"]
    selected = [int(value) for value in selected_raw]
    if any(float(expert) != value or expert < 0 for expert, value in zip(selected, selected_raw)):
        raise ValueError("top-k sidecar contains invalid expert id")
    expert_count = len(values["ffn_moe_logits"])
    experts_used = len(selected)
    if (
        expert_count == 0
        or experts_used == 0
        or len(set(selected)) != experts_used
        or any(expert >= expert_count for expert in selected)
    ):
        raise ValueError("invalid top-k expert selection")
    expected_topk = sorted(
        range(expert_count),
        key=lambda expert: (-values["ffn_moe_logits"][expert], expert),
    )[:experts_used]
    if selected != expected_topk:
        raise ValueError("top-k does not match router logits")
    if len(values["ffn_moe_probs"]) != expert_count:
        raise ValueError("router probability shape mismatch")
    if len(values["ffn_norm"]) != hidden:
        raise ValueError("ffn_norm shape mismatch")
    for stage in ("ffn_moe_weights", "ffn_moe_weights_norm"):
        if len(values[stage]) != experts_used:
            raise ValueError(f"{stage} shape mismatch")
    if len(values["ffn_moe_weights_sum"]) != 1:
        raise ValueError("weight sum shape mismatch")
    validate_routing_weights(values, selected)
    gate_count = len(values["ffn_moe_gate"])
    if gate_count % experts_used:
        raise ValueError("gate shape mismatch")
    intermediate = gate_count // experts_used
    if intermediate == 0:
        raise ValueError("empty expert intermediate shape")
    for stage in ("ffn_moe_up", "ffn_moe_swiglu"):
        if len(values[stage]) != intermediate * experts_used:
            raise ValueError(f"{stage} shape mismatch")
    for stage in ("ffn_moe_down", "ffn_moe_weighted"):
        if len(values[stage]) != hidden * experts_used:
            raise ValueError(f"{stage} shape mismatch")
    return {
        "values": values,
        "selected": selected,
        "expert_count": expert_count,
        "experts_used": experts_used,
        "intermediate": intermediate,
    }


def expert_slice(probe: dict, stage: str, expert: int, width: int) -> tuple[float, ...]:
    if expert not in probe["selected"]:
        return (0.0,) * width
    rank = probe["selected"].index(expert)
    start = rank * width
    return probe["values"][stage][start : start + width]


def sum_weighted(probe: dict, hidden: int) -> tuple[float, ...]:
    weighted = probe["values"]["ffn_moe_weighted"]
    return tuple(
        sum(float(weighted[rank * hidden + index]) for rank in range(probe["experts_used"]))
        for index in range(hidden)
    )


def compare_same_input(
    oracle_dir: Path,
    attention_dir: Path,
    moe_dir: Path,
    layer: int,
    expected_vcur_count: int,
    expected_kqv_out_count: int,
) -> dict:
    oracle_input = read_f32(oracle_dir / f"layer-{layer}.f32")
    hidden = len(oracle_input)
    oracle_attention = {
        stage: read_f32(oracle_dir / f"{stage}-{layer}.f32")
        for stage in ("attn_norm", "Vcur", "kqv_out", "ffn_inp")
    }
    qx_attention = {
        stage: read_f32(attention_dir / f"{stage}-{layer}.f32")
        for stage in ("attn_norm", "Vcur", "kqv_out", "attn_out", "ffn_inp")
    }
    for values in (oracle_attention["Vcur"], qx_attention["Vcur"]):
        if len(values) != expected_vcur_count:
            raise ValueError(f"Vcur count mismatch: {len(values)} != {expected_vcur_count}")
    for values in (oracle_attention["kqv_out"], qx_attention["kqv_out"]):
        if len(values) != expected_kqv_out_count:
            raise ValueError(
                f"kqv_out count mismatch: {len(values)} != {expected_kqv_out_count}"
            )
    for stage in ("attn_norm", "ffn_inp"):
        if len(oracle_attention[stage]) != hidden or len(qx_attention[stage]) != hidden:
            raise ValueError(f"{stage} shape mismatch")
    for stage in ("Vcur", "kqv_out"):
        if len(qx_attention[stage]) != len(oracle_attention[stage]):
            raise ValueError(f"{stage} shape mismatch")
    if len(qx_attention["attn_out"]) != hidden:
        raise ValueError("attn_out shape mismatch")
    if qx_attention["ffn_inp"] != add_f32(oracle_input, qx_attention["attn_out"]):
        raise ValueError("ffn input contradicts layer input plus attention output")

    oracle_probe = load_probe(oracle_dir, layer, hidden)
    qx_probe = load_probe(moe_dir, layer, hidden)
    if (
        qx_probe["expert_count"] != oracle_probe["expert_count"]
        or qx_probe["experts_used"] != oracle_probe["experts_used"]
        or qx_probe["intermediate"] != oracle_probe["intermediate"]
    ):
        raise ValueError("oracle and QX MoE shape mismatch")

    oracle_attention_output = subtract(oracle_attention["ffn_inp"], oracle_input)
    attention_stages = {
        "attn_norm": metrics(qx_attention["attn_norm"], oracle_attention["attn_norm"]),
        "Vcur": metrics(qx_attention["Vcur"], oracle_attention["Vcur"]),
        "kqv_out": metrics(qx_attention["kqv_out"], oracle_attention["kqv_out"]),
        "attention_output": metrics(qx_attention["attn_out"], oracle_attention_output),
        "ffn_input": metrics(qx_attention["ffn_inp"], oracle_attention["ffn_inp"]),
    }
    moe_stage_names = {
        "ffn_norm": "ffn_norm",
        "router_logits": "ffn_moe_logits",
        "router_probs": "ffn_moe_probs",
        "topk": "ffn_moe_topk",
        "weights": "ffn_moe_weights",
        "weight_sum": "ffn_moe_weights_sum",
        "weights_norm": "ffn_moe_weights_norm",
        "gate": "ffn_moe_gate",
        "up": "ffn_moe_up",
        "swiglu": "ffn_moe_swiglu",
        "down": "ffn_moe_down",
        "weighted": "ffn_moe_weighted",
    }
    moe_stages = {
        label: metrics(qx_probe["values"][stage], oracle_probe["values"][stage])
        for label, stage in moe_stage_names.items()
    }
    qx_moe_output = sum_weighted(qx_probe, hidden)
    oracle_moe_output = read_f32(oracle_dir / f"ffn_moe_out-{layer}.f32")
    oracle_layer_output = read_f32(oracle_dir / f"l_out-{layer}.f32")
    qx_layer_output = add(qx_attention["ffn_inp"], qx_moe_output)

    return {
        "schema": 1,
        "passed": True,
        "mode": "same_input",
        "layer": layer,
        "hidden": hidden,
        "intermediate": qx_probe["intermediate"],
        "routing": {
            "oracle": oracle_probe["selected"],
            "qx": qx_probe["selected"],
            "exact": qx_probe["selected"] == oracle_probe["selected"],
        },
        "checkpoints": {
            "attention": attention_stages,
            "moe": moe_stages,
        },
        "reconstruction": {
            "moe_output": metrics(qx_moe_output, oracle_moe_output),
            "layer_output": metrics(qx_layer_output, oracle_layer_output),
        },
    }


def compare(
    oracle_dir: Path,
    qx_dir: Path,
    nominal_moe_dir: Path,
    perturbed_moe_dir: Path,
    layer: int,
    step: int,
) -> dict:
    oracle_input = read_f32(oracle_dir / f"layer-{layer}.f32")
    oracle_ffn_input = read_f32(oracle_dir / f"ffn_inp-{layer}.f32")
    oracle_output = read_f32(oracle_dir / f"l_out-{layer}.f32")
    qx_input = read_f32(qx_dir / f"step-{step}-layer-{layer}-input.f32")
    qx_ffn_input = read_f32(qx_dir / f"step-{step}-layer-{layer}-ffn-inp.f32")
    qx_output = read_f32(qx_dir / f"step-{step}-layer-{layer}-output.f32")
    hidden = len(oracle_input)
    if hidden == 0 or any(len(values) != hidden for values in (oracle_ffn_input, oracle_output, qx_input, qx_ffn_input, qx_output)):
        raise ValueError("layer sidecar shape mismatch")

    nominal = load_probe(nominal_moe_dir, layer, hidden)
    perturbed = load_probe(perturbed_moe_dir, layer, hidden)
    if nominal["expert_count"] != perturbed["expert_count"]:
        raise ValueError("expert count mismatch")
    if nominal["experts_used"] != perturbed["experts_used"]:
        raise ValueError("top-k width mismatch")
    if nominal["intermediate"] != perturbed["intermediate"]:
        raise ValueError("expert intermediate shape mismatch")

    layer_input_metrics = metrics(qx_input, oracle_input)
    attention_nominal = subtract(oracle_ffn_input, oracle_input)
    attention_perturbed = subtract(qx_ffn_input, qx_input)
    attention_metrics = metrics(attention_perturbed, attention_nominal)
    ffn_input_metrics = metrics(qx_ffn_input, oracle_ffn_input)
    nominal_moe = sum_weighted(nominal, hidden)
    perturbed_moe = sum_weighted(perturbed, hidden)
    moe_metrics = metrics(perturbed_moe, nominal_moe)
    output_metrics = metrics(qx_output, oracle_output)

    nominal_reconstructed = add(oracle_ffn_input, nominal_moe)
    perturbed_reconstructed = add(qx_ffn_input, perturbed_moe)
    input_delta_l2 = float(layer_input_metrics["delta_l2"])
    ffn_delta_l2 = float(ffn_input_metrics["delta_l2"])
    gains = {
        "attention_output_from_layer_input": safe_gain(float(attention_metrics["delta_l2"]), input_delta_l2, "attention_output_from_layer_input"),
        "ffn_input_from_layer_input": safe_gain(ffn_delta_l2, input_delta_l2, "ffn_input_from_layer_input"),
        "moe_output_from_ffn_input": safe_gain(float(moe_metrics["delta_l2"]), ffn_delta_l2, "moe_output_from_ffn_input"),
        "layer_output_from_layer_input": safe_gain(float(output_metrics["delta_l2"]), input_delta_l2, "layer_output_from_layer_input"),
    }

    nominal_ids = nominal["selected"]
    perturbed_ids = perturbed["selected"]
    union = sorted(set(nominal_ids) | set(perturbed_ids))
    expert_reports = []
    for expert in union:
        item = {
            "expert_id": expert,
            "nominal_rank": nominal_ids.index(expert) if expert in nominal_ids else None,
            "perturbed_rank": perturbed_ids.index(expert) if expert in perturbed_ids else None,
        }
        for short_name, stage in EXPERT_STAGES:
            width = nominal["intermediate"] if short_name in ("gate", "up", "swiglu") else hidden
            item[short_name] = metrics(
                expert_slice(perturbed, stage, expert, width),
                expert_slice(nominal, stage, expert, width),
            )
        item["delta_l2"] = item["weighted"]["delta_l2"]
        expert_reports.append(item)
    dominant = max(expert_reports, key=lambda item: float(item["delta_l2"]))

    stage_path_deltas = {
        "ffn_norm": metrics(perturbed["values"]["ffn_norm"], nominal["values"]["ffn_norm"]),
        "router_logits": metrics(perturbed["values"]["ffn_moe_logits"], nominal["values"]["ffn_moe_logits"]),
        "router_probs": metrics(perturbed["values"]["ffn_moe_probs"], nominal["values"]["ffn_moe_probs"]),
        "moe_output": moe_metrics,
    }
    first_amplification = None
    if gains["attention_output_from_layer_input"] > 2.0:
        first_amplification = "attention_output_derived"
    elif gains["ffn_input_from_layer_input"] > 2.0:
        first_amplification = "ffn_input"
    elif gains["moe_output_from_ffn_input"] > 2.0:
        first_amplification = "moe_output"

    return {
        "schema": 1,
        "passed": True,
        "layer": layer,
        "step": step,
        "hidden": hidden,
        "intermediate": nominal["intermediate"],
        "routing": {
            "nominal": nominal_ids,
            "perturbed": perturbed_ids,
            "nominal_weights": list(nominal["values"]["ffn_moe_weights"]),
            "perturbed_weights": list(perturbed["values"]["ffn_moe_weights"]),
            "weight_deltas": {
                "raw": metrics(perturbed["values"]["ffn_moe_weights"], nominal["values"]["ffn_moe_weights"]),
                "normalized": metrics(
                    perturbed["values"]["ffn_moe_weights_norm"],
                    nominal["values"]["ffn_moe_weights_norm"],
                ),
                "sum": metrics(
                    perturbed["values"]["ffn_moe_weights_sum"],
                    nominal["values"]["ffn_moe_weights_sum"],
                ),
            },
            "common": [expert for expert in nominal_ids if expert in perturbed_ids],
            "dropped": [expert for expert in nominal_ids if expert not in perturbed_ids],
            "added": [expert for expert in perturbed_ids if expert not in nominal_ids],
        },
        "checkpoints": {
            "layer_input": layer_input_metrics,
            "attention_output_derived": attention_metrics,
            "ffn_input": ffn_input_metrics,
            "moe_output": moe_metrics,
            "layer_output": output_metrics,
        },
        "stage_path_deltas": stage_path_deltas,
        "gains": gains,
        "experts": expert_reports,
        "dominant_expert": {
            "expert_id": dominant["expert_id"],
            "delta_l2": dominant["delta_l2"],
            "moe_delta_l2_ratio": safe_gain(float(dominant["delta_l2"]), float(moe_metrics["delta_l2"]), "dominant_expert_ratio"),
        },
        "reconstruction": {
            "nominal_vs_oracle": metrics(nominal_reconstructed, oracle_output),
            "perturbed_vs_qx": metrics(perturbed_reconstructed, qx_output),
        },
        "first_amplification_checkpoint": first_amplification,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--qx-dir", type=Path)
    parser.add_argument("--nominal-moe-dir", type=Path)
    parser.add_argument("--perturbed-moe-dir", type=Path)
    parser.add_argument("--attention-dir", type=Path)
    parser.add_argument("--same-input-moe-dir", type=Path)
    parser.add_argument("--expected-vcur-count", type=int)
    parser.add_argument("--expected-kqv-out-count", type=int)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.layer < 0 or args.step < 0:
            raise ValueError("layer and step must be non-negative")
        same_input = any(
            value is not None
            for value in (
                args.attention_dir,
                args.same_input_moe_dir,
                args.expected_vcur_count,
                args.expected_kqv_out_count,
            )
        )
        if same_input:
            if args.attention_dir is None or args.same_input_moe_dir is None:
                raise ValueError("same-input mode requires attention and MoE directories")
            if any((args.qx_dir, args.nominal_moe_dir, args.perturbed_moe_dir)):
                raise ValueError("same-input and perturbation arguments cannot be mixed")
            if args.expected_vcur_count is None or args.expected_vcur_count <= 0:
                raise ValueError("same-input mode requires a positive expected Vcur count")
            if args.expected_kqv_out_count is None or args.expected_kqv_out_count <= 0:
                raise ValueError("same-input mode requires a positive expected kqv_out count")
            report = compare_same_input(
                args.oracle_dir,
                args.attention_dir,
                args.same_input_moe_dir,
                args.layer,
                args.expected_vcur_count,
                args.expected_kqv_out_count,
            )
        else:
            if any(value is None for value in (args.qx_dir, args.nominal_moe_dir, args.perturbed_moe_dir)):
                raise ValueError("perturbation mode requires QX, nominal MoE, and perturbed MoE directories")
            report = compare(
                args.oracle_dir,
                args.qx_dir,
                args.nominal_moe_dir,
                args.perturbed_moe_dir,
                args.layer,
                args.step,
            )
    except (OSError, ValueError, OverflowError) as exc:
        print(json.dumps({"schema": 1, "passed": False, "error": str(exc)}))
        return 2
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
