import importlib.util
import json
import math
import struct
from argparse import Namespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_issue80_layer3_seams.py"


def load_module():
    spec = importlib.util.spec_from_file_location("issue80_layer3", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_f32(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def f32_add(left, right):
    return [struct.unpack("<f", struct.pack("<f", a + b))[0] for a, b in zip(left, right)]


def as_f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def write_moe_probe(directory: Path, *, mode: str, weighted=None, selected=None):
    directory.mkdir(parents=True, exist_ok=True)
    selected = selected or list(range(8))
    weighted = weighted or [0.0] * 16
    weighted[0], weighted[1] = 3.0, 4.0
    logits = [8.0 - index for index in range(8)]
    probs = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04]
    total = math.fsum(probs)
    normalized = [as_f32(value / total) for value in probs]
    fixtures = {
        "ffn_norm": [0.5, 1.0],
        "ffn_moe_logits": logits,
        "ffn_moe_probs": probs,
        "ffn_moe_topk": selected,
        "ffn_moe_weights": probs,
        "ffn_moe_weights_sum": [total],
        "ffn_moe_weights_norm": normalized,
        "ffn_moe_gate": [float(index) for index in range(8)],
        "ffn_moe_up": [float(index) for index in range(8)],
        "ffn_moe_swiglu": [float(index) for index in range(8)],
        "ffn_moe_down": [0.0] * 16,
        "ffn_moe_weighted": weighted,
    }
    for name, values in fixtures.items():
        write_f32(directory / f"{name}-3.f32", values)
    (directory / "result.json").write_text(json.dumps({
        "probe": "moe_stage", "layer": 3, "input_count": 2,
        "experts": 8, "experts_used": 8, "intermediate": 1,
        "activation_mode": mode, "selected_experts": selected,
        "routing_weights": normalized, "sidecars_written": 12,
        "router_precision": "integrated_double",
        "routing_sidecar_element_type": "f32",
        "weighted_contribution_element_type": "f32",
        "weighted_contribution_formula": "f32(integrated_double_weight*expert_f32)",
    }), encoding="utf-8")


def write_integrated(directory: Path, *, mode: str, selected=None):
    directory.mkdir(parents=True, exist_ok=True)
    selected = selected or list(range(8))
    values = {
        "input": [0.5, 1.5],
        "v-cur": [2.0],
        "kqv-out": [3.0, 4.0],
        "ffn-inp": [1.0, 2.0],
        "ffn-moe-out": [3.0, 4.0],
        "output": [4.0, 6.0],
    }
    for phase, payload in values.items():
        write_f32(directory / f"step-1-layer-3-{phase}.f32", payload)
    layers = [
        {
            "layer": layer,
            "selected_experts": selected,
            "routing_weights": [0.125] * 8,
            "attention_context_tokens": 2,
            "full_moe": True,
        }
        for layer in (3, 4)
    ]
    payload = {
        "probe": "state_loop", "prompt_token": 1124, "steps": 1,
        "layers": 5, "layers_run": 2, "start_layer": 3,
        "position_base": 1, "kv_format": "int8",
        "activation_format": mode, "residual_source": "injected_f32_replay",
        "residual_replay": {"enabled": True, "source": "f32_sidecar", "values": 2},
        "tokens": [{"step": 1, "position": 1, "input_token": 1124, "layers": layers}],
    }
    (directory / "result.json").write_text(json.dumps(payload), encoding="utf-8")


def make_case(tmp_path):
    work = tmp_path / "work"
    snapshot, residual = tmp_path / "snapshot.bin", tmp_path / "residual.f32"
    exe, model = tmp_path / "qxqxf.exe", tmp_path / "model.qxf"
    snapshot.write_bytes(b"fixed-int8-kv")
    write_f32(residual, [0.5, 1.5])
    exe.write_bytes(b"exe")
    model.write_bytes(b"model")
    for mode in ("f32", "q8_k_compat"):
        write_integrated(work / f"integrated-{mode}", mode=mode)
        write_moe_probe(work / f"moe-fixed-f32-input-{mode}", mode=mode)
        write_moe_probe(work / f"moe-control-{mode}", mode=mode)
    return work, snapshot, residual, exe, model


def analyze(module, case):
    work, snapshot, residual, exe, model = case
    return module.analyze_artifacts(
        work=work, snapshot=snapshot, residual=residual, executable=exe, model=model,
        expected_snapshot_sha256=module.sha256(snapshot),
        expected_residual_sha256=module.sha256(residual),
        hidden=2, vcur_count=1, kqv_count=2, layers=5, continuation_token=1124,
    )


def test_issue80_report_validates_provenance_and_reports_real_seams(tmp_path):
    module = load_module()
    report = analyze(module, make_case(tmp_path))

    assert report["schema"] == "qx-issue80-layer3-seams-v1"
    assert report["provenance"]["snapshot_sha256"] == module.sha256(tmp_path / "snapshot.bin")
    assert report["attention"]["input"]["byte_exact"] is True
    assert report["attention"]["post_attention_contribution_derived"]["count"] == 2
    assert report["attention"]["ffn_input"]["max_abs"] == 0.0
    assert report["routing"]["integrated"] == {"f32": list(range(8)), "q8_k_compat": list(range(8))}
    assert report["moe_same_f32_input"]["moe_output"]["count"] == 2
    assert report["bridge_controls"]["f32"]["moe_output_byte_exact"] is True
    assert report["bridge_controls"]["q8_k_compat"]["layer_output_byte_exact"] is True
    assert report["bridge_controls"]["summation_semantics"] == "ordered_rank_f32_accumulation"


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("context", "attention context token count must be 2"),
        ("position", "integrated metadata invalid"),
        ("duplicate", "integrated layers are duplicate or disordered"),
        ("routing", "integrated routing must contain ordered integer expert IDs"),
        ("nonfinite", "non-finite F32 sidecar"),
        ("truncated", "F32 sidecar count mismatch"),
        ("extra", "unexpected layer-3 sidecars"),
    ],
)
def test_issue80_report_fails_closed_for_metadata_shape_and_routing(tmp_path, mutation, error):
    module = load_module()
    case = make_case(tmp_path)
    directory = case[0] / "integrated-f32"
    payload_path = directory / "result.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if mutation == "context":
        payload["tokens"][0]["layers"][0]["attention_context_tokens"] = 1
    elif mutation == "position":
        payload["position_base"] = 0
    elif mutation == "duplicate":
        payload["tokens"][0]["layers"][1]["layer"] = 3
    elif mutation == "routing":
        payload["tokens"][0]["layers"][0]["selected_experts"][0] = 0.0
    elif mutation == "nonfinite":
        write_f32(directory / "step-1-layer-3-v-cur.f32", [math.nan])
    elif mutation == "truncated":
        write_f32(directory / "step-1-layer-3-kqv-out.f32", [3.0])
    else:
        write_f32(directory / "step-1-layer-3-attn-out.f32", [1.0, 1.0])
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        analyze(module, case)


def test_issue80_report_rejects_provenance_digest_mismatch(tmp_path):
    module = load_module()
    case = make_case(tmp_path)
    work, snapshot, residual, exe, model = case
    with pytest.raises(ValueError, match="snapshot SHA-256 mismatch"):
        module.analyze_artifacts(
            work=work, snapshot=snapshot, residual=residual, executable=exe, model=model,
            expected_snapshot_sha256="0" * 64,
            expected_residual_sha256=module.sha256(residual),
            hidden=2, vcur_count=1, kqv_count=2, layers=5, continuation_token=1124,
        )


def test_issue80_bridge_control_requires_integrated_float_summation_byte_equality(tmp_path):
    module = load_module()
    case = make_case(tmp_path)
    write_f32(case[0] / "integrated-q8_k_compat" / "step-1-layer-3-ffn-moe-out.f32", [3.0, 4.5])

    with pytest.raises(ValueError, match=r"q8_k_compat: integrated/standalone MoE output is not byte-exact .*max_abs=.*rmse="):
        analyze(module, case)


@pytest.mark.parametrize(
    ("location", "field"),
    [
        ("root", "prompt_token"), ("root", "steps"), ("root", "layers"),
        ("root", "layers_run"), ("root", "start_layer"), ("root", "position_base"),
        ("replay", "values"), ("token", "step"), ("token", "position"),
        ("token", "input_token"), ("layer", "layer"), ("layer", "attention_context_tokens"),
    ],
)
@pytest.mark.parametrize("replacement", [1.0, True])
def test_integrated_required_integer_metadata_rejects_float_and_bool(tmp_path, location, field, replacement):
    module = load_module()
    case = make_case(tmp_path)
    path = case[0] / "integrated-f32" / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    target = {
        "root": payload,
        "replay": payload["residual_replay"],
        "token": payload["tokens"][0],
        "layer": payload["tokens"][0]["layers"][0],
    }[location]
    target[field] = replacement
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrated metadata invalid|duplicate or disordered|attention context"):
        analyze(module, case)


@pytest.mark.parametrize("field", ["layer", "input_count", "experts", "experts_used", "intermediate", "sidecars_written"])
@pytest.mark.parametrize("replacement", [1.0, True])
def test_moe_required_integer_metadata_rejects_float_and_bool(tmp_path, field, replacement):
    module = load_module()
    case = make_case(tmp_path)
    path = case[0] / "moe-fixed-f32-input-f32" / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = replacement
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="standalone MoE metadata invalid"):
        analyze(module, case)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_artifacts_reject_nonstandard_constants(tmp_path, constant):
    module = load_module()
    case = make_case(tmp_path)
    path = case[0] / "integrated-f32" / "result.json"
    path.write_text('{"ignored": ' + constant + "}", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON artifact"):
        analyze(module, case)


@pytest.mark.parametrize(
    "weights",
    [None, "bad", [0.125] * 7, [True] + [0.125] * 7, ["0.125"] * 8,
     [math.nan] + [0.125] * 7, [math.inf] + [0.125] * 7,
     [-0.125] + [0.125] * 7, [0.2] * 8],
)
def test_integrated_routing_weights_fail_closed(tmp_path, weights):
    module = load_module()
    case = make_case(tmp_path)
    path = case[0] / "integrated-f32" / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if weights is None:
        del payload["tokens"][0]["layers"][0]["routing_weights"]
    else:
        payload["tokens"][0]["layers"][0]["routing_weights"] = weights
    path.write_text(json.dumps(payload), encoding="utf-8")
    error = "invalid JSON artifact" if isinstance(weights, list) and any(
        type(value) is float and not math.isfinite(value) for value in weights
    ) else "integrated routing weights invalid"
    with pytest.raises(ValueError, match=error):
        analyze(module, case)


def test_routing_requires_exactly_eight_unique_in_range_ids(tmp_path):
    module = load_module()
    for selected in (list(range(7)), list(range(7)) + [128], [0] * 8):
        case_dir = tmp_path / str(len(selected)) / str(selected[-1])
        case_dir.mkdir(parents=True)
        case = make_case(case_dir)
        write_integrated(case[0] / "integrated-f32", mode="f32", selected=selected)
        with pytest.raises(ValueError, match="integrated routing"):
            analyze(module, case)


@pytest.mark.parametrize("mutation", ["experts-used", "weights-count", "weights-value", "weights-sum"])
def test_standalone_routing_matches_native_top8_f32_schema(tmp_path, mutation):
    module = load_module()
    case = make_case(tmp_path)
    directory = case[0] / "moe-fixed-f32-input-f32"
    path = directory / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "experts-used":
        payload["experts_used"] = 7
    elif mutation == "weights-count":
        payload["routing_weights"] = payload["routing_weights"][:-1]
    elif mutation == "weights-value":
        payload["routing_weights"][0] += 0.01
    else:
        weights = [as_f32(0.2)] * 8
        payload["routing_weights"] = weights
        write_f32(directory / "ffn_moe_weights_norm-3.f32", weights)
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    error = "standalone MoE metadata invalid" if mutation == "experts-used" else "standalone routing|routing weights"
    with pytest.raises(ValueError, match=error):
        analyze(module, case)


@pytest.mark.parametrize("router_precision", [None, "legacy"])
def test_standalone_rejects_missing_or_legacy_router_precision(tmp_path, router_precision):
    module = load_module()
    case = make_case(tmp_path)
    path = case[0] / "moe-fixed-f32-input-f32" / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if router_precision is None:
        del payload["router_precision"]
    else:
        payload["router_precision"] = router_precision
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="standalone MoE metadata invalid"):
        analyze(module, case)


def run_args(module, tmp_path):
    snapshot = tmp_path / "snapshot.bin"
    residual = tmp_path / "residual.f32"
    exe = tmp_path / "qxqxf.exe"
    model = tmp_path / "model.qxf"
    snapshot.write_bytes(b"snapshot")
    write_f32(residual, [0.5, 1.5])
    exe.write_bytes(b"exe")
    model.write_bytes(b"model")
    return Namespace(
        out=tmp_path / "out", snapshot=snapshot, residual=residual, qxqxf=exe, model=model,
        snapshot_sha256=module.sha256(snapshot), residual_sha256=module.sha256(residual),
        hidden=2, vcur_count=1, kqv_count=2, layers=5, ctx=4, seed=7,
        continuation_token=1124,
    )


def fixture_invoke(command, result):
    mode = command[command.index("--activation") + 1]
    if command[1] == "state-loop-probe":
        write_integrated(result.parent, mode=mode)
    else:
        write_moe_probe(result.parent, mode=mode)


def test_run_orchestrates_two_integrated_then_four_bound_moe_consumers(tmp_path, monkeypatch):
    module = load_module()
    args = run_args(module, tmp_path)
    calls = []
    def invoke(command, result):
        calls.append(command)
        fixture_invoke(command, result)
    monkeypatch.setattr(module, "_invoke", invoke)

    report = module.run(args)

    assert [call[1] for call in calls] == ["state-loop-probe"] * 2 + ["moe-stage-probe"] * 4
    integrated = calls[:2]
    assert all("--kv-snapshot-in" in call and "--start-layer" in call and "--residual-in" in call for call in integrated)
    assert all("--final-head" not in call for call in integrated)
    assert all(call[call.index("--prompt-token") + 1] == "1124" for call in integrated)
    standalone = calls[2:]
    assert all(call.count("--router-precision") == 1 for call in standalone)
    assert all(call[call.index("--router-precision") + 1] == "integrated_double" for call in standalone)
    moe_inputs = [Path(call[call.index("--ffn-inp") + 1]) for call in calls[2:]]
    f32_input = args.out / "integrated-f32" / "step-1-layer-3-ffn-inp.f32"
    q8_input = args.out / "integrated-q8_k_compat" / "step-1-layer-3-ffn-inp.f32"
    assert moe_inputs == [f32_input, f32_input, f32_input, q8_input]
    records = report["provenance"]["moe_ffn_inputs"]
    assert len(records) == 4
    assert all(record["sha256_before"] == record["sha256_after"] for record in records)
    assert report["execution"]["commands"] == calls


def test_run_fails_if_bound_ffn_input_changes_between_consumers(tmp_path, monkeypatch):
    module = load_module()
    args = run_args(module, tmp_path)
    count = 0
    def invoke(command, result):
        nonlocal count
        fixture_invoke(command, result)
        if command[1] == "moe-stage-probe":
            count += 1
            if count == 1:
                path = args.out / "integrated-f32" / "step-1-layer-3-ffn-inp.f32"
                path.write_bytes(path.read_bytes() + b"mutation")
    monkeypatch.setattr(module, "_invoke", invoke)
    with pytest.raises(ValueError, match="MoE FFN input hash changed"):
        module.run(args)


def test_run_refuses_existing_output_directory(tmp_path):
    module = load_module()
    args = run_args(module, tmp_path)
    args.out.mkdir()
    with pytest.raises(ValueError, match="output directory already exists"):
        module.run(args)


def test_invoke_fails_closed_for_nonzero_and_malformed_stdout(tmp_path, monkeypatch):
    module = load_module()
    class Completed:
        returncode = 4
        stderr = "native failure"
        stdout = ""
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: Completed())
    with pytest.raises(RuntimeError, match="native failure"):
        module._invoke(["qxqxf"], tmp_path / "one" / "result.json")
    Completed.returncode = 0
    Completed.stdout = "not-json"
    with pytest.raises(ValueError, match="invalid command JSON output"):
        module._invoke(["qxqxf"], tmp_path / "two" / "result.json")
