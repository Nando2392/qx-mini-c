import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / "scripts" / "compare_layer_sensitivity.py"
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


def write_f32(path: Path, values):
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def write_probe(directory: Path, layer: int, *, experts, weighted):
    directory.mkdir()
    logits = [-10.0] * 128
    probabilities = [0.0] * 128
    for rank, expert in enumerate(experts):
        logits[expert] = 1.0 - rank * 0.1
        probabilities[expert] = 0.6 - rank * 0.2
    fixtures = {
        "ffn_norm": [1.0, 2.0],
        "ffn_moe_logits": logits,
        "ffn_moe_probs": probabilities,
        "ffn_moe_topk": [float(expert) for expert in experts],
        "ffn_moe_weights": [0.6, 0.4],
        "ffn_moe_weights_sum": [1.0],
        "ffn_moe_weights_norm": [0.6, 0.4],
        "ffn_moe_gate": [1.0, 2.0],
        "ffn_moe_up": [1.0, 2.0],
        "ffn_moe_swiglu": [1.0, 2.0],
        "ffn_moe_down": [1.0, 0.0, 0.0, 1.0],
        "ffn_moe_weighted": weighted,
    }
    for name in STAGES:
        write_f32(directory / f"{name}-{layer}.f32", fixtures[name])


def make_case(tmp_path):
    layer = 1
    oracle, qx = tmp_path / "oracle", tmp_path / "qx"
    nominal, perturbed = tmp_path / "nominal", tmp_path / "perturbed"
    oracle.mkdir()
    qx.mkdir()

    write_f32(oracle / "layer-1.f32", [1.0, 2.0])
    write_f32(oracle / "ffn_inp-1.f32", [2.0, 4.0])
    write_f32(oracle / "l_out-1.f32", [12.0, 5.0])
    write_f32(qx / "step-0-layer-1-input.f32", [1.1, 2.0])
    write_f32(qx / "step-0-layer-1-ffn-inp.f32", [2.4, 4.0])
    write_f32(qx / "step-0-layer-1-output.f32", [16.4, 6.0])

    write_probe(nominal, layer, experts=[68, 9], weighted=[10.0, 0.0, 0.0, 1.0])
    write_probe(perturbed, layer, experts=[68, 13], weighted=[14.0, 0.0, 0.0, 2.0])
    return oracle, qx, nominal, perturbed


def run_compare(oracle, qx, nominal, perturbed):
    return subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            "--oracle-dir",
            str(oracle),
            "--qx-dir",
            str(qx),
            "--nominal-moe-dir",
            str(nominal),
            "--perturbed-moe-dir",
            str(perturbed),
            "--layer",
            "1",
            "--step",
            "0",
        ],
        text=True,
        capture_output=True,
    )


def make_same_input_case(tmp_path):
    layer = 1
    oracle = tmp_path / "oracle-same-input"
    attention = tmp_path / "attention-same-input"
    moe = tmp_path / "moe-same-input"
    attention.mkdir()

    write_probe(oracle, layer, experts=[68, 9], weighted=[10.0, 0.0, 0.0, 1.0])
    write_f32(oracle / "layer-1.f32", [1.0, 2.0])
    write_f32(oracle / "attn_norm-1.f32", [0.5, 1.0])
    write_f32(oracle / "Vcur-1.f32", [3.0, 4.0])
    write_f32(oracle / "kqv_out-1.f32", [5.0, 6.0])
    write_f32(oracle / "ffn_inp-1.f32", [2.0, 4.0])
    write_f32(oracle / "ffn_moe_out-1.f32", [10.0, 1.0])
    write_f32(oracle / "l_out-1.f32", [12.0, 5.0])

    write_f32(attention / "attn_norm-1.f32", [0.5, 1.0])
    write_f32(attention / "Vcur-1.f32", [3.0, 4.0])
    write_f32(attention / "kqv_out-1.f32", [5.0, 6.0])
    write_f32(attention / "attn_out-1.f32", [1.0, 2.0])
    write_f32(attention / "ffn_inp-1.f32", [2.0, 4.0])
    write_probe(moe, layer, experts=[68, 9], weighted=[10.0, 0.0, 0.0, 1.0])
    return oracle, attention, moe


def run_same_input_compare(
    oracle,
    attention,
    moe,
    *,
    expected_vcur_count=2,
    expected_kqv_out_count=2,
):
    return subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            "--oracle-dir",
            str(oracle),
            "--attention-dir",
            str(attention),
            "--same-input-moe-dir",
            str(moe),
            "--expected-vcur-count",
            str(expected_vcur_count),
            "--expected-kqv-out-count",
            str(expected_kqv_out_count),
            "--layer",
            "1",
        ],
        text=True,
        capture_output=True,
    )


def test_compare_layer_sensitivity_reports_same_input_attention_and_moe_chain(tmp_path):
    oracle, attention, moe = make_same_input_case(tmp_path)

    completed = run_same_input_compare(oracle, attention, moe)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["passed"] is True
    assert payload["mode"] == "same_input"
    assert payload["routing"] == {
        "oracle": [68, 9],
        "qx": [68, 9],
        "exact": True,
    }
    assert payload["checkpoints"]["attention"]["ffn_input"]["max_abs"] == 0.0
    assert payload["checkpoints"]["moe"]["weighted"]["max_abs"] == 0.0
    assert payload["reconstruction"]["moe_output"]["max_abs"] == 0.0
    assert payload["reconstruction"]["layer_output"]["max_abs"] == 0.0


def test_compare_layer_sensitivity_fails_closed_for_incomplete_same_input_mode(tmp_path):
    oracle, attention, _ = make_same_input_case(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            "--oracle-dir",
            str(oracle),
            "--attention-dir",
            str(attention),
            "--layer",
            "1",
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["passed"] is False
    assert payload["error"] == "same-input mode requires attention and MoE directories"


def test_compare_layer_sensitivity_fails_closed_when_comparison_modes_are_mixed(tmp_path):
    oracle, attention, moe = make_same_input_case(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            "--oracle-dir",
            str(oracle),
            "--attention-dir",
            str(attention),
            "--same-input-moe-dir",
            str(moe),
            "--qx-dir",
            str(tmp_path / "unused-qx"),
            "--layer",
            "1",
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["passed"] is False
    assert payload["error"] == "same-input and perturbation arguments cannot be mixed"


def test_compare_layer_sensitivity_fails_closed_when_attention_residual_is_inconsistent(tmp_path):
    oracle, attention, moe = make_same_input_case(tmp_path)
    write_f32(attention / "ffn_inp-1.f32", [3.0, 4.0])

    completed = run_same_input_compare(oracle, attention, moe)

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["passed"] is False
    assert payload["error"] == "ffn input contradicts layer input plus attention output"


def test_compare_layer_sensitivity_fails_closed_for_equally_truncated_vcur_sidecars(tmp_path):
    oracle, attention, moe = make_same_input_case(tmp_path)
    write_f32(oracle / "Vcur-1.f32", [3.0])
    write_f32(attention / "Vcur-1.f32", [3.0])

    completed = run_same_input_compare(oracle, attention, moe)

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["passed"] is False
    assert payload["error"] == "Vcur count mismatch: 1 != 2"


def test_compare_layer_sensitivity_fails_closed_for_equally_truncated_kqv_sidecars(tmp_path):
    oracle, attention, moe = make_same_input_case(tmp_path)
    write_f32(oracle / "kqv_out-1.f32", [5.0])
    write_f32(attention / "kqv_out-1.f32", [5.0])

    completed = run_same_input_compare(oracle, attention, moe)

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["passed"] is False
    assert payload["error"] == "kqv_out count mismatch: 1 != 2"


def test_compare_layer_sensitivity_reports_routing_transition_and_dominant_expert(tmp_path):
    oracle, qx, nominal, perturbed = make_case(tmp_path)

    completed = run_compare(oracle, qx, nominal, perturbed)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["passed"] is True
    assert payload["routing"]["nominal"] == [68, 9]
    assert payload["routing"]["perturbed"] == [68, 13]
    assert payload["routing"]["dropped"] == [9]
    assert payload["routing"]["added"] == [13]
    assert payload["routing"]["nominal_weights"] == pytest.approx([0.6, 0.4])
    assert payload["routing"]["perturbed_weights"] == pytest.approx([0.6, 0.4])
    assert payload["routing"]["weight_deltas"]["raw"]["max_abs"] == 0.0
    assert payload["routing"]["weight_deltas"]["normalized"]["max_abs"] == 0.0
    assert payload["dominant_expert"]["expert_id"] == 68
    assert payload["dominant_expert"]["delta_l2"] == 4.0
    assert math.isclose(payload["gains"]["ffn_input_from_layer_input"], 4.0, rel_tol=1e-6)
    assert payload["gains"]["moe_output_from_ffn_input"] > 10.0
    assert payload["reconstruction"]["nominal_vs_oracle"]["max_abs"] == 0.0
    assert payload["reconstruction"]["perturbed_vs_qx"]["max_abs"] < 1e-6
    assert payload["first_amplification_checkpoint"] == "attention_output_derived"


def test_compare_layer_sensitivity_reports_no_amplification_below_threshold(tmp_path):
    oracle, qx, nominal, perturbed = make_case(tmp_path)
    logits = [-10.0] * 128
    probabilities = [0.0] * 128
    logits[68], logits[9] = 1.0, 0.9
    probabilities[68], probabilities[9] = 0.6, 0.4
    write_f32(perturbed / "ffn_moe_logits-1.f32", logits)
    write_f32(perturbed / "ffn_moe_probs-1.f32", probabilities)
    write_f32(perturbed / "ffn_moe_topk-1.f32", [68.0, 9.0])
    write_f32(perturbed / "ffn_moe_weighted-1.f32", [10.1, 0.0, 0.0, 1.0])
    write_f32(qx / "step-0-layer-1-ffn-inp.f32", [2.1, 4.0])
    write_f32(qx / "step-0-layer-1-output.f32", [12.2, 5.0])

    completed = run_compare(oracle, qx, nominal, perturbed)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["first_amplification_checkpoint"] is None
    amplification_gains = (
        payload["gains"]["attention_output_from_layer_input"],
        payload["gains"]["ffn_input_from_layer_input"],
        payload["gains"]["moe_output_from_ffn_input"],
    )
    assert max(amplification_gains) < 1.01


def test_compare_layer_sensitivity_fails_closed_when_topk_contradicts_logits(tmp_path):
    oracle, qx, nominal, perturbed = make_case(tmp_path)
    logits = [-10.0] * 128
    logits[68] = 1.0
    logits[32] = 0.9
    write_f32(perturbed / "ffn_moe_logits-1.f32", logits)

    completed = run_compare(oracle, qx, nominal, perturbed)

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["passed"] is False
    assert "top-k does not match router logits" in payload["error"]


def test_compare_layer_sensitivity_fails_closed_when_normalized_weights_contradict_raw_weights(tmp_path):
    oracle, qx, nominal, perturbed = make_case(tmp_path)
    write_f32(perturbed / "ffn_moe_weights_norm-1.f32", [0.5, 0.5])

    completed = run_compare(oracle, qx, nominal, perturbed)

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["passed"] is False
    assert "normalized weights contradict raw weights" in payload["error"]


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("raw_probability", "raw weights contradict selected router probabilities"),
        ("weight_sum", "weight sum contradicts raw weights"),
    ),
)
def test_compare_layer_sensitivity_fails_closed_when_raw_weight_contract_is_inconsistent(
    tmp_path, mutation, expected_error
):
    oracle, qx, nominal, perturbed = make_case(tmp_path)
    if mutation == "raw_probability":
        write_f32(perturbed / "ffn_moe_weights-1.f32", [0.5, 0.5])
        write_f32(perturbed / "ffn_moe_weights_norm-1.f32", [0.5, 0.5])
    else:
        write_f32(perturbed / "ffn_moe_weights_sum-1.f32", [0.9])

    completed = run_compare(oracle, qx, nominal, perturbed)

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["passed"] is False
    assert expected_error in payload["error"]


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("missing", "ffn_moe_up-1.f32"),
        ("truncated", "invalid F32 sidecar size"),
        ("zero_experts", "invalid F32 sidecar size"),
        ("nonfinite", "non-finite F32 sidecar"),
        ("zero_delta", "invalid zero or non-finite gain denominator"),
    ),
)
def test_compare_layer_sensitivity_fails_closed_for_invalid_sidecar_inputs(tmp_path, mutation, expected_error):
    oracle, qx, nominal, perturbed = make_case(tmp_path)
    if mutation == "missing":
        (perturbed / "ffn_moe_up-1.f32").unlink()
    elif mutation == "truncated":
        (perturbed / "ffn_moe_up-1.f32").write_bytes(b"\x00")
    elif mutation == "zero_experts":
        (perturbed / "ffn_moe_topk-1.f32").write_bytes(b"")
    elif mutation == "nonfinite":
        write_f32(qx / "step-0-layer-1-input.f32", [math.nan, 2.0])
    else:
        write_f32(qx / "step-0-layer-1-input.f32", [1.0, 2.0])

    completed = run_compare(oracle, qx, nominal, perturbed)

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["passed"] is False
    assert expected_error in payload["error"]


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("duplicate_topk", "invalid top-k expert selection"),
        ("out_of_range_topk", "invalid top-k expert selection"),
        ("probability_shape", "router probability shape mismatch"),
        ("weight_shape", "ffn_moe_weights shape mismatch"),
    ),
)
def test_compare_layer_sensitivity_fails_closed_for_invalid_routing_shapes(tmp_path, mutation, expected_error):
    oracle, qx, nominal, perturbed = make_case(tmp_path)
    if mutation == "duplicate_topk":
        write_f32(perturbed / "ffn_moe_topk-1.f32", [68.0, 68.0])
    elif mutation == "out_of_range_topk":
        write_f32(perturbed / "ffn_moe_topk-1.f32", [68.0, 128.0])
    elif mutation == "probability_shape":
        write_f32(perturbed / "ffn_moe_probs-1.f32", [0.0] * 127)
    else:
        write_f32(perturbed / "ffn_moe_weights-1.f32", [1.0])

    completed = run_compare(oracle, qx, nominal, perturbed)

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["passed"] is False
    assert expected_error in payload["error"]


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("fractional_topk", "top-k sidecar contains invalid expert id"),
        ("ffn_norm_shape", "ffn_norm shape mismatch"),
        ("normalized_weight_shape", "ffn_moe_weights_norm shape mismatch"),
        ("weight_sum_shape", "weight sum shape mismatch"),
        ("gate_shape", "gate shape mismatch"),
        ("up_shape", "ffn_moe_up shape mismatch"),
        ("swiglu_shape", "ffn_moe_swiglu shape mismatch"),
        ("down_shape", "ffn_moe_down shape mismatch"),
        ("weighted_shape", "ffn_moe_weighted shape mismatch"),
        ("layer_sidecar_shape", "layer sidecar shape mismatch"),
        ("expert_count", "expert count mismatch"),
        ("topk_width", "top-k width mismatch"),
        ("intermediate_shape", "expert intermediate shape mismatch"),
    ),
)
def test_compare_layer_sensitivity_fails_closed_for_every_structural_contract(
    tmp_path, mutation, expected_error
):
    oracle, qx, nominal, perturbed = make_case(tmp_path)
    if mutation == "fractional_topk":
        write_f32(perturbed / "ffn_moe_topk-1.f32", [68.5, 13.0])
    elif mutation == "ffn_norm_shape":
        write_f32(perturbed / "ffn_norm-1.f32", [1.0])
    elif mutation == "normalized_weight_shape":
        write_f32(perturbed / "ffn_moe_weights_norm-1.f32", [1.0])
    elif mutation == "weight_sum_shape":
        write_f32(perturbed / "ffn_moe_weights_sum-1.f32", [1.0, 1.0])
    elif mutation == "gate_shape":
        write_f32(perturbed / "ffn_moe_gate-1.f32", [1.0, 2.0, 3.0])
    elif mutation == "up_shape":
        write_f32(perturbed / "ffn_moe_up-1.f32", [1.0])
    elif mutation == "swiglu_shape":
        write_f32(perturbed / "ffn_moe_swiglu-1.f32", [1.0])
    elif mutation == "down_shape":
        write_f32(perturbed / "ffn_moe_down-1.f32", [1.0, 2.0, 3.0])
    elif mutation == "weighted_shape":
        write_f32(perturbed / "ffn_moe_weighted-1.f32", [1.0, 2.0, 3.0])
    elif mutation == "layer_sidecar_shape":
        write_f32(qx / "step-0-layer-1-output.f32", [1.0])
    elif mutation == "expert_count":
        logits = [-10.0] * 129
        probabilities = [0.0] * 129
        logits[68], logits[13] = 1.0, 0.9
        probabilities[68], probabilities[13] = 0.6, 0.4
        write_f32(perturbed / "ffn_moe_logits-1.f32", logits)
        write_f32(perturbed / "ffn_moe_probs-1.f32", probabilities)
    elif mutation == "topk_width":
        write_f32(perturbed / "ffn_moe_topk-1.f32", [68.0])
        write_f32(perturbed / "ffn_moe_weights-1.f32", [0.6])
        write_f32(perturbed / "ffn_moe_weights_sum-1.f32", [0.6])
        write_f32(perturbed / "ffn_moe_weights_norm-1.f32", [1.0])
        write_f32(perturbed / "ffn_moe_down-1.f32", [1.0, 0.0])
        write_f32(perturbed / "ffn_moe_weighted-1.f32", [1.0, 0.0])
    else:
        write_f32(perturbed / "ffn_moe_gate-1.f32", [1.0, 2.0, 3.0, 4.0])
        write_f32(perturbed / "ffn_moe_up-1.f32", [1.0, 2.0, 3.0, 4.0])
        write_f32(perturbed / "ffn_moe_swiglu-1.f32", [1.0, 2.0, 3.0, 4.0])

    completed = run_compare(oracle, qx, nominal, perturbed)

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["passed"] is False
    assert expected_error in payload["error"]
