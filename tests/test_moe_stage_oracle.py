import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
QX_EXE = ROOT / "build" / "qxqxf.exe"
QXF = ROOT / "models" / "Qwen3-30B-A3B-UD-IQ2_M.qxf"
GGUF = ROOT / "models" / "Qwen3-30B-A3B-UD-IQ2_M.gguf"
LLAMA_EXE = ROOT / "build" / "llama_reference_oracle.exe"
LLAMA_BUILD = ROOT / "tests" / "build_llama_reference_oracle.bat"
LLAMA_CPP_DIR = Path(os.environ.get("LLAMA_CPP_DIR", ROOT.parent / "llama.cpp-k3"))
GGML_EXE = ROOT / "build" / "ggml_reference_decode.exe"
GGML_BUILD = ROOT / "tests" / "build_ggml_reference.bat"
COMPARE_LAYER = ROOT / "scripts" / "compare_layer_sensitivity.py"


def require_local_moe_fixtures():
    if os.name != "nt" or not QX_EXE.exists() or not QXF.exists() or not GGUF.exists():
        pytest.skip("local Windows Qwen MoE fixtures are not available")
    if not (LLAMA_CPP_DIR / "include" / "llama.h").exists():
        pytest.skip("pinned llama.cpp oracle checkout is not available")


def run_oracle(output: Path, internal_layer=0, kv_type="f16"):
    env = os.environ.copy()
    env["LLAMA_CPP_DIR"] = str(LLAMA_CPP_DIR)
    build = subprocess.run(
        ["cmd.exe", "/c", str(LLAMA_BUILD)], cwd=ROOT, env=env, text=True, capture_output=True
    )
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run(
        [
            str(LLAMA_EXE),
            str(GGUF),
            str(output),
            "42",
            str(internal_layer),
            kv_type,
            "internals" if internal_layer == 0 else f"internals={internal_layer}",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_oracle_captures_layer_1_moe_internals(tmp_path):
    require_local_moe_fixtures()
    oracle_dir = tmp_path / "oracle-layer-1"
    payload = run_oracle(oracle_dir, internal_layer=1)
    assert payload["internals_captured"] == 19
    for name, count in {
        "attn_norm-1": 2048,
        "ffn_inp-1": 2048,
        "ffn_norm-1": 2048,
        "ffn_moe_logits-1": 128,
        "ffn_moe_gate-1": 768 * 8,
        "ffn_moe_down-1": 2048 * 8,
        "ffn_moe_out-1": 2048,
        "l_out-1": 2048,
    }.items():
        assert (oracle_dir / f"{name}.f32").stat().st_size == count * 4


def read_f32(path: Path):
    raw = path.read_bytes()
    return struct.unpack(f"<{len(raw) // 4}f", raw)


def max_abs(left, right):
    assert len(left) == len(right)
    return max(abs(a - b) for a, b in zip(left, right))


def test_moe_stage_probe_accepts_oracle_ffn_input_and_exports_stages(tmp_path):
    require_local_moe_fixtures()
    oracle_dir = tmp_path / "oracle"
    oracle = run_oracle(oracle_dir)
    assert oracle["internals_captured"] == 19

    qx_dir = tmp_path / "qx"
    qx_dir.mkdir()
    result = subprocess.run(
        [
            str(QX_EXE),
            "moe-stage-probe",
            "--in",
            str(QXF),
            "--layer",
            "0",
            "--ffn-inp",
            str(oracle_dir / "ffn_inp-0.f32"),
            "--out-dir",
            str(qx_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["probe"] == "moe_stage"
    assert payload["layer"] == 0
    assert payload["input_count"] == 2048
    assert payload["experts"] == 128
    assert payload["experts_used"] == 8
    assert payload["intermediate"] == 768
    assert len(payload["selected_experts"]) == 8
    assert len(payload["routing_weights"]) == 8
    assert payload["sidecars_written"] == 12

    expected_counts = {
        "ffn_norm-0": 2048,
        "ffn_moe_logits-0": 128,
        "ffn_moe_probs-0": 128,
        "ffn_moe_topk-0": 8,
        "ffn_moe_weights-0": 8,
        "ffn_moe_weights_sum-0": 1,
        "ffn_moe_weights_norm-0": 8,
        "ffn_moe_gate-0": 768 * 8,
        "ffn_moe_up-0": 768 * 8,
        "ffn_moe_swiglu-0": 768 * 8,
        "ffn_moe_down-0": 2048 * 8,
        "ffn_moe_weighted-0": 2048 * 8,
    }
    for name, count in expected_counts.items():
        assert (qx_dir / f"{name}.f32").stat().st_size == count * 4


@pytest.mark.parametrize(
    ("internal_layer", "tensor_name", "quant_name", "block_size", "expert", "rows", "activation_name", "activation_count"),
    [
        (0, "blk.0.ffn_gate_exps.weight", "iq2_xs", 74, 49, (0, 384, 767), "ffn_norm-0.f32", 2048),
        (0, "blk.0.ffn_down_exps.weight", "iq3_xxs", 98, 49, (0, 1024, 2047), "ffn_moe_swiglu-0.f32", 768),
        (1, "blk.1.ffn_gate_exps.weight", "iq2_s", 82, 0, (0, 384, 767), "ffn_norm-1.f32", 2048),
        (1, "blk.1.ffn_down_exps.weight", "iq4_xs", 136, 0, (0, 1024, 2047), "ffn_moe_swiglu-1.f32", 768),
        (41, "blk.41.ffn_down_exps.weight", "iq3_s", 110, 48, (0, 1024, 2047), "ffn_moe_swiglu-41.f32", 768),
    ],
)
def test_q8_k_expert_dot_matches_pinned_ggml_kernel(
    tmp_path, internal_layer, tensor_name, quant_name, block_size, expert, rows, activation_name, activation_count
):
    require_local_moe_fixtures()
    oracle_dir = tmp_path / "oracle"
    run_oracle(oracle_dir, internal_layer=internal_layer)
    env = os.environ.copy()
    env["LLAMA_CPP_DIR"] = str(LLAMA_CPP_DIR)
    built = subprocess.run(
        ["cmd.exe", "/c", str(GGML_BUILD)], cwd=ROOT, env=env, text=True, capture_output=True
    )
    assert built.returncode == 0, built.stdout + built.stderr

    activation = oracle_dir / activation_name
    if activation_count == 768:
        raw = activation.read_bytes()
        activation = tmp_path / "rank0-swiglu.f32"
        activation.write_bytes(raw[: activation_count * 4])

    tensor = json.loads(
        subprocess.check_output(
            [str(QX_EXE), "inspect-tensor", "--in", str(QXF), "--name", tensor_name], text=True
        )
    )
    blocks = activation_count // 256
    expert_bytes = tensor["byte_size"] // tensor["dims"][2]
    row_bytes = blocks * block_size
    for row in rows:
        qx = subprocess.run(
            [
                str(QX_EXE),
                "expert-q8-k-dot-probe",
                "--in",
                str(QXF),
                "--name",
                tensor_name,
                "--expert",
                str(expert),
                "--row",
                str(row),
                "--activation",
                str(activation),
            ],
            text=True,
            capture_output=True,
        )
        assert qx.returncode == 0, qx.stdout + qx.stderr
        qx_dot = json.loads(qx.stdout)["dot"]
        offset = tensor["offset"] + expert * expert_bytes + row * row_bytes
        ggml = subprocess.run(
            [
                str(GGML_EXE),
                f"{quant_name}_q8_k_dot",
                str(QXF),
                str(offset),
                str(blocks),
                str(activation),
            ],
            capture_output=True,
        )
        assert ggml.returncode == 0, ggml.stderr.decode(errors="replace")
        ggml_dot = struct.unpack("<f", ggml.stdout)[0]
        assert math.isclose(qx_dot, ggml_dot, rel_tol=2e-6, abs_tol=2e-6)


def test_moe_stage_q8_k_closes_expert_kernel_divergence(tmp_path):
    require_local_moe_fixtures()
    oracle_dir = tmp_path / "oracle"
    qx_dir = tmp_path / "qx-q8-k"
    qx_dir.mkdir()
    run_oracle(oracle_dir)
    completed = subprocess.run(
        [
            str(QX_EXE),
            "moe-stage-probe",
            "--in",
            str(QXF),
            "--layer",
            "0",
            "--ffn-inp",
            str(oracle_dir / "ffn_inp-0.f32"),
            "--out-dir",
            str(qx_dir),
            "--activation",
            "q8_k_compat",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["activation_mode"] == "q8_k_compat"
    assert payload["projection_kernel"] == "iq2_xs_q8_k_and_iq3_xxs_q8_k"

    for name, limit in {
        "ffn_moe_gate-0": 5e-6,
        "ffn_moe_up-0": 5e-6,
        "ffn_moe_swiglu-0": 5e-6,
        "ffn_moe_down-0": 2e-5,
    }.items():
        assert max_abs(read_f32(qx_dir / f"{name}.f32"), read_f32(oracle_dir / f"{name}.f32")) <= limit

    weighted = read_f32(qx_dir / "ffn_moe_weighted-0.f32")
    qx_moe = [sum(weighted[rank * 2048 + i] for rank in range(8)) for i in range(2048)]
    oracle_moe = read_f32(oracle_dir / "ffn_moe_out-0.f32")
    assert max_abs(qx_moe, oracle_moe) <= 1e-5


def test_moe_stage_q8_k_closes_layer_1_iq2_s_iq4_xs_divergence(tmp_path):
    require_local_moe_fixtures()
    oracle_dir = tmp_path / "oracle-layer-1"
    qx_dir = tmp_path / "qx-layer-1"
    qx_dir.mkdir()
    run_oracle(oracle_dir, internal_layer=1)
    completed = subprocess.run(
        [
            str(QX_EXE),
            "moe-stage-probe",
            "--in",
            str(QXF),
            "--layer",
            "1",
            "--ffn-inp",
            str(oracle_dir / "ffn_inp-1.f32"),
            "--out-dir",
            str(qx_dir),
            "--activation",
            "q8_k_compat",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["projection_kernel"] == "iq2_s_q8_k_and_iq4_xs_q8_k"
    assert payload["gate_up_projection_kernel"] == "iq2_s_q8_k"
    assert payload["down_projection_kernel"] == "iq4_xs_q8_k"
    for name, limit in {
        "ffn_moe_gate-1": 5e-6,
        "ffn_moe_up-1": 5e-6,
        "ffn_moe_swiglu-1": 5e-5,
        "ffn_moe_down-1": 1.5e-4,
    }.items():
        assert max_abs(read_f32(qx_dir / f"{name}.f32"), read_f32(oracle_dir / f"{name}.f32")) <= limit
    weighted = read_f32(qx_dir / "ffn_moe_weighted-1.f32")
    qx_moe = [sum(weighted[rank * 2048 + i] for rank in range(8)) for i in range(2048)]
    assert max_abs(qx_moe, read_f32(oracle_dir / "ffn_moe_out-1.f32")) <= 5e-5


def test_attention_stage_q8_k_matches_layer_1_with_same_input(tmp_path):
    require_local_moe_fixtures()
    oracle_dir = tmp_path / "oracle-layer-1"
    qx_dir = tmp_path / "qx-attention-layer-1"
    qx_dir.mkdir()
    run_oracle(oracle_dir, internal_layer=1, kv_type="f16")

    completed = subprocess.run(
        [
            str(QX_EXE),
            "attention-stage-probe",
            "--in",
            str(QXF),
            "--layer",
            "1",
            "--layer-in",
            str(oracle_dir / "layer-1.f32"),
            "--out-dir",
            str(qx_dir),
            "--activation",
            "q8_k_compat",
            "--kv",
            "f16",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["projection_kernel"] == "q5_k_q8_k"
    assert payload["v_projection_kernel"] == "q5_k_q8_k"
    assert payload["output_projection_kernel"] == "q5_k_q8_k"
    assert payload["kv_format"] == "f16"
    assert payload["single_token_softmax"] == 1.0
    assert (qx_dir / "attn_norm-1.f32").stat().st_size == 2048 * 4
    assert (qx_dir / "Vcur-1.f32").stat().st_size == 512 * 4
    assert (qx_dir / "kqv_out-1.f32").stat().st_size == 4096 * 4
    assert (qx_dir / "attn_out-1.f32").stat().st_size == 2048 * 4
    assert (qx_dir / "ffn_inp-1.f32").stat().st_size == 2048 * 4
    assert max_abs(read_f32(qx_dir / "Vcur-1.f32"), read_f32(oracle_dir / "Vcur-1.f32")) <= 1e-5
    assert max_abs(read_f32(qx_dir / "kqv_out-1.f32"), read_f32(oracle_dir / "kqv_out-1.f32")) <= 1e-5
    assert max_abs(read_f32(qx_dir / "ffn_inp-1.f32"), read_f32(oracle_dir / "ffn_inp-1.f32")) <= 1e-4


def test_attention_stage_q8_k_closes_layer_46_q6_k_output_projection(tmp_path):
    require_local_moe_fixtures()
    oracle_dir = tmp_path / "oracle-layer-46"
    qx_dir = tmp_path / "qx-attention-layer-46"
    qx_dir.mkdir()
    run_oracle(oracle_dir, internal_layer=46, kv_type="f16")

    completed = subprocess.run(
        [
            str(QX_EXE),
            "attention-stage-probe",
            "--in",
            str(QXF),
            "--layer",
            "46",
            "--layer-in",
            str(oracle_dir / "layer-46.f32"),
            "--out-dir",
            str(qx_dir),
            "--activation",
            "q8_k_compat",
            "--kv",
            "f16",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["v_projection_kernel"] == "q5_k_q8_k"
    assert payload["output_projection_kernel"] == "q6_k_q8_k"
    assert payload["projection_kernel"] == "q5_k_q6_k_q8_k"

    env = os.environ.copy()
    env["LLAMA_CPP_DIR"] = str(LLAMA_CPP_DIR)
    built = subprocess.run(
        ["cmd.exe", "/c", str(GGML_BUILD)], cwd=ROOT, env=env, text=True, capture_output=True
    )
    assert built.returncode == 0, built.stdout + built.stderr
    tensor = json.loads(
        subprocess.check_output(
            [
                str(QX_EXE),
                "inspect-tensor",
                "--in",
                str(QXF),
                "--name",
                "blk.46.attn_output.weight",
            ],
            text=True,
        )
    )
    assert tensor["ggml_type"] == 14
    blocks = tensor["dims"][0] // 256
    row_bytes = blocks * 210
    qx_attention = read_f32(qx_dir / "attn_out-46.f32")
    for row in (0, 1024, 2047):
        reference = subprocess.run(
            [
                str(GGML_EXE),
                "q6_k_q8_k_dot",
                str(QXF),
                str(tensor["offset"] + row * row_bytes),
                str(blocks),
                str(qx_dir / "kqv_out-46.f32"),
            ],
            capture_output=True,
        )
        assert reference.returncode == 0, reference.stderr.decode(errors="replace")
        expected = struct.unpack("<f", reference.stdout)[0]
        assert math.isclose(qx_attention[row], expected, rel_tol=2e-6, abs_tol=2e-6)

    assert max_abs(read_f32(qx_dir / "ffn_inp-46.f32"), read_f32(oracle_dir / "ffn_inp-46.f32")) <= 2e-6


@pytest.mark.parametrize(
    ("case", "values", "layer", "create_output", "expected_error"),
    [
        ("short", [0.0] * 2047, 0, True, "size or read mismatch"),
        ("extra", [0.0] * 2049, 0, True, "size or read mismatch"),
        ("nan", [math.nan] + [0.0] * 2047, 0, True, "non-finite"),
        ("inf", [math.inf] + [0.0] * 2047, 0, True, "non-finite"),
        ("invalid-layer", [0.0] * 2048, 48, True, "invalid MoE stage layer"),
        ("missing-output-dir", [0.0] * 2048, 0, False, "cannot open MoE sidecar"),
    ],
)
def test_moe_stage_probe_fails_closed_for_invalid_inputs(
    tmp_path, case, values, layer, create_output, expected_error
):
    require_local_moe_fixtures()
    sidecar = tmp_path / f"{case}.f32"
    sidecar.write_bytes(struct.pack(f"<{len(values)}f", *values))
    output = tmp_path / f"out-{case}"
    if create_output:
        output.mkdir()
    completed = subprocess.run(
        [
            str(QX_EXE),
            "moe-stage-probe",
            "--in",
            str(QXF),
            "--layer",
            str(layer),
            "--ffn-inp",
            str(sidecar),
            "--out-dir",
            str(output),
            "--activation",
            "q8_k_compat",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert expected_error in completed.stderr


@pytest.mark.parametrize(
    ("layer", "expected_types", "expected_row_bytes"),
    [
        (0, (17, 17, 18), (592, 592, 294)),
        (1, (22, 22, 23), (656, 656, 408)),
        (24, (17, 17, 21), (592, 592, 330)),
        (47, (22, 22, 23), (656, 656, 408)),
    ],
)
def test_real_moe_layers_preserve_quant_types_and_expert_row_strides(
    layer, expected_types, expected_row_bytes
):
    require_local_moe_fixtures()
    tensors = []
    for role in ("gate", "up", "down"):
        payload = json.loads(
            subprocess.check_output(
                [
                    str(QX_EXE),
                    "inspect-tensor",
                    "--in",
                    str(QXF),
                    "--name",
                    f"blk.{layer}.ffn_{role}_exps.weight",
                ],
                text=True,
            )
        )
        tensors.append(payload)
    assert tuple(tensor["ggml_type"] for tensor in tensors) == expected_types
    assert tuple(
        tensor["byte_size"] // tensor["dims"][2] // tensor["dims"][1] for tensor in tensors
    ) == expected_row_bytes


@pytest.mark.parametrize(
    ("layer", "expected_kernel", "expected_gate_up_kernel", "expected_down_kernel"),
    [
        (1, "iq2_s_q8_k_and_iq4_xs_q8_k", "iq2_s_q8_k", "iq4_xs_q8_k"),
        (24, "iq2_xs_q8_k_and_iq3_s_q8_k", "iq2_xs_q8_k", "iq3_s_q8_k"),
        (47, "iq2_s_q8_k_and_iq4_xs_q8_k", "iq2_s_q8_k", "iq4_xs_q8_k"),
    ],
)
def test_moe_stage_q8_k_reports_actual_fallback_for_heterogeneous_layers(
    tmp_path, layer, expected_kernel, expected_gate_up_kernel, expected_down_kernel
):
    require_local_moe_fixtures()
    sidecar = tmp_path / "ffn-inp.f32"
    sidecar.write_bytes(struct.pack("<2048f", *([0.0] * 2048)))
    output = tmp_path / "out"
    output.mkdir()
    completed = subprocess.run(
        [
            str(QX_EXE),
            "moe-stage-probe",
            "--in",
            str(QXF),
            "--layer",
            str(layer),
            "--ffn-inp",
            str(sidecar),
            "--out-dir",
            str(output),
            "--activation",
            "q8_k_compat",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["projection_kernel"] == expected_kernel
    assert payload["gate_up_projection_kernel"] == expected_gate_up_kernel
    assert payload["down_projection_kernel"] == expected_down_kernel


def test_layer_47_same_input_attention_and_moe_chain_closes(tmp_path):
    require_local_moe_fixtures()
    oracle_dir = tmp_path / "oracle-layer-47"
    attention_dir = tmp_path / "attention-layer-47"
    moe_dir = tmp_path / "moe-layer-47"
    attention_dir.mkdir()
    moe_dir.mkdir()
    run_oracle(oracle_dir, internal_layer=47, kv_type="f16")

    attention = subprocess.run(
        [
            str(QX_EXE),
            "attention-stage-probe",
            "--in",
            str(QXF),
            "--layer",
            "47",
            "--layer-in",
            str(oracle_dir / "layer-47.f32"),
            "--out-dir",
            str(attention_dir),
            "--activation",
            "q8_k_compat",
            "--kv",
            "f16",
        ],
        text=True,
        capture_output=True,
    )
    assert attention.returncode == 0, attention.stdout + attention.stderr
    attention_payload = json.loads(attention.stdout)
    assert attention_payload["v_projection_kernel"] == "q5_k_q8_k"
    assert attention_payload["output_projection_kernel"] == "q6_k_q8_k"
    assert attention_payload["kv_format"] == "f16"

    moe = subprocess.run(
        [
            str(QX_EXE),
            "moe-stage-probe",
            "--in",
            str(QXF),
            "--layer",
            "47",
            "--ffn-inp",
            str(attention_dir / "ffn_inp-47.f32"),
            "--out-dir",
            str(moe_dir),
            "--activation",
            "q8_k_compat",
        ],
        text=True,
        capture_output=True,
    )
    assert moe.returncode == 0, moe.stdout + moe.stderr
    moe_payload = json.loads(moe.stdout)
    assert moe_payload["gate_up_projection_kernel"] == "iq2_s_q8_k"
    assert moe_payload["down_projection_kernel"] == "iq4_xs_q8_k"
    assert moe_payload["selected_experts"] == [83, 3, 74, 119, 92, 28, 109, 101]

    compared = subprocess.run(
        [
            sys.executable,
            str(COMPARE_LAYER),
            "--oracle-dir",
            str(oracle_dir),
            "--attention-dir",
            str(attention_dir),
            "--same-input-moe-dir",
            str(moe_dir),
            "--expected-vcur-count",
            "512",
            "--expected-kqv-out-count",
            "4096",
            "--layer",
            "47",
        ],
        text=True,
        capture_output=True,
    )
    assert compared.returncode == 0, compared.stdout + compared.stderr
    payload = json.loads(compared.stdout)
    assert payload["routing"] == {
        "oracle": [83, 3, 74, 119, 92, 28, 109, 101],
        "qx": [83, 3, 74, 119, 92, 28, 109, 101],
        "exact": True,
    }
    assert payload["checkpoints"]["attention"]["Vcur"]["max_abs"] <= 2e-6
    assert payload["checkpoints"]["attention"]["kqv_out"]["max_abs"] <= 1.5e-4
    assert payload["checkpoints"]["attention"]["ffn_input"]["max_abs"] <= 1e-4
    assert payload["checkpoints"]["moe"]["router_logits"]["max_abs"] <= 2e-6
    assert payload["checkpoints"]["moe"]["weights_norm"]["max_abs"] <= 3e-7
    assert payload["checkpoints"]["moe"]["down"]["max_abs"] <= 2e-4
    assert payload["checkpoints"]["moe"]["weighted"]["max_abs"] <= 2e-4
    assert payload["reconstruction"]["moe_output"]["max_abs"] <= 3e-4
    assert payload["reconstruction"]["moe_output"]["rmse"] <= 1e-5
    assert payload["reconstruction"]["layer_output"]["max_abs"] <= 3e-4
    assert payload["reconstruction"]["layer_output"]["rmse"] <= 1e-5


@pytest.mark.parametrize(
    (
        "layer",
        "v_kernel",
        "output_kernel",
        "gate_up_kernel",
        "down_kernel",
        "routing",
        "router_max_abs",
        "down_max_abs",
        "weighted_max_abs",
        "layer_output_max_abs",
        "layer_output_rmse",
    ),
    [
        (24, "iq4_xs_q8_k", "iq4_xs_q8_k", "iq2_xs_q8_k", "iq3_s_q8_k",
         [10, 105, 24, 111, 101, 98, 108, 113], 4.76837158203125e-06,
         2.384185791015625e-07, 1.7881393432617188e-07,
         4.5750217395834625e-05, 1.0114135290963168e-06),
        (41, "iq4_xs_q8_k", "q6_k_q8_k", "iq2_s_q8_k", "iq3_s_q8_k",
         [48, 73, 69, 18, 96, 104, 88, 26], 2.86102294921875e-06,
         9.5367431640625e-07, 8.940696716308594e-08,
         4.762341268360615e-05, 1.0528728896676105e-06),
        (42, "iq4_xs_q8_k", "iq4_xs_q8_k", "iq2_s_q8_k", "iq4_xs_q8_k",
         [110, 64, 17, 69, 21, 41, 116, 25], 3.337860107421875e-06,
         None, 1.1920928955078125e-07, 8.474162314087152e-07, 4.0049851967789266e-08),
        (43, "iq4_xs_q8_k", "iq4_xs_q8_k", "iq2_s_q8_k", "iq4_xs_q8_k",
         [78, 82, 83, 63, 90, 100, 74, 28], 2.86102294921875e-06,
         None, 3.5762786865234375e-07, 4.6805653255432844e-05, 1.0357114130980166e-06),
        (44, "iq4_xs_q8_k", "q6_k_q8_k", "iq2_s_q8_k", "iq4_xs_q8_k",
         [40, 113, 104, 41, 72, 83, 73, 102], 2.384185791015625e-06,
         None, 2.9802322387695312e-08, 7.81775452196598e-06, 1.7660727227129046e-07),
    ],
)
def test_backward_bisect_same_input_layers_close_with_reproducible_metrics(
    tmp_path,
    layer,
    v_kernel,
    output_kernel,
    gate_up_kernel,
    down_kernel,
    routing,
    router_max_abs,
    down_max_abs,
    weighted_max_abs,
    layer_output_max_abs,
    layer_output_rmse,
):
    require_local_moe_fixtures()
    oracle_dir = tmp_path / f"oracle-layer-{layer}"
    attention_dir = tmp_path / f"attention-layer-{layer}"
    moe_dir = tmp_path / f"moe-layer-{layer}"
    attention_dir.mkdir()
    moe_dir.mkdir()
    run_oracle(oracle_dir, internal_layer=layer, kv_type="f16")

    attention = subprocess.run(
        [
            str(QX_EXE),
            "attention-stage-probe",
            "--in",
            str(QXF),
            "--layer",
            str(layer),
            "--layer-in",
            str(oracle_dir / f"layer-{layer}.f32"),
            "--out-dir",
            str(attention_dir),
            "--activation",
            "q8_k_compat",
            "--kv",
            "f16",
        ],
        text=True,
        capture_output=True,
    )
    assert attention.returncode == 0, attention.stdout + attention.stderr
    attention_payload = json.loads(attention.stdout)
    assert attention_payload["v_projection_kernel"] == v_kernel
    assert attention_payload["output_projection_kernel"] == output_kernel
    assert attention_payload["kv_format"] == "f16"

    moe = subprocess.run(
        [
            str(QX_EXE),
            "moe-stage-probe",
            "--in",
            str(QXF),
            "--layer",
            str(layer),
            "--ffn-inp",
            str(attention_dir / f"ffn_inp-{layer}.f32"),
            "--out-dir",
            str(moe_dir),
            "--activation",
            "q8_k_compat",
        ],
        text=True,
        capture_output=True,
    )
    assert moe.returncode == 0, moe.stdout + moe.stderr
    moe_payload = json.loads(moe.stdout)
    assert moe_payload["gate_up_projection_kernel"] == gate_up_kernel
    assert moe_payload["down_projection_kernel"] == down_kernel
    assert moe_payload["selected_experts"] == routing

    compared = subprocess.run(
        [
            sys.executable,
            str(COMPARE_LAYER),
            "--oracle-dir",
            str(oracle_dir),
            "--attention-dir",
            str(attention_dir),
            "--same-input-moe-dir",
            str(moe_dir),
            "--expected-vcur-count",
            "512",
            "--expected-kqv-out-count",
            "4096",
            "--layer",
            str(layer),
        ],
        text=True,
        capture_output=True,
    )
    assert compared.returncode == 0, compared.stdout + compared.stderr
    payload = json.loads(compared.stdout)
    assert payload["routing"] == {
        "oracle": routing,
        "qx": routing,
        "exact": True,
    }
    assert payload["checkpoints"]["attention"]["Vcur"]["max_abs"] <= 2e-6
    assert payload["checkpoints"]["attention"]["kqv_out"]["max_abs"] <= 1.5e-4
    assert payload["checkpoints"]["attention"]["ffn_input"]["max_abs"] <= 1e-4
    assert payload["checkpoints"]["moe"]["router_logits"]["max_abs"] == pytest.approx(
        router_max_abs, rel=0, abs=1e-12
    )
    assert payload["checkpoints"]["moe"]["weights_norm"]["max_abs"] <= 3e-7
    if down_max_abs is None:
        assert payload["checkpoints"]["moe"]["down"]["max_abs"] <= 2e-4
    else:
        assert payload["checkpoints"]["moe"]["down"]["max_abs"] == pytest.approx(
            down_max_abs, rel=0, abs=1e-12
        )
    assert payload["checkpoints"]["moe"]["weighted"]["max_abs"] == pytest.approx(
        weighted_max_abs, rel=0, abs=1e-12
    )
    assert payload["reconstruction"]["moe_output"]["max_abs"] <= 3e-4
    assert payload["reconstruction"]["moe_output"]["rmse"] <= 1e-5
    assert payload["reconstruction"]["layer_output"]["max_abs"] == pytest.approx(
        layer_output_max_abs, rel=0, abs=1e-12
    )
    assert payload["reconstruction"]["layer_output"]["rmse"] == pytest.approx(
        layer_output_rmse, rel=0, abs=1e-12
    )
