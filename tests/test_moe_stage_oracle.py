import json
import math
import os
import struct
import subprocess
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


def require_local_moe_fixtures():
    if os.name != "nt" or not QX_EXE.exists() or not QXF.exists() or not GGUF.exists():
        pytest.skip("local Windows Qwen MoE fixtures are not available")
    if not (LLAMA_CPP_DIR / "include" / "llama.h").exists():
        pytest.skip("pinned llama.cpp oracle checkout is not available")


def run_oracle(output: Path):
    env = os.environ.copy()
    env["LLAMA_CPP_DIR"] = str(LLAMA_CPP_DIR)
    build = subprocess.run(
        ["cmd.exe", "/c", str(LLAMA_BUILD)], cwd=ROOT, env=env, text=True, capture_output=True
    )
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run(
        [str(LLAMA_EXE), str(GGUF), str(output), "42", "0", "f16", "internals"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


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
    assert oracle["internals_captured"] == 18

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
    ("tensor_name", "quant_name", "block_size", "rows", "activation_name", "activation_count"),
    [
        ("blk.0.ffn_gate_exps.weight", "iq2_xs", 74, (0, 384, 767), "ffn_norm-0.f32", 2048),
        ("blk.0.ffn_down_exps.weight", "iq3_xxs", 98, (0, 1024, 2047), "ffn_moe_swiglu-0.f32", 768),
    ],
)
def test_q8_k_expert_dot_matches_pinned_ggml_kernel(
    tmp_path, tensor_name, quant_name, block_size, rows, activation_name, activation_count
):
    require_local_moe_fixtures()
    oracle_dir = tmp_path / "oracle"
    run_oracle(oracle_dir)
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
    expert = 49
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
    ("layer", "expected_kernel"),
    [
        (1, "dequant_f32"),
        (24, "iq2_xs_iq3_xxs_q8_k_with_f32_fallback"),
        (47, "dequant_f32"),
    ],
)
def test_moe_stage_q8_k_reports_actual_fallback_for_heterogeneous_layers(
    tmp_path, layer, expected_kernel
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
    assert json.loads(completed.stdout)["projection_kernel"] == expected_kernel
