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
LLAMA_CPP_DIR = ROOT.parent / "llama.cpp-k3"
BUILD_ORACLE = ROOT / "tests" / "build_llama_reference_oracle.bat"


def read_f32(path: Path) -> tuple[float, ...]:
    raw = path.read_bytes()
    assert raw and len(raw) % 4 == 0
    values = struct.unpack(f"<{len(raw) // 4}f", raw)
    assert all(math.isfinite(value) for value in values)
    return values


def metrics(actual: tuple[float, ...], expected: tuple[float, ...]) -> dict[str, float]:
    assert len(actual) == len(expected)
    deltas = [float(got) - float(want) for got, want in zip(actual, expected)]
    actual_l2 = sum(float(value) * float(value) for value in actual)
    expected_l2 = sum(float(value) * float(value) for value in expected)
    dot = sum(float(got) * float(want) for got, want in zip(actual, expected))
    return {
        "max_abs": max(map(abs, deltas)),
        "rmse": math.sqrt(sum(delta * delta for delta in deltas) / len(deltas)),
        "cosine": dot / math.sqrt(actual_l2 * expected_l2),
    }


def test_final_head_probe_q8_k_matches_same_input_llama_logits(tmp_path):
    if not all(path.exists() for path in (QX_EXE, QXF, GGUF, LLAMA_CPP_DIR)):
        pytest.skip("real QX/llama.cpp final-head fixtures are not available")

    env = os.environ.copy()
    env["LLAMA_CPP_DIR"] = str(LLAMA_CPP_DIR)
    build = subprocess.run(
        ["cmd.exe", "/c", str(BUILD_ORACLE)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr

    oracle_dir = tmp_path / "oracle"
    qx_dir = tmp_path / "qx"
    oracle_dir.mkdir()
    qx_dir.mkdir()
    oracle = subprocess.run(
        [str(LLAMA_EXE), str(GGUF), str(oracle_dir), "42", "47", "f16", "internals=47"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert oracle.returncode == 0, oracle.stdout + oracle.stderr

    probe = subprocess.run(
        [
            str(QX_EXE),
            "final-head-probe",
            "--in",
            str(QXF),
            "--residual",
            str(oracle_dir / "l_out-47.f32"),
            "--out-dir",
            str(qx_dir),
            "--activation",
            "q8_k_compat",
            "--top-n",
            "5",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    payload = json.loads(probe.stdout)

    assert payload["input_dims"] == 2048
    assert payload["vocab_size"] == 151936
    assert payload["activation_mode"] == "q8_k_compat"
    assert payload["lm_head_kernel"] == "q6_k_q8_k"
    assert payload["activation_quantizations"] == 1
    assert payload["argmax_token"] == 1124
    assert (qx_dir / "final-norm.f32").stat().st_size == 2048 * 4
    assert (qx_dir / "logits.f32").stat().st_size == 151936 * 4

    norm_metrics = metrics(read_f32(qx_dir / "final-norm.f32"), read_f32(oracle_dir / "result_norm.f32"))
    logits_metrics = metrics(read_f32(qx_dir / "logits.f32"), read_f32(oracle_dir / "logits.f32"))
    assert norm_metrics["max_abs"] <= 4e-6
    assert norm_metrics["cosine"] >= 0.999999999999
    assert logits_metrics["max_abs"] <= 3e-6
    assert logits_metrics["rmse"] <= 6e-7
    assert logits_metrics["cosine"] >= 0.999999999999


@pytest.mark.parametrize(
    "values",
    (
        [0.0] * 2047,
        [0.0] * 2049,
        [math.nan] + [0.0] * 2047,
    ),
)
def test_final_head_probe_rejects_invalid_residual_sidecars(tmp_path, values):
    if not QX_EXE.exists() or not QXF.exists():
        pytest.skip("real QX final-head fixtures are not available")

    residual = tmp_path / "residual.f32"
    output = tmp_path / "output"
    output.mkdir()
    residual.write_bytes(struct.pack(f"<{len(values)}f", *values))
    probe = subprocess.run(
        [
            str(QX_EXE),
            "final-head-probe",
            "--in",
            str(QXF),
            "--residual",
            str(residual),
            "--out-dir",
            str(output),
            "--activation",
            "q8_k_compat",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert probe.returncode == 1
    assert "F32 sidecar" in probe.stderr
    assert list(output.iterdir()) == []


def test_final_head_probe_keeps_f32_as_default(tmp_path):
    if not QX_EXE.exists() or not QXF.exists():
        pytest.skip("real QX final-head fixtures are not available")

    residual = tmp_path / "residual.f32"
    output = tmp_path / "output"
    output.mkdir()
    residual.write_bytes(struct.pack("<2048f", *([0.0] * 2048)))
    probe = subprocess.run(
        [
            str(QX_EXE),
            "final-head-probe",
            "--in",
            str(QXF),
            "--residual",
            str(residual),
            "--out-dir",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert probe.returncode == 0, probe.stdout + probe.stderr
    payload = json.loads(probe.stdout)
    assert payload["activation_mode"] == "f32"
    assert payload["lm_head_kernel"] == "dequant_f32"
    assert payload["activation_quantizations"] == 0


def test_final_head_probe_removes_partial_outputs_when_second_write_fails(tmp_path):
    if not QX_EXE.exists() or not QXF.exists():
        pytest.skip("real QX final-head fixtures are not available")

    residual = tmp_path / "residual.f32"
    output = tmp_path / "output"
    output.mkdir()
    (output / "logits.f32").mkdir()
    residual.write_bytes(struct.pack("<2048f", *([0.0] * 2048)))
    probe = subprocess.run(
        [
            str(QX_EXE),
            "final-head-probe",
            "--in",
            str(QXF),
            "--residual",
            str(residual),
            "--out-dir",
            str(output),
            "--activation",
            "q8_k_compat",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert probe.returncode == 1
    assert "cannot open final head sidecar" in probe.stderr
    assert not (output / "final-norm.f32").exists()
    assert (output / "logits.f32").is_dir()
