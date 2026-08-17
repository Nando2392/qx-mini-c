import json
import os
import struct
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / "qxqxf.exe"
MODEL = ROOT / "models" / "Qwen3-30B-A3B-UD-IQ2_M.qxf"
GGML_BUILD = ROOT / "tests" / "build_ggml_reference.bat"
GGML_EXE = ROOT / "build" / "ggml_reference_decode.exe"
LLAMA_CPP_DIR = Path(os.environ.get("LLAMA_CPP_DIR", ROOT.parent / "llama.cpp-k3"))


def run_probe(*args):
    return subprocess.run(
        [str(EXE), "q8-k-activation-probe", *args],
        text=True,
        capture_output=True,
    )


def test_q8_k_activation_probe_reports_ggml_layout_and_is_deterministic():
    first = run_probe("--values", "256")
    second = run_probe("--values", "256")
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    payload = json.loads(first.stdout)
    assert payload["activation_format"] == "q8_k_compat"
    assert payload["values"] == 256
    assert payload["blocks"] == 1
    assert payload["block_bytes"] == 292
    assert payload["workspace_bytes"] == 292
    assert payload["finite"] is True
    assert payload["checksum"] == json.loads(second.stdout)["checksum"]


@pytest.mark.parametrize("inject", ["none", "positive", "negative", "edge"])
def test_q8_k_activation_bytes_match_pinned_ggml_oracle(tmp_path, inject):
    if os.name != "nt" or not (LLAMA_CPP_DIR / "ggml" / "src" / "ggml-quants.h").exists():
        pytest.skip("local Windows ggml oracle is not available")
    payload = json.loads(run_probe("--values", "256", "--inject", inject).stdout)
    qx_bytes = bytes.fromhex(payload["block_hex"])
    assert len(qx_bytes) == 292
    if inject == "positive":
        values = [(index % 127) / 127.0 for index in range(256)]
    elif inject == "negative":
        values = [-(index % 127) / 127.0 for index in range(256)]
    elif inject == "edge":
        values = [-1.0 if index % 2 == 0 else 1.0 for index in range(256)]
    else:
        values = [(index % 251 - 125) / 127.0 for index in range(256)]
    source = tmp_path / "activation.f32"
    source.write_bytes(struct.pack("<256f", *values))
    env = os.environ.copy()
    env["LLAMA_CPP_DIR"] = str(LLAMA_CPP_DIR)
    built = subprocess.run(["cmd.exe", "/c", str(GGML_BUILD)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert built.returncode == 0, built.stdout + built.stderr
    oracle = subprocess.run([str(GGML_EXE), "q8_k_quantize", str(source)], cwd=ROOT, capture_output=True)
    assert oracle.returncode == 0, oracle.stderr.decode(errors="replace")
    assert qx_bytes == oracle.stdout


def test_q8_k_metadata_reports_not_used_without_projection_path():
    if not MODEL.exists():
        pytest.skip("real Qwen runtime fixture is not available")
    result = subprocess.run(
        [str(EXE), "state-loop-probe", "--in", str(MODEL), "--prompt-token", "42", "--steps", "1", "--layers", "1", "--ctx", "4", "--kv", "int8", "--activation", "q8_k_compat"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["projection_kernel"] == "not_used"
    assert payload["activation_workspace_bytes"] == 0


def test_q8_k_activation_probe_rejects_non_finite_input_and_recovers():
    for injected in ("nan", "inf"):
        failed = run_probe("--values", "256", "--inject", injected)
        assert failed.returncode == 1
        assert "contains NaN or Inf" in failed.stderr
    recovered = run_probe("--values", "256")
    assert recovered.returncode == 0, recovered.stderr


def test_q8_k_activation_probe_quantizes_zero_block_without_division_by_zero():
    result = run_probe("--values", "256", "--inject", "zero")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["first_scale"] == 0.0
    assert payload["quant_sum"] == 0
    assert payload["block_bytes"] == 292


def test_q8_k_activation_probe_rejects_partial_workspace_and_overflow_sizes():
    for values in ("255", "257"):
        partial = run_probe("--values", values)
        assert partial.returncode == 1
        assert "non-zero multiple of 256" in partial.stderr
    insufficient = run_probe("--values", "4352")
    assert insufficient.returncode == 1
    assert "workspace is too small" in insufficient.stderr
    overflow = run_probe("--values", "4294967296")
    assert overflow.returncode == 2
    assert "invalid --values" in overflow.stderr
