import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GGUF = ROOT / "models" / "Qwen3-30B-A3B-UD-IQ2_M.gguf"
QXF = ROOT / "models" / "Qwen3-30B-A3B-UD-IQ2_M.qxf"
QX_EXE = ROOT / "build" / "qxqxf.exe"
LLAMA_EXE = ROOT / "build" / "llama_reference_oracle.exe"
LLAMA_CPP_DIR = ROOT.parent / "llama.cpp-k3"
BUILD = ROOT / "tests" / "build_llama_reference_oracle.bat"
COMPARE_RESIDUALS = ROOT / "scripts" / "compare_residuals.py"
COMPARE_LOGITS = ROOT / "scripts" / "compare_logits.py"


def run_json(command, expected_returncode=0):
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == expected_returncode, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_real_end_to_end_parity_is_reproducibly_refuted(tmp_path):
    if not all(path.exists() for path in (GGUF, QXF, QX_EXE, LLAMA_CPP_DIR)):
        pytest.skip("real QX/llama.cpp parity fixtures are not available")

    env = os.environ.copy()
    env["LLAMA_CPP_DIR"] = str(LLAMA_CPP_DIR)
    build = subprocess.run(
        ["cmd.exe", "/c", str(BUILD)], cwd=ROOT, env=env, text=True, capture_output=True
    )
    assert build.returncode == 0, build.stdout + build.stderr
    assert LLAMA_EXE.is_file()

    for qx_kv, llama_kv in (("f32", "f16"), ("int8", "q8_0")):
        qx_dir = tmp_path / f"qx-{qx_kv}"
        llama_dir = tmp_path / f"llama-{llama_kv}"
        qx_dir.mkdir()
        llama_dir.mkdir()

        oracle = run_json([
            str(LLAMA_EXE), str(GGUF), str(llama_dir), "42", "0,1,24,47", llama_kv, "internals",
        ])
        assert oracle["internals_captured"] == 18
        assert oracle["logits"]["argmax"] == 1124

        qx = run_json([
            str(QX_EXE), "state-loop-probe", "--in", str(QXF), "--prompt-token", "42",
            "--steps", "1", "--layers", "48", "--ctx", "4", "--kv", qx_kv,
            "--temperature", "0", "--seed", "7", "--full-moe", "--final-head",
            "--top-n", "32", "--dump-residuals", str(qx_dir),
        ])
        assert qx["tokens"][0]["selected_token"] == 1124

        inputs = run_json([
            sys.executable, str(COMPARE_RESIDUALS), "--qx-dir", str(qx_dir),
            "--llama-dir", str(llama_dir), "--layers", "0,1,24,47", "--phase", "input",
        ], expected_returncode=1)
        assert inputs["first_divergent_layer"] == 1
        assert inputs["layers"][0]["max_abs"] == 0
        assert inputs["layers"][0]["cosine"] == 1

        pre_head = run_json([
            sys.executable, str(COMPARE_RESIDUALS), "--qx-dir", str(qx_dir),
            "--llama-dir", str(llama_dir), "--layers", "47", "--phase", "output",
        ], expected_returncode=1)
        assert pre_head["first_divergent_layer"] == 47
        assert pre_head["layers"][0]["max_abs"] > 100
        assert pre_head["layers"][0]["cosine"] < 0.5

        logits = run_json([
            sys.executable, str(COMPARE_LOGITS), "--qx", str(qx_dir / "step-0-logits.f32"),
            "--llama", str(llama_dir / "logits.f32"), "--max-abs", "0.1",
            "--rmse", "0.1", "--min-cosine", "0.99",
        ], expected_returncode=1)
        assert logits["count"] == 151936
        assert logits["argmax_match"] is True
        assert logits["pass"] is False
