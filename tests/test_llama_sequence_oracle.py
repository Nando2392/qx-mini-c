import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "Qwen3-30B-A3B-UD-IQ2_M.gguf"
QXF = ROOT / "models" / "Qwen3-30B-A3B-UD-IQ2_M.qxf"
QXT = ROOT / "models" / "Qwen3-30B-A3B.qxt"
QX_EXE = ROOT / "build" / "qxqxf.exe"
LLAMA_CPP_DIR = ROOT.parent / "llama.cpp-k3"
BUILD = ROOT / "tests" / "build_llama_reference_oracle.bat"
EXE = ROOT / "build" / "llama_sequence_oracle.exe"


def test_llama_sequence_oracle_runs_fixed_token_and_hello_ids():
    if not MODEL.exists() or not LLAMA_CPP_DIR.exists():
        pytest.skip("real llama.cpp sequence fixtures are not available")

    env = os.environ.copy()
    env["LLAMA_CPP_DIR"] = str(LLAMA_CPP_DIR)
    build = subprocess.run(
        ["cmd.exe", "/c", str(BUILD)], cwd=ROOT, env=env, text=True, capture_output=True
    )
    assert build.returncode == 0, build.stdout + build.stderr
    assert EXE.is_file()

    cases = (("42", [42]), ("9707,0", [9707, 0]))
    for token_csv, prompt_tokens in cases:
        result = subprocess.run(
            [str(EXE), str(MODEL), token_csv, "2", "f16"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert payload["llama_commit"] != "unknown"
        assert payload["kv_type"] == "f16"
        assert payload["prompt_tokens"] == prompt_tokens
        assert payload["generation_steps"] == 2
        assert len(payload["generated_tokens"]) == 2
        assert all(0 <= token < payload["n_vocab"] for token in payload["generated_tokens"])


def test_qx_and_llama_sequence_comparison_is_explicit(tmp_path):
    if not all(path.exists() for path in (MODEL, QXF, QXT, QX_EXE, EXE)):
        pytest.skip("real QX/llama.cpp sequence fixtures are not available")

    def llama(tokens, kv):
        return json.loads(subprocess.check_output(
            [str(EXE), str(MODEL), tokens, "2", kv], text=True
        ))["generated_tokens"]

    llama_42_f16 = llama("42", "f16")
    llama_42_q8 = llama("42", "q8_0")
    llama_hello_f16 = llama("9707,0", "f16")
    llama_hello_q8 = llama("9707,0", "q8_0")
    assert llama_42_f16 == [1124, 50853]
    assert llama_42_q8 == [1124, 50853]
    assert llama_hello_f16 == [358, 1184]
    assert llama_hello_q8 == [358, 614]

    qx_42 = json.loads(subprocess.check_output([
        str(QX_EXE), "state-loop-probe", "--in", str(QXF), "--prompt-token", "42",
        "--steps", "2", "--layers", "48", "--ctx", "4", "--kv", "int8",
        "--temperature", "0", "--seed", "7", "--full-moe", "--final-head", "--top-n", "5",
    ], text=True))
    qx_42_generated = [step["selected_token"] for step in qx_42["tokens"]]

    prompt = tmp_path / "hello.txt"
    prompt.write_text("Hello!", encoding="utf-8")
    qx_hello = json.loads(subprocess.check_output([
        str(QX_EXE), "prompt-state-loop-probe", "--in", str(QXF), "--tokenizer", str(QXT),
        "--text-file", str(prompt), "--generate", "2", "--layers", "48", "--ctx", "4",
        "--kv", "int8", "--temperature", "0", "--seed", "7", "--full-moe", "--final-head", "--top-n", "5",
    ], text=True))
    qx_hello_generated = [step["selected_token"] for step in qx_hello["tokens"] if step["phase"] == "generate"]

    assert qx_42_generated == [1124, 50853]
    assert qx_hello["prompt_token_ids"] == [9707, 0]
    assert qx_hello_generated == [358, 1184]
    assert qx_42_generated == llama_42_f16 == llama_42_q8
    assert qx_hello_generated == llama_hello_f16
    assert qx_hello_generated[0] == llama_hello_q8[0]
    assert qx_hello_generated[1] != llama_hello_q8[1]
