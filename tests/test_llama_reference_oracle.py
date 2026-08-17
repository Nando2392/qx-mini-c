import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "tests" / "build_llama_reference_oracle.bat"
EXE = ROOT / "build" / "llama_reference_oracle.exe"
MODEL = ROOT / "models" / "Qwen3-30B-A3B-UD-IQ2_M.gguf"
LLAMA_CPP_DIR = Path(os.environ.get("LLAMA_CPP_DIR", ROOT.parent / "llama.cpp-k3"))


def test_llama_reference_oracle_rebuilds_and_runs_standalone(tmp_path):
    if os.name != "nt" or not MODEL.exists() or not (LLAMA_CPP_DIR / "include" / "llama.h").exists():
        pytest.skip("local Windows llama.cpp oracle fixtures are not available")

    env = os.environ.copy()
    env["LLAMA_CPP_DIR"] = str(LLAMA_CPP_DIR)
    build = subprocess.run(
        ["cmd.exe", "/c", str(BUILD)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr

    for dll in ("llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll"):
        assert (ROOT / "build" / dll).is_file()

    for kv_type in ("f16", "q8_0"):
        output = tmp_path / kv_type
        result = subprocess.run(
            [str(EXE), str(MODEL), str(output), "42", "0,1", kv_type, "internals"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert payload["llama_commit"] != "unknown"
        assert payload["kv_type"] == kv_type
        assert payload["token_id"] == 42
        assert payload["n_embd"] == 2048
        assert payload["n_layer"] == 48
        assert payload["n_vocab"] == 151936
        assert payload["logits"]["argmax"] == 1124
        assert payload["internals_captured"] == 6
        assert (output / "layer-0.f32").stat().st_size == 2048 * 4
        assert (output / "layer-1.f32").stat().st_size == 2048 * 4
        assert (output / "logits.f32").stat().st_size == 151936 * 4
        for name in ("ffn_inp-0", "ffn_moe_out-0", "l_out-0"):
            assert (output / f"{name}.f32").stat().st_size == 2048 * 4
        assert (output / "Vcur-0.f32").stat().st_size == 512 * 4
        assert (output / "kqv_out-0.f32").stat().st_size == 4096 * 4
        assert (output / "l_out-47.f32").stat().st_size == 2048 * 4
