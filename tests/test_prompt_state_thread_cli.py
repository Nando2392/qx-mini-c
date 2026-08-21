from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
QXQXF = ROOT / "build" / "qxqxf.exe"


def prompt_state_command(threads: str) -> list[str]:
    return [
        str(QXQXF),
        "prompt-state-loop-probe",
        "--in",
        "missing-model.qxf",
        "--tokenizer",
        "missing-tokenizer.qxt",
        "--text-file",
        "missing-prompt.txt",
        "--generate",
        "1",
        "--layers",
        "48",
        "--ctx",
        "16",
        "--kv",
        "int8",
        "--activation",
        "f32",
        "--thread-policy",
        "serial",
        "--threads",
        threads,
        "--temperature",
        "0",
        "--seed",
        "7",
        "--full-moe",
        "--final-head",
    ]


@pytest.mark.skipif(not QXQXF.exists(), reason="qxqxf.exe must be built before CLI tests")
@pytest.mark.parametrize("threads", ["abc", "1abc", "4294967296"])
def test_prompt_state_loop_probe_rejects_malformed_threads_before_file_io(threads: str):
    result = subprocess.run(prompt_state_command(threads), cwd=ROOT, capture_output=True, text=True)

    assert result.returncode == 2
    assert "invalid --threads" in result.stderr
    assert "text file read failed" not in result.stderr
    assert "failed to open" not in result.stderr
