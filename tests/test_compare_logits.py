import json
import struct
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / "scripts" / "compare_logits.py"


def write_f32(path, values):
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def test_compare_logits_reports_metrics_and_argmax(tmp_path):
    qx = tmp_path / "qx.f32"
    llama = tmp_path / "llama.f32"
    write_f32(qx, [1.0, 3.0, 2.0])
    write_f32(llama, [1.0, 3.0, 2.0])
    result = subprocess.run(
        [sys.executable, str(COMPARE), "--qx", str(qx), "--llama", str(llama)],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["count"] == 3
    assert payload["qx_argmax"] == 1
    assert payload["llama_argmax"] == 1
    assert payload["argmax_match"] is True
    assert payload["max_abs"] == 0
    assert payload["rmse"] == 0
    assert payload["cosine"] == 1


def test_compare_logits_fails_threshold_or_argmax_mismatch(tmp_path):
    qx = tmp_path / "qx.f32"
    llama = tmp_path / "llama.f32"
    write_f32(qx, [2.0, 1.0])
    write_f32(llama, [1.0, 2.0])
    result = subprocess.run(
        [
            sys.executable, str(COMPARE), "--qx", str(qx), "--llama", str(llama),
            "--max-abs", "0.1", "--rmse", "0.1", "--min-cosine", "0.99",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["argmax_match"] is False
    assert payload["pass"] is False
