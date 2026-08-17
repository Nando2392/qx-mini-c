import json
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / "scripts" / "compare_residuals.py"


def write_f32(path, values):
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def run_compare(qx_dir, llama_dir, *extra):
    return subprocess.run(
        [sys.executable, str(COMPARE), "--qx-dir", str(qx_dir), "--llama-dir", str(llama_dir), "--layers", "0,1", *extra],
        text=True,
        capture_output=True,
    )


def test_compare_residuals_reports_first_divergent_layer(tmp_path):
    qx_dir, llama_dir = tmp_path / "qx", tmp_path / "llama"
    qx_dir.mkdir()
    llama_dir.mkdir()
    write_f32(qx_dir / "step-0-layer-0-input.f32", [1.0, 2.0, 3.0])
    write_f32(llama_dir / "layer-0.f32", [1.0, 2.0, 3.0])
    write_f32(qx_dir / "step-0-layer-1-input.f32", [1.0, 2.0, 4.0])
    write_f32(llama_dir / "layer-1.f32", [1.0, 2.0, 3.0])
    result = run_compare(qx_dir, llama_dir)
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["passed"] is False
    assert payload["first_divergent_layer"] == 1
    assert payload["layers"][0]["max_abs"] == 0
    assert payload["layers"][0]["cosine"] == 1
    assert payload["layers"][1]["max_abs"] == 1


def test_compare_residuals_passes_with_explicit_tolerance(tmp_path):
    qx_dir, llama_dir = tmp_path / "qx", tmp_path / "llama"
    qx_dir.mkdir()
    llama_dir.mkdir()
    for layer in (0, 1):
        write_f32(qx_dir / f"step-0-layer-{layer}-input.f32", [1.0, 2.0, 3.001])
        write_f32(llama_dir / f"layer-{layer}.f32", [1.0, 2.0, 3.0])
    result = run_compare(qx_dir, llama_dir, "--max-abs", "0.01", "--min-cosine", "0.999")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["first_divergent_layer"] is None


def test_compare_residuals_rejects_malformed_sidecar(tmp_path):
    qx_dir, llama_dir = tmp_path / "qx", tmp_path / "llama"
    qx_dir.mkdir()
    llama_dir.mkdir()
    (qx_dir / "step-0-layer-0-input.f32").write_bytes(b"bad")
    write_f32(llama_dir / "layer-0.f32", [1.0])
    result = subprocess.run(
        [sys.executable, str(COMPARE), "--qx-dir", str(qx_dir), "--llama-dir", str(llama_dir), "--layers", "0"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "invalid F32 sidecar size" in json.loads(result.stdout)["error"]
