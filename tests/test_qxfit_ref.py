import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qxfit_ref.py"

def run(*args):
    out = subprocess.check_output([sys.executable, str(SCRIPT), *args], text=True)
    return json.loads(out)

def test_qwen3_8b_int8_4k_fits_custom_q3q4():
    p = run("--model", "qwen3-8b", "--weight-gib", "3.3", "--ctx", "4096", "--kv", "int8")
    assert p["feasible"] is True
    assert abs(p["kv_gib"] - 0.281) < 0.01
    assert p["total_active_gib"] < 4.1

def test_qwen3_8b_fp16_16k_is_heavy_but_accounted():
    p = run("--model", "qwen3-8b", "--weight-gib", "4.68", "--ctx", "16384", "--kv", "fp16")
    assert abs(p["kv_gib"] - 2.25) < 0.01
    assert p["total_active_gib"] > 7.0

def test_qwen3_30b_q3s_is_risky_but_combined_feasible_on_paper():
    p = run("--model", "qwen3-30b-a3b", "--weight-gib", "12.38", "--ctx", "4096", "--kv", "int8")
    assert abs(p["kv_gib"] - 0.188) < 0.01
    assert p["suggested_ram_weights_gib"] > 8.0
