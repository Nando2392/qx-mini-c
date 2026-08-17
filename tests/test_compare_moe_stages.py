import json
import struct
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / "scripts" / "compare_moe_stages.py"


def write_f32(path, values):
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def test_compare_moe_stages_reports_stage_and_expert_metrics(tmp_path):
    qx_dir, llama_dir = tmp_path / "qx", tmp_path / "llama"
    qx_dir.mkdir()
    llama_dir.mkdir()
    fixtures = {
        "ffn_norm": [1.0, 2.0],
        "ffn_moe_logits": [0.1, 0.2, 0.3, 0.4],
        "ffn_moe_probs": [0.1, 0.2, 0.3, 0.4],
        "ffn_moe_topk": [3.0, 2.0],
        "ffn_moe_weights": [0.4, 0.3],
        "ffn_moe_weights_sum": [0.7],
        "ffn_moe_weights_norm": [4 / 7, 3 / 7],
        "ffn_moe_gate": [1.0, 2.0, 3.0, 4.0],
        "ffn_moe_up": [1.0, 2.0, 3.0, 4.0],
        "ffn_moe_swiglu": [1.0, 2.0, 3.0, 4.0],
        "ffn_moe_down": [1.0, 2.0, 3.0, 4.0],
        "ffn_moe_weighted": [1.0, 2.0, 3.0, 4.0],
    }
    for name, values in fixtures.items():
        write_f32(qx_dir / f"{name}-0.f32", values)
        expected = list(values)
        if name == "ffn_moe_gate":
            expected[2] = 2.5
        write_f32(llama_dir / f"{name}-0.f32", expected)

    completed = subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            "--qx-dir",
            str(qx_dir),
            "--llama-dir",
            str(llama_dir),
            "--layer",
            "0",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["experts_used"] == 2
    assert payload["selected_experts"] == [3, 2]
    assert payload["stages"]["ffn_moe_gate"]["max_abs"] == 0.5
    assert payload["stages"]["ffn_moe_gate"]["max_abs_index"] == 2
    assert payload["experts"][1]["expert_id"] == 2
    assert payload["experts"][1]["gate"]["max_abs"] == 0.5
    assert payload["experts"][0]["gate"]["max_abs"] == 0


def test_compare_moe_stages_fails_closed_on_non_finite_or_shape_mismatch(tmp_path):
    qx_dir, llama_dir = tmp_path / "qx", tmp_path / "llama"
    qx_dir.mkdir()
    llama_dir.mkdir()
    write_f32(qx_dir / "ffn_norm-0.f32", [1.0, float("nan")])
    write_f32(llama_dir / "ffn_norm-0.f32", [1.0])
    completed = subprocess.run(
        [sys.executable, str(COMPARE), "--qx-dir", str(qx_dir), "--llama-dir", str(llama_dir), "--layer", "0"],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["passed"] is False
    assert "non-finite" in payload["error"] or "length mismatch" in payload["error"]
