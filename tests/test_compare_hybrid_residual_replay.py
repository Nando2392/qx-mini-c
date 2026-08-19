import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_hybrid_residual_replay.py"


def write_f32(path: Path, values: list[float]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def make_fixture(tmp_path: Path) -> tuple[Path, Path]:
    oracle = tmp_path / "oracle"
    hybrids = tmp_path / "hybrids"
    oracle.mkdir()
    write_f32(oracle / "layer-0.f32", [1.0, 2.0])
    write_f32(oracle / "layer-1.f32", [1.5, 2.5])
    write_f32(oracle / "l_out-1.f32", [2.0, 3.0])
    for start_layer, outputs in {
        0: {0: [1.6, 2.4], 1: [2.2, 2.8]},
        1: {1: [2.0, 3.0]},
    }.items():
        run_dir = hybrids / f"start-{start_layer}"
        run_dir.mkdir(parents=True)
        for layer, values in outputs.items():
            write_f32(run_dir / f"step-0-layer-{layer}-output.f32", values)
        (run_dir / "result.json").write_text(
            json.dumps(
                {
                    "start_layer": start_layer,
                    "kv_format": "f16",
                    "residual_source": "injected_f32_replay",
                    "residual_replay": {
                        "enabled": True,
                        "source": "f32_sidecar",
                        "values": 2,
                    }
                }
            ),
            encoding="utf-8",
        )
    return oracle, hybrids


def run_script(oracle: Path, hybrids: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--oracle-dir",
            str(oracle),
            "--hybrid-dir",
            str(hybrids),
            "--layers",
            "2",
            "--expected-count",
            "2",
            "--kv-format",
            "f16",
        ],
        text=True,
        capture_output=True,
    )


def test_hybrid_report_separates_incoming_accumulation_from_suffix_error(tmp_path):
    oracle, hybrids = make_fixture(tmp_path)

    completed = run_script(oracle, hybrids)

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["mode"]["residual_source"] == "llama.cpp layer-N.f32"
    assert report["mode"]["kv_source"] == "QX recomputed; oracle KV is not injected"
    assert report["mode"]["kv_format"] == "f16"
    assert [row["start_layer"] for row in report["rows"]] == [0, 1]
    assert report["rows"][0]["incoming_accumulated"]["rmse"] == 0.0
    assert report["rows"][1]["incoming_accumulated"]["rmse"] == pytest.approx(0.1)
    assert report["rows"][1]["final_after_replay"]["rmse"] == 0.0
    assert "suffix_gain_over_incoming_l2" not in report["rows"][1]


def test_hybrid_report_rejects_kv_format_metadata_mismatch(tmp_path):
    oracle, hybrids = make_fixture(tmp_path)
    metadata_path = hybrids / "start-1" / "result.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["kv_format"] = "f32"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    completed = run_script(oracle, hybrids)

    assert completed.returncode != 0
    assert "expected kv_format f16" in completed.stderr


def test_hybrid_report_rejects_residual_source_metadata_mismatch(tmp_path):
    oracle, hybrids = make_fixture(tmp_path)
    metadata_path = hybrids / "start-1" / "result.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["residual_source"] = "embedding_rmsnorm"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    completed = run_script(oracle, hybrids)

    assert completed.returncode != 0
    assert "expected residual_source injected_f32_replay" in completed.stderr


@pytest.mark.parametrize("invalid_bytes", [b"\x00\x00\x00", struct.pack("<2f", 1.0, float("nan"))])
def test_hybrid_report_rejects_malformed_or_nonfinite_sidecars(tmp_path, invalid_bytes):
    oracle, hybrids = make_fixture(tmp_path)
    (oracle / "layer-1.f32").write_bytes(invalid_bytes)

    completed = run_script(oracle, hybrids)

    assert completed.returncode != 0
    assert "layer-1.f32" in completed.stderr
