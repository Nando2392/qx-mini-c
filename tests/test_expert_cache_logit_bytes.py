from __future__ import annotations

import hashlib
import importlib.util
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "expert_cache_logit_bytes", ROOT / "scripts" / "expert_cache_logit_bytes.py"
)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def _capture(root: Path, name: str, values: list[list[float]]) -> tuple[Path, list[dict[str, object]]]:
    run = root / name
    run.mkdir()
    command = ["qxqxf.exe", "state-loop-probe"]
    capture = MOD.enable_full_logit_capture(command, run)
    assert command[-2:] == ["--dump-residuals", str(capture)]
    steps = []
    for index, logits in enumerate(values):
        selected = max(range(len(logits)), key=logits.__getitem__)
        steps.append({"step": index, "phase": "generate", "selected_token": selected})
        (capture / f"step-{index}-logits.f32").write_bytes(struct.pack(f"<{len(logits)}f", *logits))
    return capture, MOD.collect_generated_logits(capture, steps, len(values[0]), root)


def test_collects_only_generated_full_vocab_files_and_compares_raw_bytes(tmp_path: Path) -> None:
    values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    _, none = _capture(tmp_path, "none", values)
    _, resident = _capture(tmp_path, "resident", values)
    result = MOD.require_exact_policy_bytes([none], [resident], tmp_path)
    assert result == {
        "full_logit_bytes_equal": True,
        "comparison_method": "direct_chunked_byte_comparison_not_hash_equality",
        "generated_steps": [0, 1],
        "generated_token_ids": [2, 2],
        "none_runs_compared": 1,
        "resident_runs_compared": 1,
        "runs_compared": 2,
        "bytes_embedded_in_json": False,
    }
    assert all(set(item) == {
        "step", "selected_token", "argmax_from_raw_f32", "path", "size_bytes",
        "f32_count", "sha256",
    } for item in none)


def test_rejects_same_shape_different_raw_bytes(tmp_path: Path) -> None:
    _, none = _capture(tmp_path, "none", [[1.0, 2.0], [3.0, 4.0]])
    _, resident = _capture(tmp_path, "resident", [[1.0, 2.0], [3.0, 4.5]])
    with pytest.raises(MOD.FullLogitBytesError, match="bytes differ at generated step 1"):
        MOD.require_exact_policy_bytes([none], [resident], tmp_path)


def test_rejects_missing_frozen_logit_file(tmp_path: Path) -> None:
    _, none = _capture(tmp_path, "none", [[1.0, 2.0]])
    _, resident = _capture(tmp_path, "resident", [[1.0, 2.0]])
    (tmp_path / resident[0]["path"]).unlink()

    with pytest.raises(MOD.FullLogitBytesError, match="frozen full-logit artifact changed"):
        MOD.require_exact_policy_bytes([none], [resident], tmp_path)


def test_rejects_changed_bytes_even_when_record_hash_is_updated(tmp_path: Path) -> None:
    _, none = _capture(tmp_path, "none", [[1.0, 2.0]])
    _, resident = _capture(tmp_path, "resident", [[1.0, 2.0]])
    resident_path = tmp_path / resident[0]["path"]
    changed = struct.pack("<2f", 1.0, 2.5)
    resident_path.write_bytes(changed)
    resident[0]["sha256"] = hashlib.sha256(changed).hexdigest()

    with pytest.raises(MOD.FullLogitBytesError, match="bytes differ at generated step 0"):
        MOD.require_exact_policy_bytes([none], [resident], tmp_path)


def test_rejects_missing_policy_run(tmp_path: Path) -> None:
    _, none = _capture(tmp_path, "none", [[1.0, 2.0]])

    with pytest.raises(MOD.FullLogitBytesError, match="both NONE and resident full-logit runs are required"):
        MOD.require_exact_policy_bytes([none], [], tmp_path)


def test_rejects_stale_hash_after_logit_file_changes(tmp_path: Path) -> None:
    _, none = _capture(tmp_path, "none", [[1.0, 2.0]])
    _, resident = _capture(tmp_path, "resident", [[1.0, 2.0]])
    resident_path = tmp_path / resident[0]["path"]
    resident_path.write_bytes(struct.pack("<2f", 1.0, 2.5))

    with pytest.raises(MOD.FullLogitBytesError, match="frozen full-logit artifact changed"):
        MOD.require_exact_policy_bytes([none], [resident], tmp_path)


def test_rejects_truncated_vocabulary_file(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    capture = MOD.enable_full_logit_capture(["qxqxf.exe", "state-loop-probe"], run)
    (capture / "step-0-logits.f32").write_bytes(b"1234")
    with pytest.raises(MOD.FullLogitBytesError, match="full vocabulary F32 size"):
        MOD.collect_generated_logits(
            capture, [{"step": 0, "phase": "generate", "selected_token": 0}], 2, tmp_path
        )


def test_rejects_phantom_prompt_state_loop_dump_flag(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    command = ["qxqxf.exe", "prompt-state-loop-probe"]
    with pytest.raises(MOD.FullLogitBytesError, match="has no dump CLI option"):
        MOD.enable_full_logit_capture(command, run)
    assert command == ["qxqxf.exe", "prompt-state-loop-probe"]
    assert not (run / "full-logit-bytes").exists()
