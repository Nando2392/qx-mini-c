from __future__ import annotations

import copy
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "native_4k_acceptance.py"
SPEC = importlib.util.spec_from_file_location("native_4k_acceptance", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def test_only_raw_fixture() -> dict:
    """Synthetic Issue 84 fixture; never native acceptance evidence."""
    profile = {
        "requested_io_backend": "buffered", "effective_io_backend": "buffered",
        "requested_scratch_policy": "ephemeral", "effective_scratch_policy": "ephemeral",
        "requested_kernel_policy": "baseline", "effective_kernel_policy": "baseline",
        "requested_thread_policy": "serial", "effective_thread_policy": "serial",
        "requested_thread_count": 1, "effective_thread_count": 1, "sampled_steps": 2,
        "workers_used": 1, "scratch_peak_capacity_bytes": 0, "scratch_growth_events": 0,
        "temporary_blocks_decoded": 1, "temporary_floats_materialized": 1,
        "temporary_bytes_materialized": 4, "fused_final_head_dot_calls": 0,
        "baseline_final_head_dot_calls": 2, "final_head_q6_k_blocks": 2,
        "final_head_parallel_jobs": 0, "final_head_serial_jobs": 2,
        "final_head_fallback_jobs": 2, "full_logits_checksums": ["11", "22"],
    }
    return {
        "payload": {"prompt_token_count": 4095, "generated_token_ids": [7, 8],
            "generated_text": "ok", "stop_reason": "max_tokens",
            "timing": {"prefill_seconds": 1.0, "decode_seconds": 0.5},
            "execution_profile": profile,
            "capacity_profile": {"executed_forward_steps": 4096,
                "prompt_forward_steps": 4095, "generated_input_forward_steps": 1,
                "first_eos_output_index": None}},
        "wall_seconds": 2.0, "sampled_peak_rss_bytes": 123456,
        "stdout": {"path": "stdout.bin", "size_bytes": 2, "sha256": "a" * 64},
        "stderr": {"path": "stderr.bin", "size_bytes": 0, "sha256": "b" * 64},
    }


def test_bounded_prompt_construction_uses_actual_counts_within_cli_limit() -> None:
    calls: list[str] = []

    def encode(text: str) -> dict:
        calls.append(text)
        count = 4095 if text.startswith("~!") else 1
        return {"token_count": count, "token_ids": [3] * count}

    result = runner.construct_exact_prompt(4095, encode)
    assert result["token_count"] == 4095
    assert result["text"] == ("~!" * 2048)[:4095]
    assert 1 <= len(calls) <= 8
    assert all(len(text.encode("utf-8")) <= runner.TOKENIZER_MAX_INPUT_BYTES for text in calls)


def test_prompt_construction_rejects_target_larger_than_cli_byte_limit() -> None:
    with pytest.raises(ValueError, match="4096-byte"):
        runner.construct_exact_prompt(4097, lambda _text: pytest.fail("must fail before encoding"))


def test_validator_enforces_fixed_4k_contract(test_only_raw_fixture: dict) -> None:
    compact = runner.validate_native_run(copy.deepcopy(test_only_raw_fixture))
    assert compact["capacity_profile"]["executed_forward_steps"] == 4096
    for mutation, match in [
        (("capacity_profile", "executed_forward_steps", 4095), "4096"),
        (("capacity_profile", "first_eos_output_index", 0), "EOS"),
        (("execution_profile", "effective_thread_count", 2), "thread"),
        (("execution_profile", "full_logits_checksums", ["11"]), "two"),
        (("timing", "prefill_seconds", float("nan")), "finite"),
    ]:
        raw = copy.deepcopy(test_only_raw_fixture)
        section, key, value = mutation
        raw["payload"][section][key] = value
        with pytest.raises(ValueError, match=match):
            runner.validate_native_run(raw)


def _fake_child(tmp_path: Path, body: str) -> list[str]:
    child = tmp_path / "fake_child.py"
    child.write_text(body, encoding="utf-8")
    return [sys.executable, str(child)]


def test_supervisor_timeout_retains_logs_and_nonpass_journal(tmp_path: Path) -> None:
    cmd = _fake_child(tmp_path, "import time; print('started', flush=True); time.sleep(30)\n")
    with pytest.raises(runner.RunGateError, match="deadline"):
        runner.run_supervised(cmd, tmp_path, deadline_seconds=0.35, rss_ceiling_bytes=512 << 20,
                              sample_interval_seconds=0.05, heartbeat_seconds=0.1)
    assert b"started" in (tmp_path / "stdout.bin").read_bytes()
    journal = json.loads((tmp_path / "phase-journal.json").read_text())
    assert journal["status"] == "timeout" and journal["elapsed_seconds"] > 0


def test_supervisor_rss_gate_terminates_and_records_nonpass(tmp_path: Path) -> None:
    cmd = _fake_child(tmp_path, "import time; x=bytearray(16*1024*1024); print(len(x), flush=True); time.sleep(30)\n")
    with pytest.raises(runner.RunGateError, match="RSS"):
        runner.run_supervised(cmd, tmp_path, deadline_seconds=3, rss_ceiling_bytes=2 << 20,
                              sample_interval_seconds=0.05, heartbeat_seconds=0.1)
    journal = json.loads((tmp_path / "phase-journal.json").read_text())
    assert journal["status"] == "rss_exceeded"
    assert journal["peak_rss_bytes"] > 2 << 20


def test_supervisor_cancel_cleans_child_and_journals(tmp_path: Path) -> None:
    cmd = _fake_child(tmp_path, "import time; print('wait', flush=True); time.sleep(30)\n")
    cancel = threading.Event()
    threading.Timer(0.2, cancel.set).start()
    with pytest.raises(runner.RunGateError, match="cancel"):
        runner.run_supervised(cmd, tmp_path, deadline_seconds=3, rss_ceiling_bytes=512 << 20,
                              sample_interval_seconds=0.05, heartbeat_seconds=0.1,
                              cancel_requested=cancel.is_set)
    assert json.loads((tmp_path / "phase-journal.json").read_text())["status"] == "cancelled"


def test_output_directory_must_be_new(tmp_path: Path) -> None:
    existing = tmp_path / "evidence"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        runner.create_output_directory(existing)


def test_report_validator_rejects_legacy_incomplete_fixture(test_only_raw_fixture: dict) -> None:
    case = runner.validate_native_run(copy.deepcopy(test_only_raw_fixture))
    report = runner.test_only_report_fixture(case)
    # This legacy fixture deliberately lacks production provenance. Complete
    # fixtures and corruption/path checks live in test_native_4k_report_validation.
    with pytest.raises(ValueError, match="claim scope mismatch"):
        runner.validate_report(report)
