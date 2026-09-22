from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "native_capacity_acceptance.py"
SPEC = importlib.util.spec_from_file_location("native_capacity_acceptance", SCRIPT)
assert SPEC and SPEC.loader
capacity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capacity)


@pytest.fixture
def test_only_raw_fixture() -> callable:
    """Synthetic validator fixture only; never publication/acceptance evidence."""
    def make(positions: int) -> dict:
        prompt_count = positions - 1
        profile = {
            "requested_io_backend": "buffered",
            "effective_io_backend": "buffered",
            "requested_scratch_policy": "ephemeral",
            "effective_scratch_policy": "ephemeral",
            "requested_kernel_policy": "baseline",
            "effective_kernel_policy": "baseline",
            "requested_thread_policy": "serial",
            "effective_thread_policy": "serial",
            "requested_thread_count": 1,
            "effective_thread_count": 1,
            "sampled_steps": 2,
            "workers_used": 1,
            "scratch_peak_capacity_bytes": 0,
            "scratch_growth_events": 0,
            "temporary_blocks_decoded": 200,
            "temporary_floats_materialized": 300,
            "temporary_bytes_materialized": 1200,
            "fused_final_head_dot_calls": 0,
            "baseline_final_head_dot_calls": 100,
            "final_head_q6_k_blocks": 100,
            "final_head_parallel_jobs": 0,
            "final_head_serial_jobs": 2,
            "final_head_fallback_jobs": 2,
            "full_logits_checksums": ["123", "456"],
        }
        return {
            "wall_seconds": 3.0,
            "sampled_peak_rss_bytes": 4096,
            "payload": {
                "prompt_token_count": prompt_count,
                "generated_token_ids": [358, 1184],
                "generated_text": " test",
                "stop_reason": "max_tokens",
                "timing": {"prefill_seconds": 1.0, "decode_seconds": 2.0},
                "execution_profile": profile,
                "capacity_profile": {
                    "executed_forward_steps": positions,
                    "prompt_forward_steps": prompt_count,
                    "generated_input_forward_steps": 1,
                    "first_eos_output_index": None,
                },
            },
        }
    return make


@pytest.mark.parametrize("positions", capacity.POSITIONS)
def test_validator_accepts_only_complete_test_fixture(test_only_raw_fixture, positions: int) -> None:
    compact = capacity.validate_and_compact_run(test_only_raw_fixture(positions), positions)
    assert compact["positions"] == positions
    assert compact["capacity_profile"]["prompt_forward_steps"] == positions - 1
    assert len(compact["execution_profile"]["full_logits_checksums"]) == 2


@pytest.mark.parametrize("positions", capacity.POSITIONS)
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("executed_forward_steps", 0, "does not prove position"),
        ("prompt_forward_steps", 0, "does not prove position"),
        ("generated_input_forward_steps", 0, "does not prove position"),
        ("generated_input_forward_steps", True, "must be an integer"),
    ],
)
def test_capacity_counter_mismatches_fail_closed(test_only_raw_fixture, positions, field, value, message) -> None:
    raw = test_only_raw_fixture(positions)
    raw["payload"]["capacity_profile"][field] = value
    with pytest.raises(ValueError, match=message):
        capacity.validate_and_compact_run(raw, positions)


@pytest.mark.parametrize("positions", capacity.POSITIONS)
def test_early_eos_cannot_claim_last_position(test_only_raw_fixture, positions: int) -> None:
    raw = test_only_raw_fixture(positions)
    raw["payload"]["generated_token_ids"] = [358]
    raw["payload"]["stop_reason"] = "eos"
    raw["payload"]["capacity_profile"]["generated_input_forward_steps"] = 0
    raw["payload"]["capacity_profile"]["first_eos_output_index"] = 0
    raw["payload"]["execution_profile"]["sampled_steps"] = 1
    raw["payload"]["execution_profile"]["full_logits_checksums"] = ["123"]
    with pytest.raises(ValueError, match="early EOS is a gate failure"):
        capacity.validate_and_compact_run(raw, positions)


@pytest.mark.parametrize("positions", capacity.POSITIONS)
def test_invalid_timing_and_checksum_count_fail_closed(test_only_raw_fixture, positions: int) -> None:
    raw = test_only_raw_fixture(positions)
    raw["payload"]["timing"]["prefill_seconds"] = float("nan")
    with pytest.raises(ValueError, match="finite and positive"):
        capacity.validate_and_compact_run(raw, positions)

    raw = test_only_raw_fixture(positions)
    raw["payload"]["execution_profile"]["full_logits_checksums"] = ["123"]
    with pytest.raises(ValueError, match="exactly two"):
        capacity.validate_and_compact_run(raw, positions)


@pytest.mark.parametrize("positions", capacity.POSITIONS)
def test_prompt_and_token_ranges_fail_closed(test_only_raw_fixture, positions: int) -> None:
    raw = test_only_raw_fixture(positions)
    raw["payload"]["prompt_token_count"] -= 1
    with pytest.raises(ValueError, match="exact constructed target"):
        capacity.validate_and_compact_run(raw, positions)

    raw = test_only_raw_fixture(positions)
    raw["payload"]["generated_token_ids"][1] = capacity.VOCAB_COUNT
    with pytest.raises(ValueError, match="out of vocabulary"):
        capacity.validate_and_compact_run(raw, positions)


def test_prompt_construction_uses_measured_counts_not_repeat_assumption() -> None:
    observed: list[str] = []
    def test_only_encode(text: str) -> dict:
        observed.append(text)
        # Deliberately hold the count once so repetitions are not equivalent to tokens.
        count = max(1, len(text.split()) - (1 if len(text.split()) >= 3 else 0))
        return {"token_count": count, "token_ids": list(range(count))}

    prompts = capacity.construct_exact_prompts((2, 4), test_only_encode)
    assert prompts[2]["token_count"] == 2
    assert prompts[4]["token_count"] == 4
    assert len(observed) > 4


def test_prompt_construction_fails_when_exact_target_is_unreachable() -> None:
    def test_only_encode(text: str) -> dict:
        count = len(text.split()) * 2
        return {"token_count": count, "token_ids": list(range(count))}

    with pytest.raises(ValueError, match="did not tokenize to exact targets"):
        capacity.construct_exact_prompts((3,), test_only_encode)


def test_cli_missing_artifact_fails_closed_without_report(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist.json"
    completed = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--qx-exe", str(tmp_path / "missing-qxqxf.exe"),
            "--model", str(tmp_path / "missing.qxf"),
            "--tokenizer", str(tmp_path / "missing.qxt"),
            "--output", str(output),
        ],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr.startswith("capacity acceptance failed closed: missing required artifact:")
    assert not output.exists()


def test_report_rejects_changed_native_snapshot(test_only_raw_fixture) -> None:
    cases = [capacity.validate_and_compact_run(test_only_raw_fixture(position), position) for position in capacity.POSITIONS]
    report = {
        "schema": capacity.SCHEMA,
        "issue": 83,
        "status": "pass",
        "claim_scope": "test only",
        "fixed_contract": {},
        "provenance": {"native_inputs_pre": {"hash": "a"}, "native_inputs_post": {"hash": "b"}},
        "reproduction": {},
        "cases": cases,
        "gates": {},
    }
    with pytest.raises(ValueError, match="changed during acceptance"):
        capacity.validate_report(copy.deepcopy(report))


def test_production_reproduction_metadata_passes_privacy_gate(test_only_raw_fixture) -> None:
    report = {
        "schema": capacity.SCHEMA, "issue": 83, "status": "pass",
        "claim_scope": "synthetic test fixture, not measured evidence",
        "fixed_contract": {},
        "provenance": {"native_inputs_pre": {"hash": "a"}, "native_inputs_post": {"hash": "a"}},
        "reproduction": capacity.reproduction_metadata(),
        "cases": [capacity.validate_and_compact_run(test_only_raw_fixture(p), p) for p in capacity.POSITIONS],
        "gates": {},
    }
    capacity.validate_report(report)
    report["reproduction"]["python"] = "C:/Users/private-user/python.exe"
    with pytest.raises(ValueError, match="absolute Windows path"):
        capacity.validate_report(report)
