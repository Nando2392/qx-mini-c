from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "native_cpu_policy_benchmark.py"
SPEC = importlib.util.spec_from_file_location("native_cpu_policy_benchmark", SCRIPT)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


@pytest.fixture
def output_fixture() -> dict[str, Any]:
    """Synthetic unit-test-only output signature; never benchmark evidence."""
    return {
        "generated_token_ids": [101, 202],
        "generated_text": "fixture-only",
        "full_logits_checksums": ["123", "456"],
    }


@pytest.fixture
def profile_fixture() -> dict[str, Any]:
    """Synthetic unit-test-only compact profile; never benchmark evidence."""
    return {"fixture_only": True}


@pytest.fixture
def run_fixture(output_fixture: dict[str, Any], profile_fixture: dict[str, Any]) -> dict[str, Any]:
    """Synthetic unit-test-only measured run; never benchmark evidence."""
    return {
        "wall_seconds": 20.0,
        "native_prefill_seconds": 4.0,
        "native_decode_seconds": 10.0,
        "native_cpu_seconds": 14.0,
        "sampled_peak_rss_bytes": 1000,
        "output": copy.deepcopy(output_fixture),
        "execution_profile": copy.deepcopy(profile_fixture),
    }


def _raw_native_run(cell: dict[str, Any]) -> dict[str, Any]:
    """Build a policy-consistent native payload for validator unit tests."""
    baseline = cell["kernel_policy"] == "baseline"
    persistent = cell["scratch_policy"] == "persistent"
    pooled = cell["thread_policy"] == "pool"
    profile = {
        "requested_io_backend": cell["io_backend"],
        "effective_io_backend": cell["io_backend"],
        "requested_scratch_policy": cell["scratch_policy"],
        "effective_scratch_policy": cell["scratch_policy"],
        "requested_kernel_policy": cell["kernel_policy"],
        "effective_kernel_policy": cell["kernel_policy"],
        "requested_thread_policy": cell["thread_policy"],
        "effective_thread_policy": cell["thread_policy"],
        "requested_thread_count": cell["threads"],
        "effective_thread_count": cell["threads"],
        "sampled_steps": 2,
        "workers_used": cell["threads"] if pooled else 1,
        "scratch_peak_capacity_bytes": 4096 if persistent else 0,
        "scratch_growth_events": 1 if persistent else 0,
        "temporary_blocks_decoded": 1 if baseline else 0,
        "temporary_floats_materialized": 256 if baseline else 0,
        "temporary_bytes_materialized": 1024 if baseline else 0,
        "fused_final_head_dot_calls": 0 if baseline else 1,
        "baseline_final_head_dot_calls": 1 if baseline else 0,
        "final_head_q6_k_blocks": 1,
        "final_head_parallel_jobs": 1 if pooled else 0,
        "final_head_serial_jobs": 0 if pooled else 1,
        "final_head_fallback_jobs": 1 if baseline else 0,
        "full_logits_checksums": ["123", "456"],
    }
    return {
        "wall_seconds": 20.0,
        "sampled_peak_rss_bytes": 1000,
        "payload": {
            "prompt_token_count": 2,
            "generated_token_ids": [101, 202],
            "generated_text": "fixture-only",
            "stop_reason": "max_tokens",
            "timing": {"prefill_seconds": 4.0, "decode_seconds": 10.0},
            "execution_profile": profile,
        },
    }


@pytest.fixture
def report_fixture(run_fixture: dict[str, Any]) -> dict[str, Any]:
    """Synthetic unit-test-only valid report assembled through production aggregators."""
    cells = []
    for definition in benchmark.CELLS:
        warmup = copy.deepcopy(run_fixture)
        measured = []
        for index in range(5):
            run = copy.deepcopy(run_fixture)
            run["wall_seconds"] += index * 0.1
            run["native_prefill_seconds"] += index * 0.01
            run["native_decode_seconds"] += index * 0.02
            run["native_cpu_seconds"] = run["native_prefill_seconds"] + run["native_decode_seconds"]
            run["sampled_peak_rss_bytes"] += index
            measured.append(run)
        cells.append({
            **definition,
            "command": benchmark.sanitized_command(definition),
            "warmups": [warmup],
            "measured": measured,
            "summary": benchmark.summarize_runs(measured),
        })
    digest = "a" * 64
    snapshot = {
        "git_revision": "b" * 40,
        "source_manifest_sha256": digest,
        "source_files": [
            {"path": path, "size_bytes": 1, "sha256": digest}
            for path in benchmark.SOURCE_FILES
        ],
        "native_executable": {"path": "build/qxqxf.exe", "size_bytes": 1, "sha256": digest},
    }
    report = {
        "schema": benchmark.SCHEMA,
        "measurement": "issue-82-native-cli-cpu-policy",
        "status": "pass",
        "claim_scope": "fixture-only",
        "fixed_contract": {"fixture_only": True},
        "provenance": {
            "platform": {"fixture_only": True},
            "psutil_version": "fixture",
            "native_inputs_pre": copy.deepcopy(snapshot),
            "native_inputs_post": copy.deepcopy(snapshot),
            "model_qxf": {"path": "models/model.qxf", "size_bytes": 1, "sha256": digest},
            "tokenizer_qxt": {"path": "models/model.qxt", "size_bytes": 1, "sha256": digest},
            "prompt": {
                "text": benchmark.PROMPT_TEXT,
                "utf8_sha256": benchmark._sha256_bytes(benchmark.PROMPT_TEXT.encode("utf-8")),
                "bytes": len(benchmark.PROMPT_TEXT.encode("utf-8")),
            },
        },
        "phase_definitions": {"fixture_only": True},
        "reproducible_command": ["fixture-only"],
        "cells": cells,
        "output_equivalence": benchmark.validate_exact_outputs(cells),
        "decision": benchmark.build_decision(cells),
    }
    return report


@pytest.mark.parametrize("position", [0, 4])
@pytest.mark.parametrize("field", benchmark.SUMMARY_FIELDS)
def test_summarize_runs_rejects_missing_required_metric(
    run_fixture: dict[str, Any], position: int, field: str
) -> None:
    runs = [copy.deepcopy(run_fixture) for _ in range(5)]
    runs[position].pop(field)

    with pytest.raises(ValueError, match=f"measured run missing required field: {field}"):
        benchmark.summarize_runs(runs)


@pytest.mark.parametrize("bad_value", [True, "10.0"])
def test_summarize_runs_rejects_bool_and_string_metrics(
    run_fixture: dict[str, Any], bad_value: object
) -> None:
    runs = [copy.deepcopy(run_fixture) for _ in range(5)]
    runs[-1]["native_decode_seconds"] = bad_value

    with pytest.raises(ValueError, match="measured run field native_decode_seconds must be numeric"):
        benchmark.summarize_runs(runs)


def test_output_gate_compares_warmups_as_well_as_measured(report_fixture: dict[str, Any]) -> None:
    report_fixture["cells"][2]["warmups"][0]["output"]["full_logits_checksums"][1] = "999"

    with pytest.raises(ValueError, match="generated IDs/text/full logits checksums differ from baseline"):
        benchmark.validate_exact_outputs(report_fixture["cells"])


def test_output_gate_rejects_missing_cell(report_fixture: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="complete four-cell matrix"):
        benchmark.validate_exact_outputs(report_fixture["cells"][:-1])


def test_validate_report_accepts_complete_fixture(report_fixture: dict[str, Any]) -> None:
    benchmark.validate_report(report_fixture)


@pytest.mark.parametrize("cell", benchmark.CELLS, ids=lambda cell: cell["name"])
def test_validate_run_accepts_policy_specific_zero_counters(cell: dict[str, Any]) -> None:
    compact = benchmark.validate_and_compact_run(_raw_native_run(cell), cell)

    assert compact["execution_profile"]["effective_scratch_policy"] == cell["scratch_policy"]


@pytest.mark.parametrize("field", ["scratch_peak_capacity_bytes", "scratch_growth_events"])
def test_validate_run_rejects_zero_persistent_scratch_counters(field: str) -> None:
    cell = benchmark.CELLS[1]
    raw = _raw_native_run(cell)
    raw["payload"]["execution_profile"][field] = 0

    with pytest.raises(ValueError, match=rf"execution_profile\.{field} must be positive"):
        benchmark.validate_and_compact_run(raw, cell)


@pytest.mark.parametrize("field", ["scratch_peak_capacity_bytes", "scratch_growth_events"])
@pytest.mark.parametrize("bad_value", [-1, True, None], ids=["negative", "bool", "missing"])
def test_validate_run_rejects_invalid_scratch_counters(field: str, bad_value: object) -> None:
    cell = benchmark.CELLS[0]
    raw = _raw_native_run(cell)
    profile = raw["payload"]["execution_profile"]
    if bad_value is None:
        profile.pop(field)
    else:
        profile[field] = bad_value

    with pytest.raises(ValueError):
        benchmark.validate_and_compact_run(raw, cell)


def test_validate_report_rejects_frozen_binary_drift(report_fixture: dict[str, Any]) -> None:
    report_fixture["provenance"]["native_inputs_post"]["native_executable"]["sha256"] = "c" * 64

    with pytest.raises(ValueError, match="frozen native source/executable changed during campaign"):
        benchmark.validate_report(report_fixture)


def test_validate_report_rejects_missing_model_hash(report_fixture: dict[str, Any]) -> None:
    report_fixture["provenance"]["model_qxf"].pop("sha256")

    with pytest.raises(ValueError, match=r"model_qxf\.sha256 must be a lowercase SHA-256"):
        benchmark.validate_report(report_fixture)


def test_validate_report_rejects_source_manifest_schema(report_fixture: dict[str, Any]) -> None:
    report_fixture["provenance"]["native_inputs_pre"]["source_files"].pop()

    with pytest.raises(ValueError, match="source_files must contain the fixed source set"):
        benchmark.validate_report(report_fixture)


def test_validate_report_recomputes_summary(report_fixture: dict[str, Any]) -> None:
    report_fixture["cells"][1]["summary"]["native_decode_seconds"]["median"] = 0.1

    with pytest.raises(ValueError, match="report summary does not match measured runs"):
        benchmark.validate_report(report_fixture)


def test_decision_requires_decode_gain_beyond_mad_and_rss_ceiling(report_fixture: dict[str, Any]) -> None:
    cells = report_fixture["cells"]
    # Make one candidate >10% faster and outside combined MAD, but 11% above RSS baseline.
    for run in cells[1]["measured"]:
        run["native_decode_seconds"] = 8.0
        run["native_cpu_seconds"] = run["native_prefill_seconds"] + 8.0
        run["sampled_peak_rss_bytes"] = 1110
    cells[1]["summary"] = benchmark.summarize_runs(cells[1]["measured"])

    decision = benchmark.build_decision(cells)
    candidate = decision["candidates"][0]
    assert candidate["decode_gain_at_least_10_percent_and_beyond_combined_mad"] is True
    assert candidate["rss_within_10_percent_ceiling"] is False
    assert candidate["recommend"] is False
    assert decision["default_promotion"] is False
