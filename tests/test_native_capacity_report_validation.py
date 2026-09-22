from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "wiki/evidence/issue-83-native-capacity-report.json"
SCRIPT = ROOT / "scripts/validate_native_capacity_report.py"
EXPECTED_REPORT_SHA256 = "4fedd8cdf44496491e92a288304c5889b9f7a4634997d8892002158fe47ef91d"

SPEC = importlib.util.spec_from_file_location("validate_native_capacity_report", SCRIPT)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


@pytest.fixture
def measured_report() -> dict:
    raw = REPORT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_REPORT_SHA256
    return json.loads(raw)


def reject(report: dict) -> None:
    with pytest.raises((TypeError, ValueError, KeyError)):
        validator.validate_report(report)


def test_immutable_measured_report_passes_strict_offline_validation(measured_report: dict) -> None:
    validator.validate_report(copy.deepcopy(measured_report))


def test_cli_validates_existing_report_without_writing(measured_report: dict) -> None:
    before = REPORT.read_bytes()
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--report", str(REPORT)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "report": "wiki/evidence/issue-83-native-capacity-report.json",
        "sha256": EXPECTED_REPORT_SHA256,
        "status": "pass",
    }
    assert REPORT.read_bytes() == before


def test_cli_rejects_self_consistent_forged_provenance(measured_report: dict, tmp_path: Path) -> None:
    measured_report["provenance"]["model_qxf"]["sha256"] = "0" * 64
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(measured_report), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--report", str(forged)],
        cwd=ROOT, capture_output=True, encoding="utf-8", errors="strict",
    )
    assert completed.returncode == 2
    assert "immutable report SHA-256 mismatch" in completed.stderr
    assert not completed.stdout


@pytest.mark.parametrize("field", [
    "schema", "issue", "status", "claim_scope", "fixed_contract",
    "provenance", "reproduction", "cases", "gates",
])
def test_every_top_level_field_is_required(measured_report: dict, field: str) -> None:
    del measured_report[field]
    reject(measured_report)


@pytest.mark.parametrize("path", [
    ("fixed_contract", "positions"), ("fixed_contract", "runs_per_position"),
    ("fixed_contract", "total_native_runs"), ("fixed_contract", "layers"),
    ("fixed_contract", "activation"), ("fixed_contract", "kv"),
    ("fixed_contract", "sampling"), ("fixed_contract", "max_tokens"),
    ("fixed_contract", "io_backend"), ("fixed_contract", "scratch_policy"),
    ("fixed_contract", "kernel_policy"), ("fixed_contract", "thread_policy"),
    ("fixed_contract", "threads"), ("fixed_contract", "per_case_timeout_seconds"),
    ("fixed_contract", "global_timeout_seconds"), ("fixed_contract", "rss_sample_interval_seconds"),
    ("gates", "status"), ("gates", "ordered_positions"),
    ("gates", "all_prompt_tokens_consumed"), ("gates", "generated_input_forward_steps_each"),
    ("gates", "checksum_count_each"), ("gates", "early_eos"),
    ("gates", "finite_timings_and_positive_sampled_rss"),
    ("reproduction", "working_directory"), ("reproduction", "python"),
    ("reproduction", "command"),
    ("provenance", "platform"), ("provenance", "psutil_version"),
    ("provenance", "native_inputs_pre"), ("provenance", "native_inputs_post"),
    ("provenance", "model_qxf"), ("provenance", "tokenizer_qxt"),
    ("provenance", "prompts"),
])
def test_every_strict_metadata_field_is_required(measured_report: dict, path: tuple[str, str]) -> None:
    del measured_report[path[0]][path[1]]
    reject(measured_report)


@pytest.mark.parametrize(("path", "value"), [
    (("claim_scope",), ""), (("fixed_contract",), {}), (("gates",), {}),
    (("reproduction",), {}), (("provenance", "model_qxf"), {}),
    (("fixed_contract", "threads"), True), (("fixed_contract", "max_tokens"), 0),
    (("fixed_contract", "layers"), -1), (("fixed_contract", "global_timeout_seconds"), float("nan")),
    (("fixed_contract", "rss_sample_interval_seconds"), False),
    (("gates", "all_prompt_tokens_consumed"), 1), (("gates", "early_eos"), 0),
    (("provenance", "psutil_version"), ""), (("provenance", "prompts"), []),
    (("provenance", "model_qxf", "size_bytes"), True),
    (("provenance", "model_qxf", "sha256"), "0" * 63),
    (("provenance", "tokenizer_qxt", "size_bytes"), 0),
    (("provenance", "tokenizer_qxt", "path"), "C:/secret/model.qxt"),
    (("provenance", "tokenizer_qxt", "path"), "../model.qxt"),
    (("cases", 0, "command"), []), (("cases", 1, "command"), ["wrong"]),
    (("cases", 0, "timing", "native_cpu_seconds"), True),
    (("cases", 0, "timing", "native_cpu_seconds"), 0),
    (("cases", 1, "timing", "native_cpu_seconds"), -1),
    (("cases", 1, "timing", "native_cpu_seconds"), float("nan")),
    (("cases", 1, "timing", "native_cpu_seconds"), 1.0),
])
def test_empty_wrong_type_bool_nonfinite_nonpositive_and_drift_fail_closed(
    measured_report: dict, path: tuple, value: object,
) -> None:
    target = measured_report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    reject(measured_report)


def test_fixed_contract_gates_claim_scope_and_reproduction_are_exact(measured_report: dict) -> None:
    for section, field, value in [
        ("fixed_contract", "activation", "f16"),
        ("gates", "status", "PASS"),
        (None, "claim_scope", measured_report["claim_scope"] + " extra"),
        ("reproduction", "working_directory", "."),
    ]:
        changed = copy.deepcopy(measured_report)
        (changed if section is None else changed[section])[field] = value
        reject(changed)


def test_artifact_and_source_schemas_are_exact_and_manifest_is_recomputed(measured_report: dict) -> None:
    changed = copy.deepcopy(measured_report)
    changed["provenance"]["model_qxf"]["extra"] = "forged"
    reject(changed)

    for index in (0, -1):
        changed = copy.deepcopy(measured_report)
        changed["provenance"]["native_inputs_pre"]["source_files"][index]["sha256"] = "0" * 64
        changed["provenance"]["native_inputs_post"] = copy.deepcopy(changed["provenance"]["native_inputs_pre"])
        reject(changed)

    changed = copy.deepcopy(measured_report)
    for side in ("native_inputs_pre", "native_inputs_post"):
        changed["provenance"][side]["source_manifest_sha256"] = "0" * 64
    reject(changed)

    changed = copy.deepcopy(measured_report)
    records = changed["provenance"]["native_inputs_pre"]["source_files"]
    records.append(copy.deepcopy(records[0]))
    changed["provenance"]["native_inputs_post"] = copy.deepcopy(changed["provenance"]["native_inputs_pre"])
    reject(changed)


def test_case_schema_command_positions_and_native_cpu_sum_are_strict(measured_report: dict) -> None:
    for mutation in ("missing", "extra", "position", "command", "sum"):
        changed = copy.deepcopy(measured_report)
        case = changed["cases"][1]
        if mutation == "missing":
            del case["command"]
        elif mutation == "extra":
            case["extra"] = 1
        elif mutation == "position":
            case["positions"] = 128
        elif mutation == "command":
            case["command"][11] = "128"
        else:
            case["timing"]["native_cpu_seconds"] += 0.01
        reject(changed)
