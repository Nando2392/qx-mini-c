from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "wiki" / "evidence" / "issue-82-native-cpu-policy-report.json"
CORRECTION = ROOT / "wiki" / "evidence" / "issue-82-timing-semantics.json"
REPORT_SHA256 = "ed801a0debc96d71dc7ea634cca434b37be0ed10024f84bb3bcfa2e53bda4125"
MSVC_CLOCK_URL = "https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/clock?view=msvc-170"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_correction_binds_immutable_original_report() -> None:
    correction = _load(CORRECTION)

    assert _sha256(REPORT) == REPORT_SHA256
    assert correction["derived_from"]["path"] == REPORT.relative_to(ROOT).as_posix()
    assert correction["derived_from"]["raw_sha256"] == REPORT_SHA256
    assert correction["measurement_integrity"] == {
        "measured_values_modified": False,
        "summaries_modified": False,
        "status_modified": False,
        "decisions_modified": False,
        "recommended_cells": [],
        "default_promotion": False,
    }


def test_correction_defines_msvc_elapsed_semantics_and_phase_boundaries() -> None:
    correction = _load(CORRECTION)
    semantics = correction["corrected_semantics"]

    assert semantics["timer"]["implementation"] == "MSVC CRT clock() / CLOCKS_PER_SEC"
    assert semantics["timer"]["meaning"] == "phase-local elapsed wall-clock duration"
    assert semantics["timer"]["not_process_cpu_time"] is True
    assert semantics["timer"]["not_summed_worker_cpu_time"] is True
    assert semantics["timer"]["primary_source_url"] == MSVC_CLOCK_URL
    assert semantics["aliases"] == {
        "native_cpu_seconds": "native_phase_elapsed_seconds",
        "native_prefill_seconds": "native_prefill_elapsed_seconds",
        "native_decode_seconds": "native_decode_elapsed_seconds",
    }
    assert semantics["phases"]["prefill"] == (
        "Processes every prompt token except the final prompt token."
    )
    assert semantics["phases"]["decode"] == (
        "Processes the final prompt token to produce the first generated token, then processes "
        "each subsequent generated-token input to produce the next output."
    )


def test_correction_is_documentary_not_native_timer_runtime_verification() -> None:
    correction = _load(CORRECTION)

    assert correction["verification_scope"]["test_kind"] == "documentary semantics and integrity"
    assert correction["verification_scope"]["native_timer_runtime_verified"] is False
    assert correction["verification_scope"]["runtime_verification_claim"] == "none"
