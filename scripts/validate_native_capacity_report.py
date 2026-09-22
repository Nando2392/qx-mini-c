#!/usr/bin/env python
"""Strict, offline, read-only audit for the immutable Issue 83 report."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

# Keep this file directly importable by tests as well as executable as a script.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from native_capacity_acceptance import (  # noqa: E402
    POSITIONS,
    SOURCE_FILES,
    reproduction_metadata,
    sanitized_command,
    validate_report as original_validate_report,
)

ROOT = Path(__file__).resolve().parents[1]
CLAIM_SCOPE = (
    "Two finite runs on one pinned native CPU executable/model/tokenizer: actual 128 and 256 "
    "forward positions only; no quality, semantic, global parity, throughput, or default-promotion claim."
)
FIXED_CONTRACT = {
    "positions": [128, 256], "runs_per_position": 1, "total_native_runs": 2,
    "layers": 48, "activation": "f32", "kv": "int8", "sampling": "greedy",
    "max_tokens": 2, "io_backend": "buffered", "scratch_policy": "ephemeral",
    "kernel_policy": "baseline", "thread_policy": "serial", "threads": 1,
    "per_case_timeout_seconds": 3600.0, "global_timeout_seconds": 7200.0,
    "rss_sample_interval_seconds": 0.005,
}
GATES = {
    "status": "pass", "ordered_positions": [128, 256],
    "all_prompt_tokens_consumed": True, "generated_input_forward_steps_each": 1,
    "checksum_count_each": 2, "early_eos": False,
    "finite_timings_and_positive_sampled_rss": True,
}
CASE_KEYS = {
    "positions", "prompt_token_count", "generated_token_ids", "generated_text", "stop_reason",
    "timing", "wall_seconds", "sampled_peak_rss_bytes", "execution_profile", "capacity_profile", "command",
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
ARTIFACT_KEYS = {"path", "size_bytes", "sha256"}
NATIVE_INPUT_KEYS = {"git_revision", "source_manifest_sha256", "source_files", "native_executable"}
PROMPT_KEYS = {"positions", "token_count", "bytes", "utf8_sha256", "token_ids_sha256", "construction"}
PLATFORM_KEYS = {"system", "release", "machine", "processor", "python"}
PROVENANCE_KEYS = {
    "platform", "psutil_version", "native_inputs_pre", "native_inputs_post",
    "model_qxf", "tokenizer_qxt", "prompts",
}


def _object(value: object, label: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} schema mismatch")
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _portable_path(value: object, label: str, expected: str | None = None) -> str:
    path = _nonempty_string(value, label)
    if "\\" in path or re.match(r"^[A-Za-z]:", path) or path.startswith("/"):
        raise ValueError(f"{label} must be a portable repository-relative path")
    pure = PurePosixPath(path)
    if any(part in ("", ".", "..") for part in pure.parts) or pure.as_posix() != path:
        raise ValueError(f"{label} must be a normalized repository-relative path")
    if expected is not None and path != expected:
        raise ValueError(f"{label} mismatch")
    return path


def _artifact(value: object, label: str, expected_path: str) -> None:
    record = _object(value, label, ARTIFACT_KEYS)
    _portable_path(record["path"], f"{label}.path", expected_path)
    _positive_int(record["size_bytes"], f"{label}.size_bytes")
    _sha256(record["sha256"], f"{label}.sha256")


def _manifest_hash(records: list[dict[str, Any]]) -> str:
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact(value: object, expected: object, label: str) -> None:
    """Compare JSON values without Python's bool/int equality coercion."""
    if type(value) is not type(expected):
        raise ValueError(f"{label} type mismatch")
    if isinstance(expected, dict):
        if set(value) != set(expected):
            raise ValueError(f"{label} schema mismatch")
        for key in expected:
            _exact(value[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if len(value) != len(expected):
            raise ValueError(f"{label} length mismatch")
        for index, item in enumerate(expected):
            _exact(value[index], item, f"{label}[{index}]")
    elif value != expected:
        raise ValueError(f"{label} mismatch")


def _native_inputs(value: object, label: str) -> None:
    snapshot = _object(value, label, NATIVE_INPUT_KEYS)
    if not isinstance(snapshot["git_revision"], str) or HEX40.fullmatch(snapshot["git_revision"]) is None:
        raise ValueError(f"{label}.git_revision must be a lowercase SHA-1")
    records = snapshot["source_files"]
    if not isinstance(records, list) or len(records) != len(SOURCE_FILES):
        raise ValueError(f"{label}.source_files must be the exact source list")
    paths: list[str] = []
    for index, (record, expected_path) in enumerate(zip(records, SOURCE_FILES)):
        _artifact(record, f"{label}.source_files[{index}]", expected_path)
        paths.append(record["path"])
    if paths != list(SOURCE_FILES) or len(paths) != len(set(paths)):
        raise ValueError(f"{label}.source_files must be exact, ordered, and unique")
    expected_manifest = _manifest_hash(records)
    if _sha256(snapshot["source_manifest_sha256"], f"{label}.source_manifest_sha256") != expected_manifest:
        raise ValueError(f"{label}.source_manifest_sha256 mismatch")
    _artifact(snapshot["native_executable"], f"{label}.native_executable", "build/qxqxf.exe")


def _normalized_command(command: object, label: str) -> list[str]:
    if not isinstance(command, list) or len(command) != 24 or any(not isinstance(item, str) or not item for item in command):
        raise ValueError(f"{label} must be a non-empty string array")
    result = list(command)
    for index in (0, 3, 5):
        result[index] = result[index].replace("\\", "/")
        _portable_path(result[index], f"{label}[{index}]")
    return result


def _validate_provenance(value: object) -> None:
    provenance = _object(value, "provenance", PROVENANCE_KEYS)
    platform = _object(provenance["platform"], "provenance.platform", PLATFORM_KEYS)
    for key, item in platform.items():
        _nonempty_string(item, f"provenance.platform.{key}")
    _nonempty_string(provenance["psutil_version"], "provenance.psutil_version")
    _native_inputs(provenance["native_inputs_pre"], "provenance.native_inputs_pre")
    _native_inputs(provenance["native_inputs_post"], "provenance.native_inputs_post")
    if provenance["native_inputs_pre"] != provenance["native_inputs_post"]:
        raise ValueError("native input snapshots differ")
    _artifact(provenance["model_qxf"], "provenance.model_qxf", "models/Qwen3-30B-A3B-UD-IQ2_M.qxf")
    _artifact(provenance["tokenizer_qxt"], "provenance.tokenizer_qxt", "models/Qwen3-30B-A3B.qxt")
    prompts = provenance["prompts"]
    if not isinstance(prompts, list) or len(prompts) != 2:
        raise ValueError("provenance.prompts must contain exactly two records")
    construction = "repeated ASCII 'a' pattern, each candidate measured by canonical qxqxf tokenizer-encode"
    for prompt, positions in zip(prompts, POSITIONS):
        record = _object(prompt, f"provenance.prompts[{positions}]", PROMPT_KEYS)
        if record["positions"] != positions or record["token_count"] != positions - 1:
            raise ValueError("prompt positions/token count mismatch")
        text = " ".join(["a"] * (positions - 1)).encode("utf-8")
        if record["bytes"] != len(text) or record["utf8_sha256"] != hashlib.sha256(text).hexdigest():
            raise ValueError("prompt byte record mismatch")
        _sha256(record["token_ids_sha256"], "prompt token_ids_sha256")
        if record["construction"] != construction:
            raise ValueError("prompt construction mismatch")


def validate_report(report: dict[str, Any]) -> None:
    """Validate all published metadata without reading model/tokenizer/native files."""
    original_validate_report(report)
    if report["claim_scope"] != CLAIM_SCOPE:
        raise ValueError("claim_scope mismatch")
    _exact(report["fixed_contract"], FIXED_CONTRACT, "fixed_contract")
    _exact(report["gates"], GATES, "gates")
    _exact(report["reproduction"], reproduction_metadata(), "reproduction")
    _validate_provenance(report["provenance"])
    for case, positions in zip(report["cases"], POSITIONS):
        _object(case, f"case {positions}", CASE_KEYS)
        expected = [item.replace("\\", "/") for item in sanitized_command(positions)]
        if _normalized_command(case["command"], f"case {positions}.command") != expected:
            raise ValueError(f"case {positions}.command mismatch")
        timing = case["timing"]
        if not isinstance(timing, dict) or set(timing) != {"prefill_seconds", "decode_seconds", "native_cpu_seconds"}:
            raise ValueError(f"case {positions}.timing schema mismatch")
        native = timing["native_cpu_seconds"]
        if isinstance(native, bool) or not isinstance(native, (int, float)) or not math.isfinite(float(native)) or native <= 0:
            raise ValueError(f"case {positions}.timing.native_cpu_seconds must be finite and positive")
        expected_native = float(timing["prefill_seconds"]) + float(timing["decode_seconds"])
        if not math.isclose(float(native), expected_native, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"case {positions}.timing.native_cpu_seconds sum mismatch")


def _load_json(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    # This release audit is bound to the independently reviewed measurement,
    # not a general authenticity claim about arbitrary self-consistent reports.
    raw = path.read_bytes()
    expected = "4fedd8cdf44496491e92a288304c5889b9f7a4634997d8892002158fe47ef91d"
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("immutable report SHA-256 mismatch")
    data = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
    if not isinstance(data, dict):
        raise ValueError("report root must be an object")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict offline audit of an existing Issue 83 report")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    path = args.report.resolve()
    try:
        if not path.is_file():
            raise ValueError(f"report does not exist: {path}")
        report = _load_json(path)
        validate_report(report)
        try:
            shown = path.relative_to(ROOT).as_posix()
        except ValueError:
            shown = str(path)
        output = {"report": shown, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "status": "pass"}
        print(json.dumps(output, sort_keys=True))
        return 0
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError, IndexError) as exc:
        print(f"native capacity report validation failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
