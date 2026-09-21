#!/usr/bin/env python
"""Bounded, fail-closed benchmark for the native ``qxqxf generate`` CPU policies.

The campaign is intentionally fixed: Hello!, F32/INT8, max-tokens=2, ctx=3,
one warmup and five measured runs for each of four policy cells.  Native source
and executable bytes are frozen before execution and checked again before a
report can be published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError as exc:  # pragma: no cover - exercised by CLI environments
    raise SystemExit("psutil is required; install it explicitly before running this benchmark") from exc

SCHEMA = 1
PROMPT_TEXT = "Hello!"
WARMUPS = 1
REPETITIONS = 5
PER_RUN_TIMEOUT_SECONDS = 180.0
GLOBAL_TIMEOUT_SECONDS = 1800.0
RSS_SAMPLE_INTERVAL_SECONDS = 0.005
RSS_CEILING_RATIO = 1.10
MIN_DECODE_GAIN_RATIO = 0.10

CELLS = (
    {"name": "baseline", "io_backend": "buffered", "scratch_policy": "ephemeral", "kernel_policy": "baseline", "thread_policy": "serial", "threads": 1},
    {"name": "persistent_fused_serial1", "io_backend": "buffered", "scratch_policy": "persistent", "kernel_policy": "fused", "thread_policy": "serial", "threads": 1},
    {"name": "persistent_fused_pool2", "io_backend": "buffered", "scratch_policy": "persistent", "kernel_policy": "fused", "thread_policy": "pool", "threads": 2},
    {"name": "mmap_persistent_fused_pool2", "io_backend": "mmap", "scratch_policy": "persistent", "kernel_policy": "fused", "thread_policy": "pool", "threads": 2},
)
SOURCE_FILES = (
    "build_msvc.bat",
    "include/qx_format.h",
    "src/qx_format.c",
    "src/qx_qxf_main.c",
)
PROFILE_KEYS = {
    "requested_io_backend", "effective_io_backend",
    "requested_scratch_policy", "effective_scratch_policy",
    "requested_kernel_policy", "effective_kernel_policy",
    "requested_thread_policy", "effective_thread_policy",
    "requested_thread_count", "effective_thread_count", "sampled_steps",
    "workers_used", "scratch_peak_capacity_bytes", "scratch_growth_events",
    "temporary_blocks_decoded", "temporary_floats_materialized",
    "temporary_bytes_materialized", "fused_final_head_dot_calls",
    "baseline_final_head_dot_calls", "final_head_q6_k_blocks",
    "final_head_parallel_jobs", "final_head_serial_jobs",
    "final_head_fallback_jobs", "full_logits_checksums",
}
COUNTER_FIELDS = (
    "workers_used", "scratch_peak_capacity_bytes", "scratch_growth_events",
    "temporary_blocks_decoded", "temporary_floats_materialized",
    "temporary_bytes_materialized", "fused_final_head_dot_calls",
    "baseline_final_head_dot_calls", "final_head_q6_k_blocks",
    "final_head_parallel_jobs", "final_head_serial_jobs",
    "final_head_fallback_jobs",
)
SUMMARY_FIELDS = (
    "wall_seconds", "native_prefill_seconds", "native_decode_seconds",
    "native_cpu_seconds", "sampled_peak_rss_bytes",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _require_positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _portable_path(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"artifact must be inside repository: {path}") from exc
    return relative.as_posix()


def artifact_record(path: Path, root: Path) -> dict[str, Any]:
    return {"path": _portable_path(path, root), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def freeze_native_inputs(root: Path, qx_exe: Path) -> dict[str, Any]:
    files = [root / item for item in SOURCE_FILES]
    for path in (*files, qx_exe):
        if not path.is_file():
            raise ValueError(f"missing frozen native input: {path}")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise ValueError("git revision must be a lowercase 40-character SHA-1")
    source_records = [artifact_record(path, root) for path in files]
    source_manifest = json.dumps(source_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "git_revision": revision,
        "source_manifest_sha256": _sha256_bytes(source_manifest),
        "source_files": source_records,
        "native_executable": artifact_record(qx_exe, root),
    }


def build_command(qx_exe: Path, model: Path, tokenizer: Path, prompt: Path, cell: dict[str, Any]) -> list[str]:
    return [
        str(qx_exe), "generate", "--in", str(model), "--tokenizer", str(tokenizer),
        "--text-file", str(prompt), "--max-tokens", "2", "--ctx", "3",
        "--io-backend", cell["io_backend"], "--scratch-policy", cell["scratch_policy"],
        "--kernel-policy", cell["kernel_policy"], "--thread-policy", cell["thread_policy"],
        "--threads", str(cell["threads"]), "--execution-profile",
    ]


def sanitized_command(cell: dict[str, Any]) -> list[str]:
    return build_command(
        Path("build/qxqxf.exe"), Path("models/Qwen3-30B-A3B-UD-IQ2_M.qxf"),
        Path("models/Qwen3-30B-A3B.qxt"), Path("<temporary-Hello-prompt>"), cell,
    )


def one_run(command: list[str], *, deadline: float) -> dict[str, Any]:
    """Run once with sampled RSS, bounded time, and deadlock-safe stdout capture."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("global campaign timeout reached before run start")
    timeout = min(PER_RUN_TIMEOUT_SECONDS, remaining)
    started = time.perf_counter()
    with tempfile.TemporaryFile() as stdout_file:
        process = subprocess.Popen(command, stdout=stdout_file, stderr=subprocess.PIPE, text=True)
        tracked = psutil.Process(process.pid)
        peak_rss = 0
        timed_out = False
        run_deadline = time.monotonic() + timeout
        while process.poll() is None:
            try:
                peak_rss = max(peak_rss, tracked.memory_info().rss)
            except psutil.Error:
                pass
            if time.monotonic() >= run_deadline:
                timed_out = True
                process.kill()
                break
            time.sleep(RSS_SAMPLE_INTERVAL_SECONDS)
        if timed_out:
            process.wait(timeout=10)
        stderr = process.stderr.read() if process.stderr else ""
        elapsed = time.perf_counter() - started
        if timed_out:
            raise TimeoutError(f"native CLI run exceeded {timeout:.3f} seconds")
        if process.returncode:
            raise RuntimeError(f"native CLI failed ({process.returncode}): {stderr[-2000:]}")
        stdout_file.seek(0)
        raw = stdout_file.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("native CLI did not emit one valid UTF-8 JSON value") from exc
    if not isinstance(payload, dict):
        raise ValueError("native CLI JSON must be an object")
    if peak_rss <= 0:
        raise ValueError("sampled peak RSS must be positive")
    return {"wall_seconds": elapsed, "sampled_peak_rss_bytes": peak_rss, "payload": payload}


def validate_and_compact_run(raw: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    payload = _require_object(raw.get("payload"), "payload")
    if set(payload) != {"prompt_token_count", "generated_token_ids", "generated_text", "stop_reason", "timing", "execution_profile"}:
        raise ValueError("native CLI payload schema mismatch")
    _require_exact_int(payload.get("prompt_token_count"), "prompt_token_count", minimum=1)
    ids = payload.get("generated_token_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError("generated_token_ids must be a non-empty array")
    ids = [_require_exact_int(item, "generated_token_ids item") for item in ids]
    text = payload.get("generated_text")
    if not isinstance(text, str):
        raise ValueError("generated_text must be a string")
    if payload.get("stop_reason") not in {"max_tokens", "eos"}:
        raise ValueError("stop_reason is invalid")
    timing = _require_object(payload.get("timing"), "timing")
    if set(timing) != {"prefill_seconds", "decode_seconds"}:
        raise ValueError("timing schema mismatch")
    prefill = _require_positive_number(timing.get("prefill_seconds"), "timing.prefill_seconds")
    decode = _require_positive_number(timing.get("decode_seconds"), "timing.decode_seconds")
    profile = _require_object(payload.get("execution_profile"), "execution_profile")
    if set(profile) != PROFILE_KEYS:
        raise ValueError("execution_profile schema mismatch")
    expected_policies = {
        "io_backend": cell["io_backend"], "scratch_policy": cell["scratch_policy"],
        "kernel_policy": cell["kernel_policy"], "thread_policy": cell["thread_policy"],
    }
    for policy, expected in expected_policies.items():
        if profile.get(f"requested_{policy}") != expected or profile.get(f"effective_{policy}") != expected:
            raise ValueError(f"execution_profile {policy} requested/effective mismatch")
    for key in ("requested_thread_count", "effective_thread_count"):
        if _require_exact_int(profile.get(key), f"execution_profile.{key}", minimum=1) != cell["threads"]:
            raise ValueError(f"execution_profile.{key} mismatch")
    sampled_steps = _require_exact_int(profile.get("sampled_steps"), "execution_profile.sampled_steps", minimum=1)
    if sampled_steps != len(ids):
        raise ValueError("execution_profile.sampled_steps does not match generated_token_ids")
    for field in COUNTER_FIELDS:
        _require_exact_int(profile.get(field), f"execution_profile.{field}")
    if profile["workers_used"] < 1 or profile["workers_used"] > cell["threads"]:
        raise ValueError("execution_profile.workers_used is outside requested bounds")
    scratch_fields = ("scratch_peak_capacity_bytes", "scratch_growth_events")
    if cell["scratch_policy"] == "persistent":
        for field in scratch_fields:
            if profile[field] <= 0:
                raise ValueError(f"execution_profile.{field} must be positive for persistent scratch")
    elif any(profile[field] != 0 for field in scratch_fields):
        raise ValueError("ephemeral scratch counters must be zero")
    if cell["kernel_policy"] == "baseline":
        if profile["baseline_final_head_dot_calls"] <= 0 or profile["fused_final_head_dot_calls"] != 0:
            raise ValueError("baseline kernel counters do not prove the requested path")
    else:
        if profile["fused_final_head_dot_calls"] <= 0 or profile["baseline_final_head_dot_calls"] != 0:
            raise ValueError("fused kernel counters do not prove the requested path")
    if cell["thread_policy"] == "serial":
        if profile["workers_used"] != 1 or profile["final_head_parallel_jobs"] != 0 or profile["final_head_serial_jobs"] <= 0:
            raise ValueError("serial counters do not prove the requested path")
    else:
        if profile["workers_used"] != cell["threads"] or profile["final_head_parallel_jobs"] <= 0 or profile["final_head_serial_jobs"] != 0 or profile["final_head_fallback_jobs"] != 0:
            raise ValueError("pool counters do not prove the requested path")
    checksums = profile.get("full_logits_checksums")
    if not isinstance(checksums, list) or len(checksums) != sampled_steps:
        raise ValueError("full_logits_checksums must cover every sampled step")
    if any(not isinstance(item, str) or not item.isdecimal() for item in checksums):
        raise ValueError("full_logits_checksums must be decimal strings")
    wall = _require_positive_number(raw.get("wall_seconds"), "wall_seconds")
    rss = _require_exact_int(raw.get("sampled_peak_rss_bytes"), "sampled_peak_rss_bytes", minimum=1)
    return {
        "wall_seconds": wall,
        "native_prefill_seconds": prefill,
        "native_decode_seconds": decode,
        "native_cpu_seconds": prefill + decode,
        "sampled_peak_rss_bytes": rss,
        "output": {"generated_token_ids": ids, "generated_text": text, "full_logits_checksums": list(checksums)},
        "execution_profile": json.loads(json.dumps(profile)),
    }


def summarize(values: list[float]) -> dict[str, Any]:
    if not isinstance(values, list) or not values:
        raise ValueError("summary values must be a non-empty list")
    checked = [_require_positive_number(value, "summary value") for value in values]
    median = statistics.median(checked)
    return {
        "count": len(checked), "min": min(checked), "median": median, "max": max(checked),
        "mad": statistics.median(abs(value - median) for value in checked),
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(runs, list) or not runs:
        raise ValueError("measured runs must be a non-empty list")
    objects = [_require_object(run, "measured run") for run in runs]
    for run in objects:
        for field in SUMMARY_FIELDS:
            if field not in run:
                raise ValueError(f"measured run missing required field: {field}")
            if type(run[field]) not in (int, float):
                raise ValueError(f"measured run field {field} must be numeric")
            _require_positive_number(run[field], f"measured run field {field}")
        _require_object(run.get("output"), "measured run output")
        _require_object(run.get("execution_profile"), "measured run execution_profile")
    return {field: summarize([float(run[field]) for run in objects]) for field in SUMMARY_FIELDS}


def validate_exact_outputs(cells: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(cells, list) or len(cells) != len(CELLS):
        raise ValueError("output gate requires the complete four-cell matrix")
    baseline_output: dict[str, Any] | None = None
    compared = 0
    for cell in cells:
        cell_obj = _require_object(cell, "benchmark cell")
        warmups = cell_obj.get("warmups")
        measured = cell_obj.get("measured")
        if not isinstance(warmups, list) or len(warmups) != WARMUPS:
            raise ValueError("each cell must contain exactly one warmup")
        if not isinstance(measured, list) or len(measured) != REPETITIONS:
            raise ValueError("each cell must contain exactly five measured runs")
        for run in warmups + measured:
            output = _require_object(_require_object(run, "benchmark run").get("output"), "benchmark run output")
            if baseline_output is None:
                baseline_output = json.loads(json.dumps(output))
            elif output != baseline_output:
                raise ValueError("generated IDs/text/full logits checksums differ from baseline")
            compared += 1
    if baseline_output is None:
        raise ValueError("output gate has no baseline")
    return {"status": "pass", "comparison": "exact", "runs_compared_including_warmups": compared, "baseline_output": baseline_output}


def build_decision(cells: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {_require_object(cell, "benchmark cell").get("name"): cell for cell in cells}
    if set(by_name) != {cell["name"] for cell in CELLS}:
        raise ValueError("decision requires the exact four benchmark cells")
    baseline = _require_object(by_name["baseline"].get("summary"), "baseline summary")
    base_decode = _require_object(baseline.get("native_decode_seconds"), "baseline decode summary")
    base_rss = _require_object(baseline.get("sampled_peak_rss_bytes"), "baseline RSS summary")
    results: list[dict[str, Any]] = []
    for definition in CELLS[1:]:
        summary = _require_object(by_name[definition["name"]].get("summary"), "candidate summary")
        decode = _require_object(summary.get("native_decode_seconds"), "candidate decode summary")
        rss = _require_object(summary.get("sampled_peak_rss_bytes"), "candidate RSS summary")
        baseline_median = _require_positive_number(base_decode.get("median"), "baseline decode median")
        candidate_median = _require_positive_number(decode.get("median"), "candidate decode median")
        decode_gain = (baseline_median - candidate_median) / baseline_median
        median_delta = baseline_median - candidate_median
        combined_mad = _require_positive_number(base_decode.get("mad"), "baseline decode MAD") if base_decode.get("mad") != 0 else 0.0
        candidate_mad = _require_positive_number(decode.get("mad"), "candidate decode MAD") if decode.get("mad") != 0 else 0.0
        beyond_mad = median_delta > combined_mad + candidate_mad
        gain_pass = decode_gain >= MIN_DECODE_GAIN_RATIO and beyond_mad
        rss_ratio = _require_positive_number(rss.get("median"), "candidate RSS median") / _require_positive_number(base_rss.get("median"), "baseline RSS median")
        rss_pass = rss_ratio <= RSS_CEILING_RATIO
        results.append({
            "cell": definition["name"], "decode_gain_ratio": decode_gain,
            "decode_median_delta_seconds": median_delta,
            "combined_decode_mad_seconds": combined_mad + candidate_mad,
            "decode_gain_at_least_10_percent_and_beyond_combined_mad": gain_pass,
            "rss_ratio_to_baseline": rss_ratio, "rss_within_10_percent_ceiling": rss_pass,
            "recommend": gain_pass and rss_pass,
        })
    recommended = [item["cell"] for item in results if item["recommend"]]
    return {
        "status": "pass", "predeclared_rule": {
            "minimum_median_decode_gain_ratio": MIN_DECODE_GAIN_RATIO,
            "must_exceed_combined_decode_mad": True,
            "maximum_median_rss_ratio_to_baseline": RSS_CEILING_RATIO,
            "recommend_only_when_decode_and_rss_pass": True,
        },
        "candidates": results, "recommended_cells": recommended,
        "default_promotion": False,
        "conclusion": "Negative results are valid; this benchmark never promotes defaults automatically.",
    }


def validate_report(report: dict[str, Any]) -> None:
    report = _require_object(report, "report")
    required = {
        "schema", "measurement", "status", "claim_scope", "fixed_contract", "provenance",
        "phase_definitions", "reproducible_command", "cells", "output_equivalence", "decision",
    }
    if set(report) != required:
        raise ValueError("report top-level schema mismatch")
    if report.get("schema") != SCHEMA or report.get("measurement") != "issue-82-native-cli-cpu-policy" or report.get("status") != "pass":
        raise ValueError("report identity/status mismatch")
    provenance = _require_object(report.get("provenance"), "provenance")
    if set(provenance) != {"platform", "psutil_version", "native_inputs_pre", "native_inputs_post", "model_qxf", "tokenizer_qxt", "prompt"}:
        raise ValueError("provenance schema mismatch")
    pre = _require_object(provenance.get("native_inputs_pre"), "native_inputs_pre")
    post = _require_object(provenance.get("native_inputs_post"), "native_inputs_post")
    for snapshot_name, snapshot in (("native_inputs_pre", pre), ("native_inputs_post", post)):
        if set(snapshot) != {"git_revision", "source_manifest_sha256", "source_files", "native_executable"}:
            raise ValueError(f"{snapshot_name} schema mismatch")
        revision = snapshot.get("git_revision")
        if not isinstance(revision, str) or len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
            raise ValueError(f"{snapshot_name}.git_revision must be a lowercase SHA-1")
        _require_sha256(snapshot.get("source_manifest_sha256"), f"{snapshot_name}.source_manifest_sha256")
        source_files = snapshot.get("source_files")
        if not isinstance(source_files, list) or len(source_files) != len(SOURCE_FILES):
            raise ValueError(f"{snapshot_name}.source_files must contain the fixed source set")
        if [item.get("path") if isinstance(item, dict) else None for item in source_files] != list(SOURCE_FILES):
            raise ValueError(f"{snapshot_name}.source_files paths mismatch")
        for index, artifact in enumerate(source_files):
            artifact = _require_object(artifact, f"{snapshot_name}.source_files[{index}]")
            if set(artifact) != {"path", "size_bytes", "sha256"}:
                raise ValueError(f"{snapshot_name}.source_files[{index}] schema mismatch")
            _require_exact_int(artifact.get("size_bytes"), f"{snapshot_name}.source_files[{index}].size_bytes", minimum=1)
            _require_sha256(artifact.get("sha256"), f"{snapshot_name}.source_files[{index}].sha256")
        executable = _require_object(snapshot.get("native_executable"), f"{snapshot_name}.native_executable")
        if set(executable) != {"path", "size_bytes", "sha256"} or executable.get("path") != "build/qxqxf.exe":
            raise ValueError(f"{snapshot_name}.native_executable schema/path mismatch")
        _require_exact_int(executable.get("size_bytes"), f"{snapshot_name}.native_executable.size_bytes", minimum=1)
        _require_sha256(executable.get("sha256"), f"{snapshot_name}.native_executable.sha256")
    if pre != post:
        raise ValueError("frozen native source/executable changed during campaign")
    for artifact_name in ("model_qxf", "tokenizer_qxt"):
        artifact = _require_object(provenance.get(artifact_name), artifact_name)
        _require_sha256(artifact.get("sha256"), f"{artifact_name}.sha256")
        if not isinstance(artifact.get("path"), str) or Path(artifact["path"]).is_absolute():
            raise ValueError(f"{artifact_name}.path must be repository-relative")
        _require_exact_int(artifact.get("size_bytes"), f"{artifact_name}.size_bytes", minimum=1)
    prompt = _require_object(provenance.get("prompt"), "prompt")
    if prompt != {"text": PROMPT_TEXT, "utf8_sha256": _sha256_bytes(PROMPT_TEXT.encode("utf-8")), "bytes": len(PROMPT_TEXT.encode("utf-8"))}:
        raise ValueError("prompt provenance mismatch")
    cells = report.get("cells")
    if not isinstance(cells, list) or len(cells) != len(CELLS):
        raise ValueError("report must contain four cells")
    expected_names = [cell["name"] for cell in CELLS]
    if [cell.get("name") if isinstance(cell, dict) else None for cell in cells] != expected_names:
        raise ValueError("report cells are missing, reordered, or renamed")
    for cell in cells:
        warmups = cell.get("warmups")
        measured = cell.get("measured")
        if not isinstance(warmups, list) or len(warmups) != WARMUPS:
            raise ValueError("report cell warmup count mismatch")
        if not isinstance(measured, list) or len(measured) != REPETITIONS:
            raise ValueError("report cell measured count mismatch")
        expected_summary = summarize_runs(measured)
        if cell.get("summary") != expected_summary:
            raise ValueError("report summary does not match measured runs")
    expected_output = validate_exact_outputs(cells)
    if report.get("output_equivalence") != expected_output:
        raise ValueError("report output equivalence gate mismatch")
    expected_decision = build_decision(cells)
    if report.get("decision") != expected_decision:
        raise ValueError("report decision gate mismatch")
    encoded = json.dumps(report, sort_keys=True)
    if str(Path.cwd().anchor).lower() in encoded.lower() or "C:/Users/" in encoded or "C:\\\\Users\\\\" in encoded:
        raise ValueError("report contains an absolute Windows path")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded Issue 82 native CLI CPU policy benchmark")
    parser.add_argument("--qx-exe", type=Path, default=Path("build/qxqxf.exe"))
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-30B-A3B-UD-IQ2_M.qxf"))
    parser.add_argument("--tokenizer", type=Path, default=Path("models/Qwen3-30B-A3B.qxt"))
    parser.add_argument("--output", type=Path, default=Path("wiki/evidence/issue-82-native-cpu-policy-report.json"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    qx_exe = (root / args.qx_exe).resolve() if not args.qx_exe.is_absolute() else args.qx_exe.resolve()
    model = (root / args.model).resolve() if not args.model.is_absolute() else args.model.resolve()
    tokenizer = (root / args.tokenizer).resolve() if not args.tokenizer.is_absolute() else args.tokenizer.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    try:
        for path in (qx_exe, model, tokenizer):
            if not path.is_file():
                raise ValueError(f"missing required artifact: {path}")
        _portable_path(qx_exe, root); _portable_path(model, root); _portable_path(tokenizer, root); _portable_path(output, root)
        if output.exists() and not args.overwrite:
            raise ValueError(f"output already exists (use --overwrite): {_portable_path(output, root)}")
        frozen_pre = freeze_native_inputs(root, qx_exe)
        deadline = time.monotonic() + GLOBAL_TIMEOUT_SECONDS
        cells: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="qx-issue82-") as temp_dir:
            prompt = Path(temp_dir) / "prompt.txt"
            prompt.write_text(PROMPT_TEXT, encoding="utf-8", newline="")
            for definition in CELLS:
                command = build_command(qx_exe, model, tokenizer, prompt, definition)
                warmups = [validate_and_compact_run(one_run(command, deadline=deadline), definition) for _ in range(WARMUPS)]
                measured = [validate_and_compact_run(one_run(command, deadline=deadline), definition) for _ in range(REPETITIONS)]
                cells.append({
                    **definition, "command": sanitized_command(definition), "warmups": warmups,
                    "measured": measured, "summary": summarize_runs(measured),
                })
        frozen_post = freeze_native_inputs(root, qx_exe)
        if frozen_pre != frozen_post:
            raise ValueError("frozen native source/executable changed during campaign")
        output_gate = validate_exact_outputs(cells)
        decision = build_decision(cells)
        report = {
            "schema": SCHEMA, "measurement": "issue-82-native-cli-cpu-policy", "status": "pass",
            "claim_scope": "One pinned native CLI executable, QXF, QXT, prompt and four fixed CPU policy cells; not a general throughput claim or automatic default promotion.",
            "fixed_contract": {
                "activation": "f32", "kv": "int8", "prompt": PROMPT_TEXT, "max_tokens": 2, "ctx": 3,
                "warmups_per_cell": WARMUPS, "measured_runs_per_cell": REPETITIONS,
                "total_runs_including_warmups": len(CELLS) * (WARMUPS + REPETITIONS),
                "per_run_timeout_seconds": PER_RUN_TIMEOUT_SECONDS, "global_timeout_seconds": GLOBAL_TIMEOUT_SECONDS,
                "rss_sample_interval_seconds": RSS_SAMPLE_INTERVAL_SECONDS,
            },
            "provenance": {
                "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "processor": platform.processor(), "python": platform.python_version()},
                "psutil_version": psutil.__version__, "native_inputs_pre": frozen_pre, "native_inputs_post": frozen_post,
                "model_qxf": artifact_record(model, root), "tokenizer_qxt": artifact_record(tokenizer, root),
                "prompt": {"text": PROMPT_TEXT, "utf8_sha256": _sha256_bytes(PROMPT_TEXT.encode("utf-8")), "bytes": len(PROMPT_TEXT.encode("utf-8"))},
            },
            "phase_definitions": {
                "wall_seconds": "End-to-end process wall time including startup, model/tokenizer I/O, native compute, JSON output, and teardown.",
                "native_prefill_seconds": "Native CPU clock timer reported by qxqxf for prompt prefill; not wall/startup time.",
                "native_decode_seconds": "Native CPU clock timer reported by qxqxf for generation decode; not wall/startup time.",
                "native_cpu_seconds": "Sum of native prefill_seconds and decode_seconds; kept separate from end-to-end wall time.",
                "sampled_peak_rss_bytes": "Maximum process RSS sampled by psutil every 5 ms.",
            },
            "reproducible_command": ["python", "scripts/native_cpu_policy_benchmark.py", "--qx-exe", "build/qxqxf.exe", "--model", "models/Qwen3-30B-A3B-UD-IQ2_M.qxf", "--tokenizer", "models/Qwen3-30B-A3B.qxt", "--output", "wiki/evidence/issue-82-native-cpu-policy-report.json", "--overwrite"],
            "cells": cells, "output_equivalence": output_gate, "decision": decision,
        }
        validate_report(report)
        encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_bytes(encoded)
        os.replace(temporary, output)
        print(json.dumps({"status": "pass", "output": _portable_path(output, root), "sha256": _sha256_bytes(encoded), "recommended_cells": decision["recommended_cells"]}, sort_keys=True))
        return 0
    except (OSError, subprocess.SubprocessError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"benchmark failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
