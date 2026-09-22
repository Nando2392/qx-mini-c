#!/usr/bin/env python
"""Issue 83: bounded real native capacity acceptance at 128 and 256 positions.

Exactly one default-policy F32/INT8 native CLI run is made for each position
budget. Prompts are constructed with the canonical QXT CLI and accepted only
when that CLI reports the exact required token count. Publication is fail-closed
on counters, profiles, finite timings, RSS, checksums, and frozen provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

try:
    import psutil
except ImportError as exc:  # pragma: no cover
    raise SystemExit("psutil is required for Issue 83 capacity acceptance") from exc

SCHEMA = "qx-issue83-native-capacity-v1"
POSITIONS = (128, 256)
MAX_TOKENS = 2
PER_CASE_TIMEOUT_SECONDS = 3600.0
GLOBAL_TIMEOUT_SECONDS = 7200.0
RSS_SAMPLE_INTERVAL_SECONDS = 0.005
VOCAB_COUNT = 151936
SOURCE_FILES = (
    "build_msvc.bat",
    "include/qx_format.h",
    "src/qx_format.c",
    "src/qx_qxf_main.c",
    "scripts/native_capacity_acceptance.py",
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
PROFILE_INTEGER_FIELDS = PROFILE_KEYS - {
    "requested_io_backend", "effective_io_backend",
    "requested_scratch_policy", "effective_scratch_policy",
    "requested_kernel_policy", "effective_kernel_policy",
    "requested_thread_policy", "effective_thread_policy",
    "full_logits_checksums",
}
CAPACITY_KEYS = {
    "executed_forward_steps", "prompt_forward_steps",
    "generated_input_forward_steps", "first_eos_output_index",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_exact_int(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def require_positive_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def portable_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"artifact must be inside repository: {path}") from exc


def artifact_record(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing required artifact: {path}")
    return {
        "path": portable_path(path, root),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def freeze_native_inputs(root: Path, qx_exe: Path) -> dict[str, Any]:
    records = [artifact_record(root / relative, root) for relative in SOURCE_FILES]
    manifest = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise ValueError("git revision must be a lowercase 40-character SHA-1")
    return {
        "git_revision": revision,
        "source_manifest_sha256": sha256_bytes(manifest),
        "source_files": records,
        "native_executable": artifact_record(qx_exe, root),
    }


def tokenizer_encode(qx_exe: Path, tokenizer: Path, prompt: Path, root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(qx_exe), "tokenizer-encode", "--tokenizer", str(tokenizer), "--text-file", str(prompt)],
        cwd=root, capture_output=True, text=True, timeout=60.0,
    )
    if completed.returncode:
        raise RuntimeError(f"tokenizer-encode failed ({completed.returncode}): {completed.stderr[-2000:]}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("tokenizer-encode did not emit valid JSON") from exc
    payload = require_object(payload, "tokenizer-encode payload")
    ids = payload.get("token_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError("tokenizer-encode token_ids must be a non-empty array")
    checked = [require_exact_int(item, "tokenizer token id") for item in ids]
    if any(item >= VOCAB_COUNT for item in checked):
        raise ValueError("tokenizer-encode emitted an out-of-range token id")
    if require_exact_int(payload.get("token_count"), "tokenizer token_count", 1) != len(checked):
        raise ValueError("tokenizer token_count does not match token_ids")
    return {"token_count": len(checked), "token_ids": checked}


def construct_exact_prompts(
    targets: tuple[int, ...], encode: Callable[[str], dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Grow a fixed ASCII pattern and retain only CLI-verified exact counts."""
    if not targets or tuple(sorted(set(targets))) != targets or targets[0] < 1:
        raise ValueError("prompt targets must be unique increasing positive integers")
    wanted = set(targets)
    found: dict[int, dict[str, Any]] = {}
    text = ""
    # The repeated pattern is intentionally simple and bounded. Counts are never
    # inferred from repetitions: every candidate is measured by tokenizer-encode.
    for repetition in range(1, targets[-1] * 3 + 1):
        text = "a" if repetition == 1 else text + " a"
        encoded = require_object(encode(text), "test-only tokenizer fixture result")
        count = require_exact_int(encoded.get("token_count"), "constructed prompt token_count", 1)
        ids = encoded.get("token_ids")
        if not isinstance(ids, list) or len(ids) != count:
            raise ValueError("constructed prompt token_ids/count mismatch")
        for token_id in ids:
            if require_exact_int(token_id, "constructed prompt token id") >= VOCAB_COUNT:
                raise ValueError("constructed prompt token id is out of range")
        if count in wanted and count not in found:
            found[count] = {"text": text, "token_count": count, "token_ids": list(ids)}
        if count > targets[-1]:
            break
        if len(found) == len(targets):
            break
    missing = [target for target in targets if target not in found]
    if missing:
        raise ValueError(f"repeated ASCII pattern did not tokenize to exact targets: {missing}")
    return found


def generation_command(qx_exe: Path, model: Path, tokenizer: Path, prompt: Path, positions: int) -> list[str]:
    return [
        str(qx_exe), "generate", "--in", str(model), "--tokenizer", str(tokenizer),
        "--text-file", str(prompt), "--max-tokens", str(MAX_TOKENS), "--ctx", str(positions),
        "--io-backend", "buffered", "--scratch-policy", "ephemeral",
        "--kernel-policy", "baseline", "--thread-policy", "serial", "--threads", "1",
        "--execution-profile", "--capacity-profile",
    ]


def sanitized_command(positions: int) -> list[str]:
    return generation_command(
        Path("build/qxqxf.exe"), Path("models/Qwen3-30B-A3B-UD-IQ2_M.qxf"),
        Path("models/Qwen3-30B-A3B.qxt"), Path(f"<temporary-{positions - 1}-token-prompt>"), positions,
    )


def one_run(command: list[str], deadline: float) -> dict[str, Any]:
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        raise TimeoutError("global acceptance timeout reached before case start")
    timeout = min(PER_CASE_TIMEOUT_SECONDS, remaining)
    started = time.perf_counter()
    with tempfile.TemporaryFile() as stdout_file:
        process = subprocess.Popen(command, stdout=stdout_file, stderr=subprocess.PIPE, text=True)
        tracked = psutil.Process(process.pid)
        peak_rss = 0
        timed_out = False
        case_deadline = time.monotonic() + timeout
        while process.poll() is None:
            try:
                peak_rss = max(peak_rss, tracked.memory_info().rss)
            except psutil.Error:
                pass
            if time.monotonic() >= case_deadline:
                timed_out = True
                process.kill()
                break
            time.sleep(RSS_SAMPLE_INTERVAL_SECONDS)
        if timed_out:
            process.wait(timeout=10)
        stderr = process.stderr.read() if process.stderr else ""
        wall = time.perf_counter() - started
        if timed_out:
            raise TimeoutError(f"native capacity case exceeded {timeout:.3f} seconds")
        if process.returncode:
            raise RuntimeError(f"native capacity case failed ({process.returncode}): {stderr[-2000:]}")
        stdout_file.seek(0)
        raw = stdout_file.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("native capacity case did not emit one valid UTF-8 JSON value") from exc
    if peak_rss <= 0:
        raise ValueError("sampled peak RSS must be positive")
    return {"payload": require_object(payload, "native payload"), "wall_seconds": wall, "sampled_peak_rss_bytes": peak_rss}


def validate_and_compact_run(raw: dict[str, Any], positions: int) -> dict[str, Any]:
    prompt_target = positions - 1
    payload = require_object(raw.get("payload"), "payload")
    expected_payload_keys = {
        "prompt_token_count", "generated_token_ids", "generated_text", "stop_reason",
        "timing", "execution_profile", "capacity_profile",
    }
    if set(payload) != expected_payload_keys:
        raise ValueError("native CLI payload schema mismatch")
    if require_exact_int(payload.get("prompt_token_count"), "prompt_token_count", 1) != prompt_target:
        raise ValueError("prompt_token_count does not match exact constructed target")
    ids = payload.get("generated_token_ids")
    if not isinstance(ids, list) or len(ids) != MAX_TOKENS:
        raise ValueError("generated_token_ids must contain exactly two tokens; early EOS is a gate failure")
    ids = [require_exact_int(item, "generated token id") for item in ids]
    if any(item >= VOCAB_COUNT for item in ids):
        raise ValueError("generated token id is out of vocabulary range")
    if not isinstance(payload.get("generated_text"), str):
        raise ValueError("generated_text must be a string")
    if payload.get("stop_reason") != "max_tokens":
        raise ValueError("early EOS is a gate failure for position acceptance")
    timing = require_object(payload.get("timing"), "timing")
    if set(timing) != {"prefill_seconds", "decode_seconds"}:
        raise ValueError("timing schema mismatch")
    prefill = require_positive_finite(timing.get("prefill_seconds"), "timing.prefill_seconds")
    decode = require_positive_finite(timing.get("decode_seconds"), "timing.decode_seconds")

    profile = require_object(payload.get("execution_profile"), "execution_profile")
    if set(profile) != PROFILE_KEYS:
        raise ValueError("execution_profile schema mismatch")
    expected_policies = {
        "io_backend": "buffered", "scratch_policy": "ephemeral",
        "kernel_policy": "baseline", "thread_policy": "serial",
    }
    for policy, expected in expected_policies.items():
        if profile.get(f"requested_{policy}") != expected or profile.get(f"effective_{policy}") != expected:
            raise ValueError(f"execution_profile {policy} mismatch")
    for field in PROFILE_INTEGER_FIELDS:
        require_exact_int(profile.get(field), f"execution_profile.{field}")
    if profile["requested_thread_count"] != 1 or profile["effective_thread_count"] != 1:
        raise ValueError("execution_profile thread count mismatch")
    if profile["sampled_steps"] != MAX_TOKENS or profile["workers_used"] != 1:
        raise ValueError("execution_profile sampled steps/workers mismatch")
    if profile["scratch_peak_capacity_bytes"] != 0 or profile["scratch_growth_events"] != 0:
        raise ValueError("ephemeral scratch counters must be zero")
    if profile["baseline_final_head_dot_calls"] <= 0 or profile["fused_final_head_dot_calls"] != 0:
        raise ValueError("baseline counters do not prove the requested kernel path")
    if profile["final_head_parallel_jobs"] != 0 or profile["final_head_serial_jobs"] <= 0:
        raise ValueError("serial counters do not prove the requested thread path")
    checksums = profile.get("full_logits_checksums")
    if not isinstance(checksums, list) or len(checksums) != MAX_TOKENS:
        raise ValueError("full_logits_checksums must contain exactly two entries")
    if any(not isinstance(item, str) or not item.isdecimal() or int(item) < 0 for item in checksums):
        raise ValueError("full_logits_checksums must be unsigned decimal strings")

    capacity = require_object(payload.get("capacity_profile"), "capacity_profile")
    if set(capacity) != CAPACITY_KEYS:
        raise ValueError("capacity_profile schema mismatch")
    if capacity.get("first_eos_output_index") is not None:
        raise ValueError("first_eos_output_index must be null for accepted capacity cases")
    expected_capacity = {
        "executed_forward_steps": positions,
        "prompt_forward_steps": prompt_target,
        "generated_input_forward_steps": 1,
    }
    for field, expected in expected_capacity.items():
        if require_exact_int(capacity.get(field), f"capacity_profile.{field}") != expected:
            raise ValueError(f"capacity_profile.{field} does not prove position {positions}")
    wall = require_positive_finite(raw.get("wall_seconds"), "wall_seconds")
    rss = require_exact_int(raw.get("sampled_peak_rss_bytes"), "sampled_peak_rss_bytes", 1)
    return {
        "positions": positions,
        "prompt_token_count": prompt_target,
        "generated_token_ids": ids,
        "generated_text": payload["generated_text"],
        "stop_reason": payload["stop_reason"],
        "timing": {"prefill_seconds": prefill, "decode_seconds": decode, "native_cpu_seconds": prefill + decode},
        "wall_seconds": wall,
        "sampled_peak_rss_bytes": rss,
        "execution_profile": json.loads(json.dumps(profile)),
        "capacity_profile": json.loads(json.dumps(capacity)),
    }


def validate_report(report: dict[str, Any]) -> None:
    required = {"schema", "issue", "status", "claim_scope", "fixed_contract", "provenance", "reproduction", "cases", "gates"}
    if set(require_object(report, "report")) != required:
        raise ValueError("report top-level schema mismatch")
    if report.get("schema") != SCHEMA or report.get("issue") != 83 or report.get("status") != "pass":
        raise ValueError("report identity/status mismatch")
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != 2 or [case.get("positions") for case in cases if isinstance(case, dict)] != list(POSITIONS):
        raise ValueError("report must contain exactly the ordered 128/256 cases")
    for case, position in zip(cases, POSITIONS):
        validate_and_compact_run({
            "payload": {
                "prompt_token_count": case["prompt_token_count"],
                "generated_token_ids": case["generated_token_ids"],
                "generated_text": case["generated_text"],
                "stop_reason": case["stop_reason"],
                "timing": {"prefill_seconds": case["timing"]["prefill_seconds"], "decode_seconds": case["timing"]["decode_seconds"]},
                "execution_profile": case["execution_profile"],
                "capacity_profile": case["capacity_profile"],
            },
            "wall_seconds": case["wall_seconds"],
            "sampled_peak_rss_bytes": case["sampled_peak_rss_bytes"],
        }, position)
    provenance = require_object(report.get("provenance"), "provenance")
    if provenance.get("native_inputs_pre") != provenance.get("native_inputs_post"):
        raise ValueError("frozen native inputs changed during acceptance")
    encoded = json.dumps(report, sort_keys=True)
    if "C:/Users/" in encoded or "C:\\\\Users\\\\" in encoded:
        raise ValueError("report contains an absolute Windows path")


def reproduction_metadata() -> dict[str, Any]:
    return {
        "working_directory": "repository_root",
        "python": "Python 3.11 environment with psutil installed; select its interpreter explicitly",
        "command": ["python", "scripts/native_capacity_acceptance.py", "--qx-exe", "build/qxqxf.exe", "--model", "models/Qwen3-30B-A3B-UD-IQ2_M.qxf", "--tokenizer", "models/Qwen3-30B-A3B.qxt", "--output", "wiki/evidence/issue-83-native-capacity-report.json"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue 83 real native 128/256-position acceptance")
    parser.add_argument("--qx-exe", type=Path, default=Path("build/qxqxf.exe"))
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-30B-A3B-UD-IQ2_M.qxf"))
    parser.add_argument("--tokenizer", type=Path, default=Path("models/Qwen3-30B-A3B.qxt"))
    parser.add_argument("--output", type=Path, default=Path("wiki/evidence/issue-83-native-capacity-report.json"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda value: value.resolve() if value.is_absolute() else (root / value).resolve()
    qx_exe, model, tokenizer, output = map(resolve, (args.qx_exe, args.model, args.tokenizer, args.output))
    try:
        for path in (qx_exe, model, tokenizer):
            if not path.is_file():
                raise ValueError(f"missing required artifact: {path}")
            portable_path(path, root)
        portable_path(output, root)
        if output.exists() and not args.overwrite:
            raise ValueError(f"output already exists (use --overwrite): {portable_path(output, root)}")
        deadline = time.monotonic() + GLOBAL_TIMEOUT_SECONDS
        with tempfile.TemporaryDirectory(prefix="qx-issue83-") as temp_dir:
            prompt_path = Path(temp_dir) / "prompt.txt"
            def encode_text(text: str) -> dict[str, Any]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("global acceptance timeout reached during prompt construction")
                prompt_path.write_text(text, encoding="utf-8", newline="")
                return tokenizer_encode(qx_exe, tokenizer, prompt_path, root)
            prompts = construct_exact_prompts(tuple(position - 1 for position in POSITIONS), encode_text)
            frozen_pre = freeze_native_inputs(root, qx_exe)
            model_record = artifact_record(model, root)
            tokenizer_record = artifact_record(tokenizer, root)
            cases: list[dict[str, Any]] = []
            prompt_records: list[dict[str, Any]] = []
            for positions in POSITIONS:
                prompt = prompts[positions - 1]
                prompt_path.write_text(prompt["text"], encoding="utf-8", newline="")
                confirmed = tokenizer_encode(qx_exe, tokenizer, prompt_path, root)
                if confirmed != {"token_count": prompt["token_count"], "token_ids": prompt["token_ids"]}:
                    raise ValueError("prompt tokenization changed between construction and execution")
                raw = one_run(generation_command(qx_exe, model, tokenizer, prompt_path, positions), deadline)
                case = validate_and_compact_run(raw, positions)
                case["command"] = sanitized_command(positions)
                cases.append(case)
                prompt_bytes = prompt["text"].encode("utf-8")
                prompt_records.append({
                    "positions": positions, "token_count": prompt["token_count"],
                    "bytes": len(prompt_bytes), "utf8_sha256": sha256_bytes(prompt_bytes),
                    "token_ids_sha256": sha256_bytes(json.dumps(prompt["token_ids"], separators=(",", ":")).encode("ascii")),
                    "construction": "repeated ASCII 'a' pattern, each candidate measured by canonical qxqxf tokenizer-encode",
                })
        frozen_post = freeze_native_inputs(root, qx_exe)
        if frozen_pre != frozen_post:
            raise ValueError("frozen native inputs changed during acceptance")
        report = {
            "schema": SCHEMA,
            "issue": 83,
            "status": "pass",
            "claim_scope": "Two finite runs on one pinned native CPU executable/model/tokenizer: actual 128 and 256 forward positions only; no quality, semantic, global parity, throughput, or default-promotion claim.",
            "fixed_contract": {
                "positions": list(POSITIONS), "runs_per_position": 1, "total_native_runs": 2,
                "layers": 48, "activation": "f32", "kv": "int8", "sampling": "greedy",
                "max_tokens": MAX_TOKENS, "io_backend": "buffered", "scratch_policy": "ephemeral",
                "kernel_policy": "baseline", "thread_policy": "serial", "threads": 1,
                "per_case_timeout_seconds": PER_CASE_TIMEOUT_SECONDS,
                "global_timeout_seconds": GLOBAL_TIMEOUT_SECONDS,
                "rss_sample_interval_seconds": RSS_SAMPLE_INTERVAL_SECONDS,
            },
            "provenance": {
                "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "processor": platform.processor(), "python": platform.python_version()},
                "psutil_version": psutil.__version__, "native_inputs_pre": frozen_pre, "native_inputs_post": frozen_post,
                "model_qxf": model_record, "tokenizer_qxt": tokenizer_record, "prompts": prompt_records,
            },
            "reproduction": reproduction_metadata(),
            "cases": cases,
            "gates": {
                "status": "pass", "ordered_positions": list(POSITIONS), "all_prompt_tokens_consumed": True,
                "generated_input_forward_steps_each": 1, "checksum_count_each": 2,
                "early_eos": False, "finite_timings_and_positive_sampled_rss": True,
            },
        }
        validate_report(report)
        encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_bytes(encoded)
        os.replace(temporary, output)
        print(json.dumps({"status": "pass", "output": portable_path(output, root), "sha256": sha256_bytes(encoded), "positions": list(POSITIONS)}, sort_keys=True))
        return 0
    except (OSError, subprocess.SubprocessError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"capacity acceptance failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
