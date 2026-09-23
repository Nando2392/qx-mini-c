#!/usr/bin/env python
"""Issue 84 bounded, durable native CPU 4096-position acceptance runner.

A journal is crash evidence, not computation resume.  This program never retries
or resumes native inference and never makes quality, throughput, or promotion claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import psutil

SCHEMA = "qx-issue84-native-cpu4k-v1"
POSITIONS = 4096
PROMPT_TOKENS = 4095
TOKENIZER_MAX_INPUT_BYTES = 4096
PROMPT_PROBE_PATTERNS = ("~!", "~?", "~;", "~|", "~^", "~\n", "|\n", "^\n")
MAX_TOKENS = 2
NATIVE_DEADLINE_SECONDS = 8 * 60 * 60
RSS_CEILING_BYTES = 512 * 1024 * 1024
RSS_SAMPLE_INTERVAL_SECONDS = 0.1
HEARTBEAT_SECONDS = 10.0
VOCAB_COUNT = 151936
SOURCE_FILES = (
    "build_msvc.bat", "include/qx_format.h", "include/qx_tokenizer.h",
    "src/qx_format.c", "src/qx_qxf_main.c", "src/qx_tokenizer.c",
    "scripts/native_capacity_acceptance.py", "scripts/native_4k_acceptance.py",
)
PROFILE_KEYS = {
    "requested_io_backend", "effective_io_backend", "requested_scratch_policy",
    "effective_scratch_policy", "requested_kernel_policy", "effective_kernel_policy",
    "requested_thread_policy", "effective_thread_policy", "requested_thread_count",
    "effective_thread_count", "sampled_steps", "workers_used", "scratch_peak_capacity_bytes",
    "scratch_growth_events", "temporary_blocks_decoded", "temporary_floats_materialized",
    "temporary_bytes_materialized", "fused_final_head_dot_calls",
    "baseline_final_head_dot_calls", "final_head_q6_k_blocks", "final_head_parallel_jobs",
    "final_head_serial_jobs", "final_head_fallback_jobs", "full_logits_checksums",
}
CAPACITY_KEYS = {"executed_forward_steps", "prompt_forward_steps", "generated_input_forward_steps", "first_eos_output_index"}
FIXED_CONTRACT = {
    "positions": 4096, "prompt_tokens": 4095, "max_tokens": 2, "ctx": 4096,
    "runs": 1, "activation": "f32", "kv": "int8", "io_backend": "buffered",
    "scratch_policy": "ephemeral", "kernel_policy": "baseline",
    "thread_policy": "serial", "threads": 1, "native_deadline_seconds": 28800,
    "rss_ceiling_bytes": 536870912, "rss_sample_interval_seconds": 0.1,
    "heartbeat_seconds": 10.0,
}

class RunGateError(RuntimeError):
    pass


def _int(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"artifact must be inside repository: {path}") from exc


def artifact_record(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing required artifact: {path}")
    return {"path": portable_path(path, root), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def create_output_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    return path


def tokenizer_encode(executable: Path, tokenizer: Path, prompt: Path, root: Path) -> dict[str, Any]:
    result = subprocess.run([str(executable), "tokenizer-encode", "--tokenizer", str(tokenizer), "--text-file", str(prompt)],
                            cwd=root, capture_output=True, text=True, timeout=120, check=False)
    if result.returncode:
        raise RuntimeError(f"tokenizer-encode failed ({result.returncode}): {result.stderr[-2000:]}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("tokenizer-encode did not emit JSON") from exc
    ids = payload.get("token_ids") if isinstance(payload, dict) else None
    count = _int(payload.get("token_count") if isinstance(payload, dict) else None, "token_count", 1)
    if not isinstance(ids, list) or len(ids) != count or any(_int(x, "token id") >= VOCAB_COUNT for x in ids):
        raise ValueError("tokenizer token_ids/count invalid")
    return {"token_count": count, "token_ids": ids}


def construct_exact_prompt(target: int, encode: Callable[[str], dict[str, Any]], max_attempts: int = 8) -> dict[str, Any]:
    """Try at most eight byte-bounded ASCII patterns with the canonical tokenizer."""
    target = _int(target, "target", 1)
    max_attempts = _int(max_attempts, "max_attempts", 1)
    if target > TOKENIZER_MAX_INPUT_BYTES:
        raise ValueError("target cannot fit within tokenizer 4096-byte input limit")
    observed: list[str] = []
    for attempt, pattern in enumerate(PROMPT_PROBE_PATTERNS[:max_attempts], 1):
        text = (pattern * ((target + len(pattern) - 1) // len(pattern)))[:target]
        if len(text.encode("utf-8")) > TOKENIZER_MAX_INPUT_BYTES:
            raise ValueError("constructed prompt exceeds tokenizer 4096-byte input limit")
        encoded = encode(text)
        count = _int(encoded.get("token_count"), "constructed token_count", 1)
        ids = encoded.get("token_ids")
        if not isinstance(ids, list) or len(ids) != count:
            raise ValueError("constructed token_ids/count mismatch")
        observed.append(f"{pattern.encode('ascii').hex()}:{count}")
        if count == target:
            return {"text": text, "token_count": count, "token_ids": list(ids), "attempts": attempt}
    raise ValueError(f"no byte-bounded pattern reached exactly {target} tokens in <= {max_attempts} canonical attempts ({','.join(observed)})")


def _journal(output_dir: Path, phase: str, status: str, started: float, peak: int, **extra: Any) -> None:
    value = {"schema": SCHEMA, "phase": phase, "status": status,
             "elapsed_seconds": max(0.0, time.monotonic() - started), "peak_rss_bytes": peak,
             "heartbeat_interval_seconds": HEARTBEAT_SECONDS, "computation_resume_supported": False}
    value.update(extra)
    atomic_json(output_dir / "phase-journal.json", value)


def _stop_tree(process: subprocess.Popen[Any], grace: float = 2.0, *, job_handle: int | None = None) -> None:
    """Kill the contained tree and verify every process observed in the job is gone."""
    identities: dict[int, psutil.Process] = {}
    kernel32 = None
    if os.name == "nt" and job_handle:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        for capacity in (64, 1024, 16384):
            size = 8 + capacity * ctypes.sizeof(ctypes.c_size_t)
            buffer = ctypes.create_string_buffer(size)
            if kernel32.QueryInformationJobObject(wintypes.HANDLE(job_handle), 3, buffer, size, None):
                count = ctypes.c_uint32.from_buffer(buffer, 4).value
                values = (ctypes.c_size_t * count).from_buffer(buffer, 8)
                for value in values:
                    try: identities[int(value)] = psutil.Process(int(value))
                    except psutil.Error: pass
                break
            if ctypes.get_last_error() != 234:
                raise RunGateError(f"process-tree enumeration failed: winerror {ctypes.get_last_error()}")
        else:
            raise RunGateError("process-tree enumeration exceeded bounded capacity")
    try:
        parent = psutil.Process(process.pid)
        for item in [parent, *parent.children(recursive=True)]: identities.setdefault(item.pid, item)
    except psutil.Error:
        pass
    if kernel32 is not None:
        if not kernel32.CloseHandle(wintypes.HANDLE(job_handle)):
            raise RunGateError(f"closing kill-on-close job failed: winerror {ctypes.get_last_error()}")
    else:
        if os.name != "nt":
            try: os.killpg(process.pid, signal.SIGTERM)
            except OSError: pass
        else:
            try: process.terminate()
            except OSError: pass
        _, alive = psutil.wait_procs(list(identities.values()), timeout=grace)
        for item in alive:
            try: item.kill()
            except psutil.Error: pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try: process.kill(); process.wait(timeout=grace)
        except (OSError, subprocess.TimeoutExpired): pass
    _, alive = psutil.wait_procs(list(identities.values()), timeout=grace)
    if process.poll() is None or alive:
        raise RunGateError(f"process-tree cleanup failed; surviving PIDs: {sorted(item.pid for item in alive)}")


def run_supervised(command: list[str], output_dir: Path, *, deadline_seconds: float = NATIVE_DEADLINE_SECONDS,
                   rss_ceiling_bytes: int = RSS_CEILING_BYTES,
                   sample_interval_seconds: float = RSS_SAMPLE_INTERVAL_SECONDS,
                   heartbeat_seconds: float = HEARTBEAT_SECONDS,
                   cancel_requested: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Run one child in a Windows kill-on-close job, with durable binary spools."""
    stdout_path, stderr_path = output_dir / "stdout.bin", output_dir / "stderr.bin"
    started = time.monotonic(); peak = 0; process: subprocess.Popen[Any] | None = None
    job_handle: int | None = None; status = "failed"
    _journal(output_dir, "native", "running", started, peak)
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            if os.name == "nt":
                import ctypes
                from ctypes import wintypes
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
                kernel32.CreateJobObjectW.restype = wintypes.HANDLE
                kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
                kernel32.SetInformationJobObject.restype = wintypes.BOOL
                kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
                kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
                kernel32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                kernel32.OpenThread.restype = wintypes.HANDLE
                kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
                kernel32.ResumeThread.restype = wintypes.DWORD
                kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
                kernel32.CloseHandle.restype = wintypes.BOOL
                class BASIC_LIMIT(ctypes.Structure):
                    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong), ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]
                class IO_COUNTERS(ctypes.Structure):
                    _fields_ = [(name, ctypes.c_ulonglong) for name in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]
                class EXTENDED_LIMIT(ctypes.Structure):
                    _fields_ = [("BasicLimitInformation", BASIC_LIMIT), ("IoInfo", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]
                raw_job = kernel32.CreateJobObjectW(None, None)
                job_handle = int(raw_job) if raw_job else None
                if not job_handle: raise RunGateError(f"CreateJobObject failed: winerror {ctypes.get_last_error()}")
                limits = EXTENDED_LIMIT(); limits.BasicLimitInformation.LimitFlags = 0x2000
                if not kernel32.SetInformationJobObject(wintypes.HANDLE(job_handle), 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                    error = ctypes.get_last_error(); kernel32.CloseHandle(wintypes.HANDLE(job_handle)); job_handle = None
                    raise RunGateError(f"SetInformationJobObject failed: winerror {error}")
                # WinBase.h CREATE_SUSPENDED is not exported by every Python build.
                create_suspended = 0x00000004
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr, creationflags=create_suspended | subprocess.CREATE_NEW_PROCESS_GROUP)
                if not kernel32.AssignProcessToJobObject(wintypes.HANDLE(job_handle), wintypes.HANDLE(process._handle)):
                    error = ctypes.get_last_error(); process.kill(); process.wait(); kernel32.CloseHandle(wintypes.HANDLE(job_handle)); job_handle = None
                    raise RunGateError(f"AssignProcessToJobObject failed: winerror {error}")
                threads = psutil.Process(process.pid).threads()
                if len(threads) != 1:
                    _stop_tree(process, job_handle=job_handle); job_handle = None
                    raise RunGateError("suspended child did not expose exactly one initial thread")
                thread = kernel32.OpenThread(0x0002, False, threads[0].id)
                resumed = kernel32.ResumeThread(thread) if thread else 0xFFFFFFFF
                if not thread or resumed == 0xFFFFFFFF:
                    error = ctypes.get_last_error()
                    if thread: kernel32.CloseHandle(thread)
                    _stop_tree(process, job_handle=job_handle); job_handle = None
                    raise RunGateError(f"ResumeThread failed: winerror {error}")
                kernel32.CloseHandle(thread)
            else:
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            tracked = psutil.Process(process.pid); next_heartbeat = started + heartbeat_seconds
            while process.poll() is None:
                try:
                    rss = tracked.memory_info().rss + sum((c.memory_info().rss for c in tracked.children(recursive=True)), 0)
                    peak = max(peak, rss)
                except psutil.Error: pass
                now = time.monotonic()
                if peak > rss_ceiling_bytes: status = "rss_exceeded"; raise RunGateError("sampled RSS ceiling exceeded")
                if cancel_requested and cancel_requested(): status = "cancelled"; raise RunGateError("native run cancelled")
                if now - started >= deadline_seconds: status = "timeout"; raise RunGateError("native deadline exceeded")
                if now >= next_heartbeat:
                    _journal(output_dir, "native", "running", started, peak); next_heartbeat = now + heartbeat_seconds
                time.sleep(sample_interval_seconds)
            if process.returncode:
                status = "failed"; raise RunGateError(f"native child failed with exit code {process.returncode}")
            if job_handle:
                class ACCOUNTING(ctypes.Structure):
                    _fields_ = [(name, ctypes.c_longlong) for name in ("TotalUserTime", "TotalKernelTime", "ThisPeriodTotalUserTime", "ThisPeriodTotalKernelTime")] + [(name, wintypes.DWORD) for name in ("TotalPageFaultCount", "TotalProcesses", "ActiveProcesses", "TotalTerminatedProcesses")]
                accounting = ACCOUNTING()
                if not kernel32.QueryInformationJobObject(wintypes.HANDLE(job_handle), 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None):
                    raise RunGateError(f"job accounting failed: winerror {ctypes.get_last_error()}")
                if accounting.ActiveProcesses:
                    status = "descendants_survived_parent"; raise RunGateError("native parent exited while a descendant remained alive")
        raw = stdout_path.read_bytes()
        try: payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise RunGateError("native stdout is not one UTF-8 JSON value") from exc
        result = {"payload": payload, "wall_seconds": time.monotonic() - started, "sampled_peak_rss_bytes": peak, "stdout": artifact_record(stdout_path, output_dir), "stderr": artifact_record(stderr_path, output_dir)}
        try: _journal(output_dir, "native", "completed", started, peak, returncode=process.returncode)
        except Exception: status = "failed"; raise
        status = "completed"
        if job_handle:
            if not kernel32.CloseHandle(wintypes.HANDLE(job_handle)):
                status = "cleanup_failed"; raise RunGateError(f"closing completed job failed: winerror {ctypes.get_last_error()}")
            job_handle = None
        return result
    except KeyboardInterrupt as exc:
        status = "cancelled"; raise RunGateError("native run cancelled") from exc
    finally:
        if process is not None and status != "completed":
            try: _stop_tree(process, job_handle=job_handle); job_handle = None
            except Exception:
                status = "cleanup_failed"
                _journal(output_dir, "native", status, started, peak, returncode=process.poll())
                raise
        if status != "completed":
            _journal(output_dir, "native", status, started, peak, returncode=None if process is None else process.poll())

def generation_command(exe: Path, model: Path, tokenizer: Path, prompt: Path) -> list[str]:
    return [str(exe), "generate", "--in", str(model), "--tokenizer", str(tokenizer), "--text-file", str(prompt),
            "--max-tokens", "2", "--ctx", "4096", "--io-backend", "buffered", "--scratch-policy", "ephemeral",
            "--kernel-policy", "baseline", "--thread-policy", "serial", "--threads", "1", "--execution-profile", "--capacity-profile"]


def validate_native_run(raw: dict[str, Any]) -> dict[str, Any]:
    payload = raw.get("payload")
    if not isinstance(payload, dict) or set(payload) != {"prompt_token_count", "generated_token_ids", "generated_text", "stop_reason", "timing", "execution_profile", "capacity_profile"}:
        raise ValueError("native payload schema mismatch")
    if _int(payload["prompt_token_count"], "prompt_token_count", 1) != PROMPT_TOKENS: raise ValueError("prompt count must be 4095")
    ids = payload["generated_token_ids"]
    if not isinstance(ids, list) or len(ids) != 2: raise ValueError("exactly two generated token IDs required; early EOS")
    ids = [_int(x, "generated token id") for x in ids]
    if any(x >= VOCAB_COUNT for x in ids): raise ValueError("generated token ID out of range")
    if payload["stop_reason"] != "max_tokens": raise ValueError("EOS is not accepted")
    if not isinstance(payload["generated_text"], str): raise ValueError("generated_text must be string")
    timing = payload["timing"]
    if not isinstance(timing, dict) or set(timing) != {"prefill_seconds", "decode_seconds"}: raise ValueError("timing schema mismatch")
    prefill, decode = _positive(timing["prefill_seconds"], "prefill timing"), _positive(timing["decode_seconds"], "decode timing")
    profile = payload["execution_profile"]
    if not isinstance(profile, dict) or set(profile) != PROFILE_KEYS: raise ValueError("execution profile schema mismatch")
    for name, expected in (("io_backend", "buffered"), ("scratch_policy", "ephemeral"), ("kernel_policy", "baseline"), ("thread_policy", "serial")):
        if profile[f"requested_{name}"] != expected or profile[f"effective_{name}"] != expected: raise ValueError(f"{name} policy mismatch")
    integer_fields = PROFILE_KEYS - {k for k in PROFILE_KEYS if k.startswith("requested_") or k.startswith("effective_")} - {"full_logits_checksums"}
    for key in integer_fields: _int(profile[key], f"execution_profile.{key}")
    if profile["requested_thread_count"] != 1 or profile["effective_thread_count"] != 1 or profile["workers_used"] != 1: raise ValueError("thread contract mismatch")
    if profile["sampled_steps"] != 2: raise ValueError("sampled_steps must be two")
    if profile["scratch_peak_capacity_bytes"] or profile["scratch_growth_events"]: raise ValueError("ephemeral counters mismatch")
    if profile["fused_final_head_dot_calls"] or profile["baseline_final_head_dot_calls"] <= 0: raise ValueError("baseline counters mismatch")
    if profile["final_head_parallel_jobs"] or profile["final_head_serial_jobs"] <= 0: raise ValueError("serial counters mismatch")
    sums = profile["full_logits_checksums"]
    if not isinstance(sums, list) or len(sums) != 2 or any(not isinstance(x, str) or not x.isdecimal() for x in sums): raise ValueError("exactly two unsigned checksums required")
    capacity = payload["capacity_profile"]
    if not isinstance(capacity, dict) or set(capacity) != CAPACITY_KEYS: raise ValueError("capacity profile schema mismatch")
    if capacity["first_eos_output_index"] is not None: raise ValueError("EOS output index must be null")
    for key, expected in (("executed_forward_steps", 4096), ("prompt_forward_steps", 4095), ("generated_input_forward_steps", 1)):
        if _int(capacity[key], key) != expected: raise ValueError(f"{key} must prove {expected if key != 'executed_forward_steps' else 4096}")
    wall, rss = _positive(raw.get("wall_seconds"), "wall_seconds"), _int(raw.get("sampled_peak_rss_bytes"), "sampled RSS", 1)
    if rss > RSS_CEILING_BYTES: raise ValueError("sampled RSS ceiling exceeded")
    artifacts = {}
    for name in ("stdout", "stderr"):
        record = raw.get(name)
        if not isinstance(record, dict) or set(record) != {"path", "size_bytes", "sha256"} or not isinstance(record["path"], str) or len(record["sha256"]) != 64:
            raise ValueError(f"{name} artifact shape mismatch")
        _int(record["size_bytes"], f"{name}.size_bytes")
        artifacts[name] = dict(record)
    return {"positions": 4096, "prompt_token_count": 4095, "generated_token_ids": ids, "generated_text": payload["generated_text"],
            "stop_reason": "max_tokens", "timing": {"prefill_seconds": prefill, "decode_seconds": decode, "native_cpu_seconds": prefill + decode},
            "wall_seconds": wall, "sampled_peak_rss_bytes": rss, "execution_profile": json.loads(json.dumps(profile)),
            "capacity_profile": json.loads(json.dumps(capacity)), "artifacts": artifacts}


CLAIM_SCOPE = "One native CPU run proving exactly 4096 executed positions only; no quality, throughput, thermal, soak, CUDA, or default-promotion claim."
PASS_GATES = {"status": "pass", "exact_4096_counters": True, "two_outputs_and_checksums": True,
              "no_early_eos": True, "rss_within_ceiling": True, "inputs_unchanged": True}
PLATFORM_KEYS = {"system", "release", "machine", "processor", "python", "psutil"}
INPUT_KEYS = {"git_revision", "source_files", "source_manifest_sha256", "native_executable", "model_qxf", "tokenizer_qxt"}
ARTIFACT_KEYS = {"path", "size_bytes", "sha256"}


def _hex64(value: object, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be lowercase SHA-256")


def _artifact(record: object, label: str, *, expected_path: str | None = None) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != ARTIFACT_KEYS:
        raise ValueError(f"{label} artifact schema mismatch")
    path = record.get("path")
    if not isinstance(path, str) or not path or "\\" in path or "//" in path:
        raise ValueError(f"{label}.path must be a non-empty canonical repository-relative POSIX path")
    parts = path.split("/")
    if path.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"{label}.path must be confined and traversal-free")
    if len(parts[0]) >= 2 and parts[0][1] == ":":
        raise ValueError(f"{label}.path must not be an absolute Windows path")
    if expected_path is not None and path != expected_path:
        raise ValueError(f"{label}.path mismatch")
    _int(record.get("size_bytes"), f"{label}.size_bytes")
    _hex64(record.get("sha256"), f"{label}.sha256")
    return record


def _validate_inputs(inputs: object, label: str) -> dict[str, Any]:
    if not isinstance(inputs, dict) or set(inputs) != INPUT_KEYS:
        raise ValueError(f"{label} schema mismatch")
    revision = inputs.get("git_revision")
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(f"{label}.git_revision must be a lowercase SHA-1")
    records = inputs.get("source_files")
    if not isinstance(records, list) or len(records) != len(SOURCE_FILES):
        raise ValueError(f"{label}.source_files must be the complete source manifest")
    for index, (record, expected) in enumerate(zip(records, SOURCE_FILES)):
        _artifact(record, f"{label}.source_files[{index}]", expected_path=expected)
    manifest = sha256_bytes(json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    _hex64(inputs.get("source_manifest_sha256"), f"{label}.source_manifest_sha256")
    if inputs["source_manifest_sha256"] != manifest:
        raise ValueError(f"{label}.source_manifest_sha256 does not match source_files")
    for name in ("native_executable", "model_qxf", "tokenizer_qxt"):
        _artifact(inputs.get(name), f"{label}.{name}")
    return inputs


def validate_report(report: dict[str, Any]) -> None:
    """Validate the closed report schema and internal consistency, without filesystem trust."""
    required = {"schema", "issue", "status", "claim_scope", "fixed_contract", "provenance", "reproduction", "case", "gates"}
    if not isinstance(report, dict) or set(report) != required: raise ValueError("report top-level schema mismatch")
    if report["schema"] != SCHEMA or type(report["issue"]) is not int or report["issue"] != 84 or report["status"] != "pass": raise ValueError("report identity/status mismatch")
    if report["claim_scope"] != CLAIM_SCOPE: raise ValueError("claim scope mismatch")
    if report["fixed_contract"] != FIXED_CONTRACT or any(type(report["fixed_contract"].get(k)) is not type(v) for k, v in FIXED_CONTRACT.items()):
        raise ValueError("fixed contract mismatch")
    reproduction = report["reproduction"]
    if reproduction != reproduction_metadata() or not isinstance(reproduction, dict): raise ValueError("reproduction metadata mismatch")
    case = report["case"]
    if not isinstance(case, dict) or set(case) != {"positions", "prompt_token_count", "generated_token_ids", "generated_text", "stop_reason", "timing", "wall_seconds", "sampled_peak_rss_bytes", "execution_profile", "capacity_profile", "artifacts"}:
        raise ValueError("case artifact schema mismatch")
    if not isinstance(case.get("artifacts"), dict) or set(case["artifacts"]) != {"stdout", "stderr"}:
        raise ValueError("case artifacts schema mismatch")
    for name in ("stdout", "stderr"):
        _artifact(case["artifacts"].get(name), f"case.artifacts.{name}")
    reconstructed = {"payload": {"prompt_token_count": case["prompt_token_count"],
        "generated_token_ids": case["generated_token_ids"], "generated_text": case["generated_text"],
        "stop_reason": case["stop_reason"], "timing": {"prefill_seconds": case["timing"]["prefill_seconds"],
        "decode_seconds": case["timing"]["decode_seconds"]}, "execution_profile": case["execution_profile"],
        "capacity_profile": case["capacity_profile"]}, "wall_seconds": case["wall_seconds"],
        "sampled_peak_rss_bytes": case["sampled_peak_rss_bytes"], "stdout": case["artifacts"]["stdout"],
        "stderr": case["artifacts"]["stderr"]}
    if validate_native_run(reconstructed) != case:
        raise ValueError("case is not the canonical validated 4096 result")
    provenance = report["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {"platform", "inputs_pre", "inputs_post"}:
        raise ValueError("provenance schema mismatch")
    platform_record = provenance["platform"]
    if not isinstance(platform_record, dict) or set(platform_record) != PLATFORM_KEYS or any(
        not isinstance(v, str) or not v or "/" in v or "\\" in v for v in platform_record.values()
    ):
        raise ValueError("provenance platform schema mismatch or absolute path")
    pre = _validate_inputs(provenance["inputs_pre"], "provenance.inputs_pre")
    post = _validate_inputs(provenance["inputs_post"], "provenance.inputs_post")
    if pre != post: raise ValueError("frozen inputs changed")
    if report["gates"] != PASS_GATES or any(type(report["gates"].get(k)) is not type(v) for k, v in PASS_GATES.items()):
        raise ValueError("gates mismatch")


def _verify_artifact(record: dict[str, Any], base: Path, label: str) -> None:
    relative = Path(record["path"])
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} artifact escapes its allowed directory") from exc
    if not candidate.is_file(): raise ValueError(f"{label} artifact is missing")
    if candidate.stat().st_size != record["size_bytes"]: raise ValueError(f"{label} artifact size mismatch")
    if sha256_file(candidate) != record["sha256"]: raise ValueError(f"{label} artifact SHA-256 mismatch")


def report_artifacts(report: dict[str, Any], root: Path, output_dir: Path) -> None:
    """Independently authenticate every report artifact against confined on-disk bytes."""
    root, output_dir = root.resolve(), output_dir.resolve()
    if not root.is_dir(): raise ValueError("root must be an existing directory")
    try:
        output_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("output_dir must be confined inside root") from exc
    if not output_dir.is_dir(): raise ValueError("output_dir must be an existing directory")
    provenance = report["provenance"]
    for side in ("inputs_pre", "inputs_post"):
        inputs = provenance[side]
        for index, record in enumerate(inputs["source_files"]):
            _verify_artifact(record, root, f"{side}.source_files[{index}]")
        for name in ("native_executable", "model_qxf", "tokenizer_qxt"):
            _verify_artifact(inputs[name], root, f"{side}.{name}")
    for name in ("stdout", "stderr"):
        _verify_artifact(report["case"]["artifacts"][name], output_dir, name)


def freeze_inputs(root: Path, exe: Path, model: Path, tokenizer: Path) -> dict[str, Any]:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    records = [artifact_record(root / name, root) for name in SOURCE_FILES]
    return {"git_revision": revision, "source_files": records, "source_manifest_sha256": sha256_bytes(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()),
            "native_executable": artifact_record(exe, root), "model_qxf": artifact_record(model, root), "tokenizer_qxt": artifact_record(tokenizer, root)}


def reproduction_metadata() -> dict[str, Any]:
    return {"working_directory": "repository_root", "prepare_command": ["python", "scripts/native_4k_acceptance.py", "--prepare-only", "--output-dir", "wiki/evidence/issue-84-cpu4k", "--qx-exe", "build/qxqxf.exe", "--model", "models/Qwen3-30B-A3B-UD-IQ2_M.qxf", "--tokenizer", "models/Qwen3-30B-A3B.qxt"],
            "run_command": ["python", "scripts/native_4k_acceptance.py", "--output-dir", "wiki/evidence/issue-84-cpu4k", "--qx-exe", "build/qxqxf.exe", "--model", "models/Qwen3-30B-A3B-UD-IQ2_M.qxf", "--tokenizer", "models/Qwen3-30B-A3B.qxt"]}


def test_only_report_fixture(case: dict[str, Any]) -> dict[str, Any]:
    """Build a validator fixture only; it is not evidence and is never written by CLI."""
    snap = {"test_only": True}
    return {"schema": SCHEMA, "issue": 84, "status": "pass", "claim_scope": "synthetic test-only fixture, not acceptance evidence",
            "fixed_contract": dict(FIXED_CONTRACT), "provenance": {"platform": {"python": platform.python_version()}, "inputs_pre": snap, "inputs_post": snap},
            "reproduction": reproduction_metadata(), "case": case,
            "gates": {"status": "pass", "exact_4096_counters": True, "two_outputs_and_checksums": True, "no_early_eos": True, "rss_within_ceiling": True, "inputs_unchanged": True}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue 84 bounded durable native CPU4K acceptance")
    parser.add_argument("--qx-exe", type=Path, default=Path("build/qxqxf.exe")); parser.add_argument("--model", type=Path, default=Path("models/Qwen3-30B-A3B-UD-IQ2_M.qxf")); parser.add_argument("--tokenizer", type=Path, default=Path("models/Qwen3-30B-A3B.qxt")); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    resolve = lambda p: p.resolve() if p.is_absolute() else (root / p).resolve()
    exe, model, tokenizer, output = map(resolve, (args.qx_exe, args.model, args.tokenizer, args.output_dir))
    started: float | None = None; phase = "prepare"; peak_rss_bytes = 0; report_published = False
    try:
        for path in (exe, model, tokenizer): artifact_record(path, root)
        portable_path(output, root); create_output_directory(output)
        required_space = model.stat().st_size + (1 << 30)
        free_space = shutil.disk_usage(output.parent).free
        if free_space < required_space: raise ValueError(f"insufficient free disk: need {required_space}, have {free_space}")
        started = time.monotonic(); _journal(output, "prepare", "running", started, 0, free_disk_bytes=free_space, required_disk_bytes=required_space)
        prompt_path = output / "prompt.txt"
        def encode(text: str) -> dict[str, Any]:
            prompt_path.write_text(text, encoding="utf-8", newline="")
            return tokenizer_encode(exe, tokenizer, prompt_path, root)
        prompt = construct_exact_prompt(PROMPT_TOKENS, encode)
        prompt_path.write_text(prompt["text"], encoding="utf-8", newline="")
        confirmed = tokenizer_encode(exe, tokenizer, prompt_path, root)
        if confirmed["token_ids"] != prompt["token_ids"]: raise ValueError("prompt tokenization changed")
        frozen_pre = freeze_inputs(root, exe, model, tokenizer)
        prep = {"schema": SCHEMA, "issue": 84, "status": "prepared", "computation_started": False,
                "prompt": {"token_count": 4095, "construction_attempts": prompt["attempts"], "artifact": artifact_record(prompt_path, output),
                           "token_ids_sha256": sha256_bytes(json.dumps(prompt["token_ids"], separators=(",", ":")).encode("ascii"))},
                "inputs": frozen_pre, "fixed_contract": FIXED_CONTRACT, "reproduction": reproduction_metadata()}
        atomic_json(output / "preparation.json", prep); _journal(output, "prepare", "prepared", started, 0)
        if args.prepare_only:
            print(json.dumps({"status": "prepared", "output_dir": portable_path(output, root)}, sort_keys=True)); return 0
        phase = "native"
        raw = run_supervised(generation_command(exe, model, tokenizer, prompt_path), output)
        peak_rss_bytes = _int(raw.get("sampled_peak_rss_bytes"), "sampled RSS", 1)
        phase = "acceptance"; _journal(output, phase, "running", started, peak_rss_bytes)
        case = validate_native_run(raw); frozen_post = freeze_inputs(root, exe, model, tokenizer)
        report = {"schema": SCHEMA, "issue": 84, "status": "pass", "claim_scope": "One native CPU run proving exactly 4096 executed positions only; no quality, throughput, thermal, soak, CUDA, or default-promotion claim.",
                  "fixed_contract": FIXED_CONTRACT, "provenance": {"platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "processor": platform.processor(), "python": platform.python_version(), "psutil": psutil.__version__}, "inputs_pre": frozen_pre, "inputs_post": frozen_post},
                  "reproduction": reproduction_metadata(), "case": case,
                  "gates": {"status": "pass", "exact_4096_counters": True, "two_outputs_and_checksums": True, "no_early_eos": True, "rss_within_ceiling": True, "inputs_unchanged": True}}
        validate_report(report); report_artifacts(report, root, output); atomic_json(output / "report.json", report); report_published = True
        report_hash = sha256_file(output / "report.json")
        _journal(output, phase, "completed", started, case["sampled_peak_rss_bytes"], report_sha256=report_hash)
        print(json.dumps({"status": "pass", "report": portable_path(output / "report.json", root), "sha256": report_hash}, sort_keys=True)); return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        if report_published:
            try: (output / "report.json").unlink(missing_ok=True)
            except OSError: pass
        preserve_native_terminal = False
        if started is not None and phase == "native":
            try:
                existing = json.loads((output / "phase-journal.json").read_text(encoding="utf-8"))
                preserve_native_terminal = (
                    isinstance(existing, dict)
                    and existing.get("schema") == SCHEMA
                    and existing.get("phase") == "native"
                    and existing.get("status") not in (None, "running")
                    and isinstance(existing.get("elapsed_seconds"), (int, float))
                    and isinstance(existing.get("peak_rss_bytes"), int)
                )
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
        if started is not None and not preserve_native_terminal:
            try: _journal(output, phase, "failed", started, peak_rss_bytes, error_type=type(exc).__name__, error=str(exc))
            except Exception as journal_exc: print(f"terminal failure journal could not be written: {journal_exc}", file=sys.stderr)
        print(f"CPU4K acceptance failed closed: {exc}", file=sys.stderr); return 2

if __name__ == "__main__":
    raise SystemExit(main())
