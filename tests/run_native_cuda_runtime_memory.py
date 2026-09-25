from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HARD_TIMEOUT_SECONDS = 180
GPU_BUDGET_BYTES = 8 * 1024 * 1024
HOST_BUDGET_BYTES = 32 * 1024 * 1024


class GateError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_absolute_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise GateError(f"{label} path must be absolute")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise GateError(f"{label} is not a file")
    return resolved


def _native_non_negative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0 or value > 0xFFFFFFFFFFFFFFFF:
        raise GateError(f"{label} must be a native non-negative uint64")
    return value


def _native_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise GateError(f"{label} must be a native boolean")
    return value


def _snapshot(driver: Path, model: Path) -> dict[str, dict[str, int | str]]:
    artifacts = {
        "driver_executable": driver,
        "model": model,
        "include/qx_format.h": ROOT / "include" / "qx_format.h",
        "src/qx_format.c": ROOT / "src" / "qx_format.c",
        "src/qx_cuda_final_head.cu": ROOT / "src" / "qx_cuda_final_head.cu",
        "tests/native_cuda_runtime_memory_driver.cu": ROOT / "tests" / "native_cuda_runtime_memory_driver.cu",
    }
    snapshot: dict[str, dict[str, int | str]] = {}
    for name, path in artifacts.items():
        if not path.is_file():
            raise GateError(f"frozen artifact is missing: {name}")
        snapshot[name] = {"bytes": path.stat().st_size, "raw_sha256": sha256_file(path)}
    return snapshot


def _json_load_strict(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise GateError(f"driver JSON contains non-finite constant: {value}")

    try:
        return json.loads(text, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise GateError(f"driver stdout is not one valid JSON document: {exc}") from exc


def validate_driver_report(payload: object, *, require_pass: bool = True) -> dict[str, Any]:
    if type(payload) is not dict:
        raise GateError("driver report must be an object")
    report = payload
    if report.get("schema") != "qx.native-cuda-runtime-memory.v1":
        raise GateError("driver report schema mismatch")
    if report.get("same_process") is not True or report.get("process_restart_substitution") is not False:
        raise GateError("report must prove same-process calls without process restart substitution")
    for field, expected in (("warmup_calls", 1), ("measured_calls", 3), ("outputs_per_call", 2)):
        if _native_non_negative_int(report.get(field), field) != expected:
            raise GateError(f"{field} must equal {expected}")

    budgets = report.get("budgets")
    if type(budgets) is not dict:
        raise GateError("budgets must be an object")
    if _native_non_negative_int(budgets.get("gpu_post_warmup_bytes"), "gpu budget") != GPU_BUDGET_BYTES:
        raise GateError("GPU post-warmup budget must be exactly 8 MiB")
    if _native_non_negative_int(budgets.get("host_post_warmup_bytes"), "host budget") != HOST_BUDGET_BYTES:
        raise GateError("host post-warmup budget must be exactly 32 MiB")
    reason = budgets.get("host_budget_reason")
    if type(reason) is not str or "255 MiB" not in reason:
        raise GateError("host budget reason must explicitly reject 255 MiB retention")

    samples = report.get("samples")
    if type(samples) is not list or len(samples) != 8:
        raise GateError("samples must contain before/after points for one warmup and three measured calls")
    expected_phases = ["warmup_before", "warmup_after"]
    for call_index in range(1, 4):
        expected_phases.extend(("measured_before", "measured_after"))
    for ordinal, (sample, expected_phase) in enumerate(zip(samples, expected_phases)):
        if type(sample) is not dict:
            raise GateError(f"sample {ordinal} must be an object")
        if sample.get("phase") != expected_phase:
            raise GateError(f"sample {ordinal} phase mismatch")
        expected_call = 0 if ordinal < 2 else (ordinal - 2) // 2 + 1
        if _native_non_negative_int(sample.get("call_index"), f"sample {ordinal} call_index") != expected_call:
            raise GateError(f"sample {ordinal} call_index mismatch")
        for field in (
            "cuda_free_bytes", "cuda_total_bytes", "process_private_bytes",
            "process_working_set_bytes",
        ):
            _native_non_negative_int(sample.get(field), f"sample {ordinal} {field}")
        if sample["cuda_free_bytes"] > sample["cuda_total_bytes"]:
            raise GateError(f"sample {ordinal} reports CUDA free bytes above total bytes")

    calls = report.get("calls")
    if type(calls) is not list or len(calls) != 4:
        raise GateError("calls must contain one warmup and three measured calls")
    stable_ids: list[int] | None = None
    for index, call in enumerate(calls):
        if type(call) is not dict:
            raise GateError(f"call {index} must be an object")
        expected_phase = "warmup" if index == 0 else "measured"
        if call.get("phase") != expected_phase:
            raise GateError(f"call {index} phase mismatch")
        if _native_non_negative_int(call.get("call_index"), f"call {index} call_index") != index:
            raise GateError(f"call {index} call_index mismatch")
        token_ids = call.get("token_ids")
        if type(token_ids) is not list or len(token_ids) != 2:
            raise GateError(f"call {index} must contain exactly two token IDs")
        for token_index, token_id in enumerate(token_ids):
            _native_non_negative_int(token_id, f"call {index} token {token_index}")
        if stable_ids is None:
            stable_ids = token_ids
        elif token_ids != stable_ids:
            raise GateError("token IDs are not stable across all same-process calls")
        expected_counters = {"weight_uploads": 1, "kernel_launches": 2, "cpu_fallbacks": 0}
        for field, expected in expected_counters.items():
            if _native_non_negative_int(call.get(field), f"call {index} {field}") != expected:
                raise GateError(f"call {index} {field} must equal {expected}")

    measurements = report.get("measurements")
    if type(measurements) is not dict:
        raise GateError("measurements must be an object")
    gpu_fields = ("gpu_retained_growth_bytes", "gpu_post_warmup_free_span_bytes")
    host_fields = (
        "process_private_retained_growth_bytes", "process_private_post_warmup_span_bytes",
        "process_working_set_retained_growth_bytes", "process_working_set_post_warmup_span_bytes",
    )
    gpu_values = [_native_non_negative_int(measurements.get(field), field) for field in gpu_fields]
    host_values = [_native_non_negative_int(measurements.get(field), field) for field in host_fields]

    gates = report.get("gates")
    if type(gates) is not dict:
        raise GateError("gates must be an object")
    for field in (
        "cuda_total_stable", "gpu_within_budget", "host_within_budget",
        "stable_token_ids", "exact_per_call_counters", "leak_claim",
    ):
        _native_bool(gates.get(field), f"gate {field}")
    if gates["leak_claim"] is not False:
        raise GateError("cudaMemGetInfo evidence must not claim a leak")
    interpretation = report.get("cuda_mem_get_info_interpretation")
    if type(interpretation) is not str or "shared-device" not in interpretation or "not a leak claim" not in interpretation:
        raise GateError("shared-device CUDA noise interpretation is missing")

    computed_gpu_ok = gates["cuda_total_stable"] and max(gpu_values) <= GPU_BUDGET_BYTES
    computed_host_ok = max(host_values) <= HOST_BUDGET_BYTES
    if gates["gpu_within_budget"] is not computed_gpu_ok:
        raise GateError("GPU gate disagrees with measured bytes")
    if gates["host_within_budget"] is not computed_host_ok:
        raise GateError("host gate disagrees with measured bytes")
    if gates["stable_token_ids"] is not True or gates["exact_per_call_counters"] is not True:
        raise GateError("determinism/counter gates must pass")

    expected_status = (
        "inconclusive_shared_device_noise" if not computed_gpu_ok
        else "fail_host_retained_growth" if not computed_host_ok
        else "pass"
    )
    if report.get("status") != expected_status:
        raise GateError("status disagrees with measured memory gates")
    if require_pass and expected_status != "pass":
        raise GateError(f"runtime memory gate did not pass: {expected_status}")
    return report


def _kill_owned_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True, text=True, check=False, timeout=10,
        )
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired as exc:
        raise GateError(f"owned process {process.pid} survived forced timeout cleanup") from exc


def run_gate(driver: Path, model: Path, output_dir: Path) -> dict[str, Any]:
    driver = _require_absolute_file(driver, "driver")
    model = _require_absolute_file(model, "model")
    if not output_dir.is_absolute():
        raise GateError("output path must be absolute")
    if output_dir.exists():
        raise GateError("output directory must not already exist")
    output_dir.mkdir(parents=True)

    before = _snapshot(driver, model)
    (output_dir / "hashes-before.json").write_text(
        json.dumps(before, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    env = os.environ.copy()
    started = time.perf_counter()
    process = subprocess.Popen(
        [str(driver), str(model)], cwd=ROOT, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="strict",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=HARD_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_owned_process_tree(process)
        stdout, stderr = process.communicate()
    elapsed = time.perf_counter() - started

    (output_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (output_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    (output_dir / "exitcode").write_text(f"{process.returncode}\n", encoding="ascii")
    after = _snapshot(driver, model)
    (output_dir / "hashes-after.json").write_text(
        json.dumps(after, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if before != after:
        raise GateError("raw before/after artifact hash manifests differ")
    if timed_out:
        raise GateError(f"driver exceeded the total {HARD_TIMEOUT_SECONDS}-second hard timeout; owned process was killed")
    if not stdout.strip():
        raise GateError("driver emitted no JSON report")

    payload = validate_driver_report(_json_load_strict(stdout), require_pass=False)
    (output_dir / "driver-report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    accepted = process.returncode == 0 and payload["status"] == "pass" and stderr == ""
    summary = {
        "schema": "qx.native-cuda-runtime-memory-run.v1",
        "status": "pass" if accepted else payload["status"],
        "hard_timeout_seconds": HARD_TIMEOUT_SECONDS,
        "elapsed_seconds": elapsed,
        "driver_returncode": process.returncode,
        "stderr_empty": stderr == "",
        "raw_hashes_unchanged": True,
        "driver_report": payload,
    }
    if not math.isfinite(elapsed):
        raise GateError("elapsed time is not finite")
    (output_dir / "run-report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not accepted:
        raise GateError(f"driver did not satisfy the gate: returncode={process.returncode}, status={payload['status']}")
    validate_driver_report(payload, require_pass=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the same-process native CUDA runtime memory gate")
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_gate(args.driver, args.model, args.out)
    except (GateError, OSError, UnicodeError) as exc:
        print(f"native CUDA runtime memory gate: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
