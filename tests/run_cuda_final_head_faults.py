from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VCVARS = Path(
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)
BACKEND = ROOT / "src" / "qx_cuda_final_head.cu"
DRIVER_SOURCE = ROOT / "tests" / "cuda_final_head_fault_driver.cu"
EXPECTED_BACKEND_SHA256 = "a7a43140358be5cc4950f5f0d670e303fb2ca422a383584e71e743eb2dfe5e59"
BUILD_TIMEOUT_SECONDS = 120
RUN_TIMEOUT_SECONDS = 60


class GateError(RuntimeError):
    pass


def resolve_cuda_root(explicit: Path | None) -> Path:
    configured = explicit
    if configured is None:
        value = os.environ.get("QX_CUDA_ROOT") or os.environ.get("CUDA_PATH")
        if not value:
            raise GateError(
                "CUDA root is required; pass --cuda-root or set QX_CUDA_ROOT or CUDA_PATH"
            )
        configured = Path(value)
    required = (
        configured / "bin" / "nvcc.exe",
        configured / "lib" / "x64" / "cudart.lib",
        configured / "include" / "cuda_runtime.h",
    )
    for path in required:
        if not path.is_file():
            raise GateError(f"invalid CUDA root {configured}: missing {path}")
    return configured


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def snapshot() -> dict[str, dict[str, int | str]]:
    paths = {
        "src/qx_cuda_final_head.cu": BACKEND,
        "include/qx_cuda_final_head.h": ROOT / "include" / "qx_cuda_final_head.h",
        "tests/cuda_final_head_fault_driver.cu": DRIVER_SOURCE,
        "tests/run_cuda_final_head_faults.py": Path(__file__).resolve(),
    }
    result: dict[str, dict[str, int | str]] = {}
    for label, path in paths.items():
        if not path.is_file():
            raise GateError(f"tracked source is missing: {label}")
        result[label] = {"bytes": path.stat().st_size, "raw_sha256": sha256_file(path)}
    return result


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def kill_owned_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    subprocess.run(
        ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if process.poll() is None:
        process.kill()
    process.wait(timeout=10)


def run_bounded(
    args: list[str], *, env: dict[str, str], timeout: int
) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.perf_counter()
    process = subprocess.Popen(
        args,
        cwd=ROOT,
        env=env,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        kill_owned_process_tree(process)
        stdout, stderr = process.communicate()
        raise GateError(f"command exceeded hard timeout of {timeout}s: {args[0]}") from exc
    elapsed = time.perf_counter() - started
    return subprocess.CompletedProcess(args, process.returncode, stdout, stderr), elapsed


def strict_json(text: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise GateError(f"driver JSON contains non-finite constant: {value}")

    try:
        payload = json.loads(text, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise GateError(f"driver stdout is not one JSON document: {exc}") from exc
    if type(payload) is not dict:
        raise GateError("driver report must be an object")
    expected = {
        "schema": "qx.cuda-final-head-faults.v1",
        "status": "pass",
        "tests_passed": 10,
        "whole_backend_mocked": False,
        "production_source_included": True,
        "exact_alloc_free_tracking": True,
        "silent_fallbacks": 0,
    }
    for field, value in expected.items():
        if payload.get(field) != value or type(payload.get(field)) is not type(value):
            raise GateError(f"driver report field {field!r} does not match the fail-closed contract")
    device = payload.get("real_device")
    if type(device) is not str or not device:
        raise GateError("driver did not identify the real CUDA device")
    launch = payload.get("launch_injection")
    if type(launch) is not str or "not an actually failed device kernel" not in launch:
        raise GateError("driver did not distinguish launch API injection from a failed device kernel")
    return payload


def run_gate(output_dir: Path, cuda_root: Path | None = None) -> dict[str, Any]:
    cuda_root = resolve_cuda_root(cuda_root)
    if not output_dir.is_absolute():
        raise GateError("--out must be an absolute path")
    if output_dir.exists():
        raise GateError("--out must not already exist")
    nvcc = cuda_root / "bin" / "nvcc.exe"
    if not VCVARS.is_file():
        raise GateError(f"MSVC environment script is missing: {VCVARS}")
    if sha256_file(BACKEND) != EXPECTED_BACKEND_SHA256:
        raise GateError("production backend hash differs from the frozen Issue 85 source hash")

    output_dir.mkdir(parents=True)
    before = snapshot()
    write_json(output_dir / "source-hashes-before.json", before)

    executable = output_dir / "cuda_final_head_fault_driver.exe"
    command_file = output_dir / "build_fault_driver.cmd"
    command_lines = [
        "@echo off",
        "setlocal",
        f'call "{VCVARS}" >nul',
        "if errorlevel 1 exit /b %errorlevel%",
        f'"{nvcc}" -arch=sm_89 -cudart shared -Iinclude '
        f'"{DRIVER_SOURCE}" -o "{executable}"',
        "exit /b %errorlevel%",
    ]
    command_file.write_text("\n".join(command_lines) + "\n", encoding="utf-8", newline="\r\n")
    raw_command = command_file.read_bytes()
    if b"\r\n" not in raw_command or b"\n" in raw_command.replace(b"\r\n", b""):
        raise GateError("generated MSVC/NVCC command file is not CRLF-only")

    env = os.environ.copy()
    cuda_bin = str(cuda_root / "bin")
    env["PATH"] = cuda_bin + os.pathsep + env.get("PATH", "")
    build_args = ["cmd.exe", "/d", "/s", "/c", str(command_file)]
    build, build_elapsed = run_bounded(build_args, env=env, timeout=BUILD_TIMEOUT_SECONDS)
    (output_dir / "build.stdout.log").write_text(build.stdout, encoding="utf-8", newline="\n")
    (output_dir / "build.stderr.log").write_text(build.stderr, encoding="utf-8", newline="\n")
    if build.returncode != 0 or not executable.is_file():
        raise GateError(f"isolated NVCC build failed with return code {build.returncode}")

    run_args = [str(executable)]
    run, run_elapsed = run_bounded(run_args, env=env, timeout=RUN_TIMEOUT_SECONDS)
    (output_dir / "driver.stdout.log").write_text(run.stdout, encoding="utf-8", newline="\n")
    (output_dir / "driver.stderr.log").write_text(run.stderr, encoding="utf-8", newline="\n")
    (output_dir / "driver.exitcode").write_text(f"{run.returncode}\n", encoding="ascii", newline="\n")

    after = snapshot()
    write_json(output_dir / "source-hashes-after.json", after)
    if before != after:
        raise GateError("tracked source hashes changed during build or execution")
    if after["src/qx_cuda_final_head.cu"]["raw_sha256"] != EXPECTED_BACKEND_SHA256:
        raise GateError("production backend hash changed during the fault gate")
    if run.returncode != 0:
        detail = run.stderr.strip() or run.stdout.strip()
        raise GateError(f"real-GPU fault driver failed with return code {run.returncode}: {detail}")
    if run.stderr != "":
        raise GateError("passing fault driver emitted stderr")
    driver_report = strict_json(run.stdout)

    manifest = {
        "schema": "qx.cuda-final-head-fault-run.v1",
        "status": "pass",
        "build": {
            "args": build_args,
            "shell": False,
            "cmd_crlf_only": True,
            "timeout_seconds": BUILD_TIMEOUT_SECONDS,
            "elapsed_seconds": build_elapsed,
            "returncode": build.returncode,
            "cuda_bin_path_prefixed": cuda_bin,
            "executable": str(executable),
            "executable_sha256": sha256_file(executable),
        },
        "run": {
            "args": run_args,
            "shell": False,
            "timeout_seconds": RUN_TIMEOUT_SECONDS,
            "elapsed_seconds": run_elapsed,
            "returncode": run.returncode,
            "stderr_empty": run.stderr == "",
        },
        "source_hashes_unchanged": True,
        "frozen_backend_sha256": EXPECTED_BACKEND_SHA256,
        "historical_manifests_modified": False,
        "driver_report": driver_report,
    }
    write_json(output_dir / "fault-run-manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and run the Issue 85 real-CUDA fault-injection gate")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--cuda-root",
        type=Path,
        help="CUDA toolkit root (preferred over QX_CUDA_ROOT, then CUDA_PATH)",
    )
    args = parser.parse_args(argv)
    try:
        manifest = run_gate(args.out, args.cuda_root)
    except (GateError, OSError, UnicodeError, subprocess.SubprocessError) as exc:
        print(f"CUDA final-head fault gate: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
