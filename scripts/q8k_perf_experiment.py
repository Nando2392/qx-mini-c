#!/usr/bin/env python
"""Benchmark Q8_K compatible activation mode vs F32.

Measures:
- Latency (one token, 48 layers, token 42)
- Peak RSS memory usage
- Workspace allocation comparison

Usage:
    python -m pip install -r scripts/requirements-q8k-perf.txt
    python scripts/q8k_perf_experiment.py \
      --qx-exe build/qxqxf.exe \
      --model models/Qwen3-30B-A3B-UD-IQ2_M.qxf \
      --kv int8 \
      --repetitions 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import psutil


def one_run(command: list[str]) -> dict:
    """Run a benchmark and return timing + memory metrics."""
    started = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    tracked = psutil.Process(process.pid)
    peak_rss = 0
    while process.poll() is None:
        try:
            peak_rss = max(peak_rss, tracked.memory_info().rss)
        except psutil.Error:
            pass
        time.sleep(0.005)
    stderr = process.stderr.read() if process.stderr else ""
    elapsed = time.perf_counter() - started
    if process.returncode:
        raise RuntimeError(f"benchmark failed ({process.returncode}): {stderr[-2000:]}")
    return {"elapsed_seconds": elapsed, "peak_rss_bytes": peak_rss}


def summarize(values: list[float]) -> dict:
    """Return statistical summary of timing values."""
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    return {
        "count": len(values),
        "min": min(values),
        "median": median,
        "max": max(values),
        "mad": statistics.median(deviations),
        "pstdev": statistics.pstdev(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qx-exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kv", choices=("f32", "int8"), default="int8")
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()

    for path in (args.qx_exe, args.model):
        if not path.exists():
            parser.error(f"missing fixture: {path}")
    if args.repetitions < 3:
        parser.error("repetitions must be >= 3")

    runs = []
    for activation in ("f32", "q8_k_compat"):
        command = [
            str(args.qx_exe), "state-loop-probe", "--in", str(args.model),
            "--prompt-token", "42", "--steps", "1", "--layers", "48", "--ctx", "4",
            "--kv", args.kv, "--activation", activation, "--temperature", "0",
            "--seed", "7", "--full-moe",
        ]
        cold = one_run(command)
        warm = [one_run(command) for _ in range(args.repetitions)]
        runs.append({
            "activation_format": activation,
            "kv_format": args.kv,
            "command": command,
            "cold": cold,
            "warm": warm,
            "elapsed_seconds": summarize([item["elapsed_seconds"] for item in warm]),
            "peak_rss_bytes": summarize([float(item["peak_rss_bytes"]) for item in warm]),
            "activation_workspace_bytes": 0 if activation == "f32" else 16 * 292,
        })

    print(json.dumps({
        "schema": 1,
        "measurement": "one-token-48-layer-latency",
        "runs": runs,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())