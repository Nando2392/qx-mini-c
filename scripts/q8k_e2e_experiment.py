#!/usr/bin/env python
"""Ejecuta la matriz F32/Q8_K x KV F32/INT8 y compara sidecars con llama.cpp."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
import time
from pathlib import Path


MODES = (("f32", "f32"), ("f32", "int8"), ("q8_k_compat", "f32"), ("q8_k_compat", "int8"))


def checkpoint_pairs(out: Path, oracle: Path) -> dict[str, tuple[Path, Path]]:
    return {
        "layer-0-input": (out / "step-0-layer-0-input.f32", oracle / "layer-0.f32"),
        "v-cur-0": (out / "step-0-layer-0-v-cur.f32", oracle / "Vcur-0.f32"),
        "kqv-out-0": (out / "step-0-layer-0-kqv-out.f32", oracle / "kqv_out-0.f32"),
        "ffn-inp-0": (out / "step-0-layer-0-ffn-inp.f32", oracle / "ffn_inp-0.f32"),
        "ffn-moe-out-0": (out / "step-0-layer-0-ffn-moe-out.f32", oracle / "ffn_moe_out-0.f32"),
        "l-out-0": (out / "step-0-layer-0-output.f32", oracle / "l_out-0.f32"),
        "layer-1-input": (out / "step-0-layer-1-input.f32", oracle / "layer-1.f32"),
        "layer-2-input": (out / "step-0-layer-2-input.f32", oracle / "layer-2.f32"),
        "layer-24-input": (out / "step-0-layer-24-input.f32", oracle / "layer-24.f32"),
        "layer-47-input": (out / "step-0-layer-47-input.f32", oracle / "layer-47.f32"),
        "l-out-47": (out / "step-0-layer-47-output.f32", oracle / "l_out-47.f32"),
        "logits": (out / "step-0-logits.f32", oracle / "logits.f32"),
    }


def load_f32(path: Path) -> list[float]:
    raw = path.read_bytes()
    if not raw or len(raw) % 4:
        raise ValueError(f"invalid F32 sidecar: {path}")
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def metrics(actual: list[float], expected: list[float]) -> dict:
    if len(actual) != len(expected) or not actual:
        raise ValueError("sidecar dimensions differ")
    deltas = [abs(a - b) for a, b in zip(actual, expected)]
    index = max(range(len(deltas)), key=deltas.__getitem__)
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(actual, expected)) / len(actual))
    denom = math.sqrt(sum(a * a for a in actual) * sum(b * b for b in expected))
    cosine = sum(a * b for a, b in zip(actual, expected)) / denom if denom else 1.0
    return {"count": len(actual), "max_abs": deltas[index], "rmse": rmse, "cosine": cosine, "max_abs_index": index}


def run(args: argparse.Namespace, activation: str, kv: str) -> dict:
    out = args.out / f"{activation}-{kv}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    command = [
        str(args.qx_exe), "state-loop-probe", "--in", str(args.model),
        "--prompt-token", "42", "--steps", "1", "--layers", "48",
        "--ctx", "4", "--kv", kv, "--activation", activation,
        "--temperature", "0", "--seed", "7", "--full-moe", "--final-head",
        "--top-n", "5", "--dump-residuals", str(out),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(f"QX failed ({completed.returncode}): {completed.stderr[-2000:]}")
    payload = json.loads(completed.stdout)
    head = payload["tokens"][0]["final_head"]
    comparisons = {}
    for checkpoint, (actual, expected) in checkpoint_pairs(out, args.oracle).items():
        comparisons[checkpoint] = metrics(load_f32(actual), load_f32(expected))
    comparisons["final-norm"] = metrics(head["final_norm_raw"], load_f32(args.oracle / "result_norm.f32"))
    return {
        "activation_format": payload["activation_format"],
        "kv_format": payload["kv_format"],
        "projection_kernel": payload["projection_kernel"],
        "activation_workspace_bytes": payload["activation_workspace_bytes"],
        "elapsed_seconds": elapsed,
        "argmax": head["argmax_token"],
        "argmax_logit": head["argmax_logit"],
        "top": head["top_tokens"],
        "comparisons": comparisons,
        "command": command,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qx-exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.qx_exe, args.model, args.oracle):
        if not path.exists():
            parser.error(f"missing fixture: {path}")
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"schema": 1, "experiment": "f32-vs-q8k-e2e", "runs": []}
    for activation, kv in MODES:
        result["runs"].append(run(args, activation, kv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
