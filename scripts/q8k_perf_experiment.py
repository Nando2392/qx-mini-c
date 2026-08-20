#!/usr/bin/env python
"""Fail-closed CPU inference A/B baseline for F32 vs Q8_K compatible.

Measures:
- startup/model-load wall latency
- native prefill and decode latency/tokens per second
- end-to-end wall latency and peak RSS
- deterministic per-mode output signatures

Usage:
    python -m pip install -r scripts/requirements-q8k-perf.txt
    python scripts/q8k_perf_experiment.py \
      --qx-exe build/qxqxf.exe \
      --source-model models/Qwen3-30B-A3B-UD-IQ2_M.gguf \
      --model models/Qwen3-30B-A3B-UD-IQ2_M.qxf \
      --tokenizer models/Qwen3-30B-A3B.qxt \
      --prompt-file tests/fixtures/q8k_perf_prompt.txt \
      --output wiki/evidence/issue-23-cpu-baseline.json \
      --kv int8 \
      --repetitions 5
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

import psutil


def one_run(command: list[str]) -> dict[str, Any]:
    """Run one command, capture JSON without pipe deadlock, and sample RSS."""
    started = time.perf_counter()
    with tempfile.TemporaryFile() as stdout_file:
        process = subprocess.Popen(command, stdout=stdout_file, stderr=subprocess.PIPE, text=True)
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
        stdout_file.seek(0)
        raw = stdout_file.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("benchmark command did not emit one valid UTF-8 JSON value") from exc
    if not isinstance(payload, dict):
        raise ValueError("benchmark command JSON must be an object")
    return {"wall_elapsed_seconds": elapsed, "peak_rss_bytes": peak_rss, "payload": payload}


def summarize(values: list[float]) -> dict:
    """Return statistical summary of timing values."""
    if not values or any(isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("summary values must be finite positive numbers")
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


def _require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_exact_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _require_positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def extract_output_signature(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract deterministic output evidence from a native runtime payload."""
    prompt_ids = payload.get("prompt_token_ids")
    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise ValueError("prompt_token_ids must be a non-empty array")
    prompt_ids = [_require_exact_int(token, "prompt_token_ids item") for token in prompt_ids]
    tokens = payload.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("tokens must be a non-empty array")
    selected: list[int] = []
    residual_checksums: list[int] = []
    norm_checksums: list[int] = []
    logits_checksums: list[int] = []
    for index, raw_token in enumerate(tokens):
        token = _require_object(raw_token, f"tokens[{index}]")
        selected_token = token.get("selected_token")
        if selected_token is None:
            continue
        selected.append(_require_exact_int(selected_token, f"tokens[{index}].selected_token"))
        head = _require_object(token.get("final_head"), f"tokens[{index}].final_head")
        residual_checksums.append(_require_exact_int(head.get("final_residual_checksum"), "final_residual_checksum"))
        norm_checksums.append(_require_exact_int(head.get("final_norm_checksum"), "final_norm_checksum"))
        logits_checksums.append(_require_exact_int(head.get("logits_checksum"), "logits_checksum"))
        if _require_exact_int(head.get("argmax_token"), "argmax_token") != selected[-1]:
            raise ValueError("argmax_token does not match selected_token")
    if not selected:
        raise ValueError("runtime payload has no selected tokens")
    if payload.get("cache_readback_ok") is not True:
        raise ValueError("cache_readback_ok must be true")
    final_token = _require_exact_int(payload.get("final_token"), "final_token")
    if final_token != selected[-1]:
        raise ValueError("final_token does not match the final selected token")
    return {
        "prompt_token_ids": prompt_ids,
        "selected_tokens": selected,
        "final_residual_checksums": residual_checksums,
        "final_norm_checksums": norm_checksums,
        "logits_checksums": logits_checksums,
        "final_token": final_token,
        "k_cache_checksum": _require_exact_int(payload.get("k_cache_checksum"), "k_cache_checksum"),
        "v_cache_checksum": _require_exact_int(payload.get("v_cache_checksum"), "v_cache_checksum"),
        "cache_readback_ok": True,
    }


def validate_native_payload(
    payload: dict[str, Any], *, activation: str, kv: str, layers: int,
    ctx: int, generation_steps: int,
) -> dict[str, Any]:
    """Validate native timing, modality and output evidence fail-closed."""
    expected = {
        "probe": "state_loop", "activation_format": activation,
        "kv_format": kv, "layers": layers, "ctx_tokens": ctx,
        "generation_steps": generation_steps,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{key} does not match the fixed benchmark contract")
    prompt_ids = payload.get("prompt_token_ids")
    if not isinstance(prompt_ids, list) or len(prompt_ids) < 2:
        raise ValueError("prompt_token_ids must contain at least two fixed prompt tokens")
    prompt_count = _require_exact_int(payload.get("prompt_token_count"), "prompt_token_count")
    if prompt_count != len(prompt_ids):
        raise ValueError("prompt_token_count does not match prompt_token_ids")
    expected_prefill_tokens = prompt_count - 1
    raw_tokens = payload.get("tokens")
    if not isinstance(raw_tokens, list) or len(raw_tokens) != expected_prefill_tokens + generation_steps:
        raise ValueError("runtime token evidence count does not match prompt and generation contract")
    observed_phases = [
        _require_object(raw_token, f"tokens[{index}]").get("phase")
        for index, raw_token in enumerate(raw_tokens)
    ]
    expected_phases = ["prefill"] * expected_prefill_tokens + ["generate"] * generation_steps
    if observed_phases != expected_phases:
        raise ValueError("runtime token phases do not match prefill/decode contract")
    bench = _require_object(payload.get("bench"), "bench")
    if bench.get("enabled") is not True:
        raise ValueError("native bench must be enabled")
    _require_positive_number(bench.get("elapsed_sec"), "bench.elapsed_sec")
    phases = _require_object(bench.get("phases"), "bench.phases")
    if set(phases) != {"prefill", "decode"}:
        raise ValueError("bench.phases must contain exactly prefill and decode")
    for phase_name in ("prefill", "decode"):
        phase = _require_object(phases[phase_name], f"bench.phases.{phase_name}")
        if _require_exact_int(phase.get("tokens"), f"{phase_name}.tokens") <= 0:
            raise ValueError(f"{phase_name}.tokens must be positive")
        _require_positive_number(phase.get("elapsed_sec"), f"{phase_name}.elapsed_sec")
        _require_positive_number(phase.get("tokens_per_second"), f"{phase_name}.tokens_per_second")
    if _require_exact_int(phases["prefill"].get("tokens"), "prefill.tokens") != expected_prefill_tokens:
        raise ValueError("prefill.tokens does not match fixed prompt length")
    if _require_exact_int(phases["decode"].get("tokens"), "decode.tokens") != generation_steps:
        raise ValueError("decode token count does not match generation_steps")
    observed_selected_count = sum(
        _require_object(raw_token, f"tokens[{index}]").get("selected_token") is not None
        for index, raw_token in enumerate(raw_tokens)
    )
    if observed_selected_count != generation_steps:
        raise ValueError("selected-token count does not match generation_steps")
    signature = extract_output_signature(payload)
    if len(signature["selected_tokens"]) != generation_steps:
        raise ValueError("selected-token count does not match generation_steps")
    return signature


def validate_output_contract(signatures: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Require per-cell determinism and document bounded cross-mode drift."""
    if set(signatures) != {"f32", "q8_k_compat"}:
        raise ValueError("output evidence must contain exactly f32 and q8_k_compat")
    canonical: dict[str, dict[str, Any]] = {}
    for activation, items in signatures.items():
        if not items:
            raise ValueError(f"{activation} has no measured output signatures")
        if any(item != items[0] for item in items[1:]):
            raise ValueError(f"{activation} output is not deterministic across repetitions")
        canonical[activation] = items[0]
    f32 = canonical["f32"]
    q8k = canonical["q8_k_compat"]
    if f32["prompt_token_ids"] != q8k["prompt_token_ids"]:
        raise ValueError("prompt token IDs differ across activation modes")
    selected_equal = f32["selected_tokens"] == q8k["selected_tokens"]
    checksum_fields = (
        "final_residual_checksums", "final_norm_checksums", "logits_checksums",
        "k_cache_checksum", "v_cache_checksum",
    )
    checksums_equal = all(f32[field] == q8k[field] for field in checksum_fields)
    return {
        "status": "pass",
        "prompt_token_ids": f32["prompt_token_ids"],
        "selected_tokens": f32["selected_tokens"] if selected_equal else None,
        "selected_tokens_equal_across_modes": selected_equal,
        "selected_tokens_by_mode": {
            "f32": f32["selected_tokens"],
            "q8_k_compat": q8k["selected_tokens"],
        },
        "numeric_checksums_equal_across_modes": checksums_equal,
        "allowed_cross_mode_differences": [
            "selected-token sequence because q8_k_compat is an opt-in numerical mode without global greedy parity",
            "final residual, final norm, logits, and KV cache checksums",
        ],
    }


def build_inference_command(
    *, qx_exe: Path, model: Path, tokenizer: Path, prompt_file: Path,
    activation: str, kv: str, layers: int, ctx: int,
    generation_steps: int, seed: int,
) -> list[str]:
    """Build the fixed real-prompt inference command for one A/B cell."""
    return [
        str(qx_exe), "prompt-state-loop-probe", "--in", str(model),
        "--tokenizer", str(tokenizer), "--text-file", str(prompt_file),
        "--generate", str(generation_steps), "--layers", str(layers),
        "--ctx", str(ctx), "--kv", kv, "--activation", activation,
        "--temperature", "0", "--seed", str(seed), "--full-moe",
        "--final-head", "--bench",
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_provenance(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256_file(path)}


def build_artifact_provenance(
    *, benchmark_script: Path, qx_exe: Path, source_model_gguf: Path, model_qxf: Path,
    tokenizer_qxt: Path, prompt: Path,
) -> dict[str, dict[str, Any]]:
    return {
        "benchmark_script": artifact_provenance(benchmark_script),
        "qx_exe": artifact_provenance(qx_exe),
        "source_model_gguf": artifact_provenance(source_model_gguf),
        "model_qxf": artifact_provenance(model_qxf),
        "tokenizer_qxt": artifact_provenance(tokenizer_qxt),
        "prompt": artifact_provenance(prompt),
    }


def source_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
        text=True, capture_output=True, check=True,
    )
    revision = result.stdout.strip()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("git revision is not a lowercase 40-character SHA-1")
    return revision


def compact_run(
    raw: dict[str, Any], *, activation: str, kv: str, layers: int,
    ctx: int, generation_steps: int,
) -> dict[str, Any]:
    payload = _require_object(raw.get("payload"), "runtime payload")
    signature = validate_native_payload(
        payload, activation=activation, kv=kv, layers=layers, ctx=ctx,
        generation_steps=generation_steps,
    )
    bench = _require_object(payload["bench"], "bench")
    phases = _require_object(bench["phases"], "bench.phases")
    prefill = _require_object(phases["prefill"], "bench.phases.prefill")
    decode = _require_object(phases["decode"], "bench.phases.decode")
    peak_rss = _require_exact_int(raw.get("peak_rss_bytes"), "peak_rss_bytes")
    if peak_rss <= 0:
        raise ValueError("peak_rss_bytes must be positive")
    return {
        "total_latency_seconds": _require_positive_number(raw.get("wall_elapsed_seconds"), "wall_elapsed_seconds"),
        "native_process_seconds": _require_positive_number(bench.get("elapsed_sec"), "bench.elapsed_sec"),
        "prefill_latency_seconds": _require_positive_number(prefill.get("elapsed_sec"), "prefill.elapsed_sec"),
        "prefill_tokens_per_second": _require_positive_number(prefill.get("tokens_per_second"), "prefill.tokens_per_second"),
        "decode_latency_seconds": _require_positive_number(decode.get("elapsed_sec"), "decode.elapsed_sec"),
        "decode_tokens_per_second": _require_positive_number(decode.get("tokens_per_second"), "decode.tokens_per_second"),
        "peak_rss_bytes": peak_rss,
        "output_signature": signature,
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "total_latency_seconds", "native_process_seconds",
        "prefill_latency_seconds", "prefill_tokens_per_second",
        "decode_latency_seconds", "decode_tokens_per_second", "peak_rss_bytes",
    )
    return {field: summarize([float(run[field]) for run in runs]) for field in fields}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed CPU F32/Q8_K inference A/B baseline")
    parser.add_argument("--qx-exe", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kv", choices=("f32", "int8"), default="int8")
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--ctx", type=int, default=16)
    parser.add_argument("--generate", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in (args.qx_exe, args.source_model, args.model, args.tokenizer, args.prompt_file):
        if not path.is_file():
            parser.error(f"missing fixture: {path}")
    if not args.prompt_file.read_bytes():
        parser.error("prompt file must not be empty")
    if args.repetitions < 3:
        parser.error("repetitions must be >= 3")
    if args.warmups < 1:
        parser.error("warmups must be >= 1")
    if args.layers != 48:
        parser.error("this final-head baseline requires exactly 48 layers")
    if args.ctx < 2 or args.generate < 1 or args.seed < 0:
        parser.error("ctx must be >= 2, generate >= 1, and seed >= 0")
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists (use --overwrite): {args.output}")

    try:
        startup_command = [str(args.qx_exe), "inspect", "--in", str(args.model)]
        startup_warmups = [one_run(startup_command) for _ in range(args.warmups)]
        startup_measured = [one_run(startup_command) for _ in range(args.repetitions)]
        for run in startup_warmups + startup_measured:
            if _require_object(run["payload"], "inspect payload").get("magic") != "QXF1":
                raise ValueError("startup/model-load probe did not inspect a QXF1 model")

        cells: list[dict[str, Any]] = []
        signatures: dict[str, list[dict[str, Any]]] = {}
        for activation in ("f32", "q8_k_compat"):
            command = build_inference_command(
                qx_exe=args.qx_exe, model=args.model, tokenizer=args.tokenizer,
                prompt_file=args.prompt_file, activation=activation, kv=args.kv,
                layers=args.layers, ctx=args.ctx, generation_steps=args.generate,
                seed=args.seed,
            )
            warmups = [
                compact_run(one_run(command), activation=activation, kv=args.kv,
                            layers=args.layers, ctx=args.ctx, generation_steps=args.generate)
                for _ in range(args.warmups)
            ]
            measured = [
                compact_run(one_run(command), activation=activation, kv=args.kv,
                            layers=args.layers, ctx=args.ctx, generation_steps=args.generate)
                for _ in range(args.repetitions)
            ]
            signatures[activation] = [run["output_signature"] for run in warmups + measured]
            cells.append({
                "activation_format": activation,
                "command": command,
                "warmups": warmups,
                "measured": measured,
                "summary": summarize_runs(measured),
            })

        output_gate = validate_output_contract(signatures)
        by_mode = {cell["activation_format"]: cell for cell in cells}
        f32_summary = by_mode["f32"]["summary"]
        q8k_summary = by_mode["q8_k_compat"]["summary"]
        comparisons = {}
        for field in ("total_latency_seconds", "prefill_latency_seconds", "decode_latency_seconds"):
            comparisons[f"{field}_speedup_f32_over_q8_k"] = (
                f32_summary[field]["median"] / q8k_summary[field]["median"]
            )
        comparisons["peak_rss_median_delta_bytes_q8_k_minus_f32"] = (
            q8k_summary["peak_rss_bytes"]["median"] - f32_summary["peak_rss_bytes"]["median"]
        )

        report = {
            "schema": 2,
            "measurement": "cpu-inference-ab-baseline",
            "status": "pass",
            "claim_scope": "Pinned CPU, model, QXF, tokenizer, prompt, revision and arguments only; not global throughput or model/logit parity.",
            "provenance": {
                "revision": source_revision(),
                "artifacts": build_artifact_provenance(
                    benchmark_script=Path(__file__), qx_exe=args.qx_exe,
                    source_model_gguf=args.source_model,
                    model_qxf=args.model, tokenizer_qxt=args.tokenizer,
                    prompt=args.prompt_file,
                ),
                "platform": {
                    "system": platform.system(), "release": platform.release(),
                    "machine": platform.machine(), "processor": platform.processor(),
                    "python": platform.python_version(),
                },
                "fixed_arguments": {
                    "activation_modes": ["f32", "q8_k_compat"],
                    "kv": args.kv, "layers": args.layers, "ctx": args.ctx,
                    "generation_steps": args.generate, "seed": args.seed,
                    "temperature": 0, "warmups": args.warmups,
                    "repetitions": args.repetitions,
                },
            },
            "phase_definitions": {
                "startup_model_load": "Wall-clock qxqxf inspect process for the same QXF; reported separately and not subtracted from inference.",
                "prefill": "Native C clock() elapsed sum for fixed prompt positions before the first generation output, including per-position JSON emission.",
                "decode": "Native C clock() elapsed sum for generation positions including the complete final head and per-position JSON emission.",
                "total": "Wall-clock prompt-state-loop process including tokenizer/model open, allocation, prefill, decode, JSON emission and teardown.",
                "peak_rss": "Maximum process resident set sampled by psutil at 5 ms intervals.",
            },
            "startup_model_load": {
                "command": startup_command,
                "warmups": [run["wall_elapsed_seconds"] for run in startup_warmups],
                "measured": [run["wall_elapsed_seconds"] for run in startup_measured],
                "summary_seconds": summarize([run["wall_elapsed_seconds"] for run in startup_measured]),
            },
            "cells": cells,
            "output_contract": output_gate,
            "comparisons": comparisons,
        }
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        print(f"benchmark failed closed: {exc}", file=sys.stderr)
        return 2

    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": "pass", "output": str(args.output),
        "report_sha256": hashlib.sha256(encoded).hexdigest(),
        "selected_tokens_by_mode": output_gate["selected_tokens_by_mode"],
        "comparisons": comparisons,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())