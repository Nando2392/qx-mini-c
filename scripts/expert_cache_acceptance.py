#!/usr/bin/env python
"""Issue 86 bounded real A/B acceptance harness.

This runner performs no build.  It parses the QXF directory itself, derives a
packed top-k/all-layer working-set budget, freezes inputs with streaming SHA-256,
and runs the versioned test-only acceptance driver for F32 and Q8_K
compatibility.  The driver exports complete raw F32 logits without adding a
phantom production-CLI flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from expert_cache_logit_bytes import (
    FullLogitBytesError,
    collect_generated_logits,
    require_exact_policy_bytes,
)

try:
    import psutil
except ImportError as exc:  # pragma: no cover
    raise SystemExit("psutil is required for Issue 86 acceptance") from exc

SCHEMA = "qx-issue86-expert-cache-real-ab-v2"
PROMPT_TEXT = "Hi"
VOCAB_SIZE = 151_936
EXPECTED_F32_TOKENS = [1124, 77]
ACTIVATIONS = ("f32", "q8_k_compat")
POLICIES = ("none", "resident-packed")
WARMUPS = 1
MEASURED = 3
LAYERS = 48
CTX = 16
GENERATE = 2
TOP_K = 8
PER_PROCESS_TIMEOUT_SECONDS = 120.0
PARENT_TIMEOUT_SECONDS = 2100.0
RSS_SAMPLE_INTERVAL_SECONDS = 0.01
MAX_REASONABLE_CACHE_BYTES = 2 * 1024**3
HELPER_FILES = (
    "scripts/build_expert_cache_acceptance_driver.cmd",
    "scripts/expert_cache_acceptance.py",
    "scripts/expert_cache_logit_bytes.py",
    "tests/expert_cache_acceptance_driver.c",
)
SOURCE_SUFFIXES = {".c", ".h", ".cc", ".cpp", ".hpp"}
EXPECTED_RESIDENT_REQUESTS = GENERATE * LAYERS * TOP_K * 3
MAX_CANONICAL_EXPERT_SLICE_BYTES = 802_688
CACHE_FIELDS = {
    "policy", "enabled", "budget_bytes", "requests", "hits", "misses",
    "loads", "evictions", "current_resident_packed_bytes",
    "peak_resident_packed_bytes", "buffered_bytes_read",
    "buffered_bytes_avoided",
}


class AcceptanceError(RuntimeError):
    """Fail-closed acceptance error."""


@dataclass(frozen=True)
class QxfTensor:
    """QXF tensor-directory fields needed for budget arithmetic."""

    name: str
    offset: int
    byte_size: int


@dataclass(frozen=True)
class QxfMetadata:
    """Validated QXF shape and packed expert tensor directory."""

    file_size: int
    layers: int
    experts: int
    experts_per_token: int
    tensors: dict[str, QxfTensor]
    hidden: int = 0
    intermediate: int = 0
    moe_intermediate: int = 0


def exact_int(value: object, label: str, minimum: int = 0) -> int:
    """Require a non-boolean integer with a lower bound."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AcceptanceError(f"{label} must be an integer >= {minimum}")
    return value


def positive_number(value: object, label: str) -> float:
    """Require a finite positive numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcceptanceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise AcceptanceError(f"{label} must be finite and positive")
    return result


def object_value(value: object, label: str) -> dict[str, Any]:
    """Require a JSON object."""
    if not isinstance(value, dict):
        raise AcceptanceError(f"{label} must be an object")
    return value


def check_deadline(deadline: float | None, phase: str) -> None:
    """Fail closed when the parent budget expires outside native execution."""
    if deadline is not None and time.monotonic() >= deadline:
        raise AcceptanceError(f"parent hard timeout reached during {phase}")


def sha256_file(path: Path, deadline: float | None = None) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            check_deadline(deadline, f"hashing {path}")
            digest.update(chunk)
    check_deadline(deadline, f"finalizing hash for {path}")
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return SHA-256 for in-memory bytes."""
    return hashlib.sha256(data).hexdigest()


def portable_path(path: Path, root: Path) -> str:
    """Return a repository-relative POSIX path or reject the path."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise AcceptanceError(f"path must remain inside repository: {path}") from exc


def artifact(path: Path, root: Path, deadline: float | None = None) -> dict[str, Any]:
    """Record a portable, streaming-hashed file artifact."""
    if not path.is_file():
        raise AcceptanceError(f"missing required file: {path}")
    return {
        "path": portable_path(path, root),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path, deadline),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Durably replace a JSON artifact."""
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_qxf(path: Path) -> QxfMetadata:
    """Parse the native QXF header/directory without invoking model code."""
    actual_size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(272)
        if len(prefix) < 108 or prefix[:4] != b"QXF1":
            raise AcceptanceError("model is not a complete QXF1 file")
        version, header_size, entry_size, tensor_count = struct.unpack_from("<4I", prefix, 4)
        directory_offset, _data_offset, declared_size, _manifest_checksum = struct.unpack_from("<4Q", prefix, 24)
        manifest = struct.unpack_from("<13I", prefix, 56)
        layers, hidden, intermediate = manifest[2], manifest[3], manifest[4]
        experts, experts_per_token, moe_intermediate = manifest[10], manifest[11], manifest[12]
        if version != 1 or header_size < 108 or entry_size < 176:
            raise AcceptanceError("unsupported QXF header or directory layout")
        if declared_size != actual_size:
            raise AcceptanceError(f"QXF declared size {declared_size} != actual size {actual_size}")
        if layers != LAYERS or experts <= 0 or experts_per_token != TOP_K:
            raise AcceptanceError(
                f"QXF manifest mismatch: layers={layers}, experts={experts}, top_k={experts_per_token}"
            )
        if hidden <= 0 or intermediate <= 0 or moe_intermediate <= 0:
            raise AcceptanceError(
                "QXF architecture widths must be positive: "
                f"hidden={hidden}, intermediate={intermediate}, moe_intermediate={moe_intermediate}"
            )
        if directory_offset + tensor_count * entry_size > actual_size:
            raise AcceptanceError("QXF directory extends beyond file")
        tensors: dict[str, QxfTensor] = {}
        stream.seek(directory_offset)
        for index in range(tensor_count):
            raw = stream.read(entry_size)
            if len(raw) != entry_size:
                raise AcceptanceError(f"short QXF directory entry {index}")
            name_bytes = raw[:96].split(b"\0", 1)[0]
            try:
                name = name_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise AcceptanceError(f"invalid UTF-8 tensor name at entry {index}") from exc
            offset, byte_size = struct.unpack_from("<QQ", raw, 144)
            if not name or offset > actual_size or byte_size > actual_size - offset:
                raise AcceptanceError(f"invalid tensor span at directory entry {index}")
            if name in tensors:
                raise AcceptanceError(f"duplicate QXF tensor name: {name}")
            tensors[name] = QxfTensor(name, offset, byte_size)
    return QxfMetadata(
        actual_size, layers, experts, experts_per_token, tensors,
        hidden=hidden, intermediate=intermediate, moe_intermediate=moe_intermediate,
    )


def derive_budget(metadata: QxfMetadata, available_bytes: int) -> dict[str, Any]:
    """Derive a top-k/all-layer packed budget from actual gate/up/down sizes."""
    per_layer: list[dict[str, Any]] = []
    working_set = 0
    maximum_slice = 0
    for layer in range(metadata.layers):
        sizes: dict[str, int] = {}
        for kind in ("gate", "up", "down"):
            name = f"blk.{layer}.ffn_{kind}_exps.weight"
            tensor = metadata.tensors.get(name)
            if tensor is None:
                raise AcceptanceError(f"missing QXF expert tensor: {name}")
            if tensor.byte_size % metadata.experts:
                raise AcceptanceError(f"expert tensor is not evenly divisible: {name}")
            size = tensor.byte_size // metadata.experts
            if size <= 0:
                raise AcceptanceError(f"empty packed expert slice: {name}")
            sizes[kind] = size
            maximum_slice = max(maximum_slice, size)
        one_expert = sum(sizes.values())
        layer_working = one_expert * metadata.experts_per_token
        working_set += layer_working
        per_layer.append({
            "layer": layer,
            "gate_slice_bytes": sizes["gate"],
            "up_slice_bytes": sizes["up"],
            "down_slice_bytes": sizes["down"],
            "one_expert_gate_up_down_bytes": one_expert,
            "top_k_working_set_bytes": layer_working,
        })
    safe_host_limit = min(available_bytes // 4, MAX_REASONABLE_CACHE_BYTES)
    feasible = working_set >= maximum_slice and working_set <= safe_host_limit
    return {
        "formula": "sum(layer=0..47, top_k * (gate_slice + up_slice + down_slice))",
        "budget_bytes": working_set,
        "maximum_single_admission_bytes": maximum_slice,
        "available_host_bytes_at_preflight": available_bytes,
        "safe_host_limit_formula": "min(psutil.virtual_memory().available // 4, 2 GiB)",
        "safe_host_limit_bytes": safe_host_limit,
        "budget_at_most_available_host_bytes": working_set <= available_bytes,
        "budget_at_most_safe_host_limit": working_set <= safe_host_limit,
        "admission_fits_budget": maximum_slice <= working_set,
        "feasible": feasible,
        "scope": "one top-8 packed gate/up/down working set for every one of 48 layers; not all 128 experts",
        "per_layer": per_layer,
    }


def frozen_source_paths(root: Path) -> list[Path]:
    """Return every native source plus acceptance helper/driver/build files."""
    native = sorted(
        path
        for directory in (root / "include", root / "src")
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
    )
    helpers = [root / relative for relative in HELPER_FILES]
    paths = sorted({path.resolve() for path in (*native, *helpers)})
    if not paths:
        raise AcceptanceError("acceptance source freeze is empty")
    return paths


def freeze(
    root: Path, executable: Path, model: Path, tokenizer: Path,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Freeze source, executable, model, and tokenizer bytes."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
        timeout=30, check=True,
    ).stdout.strip()
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise AcceptanceError("git revision is not a lowercase SHA-1")
    check_deadline(deadline, "pre-run provenance collection")
    sources = [artifact(path, root, deadline) for path in frozen_source_paths(root)]
    return {
        "git_revision": revision,
        "source_files": sources,
        "source_manifest_sha256": sha256_bytes(
            json.dumps(sources, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "native_executable": artifact(executable, root, deadline),
        "model_qxf": artifact(model, root, deadline),
        "tokenizer_qxt": artifact(tokenizer, root, deadline),
    }


def command_for(
    executable: Path, model: Path, tokenizer: Path, prompt: Path,
    dump_dir: Path, activation: str, policy: str, budget: int,
) -> list[str]:
    """Build the exact versioned acceptance-driver command."""
    policy_budget = budget if policy == "resident-packed" else 0
    return [
        str(executable), "--model", str(model), "--tokenizer", str(tokenizer),
        "--prompt-file", str(prompt), "--activation", activation,
        "--expert-cache-policy", policy, "--expert-cache-budget-bytes",
        str(policy_budget), "--dump-dir", str(dump_dir), "--ctx", str(CTX),
        "--generate", str(GENERATE), "--io-backend", "buffered",
    ]


def sanitized_command(activation: str, policy: str, budget: int) -> list[str]:
    """Return the portable reproduction command stored in evidence."""
    return command_for(
        Path("build/issue86-driver-testonly/expert_cache_acceptance_driver.exe"),
        Path("models/Qwen3-30B-A3B-UD-IQ2_M.qxf"),
        Path("models/Qwen3-30B-A3B.qxt"), Path("<output-dir>/prompt.txt"),
        Path("<output-dir>/<fresh-run-dir>"), activation, policy, budget,
    )


def stop_tree(process: subprocess.Popen[bytes]) -> None:
    """Terminate the child and all descendants after a hard deadline."""
    processes: list[psutil.Process] = []
    try:
        parent = psutil.Process(process.pid)
        processes = [*parent.children(recursive=True), parent]
    except psutil.Error:
        pass
    for item in processes:
        try:
            item.kill()
        except psutil.Error:
            pass
    try:
        process.kill()
    except OSError:
        pass
    process.wait(timeout=10)
    _, alive = psutil.wait_procs(processes, timeout=5)
    if alive:
        raise AcceptanceError(f"timed-out process tree survived: {[item.pid for item in alive]}")


def run_one(command: list[str], run_dir: Path, parent_deadline: float) -> dict[str, Any]:
    """Run one process with durable logs, sampled RSS, and hard deadlines."""
    remaining = parent_deadline - time.monotonic()
    if remaining <= 0.0:
        raise AcceptanceError("parent hard timeout reached before process start")
    timeout = min(PER_PROCESS_TIMEOUT_SECONDS, remaining)
    stdout_path, stderr_path = run_dir / "stdout.json", run_dir / "stderr.log"
    started = time.monotonic()
    peak_rss = 0
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        tracked = psutil.Process(process.pid)
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            try:
                peak_rss = max(
                    peak_rss,
                    tracked.memory_info().rss
                    + sum(child.memory_info().rss for child in tracked.children(recursive=True)),
                )
            except psutil.Error:
                pass
            if time.monotonic() >= deadline:
                timed_out = True
                stop_tree(process)
                break
            time.sleep(RSS_SAMPLE_INTERVAL_SECONDS)
    elapsed = time.monotonic() - started
    evidence_root = run_dir.parent
    record = {
        "returncode": process.returncode,
        "timed_out": timed_out,
        "wall_seconds": elapsed,
        "sampled_peak_rss_bytes": peak_rss,
        "stdout": artifact(stdout_path, evidence_root),
        "stderr": artifact(stderr_path, evidence_root),
    }
    atomic_json(run_dir / "process.json", record)
    if timed_out:
        raise AcceptanceError(f"process exceeded hard timeout of {timeout:.3f} seconds")
    if process.returncode != 0:
        tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise AcceptanceError(f"native process failed ({process.returncode}): {tail}")
    if peak_rss <= 0:
        raise AcceptanceError("sampled peak RSS was not positive")
    try:
        payload = json.loads(stdout_path.read_text(encoding="utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AcceptanceError("native stdout is not exactly one UTF-8 JSON value") from exc
    record["payload"] = object_value(payload, "native payload")
    return record


def compact_run(
    raw: dict[str, Any], activation: str, policy: str, budget: int,
    capture_dir: Path | None = None, evidence_root: Path | None = None,
) -> dict[str, Any]:
    """Validate one real run and retain exact output/cache evidence."""
    payload = object_value(raw.get("payload"), "payload")
    if payload.get("activation_format") != activation or payload.get("io_backend") != "buffered":
        raise AcceptanceError("activation or buffered-I/O provenance mismatch")
    if exact_int(payload.get("layers"), "layers") != LAYERS or exact_int(payload.get("ctx_tokens"), "ctx") != CTX:
        raise AcceptanceError("layer/context provenance mismatch")
    prompt_count = exact_int(payload.get("prompt_token_count"), "prompt_token_count", 1)
    steps = exact_int(payload.get("steps"), "steps", 1)
    if steps > CTX or steps > 16:
        raise AcceptanceError("forward-position hard wall exceeded")
    if prompt_count + GENERATE - 1 != steps:
        raise AcceptanceError("prompt/generation forward-position accounting mismatch")
    token_steps = payload.get("tokens")
    if not isinstance(token_steps, list) or len(token_steps) != steps:
        raise AcceptanceError("token step array mismatch")
    generation = [object_value(item, "token step") for item in token_steps if isinstance(item, dict) and item.get("phase") == "generate"]
    if len(generation) != GENERATE:
        raise AcceptanceError("expected exactly two generated positions")
    selected = [exact_int(item.get("selected_token"), "selected token") for item in generation]
    if activation == "f32" and selected != EXPECTED_F32_TOKENS:
        raise AcceptanceError(
            f"canonical F32 generated tokens changed: {selected} != {EXPECTED_F32_TOKENS}"
        )
    checksums: list[str] = []
    for item in generation:
        final_head = object_value(item.get("final_head"), "generation final_head")
        checksum = exact_int(final_head.get("logits_checksum"), "full logits checksum")
        if final_head.get("full_vocabulary") is not True or exact_int(final_head.get("logits_computed"), "logits computed", 1) != 151936:
            raise AcceptanceError("final head did not checksum the full vocabulary")
        checksums.append(str(checksum))
    cache = object_value(payload.get("expert_cache_profile"), "expert_cache_profile")
    if set(cache) != CACHE_FIELDS:
        raise AcceptanceError("expert-cache v3 JSON field set mismatch")
    counters = {key: exact_int(cache.get(key), f"cache.{key}") for key in CACHE_FIELDS - {"policy", "enabled"}}
    expected_enabled = policy == "resident-packed"
    if cache.get("policy") != policy or cache.get("enabled") is not expected_enabled:
        raise AcceptanceError("expert-cache requested/effective policy mismatch")
    if counters["budget_bytes"] != (budget if expected_enabled else 0):
        raise AcceptanceError("expert-cache budget provenance mismatch")
    if counters["requests"] != counters["hits"] + counters["misses"]:
        raise AcceptanceError("cache requests != hits + misses")
    if counters["misses"] != counters["loads"]:
        raise AcceptanceError("cache misses != loads")
    if counters["peak_resident_packed_bytes"] > counters["budget_bytes"]:
        raise AcceptanceError("peak packed residency exceeded configured budget")
    if counters["current_resident_packed_bytes"] > counters["budget_bytes"]:
        raise AcceptanceError("current packed residency exceeded configured budget")
    if expected_enabled:
        if counters["requests"] != EXPECTED_RESIDENT_REQUESTS:
            raise AcceptanceError(
                f"resident cache requests {counters['requests']} != {EXPECTED_RESIDENT_REQUESTS}"
            )
        if counters["requests"] <= 0 or counters["loads"] <= 0 or counters["buffered_bytes_read"] <= 0:
            raise AcceptanceError("resident run did not prove real cache requests/loads/buffered reads")
        if counters["hits"] <= 0 or counters["buffered_bytes_avoided"] <= 0:
            raise AcceptanceError(
                "positive repeated-expert proof was not achieved by the bounded real workload"
            )
    elif any(counters[key] != 0 for key in counters):
        raise AcceptanceError("NONE policy must have zero inactive cache state and counters")
    full_logits = None
    if capture_dir is not None or evidence_root is not None:
        if capture_dir is None or evidence_root is None:
            raise AcceptanceError("capture_dir and evidence_root must be supplied together")
        try:
            full_logits = collect_generated_logits(
                capture_dir, generation, VOCAB_SIZE, evidence_root,
            )
        except FullLogitBytesError as exc:
            raise AcceptanceError(str(exc)) from exc
    return {
        "wall_seconds": positive_number(raw.get("wall_seconds"), "wall_seconds"),
        "sampled_peak_rss_bytes": exact_int(raw.get("sampled_peak_rss_bytes"), "RSS", 1),
        "forward_positions": steps,
        "prompt_token_count": prompt_count,
        "generated_token_ids": selected,
        "full_logits_checksums": checksums,
        "full_logit_artifacts": full_logits,
        "cache": cache,
        "process": {
            "returncode": raw["returncode"], "timed_out": raw["timed_out"],
            "stdout": raw["stdout"], "stderr": raw["stderr"],
        },
    }


def summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize measured observations without making a speedup claim."""
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise AcceptanceError(f"measured run {index} must be an object")
        for field in ("wall_seconds", "sampled_peak_rss_bytes"):
            if field not in run:
                raise AcceptanceError(f"measured run {index} missing required field: {field}")
    walls = [positive_number(run["wall_seconds"], "wall") for run in runs]
    rss = [exact_int(run["sampled_peak_rss_bytes"], "RSS", 1) for run in runs]
    return {
        "count": len(runs),
        "wall_seconds": {"min": min(walls), "median": statistics.median(walls), "max": max(walls)},
        "sampled_peak_rss_bytes": {"min": min(rss), "median": statistics.median(rss), "max": max(rss)},
        "speedup_required": False,
    }


def equivalence(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Require exact outputs and raw full-logit bytes across every A/B run."""
    comparisons: list[dict[str, Any]] = []
    for activation in ACTIVATIONS:
        by_policy = {cell["policy"]: cell for cell in cells if cell["activation"] == activation}
        if set(by_policy) != set(POLICIES):
            raise AcceptanceError(f"missing A/B cells for {activation}")
        none_runs = by_policy["none"]["warmups"] + by_policy["none"]["measured"]
        resident_runs = by_policy["resident-packed"]["warmups"] + by_policy["resident-packed"]["measured"]
        none_outputs = [(run["generated_token_ids"], run["full_logits_checksums"]) for run in none_runs]
        resident_outputs = [(run["generated_token_ids"], run["full_logits_checksums"]) for run in resident_runs]
        all_outputs = none_outputs + resident_outputs
        if not all_outputs or any(item != all_outputs[0] for item in all_outputs[1:]):
            raise AcceptanceError(f"NONE/resident exact token or full-logit-checksum mismatch for {activation}")
        evidence_root_value = by_policy["none"].get("evidence_root")
        if not isinstance(evidence_root_value, str):
            raise AcceptanceError(f"missing evidence root for {activation}")
        if any(not isinstance(run.get("full_logit_artifacts"), list) for run in none_runs + resident_runs):
            raise AcceptanceError(f"missing full-logit byte artifacts for {activation}")
        try:
            byte_result = require_exact_policy_bytes(
                [run["full_logit_artifacts"] for run in none_runs],
                [run["full_logit_artifacts"] for run in resident_runs],
                Path(evidence_root_value),
            )
        except FullLogitBytesError as exc:
            raise AcceptanceError(str(exc)) from exc
        comparisons.append({
            "activation": activation, "generated_tokens_equal": True,
            "full_logits_checksums_equal": True, "runs_compared": len(all_outputs),
            "reference": {"generated_token_ids": all_outputs[0][0], "full_logits_checksums": all_outputs[0][1]},
            "full_logit_bytes": byte_result,
        })
    return {
        "status": "pass",
        "comparisons": comparisons,
        "full_logit_bytes_equal": True,
        "comparison_method": "direct_chunked_byte_comparison_not_hash_equality",
    }


def resolve(root: Path, value: Path) -> Path:
    """Resolve a CLI path relative to repository root."""
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def reproduction(budget: int) -> dict[str, Any]:
    """Describe the parent-approved launch command and fixed subprocess matrix."""
    return {
        "working_directory": "repository_root",
        "launch_only_after_tests_green": True,
        "parent_command": [
            "python", "scripts/expert_cache_acceptance.py", "--driver-exe",
            "build/issue86-driver-testonly/expert_cache_acceptance_driver.exe",
            "--model", "models/Qwen3-30B-A3B-UD-IQ2_M.qxf", "--tokenizer",
            "models/Qwen3-30B-A3B.qxt", "--output-dir", "build/issue86-expert-cache-ab",
        ],
        "cell_commands": [sanitized_command(activation, policy, budget) for activation in ACTIVATIONS for policy in POLICIES],
    }


def main() -> int:
    """Run the bounded real A/B matrix and publish fail-closed evidence."""
    parser = argparse.ArgumentParser(description="Issue 86 bounded real expert-cache A/B acceptance")
    parser.add_argument(
        "--driver-exe", type=Path,
        default=Path("build/issue86-driver-testonly/expert_cache_acceptance_driver.exe"),
    )
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-30B-A3B-UD-IQ2_M.qxf"))
    parser.add_argument("--tokenizer", type=Path, default=Path("models/Qwen3-30B-A3B.qxt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    executable, model, tokenizer, output_dir = (
        resolve(root, args.driver_exe), resolve(root, args.model),
        resolve(root, args.tokenizer), resolve(root, args.output_dir),
    )
    try:
        parent_deadline = time.monotonic() + PARENT_TIMEOUT_SECONDS
        output_relative = portable_path(output_dir, root)
        synthetic_test_path = output_relative == "synthetic/test-data" and output_dir.name.startswith("issue86-")
        if not output_relative.startswith("build/issue86-") and not synthetic_test_path:
            raise AcceptanceError("raw output directory must be a fresh repo/build/issue86-* path")
        if output_dir.exists():
            raise AcceptanceError("raw output directory must be fresh and not already exist")
        for path in (executable, model, tokenizer):
            if not path.is_file():
                raise AcceptanceError(f"missing required input: {path}")
            portable_path(path, root)
        output_dir.mkdir(parents=True, exist_ok=False)
        metadata = parse_qxf(model)
        available = int(psutil.virtual_memory().available)
        budget = derive_budget(metadata, available)
        atomic_json(output_dir / "budget.json", budget)
        if not budget["feasible"]:
            atomic_json(output_dir / "feasibility.json", {
                "status": "blocked_before_native_allocation",
                "reason": "top-k/all-48-layer packed working set exceeds the measured safe host budget",
                "budget": budget,
                "positive_hits_claimed": False,
            })
            print("Issue 86 acceptance blocked before native allocation: bounded positive-hit budget is infeasible", file=sys.stderr)
            return 3
        prompt_path = output_dir / "prompt.txt"
        prompt_path.write_text(PROMPT_TEXT, encoding="utf-8", newline="")
        frozen_pre = freeze(root, executable, model, tokenizer, parent_deadline)
        check_deadline(parent_deadline, "pre-run hash finalization")
        cells: list[dict[str, Any]] = []
        run_index = 0
        for activation in ACTIVATIONS:
            for policy in POLICIES:
                cell_runs: list[dict[str, Any]] = []
                for phase, count in (("warmup", WARMUPS), ("measured", MEASURED)):
                    for repetition in range(count):
                        run_index += 1
                        run_dir = output_dir / f"run-{run_index:02d}-{activation}-{policy}-{phase}-{repetition + 1}"
                        run_dir.mkdir()
                        command = command_for(
                            executable, model, tokenizer, prompt_path, run_dir,
                            activation, policy, int(budget["budget_bytes"]),
                        )
                        raw = run_one(command, run_dir, parent_deadline)
                        compact = compact_run(
                            raw, activation, policy, int(budget["budget_bytes"]),
                            run_dir, output_dir,
                        )
                        compact["phase"] = phase
                        compact["repetition"] = repetition + 1
                        cell_runs.append(compact)
                warmups = [run for run in cell_runs if run["phase"] == "warmup"]
                measured = [run for run in cell_runs if run["phase"] == "measured"]
                cells.append({
                    "activation": activation, "policy": policy,
                    "q8_promotion": False,
                    "command": sanitized_command(activation, policy, int(budget["budget_bytes"])),
                    "evidence_root": str(output_dir),
                    "warmups": warmups, "measured": measured, "summary": summary(measured),
                })
        exact_output = equivalence(cells)
        frozen_post = freeze(root, executable, model, tokenizer, parent_deadline)
        check_deadline(parent_deadline, "post-run hash finalization")
        if frozen_pre != frozen_post:
            raise AcceptanceError("frozen source/executable/model/tokenizer changed during acceptance")
        report = {
            "schema": SCHEMA,
            "issue": 86,
            "status": "pass",
            "claim_scope": (
                "Real buffered-CPU F32/Q8_K NONE versus resident-packed proof only; explicit buffered bytes, "
                "not physical disk I/O; no speedup, GPU, GEMM, default-promotion, or global parity claim."
            ),
            "fixed_contract": {
                "prompt_utf8": PROMPT_TEXT,
                "forward_positions_max": 16,
                "layers": LAYERS,
                "ctx": CTX,
                "generation_steps": GENERATE,
                "kv": "int8",
                "activations": list(ACTIVATIONS),
                "policies": list(POLICIES),
                "warmups_per_cell": WARMUPS,
                "measured_runs_per_cell": MEASURED,
                "total_processes": len(ACTIVATIONS) * len(POLICIES) * (WARMUPS + MEASURED),
                "per_process_hard_timeout_seconds": PER_PROCESS_TIMEOUT_SECONDS,
                "parent_hard_timeout_seconds": PARENT_TIMEOUT_SECONDS,
                "rss_sample_interval_seconds": RSS_SAMPLE_INTERVAL_SECONDS,
            },
            "qxf_budget_derivation": budget,
            "provenance": {
                "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "python": platform.python_version(), "psutil": psutil.__version__},
                "inputs_pre": frozen_pre, "inputs_post": frozen_post,
                "prompt": artifact(prompt_path, root),
            },
            "reproduction": reproduction(int(budget["budget_bytes"])),
            "cells": cells,
            "exact_output_equivalence": exact_output,
            "gates": {
                "real_model_and_prompt": True,
                "buffered_cpu_only": True,
                "f32_and_q8_without_promotion": True,
                "positive_hits_each_resident_run": True,
                "positive_explicit_buffered_bytes_avoided_each_resident_run": True,
                "cache_counter_consistency": True,
                "packed_residency_at_most_budget": True,
                "exact_generated_tokens": True,
                "exact_full_logits_checksums": True,
                "exact_full_logit_bytes": True,
                "frozen_inputs_unchanged": True,
                "issue_contract_complete": True,
            },
        }
        atomic_json(output_dir / "report.json", report)
        print(json.dumps({
            "status": report["status"],
            "report": portable_path(output_dir / "report.json", root),
            "sha256": sha256_file(output_dir / "report.json"),
        }, sort_keys=True))
        return 0
    except (AcceptanceError, FullLogitBytesError, OSError, subprocess.SubprocessError, ValueError) as exc:
        try:
            if output_dir.is_dir():
                atomic_json(output_dir / "failure.json", {
                    "schema": SCHEMA, "status": "failed_closed", "error_type": type(exc).__name__,
                    "error": str(exc), "positive_hits_claimed": False,
                })
        except OSError:
            pass
        print(f"Issue 86 acceptance failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
