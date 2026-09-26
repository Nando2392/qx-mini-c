#!/usr/bin/env python
"""Strict raw full-logit sidecar validation for the Issue 86 test driver.

The production CLI has no full-logit dump option.  The versioned acceptance
*driver* writes raw little-endian F32 files named ``step-N-logits.f32`` into its
already-existing ``--dump-dir``.  These helpers validate that actual schema and
compare bytes directly; SHA-256 values are provenance, never the equality gate.
"""
from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path
from typing import Any, Iterable


class FullLogitBytesError(RuntimeError):
    """Fail-closed raw-logit evidence error."""


def _exact_int(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FullLogitBytesError(f"{label} must be an integer >= {minimum}")
    return value


def enable_full_logit_capture(command: list[str], run_dir: Path) -> Path:
    """Enable the shipping state-loop dump seam; never invent a prompt CLI flag."""
    if len(command) < 2 or command[1] != "state-loop-probe":
        raise FullLogitBytesError(
            "full-logit capture requires state-loop-probe --dump-residuals; "
            "prompt-state-loop-probe has no dump CLI option"
        )
    capture_dir = run_dir / "full-logit-bytes"
    capture_dir.mkdir(parents=False, exist_ok=False)
    if "--dump-residuals" in command:
        raise FullLogitBytesError("command already configures --dump-residuals")
    command.extend(("--dump-residuals", str(capture_dir)))
    return capture_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_argmax(path: Path, vocab_size: int) -> int:
    """Require finite little-endian F32 values and return first-index argmax."""
    raw = path.read_bytes()
    if len(raw) != vocab_size * 4:
        raise FullLogitBytesError("raw logit byte count changed during validation")
    best_index = -1
    best_value = -math.inf
    for index, (value,) in enumerate(struct.iter_unpack("<f", raw)):
        if not math.isfinite(value):
            raise FullLogitBytesError(f"non-finite full logit at vocabulary index {index}")
        if best_index < 0 or value > best_value:
            best_index, best_value = index, value
    return best_index


def collect_generated_logits(
    capture_dir: Path,
    token_steps: list[dict[str, Any]],
    vocab_size: int,
    evidence_root: Path,
) -> list[dict[str, Any]]:
    """Validate every actual ``tokens[]`` generate record and its step-N sidecar."""
    vocab_size = _exact_int(vocab_size, "vocab_size", 1)
    if not isinstance(token_steps, list) or not token_steps:
        raise FullLogitBytesError("token_steps must be a non-empty list")
    expected_bytes = vocab_size * 4
    expected_files: set[str] = set()
    records: list[dict[str, Any]] = []
    for record_index, token_step in enumerate(token_steps):
        if not isinstance(token_step, dict):
            raise FullLogitBytesError(f"invalid token step at index {record_index}")
        if token_step.get("phase") != "generate":
            raise FullLogitBytesError(f"token step {record_index} phase must be generate")
        step = _exact_int(token_step.get("step"), f"token step {record_index}.step")
        if step != record_index:
            raise FullLogitBytesError("generated token steps must be contiguous from zero")
        selected = _exact_int(
            token_step.get("selected_token"), f"token step {record_index}.selected_token"
        )
        if selected >= vocab_size:
            raise FullLogitBytesError("selected_token is outside the vocabulary")
        name = f"step-{step}-logits.f32"
        expected_files.add(name)
        path = capture_dir / name
        if not path.is_file():
            raise FullLogitBytesError(f"missing generated-position logits: {name}")
        size = path.stat().st_size
        if size != expected_bytes:
            raise FullLogitBytesError(
                f"{name} size {size} != full vocabulary F32 size {expected_bytes}"
            )
        argmax = _raw_argmax(path, vocab_size)
        if argmax != selected:
            raise FullLogitBytesError(
                f"{name} raw argmax {argmax} != tokens[].selected_token {selected}"
            )
        try:
            relative = path.resolve().relative_to(evidence_root.resolve()).as_posix()
        except ValueError as exc:
            raise FullLogitBytesError("logit artifact escaped evidence root") from exc
        records.append({
            "step": step,
            "selected_token": selected,
            "argmax_from_raw_f32": argmax,
            "path": relative,
            "size_bytes": size,
            "f32_count": vocab_size,
            "sha256": _sha256(path),
        })
    actual_files = {path.name for path in capture_dir.glob("step-*-logits.f32")}
    if actual_files != expected_files:
        raise FullLogitBytesError(
            f"full-logit sidecar coverage mismatch: expected {sorted(expected_files)}, "
            f"found {sorted(actual_files)}"
        )
    return records


_RECORD_FIELDS = {
    "step", "selected_token", "argmax_from_raw_f32", "path", "size_bytes",
    "f32_count", "sha256",
}


def _paths(records: list[dict[str, Any]], evidence_root: Path) -> list[Path]:
    paths: list[Path] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
            raise FullLogitBytesError("invalid full-logit artifact record")
        step = _exact_int(record["step"], "artifact step")
        selected = _exact_int(record["selected_token"], "artifact selected_token")
        if _exact_int(record["argmax_from_raw_f32"], "artifact argmax") != selected:
            raise FullLogitBytesError("stored raw argmax does not match selected token")
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise FullLogitBytesError("non-portable full-logit artifact path")
        if relative.name != f"step-{step}-logits.f32":
            raise FullLogitBytesError("artifact path does not match its step")
        path = (evidence_root / relative).resolve()
        try:
            path.relative_to(evidence_root.resolve())
        except ValueError as exc:
            raise FullLogitBytesError("logit artifact escaped evidence root") from exc
        size = _exact_int(record["size_bytes"], "artifact size_bytes", 1)
        count = _exact_int(record["f32_count"], "artifact f32_count", 1)
        if size != count * 4:
            raise FullLogitBytesError("artifact size is not its full F32 vocabulary")
        if (
            not path.is_file()
            or path.stat().st_size != size
            or _sha256(path) != record["sha256"]
            or _raw_argmax(path, count) != selected
        ):
            raise FullLogitBytesError(f"frozen full-logit artifact changed: {relative.as_posix()}")
        paths.append(path)
    return paths


def _byte_equal(left: Path, right: Path) -> bool:
    """Compare bytes directly; hashes are provenance, never the equality gate."""
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            ac = a.read(1024 * 1024)
            bc = b.read(1024 * 1024)
            if ac != bc:
                return False
            if not ac:
                return True


def require_exact_policy_bytes(
    none_runs: Iterable[list[dict[str, Any]]],
    resident_runs: Iterable[list[dict[str, Any]]],
    evidence_root: Path,
) -> dict[str, Any]:
    """Require every NONE/resident run to match the first NONE run byte-for-byte."""
    none_groups, resident_groups = list(none_runs), list(resident_runs)
    if not none_groups or not resident_groups:
        raise FullLogitBytesError("both NONE and resident full-logit runs are required")
    groups = none_groups + resident_groups
    reference_records = groups[0]
    reference = _paths(reference_records, evidence_root)
    if not reference:
        raise FullLogitBytesError("full-logit run contains no generated positions")
    steps = [item["step"] for item in reference_records]
    selected_tokens = [item["selected_token"] for item in reference_records]
    for records in groups:
        if [item.get("step") for item in records] != steps:
            raise FullLogitBytesError("generated-step coverage differs between runs")
        if [item.get("selected_token") for item in records] != selected_tokens:
            raise FullLogitBytesError("generated selected tokens differ between runs")
        paths = _paths(records, evidence_root)
        if len(paths) != len(reference):
            raise FullLogitBytesError("generated-step count differs between runs")
        for step, left, right in zip(steps, reference, paths):
            if not _byte_equal(left, right):
                raise FullLogitBytesError(f"full logit bytes differ at generated step {step}")
    return {
        "full_logit_bytes_equal": True,
        "comparison_method": "direct_chunked_byte_comparison_not_hash_equality",
        "generated_steps": steps,
        "generated_token_ids": selected_tokens,
        "none_runs_compared": len(none_groups),
        "resident_runs_compared": len(resident_groups),
        "runs_compared": len(groups),
        "bytes_embedded_in_json": False,
    }
