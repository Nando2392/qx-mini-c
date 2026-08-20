#!/usr/bin/env python3
"""Validate fail-closed accumulated KV snapshot manifests and payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any, NamedTuple


SNAPSHOT_SCHEMA = "qx-kv-snapshot-v1"
KV_FORMATS = {"f16", "f32", "int8"}
ACTIVATION_FORMATS = {"f32", "q8_k_compat"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
RUNTIME_HEADER = struct.Struct("<8s10I")
RUNTIME_MAGIC = b"QXKVSNP1"
RUNTIME_FORMAT_IDS = {"int8": 1, "f16": 2, "f32": 3}
RUNTIME_FORMAT_NAMES = {value: key for key, value in RUNTIME_FORMAT_IDS.items()}
GEOMETRY_INTEGER_FIELDS = (
    "layers",
    "positions",
    "ctx_tokens",
    "kv_heads",
    "head_dim",
    "bytes_per_vector",
    "next_token",
    "seed",
)


class ValidatedSnapshot(NamedTuple):
    manifest: dict[str, Any]
    payload: bytes


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"manifest contains non-finite JSON number {value}")


def _object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"manifest contains duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_object_from_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read snapshot manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("snapshot manifest root must be a JSON object")
    return value


def _require_object(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _require_exact_int(parent: dict[str, Any], key: str, *, minimum: int = 0) -> int:
    value = parent.get(key)
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    if value < minimum:
        raise ValueError(f"{key} is out of range")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_prompt_tokens(value: object) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError("prompt_tokens must be a non-empty array")
    if any(type(token) is not int or token < 0 or token > 0xFFFFFFFF for token in value):
        raise ValueError("prompt_tokens must contain unsigned integers")
    return value


def _validate_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    kv_format = geometry.get("kv_format")
    if kv_format not in KV_FORMATS:
        raise ValueError("unsupported canonical kv_format")
    activation_format = geometry.get("activation_format")
    if activation_format not in ACTIVATION_FORMATS:
        raise ValueError("unsupported activation_format")
    values = {
        field: _require_exact_int(
            geometry,
            field,
            minimum=0 if field in {"next_token", "seed"} else 1,
        )
        for field in GEOMETRY_INTEGER_FIELDS
    }
    prompt_tokens = _validate_prompt_tokens(geometry.get("prompt_tokens"))
    if values["positions"] != len(prompt_tokens):
        raise ValueError("positions must equal prompt_tokens length")
    if values["positions"] > values["ctx_tokens"]:
        raise ValueError("positions exceed ctx_tokens")
    if values["kv_heads"] > 0xFFFFFFFF // values["head_dim"]:
        raise ValueError("KV geometry overflow")
    return {
        "kv_format": kv_format,
        "activation_format": activation_format,
        **values,
        "prompt_tokens": prompt_tokens,
    }


def _validate_expected_geometry(actual: dict[str, Any], expected: dict[str, object]) -> None:
    if not isinstance(expected, dict):
        raise ValueError("expected_geometry must be an object")
    geometry_only = {key: value for key, value in expected.items() if key != "producer_binary_sha256"}
    validated_expected = _validate_geometry(geometry_only)
    for key, actual_value in actual.items():
        if validated_expected.get(key) != actual_value:
            raise ValueError(f"geometry mismatch for {key}")


def _resolve_payload(manifest_path: Path, payload_path: object) -> Path:
    if not isinstance(payload_path, str) or not payload_path:
        raise ValueError("payload path must be a non-empty relative path")
    candidate = Path(payload_path)
    if candidate.is_absolute():
        raise ValueError("payload path must remain inside snapshot directory")
    root = manifest_path.parent.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("payload path must remain inside snapshot directory") from exc
    if not resolved.is_file():
        raise ValueError("payload path is not a file")
    return resolved


def _validate_scale(value: object, kv_format: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("entry scale must be numeric")
    scale = float(value)
    if not math.isfinite(scale):
        raise ValueError("entry scale is non-finite")
    if kv_format == "int8":
        if scale <= 0.0:
            raise ValueError("INT8 entry scale must be positive")
    elif scale != 1.0:
        raise ValueError("F16/F32 entry scale must equal 1")
    return scale


def _validate_entries(
    value: object,
    *,
    geometry: dict[str, Any],
    payload: bytes,
) -> None:
    if not isinstance(value, list):
        raise ValueError("entries must be an array")
    expected_keys = {
        (layer, position, kind)
        for layer in range(geometry["layers"])
        for position in range(geometry["positions"])
        for kind in ("k", "v")
    }
    seen: set[tuple[int, int, str]] = set()
    vector_bytes = geometry["bytes_per_vector"]
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ValueError(f"entry {index} must be an object")
        layer = _require_exact_int(entry, "layer")
        position = _require_exact_int(entry, "position")
        kind = entry.get("kind")
        offset = _require_exact_int(entry, "offset")
        byte_count = _require_exact_int(entry, "bytes", minimum=1)
        if kind not in {"k", "v"}:
            raise ValueError("entry kind must be k or v")
        key = (layer, position, kind)
        if key not in expected_keys or key in seen:
            raise ValueError("entry coverage is incomplete or duplicated")
        seen.add(key)
        record_index = layer * geometry["positions"] + position
        record_offset = RUNTIME_HEADER.size + record_index * (2 * vector_bytes + 8)
        expected_offset = record_offset if kind == "k" else record_offset + vector_bytes
        if byte_count != vector_bytes or offset != expected_offset or offset > len(payload) - byte_count:
            raise ValueError("entry payload range is invalid")
        scale = _validate_scale(entry.get("scale"), geometry["kv_format"])
        scale_offset = record_offset + 2 * vector_bytes + (0 if kind == "k" else 4)
        runtime_scale = struct.unpack_from("<f", payload, scale_offset)[0]
        if runtime_scale != scale:
            raise ValueError("entry scale does not match runtime payload")
        block = payload[offset : offset + byte_count]
        if sha256_bytes(block) != _require_sha256(entry.get("sha256"), "entry sha256"):
            raise ValueError("entry hash mismatch")
    if seen != expected_keys:
        raise ValueError("entry coverage is incomplete or duplicated")


def _parse_runtime_payload(payload: bytes) -> dict[str, Any]:
    if len(payload) < RUNTIME_HEADER.size:
        raise ValueError("runtime snapshot header is truncated")
    (
        magic,
        version,
        layers,
        positions,
        ctx_tokens,
        kv_heads,
        head_dim,
        format_id,
        bytes_per_vector,
        next_token,
        seed,
    ) = RUNTIME_HEADER.unpack_from(payload)
    kv_format = RUNTIME_FORMAT_NAMES.get(format_id)
    if magic != RUNTIME_MAGIC or version != 2 or kv_format is None:
        raise ValueError("runtime snapshot header mismatch")
    integer_values = (layers, positions, ctx_tokens, kv_heads, head_dim, bytes_per_vector)
    if any(value == 0 for value in integer_values) or positions > ctx_tokens:
        raise ValueError("runtime snapshot geometry is invalid")
    expected_bytes = RUNTIME_HEADER.size + layers * positions * (2 * bytes_per_vector + 8) + 32
    if len(payload) != expected_bytes:
        raise ValueError("runtime snapshot length does not match header")
    if hashlib.sha256(payload[:-32]).digest() != payload[-32:]:
        raise ValueError("runtime snapshot embedded SHA-256 mismatch")
    return {
        "kv_format": kv_format,
        "layers": layers,
        "positions": positions,
        "ctx_tokens": ctx_tokens,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "bytes_per_vector": bytes_per_vector,
        "next_token": next_token,
        "seed": seed,
    }


def build_snapshot_manifest(
    snapshot_path: Path,
    *,
    model_path: Path,
    producer_binary_path: Path,
    revision: str,
    activation_format: str,
    prompt_tokens: list[int],
) -> dict[str, Any]:
    """Build provenance and per-vector hashes for a native QXKVSNP1 payload."""
    snapshot_path = Path(snapshot_path)
    model_path = Path(model_path)
    producer_binary_path = Path(producer_binary_path)
    if not snapshot_path.is_file() or not model_path.is_file() or not producer_binary_path.is_file():
        raise ValueError("snapshot, model, and producer binary must be files")
    if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision):
        raise ValueError("producer revision must be a lowercase 40-character Git id")
    if activation_format not in ACTIVATION_FORMATS:
        raise ValueError("unsupported activation_format")
    tokens = _validate_prompt_tokens(prompt_tokens)
    payload = snapshot_path.read_bytes()
    runtime = _parse_runtime_payload(payload)
    if len(tokens) != runtime["positions"]:
        raise ValueError("prompt_tokens must cover every accumulated position")
    vector_bytes = runtime["bytes_per_vector"]
    entries: list[dict[str, Any]] = []
    for layer in range(runtime["layers"]):
        for position in range(runtime["positions"]):
            record_index = layer * runtime["positions"] + position
            record_offset = RUNTIME_HEADER.size + record_index * (2 * vector_bytes + 8)
            for kind, offset, scale_offset in (
                ("k", record_offset, record_offset + 2 * vector_bytes),
                ("v", record_offset + vector_bytes, record_offset + 2 * vector_bytes + 4),
            ):
                block = payload[offset : offset + vector_bytes]
                entries.append({
                    "layer": layer,
                    "position": position,
                    "kind": kind,
                    "offset": offset,
                    "bytes": vector_bytes,
                    "sha256": sha256_bytes(block),
                    "scale": struct.unpack_from("<f", payload, scale_offset)[0],
                })
    geometry = {**runtime, "activation_format": activation_format, "prompt_tokens": list(tokens)}
    return {
        "schema": SNAPSHOT_SCHEMA,
        "model": {"size": model_path.stat().st_size, "sha256": sha256_file(model_path)},
        "producer": {"binary_sha256": sha256_file(producer_binary_path), "revision": revision},
        "geometry": geometry,
        "payload": {"path": snapshot_path.name, "bytes": len(payload), "sha256": sha256_bytes(payload)},
        "entries": entries,
    }


def load_snapshot(
    manifest_path: Path,
    *,
    model_path: Path,
    expected_geometry: dict[str, object],
) -> ValidatedSnapshot:
    """Load and validate a KV snapshot without accepting partial provenance."""
    manifest_path = Path(manifest_path)
    model_path = Path(model_path)
    manifest = _load_manifest(manifest_path)
    if manifest.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("unsupported snapshot schema")

    model = _require_object(manifest, "model")
    expected_model_size = _require_exact_int(model, "size", minimum=1)
    expected_model_hash = _require_sha256(model.get("sha256"), "model sha256")
    if not model_path.is_file() or model_path.stat().st_size != expected_model_size:
        raise ValueError("model size mismatch")
    if sha256_file(model_path) != expected_model_hash:
        raise ValueError("model hash mismatch")

    producer = _require_object(manifest, "producer")
    producer_hash = _require_sha256(producer.get("binary_sha256"), "producer binary_sha256")
    expected_producer_hash = _require_sha256(
        expected_geometry.get("producer_binary_sha256"),
        "expected producer binary_sha256",
    )
    if producer_hash != expected_producer_hash:
        raise ValueError("producer binary hash mismatch")
    revision = producer.get("revision")
    if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision):
        raise ValueError("producer revision must be a lowercase 40-character Git id")

    geometry = _validate_geometry(_require_object(manifest, "geometry"))
    _validate_expected_geometry(geometry, expected_geometry)

    payload_meta = _require_object(manifest, "payload")
    payload_path = _resolve_payload(manifest_path, payload_meta.get("path"))
    payload = payload_path.read_bytes()
    payload_bytes = _require_exact_int(payload_meta, "bytes", minimum=1)
    if len(payload) != payload_bytes:
        raise ValueError("payload length mismatch")
    if sha256_bytes(payload) != _require_sha256(payload_meta.get("sha256"), "payload sha256"):
        raise ValueError("payload hash mismatch")
    runtime = _parse_runtime_payload(payload)
    for key in ("kv_format", *GEOMETRY_INTEGER_FIELDS):
        if runtime[key] != geometry[key]:
            raise ValueError(f"runtime payload geometry mismatch for {key}")
    _validate_entries(manifest.get("entries"), geometry=geometry, payload=payload)
    return ValidatedSnapshot(manifest=manifest, payload=payload)


def _parse_prompt_tokens(value: str) -> list[int]:
    try:
        tokens = [int(part, 10) for part in value.split(",") if part]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("prompt tokens must be comma-separated integers") from exc
    try:
        return _validate_prompt_tokens(tokens)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build a manifest for a native QXKVSNP1 snapshot")
    build.add_argument("--snapshot", type=Path, required=True)
    build.add_argument("--model", type=Path, required=True)
    build.add_argument("--binary", type=Path, required=True)
    build.add_argument("--revision", required=True)
    build.add_argument("--activation", choices=sorted(ACTIVATION_FORMATS), required=True)
    build.add_argument("--prompt-tokens", type=_parse_prompt_tokens, required=True)
    build.add_argument("--out", type=Path, required=True)
    validate = subparsers.add_parser("validate", help="validate manifest, provenance, geometry, and payload")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--model", type=Path, required=True)
    validate.add_argument("--binary", type=Path, required=True)
    validate.add_argument("--expected-geometry", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        if args.out.resolve().parent != args.snapshot.resolve().parent:
            raise ValueError("manifest output must be beside its snapshot payload")
        manifest = build_snapshot_manifest(
            args.snapshot,
            model_path=args.model,
            producer_binary_path=args.binary,
            revision=args.revision,
            activation_format=args.activation,
            prompt_tokens=args.prompt_tokens,
        )
        args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"manifest": str(args.out), "payload_sha256": manifest["payload"]["sha256"]}))
        return 0
    expected = _load_manifest(args.expected_geometry)
    if not args.binary.is_file():
        raise ValueError("producer binary must be a file")
    expected["producer_binary_sha256"] = sha256_file(args.binary)
    snapshot = load_snapshot(args.manifest, model_path=args.model, expected_geometry=expected)
    print(json.dumps({"valid": True, "payload_sha256": snapshot.manifest["payload"]["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
