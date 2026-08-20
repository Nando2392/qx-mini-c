import copy
import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "kv_snapshot_replay.py"
SPEC = importlib.util.spec_from_file_location("kv_snapshot_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
KV_SNAPSHOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KV_SNAPSHOT)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_fixture(tmp_path: Path, *, kv_format: str = "f16") -> tuple[Path, Path, dict[str, object]]:
    model = tmp_path / "model.qxf"
    model.write_bytes(b"synthetic-qxf-model")
    binary = tmp_path / "qxqxf.exe"
    binary.write_bytes(b"synthetic-qxqxf-binary")

    block_size = 4
    format_id = {"int8": 1, "f16": 2, "f32": 3}[kv_format]
    payload_bytes = bytearray(struct.pack(
        "<8s10I",
        b"QXKVSNP1", 2, 2, 2, 4, 1, 2, format_id, block_size, 99, 7,
    ))
    entries = []
    for layer in range(2):
        for position in range(2):
            record_offset = len(payload_bytes)
            scale = 1.0 if kv_format != "int8" else 0.125
            for kind in ("k", "v"):
                block = bytes((layer, position, ord(kind), 255))
                offset = len(payload_bytes)
                payload_bytes.extend(block)
                entries.append(
                    {
                        "layer": layer,
                        "position": position,
                        "kind": kind,
                        "offset": offset,
                        "bytes": block_size,
                        "sha256": sha256_bytes(block),
                        "scale": scale,
                    }
                )
            assert len(payload_bytes) == record_offset + 2 * block_size
            payload_bytes.extend(struct.pack("<ff", scale, scale))
    payload_bytes.extend(hashlib.sha256(payload_bytes).digest())
    payload_bytes = bytes(payload_bytes)
    payload = tmp_path / "kv.bin"
    payload.write_bytes(payload_bytes)

    manifest = {
        "schema": "qx-kv-snapshot-v1",
        "model": {
            "size": model.stat().st_size,
            "sha256": sha256_bytes(model.read_bytes()),
        },
        "producer": {
            "binary_sha256": sha256_bytes(binary.read_bytes()),
            "revision": "0" * 40,
        },
        "geometry": {
            "kv_format": kv_format,
            "activation_format": "q8_k_compat",
            "layers": 2,
            "positions": 2,
            "ctx_tokens": 4,
            "kv_heads": 1,
            "head_dim": 2,
            "bytes_per_vector": block_size,
            "next_token": 99,
            "seed": 7,
            "prompt_tokens": [42, 9707],
        },
        "payload": {
            "path": "kv.bin",
            "bytes": len(payload_bytes),
            "sha256": sha256_bytes(payload_bytes),
        },
        "entries": entries,
    }
    manifest_path = tmp_path / "snapshot.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, model, manifest


def expected_geometry(kv_format: str = "f16") -> dict[str, object]:
    return {
        "producer_binary_sha256": sha256_bytes(b"synthetic-qxqxf-binary"),
        "kv_format": kv_format,
        "activation_format": "q8_k_compat",
        "layers": 2,
        "positions": 2,
        "ctx_tokens": 4,
        "kv_heads": 1,
        "head_dim": 2,
        "bytes_per_vector": 4,
        "next_token": 99,
        "seed": 7,
        "prompt_tokens": [42, 9707],
    }


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, allow_nan=True), encoding="utf-8")


def test_valid_snapshot_loads_exact_accumulated_matrix(tmp_path):
    manifest_path, model, _ = write_fixture(tmp_path)

    snapshot = KV_SNAPSHOT.load_snapshot(
        manifest_path,
        model_path=model,
        expected_geometry=expected_geometry(),
    )

    assert snapshot.manifest["schema"] == "qx-kv-snapshot-v1"
    assert snapshot.payload == (tmp_path / "kv.bin").read_bytes()
    assert len(snapshot.manifest["entries"]) == 2 * 2 * 2


def test_builder_derives_manifest_from_native_runtime_snapshot(tmp_path):
    manifest_path, model, expected_manifest = write_fixture(tmp_path)
    binary = tmp_path / "qxqxf.exe"

    built = KV_SNAPSHOT.build_snapshot_manifest(
        tmp_path / "kv.bin",
        model_path=model,
        producer_binary_path=binary,
        revision="0" * 40,
        activation_format="q8_k_compat",
        prompt_tokens=[42, 9707],
    )
    manifest_path.write_text(json.dumps(built), encoding="utf-8")
    loaded = KV_SNAPSHOT.load_snapshot(
        manifest_path,
        model_path=model,
        expected_geometry=expected_geometry(),
    )

    assert built == expected_manifest
    assert loaded.payload[:8] == b"QXKVSNP1"


def test_native_runtime_snapshot_rejects_same_length_payload_mutation(tmp_path):
    write_fixture(tmp_path)
    payload = bytearray((tmp_path / "kv.bin").read_bytes())
    payload[48] ^= 0x01

    with pytest.raises(ValueError, match="embedded SHA-256 mismatch"):
        KV_SNAPSHOT._parse_runtime_payload(bytes(payload))


@pytest.mark.parametrize("replacement", [True, 2.0])
def test_integer_fields_reject_bool_and_integral_float(tmp_path, replacement):
    manifest_path, model, manifest = write_fixture(tmp_path)
    manifest["geometry"]["positions"] = replacement
    write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="positions.*integer"):
        KV_SNAPSHOT.load_snapshot(
            manifest_path,
            model_path=model,
            expected_geometry=expected_geometry(),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_scale_fails_closed(tmp_path, value):
    manifest_path, model, manifest = write_fixture(tmp_path, kv_format="int8")
    manifest["entries"][0]["scale"] = value
    write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="non-finite"):
        KV_SNAPSHOT.load_snapshot(
            manifest_path,
            model_path=model,
            expected_geometry=expected_geometry("int8"),
        )


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_entry_matrix_must_be_complete_and_unique(tmp_path, mutation):
    manifest_path, model, manifest = write_fixture(tmp_path)
    if mutation == "missing":
        manifest["entries"].pop()
    else:
        manifest["entries"].append(copy.deepcopy(manifest["entries"][0]))
    write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="entry coverage"):
        KV_SNAPSHOT.load_snapshot(
            manifest_path,
            model_path=model,
            expected_geometry=expected_geometry(),
        )


def test_payload_length_and_hash_are_verified(tmp_path):
    manifest_path, model, _ = write_fixture(tmp_path)
    (tmp_path / "kv.bin").write_bytes(b"truncated")

    with pytest.raises(ValueError, match="payload (length|hash)"):
        KV_SNAPSHOT.load_snapshot(
            manifest_path,
            model_path=model,
            expected_geometry=expected_geometry(),
        )


def test_per_entry_hash_is_verified(tmp_path):
    manifest_path, model, manifest = write_fixture(tmp_path)
    manifest["entries"][3]["sha256"] = "f" * 64
    write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="entry hash"):
        KV_SNAPSHOT.load_snapshot(
            manifest_path,
            model_path=model,
            expected_geometry=expected_geometry(),
        )


def test_model_provenance_mismatch_fails_closed(tmp_path):
    manifest_path, model, _ = write_fixture(tmp_path)
    model.write_bytes(b"different-model")

    with pytest.raises(ValueError, match="model (size|hash)"):
        KV_SNAPSHOT.load_snapshot(
            manifest_path,
            model_path=model,
            expected_geometry=expected_geometry(),
        )


def test_producer_binary_provenance_mismatch_fails_closed(tmp_path):
    manifest_path, model, manifest = write_fixture(tmp_path)
    manifest["producer"]["binary_sha256"] = "f" * 64
    write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="producer binary hash"):
        KV_SNAPSHOT.load_snapshot(
            manifest_path,
            model_path=model,
            expected_geometry=expected_geometry(),
        )


def test_runtime_geometry_mismatch_fails_closed(tmp_path):
    manifest_path, model, _ = write_fixture(tmp_path)
    incompatible = expected_geometry()
    incompatible["prompt_tokens"] = [42, 0]

    with pytest.raises(ValueError, match="geometry mismatch.*prompt_tokens"):
        KV_SNAPSHOT.load_snapshot(
            manifest_path,
            model_path=model,
            expected_geometry=incompatible,
        )


def test_payload_path_cannot_escape_snapshot_directory(tmp_path):
    manifest_path, model, manifest = write_fixture(tmp_path)
    outside = tmp_path.parent / "outside-kv.bin"
    outside.write_bytes((tmp_path / "kv.bin").read_bytes())
    manifest["payload"]["path"] = "../outside-kv.bin"
    write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="payload path"):
        KV_SNAPSHOT.load_snapshot(
            manifest_path,
            model_path=model,
            expected_geometry=expected_geometry(),
        )


@pytest.mark.parametrize("root", [[], None, "snapshot"])
def test_manifest_root_must_be_an_object(tmp_path, root):
    manifest_path, model, _ = write_fixture(tmp_path)
    manifest_path.write_text(json.dumps(root), encoding="utf-8")

    with pytest.raises(ValueError, match="root.*object"):
        KV_SNAPSHOT.load_snapshot(
            manifest_path,
            model_path=model,
            expected_geometry=expected_geometry(),
        )
