import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / "qxqxf.exe"
MAKE_GGUF = ROOT / "scripts" / "make_synthetic_gguf.py"

HEADER_SIZE = 272
ENTRY_SIZE = 208
HEADER_DIR_OFFSET = 24
HEADER_DATA_OFFSET = 32
ENTRY_RANK = 104
ENTRY_DIMS = 112
ENTRY_OFFSET = 144
ENTRY_BYTE_SIZE = 152
ENTRY_GROUP_SIZE = 160
ENTRY_FLAGS = 164
MANIFEST_CHECKSUM = 48
MANIFEST_OFFSET = 56
MANIFEST_SIZE = 13 * 4
MANIFEST_LAYERS = 64
MANIFEST_VOCAB = 88
MANIFEST_EXPERTS = 96


@pytest.fixture()
def valid_qxf(tmp_path):
    if not EXE.exists():
        pytest.fail("build/qxqxf.exe is required")
    gguf = tmp_path / "synthetic.gguf"
    qxf = tmp_path / "synthetic.qxf"
    subprocess.run([sys.executable, str(MAKE_GGUF), "--out", str(gguf)], check=True, capture_output=True)
    subprocess.run([
        str(EXE), "create-from-gguf-copy", "--in", str(gguf),
        "--model", "qwen3-30b-a3b", "--quant", "q2", "--out", str(qxf),
    ], check=True, capture_output=True)
    return qxf


def fnv1a64(data):
    value = 1469598103934665603
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def refresh_manifest_checksum(raw):
    checksum = fnv1a64(raw[MANIFEST_OFFSET:MANIFEST_OFFSET + MANIFEST_SIZE])
    struct.pack_into("<Q", raw, MANIFEST_CHECKSUM, checksum)


def mutated(source, tmp_path, name, mutate):
    target = tmp_path / name
    shutil.copyfile(source, target)
    raw = bytearray(target.read_bytes())
    mutate(raw)
    target.write_bytes(raw)
    return target


def run_qxf(path, command, *args):
    return subprocess.run(
        [str(EXE), command, "--in", str(path), *args],
        text=True,
        capture_output=True,
    )


def run_inspect(path):
    return run_qxf(path, "inspect-tensor", "--name", "token_embd.weight")


def directory(raw):
    return struct.unpack_from("<Q", raw, HEADER_DIR_OFFSET)[0]


@pytest.mark.parametrize("case", ["dir_unaligned", "data_unaligned", "data_overlaps_directory"])
def test_qxf_rejects_noncanonical_header_layout(valid_qxf, tmp_path, case):
    def corrupt(raw):
        dir_offset = directory(raw)
        tensor_count = struct.unpack_from("<I", raw, 16)[0]
        if case == "dir_unaligned":
            struct.pack_into("<Q", raw, HEADER_DIR_OFFSET, dir_offset + 1)
        elif case == "data_unaligned":
            data_offset = struct.unpack_from("<Q", raw, HEADER_DATA_OFFSET)[0]
            struct.pack_into("<Q", raw, HEADER_DATA_OFFSET, data_offset + 1)
        else:
            struct.pack_into("<Q", raw, HEADER_DATA_OFFSET, dir_offset + tensor_count * ENTRY_SIZE - 1)

    result = run_inspect(mutated(valid_qxf, tmp_path, f"{case}.qxf", corrupt))
    assert result.returncode != 0
    assert "invalid QXF layout" in result.stderr


@pytest.mark.parametrize("case", ["rank_zero", "zero_active_dim", "inactive_dim", "dimension_product_overflow"])
def test_qxf_rejects_noncanonical_tensor_dimensions(valid_qxf, tmp_path, case):
    def corrupt(raw):
        entry = directory(raw)
        if case == "rank_zero":
            struct.pack_into("<I", raw, entry + ENTRY_RANK, 0)
        elif case == "zero_active_dim":
            struct.pack_into("<Q", raw, entry + ENTRY_DIMS, 0)
        elif case == "inactive_dim":
            struct.pack_into("<Q", raw, entry + ENTRY_DIMS + 2 * 8, 1)
        else:
            struct.pack_into("<Q", raw, entry + ENTRY_DIMS, 0xFFFFFFFFFFFFFFFF)
            struct.pack_into("<Q", raw, entry + ENTRY_DIMS + 8, 2)

    result = run_inspect(mutated(valid_qxf, tmp_path, f"{case}.qxf", corrupt))
    assert result.returncode != 0
    assert "invalid tensor dimensions" in result.stderr


def test_qxf_rejects_duplicate_tensor_names(valid_qxf, tmp_path):
    def corrupt(raw):
        entry = directory(raw)
        raw[entry + ENTRY_SIZE:entry + ENTRY_SIZE + 96] = raw[entry:entry + 96]

    result = run_inspect(mutated(valid_qxf, tmp_path, "duplicate-name.qxf", corrupt))
    assert result.returncode != 0
    assert "duplicate tensor name" in result.stderr


@pytest.mark.parametrize("case", ["before_data", "unaligned", "overlap"])
def test_qxf_rejects_invalid_tensor_placement(valid_qxf, tmp_path, case):
    def corrupt(raw):
        entry = directory(raw)
        data_offset = struct.unpack_from("<Q", raw, HEADER_DATA_OFFSET)[0]
        first_offset = struct.unpack_from("<Q", raw, entry + ENTRY_OFFSET)[0]
        if case == "before_data":
            struct.pack_into("<Q", raw, entry + ENTRY_OFFSET, data_offset - 4096)
        elif case == "unaligned":
            struct.pack_into("<Q", raw, entry + ENTRY_OFFSET, first_offset + 1)
        else:
            struct.pack_into("<Q", raw, entry + ENTRY_SIZE + ENTRY_OFFSET, first_offset)

    result = run_inspect(mutated(valid_qxf, tmp_path, f"{case}.qxf", corrupt))
    assert result.returncode != 0
    expected = "overlapping tensor ranges" if case == "overlap" else "invalid tensor placement"
    assert expected in result.stderr


def test_qxf_rejects_zero_size_data_tensor(valid_qxf, tmp_path):
    def corrupt(raw):
        struct.pack_into("<Q", raw, directory(raw) + ENTRY_BYTE_SIZE, 0)

    result = run_inspect(mutated(valid_qxf, tmp_path, "zero-size.qxf", corrupt))
    assert result.returncode != 0
    assert "invalid tensor placement" in result.stderr


@pytest.mark.parametrize("case", ["byte_size_mismatch", "quant_row_not_block_divisible"])
def test_qxf_rejects_tensor_size_inconsistent_with_dimensions(valid_qxf, tmp_path, case):
    def corrupt(raw):
        entry = directory(raw)
        if case == "byte_size_mismatch":
            byte_size = struct.unpack_from("<Q", raw, entry + ENTRY_BYTE_SIZE)[0]
            struct.pack_into("<Q", raw, entry + ENTRY_BYTE_SIZE, byte_size - 1)
        else:
            struct.pack_into("<Q", raw, entry + ENTRY_DIMS, 255)

    result = run_inspect(mutated(valid_qxf, tmp_path, f"{case}.qxf", corrupt))
    assert result.returncode != 0
    assert "tensor byte size inconsistent with dimensions" in result.stderr


@pytest.mark.parametrize(("offset", "value"), [(96, 99), (100, 99), (ENTRY_GROUP_SIZE, 0), (ENTRY_FLAGS, 99)])
def test_qxf_rejects_invalid_tensor_metadata(valid_qxf, tmp_path, offset, value):
    def corrupt(raw):
        struct.pack_into("<I", raw, directory(raw) + offset, value)

    result = run_inspect(mutated(valid_qxf, tmp_path, f"metadata-{offset}.qxf", corrupt))
    assert result.returncode != 0
    assert "invalid tensor metadata" in result.stderr


def test_qxf_rejects_trailing_undeclared_bytes(valid_qxf, tmp_path):
    out = tmp_path / "trailing.qxf"
    out.write_bytes(valid_qxf.read_bytes() + b"undeclared")
    result = run_inspect(out)
    assert result.returncode != 0
    assert "declared QXF file size does not match file" in result.stderr


def test_qxf_accepts_nonoverlapping_directory_entries_in_logical_order(valid_qxf, tmp_path):
    def reorder(raw):
        entry = directory(raw)
        first = bytes(raw[entry:entry + ENTRY_SIZE])
        second = bytes(raw[entry + ENTRY_SIZE:entry + 2 * ENTRY_SIZE])
        raw[entry:entry + ENTRY_SIZE] = second
        raw[entry + ENTRY_SIZE:entry + 2 * ENTRY_SIZE] = first

    out = mutated(valid_qxf, tmp_path, "logical-order.qxf", reorder)
    result = run_inspect(out)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(("field", "offset"), [("layers", MANIFEST_LAYERS), ("experts", MANIFEST_EXPERTS)])
def test_qxf_rejects_zero_manifest_divisors(valid_qxf, tmp_path, field, offset):
    def corrupt(raw):
        struct.pack_into("<I", raw, offset, 0)
        refresh_manifest_checksum(raw)

    result = run_qxf(mutated(valid_qxf, tmp_path, f"zero-{field}.qxf", corrupt), "bench-expert-load", "--iters", "1")
    assert result.returncode != 0
    assert "invalid QXF manifest" in result.stderr


@pytest.mark.parametrize(
    ("command", "args"),
    [
        ("token-embedding", ("--token-id", "42")),
        ("forward-schedule", ("--token-id", "42", "--top-k", "8")),
        ("logits-probe", ("--activation", "1", "--top-n", "2", "--scan", "4", "--seed", "7")),
        ("residual-vector-probe", ("--token-id", "42", "--dims", "64", "--seed", "7")),
        ("token-forward-probe", ("--token-id", "42", "--layers", "1", "--top-k", "1", "--blocks", "1", "--seed", "7")),
    ],
)
def test_legacy_embedding_paths_reject_nondivisible_rows(valid_qxf, command, args):
    result = run_qxf(valid_qxf, command, *args)
    assert result.returncode != 0
    assert "tensor byte size is not divisible by row count" in result.stderr


def test_matvec_stub_rejects_truncated_requested_span(valid_qxf):
    result = run_qxf(valid_qxf, "matvec-stub", "--name", "token_embd.weight", "--rows", "100")
    assert result.returncode != 0
    assert "requested matvec span outside tensor" in result.stderr


def test_logits_probe_is_explicitly_synthetic(valid_qxf, tmp_path):
    def make_small_valid_head(raw):
        entry = directory(raw)
        raw[entry:entry + 96] = b"fixture.partial\0" + b"\0" * (96 - len("fixture.partial") - 1)
        second = entry + ENTRY_SIZE
        raw[second:second + 96] = b"token_embd.weight\0" + b"\0" * (96 - len("token_embd.weight") - 1)
        struct.pack_into("<I", raw, MANIFEST_VOCAB, 256)
        refresh_manifest_checksum(raw)

    fixture = mutated(valid_qxf, tmp_path, "valid-small-head.qxf", make_small_valid_head)
    result = run_qxf(fixture, "logits-probe", "--activation", "1", "--top-n", "2", "--scan", "4", "--seed", "7")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["synthetic"] is True
