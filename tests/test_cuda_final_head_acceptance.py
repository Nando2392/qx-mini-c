from __future__ import annotations

import json
import math
import os
import struct
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "include" / "qx_cuda_final_head.h"
CPU_BUILD = ROOT / "build_msvc.bat"
REQUIRE_GPU_ENV = "QX_REQUIRE_CUDA_FINAL_HEAD"
DRIVER_ENV = "QX_CUDA_FINAL_HEAD_DRIVER"
BLOCK_ELEMENTS = 256
BLOCK_BYTES = 210
HIDDEN = 512
VOCAB = 513
CALLS = 2


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _decode_q6_k_block(block: bytes) -> tuple[float, ...]:
    """Independent GGML Q6_K byte-layout oracle; no QX code is imported."""
    assert len(block) == BLOCK_BYTES
    d = struct.unpack_from("<e", block, 208)[0]
    scales = struct.unpack_from("<16b", block, 192)
    decoded: list[float] = []
    for index in range(BLOCK_ELEMENTS):
        half = index // 128
        local = index % 128
        lane = local % 32
        section = local // 32
        ql_offset = half * 64 + lane + (32 if section in (1, 3) else 0)
        packed_low = block[ql_offset]
        low_four = packed_low & 0x0F if section < 2 else packed_low >> 4
        high_two = (block[128 + half * 32 + lane] >> (2 * section)) & 0x03
        quant = low_four | (high_two << 4)
        scale = scales[half * 8 + (lane // 16) + section * 2]
        decoded.append(float(d) * scale * (quant - 32))
    return tuple(decoded)


def _fixed_golden_block() -> bytes:
    block = bytearray(BLOCK_BYTES)
    block[:128] = bytes([0xA5]) * 128
    block[128:192] = bytes([0xE4]) * 64
    block[192:208] = struct.pack("<16b", *range(-8, 0), *range(1, 9))
    struct.pack_into("<e", block, 208, 0.5)
    return bytes(block)


def _synthetic_case() -> tuple[bytes, tuple[float, ...], tuple[float, ...]]:
    """Produce deterministic packed rows and an independently decoded F32 result."""
    row_bytes = (HIDDEN // BLOCK_ELEMENTS) * BLOCK_BYTES
    weights = bytearray(VOCAB * row_bytes)
    state = 0x85C0FFEE
    for row in range(VOCAB):
        for block_index in range(HIDDEN // BLOCK_ELEMENTS):
            offset = row * row_bytes + block_index * BLOCK_BYTES
            for byte_index in range(192):
                state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
                weights[offset + byte_index] = state >> 24
            scales = tuple(((row * 5 + block_index * 3 + i * 7) % 31) - 15 for i in range(16))
            weights[offset + 192 : offset + 208] = struct.pack("<16b", *scales)
            struct.pack_into("<e", weights, offset + 208, (0.5, -0.25, 1.0, 2.0)[(row + block_index) % 4])

    activation = tuple(_f32(math.sin(i * 0.03125) * 0.125 + ((i % 11) - 5) * 0.0005) for i in range(HIDDEN))
    expected: list[float] = []
    for row in range(VOCAB):
        row_offset = row * row_bytes
        total = 0.0
        for block_index in range(HIDDEN // BLOCK_ELEMENTS):
            offset = row_offset + block_index * BLOCK_BYTES
            decoded = _decode_q6_k_block(bytes(weights[offset : offset + BLOCK_BYTES]))
            start = block_index * BLOCK_ELEMENTS
            total += sum(float(weight) * float(value) for weight, value in zip(decoded, activation[start : start + BLOCK_ELEMENTS]))
        expected.append(_f32(total))
    return bytes(weights), activation, tuple(expected)


def _metrics(actual: tuple[float, ...], expected: tuple[float, ...]) -> dict[str, float]:
    assert len(actual) == len(expected) and actual
    deltas = [got - want for got, want in zip(actual, expected)]
    dot = sum(got * want for got, want in zip(actual, expected))
    actual_l2 = sum(value * value for value in actual)
    expected_l2 = sum(value * value for value in expected)
    assert actual_l2 > 0.0 and expected_l2 > 0.0
    return {
        "max_abs": max(abs(delta) for delta in deltas),
        "rmse": math.sqrt(sum(delta * delta for delta in deltas) / len(deltas)),
        "cosine": dot / math.sqrt(actual_l2 * expected_l2),
    }


def _gpu_driver_or_skip() -> Path:
    required = os.environ.get(REQUIRE_GPU_ENV) == "1"
    if not required:
        pytest.skip(f"GPU execution is opt-in; set {REQUIRE_GPU_ENV}=1 for release acceptance")
    configured = os.environ.get(DRIVER_ENV)
    assert configured, f"{DRIVER_ENV} must name the backend-owned acceptance driver when {REQUIRE_GPU_ENV}=1"
    driver = Path(configured)
    assert driver.is_file(), f"configured CUDA final-head driver does not exist: {driver}"
    return driver


def test_q6_k_reference_decoder_has_fixed_signed_scale_fp16_golden() -> None:
    decoded = _decode_q6_k_block(_fixed_golden_block())
    expected_runs = (
        108.0, 94.5, 33.0, 27.5, -20.0, -15.0, -26.0, -13.0,
        -13.5, -27.0, -16.5, -22.0, 25.0, 30.0, 91.0, 104.0,
    )
    for run, expected in enumerate(expected_runs):
        assert decoded[run * 16 : (run + 1) * 16] == (expected,) * 16


def test_public_header_is_portable_and_exposes_accounting_contract() -> None:
    text = HEADER.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "#include <cuda" not in lowered
    assert "#include <driver" not in lowered
    for field in (
        "resident_weight_bytes", "persistent_allocations", "weight_uploads",
        "weight_upload_bytes", "host_to_device_bytes", "device_to_host_bytes",
        "kernel_launches", "cpu_fallbacks",
    ):
        assert f"uint64_t {field};" in text
    for symbol in (
        "qx_cuda_final_head_is_available", "qx_cuda_final_head_create_q6_k_f32",
        "qx_cuda_final_head_compute_f32", "qx_cuda_final_head_get_counters",
        "qx_cuda_final_head_destroy",
    ):
        assert symbol in text


def test_cpu_default_build_links_only_the_unavailable_stub() -> None:
    text = CPU_BUILD.read_text(encoding="utf-8").lower().replace("/", "\\")
    assert "src\\qx_cuda_final_head_stub.c" in text
    for forbidden in ("qx_cuda_final_head.cu", "cudart", "nvcuda", "qxqxf_cuda"):
        assert forbidden not in text


def test_cuda_backend_matches_independent_q6_k_oracle_and_exact_counters(tmp_path: Path) -> None:
    driver = _gpu_driver_or_skip()
    weights, activation, expected = _synthetic_case()
    weights_path = tmp_path / "weights.q6_k"
    activation_path = tmp_path / "activation.f32"
    logits_path = tmp_path / "logits.f32"
    repeat_path = tmp_path / "repeat-logits.f32"
    weights_path.write_bytes(weights)
    activation_path.write_bytes(struct.pack(f"<{HIDDEN}f", *activation))

    command = [
        str(driver), "--weights", str(weights_path), "--activation", str(activation_path),
        "--hidden", str(HIDDEN), "--vocab", str(VOCAB), "--calls", str(CALLS),
        "--logits", str(logits_path), "--repeat-logits", str(repeat_path), "--json",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)

    assert payload["schema"] == "qx.cuda-final-head-driver.v1"
    assert payload["status"] == "pass"
    assert payload["backend"] == "cuda"
    assert payload["hidden"] == HIDDEN
    assert payload["vocab"] == VOCAB
    assert payload["calls"] == CALLS
    assert isinstance(payload["device"]["name"], str) and payload["device"]["name"]
    assert type(payload["device"]["compute_capability_major"]) is int
    assert type(payload["device"]["compute_capability_minor"]) is int

    raw_logits = logits_path.read_bytes()
    repeat_logits = repeat_path.read_bytes()
    assert len(raw_logits) == VOCAB * 4
    assert repeat_logits == raw_logits
    actual = struct.unpack(f"<{VOCAB}f", raw_logits)
    assert all(math.isfinite(value) for value in actual)
    measured = _metrics(actual, expected)
    assert measured["max_abs"] <= 1e-4
    assert measured["rmse"] <= 2e-5
    assert measured["cosine"] >= 0.999999

    expected_counters = {
        "resident_weight_bytes": len(weights),
        "persistent_allocations": 3,
        "weight_uploads": 1,
        "weight_upload_bytes": len(weights),
        "host_to_device_bytes": CALLS * HIDDEN * 4,
        "device_to_host_bytes": CALLS * VOCAB * 4,
        "kernel_launches": CALLS,
        "cpu_fallbacks": 0,
    }
    assert payload["counters"] == expected_counters
