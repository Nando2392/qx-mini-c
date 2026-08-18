import json
import math
import os
import struct
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / "qxqxf.exe"
GGML_REFERENCE = ROOT / "build" / "ggml_reference_decode.exe"
LLAMA_CPP_DIR = Path(os.environ.get("LLAMA_CPP_DIR", ROOT.parent / "llama.cpp-k3"))
GGML_REFERENCE_DLLS = LLAMA_CPP_DIR / "build-k3-cpu" / "bin" / "Release"
MODEL = ROOT / "models" / "Qwen3-30B-A3B-UD-IQ2_M.qxf"
IQ4_VALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fnv1a64(data):
    value = 1469598103934665603
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def qxf_with_manifest_u32(source, destination, manifest_field_offset, value):
    with source.open("rb") as handle:
        fixed = handle.read(56)
        data_offset = struct.unpack_from("<Q", fixed, 32)[0]
        file_size = struct.unpack_from("<Q", fixed, 40)[0]
        handle.seek(0)
        prefix = bytearray(handle.read(data_offset))
    struct.pack_into("<I", prefix, 56 + manifest_field_offset, value)
    struct.pack_into("<Q", prefix, 48, fnv1a64(prefix[56:108]))
    with destination.open("wb") as handle:
        handle.write(prefix)
        handle.seek(file_size - 1)
        handle.write(b"\0")


def scale_min_k4(index, scales):
    if index < 4:
        return scales[index] & 63, scales[index + 4] & 63
    return (scales[index + 4] & 15) | ((scales[index - 4] >> 6) << 4), (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4)


def decode_q4_k(block):
    d, dmin = struct.unpack_from("<ee", block)
    scales, quants = block[4:16], block[16:144]
    result, quant_offset, scale_index = [], 0, 0
    for _ in range(4):
        scale_a, min_a = scale_min_k4(scale_index, scales)
        scale_b, min_b = scale_min_k4(scale_index + 1, scales)
        chunk = quants[quant_offset : quant_offset + 32]
        result.extend(f32(d * scale_a * (value & 15) - dmin * min_a) for value in chunk)
        result.extend(f32(d * scale_b * (value >> 4) - dmin * min_b) for value in chunk)
        quant_offset += 32
        scale_index += 2
    return result


def decode_iq4_xs(block):
    d = struct.unpack_from("<e", block)[0]
    scales_high = struct.unpack_from("<H", block, 2)[0]
    scales_low, quants = block[4:8], block[8:136]
    result = []
    for group in range(8):
        local_scale = ((scales_low[group // 2] >> (4 * (group % 2))) & 15) | (((scales_high >> (2 * group)) & 3) << 4)
        multiplier = f32(d * (local_scale - 32))
        chunk = quants[group * 16 : group * 16 + 16]
        result.extend(f32(multiplier * IQ4_VALUES[value & 15]) for value in chunk)
        result.extend(f32(multiplier * IQ4_VALUES[value >> 4]) for value in chunk)
    return result


def metadata(name):
    return json.loads(subprocess.check_output([str(EXE), "inspect-tensor", "--in", str(MODEL), "--name", name], text=True))


def read_at(offset, size):
    with MODEL.open("rb") as handle:
        handle.seek(offset)
        return handle.read(size)


def residual_for_token(token, norm_name):
    embedding, norm = metadata("token_embd.weight"), metadata(norm_name)
    row_bytes = 8 * 144
    row = read_at(embedding["offset"] + token * row_bytes, row_bytes)
    decoded = []
    for block in range(8):
        decoded.extend(decode_q4_k(row[block * 144 : (block + 1) * 144]))
    rms = math.sqrt(sum(value * value for value in decoded) / 2048 + 1e-6)
    norm_values = struct.unpack("<2048f", read_at(norm["offset"], 2048 * 4))
    return [f32(f32(value / rms) * norm_values[index]) for index, value in enumerate(decoded)]


def embedding_for_token(token):
    embedding = metadata("token_embd.weight")
    row_bytes = 8 * 144
    row = read_at(embedding["offset"] + token * row_bytes, row_bytes)
    decoded = []
    for block in range(8):
        decoded.extend(decode_q4_k(row[block * 144 : (block + 1) * 144]))
    return decoded


def iq4_row_dot(tensor, row, residual):
    blocks_per_row = tensor["dims"][0] // 256
    row_bytes = blocks_per_row * 136
    raw = read_at(tensor["offset"] + row * row_bytes, row_bytes)
    weights = []
    for block in range(blocks_per_row):
        weights.extend(decode_iq4_xs(raw[block * 136 : (block + 1) * 136]))
    return f32(sum(float(weight) * float(value) for weight, value in zip(weights, residual)))


def ggml_reference_row(tensor, expert, row, quant_name, block_size):
    blocks = tensor["dims"][0] // 256
    expert_bytes = tensor["byte_size"] // tensor["dims"][2]
    offset = tensor["offset"] + expert * expert_bytes + row * blocks * block_size
    env = os.environ.copy()
    env["PATH"] = str(GGML_REFERENCE_DLLS) + os.pathsep + env.get("PATH", "")
    raw = subprocess.check_output([str(GGML_REFERENCE), quant_name, str(MODEL), str(offset), str(blocks)], env=env)
    return struct.unpack(f"<{blocks * 256}f", raw)


def ggml_reference_full_q6_logits(tensor, activation, activation_path):
    activation_path.write_bytes(struct.pack(f"<{len(activation)}f", *activation))
    env = os.environ.copy()
    env["PATH"] = str(GGML_REFERENCE_DLLS) + os.pathsep + env.get("PATH", "")
    raw = subprocess.check_output(
        [str(GGML_REFERENCE), "q6_k_logits", str(MODEL), str(tensor["offset"]), str(tensor["dims"][1]), str(activation_path)],
        env=env,
    )
    return struct.unpack(f"<{tensor['dims'][1]}f", raw)


def test_real_iq4_xs_full_rows_match_independent_python_decoder():
    if not EXE.exists() or not MODEL.exists():
        pytest.skip("real Qwen QXF is not available")
    payload = json.loads(subprocess.check_output([str(EXE), "real-qkv-golden-probe", "--in", str(MODEL), "--layer", "0", "--token-a", "42", "--token-b", "43", "--q-heads-run", "32", "--seed", "7"], text=True))
    assert payload["iq4_xs_decoder_gate"] == "external_python_full_row"
    residual_a = residual_for_token(42, "blk.0.attn_norm.weight")
    residual_b = residual_for_token(43, "blk.0.attn_norm.weight")
    q_tensor = metadata("blk.0.attn_q.weight")
    k_tensor = metadata("blk.0.attn_k.weight")
    v_tensor = metadata("blk.0.attn_v.weight")
    checks = [
        (payload["q_raw"][0], iq4_row_dot(q_tensor, 0, residual_b)),
        (payload["q_raw"][1024], iq4_row_dot(q_tensor, 1024, residual_b)),
        (payload["q_raw"][4095], iq4_row_dot(q_tensor, 4095, residual_b)),
        (payload["k_raw"][0][0], iq4_row_dot(k_tensor, 0, residual_a)),
        (payload["k_raw"][0][128], iq4_row_dot(k_tensor, 128, residual_a)),
        (payload["k_raw"][1][511], iq4_row_dot(k_tensor, 511, residual_b)),
        (payload["v_raw"][0][0], iq4_row_dot(v_tensor, 0, residual_a)),
        (payload["v_raw"][1][511], iq4_row_dot(v_tensor, 511, residual_b)),
    ]
    for got, expected in checks:
        assert math.isclose(got, expected, rel_tol=4e-6, abs_tol=4e-6)
    assert payload["q_heads_run"] == 32
    assert payload["kv_heads_total"] == 4
    assert payload["full_head_coverage"] is True


def test_real_attention_output_4096_to_2048_matches_independent_decoder():
    if not EXE.exists() or not MODEL.exists():
        pytest.skip("real Qwen QXF is not available")
    payload = json.loads(subprocess.check_output([str(EXE), "real-qkv-golden-probe", "--in", str(MODEL), "--layer", "0", "--token-a", "42", "--token-b", "43", "--q-heads-run", "32", "--seed", "7"], text=True))
    assert payload["output_projection"] == "real_iq4_xs_4096_to_2048"
    assert len(payload["attention_context_raw"]) == 4096
    assert len(payload["output_raw"]) == 2048
    output_tensor = metadata("blk.0.attn_output.weight")
    assert output_tensor["dims"] == [4096, 2048]
    for row in (0, 1024, 2047):
        expected = iq4_row_dot(output_tensor, row, payload["attention_context_raw"])
        assert math.isclose(payload["output_raw"][row], expected, rel_tol=5e-6, abs_tol=5e-6)


def test_real_state_loop_uses_all_heads_and_full_output_projection():
    tokens = Path.home() / "AppData" / "Local" / "Temp" / "qwen3-a3b.tokens.tsv"
    if not EXE.exists() or not MODEL.exists() or not tokens.exists():
        pytest.skip("real Qwen runtime fixtures are not available")
    payload = json.loads(subprocess.check_output([str(EXE), "state-loop-probe", "--in", str(MODEL), "--tokens", str(tokens), "--prompt-token", "42", "--steps", "1", "--layers", "1", "--ctx", "4", "--kv", "int8", "--top-k", "3", "--scan", "64", "--temperature", "0", "--seed", "7", "--rope-gqa-attention", "--residual-dims", "2048", "--norm", "blk.0.attn_norm.weight"], text=True))
    layer = payload["tokens"][0]["layers"][0]
    assert {layer["q_ggml_type"], layer["k_ggml_type"], layer["v_ggml_type"], layer["output_ggml_type"]} == {23}
    assert layer["q_heads_run"] == 32
    assert layer["kv_heads_touched"] == 4
    assert layer["output_projection_input_dims"] == 4096
    assert layer["output_projection_output_dims"] == 2048
    assert layer["attention_output_vector_values"] == 2048
    assert layer["attention_output_vector_checksum"] > 0


def test_post_attention_residual_and_ffn_rmsnorm_match_python():
    if not EXE.exists() or not MODEL.exists():
        pytest.skip("real Qwen QXF is not available")
    payload = json.loads(subprocess.check_output([str(EXE), "real-qkv-golden-probe", "--in", str(MODEL), "--layer", "0", "--token-a", "42", "--token-b", "43", "--q-heads-run", "32", "--seed", "7"], text=True))
    assert payload["post_attention_norm_tensor"] == "blk.0.ffn_norm.weight"
    assert len(payload["residual_after_attention"]) == 2048
    assert len(payload["ffn_norm_raw"]) == 2048
    embedding = embedding_for_token(43)
    residual = [f32(base + delta) for base, delta in zip(embedding, payload["output_raw"])]
    norm_meta = metadata("blk.0.ffn_norm.weight")
    norm_weights = struct.unpack("<2048f", read_at(norm_meta["offset"], 2048 * 4))
    rms = math.sqrt(sum(value * value for value in residual) / 2048 + 1e-6)
    normalized = [f32(f32(value / rms) * norm_weights[index]) for index, value in enumerate(residual)]
    for row in (0, 1024, 2047):
        assert math.isclose(payload["residual_after_attention"][row], residual[row], rel_tol=2e-6, abs_tol=2e-6)
        assert math.isclose(payload["ffn_norm_raw"][row], normalized[row], rel_tol=3e-6, abs_tol=3e-6)


def test_real_router_logits_softmax_and_top8_match_python():
    if not EXE.exists() or not MODEL.exists():
        pytest.skip("real Qwen QXF is not available")
    payload = json.loads(subprocess.check_output([str(EXE), "real-qkv-golden-probe", "--in", str(MODEL), "--layer", "0", "--token-a", "42", "--token-b", "43", "--q-heads-run", "32", "--seed", "7"], text=True))
    assert payload["router_tensor"] == "blk.0.ffn_gate_inp.weight"
    assert payload["router_norm_topk_prob"] is True
    router = metadata("blk.0.ffn_gate_inp.weight")
    raw = read_at(router["offset"], router["byte_size"])
    hidden = payload["ffn_norm_raw"]
    logits = []
    for expert in range(128):
        weights = struct.unpack_from("<2048f", raw, expert * 2048 * 4)
        logits.append(sum(float(weight) * float(value) for weight, value in zip(weights, hidden)))
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    denominator = sum(exponentials)
    probabilities = [value / denominator for value in exponentials]
    selected = sorted(range(128), key=lambda expert: probabilities[expert], reverse=True)[:8]
    assert payload["selected_experts"] == selected
    for expert in (0, 63, 127):
        assert math.isclose(payload["router_logits"][expert], logits[expert], rel_tol=3e-6, abs_tol=3e-6)
        assert math.isclose(payload["router_probs"][expert], probabilities[expert], rel_tol=3e-6, abs_tol=3e-6)
    selected_sum = sum(probabilities[expert] for expert in selected)
    for got, expert in zip(payload["routing_weights"], selected):
        assert math.isclose(got, probabilities[expert] / selected_sum, rel_tol=3e-6, abs_tol=3e-6)
    assert math.isclose(sum(payload["routing_weights"]), 1.0, rel_tol=1e-12, abs_tol=1e-12)


def test_real_top8_swiglu_moe_produces_full_layer_residual():
    if not EXE.exists() or not MODEL.exists():
        pytest.skip("real Qwen QXF is not available")
    payload = json.loads(subprocess.check_output([str(EXE), "real-qkv-golden-probe", "--in", str(MODEL), "--layer", "0", "--token-a", "42", "--token-b", "43", "--q-heads-run", "32", "--seed", "7", "--full-moe"], text=True))
    assert payload["moe_mode"] == "real_top8_swiglu"
    assert payload["moe_intermediate"] == 768
    assert payload["experts_run"] == 8
    assert payload["gate_ggml_type"] == 17
    assert payload["up_ggml_type"] == 17
    assert payload["down_ggml_type"] == 18
    assert len(payload["moe_output_raw"]) == 2048
    assert len(payload["layer_output_raw"]) == 2048
    assert all(math.isfinite(value) for value in payload["moe_output_raw"])
    assert payload["moe_output_l2"] > 0
    for row in (0, 1024, 2047):
        expected = f32(payload["residual_after_attention"][row] + payload["moe_output_raw"][row])
        assert math.isclose(payload["layer_output_raw"][row], expected, rel_tol=2e-6, abs_tol=2e-6)


def test_expert_decoders_match_external_llama_cpp_reference():
    if not EXE.exists() or not MODEL.exists() or not GGML_REFERENCE.exists():
        pytest.skip("external llama.cpp reference decoder is not available")
    payload = json.loads(subprocess.check_output([str(EXE), "real-qkv-golden-probe", "--in", str(MODEL), "--layer", "0", "--token-a", "42", "--token-b", "43", "--q-heads-run", "32", "--seed", "7", "--full-moe"], text=True))
    expert = payload["selected_experts"][0]
    gate_tensor = metadata("blk.0.ffn_gate_exps.weight")
    up_tensor = metadata("blk.0.ffn_up_exps.weight")
    down_tensor = metadata("blk.0.ffn_down_exps.weight")
    for row in (0, 384, 767):
        gate_weights = ggml_reference_row(gate_tensor, expert, row, "iq2_xs", 74)
        up_weights = ggml_reference_row(up_tensor, expert, row, "iq2_xs", 74)
        gate_expected = f32(sum(float(weight) * float(value) for weight, value in zip(gate_weights, payload["ffn_norm_raw"])))
        up_expected = f32(sum(float(weight) * float(value) for weight, value in zip(up_weights, payload["ffn_norm_raw"])))
        assert math.isclose(payload["expert0_gate_raw"][row], gate_expected, rel_tol=5e-6, abs_tol=5e-6)
        assert math.isclose(payload["expert0_up_raw"][row], up_expected, rel_tol=5e-6, abs_tol=5e-6)
    for row in (0, 1024, 2047):
        down_weights = ggml_reference_row(down_tensor, expert, row, "iq3_xxs", 98)
        down_expected = f32(sum(float(weight) * float(value) for weight, value in zip(down_weights, payload["expert0_hidden_raw"])))
        assert math.isclose(payload["expert0_down_raw"][row], down_expected, rel_tol=5e-6, abs_tol=5e-6)


def test_state_loop_propagates_real_attention_and_moe_across_two_layers():
    tokens = Path.home() / "AppData" / "Local" / "Temp" / "qwen3-a3b.tokens.tsv"
    if not EXE.exists() or not MODEL.exists() or not tokens.exists():
        pytest.skip("real Qwen runtime fixtures are not available")
    payload = json.loads(subprocess.check_output([str(EXE), "state-loop-probe", "--in", str(MODEL), "--tokens", str(tokens), "--prompt-token", "42", "--steps", "1", "--layers", "2", "--ctx", "4", "--kv", "int8", "--top-k", "3", "--scan", "64", "--temperature", "0", "--seed", "7", "--full-moe"], text=True))
    assert payload["residual_source"] == "real_attention_moe_carry"
    assert len(payload["tokens"]) == 1
    layer0, layer1 = payload["tokens"][0]["layers"]
    assert layer0["full_moe"] is True
    assert layer1["full_moe"] is True
    assert layer0["qk_head_norm"] is True
    assert layer1["qk_head_norm"] is True
    assert layer0["attention_context_tokens"] == 1
    assert layer1["attention_context_tokens"] == 1
    assert layer0["experts_run"] == 8
    assert layer1["experts_run"] == 8
    assert len(layer0["selected_experts"]) == 8
    assert len(layer1["selected_experts"]) == 8
    assert math.isclose(sum(layer0["routing_weights"]), 1.0, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(sum(layer1["routing_weights"]), 1.0, rel_tol=1e-12, abs_tol=1e-12)
    assert layer0["residual_output_checksum"] == layer1["residual_input_checksum"]
    assert layer0["residual_output_checksum"] != layer0["residual_input_checksum"]
    assert layer1["residual_output_checksum"] != layer1["residual_input_checksum"]
    assert layer0["gate_ggml_type"] == 17
    assert layer0["down_ggml_type"] == 18
    assert layer1["gate_ggml_type"] == 22
    assert layer1["down_ggml_type"] == 23
    assert layer0["moe_output_l2"] > 0
    assert layer1["moe_output_l2"] > 0
    invalid = subprocess.run(
        [str(EXE), "state-loop-probe", "--in", str(MODEL), "--prompt-token", "42", "--steps", "1", "--layers", "2", "--ctx", "4", "--kv", "int8", "--seed", "7", "--full-moe", "--norm", "blk.0.attn_norm.weight"],
        text=True,
        capture_output=True,
    )
    assert invalid.returncode != 0
    assert "--norm cannot be combined with --full-moe" in invalid.stderr


def test_state_loop_can_dump_lossless_layer_residuals(tmp_path):
    if not EXE.exists() or not MODEL.exists():
        pytest.skip("real Qwen runtime fixtures are not available")
    dump_dir = tmp_path / "residuals"
    dump_dir.mkdir()
    payload = json.loads(
        subprocess.check_output(
            [
                str(EXE), "state-loop-probe", "--in", str(MODEL),
                "--prompt-token", "42", "--steps", "1", "--layers", "2",
                "--ctx", "4", "--kv", "int8", "--temperature", "0",
                "--seed", "7", "--full-moe", "--dump-residuals", str(dump_dir),
            ],
            text=True,
        )
    )
    assert payload["residual_dump"] is True
    assert payload["residual_dump_count"] == 12
    phase_counts = {
        "input": 2048,
        "v-cur": 512,
        "kqv-out": 4096,
        "ffn-inp": 2048,
        "ffn-moe-out": 2048,
        "output": 2048,
    }
    for layer in range(2):
        for phase, count in phase_counts.items():
            path = dump_dir / f"step-0-layer-{layer}-{phase}.f32"
            assert path.stat().st_size == count * 4
            values = struct.unpack(f"<{count}f", path.read_bytes())
            assert all(math.isfinite(value) for value in values)


def test_state_loop_supports_diagnostic_f32_kv(tmp_path):
    if not EXE.exists() or not MODEL.exists():
        pytest.skip("real Qwen runtime fixtures are not available")
    dump_dir = tmp_path / "f32-residuals"
    dump_dir.mkdir()
    payload = json.loads(
        subprocess.check_output(
            [
                str(EXE), "state-loop-probe", "--in", str(MODEL),
                "--prompt-token", "42", "--steps", "1", "--layers", "1",
                "--ctx", "4", "--kv", "f32", "--temperature", "0",
                "--seed", "7", "--full-moe", "--dump-residuals", str(dump_dir),
            ],
            text=True,
        )
    )
    assert payload["kv_format"] == "f32"
    assert "diagnostic F32 KV" in payload["note"]
    assert payload["bytes_per_k_or_v"] == 512 * 4
    assert payload["tokens"][0]["layers"][0]["kv_scale_source"] == "lossless_f32"
    assert payload["residual_dump_count"] == 6
    assert (dump_dir / "step-0-layer-0-v-cur.f32").stat().st_size == 512 * 4
    assert (dump_dir / "step-0-layer-0-kqv-out.f32").stat().st_size == 4096 * 4
    assert (dump_dir / "step-0-layer-0-ffn-inp.f32").stat().st_size == 2048 * 4


def test_q8_k_compat_reports_combined_expert_kernel_families_for_two_layers():
    if not EXE.exists() or not MODEL.exists():
        pytest.skip("real Qwen runtime fixtures are not available")
    payload = json.loads(
        subprocess.check_output(
            [str(EXE), "state-loop-probe", "--in", str(MODEL), "--prompt-token", "42", "--steps", "1", "--layers", "2", "--ctx", "4", "--kv", "f32", "--activation", "q8_k_compat", "--temperature", "0", "--seed", "7", "--full-moe"],
            text=True,
        )
    )
    assert payload["activation_format"] == "q8_k_compat"
    assert payload["projection_kernel"] == "iq4_xs_q5_k_q8_k"
    assert payload["moe_projection_kernel"] == "iq2_xs_iq3_xxs_iq2_s_iq4_xs_q8_k"
    assert payload["moe_gate_up_projection_kernel"] == "iq2_xs_iq2_s_q8_k"
    assert payload["moe_down_projection_kernel"] == "iq3_xxs_iq4_xs_q8_k"
    assert payload["layers_run"] == 2
    assert payload["tokens"][0]["layers"][1]["q_ggml_type"] == 13


def test_state_loop_exposes_explicit_q8_k_compat_activation_mode(tmp_path):
    if not EXE.exists() or not MODEL.exists():
        pytest.skip("real Qwen runtime fixtures are not available")
    dump_dir = tmp_path / "q8-k-residuals"
    dump_dir.mkdir()
    payload = json.loads(
        subprocess.check_output(
            [
                str(EXE), "state-loop-probe", "--in", str(MODEL),
                "--prompt-token", "42", "--steps", "1", "--layers", "1",
                "--ctx", "4", "--kv", "f32", "--activation", "q8_k_compat",
                "--temperature", "0", "--seed", "7", "--full-moe",
                "--dump-residuals", str(dump_dir),
            ],
            text=True,
        )
    )
    assert payload["activation_format"] == "q8_k_compat"
    assert payload["projection_kernel"] == "iq4_xs_q8_k"
    assert payload["activation_workspace_bytes"] == 16 * 292
    assert payload["moe_projection_kernel"] == "iq2_xs_q8_k_and_iq3_xxs_q8_k"
    assert payload["moe_gate_up_projection_kernel"] == "iq2_xs_q8_k"
    assert payload["moe_down_projection_kernel"] == "iq3_xxs_q8_k"
    assert payload["moe_activation_workspace_bytes"] == 8 * 292
    assert (dump_dir / "step-0-layer-0-v-cur.f32").stat().st_size == 512 * 4




def test_state_loop_propagates_one_real_token_across_all_48_layers():
    if not EXE.exists() or not MODEL.exists():
        pytest.skip("real Qwen runtime fixtures are not available")
    payload = json.loads(
        subprocess.check_output(
            [str(EXE), "state-loop-probe", "--in", str(MODEL), "--prompt-token", "42", "--steps", "1", "--layers", "48", "--ctx", "4", "--kv", "int8", "--top-k", "3", "--scan", "64", "--temperature", "0", "--seed", "7", "--full-moe"],
            text=True,
        )
    )
    layers = payload["tokens"][0]["layers"]
    assert payload["layers"] == 48
    assert payload["layers_run"] == 48
    assert payload["kv_appends"] == 48
    assert payload["cache_readback_ok"] is True
    assert len(layers) == 48
    assert [layer["layer"] for layer in layers] == list(range(48))
    for index, layer in enumerate(layers):
        assert layer["full_moe"] is True
        assert layer["qk_head_norm"] is True
        assert layer["attention_context_tokens"] == 1
        assert layer["q_tensor"] == f"blk.{index}.attn_q.weight"
        assert layer["k_tensor"] == f"blk.{index}.attn_k.weight"
        assert layer["v_tensor"] == f"blk.{index}.attn_v.weight"
        assert layer["output_tensor"] == f"blk.{index}.attn_output.weight"
        assert layer["experts_run"] == 8
        assert len(layer["selected_experts"]) == 8
        assert len(set(layer["selected_experts"])) == 8
        assert layer["moe_output_l2"] > 0
        assert layer["residual_output_checksum"] != layer["residual_input_checksum"]
        if index:
            assert layers[index - 1]["residual_output_checksum"] == layer["residual_input_checksum"]
    assert layers[0]["residual_input_checksum"] != layers[-1]["residual_output_checksum"]


def test_state_loop_applies_real_final_norm_and_complete_lm_head(tmp_path):
    if not EXE.exists() or not MODEL.exists():
        pytest.skip("real Qwen runtime fixtures are not available")
    dump_dir = tmp_path / "final-head-dump"
    dump_dir.mkdir()
    payload = json.loads(
        subprocess.check_output(
            [str(EXE), "state-loop-probe", "--in", str(MODEL), "--prompt-token", "42", "--steps", "1", "--layers", "48", "--ctx", "4", "--kv", "int8", "--temperature", "0", "--seed", "7", "--full-moe", "--final-head", "--top-n", "5", "--dump-residuals", str(dump_dir)],
            text=True,
        )
    )
    head = payload["tokens"][0]["final_head"]
    assert head["enabled"] is True
    assert head["norm_tensor"] == "output_norm.weight"
    assert head["norm_ggml_type"] == 0
    assert head["norm_values"] == 2048
    assert head["norm_checksum_verified"] is True
    assert head["norm_raw_checksum"] > 0
    assert len(head["final_residual_raw"]) == 2048
    assert len(head["final_norm_raw"]) == 2048
    assert head["final_residual_checksum"] == payload["tokens"][0]["layers"][-1]["residual_output_checksum"]
    assert head["final_norm_checksum"] != head["final_residual_checksum"]
    assert head["lm_head_tensor"] == "output.weight"
    assert head["lm_head_ggml_type"] == 14
    assert head["lm_head_decoder"] == "Q6_K"
    assert head["lm_head_checksum_verified"] is True
    assert head["lm_head_raw_checksum"] > 0
    assert head["input_dims"] == 2048
    assert head["vocab_size"] == 151936
    assert head["logits_computed"] == 151936
    assert head["full_vocabulary"] is True
    assert head["logits_checksum"] > 0
    logits_path = dump_dir / "step-0-logits.f32"
    assert payload["logits_dump_count"] == 1
    assert logits_path.stat().st_size == 151936 * 4
    assert fnv1a64(logits_path.read_bytes()) == head["logits_checksum"]
    assert math.isfinite(head["logits_rms"])
    assert head["logits_rms"] > 0
    assert len(head["top_tokens"]) == 5
    assert head["argmax_token"] == head["top_tokens"][0]["token"]
    assert head["argmax_logit"] == head["top_tokens"][0]["logit"]
    assert all(head["top_tokens"][index]["logit"] >= head["top_tokens"][index + 1]["logit"] for index in range(4))
    base = [str(EXE), "state-loop-probe", "--in", str(MODEL), "--prompt-token", "42", "--steps", "1", "--layers", "48", "--ctx", "4", "--kv", "int8", "--temperature", "0", "--seed", "7", "--full-moe", "--final-head"]
    invalid_commands = [
        base + ["--layers", "47"],
        base + ["--layers", "49"],
        base + ["--steps", "0"],
        base + ["--steps", "65"],
        base + ["--temperature", "0.1"],
        base + ["--bench"],
        [arg for arg in base if arg != "--full-moe"],
    ]
    for command in invalid_commands:
        invalid = subprocess.run(
            command,
            text=True,
            capture_output=True,
        )
        assert invalid.returncode != 0
        assert "--final-head requires --full-moe, 1..64 steps, all manifest layers, temperature 0, and no --bench" in invalid.stderr
    oversized_ctx = subprocess.run(base + ["--ctx", "4097"], text=True, capture_output=True)
    assert oversized_ctx.returncode != 0
    assert "state loop context must be between 1 and 4096" in oversized_ctx.stderr
    zero_ctx = subprocess.run(base + ["--ctx", "0"], text=True, capture_output=True)
    assert zero_ctx.returncode != 0
    assert "state loop context must be between 1 and 4096" in zero_ctx.stderr
    beyond_ctx = subprocess.run(base + ["--steps", "5"], text=True, capture_output=True)
    assert beyond_ctx.returncode != 0
    assert "steps exceed ctx" in beyond_ctx.stderr
    zero_kv_heads = tmp_path / "zero-kv-heads.qxf"
    qxf_with_manifest_u32(MODEL, zero_kv_heads, 24, 0)
    malformed_command = [str(zero_kv_heads) if arg == str(MODEL) else arg for arg in base]
    malformed = subprocess.run(malformed_command, text=True, capture_output=True)
    assert malformed.returncode != 0
    assert "invalid QXF manifest" in malformed.stderr


def test_final_norm_and_argmax_match_independent_llama_cpp_reference(tmp_path):
    if not EXE.exists() or not MODEL.exists() or not GGML_REFERENCE.exists():
        pytest.skip("real Qwen fixtures or external llama.cpp reference are not available")
    payload = json.loads(
        subprocess.check_output(
            [str(EXE), "state-loop-probe", "--in", str(MODEL), "--prompt-token", "42", "--steps", "1", "--layers", "48", "--ctx", "4", "--kv", "int8", "--temperature", "0", "--seed", "7", "--full-moe", "--final-head", "--top-n", "5"],
            text=True,
        )
    )
    head = payload["tokens"][0]["final_head"]
    residual = head["final_residual_raw"]
    norm_tensor = metadata("output_norm.weight")
    norm_weights = struct.unpack("<2048f", read_at(norm_tensor["offset"], norm_tensor["byte_size"]))
    rms = math.sqrt(sum(float(value) * float(value) for value in residual) / 2048 + 1e-6)
    expected_norm = [f32(f32(float(value) / rms) * norm_weights[index]) for index, value in enumerate(residual)]
    for got, expected in zip(head["final_norm_raw"], expected_norm):
        assert math.isclose(got, expected, rel_tol=3e-6, abs_tol=3e-6)
    output_tensor = metadata("output.weight")
    reference_logits = ggml_reference_full_q6_logits(output_tensor, head["final_norm_raw"], tmp_path / "final-norm.f32")
    assert head["logits_checksum"] == fnv1a64(struct.pack(f"<{len(reference_logits)}f", *reference_logits))
    reference_top = sorted(range(len(reference_logits)), key=lambda token: reference_logits[token], reverse=True)[:5]
    assert head["argmax_token"] == reference_top[0]
    assert [item["token"] for item in head["top_tokens"]] == reference_top
    for item in head["top_tokens"]:
        assert math.isclose(item["logit"], reference_logits[item["token"]], rel_tol=4e-6, abs_tol=4e-6)
    for token in (0, 75968, 151935):
        assert math.isfinite(reference_logits[token])


def test_state_loop_runs_two_real_greedy_tokens_from_selected_embedding(tmp_path):
    if not EXE.exists() or not MODEL.exists() or not GGML_REFERENCE.exists():
        pytest.skip("real Qwen runtime fixtures are not available")
    common = [str(EXE), "state-loop-probe", "--in", str(MODEL), "--layers", "48", "--ctx", "4", "--kv", "int8", "--temperature", "0", "--seed", "7", "--full-moe", "--final-head", "--top-n", "5"]
    payload = json.loads(subprocess.check_output(common + ["--prompt-token", "42", "--steps", "2"], text=True))
    assert payload["steps"] == 2
    assert payload["layers_run"] == 96
    assert payload["kv_appends"] == 96
    assert payload["cache_readback_ok"] is True
    assert len(payload["tokens"]) == 2
    first, second = payload["tokens"]
    assert first["input_token"] == 42
    assert first["selected_token"] == 1124
    assert second["input_token"] == first["selected_token"]
    assert second["selected_token"] == 50853
    assert second["position"] == 1
    assert all(layer["attention_context_tokens"] == 1 for layer in first["layers"])
    assert all(layer["attention_context_tokens"] == 2 for layer in second["layers"])
    assert all(layer["kv_token"] == 0 for layer in first["layers"])
    assert all(layer["kv_token"] == 1 for layer in second["layers"])
    assert first["final_head"]["logits_computed"] == 151936
    assert second["final_head"]["logits_computed"] == 151936
    assert first["final_head"]["logits_checksum"] == 10967348620636053936
    assert second["final_head"]["logits_checksum"] == 14548714559300682082
    assert second["residual_checksum"] != first["final_head"]["final_residual_checksum"]
    embedding = metadata("token_embd.weight")
    assert embedding["ggml_type"] == 12
    embedding_row_bytes = 8 * 144
    prompt_embedding_raw = read_at(embedding["offset"] + first["input_token"] * embedding_row_bytes, embedding_row_bytes)
    selected_embedding_raw = read_at(embedding["offset"] + second["input_token"] * embedding_row_bytes, embedding_row_bytes)
    assert first["residual_checksum"] == fnv1a64(prompt_embedding_raw)
    assert second["residual_checksum"] == fnv1a64(selected_embedding_raw)
    assert prompt_embedding_raw != selected_embedding_raw
    output_tensor = metadata("output.weight")
    second_reference_logits = ggml_reference_full_q6_logits(output_tensor, second["final_head"]["final_norm_raw"], tmp_path / "second-final-norm.f32")
    assert second["final_head"]["logits_checksum"] == fnv1a64(struct.pack(f"<{len(second_reference_logits)}f", *second_reference_logits))
    assert second["selected_token"] == max(range(len(second_reference_logits)), key=second_reference_logits.__getitem__)
