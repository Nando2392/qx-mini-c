"""Deterministic end-to-end smoke check for the QXF C runtime."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXE = ROOT / "build" / "qxqxf.exe"


def run(*args: object) -> str:
    return subprocess.check_output([str(arg) for arg in args], cwd=ROOT, text=True)


def fnv1a64(data: bytes | bytearray) -> int:
    value = 1469598103934665603
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def make_compact_valid_embedding(qxf: Path) -> None:
    raw = bytearray(qxf.read_bytes())
    dir_offset = struct.unpack_from("<Q", raw, 24)[0]
    tensor_count = struct.unpack_from("<I", raw, 16)[0]
    entry_size = 208
    source_row = None
    for index in range(tensor_count):
        entry = dir_offset + index * entry_size
        if bytes(raw[entry:entry + 96]).split(b"\0", 1)[0] == b"blk.0.attn_q.weight":
            source_offset = struct.unpack_from("<Q", raw, entry + 144)[0]
            source_row = bytes(raw[source_offset:source_offset + 8 * 144])
            break
    assert source_row is not None and len(source_row) == 8 * 144
    first = bytearray(raw[dir_offset:dir_offset + entry_size])
    entries_end = dir_offset + tensor_count * entry_size
    raw[dir_offset:entries_end - entry_size] = raw[dir_offset + entry_size:entries_end]
    token_offset = (len(raw) + 4095) & ~4095
    raw.extend(b"\0" * (token_offset - len(raw)))
    token_data = source_row * 8
    raw.extend(token_data)
    struct.pack_into("<I", first, 104, 2)
    struct.pack_into("<4Q", first, 112, 2048, 8, 0, 0)
    struct.pack_into("<Q", first, 144, token_offset)
    struct.pack_into("<Q", first, 152, len(token_data))
    struct.pack_into("<Q", first, 168, fnv1a64(token_data))
    raw[entries_end - entry_size:entries_end] = first
    struct.pack_into("<I", raw, 88, 8)
    struct.pack_into("<Q", raw, 48, fnv1a64(raw[56:108]))
    struct.pack_into("<Q", raw, 40, len(raw))
    qxf.write_bytes(raw)


def main() -> int:
    if not EXE.exists():
        subprocess.check_call(["cmd.exe", "/c", "build_msvc.bat"], cwd=ROOT)

    with tempfile.TemporaryDirectory(prefix="qx-smoke-") as temp:
        temp_path = Path(temp)
        gguf = temp_path / "synthetic.gguf"
        qxf = temp_path / "synthetic.qxf"
        tokens = temp_path / "tokens.tsv"
        tokenizer = temp_path / "tokenizer.qxt"
        prompt = temp_path / "prompt.txt"

        run(sys.executable, ROOT / "scripts" / "make_synthetic_gguf.py", "--out", gguf)
        run(EXE, "create-from-gguf-copy", "--in", gguf, "--model", "qwen3-30b-a3b", "--quant", "q2", "--out", qxf)
        malformed = subprocess.run(
            [str(EXE), "token-embedding", "--in", str(qxf), "--token-id", "2"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert malformed.returncode != 0
        assert "tensor byte size is not divisible by row count" in malformed.stderr
        make_compact_valid_embedding(qxf)
        run(EXE, "tokenizer-export", "--gguf", gguf, "--out", tokens)
        run(sys.executable, ROOT / "scripts" / "export_qwen3_tokenizer.py", "--gguf", gguf, "--out", tokenizer)
        prompt.write_text("Hello", encoding="utf-8")
        tokenizer_summary = json.loads(run(EXE, "tokenizer-inspect", "--tokenizer", tokenizer))
        encoded = json.loads(run(EXE, "tokenizer-encode", "--tokenizer", tokenizer, "--text-file", prompt))
        decoded = json.loads(run(EXE, "tokenizer-decode", "--tokenizer", tokenizer, "--ids", "7"))
        result = json.loads(
            run(
                EXE,
                "state-loop-probe",
                "--in",
                qxf,
                "--tokens",
                tokens,
                "--prompt-token",
                2,
                "--steps",
                2,
                "--layers",
                2,
                "--ctx",
                16,
                "--kv",
                "int8",
                "--top-k",
                3,
                "--scan",
                32,
                "--temperature",
                0,
                "--seed",
                7,
                "--rope-gqa-attention",
                "--residual-dims",
                1152,
                "--norm",
                "blk.0.attn_norm.weight",
            )
        )
        golden = json.loads(run(EXE, "rope-gqa-golden-probe", "--tokens", 2, "--q-heads-run", 9, "--seed", 7))

    real_model = ROOT / "models" / "Qwen3-30B-A3B-UD-IQ2_M.qxf"
    real_golden = None
    real_state = None
    if real_model.exists():
        real_golden = json.loads(run(EXE, "real-qkv-golden-probe", "--in", real_model, "--layer", 0, "--token-a", 42, "--token-b", 43, "--q-heads-run", 32, "--seed", 7, "--full-moe"))
        real_state = json.loads(run(EXE, "state-loop-probe", "--in", real_model, "--prompt-token", 42, "--steps", 2, "--layers", 48, "--ctx", 4, "--kv", "int8", "--temperature", 0, "--seed", 7, "--full-moe", "--final-head", "--top-n", 5))

    layer0 = result["tokens"][0]["layers"][0]
    assert result["delta_source"] == "rope_gqa_attention"
    assert result["cache_readback_ok"] is True
    assert result["layers_run"] == 4
    assert result["kv_appends"] == 4
    assert layer0["persistent_kv"] is True
    assert layer0["kv_scale_source"] == "dynamic_per_vector"
    assert layer0["k_scale"] > 0
    assert layer0["v_scale"] > 0
    assert abs(layer0["softmax_sum"] - 1.0) < 1e-6
    assert layer0["attention_mode"] == "rope_gqa_full_heads"
    assert layer0["rope_applied"] is True
    assert layer0["q_heads_run"] == 2
    assert layer0["kv_heads_touched"] == 1
    assert layer0["gqa_group_size"] == 8
    assert abs(layer0["softmax_sum_min"] - 1.0) < 1e-6
    assert abs(layer0["softmax_sum_max"] - 1.0) < 1e-6
    assert layer0["attention_output_vector_checksum"] > 0
    assert golden["probe"] == "rope_gqa_golden"
    assert golden["rope_layout"] == "qwen_split_half"
    assert golden["q_heads_run"] == 9
    assert golden["kv_heads_touched"] == 2
    assert abs(golden["softmax_sum_min"] - 1.0) < 1e-9
    assert abs(golden["softmax_sum_max"] - 1.0) < 1e-9
    assert tokenizer_summary["checksum_verified"] is True
    assert encoded["token_ids"] == [7]
    assert decoded["text"] == "Hello"
    if real_golden is not None:
        assert real_golden["probe"] == "real_qkv_golden"
        assert real_golden["projection_layout"] == "contiguous_tensor_rows"
        assert real_golden["projection_input_dims"] == 2048
        assert real_golden["projection_blocks_per_row"] == 8
        assert real_golden["q_heads_run"] == 32
        assert real_golden["kv_heads_total"] == 4
        assert real_golden["full_head_coverage"] is True
        assert real_golden["post_attention_norm_tensor"] == "blk.0.ffn_norm.weight"
        assert len(real_golden["selected_experts"]) == 8
        assert real_golden["moe_mode"] == "real_top8_swiglu"
        assert real_golden["experts_run"] == 8
        assert real_golden["moe_output_l2"] > 0
        assert len(real_golden["layer_output_raw"]) == 2048
    if real_state is not None:
        real_layers = real_state["tokens"][0]["layers"]
        second_layers = real_state["tokens"][1]["layers"]
        assert real_state["residual_source"] == "real_attention_moe_carry"
        assert real_state["layers_run"] == 96
        assert real_state["kv_appends"] == 96
        assert [token["input_token"] for token in real_state["tokens"]] == [42, 1124]
        assert [token["selected_token"] for token in real_state["tokens"]] == [1124, 50853]
        assert [token["position"] for token in real_state["tokens"]] == [0, 1]
        assert len(real_layers) == 48
        assert len(second_layers) == 48
        assert all(layer["full_moe"] is True for layer in real_layers)
        assert all(layer["qk_head_norm"] is True for layer in real_layers)
        assert all(layer["experts_run"] == 8 for layer in real_layers)
        assert all(real_layers[index - 1]["residual_output_checksum"] == real_layers[index]["residual_input_checksum"] for index in range(1, 48))
        assert all(layer["attention_context_tokens"] == 2 for layer in second_layers)
        assert real_state["tokens"][1]["residual_checksum"] != real_state["tokens"][0]["final_head"]["final_residual_checksum"]
        real_head = real_state["tokens"][0]["final_head"]
        assert real_head["norm_tensor"] == "output_norm.weight"
        assert real_head["lm_head_tensor"] == "output.weight"
        assert real_head["lm_head_ggml_type"] == 14
        assert real_head["logits_computed"] == 151936
        assert real_head["argmax_token"] == real_head["top_tokens"][0]["token"]

    print(
        json.dumps(
            {
                "status": "pass",
                "probe": result["probe"],
                "delta_source": result["delta_source"],
                "layers_run": result["layers_run"],
                "kv_appends": result["kv_appends"],
                "cache_readback_ok": result["cache_readback_ok"],
                "softmax_sum": layer0["softmax_sum"],
                "golden_probe": golden["probe"],
                "tokenizer_qxt2": tokenizer_summary["checksum_verified"],
                "tokenizer_ids": encoded["token_ids"],
                "real_qkv_golden": real_golden is not None,
                "real_48_layer_state": real_state is not None,
                "real_layers_run": real_state["layers_run"] if real_state is not None else 0,
                "real_carry_links": 47 if real_state is not None else 0,
                "real_final_head": real_state is not None,
                "real_argmax_token": real_state["tokens"][0]["final_head"]["argmax_token"] if real_state is not None else None,
                "real_autoregressive_sequence": [token["selected_token"] for token in real_state["tokens"]] if real_state is not None else [],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
