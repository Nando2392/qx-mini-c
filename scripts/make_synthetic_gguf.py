#!/usr/bin/env python
import argparse
import struct
from pathlib import Path

GGUF_UINT32 = 4
GGUF_INT32 = 5
GGUF_BOOL = 7
GGUF_STRING = 8
GGUF_ARRAY = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q4_K = 12
GGML_TYPE_F32 = 0
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23


def wstr(f, s):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def meta_u32(f, key, value):
    wstr(f, key)
    f.write(struct.pack("<I", GGUF_UINT32))
    f.write(struct.pack("<I", value))


def meta_str(f, key, value):
    wstr(f, key)
    f.write(struct.pack("<I", GGUF_STRING))
    wstr(f, value)


def meta_str_array(f, key, values):
    wstr(f, key)
    f.write(struct.pack("<I", GGUF_ARRAY))
    f.write(struct.pack("<IQ", GGUF_STRING, len(values)))
    for value in values:
        wstr(f, value)


def meta_i32_array(f, key, values):
    wstr(f, key)
    f.write(struct.pack("<I", GGUF_ARRAY))
    f.write(struct.pack("<IQ", GGUF_INT32, len(values)))
    for value in values:
        f.write(struct.pack("<i", value))


def meta_bool(f, key, value):
    wstr(f, key)
    f.write(struct.pack("<IB", GGUF_BOOL, int(value)))


def tensor(f, name, dims, typ, offset):
    wstr(f, name)
    f.write(struct.pack("<I", len(dims)))
    for d in dims:
        f.write(struct.pack("<Q", d))
    f.write(struct.pack("<I", typ))
    f.write(struct.pack("<Q", offset))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/fixtures/qwen3-30b-a3b-mini.gguf")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer_tokens = [f"tok{i}" for i in range(64)]
    tokenizer_tokens[:11] = ["H", "e", "l", "o", "He", "ll", "Hell", "Hello", "Ġ", "world", "<|im_start|>"]
    tokenizer_types = [1] * 64
    tokenizer_types[10] = 3
    kvs = [
        ("str", "general.architecture", "qwen3moe"),
        ("str", "general.name", "synthetic-qwen3-30b-a3b-mini"),
        ("u32", "general.alignment", 32),
        ("u32", "qwen3moe.block_count", 48),
        ("u32", "qwen3moe.embedding_length", 2048),
        ("u32", "qwen3moe.feed_forward_length", 6144),
        ("u32", "qwen3moe.attention.head_count", 32),
        ("u32", "qwen3moe.attention.head_count_kv", 4),
        ("u32", "qwen3moe.expert_count", 128),
        ("u32", "qwen3moe.expert_used_count", 8),
        ("str", "tokenizer.ggml.model", "gpt2"),
        ("str", "tokenizer.ggml.pre", "qwen2"),
        ("str_array", "tokenizer.ggml.tokens", tokenizer_tokens),
        ("i32_array", "tokenizer.ggml.token_type", tokenizer_types),
        ("str_array", "tokenizer.ggml.merges", ["H e", "l l", "He ll", "Hell o"]),
        ("u32", "tokenizer.ggml.bos_token_id", 10),
        ("u32", "tokenizer.ggml.eos_token_id", 10),
        ("bool", "tokenizer.ggml.add_bos_token", False),
        ("bool", "tokenizer.ggml.add_eos_token", False),
    ]
    tensors = [
        ("token_embd.weight", [2048, 151936], GGML_TYPE_Q4_K, 0),
        ("blk.0.attn_q.weight", [256, 256], GGML_TYPE_Q4_K, 2320384),
        ("blk.0.attn_k.weight", [256, 256], GGML_TYPE_Q4_K, 2357248),
        ("blk.0.attn_v.weight", [256, 256], GGML_TYPE_Q4_K, 2394112),
        ("blk.0.attn_output.weight", [256, 256], GGML_TYPE_Q4_K, 2430976),
        ("blk.0.ffn_gate_inp.weight", [2048, 128], GGML_TYPE_F32, 2048),
        ("blk.0.ffn_gate_exps.weight", [2048, 768, 128], GGML_TYPE_IQ2_XS, 1050624),
        ("blk.0.ffn_up_exps.weight", [2048, 768, 128], GGML_TYPE_IQ2_XS, 1083392),
        ("blk.0.ffn_down_exps.weight", [768, 2048, 128], GGML_TYPE_IQ3_XXS, 1116160),
        ("blk.1.ffn_gate_inp.weight", [2048, 128], GGML_TYPE_F32, 1157120),
        ("blk.1.ffn_gate_exps.weight", [2048, 768, 128], GGML_TYPE_IQ2_S, 2205696),
        ("blk.1.ffn_up_exps.weight", [2048, 768, 128], GGML_TYPE_IQ2_S, 2238464),
        ("blk.1.ffn_down_exps.weight", [768, 2048, 128], GGML_TYPE_IQ4_XS, 2271232),
        ("blk.0.attn_norm.weight", [2048], GGML_TYPE_F32, 2312192),
    ]

    with out.open("wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<IQQ", 3, len(tensors), len(kvs)))
        for typ, key, val in kvs:
            if typ == "str":
                meta_str(f, key, val)
            elif typ == "str_array":
                meta_str_array(f, key, val)
            elif typ == "i32_array":
                meta_i32_array(f, key, val)
            elif typ == "bool":
                meta_bool(f, key, val)
            else:
                meta_u32(f, key, val)
        for item in tensors:
            tensor(f, *item)
        pos = f.tell()
        pad = (-pos) % 32
        f.write(b"\0" * pad)
        data_len = 2468352
        data = bytearray((i % 251 for i in range(data_len)))
        # Keep Q4_K attention fixture scales finite. Arbitrary byte patterns can
        # encode FP16 NaNs in d/dmin and are not valid quantized tensor data.
        for tensor_offset in (2320384, 2357248, 2394112, 2430976):
            for block in range(256):
                base = tensor_offset + block * 144
                data[base:base + 4] = b"\x00\x3c\x00\x00"  # d=1.0, dmin=0.0
        # RMSNorm weights are real F32 values, not arbitrary fixture bytes.
        norm_offset = 2312192
        for index in range(2048):
            struct.pack_into("<f", data, norm_offset + index * 4, 1.0)
        f.write(data)
    print(out)


if __name__ == "__main__":
    main()
