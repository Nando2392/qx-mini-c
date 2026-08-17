#!/usr/bin/env python
"""Export the Qwen2/Qwen3 GPT-2 BPE metadata from GGUF to QXT2."""

import argparse
import json
import os
import struct
from pathlib import Path


GGUF_TYPES = {
    0: ("B", 1),
    1: ("b", 1),
    2: ("H", 2),
    3: ("h", 2),
    4: ("I", 4),
    5: ("i", 4),
    6: ("f", 4),
    7: ("B", 1),
    10: ("Q", 8),
    11: ("q", 8),
    12: ("d", 8),
}
GGUF_STRING = 8
GGUF_ARRAY = 9
MAX_STRING_BYTES = 1 << 20
MAX_ARRAY_ITEMS = 1_000_000
MAX_SIDECAR_BYTES = 256 << 20
QXT_HEADER = struct.Struct("<4sIIIIIiiIIQ")
GGUF_INTEGER_TYPES = {0, 1, 2, 3, 4, 5, 10, 11}


class GGUFError(ValueError):
    pass


class Reader:
    def __init__(self, path):
        self.path = path
        self.handle = path.open("rb")
        self.size = path.stat().st_size
        self.last_array_element_type = None

    def close(self):
        self.handle.close()

    def read(self, size):
        if size < 0 or size > self.size - self.handle.tell():
            raise GGUFError("GGUF value exceeds file bounds")
        data = self.handle.read(size)
        if len(data) != size:
            raise GGUFError("short GGUF read")
        return data

    def unpack(self, fmt):
        codec = struct.Struct("<" + fmt)
        return codec.unpack(self.read(codec.size))[0]

    def string(self):
        length = self.unpack("Q")
        if length > MAX_STRING_BYTES:
            raise GGUFError("GGUF string exceeds 1 MiB limit")
        try:
            return self.read(length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFError("GGUF tokenizer string is not valid UTF-8") from exc

    def scalar(self, value_type):
        if value_type not in GGUF_TYPES:
            raise GGUFError(f"unsupported GGUF scalar type {value_type}")
        fmt, _ = GGUF_TYPES[value_type]
        return self.unpack(fmt)

    def value(self, value_type):
        self.last_array_element_type = None
        if value_type == GGUF_STRING:
            return self.string()
        if value_type == GGUF_ARRAY:
            element_type = self.unpack("I")
            self.last_array_element_type = element_type
            count = self.unpack("Q")
            if count > MAX_ARRAY_ITEMS:
                raise GGUFError("GGUF array exceeds item limit")
            if element_type == GGUF_STRING:
                return [self.string() for _ in range(count)]
            if element_type not in GGUF_TYPES:
                raise GGUFError(f"unsupported GGUF array element type {element_type}")
            fmt, size = GGUF_TYPES[element_type]
            total = count * size
            if total > MAX_SIDECAR_BYTES:
                raise GGUFError("GGUF array exceeds byte limit")
            data = self.read(total)
            return list(struct.unpack(f"<{count}{fmt}", data)) if count else []
        return self.scalar(value_type)


def fnv1a64(data):
    value = 1469598103934665603
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def read_tokenizer_metadata(path):
    reader = Reader(path)
    try:
        if reader.read(4) != b"GGUF":
            raise GGUFError("bad GGUF magic")
        version = reader.unpack("I")
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        reader.unpack("Q")  # tensor count
        kv_count = reader.unpack("Q")
        if kv_count > MAX_ARRAY_ITEMS:
            raise GGUFError("GGUF metadata count exceeds limit")
        wanted = {
            "tokenizer.ggml.model",
            "tokenizer.ggml.pre",
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.token_type",
            "tokenizer.ggml.merges",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.add_bos_token",
            "tokenizer.ggml.add_eos_token",
        }
        metadata = {}
        for _ in range(kv_count):
            key = reader.string()
            value_type = reader.unpack("I")
            value = reader.value(value_type)
            if key in wanted:
                metadata[key] = (value_type, reader.last_array_element_type, value)
        return metadata
    finally:
        reader.close()


def validated_payload(metadata):
    def required(name, value_type, element_type=None):
        item = metadata.get(name)
        if item is None or item[0] != value_type or item[1] != element_type:
            expected = f"type {value_type}" if element_type is None else f"array element type {element_type}"
            raise GGUFError(f"{name} requires GGUF {expected}")
        return item[2]

    def optional_integer(name, default):
        item = metadata.get(name)
        if item is None:
            return default
        if item[0] not in GGUF_INTEGER_TYPES or item[1] is not None or not isinstance(item[2], int) or isinstance(item[2], bool):
            raise GGUFError(f"{name} requires GGUF integer type")
        return item[2]

    def optional_bool(name, default):
        item = metadata.get(name)
        if item is None:
            return default
        if item[0] != 7 or item[1] is not None or item[2] not in (0, 1):
            raise GGUFError(f"{name} requires GGUF bool type")
        return bool(item[2])

    model = required("tokenizer.ggml.model", GGUF_STRING)
    pre = required("tokenizer.ggml.pre", GGUF_STRING)
    tokens = required("tokenizer.ggml.tokens", GGUF_ARRAY, GGUF_STRING)
    token_types = required("tokenizer.ggml.token_type", GGUF_ARRAY, 5)
    merges = required("tokenizer.ggml.merges", GGUF_ARRAY, GGUF_STRING)
    if model != "gpt2" or pre != "qwen2":
        raise GGUFError(f"expected tokenizer model=gpt2 pre=qwen2, got {model!r}/{pre!r}")
    if not isinstance(tokens, list) or not tokens or len(tokens) > MAX_ARRAY_ITEMS:
        raise GGUFError("missing or invalid tokenizer token array")
    if not isinstance(token_types, list) or len(token_types) != len(tokens):
        raise GGUFError("token type count does not match vocabulary")
    if not isinstance(merges, list) or len(merges) > MAX_ARRAY_ITEMS:
        raise GGUFError("missing or invalid tokenizer merge array")
    bos_id = optional_integer("tokenizer.ggml.bos_token_id", -1)
    eos_id = optional_integer("tokenizer.ggml.eos_token_id", -1)
    for name, token_id in (("BOS", bos_id), ("EOS", eos_id)):
        if token_id < -1 or token_id >= len(tokens):
            raise GGUFError(f"{name} token id out of range")
    flags = int(optional_bool("tokenizer.ggml.add_bos_token", False))
    flags |= int(optional_bool("tokenizer.ggml.add_eos_token", False)) << 1
    payload = bytearray(model.encode("utf-8") + pre.encode("utf-8"))
    for token, token_type in zip(tokens, token_types):
        if not isinstance(token, str) or "\x00" in token:
            raise GGUFError("token strings must be NUL-free UTF-8")
        raw = token.encode("utf-8")
        if len(raw) > MAX_STRING_BYTES:
            raise GGUFError("token string exceeds limit")
        if not isinstance(token_type, int) or isinstance(token_type, bool) or token_type < 1 or token_type > 6:
            raise GGUFError("token type outside GGML range 1..6")
        payload += struct.pack("<II", len(raw), token_type) + raw
    parsed_merges = []
    for merge in merges:
        if not isinstance(merge, str) or "\x00" in merge:
            raise GGUFError("merge strings must be NUL-free UTF-8")
        if " " not in merge:
            raise GGUFError("malformed BPE merge")
        left, right = merge.split(" ", 1)
        if not left or not right:
            raise GGUFError("empty BPE merge symbol")
        left_raw, right_raw = left.encode("utf-8"), right.encode("utf-8")
        if len(left_raw) > MAX_STRING_BYTES or len(right_raw) > MAX_STRING_BYTES:
            raise GGUFError("merge symbol exceeds limit")
        payload += struct.pack("<II", len(left_raw), len(right_raw)) + left_raw + right_raw
        parsed_merges.append((left, right))
    if len(payload) > MAX_SIDECAR_BYTES:
        raise GGUFError("QXT2 payload exceeds 256 MiB limit")
    header = QXT_HEADER.pack(
        b"QXT2",
        2,
        len(tokens),
        len(parsed_merges),
        len(model.encode("utf-8")),
        len(pre.encode("utf-8")),
        bos_id,
        eos_id,
        flags,
        len(payload),
        fnv1a64(payload),
    )
    return header + payload, len(tokens), len(parsed_merges), flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        metadata = read_tokenizer_metadata(args.gguf)
        sidecar, vocab_count, merge_count, flags = validated_payload(metadata)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(args.out.name + ".tmp")
        temporary.write_bytes(sidecar)
        os.replace(temporary, args.out)
    except (OSError, GGUFError, struct.error) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "format": "QXT2",
        "vocab_count": vocab_count,
        "merge_count": merge_count,
        "add_bos": bool(flags & 1),
        "add_eos": bool(flags & 2),
        "output": str(args.out),
    }))


if __name__ == "__main__":
    main()
