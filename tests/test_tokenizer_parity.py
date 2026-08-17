import json
import hashlib
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "scripts" / "export_qwen3_tokenizer.py"
EXE = ROOT / "build" / "qxqxf.exe"
REAL_GGUF = ROOT / "models" / "Qwen3-30B-A3B-UD-IQ2_M.gguf"
REAL_QXF = ROOT / "models" / "Qwen3-30B-A3B-UD-IQ2_M.qxf"
HEADER = struct.Struct("<4sIIIIIiiIIQ")
ORACLE_FIXTURE = ROOT / "tests" / "fixtures" / "qwen3-tokenizer-llama-cpp-goldens.json"
LLAMA_CPP = ROOT.parent / "llama.cpp-k3"
LLAMA_TOKENIZE = Path(os.environ.get("QX_LLAMA_TOKENIZE", LLAMA_CPP / "build-k3-cpu" / "bin" / "Release" / "llama-tokenize.exe"))


def fnv1a64(data):
    value = 1469598103934665603
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def wstr(handle, value):
    raw = value.encode("utf-8")
    handle.write(struct.pack("<Q", len(raw)))
    handle.write(raw)


def metadata_string(handle, key, value):
    wstr(handle, key)
    handle.write(struct.pack("<I", 8))
    wstr(handle, value)


def metadata_u32(handle, key, value):
    wstr(handle, key)
    handle.write(struct.pack("<II", 4, value))


def metadata_bool(handle, key, value):
    wstr(handle, key)
    handle.write(struct.pack("<IB", 7, int(value)))


def metadata_array(handle, key, element_type, values):
    wstr(handle, key)
    handle.write(struct.pack("<IIQ", 9, element_type, len(values)))
    for value in values:
        if element_type == 8:
            wstr(handle, value)
        elif element_type == 5:
            handle.write(struct.pack("<i", value))
        elif element_type == 6:
            handle.write(struct.pack("<f", value))
        else:
            raise AssertionError(element_type)


def make_tokenizer_gguf(path, *, bos_writer=metadata_u32, bos_value=10, add_bos_writer=metadata_bool,
                        add_bos_value=False, token_type_element=5, extra_tokens=()):
    tokens = ["H", "e", "l", "o", "He", "ll", "Hell", "Hello", "Ġ", "world", "<|im_start|>", *extra_tokens]
    merges = ["H e", "l l", "He ll", "Hell o"]
    token_types = [1] * 10 + [3] + [1] * len(extra_tokens)
    if token_type_element == 6:
        token_types = [float(value) for value in token_types]
    metadata = [
        (metadata_string, ("tokenizer.ggml.model", "gpt2")),
        (metadata_string, ("tokenizer.ggml.pre", "qwen2")),
        (metadata_array, ("tokenizer.ggml.tokens", 8, tokens)),
        (metadata_array, ("tokenizer.ggml.token_type", token_type_element, token_types)),
        (metadata_array, ("tokenizer.ggml.merges", 8, merges)),
        (bos_writer, ("tokenizer.ggml.bos_token_id", bos_value)),
        (metadata_u32, ("tokenizer.ggml.eos_token_id", 10)),
        (add_bos_writer, ("tokenizer.ggml.add_bos_token", add_bos_value)),
        (metadata_bool, ("tokenizer.ggml.add_eos_token", False)),
    ]
    with path.open("wb") as handle:
        handle.write(b"GGUF")
        handle.write(struct.pack("<IQQ", 3, 0, len(metadata)))
        for writer, args in metadata:
            writer(handle, *args)
    return tokens, merges, token_types


def read_qxt(path):
    raw = path.read_bytes()
    header = HEADER.unpack_from(raw)
    magic, version, vocab_count, merge_count, model_len, pre_len, bos_id, eos_id, flags, payload_len, checksum = header
    payload = raw[HEADER.size:]
    assert len(payload) == payload_len
    assert fnv1a64(payload) == checksum
    cursor = 0
    model = payload[cursor:cursor + model_len].decode("utf-8")
    cursor += model_len
    pre = payload[cursor:cursor + pre_len].decode("utf-8")
    cursor += pre_len
    tokens = []
    token_types = []
    for _ in range(vocab_count):
        length, token_type = struct.unpack_from("<II", payload, cursor)
        cursor += 8
        tokens.append(payload[cursor:cursor + length].decode("utf-8"))
        token_types.append(token_type)
        cursor += length
    merges = []
    for _ in range(merge_count):
        left_len, right_len = struct.unpack_from("<II", payload, cursor)
        cursor += 8
        left = payload[cursor:cursor + left_len].decode("utf-8")
        cursor += left_len
        right = payload[cursor:cursor + right_len].decode("utf-8")
        cursor += right_len
        merges.append((left, right))
    assert cursor == len(payload)
    return {
        "magic": magic,
        "version": version,
        "model": model,
        "pre": pre,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "flags": flags,
        "tokens": tokens,
        "token_types": token_types,
        "merges": merges,
    }


def test_exporter_writes_validated_qxt2_sidecar(tmp_path):
    gguf = tmp_path / "tokenizer.gguf"
    qxt = tmp_path / "tokenizer.qxt"
    tokens, merges, token_types = make_tokenizer_gguf(gguf)
    subprocess.run([sys.executable, str(EXPORTER), "--gguf", str(gguf), "--out", str(qxt)], check=True)
    got = read_qxt(qxt)
    assert got["magic"] == b"QXT2"
    assert got["version"] == 2
    assert got["model"] == "gpt2"
    assert got["pre"] == "qwen2"
    assert got["bos_id"] == 10
    assert got["eos_id"] == 10
    assert got["flags"] == 0
    assert got["tokens"] == tokens
    assert got["token_types"] == token_types
    assert got["merges"] == [tuple(item.split(" ", 1)) for item in merges]


@pytest.mark.parametrize("kwargs", [
    {"bos_writer": metadata_string, "bos_value": "10"},
    {"add_bos_writer": metadata_string, "add_bos_value": "false"},
    {"token_type_element": 6},
])
def test_exporter_rejects_wrong_gguf_metadata_types(tmp_path, kwargs):
    gguf = tmp_path / "wrong-type.gguf"
    qxt = tmp_path / "wrong-type.qxt"
    make_tokenizer_gguf(gguf, **kwargs)
    result = subprocess.run(
        [sys.executable, str(EXPORTER), "--gguf", str(gguf), "--out", str(qxt)],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "type" in result.stderr.lower()
    assert not qxt.exists()


def test_qxt2_loader_is_fail_closed(tmp_path):
    if not EXE.exists():
        raise AssertionError("build/qxqxf.exe is required")
    gguf = tmp_path / "tokenizer.gguf"
    qxt = tmp_path / "tokenizer.qxt"
    make_tokenizer_gguf(gguf)
    subprocess.run([sys.executable, str(EXPORTER), "--gguf", str(gguf), "--out", str(qxt)], check=True)
    payload = json.loads(subprocess.check_output([str(EXE), "tokenizer-inspect", "--tokenizer", str(qxt)], text=True))
    assert payload == {
        "format": "QXT2",
        "version": 2,
        "model": "gpt2",
        "pre": "qwen2",
        "vocab_count": 11,
        "merge_count": 4,
        "bos_token_id": 10,
        "eos_token_id": 10,
        "add_bos": False,
        "add_eos": False,
        "checksum_verified": True,
    }
    corrupted = tmp_path / "corrupted.qxt"
    raw = bytearray(qxt.read_bytes())
    raw[-1] ^= 1
    corrupted.write_bytes(raw)
    bad = subprocess.run([str(EXE), "tokenizer-inspect", "--tokenizer", str(corrupted)], text=True, capture_output=True)
    assert bad.returncode != 0
    assert "checksum mismatch" in bad.stderr
    truncated = tmp_path / "truncated.qxt"
    truncated.write_bytes(qxt.read_bytes()[:-7])
    short = subprocess.run([str(EXE), "tokenizer-inspect", "--tokenizer", str(truncated)], text=True, capture_output=True)
    assert short.returncode != 0
    assert "size mismatch" in short.stderr

    semantic = tmp_path / "semantic-corruption.qxt"
    raw = bytearray(qxt.read_bytes())
    first_token_type = HEADER.size + len(b"gpt2qwen2") + 4
    struct.pack_into("<I", raw, first_token_type, 0)
    struct.pack_into("<Q", raw, 40, fnv1a64(raw[HEADER.size:]))
    semantic.write_bytes(raw)
    invalid = subprocess.run([str(EXE), "tokenizer-inspect", "--tokenizer", str(semantic)], text=True, capture_output=True)
    assert invalid.returncode != 0
    assert "invalid tokenizer token entry" in invalid.stderr

    invalid_utf8 = tmp_path / "invalid-utf8.qxt"
    raw = bytearray(qxt.read_bytes())
    first_token_text = HEADER.size + len(b"gpt2qwen2") + 8
    raw[first_token_text] = 0xFF
    struct.pack_into("<Q", raw, 40, fnv1a64(raw[HEADER.size:]))
    invalid_utf8.write_bytes(raw)
    bad_utf8 = subprocess.run([str(EXE), "tokenizer-inspect", "--tokenizer", str(invalid_utf8)], text=True, capture_output=True)
    assert bad_utf8.returncode != 0
    assert "UTF-8" in bad_utf8.stderr


def test_synthetic_qxt2_bpe_special_and_decode_work_without_model(tmp_path):
    gguf = tmp_path / "tokenizer.gguf"
    qxt = tmp_path / "tokenizer.qxt"
    make_tokenizer_gguf(gguf)
    subprocess.run([sys.executable, str(EXPORTER), "--gguf", str(gguf), "--out", str(qxt)], check=True)
    plain = tmp_path / "plain.txt"
    plain.write_text("Hello", encoding="utf-8")
    encoded = json.loads(subprocess.check_output([
        str(EXE), "tokenizer-encode", "--tokenizer", str(qxt), "--text-file", str(plain)
    ], text=True))
    assert encoded["token_ids"] == [7]
    decoded = json.loads(subprocess.check_output([
        str(EXE), "tokenizer-decode", "--tokenizer", str(qxt), "--ids", "7"
    ], text=True))
    assert decoded["text"] == "Hello"
    special = tmp_path / "special.txt"
    special.write_text("<|im_start|>", encoding="utf-8")
    special_ids = json.loads(subprocess.check_output([
        str(EXE), "tokenizer-encode", "--tokenizer", str(qxt), "--text-file", str(special), "--parse-special"
    ], text=True))
    assert special_ids["token_ids"] == [10]
    rendered = json.loads(subprocess.check_output([
        str(EXE), "tokenizer-decode", "--tokenizer", str(qxt), "--ids", "10", "--special"
    ], text=True))
    assert rendered["text"] == "<|im_start|>"

    byte_gguf = tmp_path / "byte-tokenizer.gguf"
    byte_qxt = tmp_path / "byte-tokenizer.qxt"
    make_tokenizer_gguf(byte_gguf, extra_tokens=("ÿ",))
    subprocess.run([sys.executable, str(EXPORTER), "--gguf", str(byte_gguf), "--out", str(byte_qxt)], check=True)
    invalid_json = subprocess.run(
        [str(EXE), "tokenizer-decode", "--tokenizer", str(byte_qxt), "--ids", "11"],
        capture_output=True,
    )
    assert invalid_json.returncode != 0
    assert b"decoded output is not valid UTF-8" in invalid_json.stderr


def test_llama_cpp_oracle_fixture_has_reproducible_provenance():
    fixture = json.loads(ORACLE_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["oracle"] == "llama-tokenize"
    assert len(fixture["llama_cpp_commit"]) == 40
    assert len(fixture["source_model_sha256"]) == 64
    assert fixture["source_model_size"] == 10865578560
    assert len(fixture["cases"]) == 4


@pytest.fixture(scope="module")
def real_qxt(tmp_path_factory):
    if not REAL_GGUF.exists():
        pytest.skip("real Qwen3 GGUF is not available")
    output = tmp_path_factory.mktemp("qwen3-tokenizer") / "Qwen3-30B-A3B.qxt"
    subprocess.run([sys.executable, str(EXPORTER), "--gguf", str(REAL_GGUF), "--out", str(output)], check=True)
    return output


def test_qwen3_encode_decode_matches_llama_cpp_goldens(real_qxt, tmp_path):
    fixture = json.loads(ORACLE_FIXTURE.read_text(encoding="utf-8"))
    if not LLAMA_TOKENIZE.exists():
        pytest.fail("real GGUF is present but llama-tokenize is unavailable")
    oracle_commit = subprocess.check_output(["git", "-C", str(LLAMA_CPP), "rev-parse", "HEAD"], text=True).strip()
    assert oracle_commit == fixture["llama_cpp_commit"]
    assert REAL_GGUF.stat().st_size == fixture["source_model_size"]
    digest = hashlib.sha256()
    with REAL_GGUF.open("rb") as model:
        for chunk in iter(lambda: model.read(8 << 20), b""):
            digest.update(chunk)
    assert digest.hexdigest() == fixture["source_model_sha256"]
    for index, case in enumerate(fixture["cases"]):
        text = case["text"]
        expected = case["token_ids"]
        parse_special = case["parse_special"]
        prompt = tmp_path / f"prompt-{index}.txt"
        prompt.write_text(text, encoding="utf-8", newline="")
        oracle_command = [
            str(LLAMA_TOKENIZE), "-m", str(REAL_GGUF), "-f", str(prompt),
            *fixture["oracle_args"],
        ]
        if not parse_special:
            oracle_command.append("--no-parse-special")
        oracle = subprocess.check_output(oracle_command, text=True)
        assert json.loads(oracle) == expected
        command = [str(EXE), "tokenizer-encode", "--tokenizer", str(real_qxt), "--text-file", str(prompt)]
        if parse_special:
            command.append("--parse-special")
        encoded = json.loads(subprocess.check_output(command, text=True))
        assert encoded["token_ids"] == expected
        assert encoded["input_bytes"] == len(text.encode("utf-8"))
        decoded = json.loads(subprocess.check_output([
            str(EXE), "tokenizer-decode", "--tokenizer", str(real_qxt), "--ids", ",".join(map(str, expected)), "--special"
        ], text=True))
        assert decoded["text"] == text
        assert decoded["utf8_bytes"] == len(text.encode("utf-8"))


def test_qwen3_tokenizer_rejects_invalid_inputs(real_qxt, tmp_path):
    oversized = tmp_path / "oversized.txt"
    oversized.write_bytes(b"a" * 4097)
    too_large = subprocess.run([str(EXE), "tokenizer-encode", "--tokenizer", str(real_qxt), "--text-file", str(oversized)], text=True, capture_output=True)
    assert too_large.returncode != 0
    assert "input exceeds 4096-byte limit" in too_large.stderr
    invalid_utf8 = tmp_path / "invalid-utf8.txt"
    invalid_utf8.write_bytes(b"ok\xff")
    bad_utf8 = subprocess.run([str(EXE), "tokenizer-encode", "--tokenizer", str(real_qxt), "--text-file", str(invalid_utf8)], text=True, capture_output=True)
    assert bad_utf8.returncode != 0
    assert "input is not valid UTF-8" in bad_utf8.stderr
    bad_id = subprocess.run([str(EXE), "tokenizer-decode", "--tokenizer", str(real_qxt), "--ids", "151936"], text=True, capture_output=True)
    assert bad_id.returncode != 0
    assert "token id out of range" in bad_id.stderr

    if REAL_QXF.exists():
        altered = tmp_path / "wrong-model-tokenizer.qxt"
        raw = bytearray(real_qxt.read_bytes())
        first_token_text = HEADER.size + len(b"gpt2qwen2") + 8
        raw[first_token_text] = ord("?")
        struct.pack_into("<Q", raw, 40, fnv1a64(raw[HEADER.size:]))
        altered.write_bytes(raw)
        prompt = tmp_path / "fingerprint-prompt.txt"
        prompt.write_text("Hello", encoding="utf-8")
        mismatch = subprocess.run([
            str(EXE), "prompt-state-loop-probe", "--in", str(REAL_QXF), "--tokenizer", str(altered),
            "--text-file", str(prompt), "--generate", "0", "--full-moe", "--final-head",
        ], text=True, capture_output=True)
        assert mismatch.returncode != 0
        assert "tokenizer fingerprint does not match Qwen3-30B-A3B" in mismatch.stderr


def test_tokenized_prompt_prefills_verified_greedy_loop(real_qxt, tmp_path):
    if not REAL_QXF.exists():
        pytest.skip("real Qwen3 QXF is not available")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Hello!", encoding="utf-8", newline="")
    payload = json.loads(subprocess.check_output([
        str(EXE), "prompt-state-loop-probe",
        "--in", str(REAL_QXF),
        "--tokenizer", str(real_qxt),
        "--text-file", str(prompt),
        "--generate", "2",
        "--layers", "48",
        "--ctx", "4",
        "--kv", "int8",
        "--temperature", "0",
        "--seed", "7",
        "--full-moe",
        "--final-head",
        "--top-n", "5",
    ], text=True))
    assert payload["prompt_token_ids"] == [9707, 0]
    assert payload["prompt_token_count"] == 2
    assert payload["generation_steps"] == 2
    assert payload["steps"] == 3
    assert [step["position"] for step in payload["tokens"]] == [0, 1, 2]
    assert [step["input_token"] for step in payload["tokens"][:2]] == [9707, 0]
    assert [step["phase"] for step in payload["tokens"]] == ["prefill", "generate", "generate"]
    assert "final_head" not in payload["tokens"][0]
    assert payload["tokens"][1]["selected_token"] == payload["tokens"][2]["input_token"]
    assert payload["tokens"][1]["final_head"]["logits_computed"] == 151936
    assert payload["tokens"][2]["final_head"]["logits_computed"] == 151936
    assert payload["layers_run"] == 144
    assert payload["kv_appends"] == 144
    assert payload["cache_readback_ok"] is True
