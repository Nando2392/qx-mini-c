import json
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _fnv1a64(data):
    value = 1469598103934665603
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def _make_compact_valid_embedding(qxf):
    """Replace the intentionally partial embedding with eight complete Q4_K rows."""
    raw = bytearray(qxf.read_bytes())
    dir_offset = struct.unpack_from("<Q", raw, 24)[0]
    tensor_count = struct.unpack_from("<I", raw, 16)[0]
    entry_size = 208
    source_row = None
    for index in range(tensor_count):
        entry = dir_offset + index * entry_size
        name = bytes(raw[entry:entry + 96]).split(b"\0", 1)[0]
        if name == b"blk.0.attn_q.weight":
            source_offset = struct.unpack_from("<Q", raw, entry + 144)[0]
            source_row = bytes(raw[source_offset:source_offset + 8 * 144])
            break
    assert source_row is not None and len(source_row) == 8 * 144
    first = bytearray(raw[dir_offset:dir_offset + entry_size])
    entries_end = dir_offset + tensor_count * entry_size
    raw[dir_offset:entries_end - entry_size] = raw[dir_offset + entry_size:entries_end]

    token_offset = (len(raw) + 4095) & ~4095
    raw.extend(b"\0" * (token_offset - len(raw)))
    token_data = bytearray(source_row * 8)
    raw.extend(token_data)

    struct.pack_into("<I", first, 104, 2)
    struct.pack_into("<4Q", first, 112, 2048, 8, 0, 0)
    struct.pack_into("<Q", first, 144, token_offset)
    struct.pack_into("<Q", first, 152, len(token_data))
    struct.pack_into("<Q", first, 168, _fnv1a64(token_data))
    raw[entries_end - entry_size:entries_end] = first

    struct.pack_into("<I", raw, 88, 8)
    struct.pack_into("<Q", raw, 48, _fnv1a64(raw[56:108]))
    struct.pack_into("<Q", raw, 40, len(raw))
    qxf.write_bytes(raw)


def test_synthetic_gguf_inspect_if_built(tmp_path):
    exe = ROOT / "build" / "qxqxf.exe"
    if not exe.exists():
        return
    gguf = tmp_path / "mini.gguf"
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "make_synthetic_gguf.py"), "--out", str(gguf)])
    summary = subprocess.check_output([str(exe), "inspect-gguf", "--in", str(gguf)], text=True)
    data = json.loads(summary)
    assert data["magic"] == "GGUF"
    assert data["version"] == 3
    assert data["tensor_count"] == 14
    assert data["metadata_kv_count"] >= 11
    assert data["alignment"] == 32
    assert data["architecture"] == "qwen3moe"
    assert data["qwen3_block_count"] == 48
    assert data["qwen3_expert_count"] == 128
    assert data["qwen3_expert_used_count"] == 8
    assert data["qwen3_embedding_length"] == 2048
    assert data["qwen3_attention_head_count_kv"] == 4
    assert data["data_offset"] % 32 == 0
    assert data["first_tensors"][0]["name"] == "token_embd.weight"
    assert data["first_tensors"][1]["name"] == "blk.0.attn_q.weight"
    assert data["first_tensors"][3]["name"] == "blk.0.attn_v.weight"


def test_create_qxf_from_synthetic_gguf_if_built(tmp_path):
    exe = ROOT / "build" / "qxqxf.exe"
    if not exe.exists():
        return
    gguf = tmp_path / "mini.gguf"
    qxf = tmp_path / "mini.qxf"
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "make_synthetic_gguf.py"), "--out", str(gguf)])
    subprocess.check_call([
        str(exe), "create-from-gguf", "--in", str(gguf), "--model", "qwen3-30b-a3b", "--quant", "q2", "--out", str(qxf)
    ])
    summary = subprocess.check_output([str(exe), "inspect", "--in", str(qxf)], text=True)
    data = json.loads(summary)
    assert data["model_type"] == "qwen3_moe"
    assert data["quant_type"] == "q2"
    assert data["tensor_count"] == 18771


def test_create_qxf_tensor_copy_from_synthetic_gguf_if_built(tmp_path):
    exe = ROOT / "build" / "qxqxf.exe"
    if not exe.exists():
        return
    gguf = tmp_path / "mini.gguf"
    qxf = tmp_path / "mini-copy.qxf"
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "make_synthetic_gguf.py"), "--out", str(gguf)])
    subprocess.check_call([
        str(exe), "create-from-gguf-copy", "--in", str(gguf), "--model", "qwen3-30b-a3b", "--quant", "q2", "--out", str(qxf)
    ])
    summary = subprocess.check_output([str(exe), "inspect", "--in", str(qxf)], text=True)
    data = json.loads(summary)
    assert data["model_type"] == "qwen3_moe"
    assert data["quant_type"] == "q2"
    assert data["tensor_count"] == 14
    assert data["data_offset"] == 8192
    assert data["file_size"] > data["data_offset"]
    assert qxf.stat().st_size >= data["file_size"] - 4096
    tensor = subprocess.check_output([str(exe), "inspect-tensor", "--in", str(qxf), "--name", "token_embd.weight"], text=True)
    tdata = json.loads(tensor)
    assert tdata["name"] == "token_embd.weight"
    assert tdata["byte_size"] == 1152
    assert tdata["ggml_type"] == 12
    verify = subprocess.check_output([str(exe), "verify-qxf", "--in", str(qxf), "--max", "14"], text=True)
    vdata = json.loads(verify)
    assert vdata["verified"] is True
    assert vdata["checked"] == 14
    expert = subprocess.check_output([str(exe), "expert-index", "--in", str(qxf)], text=True)
    edata = json.loads(expert)
    assert edata["model_type"] == "qwen3_moe"
    assert edata["layers"] == 48
    assert edata["experts_per_layer"] == 128
    assert edata["router_tensors"] == 2
    assert edata["packed_expert_tensors"] == 6
    assert edata["complete_layers"] == 2
    coverage = subprocess.check_output([str(exe), "expert-quant-coverage", "--in", str(qxf)], text=True)
    qcov = json.loads(coverage)
    assert qcov["probe"] == "expert_quant_coverage"
    assert qcov["complete_layers"] == 2
    assert qcov["layers"][0]["gate_ggml_type"] == 17
    assert qcov["layers"][1]["gate_ggml_type"] == 22
    assert qcov["layers"][1]["down_decoder"] == "IQ4_XS"
    plan = subprocess.check_output([str(exe), "expert-plan", "--in", str(qxf), "--vram-gib", "2.0", "--ram-gib", "6.5"], text=True)
    pdata = json.loads(plan)
    assert pdata["experts_per_token"] == 8
    assert pdata["vram_expert_slots"] > 0
    assert pdata["ram_expert_slots"] > pdata["vram_expert_slots"]
    sl = subprocess.check_output([str(exe), "expert-slice", "--in", str(qxf), "--layer", "0", "--expert", "0"], text=True)
    sdata = json.loads(sl)
    assert sdata["layer"] == 0
    assert sdata["expert"] == 0
    assert sdata["experts_per_layer"] == 128
    assert sdata["slice_exact"] is True
    assert sdata["slices"]["gate"]["tensor"] == "blk.0.ffn_gate_exps.weight"
    load = subprocess.check_output([str(exe), "expert-load", "--in", str(qxf), "--layer", "0", "--expert", "0", "--kind", "gate"], text=True)
    ldata = json.loads(load)
    assert ldata["loaded"] is True
    assert ldata["layer"] == 0
    assert ldata["expert"] == 0
    assert ldata["kind"] == "gate"
    assert ldata["byte_size"] > 0
    assert ldata["checksum"] > 0
    demo = subprocess.check_output([str(exe), "cache-demo", "--in", str(qxf), "--slots", "2", "--sequence", "0:0:gate,0:0:gate,0:1:gate,0:0:gate"], text=True)
    ddata = json.loads(demo)
    assert ddata["requests"] == 4
    assert ddata["hits"] == 2
    assert ddata["misses"] == 2
    assert ddata["slots"] == 2
    bench = subprocess.check_output([str(exe), "bench-expert-load", "--in", str(qxf), "--iters", "1", "--kind", "gate"], text=True)
    bdata = json.loads(bench)
    assert bdata["loads"] == 1
    assert bdata["bytes"] > 0
    assert bdata["mib_per_sec"] > 0
    assert bdata["avg_ms"] >= 0
    run = subprocess.check_output([str(exe), "cache-run", "--in", str(qxf), "--slots", "2", "--sequence", "0:0:gate,0:0:gate,0:1:gate,0:0:gate"], text=True)
    rdata = json.loads(run)
    assert rdata["requests"] == 4
    assert rdata["hits"] == 2
    assert rdata["misses"] == 2
    assert rdata["slots"] == 2
    assert rdata["bytes_loaded"] > 0
    assert rdata["resident_bytes"] == rdata["bytes_loaded"]
    seq_file = tmp_path / "seq.txt"
    seq_file.write_text("0:0:gate,0:0:gate,0:1:gate,0:0:gate")
    run_file = subprocess.check_output([str(exe), "cache-run", "--in", str(qxf), "--slots", "2", "--sequence-file", str(seq_file)], text=True)
    assert json.loads(run_file)["hits"] == 2
    trace = subprocess.check_output([str(exe), "route-trace", "--layers", "2", "--experts", "4", "--top-k", "2", "--tokens", "2", "--seed", "7"], text=True)
    tdata = json.loads(trace)
    assert tdata["layers"] == 2
    assert tdata["experts"] == 4
    assert tdata["top_k"] == 2
    assert tdata["tokens"] == 2
    assert tdata["requests"] == 24
    assert tdata["sequence"].count(",") == 23
    reused = subprocess.check_output([str(exe), "route-trace", "--layers", "2", "--experts", "4", "--top-k", "2", "--tokens", "2", "--seed", "7", "--reuse-pct", "100"], text=True)
    udata = json.loads(reused)
    assert udata["reuse_pct"] == 100
    assert udata["requests"] == 24
    first_token = udata["sequence"].split(",")[:12]
    second_token = udata["sequence"].split(",")[12:]
    assert first_token == second_token
    expert_run = subprocess.check_output([str(exe), "cache-run-expert", "--in", str(qxf), "--slots", "2", "--sequence", "0:0:gate,0:0:up,0:0:down,0:0:gate,0:0:up,0:0:down,0:1:gate,0:1:up,0:1:down,0:0:gate,0:0:up,0:0:down"], text=True)
    xr = json.loads(expert_run)
    assert xr["requests"] == 12
    assert xr["expert_requests"] == 4
    assert xr["hits"] == 2
    assert xr["misses"] == 2
    assert xr["slots"] == 2
    assert xr["bytes_loaded"] > 0
    complete_plan = subprocess.check_output([str(exe), "expert-cache-plan-complete", "--in", str(qxf), "--vram-gib", "2.0", "--ram-gib", "6.5", "--top-k", "8"], text=True)
    cp = json.loads(complete_plan)
    assert cp["cache_unit"] == "full_expert"
    assert cp["top_k"] == 8
    assert cp["working_set_experts_per_token"] == 384
    assert cp["vram_expert_slots"] > 0
    assert cp["ram_expert_slots"] > cp["vram_expert_slots"]
    runtime = subprocess.check_output([str(exe), "runtime-plan", "--in", str(qxf), "--ctx", "4096", "--kv", "int8", "--vram-gib", "4.2", "--ram-gib", "11.0", "--hot-vram-gib", "2.0", "--hot-ram-gib", "6.5", "--top-k", "8"], text=True)
    rp = json.loads(runtime)
    assert rp["model_type"] == "qwen3_moe"
    assert rp["ctx_tokens"] == 4096
    assert rp["kv_format"] == "int8"
    assert rp["cache_unit"] == "full_expert"
    assert rp["working_set_experts_per_token"] == 384
    assert rp["total_model_gib"] > 0
    assert rp["kv_gib"] > 0
    assert rp["feasible"] in (True, False)
    malformed_embedding = subprocess.run([str(exe), "token-embedding", "--in", str(qxf), "--token-id", "2"], text=True, capture_output=True)
    assert malformed_embedding.returncode != 0
    assert "tensor byte size is not divisible by row count" in malformed_embedding.stderr
    _make_compact_valid_embedding(qxf)
    emb = subprocess.check_output([str(exe), "token-embedding", "--in", str(qxf), "--token-id", "2"], text=True)
    ep = json.loads(emb)
    assert ep["token_id"] == 2
    assert ep["tensor"] == "token_embd.weight"
    assert ep["row_byte_size"] > 0
    assert ep["offset"] >= data["data_offset"]
    schedule = subprocess.check_output([str(exe), "forward-schedule", "--in", str(qxf), "--token-id", "2", "--top-k", "2"], text=True)
    fs = json.loads(schedule)
    assert fs["token_id"] == 2
    assert fs["layers"] == 48
    assert fs["top_k"] == 2
    assert fs["embedding"]["tensor"] == "token_embd.weight"
    assert fs["steps_per_layer"] >= 4
    assert fs["mock_forward"] is True
    qb = subprocess.check_output([str(exe), "quant-block", "--in", str(qxf), "--name", "token_embd.weight", "--block", "0"], text=True)
    qbd = json.loads(qb)
    assert qbd["tensor"] == "token_embd.weight"
    assert qbd["block_index"] == 0
    assert qbd["block_offset"] >= data["data_offset"]
    assert qbd["block_byte_size"] > 0
    assert qbd["checksum"] > 0
    assert qbd["ggml_type"] == 12
    mv = subprocess.check_output([str(exe), "matvec-stub", "--in", str(qxf), "--name", "token_embd.weight", "--rows", "2"], text=True)
    mvd = json.loads(mv)
    assert mvd["stub"] is True
    assert mvd["tensor"] == "token_embd.weight"
    assert mvd["rows"] == 2
    assert mvd["bytes_read"] > 0
    assert mvd["checksum_mix"] > 0
    dec = subprocess.check_output([str(exe), "decode-block", "--in", str(qxf), "--name", "blk.0.ffn_gate_exps.weight", "--block", "0"], text=True)
    dd = json.loads(dec)
    assert dd["tensor"] == "blk.0.ffn_gate_exps.weight"
    assert dd["ggml_type"] == 17
    assert dd["decoder"] == "IQ2_XS"
    assert dd["decoded"] is True
    assert dd["values"] == 256
    assert dd["block_byte_size"] == 74
    assert "sum" in dd
    decq4 = subprocess.check_output([str(exe), "decode-block", "--in", str(qxf), "--name", "token_embd.weight", "--block", "0"], text=True)
    dq4 = json.loads(decq4)
    assert dq4["tensor"] == "token_embd.weight"
    assert dq4["ggml_type"] == 12
    assert dq4["decoder"] == "Q4_K"
    assert dq4["decoded"] is True
    assert dq4["values"] == 256
    assert dq4["block_byte_size"] == 144
    assert "sum" in dq4
    rms = subprocess.check_output([str(exe), "rmsnorm-probe", "--in", str(qxf), "--token-id", "2", "--norm", "blk.0.attn_norm.weight", "--seed", "7"], text=True)
    rn = json.loads(rms)
    assert rn["probe"] == "rmsnorm"
    assert rn["token_id"] == 2
    assert rn["embedding_decoder"] == "Q4_K"
    assert rn["norm_tensor"] == "blk.0.attn_norm.weight"
    assert rn["norm_ggml_type"] == 0
    assert rn["values"] > 0
    assert rn["rms"] > 0
    assert rn["normalized_probe"] != 0
    attn = subprocess.check_output([str(exe), "attention-probe", "--in", str(qxf), "--layer", "0", "--blocks", "2", "--seed", "7", "--ctx", "64", "--kv", "int8", "--cache-write"], text=True)
    ap = json.loads(attn)
    assert ap["probe"] == "attention"
    assert ap["layer"] == 0
    assert ap["q"]["tensor"] == "blk.0.attn_q.weight"
    assert ap["k"]["tensor"] == "blk.0.attn_k.weight"
    assert ap["v"]["tensor"] == "blk.0.attn_v.weight"
    assert ap["o"]["tensor"] == "blk.0.attn_output.weight"
    assert ap["q"]["decoder"] == "Q4_K"
    assert ap["values"] > 0
    assert ap["attention_score_probe"] != 0
    assert ap["attention_output_probe"] != 0
    assert ap["kv_cache"]["enabled"] is True
    assert ap["kv_cache"]["kv_format"] == "int8"
    assert ap["kv_cache"]["bytes_per_token_per_layer"] == 1024
    assert ap["kv_buffer"]["enabled"] is True
    assert ap["kv_buffer"]["allocated"] is True
    assert ap["kv_buffer"]["readback_ok"] is True
    kv = subprocess.check_output([str(exe), "kv-cache-probe", "--in", str(qxf), "--ctx", "4096", "--kv", "int8", "--token", "2", "--layer", "0", "--head", "0"], text=True)
    kp = json.loads(kv)
    assert kp["probe"] == "kv_cache"
    assert kp["kv_format"] == "int8"
    assert kp["ctx_tokens"] == 4096
    assert kp["layers"] == 48
    assert kp["kv_heads"] == 4
    assert kp["head_dim"] == 128
    assert kp["bytes_per_value"] == 1
    assert kp["bytes_per_token_per_layer"] == 1024
    assert kp["total_bytes"] == 201326592
    assert kp["k_offset"] < kp["v_offset"]
    kb = subprocess.check_output([str(exe), "kv-cache-buffer-probe", "--in", str(qxf), "--ctx", "64", "--kv", "int8", "--token", "7", "--layer", "1", "--head", "2", "--seed", "7"], text=True)
    kbp = json.loads(kb)
    assert kbp["probe"] == "kv_cache_buffer"
    assert kbp["allocated"] is True
    assert kbp["kv_format"] == "int8"
    assert kbp["ctx_tokens"] == 64
    assert kbp["total_bytes"] == 3145728
    assert kbp["write_bytes"] == 256
    assert kbp["readback_ok"] is True
    assert kbp["k_checksum"] != kbp["v_checksum"]
    ac = subprocess.check_output([str(exe), "attention-cache-probe", "--in", str(qxf), "--ctx", "64", "--kv", "int8", "--layer", "0", "--tokens", "2", "--blocks", "2", "--seed", "7"], text=True)
    acp = json.loads(ac)
    assert acp["probe"] == "attention_cache"
    assert acp["kv_format"] == "int8"
    assert acp["tokens_written"] == 2
    assert acp["current_token"] == 1
    assert acp["attend_token"] == 0
    assert acp["cache_readback_ok"] is True
    assert acp["k_cache_checksum"] != acp["v_cache_checksum"]
    assert acp["attention_score_from_cache"] != 0
    assert acp["attention_output_from_cache"] != 0
    sm = subprocess.check_output([str(exe), "attention-softmax-probe", "--in", str(qxf), "--ctx", "64", "--kv", "int8", "--layer", "0", "--tokens", "4", "--blocks", "2", "--seed", "7"], text=True)
    sp = json.loads(sm)
    assert sp["probe"] == "attention_softmax"
    assert sp["tokens_written"] == 4
    assert sp["current_token"] == 3
    assert sp["attend_count"] == 3
    assert sp["causal_mask"] is True
    assert abs(sp["softmax_sum"] - 1.0) < 1e-6
    assert len(sp["scores"]) == 3
    assert len(sp["weights"]) == 3
    assert sp["cache_readback_ok"] is True
    assert sp["context_from_softmax"] != 0
    assert sp["attention_output_from_softmax"] != 0
    av = subprocess.check_output([str(exe), "attention-vector-probe", "--in", str(qxf), "--ctx", "64", "--kv", "int8", "--layer", "0", "--tokens", "4", "--dims", "16", "--seed", "7"], text=True)
    vp = json.loads(av)
    assert vp["probe"] == "attention_vector"
    assert vp["dims"] == 16
    assert vp["head_dim"] == 128
    assert vp["tokens_written"] == 4
    assert vp["attend_count"] == 3
    assert abs(vp["softmax_sum"] - 1.0) < 1e-6
    assert len(vp["scores"]) == 3
    assert len(vp["weights"]) == 3
    assert len(vp["context_first8"]) == 8
    assert vp["cache_readback_ok"] is True
    assert vp["context_l2"] > 0
    assert vp["attention_output_probe"] != 0
    mh = subprocess.check_output([str(exe), "attention-multihead-probe", "--in", str(qxf), "--ctx", "64", "--kv", "int8", "--layer", "0", "--tokens", "4", "--heads", "4", "--dims", "16", "--seed", "7"], text=True)
    mp = json.loads(mh)
    assert mp["probe"] == "attention_multihead"
    assert mp["heads_run"] == 4
    assert mp["q_heads"] == 32
    assert mp["kv_heads"] == 4
    assert mp["dims"] == 16
    assert mp["attend_count"] == 3
    assert len(mp["head_outputs"]) == 4
    assert mp["all_softmax_ok"] is True
    assert mp["cache_readback_ok"] is True
    assert mp["combined_output_probe"] != 0
    dec3 = subprocess.check_output([str(exe), "decode-block", "--in", str(qxf), "--name", "blk.0.ffn_down_exps.weight", "--block", "0"], text=True)
    d3 = json.loads(dec3)
    assert d3["tensor"] == "blk.0.ffn_down_exps.weight"
    assert d3["ggml_type"] == 18
    assert d3["decoder"] == "IQ3_XXS"
    assert d3["decoded"] is True
    assert d3["values"] == 256
    assert d3["block_byte_size"] == 98
    dec2s = subprocess.check_output([str(exe), "decode-block", "--in", str(qxf), "--name", "blk.1.ffn_gate_exps.weight", "--block", "0"], text=True)
    d2s = json.loads(dec2s)
    assert d2s["ggml_type"] == 22
    assert d2s["decoder"] == "IQ2_S"
    assert d2s["values"] == 256
    assert d2s["block_byte_size"] == 82
    dec3s = subprocess.check_output([str(exe), "decode-block", "--in", str(qxf), "--name", "blk.1.ffn_down_exps.weight", "--block", "0"], text=True)
    d3s = json.loads(dec3s)
    assert d3s["ggml_type"] == 23
    assert d3s["decoder"] == "IQ4_XS"
    assert d3s["values"] == 256
    assert d3s["block_byte_size"] == 136
    dot = subprocess.check_output([str(exe), "block-dot", "--in", str(qxf), "--name", "blk.0.ffn_gate_exps.weight", "--block", "0", "--seed", "7"], text=True)
    bd = json.loads(dot)
    assert bd["tensor"] == "blk.0.ffn_gate_exps.weight"
    assert bd["ggml_type"] == 17
    assert bd["decoder"] == "IQ2_XS"
    assert bd["values"] == 256
    assert bd["input_seed"] == 7
    assert bd["dot"] != 0
    dot3 = subprocess.check_output([str(exe), "block-dot", "--in", str(qxf), "--name", "blk.0.ffn_down_exps.weight", "--block", "0", "--seed", "7"], text=True)
    bd3 = json.loads(dot3)
    assert bd3["ggml_type"] == 18
    assert bd3["decoder"] == "IQ3_XXS"
    assert bd3["dot"] != 0
    row = subprocess.check_output([str(exe), "matvec-row", "--in", str(qxf), "--name", "blk.0.ffn_gate_exps.weight", "--start-block", "0", "--blocks", "2", "--seed", "7"], text=True)
    rd = json.loads(row)
    assert rd["tensor"] == "blk.0.ffn_gate_exps.weight"
    assert rd["ggml_type"] == 17
    assert rd["decoder"] == "IQ2_XS"
    assert rd["start_block"] == 0
    assert rd["blocks"] == 2
    assert rd["values"] == 512
    assert rd["dot"] != 0
    row3 = subprocess.check_output([str(exe), "matvec-row", "--in", str(qxf), "--name", "blk.0.ffn_down_exps.weight", "--start-block", "0", "--blocks", "2", "--seed", "7"], text=True)
    rd3 = json.loads(row3)
    assert rd3["ggml_type"] == 18
    assert rd3["decoder"] == "IQ3_XXS"
    assert rd3["values"] == 512
    assert rd3["dot"] != 0
    erow = subprocess.check_output([str(exe), "expert-row", "--in", str(qxf), "--layer", "0", "--expert", "0", "--kind", "gate", "--start-block", "0", "--blocks", "2", "--seed", "7"], text=True)
    er = json.loads(erow)
    assert er["layer"] == 0
    assert er["expert"] == 0
    assert er["kind"] == "gate"
    assert er["tensor"] == "blk.0.ffn_gate_exps.weight"
    assert er["decoder"] == "IQ2_XS"
    assert er["slice_start_block"] == 0
    assert er["absolute_start_block"] == 0
    assert er["values"] == 512
    assert er["dot"] != 0
    erow3 = subprocess.check_output([str(exe), "expert-row", "--in", str(qxf), "--layer", "0", "--expert", "0", "--kind", "down", "--start-block", "0", "--blocks", "2", "--seed", "7"], text=True)
    er3 = json.loads(erow3)
    assert er3["kind"] == "down"
    assert er3["decoder"] == "IQ3_XXS"
    assert er3["values"] == 512
    assert er3["dot"] != 0
    fwd = subprocess.check_output([str(exe), "expert-forward-probe", "--in", str(qxf), "--layer", "0", "--expert", "0", "--start-block", "0", "--blocks", "2", "--seed", "7"], text=True)
    fd = json.loads(fwd)
    assert fd["layer"] == 0
    assert fd["expert"] == 0
    assert fd["blocks"] == 2
    assert fd["values"] == 512
    assert fd["gate_decoder"] == "IQ2_XS"
    assert fd["up_decoder"] == "IQ2_XS"
    assert fd["down_decoder"] == "IQ3_XXS"
    assert fd["gate_dot"] != 0
    assert fd["up_dot"] != 0
    assert fd["hidden_probe"] != 0
    assert fd["down_dot"] != 0
    assert fd["projected_probe"] != 0
    fwd1 = subprocess.check_output([str(exe), "expert-forward-probe", "--in", str(qxf), "--layer", "1", "--expert", "0", "--start-block", "0", "--blocks", "2", "--seed", "7"], text=True)
    fd1 = json.loads(fwd1)
    assert fd1["gate_decoder"] == "IQ2_S"
    assert fd1["up_decoder"] == "IQ2_S"
    assert fd1["down_decoder"] == "IQ4_XS"
    assert fd1["projected_probe"] != 0
    router = subprocess.check_output([str(exe), "router-topk-probe", "--in", str(qxf), "--layer", "0", "--top-k", "2", "--blocks", "2", "--seed", "7"], text=True)
    rt = json.loads(router)
    assert rt["layer"] == 0
    assert rt["router_tensor"] == "blk.0.ffn_gate_inp.weight"
    assert rt["router_kernel"] == "F32_PREFIX_DOT"
    assert rt["top_k"] == 2
    assert len(rt["selected_experts"]) == 2
    assert rt["selected_experts"][0]["logit"] >= rt["selected_experts"][1]["logit"]
    assert rt["selected_experts"][0]["projected_probe"] != 0
    layer = subprocess.check_output([str(exe), "layer-forward-probe", "--in", str(qxf), "--layer", "0", "--top-k", "2", "--blocks", "2", "--seed", "7"], text=True)
    lf = json.loads(layer)
    assert lf["probe"] == "layer_forward"
    assert lf["layer"] == 0
    assert lf["top_k"] == 2
    assert lf["router_kernel"] == "F32_PREFIX_DOT"
    assert len(lf["selected_experts"]) == 2
    assert lf["selected_experts"][0]["logit"] >= lf["selected_experts"][1]["logit"]
    assert lf["expert_outputs_sum"] != 0
    assert lf["layer_output_probe"] != 0
    moe = subprocess.check_output([str(exe), "moe-forward-probe", "--in", str(qxf), "--layers", "1", "--top-k", "2", "--blocks", "2", "--seed", "7"], text=True)
    mf = json.loads(moe)
    assert mf["probe"] == "moe_forward"
    assert mf["layers_requested"] == 1
    assert mf["layers_run"] == 1
    assert mf["top_k"] == 2
    assert len(mf["layers"]) == 1
    assert mf["layers"][0]["layer"] == 0
    assert mf["layers"][0]["selected_experts"][0]["logit"] >= mf["layers"][0]["selected_experts"][1]["logit"]
    assert mf["moe_output_probe"] != 0
    token_fwd = subprocess.check_output([str(exe), "token-forward-probe", "--in", str(qxf), "--token-id", "2", "--layers", "2", "--top-k", "2", "--blocks", "2", "--seed", "7", "--norm", "blk.0.attn_norm.weight", "--attention-layer", "0"], text=True)
    tf = json.loads(token_fwd)
    assert tf["probe"] == "token_forward"
    assert tf["token_id"] == 2
    assert tf["embedding"]["tensor"] == "token_embd.weight"
    assert tf["embedding"]["decoder"] == "Q4_K"
    assert tf["rmsnorm"]["enabled"] is True
    assert tf["rmsnorm"]["norm_tensor"] == "blk.0.attn_norm.weight"
    assert tf["rmsnorm"]["rms"] > 0
    assert tf["attention"]["enabled"] is True
    assert tf["attention"]["layer"] == 0
    assert tf["attention"]["output_probe"] != 0
    token_mh = subprocess.check_output([str(exe), "token-forward-probe", "--in", str(qxf), "--token-id", "2", "--layers", "2", "--top-k", "2", "--blocks", "2", "--seed", "7", "--norm", "blk.0.attn_norm.weight", "--multihead-attention", "--attention-layer", "0", "--attention-heads", "4", "--attention-dims", "16"], text=True)
    tmh = json.loads(token_mh)
    assert tmh["attention"]["enabled"] is True
    assert tmh["attention"]["mode"] == "multihead"
    assert tmh["attention"]["heads_run"] == 4
    assert tmh["attention"]["dims"] == 16
    assert tmh["attention"]["output_probe"] != 0
    assert tmh["attention"]["output_probe"] != tf["attention"]["output_probe"]
    logits = subprocess.check_output([str(exe), "logits-probe", "--in", str(qxf), "--activation", "0.125", "--top-n", "3", "--scan", "32", "--seed", "7"], text=True)
    lp = json.loads(logits)
    assert lp["probe"] == "logits"
    assert lp["top_n"] == 3
    assert lp["scanned"] == 8
    assert lp["lm_head_tensor"] in ("output.weight", "lm_head.weight", "token_embd.weight")
    assert len(lp["top_tokens"]) == 3
    assert lp["top_tokens"][0]["logit"] >= lp["top_tokens"][1]["logit"]
    bounded = json.loads(subprocess.check_output([str(exe), "logits-probe", "--in", str(qxf), "--activation", "0.125", "--top-n", "32", "--scan", "32", "--seed", "7"], text=True))
    assert bounded["scanned"] == 8
    assert bounded["top_n"] == 8
    assert len(bounded["top_tokens"]) == 8
    sampler = subprocess.check_output([str(exe), "sampler-probe", "--in", str(qxf), "--activation", "0.125", "--top-k", "3", "--scan", "32", "--temperature", "0", "--seed", "7"], text=True)
    spm = json.loads(sampler)
    assert spm["probe"] == "sampler"
    assert spm["strategy"] == "argmax"
    assert spm["selected_token"] == lp["top_tokens"][0]["token"]
    assert spm["top_k"] == 3
    assert spm["temperature"] == 0
    tok_sidecar = tmp_path / "tokens.tsv"
    export = subprocess.check_output([str(exe), "tokenizer-export", "--gguf", str(gguf), "--out", str(tok_sidecar)], text=True)
    ex = json.loads(export)
    assert ex["exported"] is True
    assert ex["token_count"] == 64
    tokp = subprocess.check_output([str(exe), "tokenizer-probe", "--in", str(qxf), "--tokens", str(tok_sidecar), "--token-id", str(spm["selected_token"])], text=True)
    tpd = json.loads(tokp)
    assert tpd["probe"] == "tokenizer"
    assert tpd["token_id"] == spm["selected_token"]
    assert isinstance(tpd["piece"], str)
    assert tpd["source"] == "sidecar"
    compact_pieces = ["H", "e", "l", "o", "He", "ll", "Hell", "Hello"]
    assert tpd["piece"] == compact_pieces[spm["selected_token"]]
    sampler_t = subprocess.check_output([str(exe), "sampler-probe", "--in", str(qxf), "--activation", "0.125", "--top-k", "3", "--scan", "32", "--temperature", "0.7", "--seed", "7"], text=True)
    spt = json.loads(sampler_t)
    assert spt["strategy"] == "temperature_top_k"
    assert 0 <= spt["selected_rank"] < 3
    assert abs(spt["prob_sum"] - 1.0) < 1e-6
    token_logits = subprocess.check_output([str(exe), "token-forward-probe", "--in", str(qxf), "--token-id", "2", "--layers", "2", "--top-k", "2", "--blocks", "2", "--seed", "7", "--norm", "blk.0.attn_norm.weight", "--multihead-attention", "--attention-layer", "0", "--attention-heads", "4", "--attention-dims", "16", "--logits", "--top-n", "3"], text=True)
    tlog = json.loads(token_logits)
    assert tlog["logits"]["enabled"] is True
    assert len(tlog["logits"]["top_tokens"]) == 3
    bounded_token_logits = json.loads(subprocess.check_output([str(exe), "token-forward-probe", "--in", str(qxf), "--token-id", "2", "--layers", "2", "--top-k", "2", "--blocks", "2", "--seed", "7", "--norm", "blk.0.attn_norm.weight", "--logits", "--top-n", "32", "--sample", "--temperature", "0"], text=True))
    assert bounded_token_logits["logits"]["scanned"] == 8
    assert bounded_token_logits["logits"]["top_n"] == 8
    assert len(bounded_token_logits["logits"]["top_tokens"]) == 8
    assert bounded_token_logits["sampler"]["top_k"] == 8
    token_sample = subprocess.check_output([str(exe), "token-forward-probe", "--in", str(qxf), "--token-id", "2", "--layers", "2", "--top-k", "2", "--blocks", "2", "--seed", "7", "--norm", "blk.0.attn_norm.weight", "--multihead-attention", "--attention-layer", "0", "--attention-heads", "4", "--attention-dims", "16", "--logits", "--top-n", "3", "--sample", "--temperature", "0"], text=True)
    ts = json.loads(token_sample)
    assert ts["sampler"]["enabled"] is True
    assert ts["sampler"]["strategy"] == "argmax"
    assert ts["sampler"]["selected_token"] == ts["logits"]["top_tokens"][0]["token"]
    token_decoded = subprocess.check_output([str(exe), "token-forward-probe", "--in", str(qxf), "--token-id", "2", "--layers", "2", "--top-k", "2", "--blocks", "2", "--seed", "7", "--norm", "blk.0.attn_norm.weight", "--multihead-attention", "--attention-layer", "0", "--attention-heads", "4", "--attention-dims", "16", "--logits", "--top-n", "3", "--sample", "--temperature", "0", "--decode-token", "--tokens", str(tok_sidecar)], text=True)
    td = json.loads(token_decoded)
    assert td["decoded_token"]["enabled"] is True
    assert td["decoded_token"]["token_id"] == td["sampler"]["selected_token"]
    assert td["decoded_token"]["source"] == "sidecar"
    assert td["decoded_token"]["piece"] == compact_pieces[td["sampler"]["selected_token"]]
    gen = subprocess.check_output([str(exe), "generate-probe", "--in", str(qxf), "--tokens", str(tok_sidecar), "--prompt-token", "2", "--steps", "3", "--top-k", "3", "--scan", "32", "--temperature", "0", "--seed", "7"], text=True)
    gd = json.loads(gen)
    assert gd["probe"] == "generate"
    assert gd["prompt_token"] == 2
    assert gd["steps"] == 3
    assert len(gd["tokens"]) == 3
    assert gd["tokens"][0]["piece"] == compact_pieces[gd["tokens"][0]["token"]]
    assert gd["generated_text"] == "".join(t["piece"] for t in gd["tokens"])
    state = subprocess.check_output([str(exe), "state-loop-probe", "--in", str(qxf), "--tokens", str(tok_sidecar), "--prompt-token", "2", "--steps", "3", "--layers", "2", "--ctx", "16", "--kv", "int8", "--top-k", "3", "--scan", "32", "--temperature", "0", "--seed", "7"], text=True)
    sd = json.loads(state)
    assert sd["probe"] == "state_loop"
    assert sd["prompt_token"] == 2
    assert sd["steps"] == 3
    assert sd["layers_run"] == 6
    assert sd["kv_appends"] == 6
    assert sd["kv_format"] == "int8"
    assert sd["cache_readback_ok"] is True
    assert len(sd["tokens"]) == 3
    assert sd["tokens"][0]["layers"][0]["layer"] == 0
    assert sd["generated_text"] == "".join(t["piece"] for t in sd["tokens"])
    rstate = subprocess.check_output([str(exe), "state-loop-probe", "--in", str(qxf), "--tokens", str(tok_sidecar), "--prompt-token", "2", "--steps", "2", "--layers", "2", "--ctx", "16", "--kv", "int8", "--top-k", "3", "--scan", "32", "--temperature", "0", "--seed", "7", "--real-kv"], text=True)
    rsd = json.loads(rstate)
    assert rsd["kv_source"] == "projection_decode"
    assert rsd["kv_appends"] == 4
    assert rsd["cache_readback_ok"] is True
    assert rsd["tokens"][0]["layers"][0]["k_tensor"].endswith("attn_k.weight")
    assert rsd["tokens"][0]["layers"][0]["v_tensor"].endswith("attn_v.weight")
    assert rsd["tokens"][0]["layers"][0]["k_real_values"] > 0
    assert rsd["tokens"][0]["layers"][0]["v_real_values"] > 0
    pmv = subprocess.check_output([str(exe), "projection-matvec-probe", "--in", str(qxf), "--layer", "0", "--token-id", "2", "--rows", "4", "--dims", "64", "--kv", "int8", "--seed", "7"], text=True)
    pd = json.loads(pmv)
    assert pd["probe"] == "projection_matvec"
    assert pd["layer"] == 0
    assert pd["rows"] == 4
    assert pd["dims"] == 64
    assert pd["kv_format"] == "int8"
    assert pd["k_tensor"].endswith("attn_k.weight")
    assert pd["v_tensor"].endswith("attn_v.weight")
    assert pd["k_values"] == 4
    assert pd["v_values"] == 4
    assert pd["k_checksum"] > 0
    assert pd["projection_kernel"] in ("iq4_xs_window_dot", "quant_window_decode_dot")
    rv = subprocess.check_output([str(exe), "residual-vector-probe", "--in", str(qxf), "--token-id", "2", "--norm", "blk.0.attn_norm.weight", "--dims", "64", "--seed", "7"], text=True)
    rd = json.loads(rv)
    assert rd["probe"] == "residual_vector"
    assert rd["token_id"] == 2
    assert rd["dims"] == 64
    assert rd["source"] == "embedding_rmsnorm"
    assert rd["embedding_tensor"] == "token_embd.weight"
    assert rd["norm_tensor"] == "blk.0.attn_norm.weight"
    assert rd["values"] == 64
    assert rd["rms"] > 0
    assert rd["checksum"] > 0
    pmvr = subprocess.check_output([str(exe), "projection-matvec-probe", "--in", str(qxf), "--layer", "0", "--token-id", "2", "--rows", "4", "--dims", "64", "--kv", "int8", "--seed", "7", "--residual-vector", "--norm", "blk.0.attn_norm.weight"], text=True)
    pdr = json.loads(pmvr)
    assert pdr["residual_source"] == "embedding_rmsnorm"
    assert pdr["residual_values"] == 64
    smv = subprocess.check_output([str(exe), "state-loop-probe", "--in", str(qxf), "--tokens", str(tok_sidecar), "--prompt-token", "2", "--steps", "2", "--layers", "2", "--ctx", "16", "--kv", "int8", "--top-k", "3", "--scan", "32", "--temperature", "0", "--seed", "7", "--real-kv", "--projection-matvec"], text=True)
    smd = json.loads(smv)
    assert smd["kv_source"] == "projection_matvec"
    assert smd["tokens"][0]["layers"][0]["k_matvec_values"] > 0
    smvr = subprocess.check_output([str(exe), "state-loop-probe", "--in", str(qxf), "--tokens", str(tok_sidecar), "--prompt-token", "2", "--steps", "2", "--layers", "2", "--ctx", "16", "--kv", "int8", "--top-k", "3", "--scan", "32", "--temperature", "0", "--seed", "7", "--real-kv", "--projection-matvec", "--residual-vector", "--norm", "blk.0.attn_norm.weight"], text=True)
    smrd = json.loads(smvr)
    assert smrd["residual_source"] == "embedding_rmsnorm"
    assert smrd["tokens"][0]["residual_values"] == 64
    smc = subprocess.check_output([str(exe), "state-loop-probe", "--in", str(qxf), "--tokens", str(tok_sidecar), "--prompt-token", "2", "--steps", "2", "--layers", "2", "--ctx", "16", "--kv", "int8", "--top-k", "3", "--scan", "32", "--temperature", "0", "--seed", "7", "--real-kv", "--projection-matvec", "--residual-vector", "--residual-carry", "--residual-dims", "96", "--norm", "blk.0.attn_norm.weight"], text=True)
    scd = json.loads(smc)
    assert scd["residual_source"] == "embedding_rmsnorm_carry"
    assert scd["residual_dims"] == 96
    assert scd["tokens"][0]["residual_values"] == 96
    assert scd["tokens"][0]["layers"][0]["attention_delta"] != 0
    assert scd["tokens"][0]["layers"][0]["moe_delta"] != 0
    assert scd["tokens"][0]["residual_checksum_after"] > 0
    sn = subprocess.check_output([str(exe), "state-loop-probe", "--in", str(qxf), "--tokens", str(tok_sidecar), "--prompt-token", "2", "--steps", "2", "--layers", "2", "--ctx", "16", "--kv", "int8", "--top-k", "3", "--scan", "32", "--temperature", "0", "--seed", "7", "--real-kv", "--projection-matvec", "--residual-vector", "--residual-carry", "--numeric-deltas", "--residual-dims", "96", "--norm", "blk.0.attn_norm.weight"], text=True)
    snd = json.loads(sn)
    assert snd["delta_source"] == "numeric_probe"
    assert snd["tokens"][0]["layers"][0]["attention_delta_source"] == "numeric_attention"
    assert snd["tokens"][0]["layers"][0]["moe_delta_source"] == "numeric_moe"
    assert snd["tokens"][0]["layers"][0]["attention_delta"] != scd["tokens"][0]["layers"][0]["attention_delta"]
    assert snd["tokens"][0]["layers"][0]["moe_delta"] != scd["tokens"][0]["layers"][0]["moe_delta"]
    sv = subprocess.check_output([str(exe), "state-loop-probe", "--in", str(qxf), "--tokens", str(tok_sidecar), "--prompt-token", "2", "--steps", "2", "--layers", "2", "--ctx", "16", "--kv", "int8", "--top-k", "3", "--scan", "32", "--temperature", "0", "--seed", "7", "--real-kv", "--projection-matvec", "--residual-vector", "--residual-carry", "--numeric-deltas", "--delta-vectors", "--residual-dims", "96", "--norm", "blk.0.attn_norm.weight"], text=True)
    svd = json.loads(sv)
    assert svd["delta_source"] == "numeric_vectors"
    l0 = svd["tokens"][0]["layers"][0]
    assert l0["attention_delta_vector_values"] == 96
    assert l0["moe_delta_vector_values"] == 96
    assert l0["attention_delta_vector_l2"] > 0
    assert l0["moe_delta_vector_l2"] > 0
    assert l0["attention_delta_vector_checksum"] > 0
    assert l0["moe_delta_vector_checksum"] > 0
    sa = subprocess.check_output([str(exe), "state-loop-probe", "--in", str(qxf), "--tokens", str(tok_sidecar), "--prompt-token", "2", "--steps", "2", "--layers", "2", "--ctx", "16", "--kv", "int8", "--top-k", "3", "--scan", "32", "--temperature", "0", "--seed", "7", "--real-kv", "--projection-matvec", "--residual-vector", "--residual-carry", "--numeric-deltas", "--delta-vectors", "--attention-output-vector", "--residual-dims", "96", "--norm", "blk.0.attn_norm.weight"], text=True)
    sad = json.loads(sa)
    assert sad["delta_source"] == "attention_output_vector"
    al0 = sad["tokens"][0]["layers"][0]
    assert al0["attention_output_vector_values"] == 96
    assert al0["attention_output_vector_l2"] > 0
    assert al0["attention_output_vector_checksum"] > 0
    assert al0["attention_context_tokens"] == 1
    assert al0["attention_output_source"] == "kv_cache_partial"
    sca = subprocess.check_output([str(exe), "state-loop-probe", "--in", str(qxf), "--tokens", str(tok_sidecar), "--prompt-token", "2", "--steps", "2", "--layers", "2", "--ctx", "16", "--kv", "int8", "--top-k", "3", "--scan", "32", "--temperature", "0", "--seed", "7", "--real-kv", "--projection-matvec", "--residual-vector", "--residual-carry", "--numeric-deltas", "--delta-vectors", "--attention-output-vector", "--causal-attention", "--residual-dims", "96", "--norm", "blk.0.attn_norm.weight"], text=True)
    scad = json.loads(sca)
    assert scad["delta_source"] == "causal_attention"
    cal0 = scad["tokens"][0]["layers"][0]
    assert cal0["attention_output_source"] == "qkv_softmax_output_projection_partial"
    assert cal0["persistent_kv"] is True
    assert cal0["kv_scale_source"] == "dynamic_per_vector"
    assert cal0["k_scale"] > 0
    assert cal0["v_scale"] > 0
    assert cal0["q_values"] == 96
    assert cal0["attention_context_tokens"] == 1
    assert abs(cal0["softmax_sum"] - 1.0) < 1e-6
    assert cal0["attention_output_vector_l2"] > 0
    assert cal0["attention_output_vector_checksum"] > 0
    rg = subprocess.check_output([str(exe), "state-loop-probe", "--in", str(qxf), "--tokens", str(tok_sidecar), "--prompt-token", "2", "--steps", "2", "--layers", "2", "--ctx", "16", "--kv", "int8", "--top-k", "3", "--scan", "32", "--temperature", "0", "--seed", "7", "--rope-gqa-attention", "--residual-dims", "1152", "--norm", "blk.0.attn_norm.weight"], text=True)
    rgd = json.loads(rg)
    rgl0 = rgd["tokens"][1]["layers"][0]
    assert rgd["delta_source"] == "rope_gqa_attention"
    assert rgl0["attention_mode"] == "rope_gqa_full_heads"
    assert rgl0["rope_applied"] is True
    assert rgl0["rope_theta"] == 1000000
    assert rgl0["q_heads_total"] == 32
    assert rgl0["kv_heads_total"] == 4
    assert rgl0["q_heads_run"] == 2
    assert rgl0["kv_heads_touched"] == 1
    assert rgl0["gqa_group_size"] == 8
    assert rgl0["head_dim"] == 128
    assert rgl0["output_projection_input_dims"] == 256
    assert rgl0["output_projection_output_dims"] == 256
    assert abs(rgl0["softmax_sum_min"] - 1.0) < 1e-6
    assert abs(rgl0["softmax_sum_max"] - 1.0) < 1e-6
    kv_snapshot = tmp_path / "accumulated-kv.bin"
    replay_common = [
        str(exe), "state-loop-probe", "--in", str(qxf), "--tokens", str(tok_sidecar),
        "--layers", "2", "--ctx", "16", "--kv", "int8", "--top-k", "3",
        "--scan", "32", "--temperature", "0", "--seed", "7", "--rope-gqa-attention",
        "--residual-dims", "1152", "--norm", "blk.0.attn_norm.weight",
    ]
    baseline = json.loads(subprocess.check_output(
        replay_common + ["--prompt-token", "2", "--steps", "3"], text=True))
    captured = json.loads(subprocess.check_output(
        replay_common + [
            "--prompt-token", "2", "--steps", "2", "--kv-snapshot-out", str(kv_snapshot),
        ], text=True))
    assert kv_snapshot.read_bytes()[:8] == b"QXKVSNP1"
    continuation_token = captured["final_token"]
    replayed = json.loads(subprocess.check_output(
        replay_common + [
            "--prompt-token", str(continuation_token), "--steps", "1",
            "--kv-snapshot-in", str(kv_snapshot),
        ], text=True))
    assert replayed["position_base"] == 2
    assert replayed["tokens"][0]["position"] == 2
    assert replayed["tokens"][0]["input_token"] == baseline["tokens"][2]["input_token"]
    assert replayed["tokens"][0]["layers"] == baseline["tokens"][2]["layers"]
    assert replayed["tokens"][0]["selected_token"] == baseline["tokens"][2]["selected_token"]
    assert replayed["final_token"] == baseline["final_token"]

    snapshot_bytes = kv_snapshot.read_bytes()
    same_length_mutation = bytearray(snapshot_bytes)
    same_length_mutation[48] ^= 1
    for name, corrupt in (
        ("truncated", snapshot_bytes[:-1]),
        ("extra", snapshot_bytes + b"x"),
        ("bad-magic", bytes([snapshot_bytes[0] ^ 1]) + snapshot_bytes[1:]),
        ("same-length-payload-mutation", bytes(same_length_mutation)),
    ):
        corrupt_path = tmp_path / f"{name}.bin"
        corrupt_path.write_bytes(corrupt)
        rejected = subprocess.run(
            replay_common + [
                "--prompt-token", str(continuation_token), "--steps", "1",
                "--kv-snapshot-in", str(corrupt_path),
            ], text=True, capture_output=True)
        assert rejected.returncode != 0
    wrong_token = subprocess.run(
        replay_common + [
            "--prompt-token", str(continuation_token + 1), "--steps", "1",
            "--kv-snapshot-in", str(kv_snapshot),
        ], text=True, capture_output=True)
    assert wrong_token.returncode != 0
    sb = subprocess.check_output([str(exe), "state-loop-probe", "--in", str(qxf), "--tokens", str(tok_sidecar), "--prompt-token", "2", "--steps", "2", "--layers", "2", "--ctx", "16", "--kv", "int8", "--top-k", "3", "--scan", "32", "--temperature", "0", "--seed", "7", "--real-kv", "--projection-matvec", "--residual-vector", "--residual-carry", "--numeric-deltas", "--delta-vectors", "--attention-output-vector", "--bench", "--residual-dims", "96", "--norm", "blk.0.attn_norm.weight"], text=True)
    sbd = json.loads(sb)
    assert sbd["bench"]["enabled"] is True
    assert sbd["bench"]["tokens_per_second"] > 0
    assert sbd["bench"]["ms_per_token"] > 0
    assert sbd["bench"]["layer_steps_per_second"] > 0
    assert sbd["bench"]["layer_steps"] == 4
    assert sbd["bench"]["phases"]["prefill"]["tokens"] == 0
    assert sbd["bench"]["phases"]["prefill"]["elapsed_sec"] == 0
    assert sbd["bench"]["phases"]["decode"]["tokens"] == 2
    assert sbd["bench"]["phases"]["decode"]["elapsed_sec"] > 0
    assert sbd["bench"]["phases"]["decode"]["tokens_per_second"] > 0
    assert tf["layers_run"] == 2
    assert tf["top_k"] == 2
    assert tf["embedding_probe"] != 0
    assert tf["moe_output_probe"] != 0
    assert tf["token_output_probe"] != 0


def test_real_qwen_output_weight_q6k_decode_if_available():
    exe = ROOT / "build" / "qxqxf.exe"
    qxf = ROOT / "models" / "Qwen3-30B-A3B-UD-IQ2_M.qxf"
    if not exe.exists() or not qxf.exists():
        return
    dec = subprocess.check_output([str(exe), "decode-block", "--in", str(qxf), "--name", "output.weight", "--block", "0"], text=True)
    dd = json.loads(dec)
    assert dd["tensor"] == "output.weight"
    assert dd["ggml_type"] == 14
    assert dd["decoder"] == "Q6_K"
    assert dd["decoded"] is True
    assert dd["values"] == 256
    assert dd["block_byte_size"] == 210
    assert dd["sum"] != 0
