# qx-mini-c bootstrap status

## Current decision

The machine has >30 GB free disk, so the project can use the serious path:

- target utility: Qwen3-30B-A3B MoE with Q2/IQ2/Q3 and experts mmap/offload;
- keep Qwen3-8B Q4_K_M as the laboratory path for dense-kernel validation;
- keep a Qwen3-4B fallback only if the machine cannot operate the MoE path.

## Implemented in this bootstrap

- C memory-fit planner API: `include/qxfit.h`.
- C implementation: `src/qxfit.c`.
- CLI executable: `src/qx_main.c`.
- QXF binary contract: `include/qx_format.h`.
- QXF metadata writer/reader: `src/qx_format.c`.
- GGUF header/metadata/tensor directory parser: `src/qx_gguf.c`.
- QXF tensor-copy from GGUF with 4096-byte output alignment and tensor checksums.
- QXF CLI executable: `src/qx_qxf_main.c`.
- Synthetic GGUF fixture generator: `scripts/make_synthetic_gguf.py`.
- Python reference model: `scripts/qxfit_ref.py`.
- HF metadata collector: `scripts/fetch_qwen3_metadata.py`.
- Tests: `tests/test_qxfit_ref.py`, `tests/test_qxf.py`.
- MSVC build helper: `build_msvc.bat`.
- Metadata snapshot: `models/qwen3_targets.json`.

## Verified outputs

Python tests:

```text
10 passed in 0.66s
```

MSVC build:

```text
cl /nologo /std:c17 /O2 /W4 /Iinclude src\qxfit.c src\qx_main.c /Fo:build\ /Fe:build\qxfit.exe
cl /nologo /std:c17 /O2 /W4 /Iinclude src\qx_format.c src\qx_gguf.c src\qx_qxf_main.c /Fo:build\ /Fe:build\qxqxf.exe
```

Qwen3-8B custom Q3/Q4 target fit:

```json
{
  "model": "qwen3-8b",
  "weight_gib": 3.300,
  "ctx_tokens": 4096,
  "kv_format": "int8",
  "kv_gib": 0.281,
  "runtime_overhead_gib": 0.751,
  "total_active_gib": 4.051,
  "usable_ram_gib": 11.000,
  "usable_vram_gib": 4.200,
  "suggested_vram_weights_gib": 3.300,
  "suggested_ram_weights_gib": 0.000,
  "feasible": true
}
```

Qwen3-30B-A3B Q2 target fit on paper:

```json
{
  "model": "qwen3-30b-a3b",
  "weight_gib": 10.485,
  "ctx_tokens": 4096,
  "kv_format": "int8",
  "kv_gib": 0.188,
  "runtime_overhead_gib": 0.657,
  "total_active_gib": 11.143,
  "usable_ram_gib": 11.000,
  "usable_vram_gib": 4.200,
  "suggested_vram_weights_gib": 3.543,
  "suggested_ram_weights_gib": 6.942,
  "feasible": true
}
```

## QXF metadata-only smoke: Qwen3-8B lab target

```text
./build/qxqxf.exe create --model qwen3-8b --quant q3q4mix --out models/qwen3-8b-meta.qxf
./build/qxqxf.exe inspect --in models/qwen3-8b-meta.qxf
```

Output summary:

```json
{
  "magic": "QXF1",
  "version": 1,
  "model_type": "qwen3_dense",
  "quant_type": "q3q4mix",
  "layers": 36,
  "hidden": 4096,
  "intermediate": 12288,
  "q_heads": 32,
  "kv_heads": 8,
  "head_dim": 128,
  "vocab": 151936,
  "tensor_count": 327,
  "dir_offset": 4096,
  "data_offset": 73728,
  "file_size": 73728
}
```

## QXF metadata-only smoke: Qwen3-30B-A3B utility target

```text
./build/qxqxf.exe create --model qwen3-30b-a3b --quant q2 --out models/qwen3-30b-a3b-meta.qxf
./build/qxqxf.exe inspect --in models/qwen3-30b-a3b-meta.qxf
```

Output summary:

```json
{
  "magic": "QXF1",
  "version": 1,
  "model_type": "qwen3_moe",
  "quant_type": "q2",
  "layers": 48,
  "hidden": 2048,
  "intermediate": 6144,
  "q_heads": 32,
  "kv_heads": 4,
  "head_dim": 128,
  "vocab": 151936,
  "tensor_count": 18771,
  "dir_offset": 4096,
  "data_offset": 3911680,
  "file_size": 3911680
}
```

## Next engineering step

Implement F1-MoE:

1. Run `inspect-gguf` against the real Qwen3-30B-A3B Q2/IQ2 GGUF after download.
2. Tensor manifest generation from actual GGUF tensor metadata.
3. Name mapper GGUF -> QXF-MoE layout.
4. QXF tensor-copy mode with 4096-byte alignment and checksums.
5. Expert mmap/offload planning before decode kernels.

## QXF tensor-copy smoke

`create-from-gguf-copy` now reads a GGUF tensor table, computes per-tensor byte
spans from relative offsets, copies tensor bytes into QXF, aligns each tensor to
4096 bytes, and writes checksums back into the QXF directory.

```text
cmd.exe /c "build\qxqxf.exe create-from-gguf-copy --in tests\fixtures\qwen3-30b-a3b-mini.gguf --model qwen3-30b-a3b --quant q2 --out models\qwen3-30b-a3b-copy.qxf"
cmd.exe /c "build\qxqxf.exe inspect --in models\qwen3-30b-a3b-copy.qxf"
```

Verified output:

```json
{
  "magic": "QXF1",
  "version": 1,
  "model_type": "qwen3_moe",
  "quant_type": "q2",
  "tensor_count": 6,
  "dir_offset": 4096,
  "data_offset": 8192,
  "file_size": 32768
}
```

## GGUF parser smoke

The parser was verified against a synthetic Qwen3-30B-A3B mini GGUF fixture.

```text
python scripts/make_synthetic_gguf.py --out tests/fixtures/qwen3-30b-a3b-mini.gguf
./build/qxqxf.exe inspect-gguf --in tests/fixtures/qwen3-30b-a3b-mini.gguf
./build/qxqxf.exe create-from-gguf --in tests/fixtures/qwen3-30b-a3b-mini.gguf --model qwen3-30b-a3b --quant q2 --out models/qwen3-30b-a3b-from-gguf-meta.qxf
```

Verified extracted fields:

```json
{
  "magic": "GGUF",
  "version": 3,
  "tensor_count": 6,
  "metadata_kv_count": 10,
  "alignment": 32,
  "architecture": "qwen3moe",
  "qwen3_block_count": 48,
  "qwen3_expert_count": 128,
  "qwen3_expert_used_count": 8,
  "qwen3_embedding_length": 2048,
  "qwen3_attention_head_count_kv": 4
}
```


## Real Qwen3-30B-A3B conversion

Downloaded real target:

```text
models/Qwen3-30B-A3B-UD-IQ2_M.gguf
bytes: 10865578560
size: 10.119 GiB
```

Real GGUF inspection:

```text
magic: GGUF
version: 3
tensor_count: 579
metadata_kv_count: 35
architecture: qwen3moe
name: Qwen3-30B-A3B
layers: 48
hidden: 2048
ffn: 6144
experts: 128
top-k: 8
q_heads: 32
kv_heads: 4
data_offset: 5970496
```

Real QXF tensor-copy completed:

```text
models/Qwen3-30B-A3B-UD-IQ2_M.qxf
bytes: 10860081152
size: 10.114 GiB
```

QXF inspect:

```json
{
  "magic": "QXF1",
  "version": 1,
  "model_type": "qwen3_moe",
  "quant_type": "q2",
  "layers": 48,
  "hidden": 2048,
  "intermediate": 6144,
  "q_heads": 32,
  "kv_heads": 4,
  "head_dim": 128,
  "vocab": 151936,
  "tensor_count": 579,
  "dir_offset": 4096,
  "data_offset": 126976,
  "file_size": 10860081152
}
```

Verification:

```text
10 passed in 0.65s
```


## QXF read-only loader smoke

Implemented read-only QXF loader:

```text
qx_open_file
qx_find_tensor
qx_verify_tensor_checksum
qx_verify_all_tensors
```

Real tensor lookup:

```text
cmd.exe /c "build\qxqxf.exe inspect-tensor --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --name token_embd.weight"
```

Output:

```json
{
  "name": "token_embd.weight",
  "dtype": 3,
  "quant": 5,
  "rank": 2,
  "dims": [2048, 151936],
  "offset": 255389696,
  "byte_size": 175030272,
  "checksum": 7042251507533971635
}
```

Checksum spot-check:

```text
cmd.exe /c "build\qxqxf.exe verify-qxf --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --max 8"
```

Output:

```json
{"verified": true, "checked": 8, "tensor_count": 579}
```

Verification:

```text
10 passed in 1.52s
```


## MoE expert index + cache plan

Implemented MoE expert index over real QXF directory:

```text
build\qxqxf.exe expert-index --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf
```

Real output:

```json
{
  "model_type": "qwen3_moe",
  "layers": 48,
  "experts_per_layer": 128,
  "experts_per_token": 8,
  "router_tensors": 48,
  "expert_gate_tensors": 48,
  "expert_up_tensors": 48,
  "expert_down_tensors": 48,
  "complete_layers": 48,
  "packed_expert_tensors": 144,
  "expert_tensor_bytes": 9863430144,
  "avg_layer_expert_bytes": 205488128,
  "min_layer_expert_bytes": 193462272,
  "max_layer_expert_bytes": 235929600,
  "avg_single_expert_bytes": 1605376
}
```

Implemented cache sizing:

```text
build\qxqxf.exe expert-plan --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --vram-gib 2.0 --ram-gib 6.5
```

Real output:

```json
{
  "avg_single_expert_bytes": 1605376,
  "total_expert_bytes": 9863430144,
  "total_experts": 6144,
  "hot_vram_gib": 2.000,
  "hot_ram_gib": 6.500,
  "vram_expert_slots": 1337,
  "ram_expert_slots": 4347,
  "vram_layer_equivalent": 10.45,
  "ram_layer_equivalent": 33.96,
  "experts_per_token": 8
}
```

Verification:

```text
10 passed in 1.88s
```


## Expert slice addressing

Implemented packed MoE expert slice addressing:

```text
build\qxqxf.exe expert-slice --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --layer 0 --expert 0
build\qxqxf.exe expert-slice --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --layer 47 --expert 127
```

Layer 0 / expert 0:

```json
{
  "layer": 0,
  "expert": 0,
  "experts_per_layer": 128,
  "slices": {
    "gate": {"tensor": "blk.0.ffn_gate_exps.weight", "offset": 517533696, "byte_size": 454656, "packed_byte_size": 58195968, "remainder": 0},
    "up": {"tensor": "blk.0.ffn_up_exps.weight", "offset": 576786432, "byte_size": 454656, "packed_byte_size": 58195968, "remainder": 0},
    "down": {"tensor": "blk.0.ffn_down_exps.weight", "offset": 440463360, "byte_size": 602112, "packed_byte_size": 77070336, "remainder": 0}
  },
  "slice_exact": true
}
```

Layer 47 / expert 127:

```json
{
  "layer": 47,
  "expert": 127,
  "experts_per_layer": 128,
  "slices": {
    "gate": {"tensor": "blk.47.ffn_gate_exps.weight", "offset": 10794033152, "byte_size": 503808, "packed_byte_size": 64487424, "remainder": 0},
    "up": {"tensor": "blk.47.ffn_up_exps.weight", "offset": 10859577344, "byte_size": 503808, "packed_byte_size": 64487424, "remainder": 0},
    "down": {"tensor": "blk.47.ffn_down_exps.weight", "offset": 10729213952, "byte_size": 835584, "packed_byte_size": 106954752, "remainder": 0}
  },
  "slice_exact": true
}
```

Verification:

```text
10 passed in 1.70s
```


## Expert load primitive + cache demo

Implemented expert slice load primitive:

```text
build\qxqxf.exe expert-load --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --layer 0 --expert 0 --kind gate
build\qxqxf.exe expert-load --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --layer 47 --expert 127 --kind down
```

Real output, layer 0 expert 0 gate:

```json
{
  "loaded": true,
  "layer": 0,
  "expert": 0,
  "kind": "gate",
  "tensor": "blk.0.ffn_gate_exps.weight",
  "offset": 517533696,
  "byte_size": 454656,
  "checksum": 11262208963638616294
}
```

Real output, layer 47 expert 127 down:

```json
{
  "loaded": true,
  "layer": 47,
  "expert": 127,
  "kind": "down",
  "tensor": "blk.47.ffn_down_exps.weight",
  "offset": 10729213952,
  "byte_size": 835584,
  "checksum": 14285278245730931606
}
```

Implemented LRU-style cache behavior demo:

```text
build\qxqxf.exe cache-demo --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --slots 2 --sequence 0:0:gate,0:0:gate,0:1:gate,0:0:gate
```

Output:

```json
{"requests": 4, "hits": 2, "misses": 2, "slots": 2}
```

Verification:

```text
1 passed in 1.37s
10 passed in 0.84s
```


## Expert load bandwidth benchmark

Implemented benchmark command:

```text
build\qxqxf.exe bench-expert-load --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --iters 128 --kind gate
```

Real output, gate:

```json
{
  "loads": 128,
  "kind": "gate",
  "bytes": 59572224,
  "mib": 56.812,
  "seconds": 0.131000,
  "avg_ms": 1.023,
  "mib_per_sec": 433.683,
  "checksum_mix": 4548474522337222801
}
```

Real output, up:

```json
{
  "loads": 128,
  "kind": "up",
  "bytes": 59572224,
  "mib": 56.812,
  "seconds": 0.138000,
  "avg_ms": 1.078,
  "mib_per_sec": 411.685,
  "checksum_mix": 173222889435899423
}
```

Real output, down:

```json
{
  "loads": 128,
  "kind": "down",
  "bytes": 84922368,
  "mib": 80.988,
  "seconds": 0.154000,
  "avg_ms": 1.203,
  "mib_per_sec": 525.898,
  "checksum_mix": 3650142174686198851
}
```

Verification:

```text
1 passed in 1.05s
10 passed in 0.90s
```


## Real expert cache with persistent slots

Implemented real cache command:

```text
build\qxqxf.exe cache-run --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --slots 2 --sequence 0:0:gate,0:0:gate,0:1:gate,0:0:gate
```

Output, no eviction:

```json
{"requests": 4, "hits": 2, "misses": 2, "slots": 2, "bytes_loaded": 909312, "resident_bytes": 909312, "checksum_mix": 6050353679935418180}
```

Eviction smoke:

```text
build\qxqxf.exe cache-run --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --slots 2 --sequence 0:0:gate,0:0:gate,0:1:gate,0:2:gate,0:0:gate
```

Output:

```json
{"requests": 5, "hits": 1, "misses": 4, "slots": 2, "bytes_loaded": 1818624, "resident_bytes": 909312, "checksum_mix": 12307258353075299401}
```

Verification:

```text
1 passed in 1.19s
10 passed in 0.89s
```


## Router trace simulator

Implemented deterministic route trace command:

```text
build\qxqxf.exe route-trace --layers 48 --experts 128 --top-k 8 --tokens 2 --seed 7
```

Generated real trace file:

```text
models/route_trace_2tok.json
requests: 2304
sequence_chars: 23714
```

Integrated with real cache runner:

```text
build\qxqxf.exe cache-run --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --slots 256 --sequence <route_trace.sequence>
```

Output:

```json
{"requests": 2304, "hits": 0, "misses": 2304, "slots": 256, "bytes_loaded": 1232928768, "resident_bytes": 151437312, "checksum_mix": 16378076191289069085}
```

Interpretation: fully random synthetic routing has no locality, so hit rate is 0%. Next trace mode should add router locality/reuse knobs to estimate useful hit rates.

Verification:

```text
1 passed in 1.25s
10 passed in 0.90s
```


## Route trace reuse + sequence-file cache run

Implemented route trace locality knob:

```text
build\qxqxf.exe route-trace --layers 48 --experts 128 --top-k 8 --tokens 4 --seed 7 --reuse-pct 100
```

Implemented cache-run sequence file input to avoid Windows argv length limit:

```text
build\qxqxf.exe cache-run --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --slots 2048 --sequence-file models\route_trace_reuse100_4tok.seq
```

Reuse 75%, 4 tokens:

```text
trace requests: 4608
sequence_chars: 47354
slots 256:  hits 0,   misses 4608, bytes_loaded 2465857536, resident_bytes 151437312
slots 512:  hits 0,   misses 4608, bytes_loaded 2465857536, resident_bytes 283963392
slots 1024: hits 0,   misses 4608, bytes_loaded 2465857536, resident_bytes 544899072
slots 2048: hits 120, misses 4488, bytes_loaded 2399649792, resident_bytes 1096826880
```

Reuse 100%, 4 tokens:

```text
trace requests: 4608
sequence_chars: 47375
slots 256:  hits 0,    misses 4608, bytes_loaded 2465857536, resident_bytes 151437312
slots 512:  hits 0,    misses 4608, bytes_loaded 2465857536, resident_bytes 283963392
slots 1024: hits 0,    misses 4608, bytes_loaded 2465857536, resident_bytes 544899072
slots 2048: hits 3456, misses 1152, bytes_loaded 616464384,  resident_bytes 616464384
```

Interpretation: one token of full top-k traffic is 1152 slice requests. Cache needs >1152 slots to retain full previous token for immediate reuse. 2048 slots works; 1024 thrashes.

Verification:

```text
1 passed in 1.48s
10 passed in 0.88s
```


## Expert-complete cache

Implemented cache keyed by full expert, not individual gate/up/down tensor slices:

```text
build\qxqxf.exe cache-run-expert --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --slots N --sequence-file models\route_trace_reuse100_4tok.seq
```

Reuse 100%, 4 tokens, expert-complete cache:

```text
slots 256:  requests 4608, expert_requests 1536, hits 0,    misses 1536, bytes_loaded 2465857536, resident_bytes 415825920
slots 512:  requests 4608, expert_requests 1536, hits 1152, misses 384,  bytes_loaded 616464384,  resident_bytes 616464384
slots 1024: requests 4608, expert_requests 1536, hits 1152, misses 384,  bytes_loaded 616464384,  resident_bytes 616464384
slots 2048: requests 4608, expert_requests 1536, hits 1152, misses 384,  bytes_loaded 616464384,  resident_bytes 616464384
```

Interpretation: expert-complete cache fixes previous slice-slot waste. With 4-token full reuse, only 384 unique layer/expert pairs exist; 512 slots enough. 256 slots thrashes because one token needs 384 expert slots.

Verification:

```text
1 passed in 1.83s
10 passed in 0.82s
```


## Expert-complete cache planner

Implemented planner for cache unit = full expert:

```text
build\qxqxf.exe expert-cache-plan-complete --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --vram-gib 2.0 --ram-gib 6.5 --top-k 8
```

Real output:

```json
{
  "cache_unit": "full_expert",
  "layers": 48,
  "experts_per_layer": 128,
  "top_k": 8,
  "working_set_experts_per_token": 384,
  "avg_full_expert_bytes": 1605376,
  "hot_vram_gib": 2.000,
  "hot_ram_gib": 6.500,
  "vram_expert_slots": 1337,
  "ram_expert_slots": 4347,
  "vram_covers_one_token": true,
  "ram_covers_one_token": true
}
```

Interpretation: with full-expert cache, 2 GiB VRAM hot cache can hold ~1337 experts, enough for one token working set of 384 layer/expert pairs at top-8 across 48 layers.

Verification:

```text
1 passed in 1.23s
10 passed in 0.87s
```


## Full runtime planner

Implemented runtime planning command:

```text
build\qxqxf.exe runtime-plan --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --ctx 4096 --kv int8 --vram-gib 4.2 --ram-gib 11.0 --hot-vram-gib 2.0 --hot-ram-gib 6.5 --top-k 8
```

Real output:

```json
{
  "model_type": "qwen3_moe",
  "cache_unit": "full_expert",
  "ctx_tokens": 4096,
  "kv_format": "int8",
  "layers": 48,
  "top_k": 8,
  "working_set_experts_per_token": 384,
  "total_model_gib": 10.114239,
  "expert_gib": 9.186035,
  "non_expert_gib": 0.928204,
  "kv_gib": 0.187500,
  "runtime_overhead_gib": 0.750,
  "hot_vram_gib": 2.000,
  "hot_ram_gib": 6.500,
  "vram_expert_slots": 1337,
  "ram_expert_slots": 4347,
  "vram_covers_one_token": true,
  "ram_covers_one_token": true,
  "usable_vram_gib": 4.200,
  "usable_ram_gib": 11.000,
  "active_vram_gib": 2.000,
  "active_ram_gib": 8.366,
  "feasible": true
}
```

Interpretation: QXF 10.114 GiB total, packed experts 9.186 GiB, non-expert resident base about 0.928 GiB. With 2.0 GiB VRAM hot experts + 6.5 GiB RAM hot experts + 4K INT8 KV, active RAM estimate is 8.366 GiB and active VRAM estimate is 2.0 GiB, feasible under 11.0 GiB RAM / 4.2 GiB VRAM usable.

Verification:

```text
1 passed in 1.01s
10 passed in 0.86s
```


## Minimal inference scheduling scaffold

Implemented token embedding row lookup:

```text
build\qxqxf.exe token-embedding --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --token-id 42
```

Real output:

```json
{
  "token_id": 42,
  "tensor": "token_embd.weight",
  "dtype": 3,
  "rank": 2,
  "hidden": 2048,
  "vocab": 151936,
  "offset": 255438080,
  "row_byte_size": 1152,
  "tensor_byte_size": 175030272
}
```

Implemented mock forward schedule over QXF metadata:

```text
build\qxqxf.exe forward-schedule --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --token-id 42 --top-k 8
```

Real output:

```json
{
  "mock_forward": true,
  "token_id": 42,
  "layers": 48,
  "top_k": 8,
  "steps_per_layer": 5,
  "embedding": {"tensor": "token_embd.weight", "offset": 255438080, "row_byte_size": 1152},
  "layer0": {
    "attention": true,
    "router": true,
    "expert_gate": true,
    "expert_up": true,
    "expert_down": true
  },
  "planned_ops": 241
}
```

Interpretation: this is not numeric inference yet. It verifies the runtime can locate token embedding bytes, discover layer-0 attention/router/expert tensors, and produce a deterministic forward execution skeleton across 48 layers.

Verification:

```text
1 passed in 1.31s
10 passed in 0.81s
```


## Quant block probe + matvec stub

Implemented raw quant block probe:

```text
build\qxqxf.exe quant-block --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --name token_embd.weight --block 0
```

Real output:

```json
{
  "tensor": "token_embd.weight",
  "dtype": 3,
  "quant": 5,
  "group_size": 64,
  "block_index": 0,
  "block_count": 683712,
  "block_offset": 255389696,
  "block_byte_size": 256,
  "checksum": 14754412774054462806,
  "dequantized": false,
  "note": "raw quant block probe; numeric dequant kernel not implemented"
}
```

Implemented matvec stub over raw quant bytes:

```text
build\qxqxf.exe matvec-stub --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --name token_embd.weight --rows 2
```

Real output:

```json
{
  "stub": true,
  "tensor": "token_embd.weight",
  "rows": 2,
  "dtype": 3,
  "quant": 5,
  "block_byte_size": 256,
  "bytes_read": 512,
  "checksum_mix": 3123787546529711065,
  "numeric_kernel": false
}
```

Interpretation: this deliberately does not fake numeric dequant/inference. It proves QXF can seek/read deterministic raw quant blocks and wire a matvec-kernel entrypoint with checksums. Next step is real GGML quant block decoding for the actual dtype.

Verification:

```text
1 passed in 1.59s
10 passed in 0.94s
```


## GGML type preservation in QXF

Preserved source GGUF `ggml_type` inside QXF tensor directory `flags`, and exposed it in `inspect-tensor` / `quant-block`.

Reference from current ggml enum:

```text
12 = GGML_TYPE_Q4_K
17 = GGML_TYPE_IQ2_XS
18 = GGML_TYPE_IQ3_XXS
```

Reconverted real QXF so the new field is populated:

```text
build\qxqxf.exe create-from-gguf-copy --in models\Qwen3-30B-A3B-UD-IQ2_M.gguf --model qwen3-30b-a3b --quant q2 --out models\Qwen3-30B-A3B-UD-IQ2_M.qxf
```

Real tensor types:

```json
{"name":"token_embd.weight", "dtype":3, "ggml_type":12, "meaning":"Q4_K"}
{"name":"blk.0.ffn_gate_exps.weight", "dtype":5, "ggml_type":17, "meaning":"IQ2_XS"}
{"name":"blk.0.ffn_up_exps.weight", "dtype":5, "ggml_type":17, "meaning":"IQ2_XS"}
{"name":"blk.0.ffn_down_exps.weight", "dtype":4, "ggml_type":18, "meaning":"IQ3_XXS"}
```

Real quant-block over expert gate:

```json
{
  "tensor": "blk.0.ffn_gate_exps.weight",
  "dtype": 5,
  "ggml_type": 17,
  "quant": 5,
  "group_size": 64,
  "block_index": 0,
  "block_count": 227328,
  "block_offset": 517533696,
  "block_byte_size": 256,
  "checksum": 14999277971911976401,
  "dequantized": false
}
```

Verification:

```text
1 passed in 1.40s
10 passed in 0.90s
```

Next decoder target: `IQ2_XS` for gate/up experts first; `IQ3_XXS` for down experts second; `Q4_K` for embedding/non-expert later.


## IQ2_XS decode-block

Research source: current `ggml-org/ggml` source (`ggml-common.h`, `ggml-quants.c`). Relevant facts:

```text
GGML_TYPE_IQ2_XS = 17
QK_K = 256
block_iq2_xs size = sizeof(fp16 d) + 32 uint16 qs + 8 uint8 scales = 74 bytes
```

Implemented generated IQ2_XS tables from ggml:

```text
src/qx_iq2xs_tables.inc
```

Implemented CLI:

```text
build\qxqxf.exe decode-block --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --name blk.0.ffn_gate_exps.weight --block 0
```

Real output gate expert block 0:

```json
{
  "tensor": "blk.0.ffn_gate_exps.weight",
  "ggml_type": 17,
  "decoder": "IQ2_XS",
  "decoded": true,
  "block_index": 0,
  "block_offset": 517533696,
  "block_byte_size": 74,
  "values": 256,
  "sum": -0.25754571,
  "min": -0.0715076923,
  "max": 0.0715076923,
  "raw_checksum": 7804969416261204372,
  "first8": [0.0133037567, -0.0133037567, 0.0133037567, -0.0415742397, -0.0415742397, 0.0133037567, 0.0133037567, -0.0133037567]
}
```

Real output up expert block 0:

```json
{
  "tensor": "blk.0.ffn_up_exps.weight",
  "ggml_type": 17,
  "decoder": "IQ2_XS",
  "decoded": true,
  "block_index": 0,
  "block_offset": 576786432,
  "block_byte_size": 74,
  "values": 256,
  "sum": -0.121747822,
  "min": -0.0401071012,
  "max": 0.0731364787,
  "raw_checksum": 7375102694807624831,
  "first8": [0.00395035744, 0.00395035744, 0.012344867, 0.012344867, 0.012344867, 0.00395035744, -0.00395035744, -0.012344867]
}
```

Verification:

```text
1 passed in 1.47s
10 passed in 0.84s
```

Next: implement `IQ3_XXS` decode for `blk.N.ffn_down_exps.weight` (`ggml_type=18`), then wire real matvec over decoded blocks.


## IQ3_XXS decode-block

Research source: current `ggml-org/ggml` source (`ggml-common.h`, `ggml-quants.c`). Relevant facts:

```text
GGML_TYPE_IQ3_XXS = 18
QK_K = 256
block_iq3_xxs size = sizeof(fp16 d) + 96 uint8 qs/scales/signs = 98 bytes
```

Extended generated decode tables:

```text
src/qx_iq2xs_tables.inc now also includes qx_iq3xxs_grid[256]
```

Implemented `decode-block` support for `IQ3_XXS` down experts.

Real output down expert block 0:

```json
{
  "tensor": "blk.0.ffn_down_exps.weight",
  "ggml_type": 18,
  "decoder": "IQ3_XXS",
  "decoded": true,
  "block_index": 0,
  "block_offset": 440463360,
  "block_byte_size": 98,
  "values": 256,
  "sum": 0.101204038,
  "min": -0.0726884007,
  "max": 0.0633092523,
  "raw_checksum": 10537357753799797417,
  "first8": [0.00287425518, 0.0258682966, 0.0445509553, 0.00862276554, -0.0143712759, 0.0201197863, -0.031616807, 0.00287425518]
}
```

Verification:

```text
1 passed in 1.51s
10 passed in 0.82s
```

Next: implement real block-dot/matvec over decoded IQ2_XS/IQ3_XXS blocks.


## Quant block-dot

Implemented real decoded block dot product for supported expert quant blocks:

```text
IQ2_XS (ggml_type=17) -> gate/up experts
IQ3_XXS (ggml_type=18) -> down experts
```

CLI:

```text
build\qxqxf.exe block-dot --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --name blk.0.ffn_gate_exps.weight --block 0 --seed 7
```

The input vector is deterministic (`deterministic_lcg_unit`) so the smoke is repeatable without pretending to run a real activation yet.

Real gate block-dot:

```json
{
  "tensor": "blk.0.ffn_gate_exps.weight",
  "ggml_type": 17,
  "decoder": "IQ2_XS",
  "block_index": 0,
  "block_offset": 517533696,
  "block_byte_size": 74,
  "values": 256,
  "input_seed": 7,
  "input_kind": "deterministic_lcg_unit",
  "dot": -0.284831268,
  "input_sum": 3.96679688,
  "weight_sum": -0.25754571,
  "raw_checksum": 7804969416261204372
}
```

Real up block-dot:

```json
{
  "tensor": "blk.0.ffn_up_exps.weight",
  "ggml_type": 17,
  "decoder": "IQ2_XS",
  "dot": -0.121760674,
  "weight_sum": -0.121747822,
  "raw_checksum": 7375102694807624831
}
```

Real down block-dot:

```json
{
  "tensor": "blk.0.ffn_down_exps.weight",
  "ggml_type": 18,
  "decoder": "IQ3_XXS",
  "block_index": 0,
  "block_offset": 440463360,
  "block_byte_size": 98,
  "values": 256,
  "input_seed": 7,
  "input_kind": "deterministic_lcg_unit",
  "dot": -0.111271971,
  "input_sum": 3.96679688,
  "weight_sum": 0.101204038,
  "raw_checksum": 10537357753799797417
}
```

Verification:

```text
1 passed in 2.45s
10 passed in 1.02s
```

Next: matvec-row over N blocks for one row/logit element, then small expert gate/up/down row probes.


## Matvec-row over quant blocks

Implemented real row-style accumulation over multiple decoded quant blocks:

```text
matvec-row = sum over N blocks of dot(decode(block), deterministic_input_chunk[256])
Supported: IQ2_XS, IQ3_XXS
```

CLI:

```text
build\qxqxf.exe matvec-row --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --name blk.0.ffn_gate_exps.weight --start-block 0 --blocks 8 --seed 7
```

Real gate row probe:

```json
{
  "tensor": "blk.0.ffn_gate_exps.weight",
  "ggml_type": 17,
  "decoder": "IQ2_XS",
  "start_block": 0,
  "blocks": 8,
  "values": 2048,
  "bytes_read": 592,
  "dot": -0.827121252,
  "input_sum": -39.0565186,
  "weight_sum": -2.19115421,
  "checksum_mix": 13072283985099508783
}
```

Real up row probe:

```json
{
  "tensor": "blk.0.ffn_up_exps.weight",
  "ggml_type": 17,
  "decoder": "IQ2_XS",
  "values": 2048,
  "bytes_read": 592,
  "dot": 0.121682341,
  "weight_sum": -0.244790643,
  "checksum_mix": 12427290363261994030
}
```

Real down row probe:

```json
{
  "tensor": "blk.0.ffn_down_exps.weight",
  "ggml_type": 18,
  "decoder": "IQ3_XXS",
  "values": 2048,
  "bytes_read": 784,
  "dot": 0.446936193,
  "weight_sum": 1.70155293,
  "checksum_mix": 2824462717470038020
}
```

Verification:

```text
1 passed in 2.99s
10 passed in 0.87s
```

Next: expert row probe selecting layer/expert/kind and mapping it to packed tensor start-block automatically.


## Expert-row automatic packed tensor addressing

Implemented `expert-row` so callers no longer pass packed tensor names manually.

CLI:

```text
build\qxqxf.exe expert-row --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --layer 0 --expert 0 --kind gate --start-block 0 --blocks 8 --seed 7
```

It resolves:

```text
(layer, expert, kind) -> blk.N.ffn_{gate|up|down}_exps.weight
slice_offset = tensor.offset + expert * slice_bytes
row probe offset = slice_offset + start_block * quant_block_size
```

Real gate expert-row:

```json
{
  "layer": 0,
  "expert": 0,
  "kind": "gate",
  "tensor": "blk.0.ffn_gate_exps.weight",
  "ggml_type": 17,
  "decoder": "IQ2_XS",
  "slice_offset": 517533696,
  "slice_byte_size": 454656,
  "slice_block_count": 6144,
  "slice_block_remainder": 0,
  "slice_start_block": 0,
  "absolute_start_block": 0,
  "blocks": 8,
  "values": 2048,
  "dot": -0.827121252,
  "checksum_mix": 13072283985099508783
}
```

Real up expert-row:

```json
{
  "layer": 0,
  "expert": 0,
  "kind": "up",
  "tensor": "blk.0.ffn_up_exps.weight",
  "ggml_type": 17,
  "decoder": "IQ2_XS",
  "slice_offset": 576786432,
  "slice_byte_size": 454656,
  "slice_block_count": 6144,
  "dot": 0.121682341,
  "checksum_mix": 12427290363261994030
}
```

Real down expert-row:

```json
{
  "layer": 0,
  "expert": 0,
  "kind": "down",
  "tensor": "blk.0.ffn_down_exps.weight",
  "ggml_type": 18,
  "decoder": "IQ3_XXS",
  "slice_offset": 440463360,
  "slice_byte_size": 602112,
  "slice_block_count": 6144,
  "slice_block_remainder": 0,
  "dot": 0.446936193,
  "checksum_mix": 2824462717470038020
}
```

Verification:

```text
1 passed in 1.19s
10 passed in 1.07s
```

Note: synthetic GGUF fixture was enlarged so per-expert slices contain enough quant blocks for expert-row tests.

Next: build `expert-forward-probe` for one expert: gate row dot + up row dot + SiLU(gate)*up, then down row projection probe.


## Expert forward probe

Implemented one-expert numeric forward probe:

```text
gate_dot = expert-row(kind=gate)
up_dot = expert-row(kind=up)
gate_silu = silu(gate_dot)
hidden_probe = gate_silu * up_dot
down_dot = expert-row(kind=down)
projected_probe = hidden_probe * down_dot
```

CLI:

```text
build\qxqxf.exe expert-forward-probe --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --layer 0 --expert 0 --start-block 0 --blocks 8 --seed 7
```

Real output:

```json
{
  "probe": "expert_forward",
  "layer": 0,
  "expert": 0,
  "start_block": 0,
  "blocks": 8,
  "values": 2048,
  "input_seed": 7,
  "gate_decoder": "IQ2_XS",
  "up_decoder": "IQ2_XS",
  "down_decoder": "IQ3_XXS",
  "gate_ggml_type": 17,
  "up_ggml_type": 17,
  "down_ggml_type": 18,
  "gate_dot": -0.827121252,
  "up_dot": 0.121682341,
  "down_dot": 0.446936193,
  "gate_silu": -0.25165504,
  "hidden_probe": -0.0306219745,
  "projected_probe": -0.0136860687,
  "gate_slice_offset": 517533696,
  "up_slice_offset": 576786432,
  "down_slice_offset": 440463360,
  "gate_checksum_mix": 13072283985099508783,
  "up_checksum_mix": 12427290363261994030,
  "down_checksum_mix": 2824462717470038020
}
```

Verification:

```text
1 passed in 1.63s
10 passed in 0.92s
```

Next: router/top-k probe using `ffn_gate_inp.weight` route logits, initially deterministic row-dot probes, then top-k expert selection feeding `expert-forward-probe`.


## Router top-k probe

Implemented `router-topk-probe` for Qwen3 MoE routing over the real router tensor:

```text
router tensor: blk.N.ffn_gate_inp.weight
kernel: F32_PREFIX_DOT
selected experts: sorted top-k logits
then runs expert forward probe fields for each selected expert
```

CLI:

```text
build\qxqxf.exe router-topk-probe --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --layer 0 --top-k 2 --blocks 8 --seed 7
```

Real router tensor:

```json
{
  "name": "blk.0.ffn_gate_inp.weight",
  "ggml_type": 0,
  "dims": [2048, 128],
  "offset": 575729664,
  "byte_size": 1048576,
  "checksum": 6657109537548853775
}
```

Real router-topk output:

```json
{
  "probe": "router_topk",
  "layer": 0,
  "router_tensor": "blk.0.ffn_gate_inp.weight",
  "router_kernel": "F32_PREFIX_DOT",
  "router_ggml_type": 0,
  "experts": 128,
  "top_k": 2,
  "blocks": 8,
  "values": 2048,
  "input_seed": 7,
  "bytes_read": 1048576,
  "checksum_mix": 17805025676850820827,
  "selected_experts": [
    {"rank": 0, "expert": 35, "logit": 1.65265287, "gate_dot": 0.137305664, "up_dot": 0.322314397, "down_dot": 0.0380495114, "projected_probe": 0.00089966357},
    {"rank": 1, "expert": 51, "logit": 1.50835166, "gate_dot": -0.167478631, "up_dot": 0.0713881327, "down_dot": -0.251976521, "projected_probe": 0.00138047028}
  ]
}
```

Verification:

```text
1 passed in 1.42s
10 passed in 1.09s
```

Implementation note: F16 helper now maps NaN/Inf half values to finite sentinels so synthetic quant fixtures always produce JSON-valid numeric probes.

Next: `layer-forward-probe` for one MoE layer: router top-k -> run selected experts -> aggregate projected probes.


## Layer forward probe

Implemented `layer-forward-probe` for one Qwen3 MoE layer:

```text
router F32 prefix-dot -> sorted top-k experts -> expert forward probes -> aggregate projected probes
```

CLI:

```text
build\qxqxf.exe layer-forward-probe --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --layer 0 --top-k 2 --blocks 8 --seed 7
```

Real output:

```json
{
  "probe": "layer_forward",
  "layer": 0,
  "router_tensor": "blk.0.ffn_gate_inp.weight",
  "router_kernel": "F32_PREFIX_DOT",
  "experts": 128,
  "top_k": 2,
  "blocks": 8,
  "values": 2048,
  "input_seed": 7,
  "router_bytes_read": 1048576,
  "router_checksum_mix": 17805025676850820827,
  "selected_experts": [
    {"rank": 0, "expert": 35, "logit": 1.65265287, "gate_dot": 0.137305664, "up_dot": 0.322314397, "down_dot": 0.0380495114, "projected_probe": 0.00089966357},
    {"rank": 1, "expert": 51, "logit": 1.50835166, "gate_dot": -0.167478631, "up_dot": 0.0713881327, "down_dot": -0.251976521, "projected_probe": 0.00138047028}
  ],
  "expert_outputs_sum": 0.00228013385,
  "layer_output_probe": 0.00228013385
}
```

Verification:

```text
1 passed in 2.05s
10 passed in 1.05s
```

Next: `moe-forward-probe` across multiple layers using the same deterministic input and top-k routing, then cache integration metrics.


## MoE forward probe multi-layer

Implemented `moe-forward-probe`:

```text
for layer 0..N-1:
  router F32 prefix-dot
  select top-k experts
  run supported expert probes
  aggregate layer_output_probe
aggregate moe_output_probe
```

CLI:

```text
build\qxqxf.exe moe-forward-probe --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --layers 2 --top-k 2 --blocks 8 --seed 7
```

Real output:

```json
{
  "probe": "moe_forward",
  "layers_requested": 2,
  "layers_run": 2,
  "top_k": 2,
  "blocks": 8,
  "input_seed": 7,
  "layers": [
    {
      "layer": 0,
      "router_kernel": "F32_PREFIX_DOT",
      "top_k": 2,
      "selected_experts": [
        {"rank": 0, "expert": 35, "logit": 1.65265287, "supported": true, "projected_probe": 0.00089966357},
        {"rank": 1, "expert": 51, "logit": 1.50835166, "supported": true, "projected_probe": 0.00138047028}
      ],
      "layer_output_probe": 0.00228013385
    },
    {
      "layer": 1,
      "router_kernel": "F32_PREFIX_DOT",
      "top_k": 2,
      "selected_experts": [
        {"rank": 0, "expert": 56, "logit": 2.09146463, "supported": false, "reason": "unsupported_expert_quant", "projected_probe": 0},
        {"rank": 1, "expert": 65, "logit": 1.49880139, "supported": false, "reason": "unsupported_expert_quant", "projected_probe": 0}
      ],
      "layer_output_probe": 0
    }
  ],
  "moe_output_probe": 0.00228013385
}
```

Discovery from real QXF:

```text
layer 0 experts: gate/up IQ2_XS (17), down IQ3_XXS (18) => supported
layer 1 experts: gate/up IQ2_S (22), down IQ4_XS (23) => unsupported for numeric expert probe yet
```

Verification:

```text
1 passed in 10.03s
10 passed in 0.98s
```

Next: implement IQ2_S (22) and IQ3_S (23) decoders so multi-layer MoE probe becomes numeric beyond layer 0.

## IQ2_S / IQ3_S decode support

Implemented real block decode support for current ggml expert quant types found in Qwen3-30B-A3B layer 1:

```text
GGML_TYPE_IQ2_S = 22, block_iq2_s = 82 bytes, QK_K=256
GGML_TYPE_IQ3_S = 23, block_iq3_s = 110 bytes, QK_K=256
source: ggml-org/ggml src/ggml-common.h + src/ggml-quants.c
```

Updated probes:

```text
decode-block
block-dot
matvec-row
expert-row
expert-forward-probe
moe-forward-probe
```

Real QXF smoke:

```text
decode-block blk.1.ffn_gate_exps.weight -> IQ2_S, sum -0.34015134, block 82 bytes
decode-block blk.1.ffn_down_exps.weight -> IQ3_S, sum 0.0227458477, block 110 bytes
moe-forward-probe --layers 2 now marks layer 1 selected experts supported=true
```

## Expert quant coverage + full MoE probe smoke

Added `expert-quant-coverage` to scan packed expert tensor quant types across all layers before deeper probes.

Corrected ggml enum handling after checking current `ggml/include/ggml.h`:

```text
17 = IQ2_XS
18 = IQ3_XXS
21 = IQ3_S
22 = IQ2_S
23 = IQ4_XS
```

Implemented IQ4_XS decode for ggml_type 23 using `block_iq4_xs` layout:

```text
block_iq4_xs = d f16 + scales_h u16 + scales_l[4] + qs[128] = 136 bytes
kvalues_iq4nl[16] from ggml-common.h
```

Real QXF coverage:

```text
complete_layers: 48
supported_layers: 48
unsupported_layers: 0
expert type triplets:
- gate/up/down: 17/17/18 = IQ2_XS/IQ2_XS/IQ3_XXS
- gate/up/down: 17/17/21 = IQ2_XS/IQ2_XS/IQ3_S
- gate/up/down: 17/17/23 = IQ2_XS/IQ2_XS/IQ4_XS
- gate/up/down: 22/22/21 = IQ2_S/IQ2_S/IQ3_S
- gate/up/down: 22/22/23 = IQ2_S/IQ2_S/IQ4_XS
```

Full 48-layer smoke:

```text
moe-forward-probe --layers 48 --top-k 2 --blocks 2 --seed 7
layers_run: 48
unsupported_selected: 0
moe_output_probe: -0.450067661
```

## Token forward scalar probe

Added `token-forward-probe` as the first token-level end-to-end scalar probe:

```text
token_embd.weight row lookup
raw embedding row checksum/projection
48-layer MoE probe
scalar residual-style aggregate
```

It is still explicitly not full inference:

```text
no attention
no RMSNorm
no real activation vector propagation
no lm_head/logits/sampler
```

Real QXF smoke:

```bash
build\qxqxf.exe token-forward-probe --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --token-id 42 --layers 48 --top-k 2 --blocks 2 --seed 7
```

Result summary:

```text
token_id: 42
embedding ggml_type: 12
embedding offset: 255438080
embedding row bytes: 1152
embedding checksum: 8543351503453051808
embedding_probe: -2.17405731
layers_run: 48
unsupported_selected: 0
moe_output_probe: 0.0562323582
token_output_probe: -2.11782495
```

## Q4_K decode + numeric embedding probe

Implemented `Q4_K` block decode for `token_embd.weight`:

```text
GGML_TYPE_Q4_K = 12
block_q4_K = d f16 + dmin f16 + scales[12] + qs[128] = 144 bytes
get_scale_min_k4 copied from current ggml logic
```

`decode-block` now supports Q4_K and `token-forward-probe` uses numeric embedding decode when row bytes contain whole Q4_K blocks.

Real QXF smoke:

```text
decode-block token_embd.weight block 0:
  decoder: Q4_K
  block_byte_size: 144
  values: 256
  sum: -0.254842281
  min: -0.140350342
  max: 0.0351467133
  raw_checksum: 15102224363924454741
```

Token 42 full probe after numeric embedding decode:

```text
embedding decoder: Q4_K
embedding numeric: true
embedding values: 2048
embedding_probe: -0.0960464004
layers_run: 48
unsupported_selected: 0
moe_output_probe: 0.0562323582
token_output_probe: -0.0398140422
```

## RMSNorm scalar probe

Added `rmsnorm-probe` and optional `--norm` integration in `token-forward-probe`.

Real tensors validated:

```text
blk.0.attn_norm.weight
  ggml_type: 0 (F32)
  dims: [2048]
  byte_size: 8192
```

Real QXF smoke:

```text
rmsnorm-probe --token-id 42 --norm blk.0.attn_norm.weight
  embedding_decoder: Q4_K
  values: 2048
  epsilon: 1e-06
  rms: 0.0144485799
  norm_weight_sum: 4.8323755
  normalized_probe: -0.434585256
```

`token-forward-probe` now accepts `--norm blk.0.attn_norm.weight` and uses the normalized scalar probe for router bias and final token scalar output.

```text
token-forward-probe token 42, 48 layers, top_k 2, blocks 2:
  rmsnorm.enabled: true
  rmsnorm.rms: 0.0144485799
  rmsnorm.normalized_probe: -0.434585256
  unsupported_selected: 0
  moe_output_probe: 0.0562323582
  token_output_probe: -0.378352898
```

## Attention scalar skeleton

Added `attention-probe` and optional `--attention-layer` integration in `token-forward-probe`.

Real tensors validated on layer 0:

```text
blk.0.attn_q.weight      ggml_type 23 IQ4_XS dims [2048, 4096]
blk.0.attn_k.weight      ggml_type 23 IQ4_XS dims [2048, 512]
blk.0.attn_v.weight      ggml_type 23 IQ4_XS dims [2048, 512]
blk.0.attn_output.weight ggml_type 23 IQ4_XS dims [4096, 2048]
```

Real smoke:

```text
attention-probe --layer 0 --blocks 2 --seed 7
  values: 512
  q.dot: -0.141278835
  k.dot: 0.343607893
  v.dot: 0.299770548
  o.dot: 0.0822988683
  attention_score_probe: -0.00214538508
  context_probe: -0.00064312326
  attention_output_probe: -5.29283165e-05
```

`token-forward-probe` with `--norm blk.0.attn_norm.weight --attention-layer 0` now includes:

```text
attention.enabled: true
attention.layer: 0
attention.values: 512
attention.output_probe: -5.29283165e-05
layers_run: 48
unsupported_selected: 0
token_output_probe: -0.378405826
```

Still not full attention: no KV cache materialization, no causal mask, no softmax, no real multi-token context yet.

## KV cache metadata planner/probe

Added `kv-cache-probe` and optional KV metadata output in `attention-probe`.

Real smoke:

```text
kv-cache-probe --ctx 4096 --kv int8 --token 42 --layer 0 --head 0
  layers: 48
  kv_heads: 4
  head_dim: 128
  bytes_per_value: 1
  bytes_per_k_or_v: 512
  bytes_per_token_per_layer: 1024
  token_stride: 1024
  layer_stride: 4194304
  total_bytes: 201326592
  total_gib: 0.187500
  k_offset: 43008
  v_offset: 43520
  layout: layer_token_kv_head_dim
```

`attention-probe --ctx 4096 --kv int8` now emits:

```text
kv_cache.enabled: true
kv_cache.bytes_per_token_per_layer: 1024
kv_cache.layer_stride: 4194304
kv_cache.total_bytes: 201326592
```

This is planner/offset metadata only. No KV buffer is allocated yet; no write/read of real cached K/V vectors yet.

## KV cache buffer write/read probe

Added `kv-cache-buffer-probe` for real heap allocation plus deterministic K/V head-slice write/readback.

Real INT8 smoke:

```text
kv-cache-buffer-probe --ctx 64 --kv int8 --token 7 --layer 1 --head 2 --seed 7
  allocated: true
  bytes_per_value: 1
  single_k_or_v_write_bytes: 128
  write_bytes: 256
  total_bytes: 3145728
  k_offset: 72960
  v_offset: 73472
  k_checksum: 12120104280957023034
  v_checksum: 11627941326168696797
  readback_ok: true
```

Real F16-layout smoke:

```text
kv-cache-buffer-probe --ctx 64 --kv f16 --token 7 --layer 1 --head 2 --seed 7
  allocated: true
  bytes_per_value: 2
  single_k_or_v_write_bytes: 256
  write_bytes: 512
  total_bytes: 6291456
  k_offset: 145920
  v_offset: 146944
  readback_ok: true
```

`attention-probe --ctx 64 --kv int8 --cache-write` now performs the same real KV cache allocation/write/readback and emits `kv_buffer.readback_ok: true`.

This is still a deterministic cache-write probe. It does not yet write real decoded K/V activations from Q/K/V matvec output.

## Attention cache activation probe

Added `attention-cache-probe`: writes quantized K/V activation probes for multiple tokens into the heap KV cache, reads the previous token back, and computes a scalar attention score from current Q against cached K plus cached V context.

Real smoke:

```text
attention-cache-probe --ctx 64 --kv int8 --layer 0 --tokens 2 --blocks 2 --seed 7
  tokens_written: 2
  current_token: 1
  attend_token: 0
  bytes_per_k_or_v: 512
  bytes_per_token_per_layer: 1024
  total_bytes: 3145728
  q_current_dot: 0.0121129454
  cached_k_mean: -0.124145508
  cached_v_mean: -1.01983643
  k_cache_checksum: 966308009196187445
  v_cache_checksum: 4500651025657082646
  cache_readback_ok: true
  attention_score_from_cache: -0.000132915547
  context_from_cache: 0.000135552117
  attention_output_from_cache: 0.0000111557858
```

This is the first KV path where attention reads a previous token from cache. It is still scalar/probe mode: Q/K/V are dot-probe activations, not full vectors; no softmax/causal mask over all prior tokens yet.

## Causal softmax over cached KV probes

Added `attention-softmax-probe`: writes K/V activation probes for multiple tokens, applies causal masking by attending only to prior tokens, computes stable softmax weights, aggregates cached V means, and applies output projection scalar probe.

Real smoke:

```text
attention-softmax-probe --ctx 64 --kv int8 --layer 0 --tokens 4 --blocks 2 --seed 7
  current_token: 3
  attend_count: 3
  causal_mask: true
  scores: [-0.00197251037, 0.0345237803, -0.0192334307]
  weights: [0.331119507, 0.34342737, 0.325453123]
  softmax_sum: 1
  cache_readback_ok: true
  context_from_softmax: -1.61556956
  attention_output_from_softmax: -0.132959547
```

This is causal softmax over cached scalar K/V probes. Still missing full vector Q/K/V, per-head vector attention, and full output projection.

## Partial vector attention over cached KV

Added `attention-vector-probe`: creates a partial per-head Q vector, writes cached INT8 K/V vectors for prior tokens, computes vector dot scores, stable softmax weights, and a partial context vector.

Real smoke:

```text
attention-vector-probe --ctx 64 --kv int8 --layer 0 --tokens 4 --dims 16 --seed 7
  head_dim: 128
  dims: 16
  scores: [-0.0593428586, -0.137967221, -0.312730095]
  weights: [0.370294342, 0.342295311, 0.287410346]
  softmax_sum: 1
  context_first8: [-1.45942988, -1.02282647, 0.911864696, 0.600216495, 1.54847737, 0.731573159, -1.43844144, 1.08945717]
  context_l2: 5.25518635
  attention_output_probe: 0.0693619383
```

This is the first vector-shaped attention path. It is still partial/probe mode: Q/K/V vectors are deterministic activations anchored by tensor dot probes, and output projection is still scalar.

## Multi-head attention loop over cached KV

Added `attention-multihead-probe`: runs a loop over Q heads, maps Q heads to grouped KV heads (`h % kv_heads`), computes per-head vector scores/softmax/context probes, and combines head outputs.

Real smoke:

```text
attention-multihead-probe --ctx 64 --kv int8 --layer 0 --tokens 4 --heads 4 --dims 16 --seed 7
  q_heads: 32
  kv_heads: 4
  heads_run: 4
  head_dim: 128
  dims: 16
  head_outputs: [-0.119485256, 0.122888545, -0.383002653, 0.0341028165]
  all_softmax_ok: true
  cache_readback_ok: true
  combined_output_probe: -0.0863741371
```

This covers grouped-query multi-head control flow. Still probe mode: no true full Q/K/V matvec vectors and no full output projection matrix multiply yet.

## Token-forward with multi-head attention probe

`token-forward-probe` now supports the multi-head attention probe path directly:

```text
token-forward-probe --token-id 42 --layers 2 --top-k 2 --blocks 2 --seed 7   --norm blk.0.attn_norm.weight   --multihead-attention --attention-layer 0 --attention-heads 4 --attention-dims 16
```

Real smoke:

```text
attention.mode: multihead
attention.values: 64
attention.score_probe: -0.000816980297
attention.output_probe: -0.0000672365538
attention.heads_run: 4
attention.dims: 16
layers_run: 2
unsupported_selected: 0
moe_output_probe: 0.00785388239
token_output_probe: -0.42679861
```

This makes the token path include embedding decode, optional RMSNorm, optional scalar/multi-head attention probe, and MoE probe. Logits/lm_head and sampler are still not implemented.

## Logits / lm_head top-token probe

Added `logits-probe` and `token-forward-probe --logits --top-n`.

Standalone real smoke over Qwen QXF:

```text
logits-probe --activation 0.125 --top-n 5 --scan 64 --seed 7
  lm_head_tensor: output.weight
  tied_embedding_fallback: false
  top_tokens: [45, 38, 58, 55, 61]
```

Token-forward real smoke:

```text
token-forward-probe ... --multihead-attention --attention-heads 4 --attention-dims 16 --logits --top-n 3
  token_output_probe: -0.42679861
  logits.enabled: true
  logits.lm_head_tensor: output.weight
  logits.top_tokens: [14, 36, 41]
```

Implementation notes:
- Uses `output.weight` when present, then `lm_head.weight`, else tied `token_embd.weight` fallback.
- `output.weight` real GGML type 14 / Q6_K is decoded by the Q6_K block decoder.
- This is top-token probe/search over scanned rows. Sampler not implemented.

## Q6_K decoder for output.weight

Implemented GGML type 14 / `Q6_K` block decode and connected it to `decode-block`, `block-dot`, and logits row scoring.

Real smoke:

```text
decode-block --name output.weight --block 0
  ggml_type: 14
  decoder: Q6_K
  block_byte_size: 210
  values: 256
  sum: 0.320626736
  min: -0.0617240667
  max: 0.0731544495
  first8: [-0.0272169113, 0.0427694321, -0.0602660179, -0.0174965858, 0.062210083, 0.0174965858, 0.00777626038, -0.00777626038]
```

Logits smoke now uses decoded Q6_K rows instead of raw-byte proxy:

```text
logits-probe --activation 0.125 --top-n 5 --scan 64 --seed 7
  lm_head_tensor: output.weight
  top_tokens: [17, 59, 63, 1, 4]
```

## Minimal sampler probe

Added `sampler-probe` and `token-forward-probe --sample`.

Standalone real smoke:

```text
sampler-probe --activation 0.125 --top-k 5 --scan 64 --temperature 0.7 --seed 7
  strategy: temperature_top_k
  selected_token: 59
  selected_rank: 1
  selected_prob: 0.205588569
  prob_sum: 1
  candidates: [17, 59, 63, 1, 4]
```

Token-forward real smoke:

```text
token-forward-probe ... --logits --top-n 3 --sample --temperature 0
  token_output_probe: -0.42679861
  logits.top_tokens: [52, 3, 57]
  sampler.strategy: argmax
  sampler.selected_token: 52
```

Sampler modes:
- `temperature <= 0`: deterministic argmax.
- `temperature > 0`: stable softmax over top-k candidates, deterministic seeded draw.

Still pending: tokenizer decode and real autoregressive generation loop.

## Tokenizer identity decode probe

Added `tokenizer-probe` and `token-forward-probe --decode-token`.

Real smoke:

```text
tokenizer-probe --token-id 52
  vocab: 151936
  source: fallback_token_id
  piece: <tok_52>
```

Integrated token-forward smoke:

```text
token-forward-probe ... --logits --top-n 3 --sample --temperature 0 --decode-token
  sampler.selected_token: 52
  decoded_token.enabled: true
  decoded_token.piece: <tok_52>
```

Important: current QXF tensor-copy path does not persist `tokenizer.ggml.tokens` arrays from GGUF metadata, so this is a reversible identity fallback, not real BPE/SentencePiece text decode yet.

Next hard step: persist tokenizer metadata from GGUF into QXF or sidecar, then decode real token pieces and build a minimal autoregressive loop.

## GGUF tokenizer sidecar export + real token piece decode

Added `tokenizer-export` and sidecar-backed `tokenizer-probe` / `token-forward-probe --decode-token --tokens`.

Synthetic RED/GREEN path:

```text
tokenizer-export --gguf mini.gguf --out tokens.tsv
  token_count: 64

tokenizer-probe --tokens tokens.tsv --token-id N
  source: sidecar
  piece: tokN
```

Real Qwen smoke:

```text
tokenizer-export --gguf Qwen3-30B-A3B-UD-IQ2_M.gguf --out qwen3-a3b.tokens.tsv
  token_count: 151936

tokenizer-probe --tokens qwen3-a3b.tokens.tsv --token-id 52
  source: sidecar
  piece: U

token-forward-probe ... --sample --temperature 0 --decode-token --tokens qwen3-a3b.tokens.tsv
  sampler.selected_token: 52
  decoded_token.source: sidecar
  decoded_token.piece: U
```

Sidecar format:

```text
# qx-tokenizer-v1
<token_id>	<escaped_piece>
```

This is now real GGUF `tokenizer.ggml.tokens` extraction into a sidecar. QXF itself still does not embed tokenizer metadata inside the `.qxf` container.

Next step: minimal generation loop over N tokens using sampled token -> decoded piece accumulation.

## Minimal generation loop probe

Added `generate-probe`: a bounded N-step probe loop that samples a token, decodes it through the tokenizer sidecar, appends text, and feeds the selected token into the next step's deterministic activation probe.

Synthetic test path verifies:

```text
generate-probe --tokens tokens.tsv --prompt-token 42 --steps 3 --top-k 3 --scan 32 --temperature 0
  probe: generate
  steps: 3
  tokens: 3 entries
  generated_text == concat(decoded pieces)
```

Real Qwen smoke:

```text
generate-probe --prompt-token 42 --steps 3 --top-k 3 --scan 64 --temperature 0 --tokens qwen3-a3b.tokens.tsv
  tokens: [10 '+', 31 '@', 34 'C']
  generated_text: +@C
```

Important: this is not full autoregressive inference yet. The selected token feeds the next step's probe activation, but the transformer residual/KV state is still not advanced through all real layers.

Next step: move from probe loop to real state loop skeleton: per-token KV append, layer iteration, and reuse sampled token as next input.

## State loop skeleton probe

Added `state-loop-probe`: a per-token / per-layer loop skeleton that appends deterministic K/V bytes into computed cache addresses, verifies checksum readback, samples the next token, decodes via tokenizer sidecar, and accumulates output text.

Synthetic test path verifies:

```text
state-loop-probe --prompt-token 42 --steps 3 --layers 2 --ctx 16 --kv int8 --top-k 3 --scan 32 --temperature 0
  probe: state_loop
  layers_run: 6
  kv_appends: 6
  cache_readback_ok: true
  generated_text == concat(decoded pieces)
```

Real Qwen smoke:

```text
state-loop-probe --prompt-token 42 --steps 3 --layers 2 --ctx 16 --kv int8 --tokens qwen3-a3b.tokens.tsv
  bytes_per_k_or_v: 512
  bytes_per_token_per_layer: 1024
  layer_stride: 16384
  tokens: [10 '+', 31 '@', 34 'C']
  layers_run: 6
  kv_appends: 6
  cache_readback_ok: true
  generated_text: +@C
```

Important: this is now a state-loop/control-flow skeleton with KV address/checksum proof, not full numerical autoregressive inference. Residual math is still probe-level and needs replacement by real layer residual updates.

Next step: replace deterministic KV append with real attention K/V vectors from Q/K/V projection probes, then persist/reuse KV across tokens.

## State loop with real projection-decoded KV

Added `state-loop-probe --real-kv`: the state loop can now fill K/V cache bytes from real attention projection tensors instead of deterministic skeleton bytes.

Implemented:

```text
--real-kv
  layer N -> blk.N.attn_k.weight / blk.N.attn_v.weight
  decode supported quant block
  quantize decoded float probe values into INT8 KV bytes
  write/checksum K/V cache slot
  emit tensor names + real value count per layer append
```

Decoder coverage expanded:

```text
GGML type 13 Q5_K, block size 176
```

Synthetic test path verifies `--real-kv` over available projection tensors:

```text
kv_source: projection_decode
kv_appends: 4
cache_readback_ok: true
k_tensor/v_tensor emitted
k_real_values/v_real_values > 0
```

Real Qwen smoke:

```text
state-loop-probe --prompt-token 42 --steps 2 --layers 2 --ctx 16 --kv int8 --real-kv
  kv_source: projection_decode
  layer0 tensors: blk.0.attn_k.weight / blk.0.attn_v.weight
  layer1 tensors: blk.1.attn_k.weight / blk.1.attn_v.weight
  k/v real values per append: 512
  layers_run: 4
  kv_appends: 4
  cache_readback_ok: true
  generated_text: +@
```

Important: K/V now comes from decoded projection tensor blocks, but the residual vector and full Q/K/V matvec are still probe-level. Next step is a real projection matvec path from residual vector into K/V cache slots.

## Projection matvec probe into KV

Added `projection-matvec-probe` and integrated `state-loop-probe --projection-matvec`.

Implemented:

```text
projection-matvec-probe
  residual probe vector (deterministic)
  -> decoded attn_k / attn_v quant rows
  -> partial dot product over --dims
  -> INT8 K/V bytes
  -> checksum output

state-loop-probe --real-kv --projection-matvec
  per token/per layer projection matvec fill
  writes K/V cache slots
  verifies checksum/readback
  samples selected token and decodes piece
```

Synthetic test path verifies:

```text
projection_matvec rows=4 dims=64
k_tensor/v_tensor are attn_k.weight / attn_v.weight
k_values/v_values == rows
state-loop kv_source == projection_matvec
k_matvec_values/v_matvec_values > 0
```

Real Qwen smoke:

```text
projection-matvec-probe --layer 0 --token-id 42 --rows 4 --dims 64 --kv int8
  k_tensor: blk.0.attn_k.weight
  v_tensor: blk.0.attn_v.weight
  k_ggml_type: 23
  v_ggml_type: 23
  k_values: 4
  v_values: 4
  k_checksum: 3554543661169652019
  v_checksum: 3554543661169652019

state-loop-probe --steps 2 --layers 2 --real-kv --projection-matvec
  kv_source: projection_matvec
  k_matvec_values/v_matvec_values: 512 per append
  layers_run: 4
  kv_appends: 4
  cache_readback_ok: true
  generated_text: +@
```

Important: K/V now comes from decoded projection rows dotted with a residual probe vector. This is closer to real inference, but residual is still synthetic/deterministic, not the true residual stream propagated through RMSNorm/attention/MLP.

Next step: replace deterministic residual probe with carried residual vector from embedding/RMSNorm and feed it through projection matvec.

## Residual vector probe into projection matvec

Added `residual-vector-probe` and integrated residual vectors into projection/state probes.

Implemented:

```text
residual-vector-probe
  token_embd.weight Q4_K decode
  -> partial embedding vector
  -> optional blk.0.attn_norm.weight F32 RMSNorm
  -> residual vector checksum / rms / l2

projection-matvec-probe --residual-vector --norm blk.0.attn_norm.weight
  residual vector from embedding/RMSNorm
  -> decoded attn_k / attn_v quant rows
  -> partial matvec dot
  -> INT8 K/V bytes

state-loop-probe --real-kv --projection-matvec --residual-vector
  per token residual vector
  -> per layer projection matvec
  -> KV append/readback
  -> sample/decode/carry token
```

Verification:

```text
cmd.exe /c build_msvc.bat
python -m pytest tests/test_gguf.py::test_create_qxf_tensor_copy_from_synthetic_gguf_if_built -q
python -m pytest tests -q
```

Result:

```text
1 passed in 2.23s
11 passed in 1.54s
```

Real Qwen smoke:

```text
residual-vector-probe --token-id 42 --norm blk.0.attn_norm.weight --dims 64
  rms: 0.0112107393
  checksum: 7643336349005103050

projection-matvec-probe --residual-vector
  residual_source: embedding_rmsnorm
  residual_values: 64
  k_values/v_values: 4

state-loop-probe --real-kv --projection-matvec --residual-vector
  residual_source: embedding_rmsnorm
  kv_source: projection_matvec
  layers_run: 4
  kv_appends: 4
  cache_readback_ok: true
  generated_text: +@
```

Important: residual is now derived from real embedding bytes plus F32 RMSNorm weights. It is still partial-width (64 dims by default) and not yet the carried full hidden-state vector through attention + MLP residual additions.

Next step: make the projection matvec use wider/full hidden slices and begin carrying a mutable residual vector through attention output / MoE output.

## Residual carry skeleton

Added `state-loop-probe --residual-carry --residual-dims N`.

Implemented:

```text
state-loop-probe --real-kv --projection-matvec --residual-vector --residual-carry
  token embedding Q4_K + RMSNorm F32 -> residual vector
  per layer projection matvec -> KV append
  compute attention_delta from K checksum
  compute moe_delta from V checksum
  residual += attention_delta + moe_delta wave
  emit residual_checksum_after
```

Verification:

```text
cmd.exe /c build_msvc.bat
python -m pytest tests/test_gguf.py::test_create_qxf_tensor_copy_from_synthetic_gguf_if_built -q
python -m pytest tests -q
```

Result:

```text
1 passed in 1.82s
11 passed in 1.70s
```

Real Qwen smoke:

```text
state-loop-probe --steps 2 --layers 2 --real-kv --projection-matvec --residual-vector --residual-carry --residual-dims 96
  residual_source: embedding_rmsnorm_carry
  residual_dims: 96
  layers_run: 4
  kv_appends: 4
  cache_readback_ok: true
  residual_checksum_after: present per token
  attention_delta/moe_delta: present per layer
  generated_text: +@
```

Important: residual carry is now mutable and layer-updated, but attention/MLP deltas are still checksum-derived skeleton deltas, not full numeric attention output projection or MoE vector output.

Next step: replace checksum-derived deltas with partial numeric attention output vector and partial MoE output vector.

## Numeric delta probes for residual carry

Added `state-loop-probe --numeric-deltas`.

Implemented:

```text
state-loop-probe --real-kv --projection-matvec --residual-vector --residual-carry --numeric-deltas
  residual vector from embedding Q4_K + RMSNorm F32
  -> projection matvec kprobe/vprobe from decoded attn_k/attn_v rows
  -> attention_delta = kprobe / residual_dims
  -> moe_delta = vprobe / (2 * residual_dims)
  -> residual vector mutable update
  -> emit numeric delta source fields
```

Verification:

```text
cmd.exe /c build_msvc.bat
python -m pytest tests/test_gguf.py::test_create_qxf_tensor_copy_from_synthetic_gguf_if_built -q
python -m pytest tests -q
```

Result:

```text
1 passed in 1.85s
11 passed in 1.78s
```

Real Qwen smoke:

```text
state-loop-probe --steps 2 --layers 2 --real-kv --projection-matvec --residual-vector --residual-carry --numeric-deltas --residual-dims 96
  delta_source: numeric_probe
  attention_delta_source: numeric_attention
  moe_delta_source: numeric_moe
  layers_run: 4
  kv_appends: 4
  cache_readback_ok: true
  generated_text: +@
```

Example real deltas:

```text
layer 0 step 0:
  attention_delta: -6.5152017e-05
  moe_delta: 1.68683883e-05
layer 1 step 0:
  attention_delta: 3.84761468e-05
  moe_delta: 7.38164518e-06
```

Important: deltas are now numeric from decoded projection matvec probe values rather than checksum-derived skeleton values. They are still scalar/vector-smeared deltas, not full attention output projection or MoE output vectors.

Next step: replace scalar numeric deltas with actual partial output vectors and add vector checksum/l2 per delta.

## Numeric delta vectors for residual carry

Added `state-loop-probe --delta-vectors`.

Implemented:

```text
state-loop-probe --numeric-deltas --delta-vectors
  scalar kprobe/vprobe deltas
  -> attention_delta_vector over residual_dims
  -> moe_delta_vector over residual_dims
  -> residual += attention_delta_vector + moe_delta_vector
  -> emit vector values/l2/checksum per layer
```

Verification:

```text
cmd.exe /c build_msvc.bat
python -m pytest tests/test_gguf.py::test_create_qxf_tensor_copy_from_synthetic_gguf_if_built -q
python -m pytest tests -q
```

Result:

```text
1 passed in 1.62s
11 passed in 1.70s
```

Real Qwen smoke:

```text
state-loop-probe --steps 2 --layers 2 --real-kv --projection-matvec --residual-vector --residual-carry --numeric-deltas --delta-vectors --residual-dims 96
  delta_source: numeric_vectors
  layers_run: 4
  kv_appends: 4
  cache_readback_ok: true
  generated_text: +@
```

Example layer output:

```text
attention_delta_vector_values: 96
moe_delta_vector_values: 96
attention_delta_vector_l2: 0.000390062848
moe_delta_vector_l2: 0.000100914877
attention_delta_vector_checksum: 7663906803160100772
moe_delta_vector_checksum: 6114503532878151341
```

Important: residual updates now use partial vectors, not only scalar-smearing. The vectors are still probe-derived from projection matvec scalars, not full dense attention output projection / MoE down projection vectors.

Next step: build actual partial attention output vector from cached K/V + output projection, then actual partial MoE output vector from selected experts.

## Partial attention output vector from KV cache

Added `state-loop-probe --attention-output-vector`.

Implemented:

```text
state-loop-probe --numeric-deltas --delta-vectors --attention-output-vector
  projection matvec writes K/V INT8
  -> partial attention output vector derived from persisted KV cache bytes
  -> residual += attention_output_vector + moe_delta_vector
  -> emit attention output values/l2/checksum/context/source
```

Verification:

```text
cmd.exe /c build_msvc.bat
python -m pytest tests/test_gguf.py::test_create_qxf_tensor_copy_from_synthetic_gguf_if_built -q
python -m pytest tests -q
```

Result:

```text
1 passed in 1.62s
11 passed in 2.12s
```

Real Qwen smoke:

```text
state-loop-probe --steps 2 --layers 2 --real-kv --projection-matvec --residual-vector --residual-carry --numeric-deltas --delta-vectors --attention-output-vector --residual-dims 96
  delta_source: attention_output_vector
  layers_run: 4
  kv_appends: 4
  cache_readback_ok: true
  generated_text: +@
```

Example layer output:

```text
attention_output_vector_values: 96
attention_output_vector_l2: 0.00248691075
attention_output_vector_checksum: 6987876588115690819
attention_context_tokens: 1
attention_output_source: kv_cache_partial
```

Important: residual carry now consumes a partial attention output vector derived from persisted KV cache data. It is still a probe approximation, not full Q projection + causal softmax + attn_output.weight dense projection.

Next step: implement actual Q projection + causal softmax over cached K/V for the attention output vector.

## Kernel optimization pass 1 + bench metrics

Research notes:

```text
llama.cpp/GGML quant kernels are block-based and performance depends on decoding/dotting quant blocks in tight backend-specific loops, minimizing allocation and dispatch overhead, and using SIMD/backend kernels where available. Current QX probe hot path was still doing per-row malloc/free and full-block decode before dot.
```

Changes:

```text
- Added stack-buffer raw block read helper for small quant blocks.
- Specialized IQ4_XS projection dot for prefix dims.
- Avoided per-row malloc/free in projection matvec hot path.
- Avoided full 256-value materialization for IQ4_XS when only prefix dims are needed.
- Added `state-loop-probe --bench` emitting elapsed_sec, tokens_per_second, ms_per_token, layer_steps, layer_steps_per_second.
```

Verification:

```text
cmd.exe /c build_msvc.bat
python -m pytest tests/test_gguf.py::test_create_qxf_tensor_copy_from_synthetic_gguf_if_built -q
python -m pytest tests -q
```

Result:

```text
1 passed in 0.79s
11 passed in 1.53s
```

Benchmark before pass 1, real Qwen smoke, 8 steps x 8 layers:

```text
wall_avg: 0.0588178 s
wall_tok_s_avg: 136.0 tok/s probe
wall_layer_steps_s_avg: 1088.1 layer-steps/s
```

Benchmark after pass 1, same command:

```text
wall_avg: 0.0550954 s
wall_tok_s_avg: 145.2 tok/s probe
wall_layer_steps_s_avg: 1161.6 layer-steps/s
```

Measured improvement:

```text
~6.8% wall throughput improvement on current probe path.
```

48-layer probe smoke after pass 1:

```text
4 steps x 48 layers
wall_tok_s: 17.95 tok/s probe
wall_layer_steps_s: 861.54 layer-steps/s
bench JSON inner loop: 33.06 tok/s CPU-clock, 1586.78 layer-steps/s
```

Important:

```text
This optimizes current probe kernels only. Full runtime still needs real Q projection, causal softmax, output projection, MoE expert matvec vectors, threading, SIMD intrinsics, and cache-aware tensor traversal before the number is meaningful as final model tok/s.
```

Next optimization target:

```text
Replace per-row random fseek/read with contiguous block-window caching and then add AVX2 dot kernels for Q4/Q5/Q6/IQ4 hot formats.
```



## Partial causal attention correctness slice

Added `state-loop-probe --causal-attention` as a correctness-first path. The flag enables the existing real projection/residual dependencies and now executes:

```text
embedding + RMSNorm residual (partial dimensions)
-> real Q/K/V quant projection matvec
-> dynamic per-vector INT8 K/V quantization with persisted scales
-> persistent K/V cache indexed by layer and token
-> Q·K / sqrt(d) scores over prior cached tokens
-> stable causal softmax
-> weighted V context
-> real attn_output.weight projection (partial dimensions)
-> residual carry
```

Evidence on the real `Qwen3-30B-A3B-UD-IQ2_M.qxf`, 2 tokens × 2 layers × 96 dimensions:

```text
delta_source=causal_attention
cache_readback_ok=true
layers_run=4
step 0: context_tokens=1, softmax_sum=1, k_scale=5.46202864e-06, v_scale=2.47443222e-06
step 1: context_tokens=2, softmax_sum=1, k_scale=3.22039887e-06, v_scale=2.95341988e-06
attention output checksums differ across steps
```

The synthetic GGUF fixture was corrected to use finite FP16 `d/dmin` values in Q4_K attention blocks. Arbitrary bytes can encode FP16 NaNs and are not a valid quantized fixture. Runtime projection dots and INT8 quantization also reject non-finite inputs defensively.

Verification:

```text
MSVC C17 /O2 /W4 build: exit 0, no warnings
focused causal/state-loop test: 1 passed
full suite: 11 passed
real Qwen causal smoke: PASS
```

Honesty boundary: this is still a 96-dimension partial attention slice. It does not yet implement full 2048 hidden dimensions, per-head RoPE/GQA layout, complete 48-layer MoE, full logits, or token equivalence against llama.cpp. Benchmark numbers from this path remain probe numbers, not final decode throughput.

## Qwen3 split-half RoPE + partial GQA

Added `state-loop-probe --rope-gqa-attention`. It extends the causal path with:

```text
Qwen3 split-half rotate_half convention
RoPE theta=1,000,000 on Q and K at the token position
independent causal softmax for every executed Q head
Q-head -> KV-head mapping with group_size=q_heads/kv_heads
per-head V context aggregation
```

The split-half layout was checked against the current Hugging Face Qwen3 implementation: `rotate_half` splits the final dimension into two halves and returns `(-x2, x1)`. The runtime therefore rotates index `i` with `i + head_dim/2`, not adjacent pairs.

Real model smoke, 2 tokens × 2 layers × 1,152 residual values:

```text
delta_source=rope_gqa_attention
cache_readback_ok=true
q_heads_total=32
kv_heads_total=4
head_dim=128
q_heads_run=9
gqa_group_size=8
kv_heads_touched=2
softmax_sum_min=1
softmax_sum_max=1
step-1 attention L2=4.74682024e-05
step-1 attention checksum=2560196670429878433
```

Verification:

```text
MSVC C17 /O2 /W4: PASS, no warnings
focused test: 1 passed
full suite: 11 passed
smoke_check.py: PASS
real Qwen smoke: PASS
```

Honesty boundary: nine Q heads and two KV heads are exercised, proving the GQA group boundary is crossed, but this is not yet full 32-head attention. Projection dots remain capped to the current 256-value quant block probe, RoPE has not yet been compared numerically against a llama.cpp/Hugging Face golden vector, and logits/tokens are not reference-equivalent.

## Independent RoPE/GQA golden vector

Added `rope-gqa-golden-probe` and an independent Python reference test. Both execute a deterministic two-token vector with Qwen3's split-half RoPE, dynamic INT8 K/V quantization, the 32-Q-head/4-KV-head GQA mapping, stable per-head softmax, and V context aggregation. The test compares C against Python for heads 0 and 8, deliberately crossing the GQA boundary from KV head 0 to KV head 1.

```text
score samples:
  -0.18515469696399092
  -0.2463259477409985
   0.42754526101969331
  -0.012447502910009959

weight samples:
  0.51528804576873499
  0.48471195423126512
  0.60825730653446175
  0.39174269346553819

context samples:
   0.38121691787450063
   0.20406551313809435
  -0.9999999962747097
   0.034675134601994395
```

Acceptance tolerance is `2e-6` for scores, probabilities, and context samples. Softmax sums remain within `1e-9` of one. This closes the synthetic math-equivalence gate; the next reference gate must feed real decoded Q/K/V projection values from the Qwen checkpoint into an external implementation rather than deterministic generated vectors.

## Real Q/K/V export and external attention reference

The reference audit exposed two prompt-fidelity bugs in the old projection path:

```text
old projection: one pseudo-random quant block per output row
correct projection: every contiguous block belonging to that tensor row

old projection: dot / sqrt(hidden)
correct projection: raw linear dot; only Q·K attention scores use / sqrt(head_dim)
```

The embedding path also repeated its first 256-value quant block. It now decodes consecutive blocks from the selected token row before RMSNorm. For this Qwen checkpoint:

```text
hidden/input dimensions: 2048
IQ4_XS blocks per Q/K/V projection row: 8
Q rows available: 4096
K/V rows available: 512 each
embedding Q4_K blocks per token row: 8
```

Added `real-qkv-golden-probe`. It exports full-precision probe outputs produced from real layer-0 Q/K/V tensors for tokens 42 and 43, then calculates split-half RoPE, dynamic INT8 K/V, GQA, causal scores, softmax, and V context in C. `tests/test_real_qkv_golden.py` consumes the exported raw Q/K/V and independently recomputes the attention math in Python.

Real checkpoint evidence, heads 0 and 8:

```text
scores:
   0.0008427251433244113
   0.0009048894307379170
  -0.0000356929314366093
  -0.0001055777371821491

weights:
  0.49998445892815163
  0.50001554107184840
  0.50001747120142930
  0.49998252879857075

context:
  -0.0011409460701757661
  -0.0011299446306302988
   0.0050463027742350690
  -0.0043131021436093610
```

The synthetic Q4_K row test independently decodes raw tensor bytes and proves that output rows map to contiguous tensor rows without hidden projection normalization. The real embedding test independently decodes consecutive Q4_K blocks and applies the checkpoint's RMSNorm weights.

Honesty boundary: the external Python attention reference starts from Q/K/V exported by the C projection kernel. It proves real-vector RoPE/GQA/softmax/context equivalence, but it does not independently decode the real checkpoint's IQ4_XS Q/K/V weights. That IQ4_XS full-row projection remains the next independent decoder gate.

## Independent IQ4_XS full-row gate + complete Q/KV head coverage

Added `tests/test_real_iq4xs_projection.py`, an independent Python IQ4_XS decoder based directly on the 136-byte GGML block layout:

```text
fp16 d
uint16 scales_h
4 packed low-scale bytes
128 packed IQ4_NL quants
256 decoded values per block
```

The test reads raw bytes directly from the 10.86 GB QXF and reconstructs complete 2,048-value rows from eight consecutive blocks. It independently reconstructs the token embedding and RMSNorm residual, then compares C projection results for:

```text
Q rows: 0, 1024, 4095
K rows: 0, 128, 511
V rows: 0, 511
```

Every checked row matches within `4e-6`. This covers the first, middle/GQA-boundary, and last output rows rather than only block-level checksums.

`real-qkv-golden-probe --q-heads-run 32` now executes:

```text
32/32 Q heads
4/4 KV heads
group size 8
4096 Q projection rows
512 K rows
512 V rows
8 IQ4_XS blocks per row
```

Full-head evidence:

```text
full_head_coverage=true
head 0 scores:  0.0008427251433244113, 0.0009048894307379170
head 31 scores: 0.0015161588008212766, 0.0009314100880941343
head 0 weights: 0.49998445892815163, 0.50001554107184840
head 31 weights: 0.50014618717401630, 0.49985381282598370
```

Verification after this gate:

```text
MSVC /O2 /W4: PASS
independent IQ4_XS test: PASS
full suite: 16 passed
smoke_check.py with 32Q/4KV: PASS
```

Honesty boundary: this closes layer-0 Q/K/V projection decoding and complete attention-head coverage for two tokens. It does not yet validate the full `attn_output.weight` 4096→2048 projection, integrate all 32 heads into the persistent multi-layer state loop, execute MoE, or compare final logits/tokens against llama.cpp.

## Full attention output projection and persistent full-head state loop

The real layer-0 correctness slice now carries the complete 4,096-value concatenated attention context through `blk.0.attn_output.weight` and emits all 2,048 output values. The tensor is real checkpoint IQ4_XS with dimensions `[4096, 2048]`, so each output row consumes 16 consecutive 136-byte IQ4_XS blocks.

`tests/test_real_iq4xs_projection.py` independently reads and decodes the raw QXF bytes for output rows `0`, `1024`, and `2047`, then computes each 4,096-term dot product in Python. C and Python match within `5e-6`.

Real checkpoint evidence:

```text
attention context values: 4096
attention context L2:     0.44175370877508735
output values:            2048
output L2:                0.8765631238759582
output FNV-1a checksum:   1851683731450459224
output samples:
  row 0:    -0.0142217353
  row 1024: -0.00186857965
  row 2047:  0.000908099697
```

The persistent `state-loop-probe` no longer derives Q-head coverage from the residual output width. It derives available heads from the real Q projection tensor and independently tracks attention-context width versus output width:

```text
Q heads run:                  32/32
KV heads touched:             4/4
output projection input:      4096
output projection output:     2048
attention output values:      2048
attention output L2:          0.928497412
attention output checksum:    12163708170389328135
```

Synthetic fixtures now report their actual tensor-backed coverage (`2` Q heads, `1` KV head, `256→256`) instead of claiming nine heads based only on requested residual dimensions.

Verification:

```text
MSVC C17 /O2 /W4: PASS
real IQ4_XS Q/K/V + output tests: 3 passed
full suite: 18 passed
smoke_check.py: PASS
```

Honesty boundary: attention projection for layer 0 is now independently gated end to end, and the state loop executes the full 32Q/4KV attention width plus 4096→2048 output projection. The loop still lacks the real post-attention residual/RMSNorm ordering, router/top-8 MoE forward, all-layer equivalence, full vocabulary logits, and token equality against a reference runtime.

## Real post-attention residual, router, and top-8 SwiGLU MoE

`real-qkv-golden-probe --full-moe` now executes the remaining layer-0 forward slice in model order:

```text
embedding residual + attention output
→ blk.0.ffn_norm.weight RMSNorm
→ blk.0.ffn_gate_inp.weight F32 router
→ softmax across 128 experts
→ top-8 probabilities without renormalization
→ IQ2_XS gate/up projections (2048→768)
→ SiLU(gate) * up
→ IQ3_XXS down projection (768→2048)
→ weighted sum of eight experts
→ post-attention residual + MoE output
```

The ordering and routing semantics were checked against Hugging Face Transformers `Qwen3MoeDecoderLayer` and `Qwen3MoeTopKRouter`: router softmax is computed over all experts, top-k is selected afterward, and `norm_topk_prob=False` leaves the selected probabilities unnormalized.

Independent Python gates now verify:

```text
embedding + attention residual add
blk.0.ffn_norm.weight RMSNorm
all 128 F32 router logits
softmax probabilities
top-8 expert IDs
eight routing weights
final residual addition at rows 0, 1024, 2047
```

Real layer-0 evidence for token 43:

```text
post-attention RMS: 0.024896273227477964
selected experts:   [49, 89, 48, 108, 4, 58, 92, 24]
routing weights:    [0.0859521696, 0.0831270745, 0.0571239175,
                     0.0470926749, 0.0394171348, 0.0378199365,
                     0.0280431577, 0.0246417165]
top-8 weight sum:   0.40321778211334025
MoE intermediate:  768
gate/up/down types: IQ2_XS / IQ2_XS / IQ3_XXS
MoE output L2:      1.2106010128440836
```

Layer output samples:

```text
row 0:    -0.0360694714
row 1024:  0.0302546993
row 2047:  0.0302934200
```

Verification:

```text
MSVC C17 /O2 /W4: PASS
real layer tests: 6 passed
full suite: 21 passed
smoke_check.py including --full-moe: PASS
```

Honesty boundary: this is a complete real forward slice for layer 0, but the packed IQ2_XS/IQ3_XXS expert math currently relies on the C decoders already exercised by block-level tests; unlike the router and residual arithmetic, complete expert rows do not yet have an independent Python decoder golden. The persistent state loop still uses probe MoE deltas rather than this full top-8 path. All 48 layers, final norm, full lm_head, tokenizer equivalence, and reference token equality remain open.

## 2026-08-17: real state propagation across layers 0 and 1

`state-loop-probe --full-moe` now executes the real attention and MoE path for two consecutive layers. Each layer performs its own attention RMSNorm, Q/K/V projections, per-head Q/K RMSNorm, split-half RoPE, GQA, dynamic INT8 KV, output projection, attention residual, FFN RMSNorm, global router softmax, unnormalized top-8 selection, SwiGLU experts and weighted residual. The Q/K normalization order is independently checked in Python against raw `blk.N.attn_q_norm.weight` and `blk.N.attn_k_norm.weight` values emitted by the real-model golden probe.

Measured one-token probe:

```text
elapsed: ~0.34 s for two layers
layer 0 experts: [49, 89, 92, 48, 108, 58, 4, 38]
layer 0 types:   IQ2_XS / IQ2_XS / IQ3_XXS
layer 0 MoE L2:  1.1315761998040383
layer 0 output checksum: 15037121990391945191

layer 1 input checksum:  15037121990391945191
layer 1 experts: [68, 13, 28, 73, 63, 32, 124, 114]
layer 1 types:   IQ2_S / IQ2_S / IQ4_XS
layer 1 MoE L2:  2.7100691687759846
layer 1 output checksum: 9586706624653950598
```

Verification:

```text
MSVC C17 /O2 /W4: PASS
pytest: 25 passed
smoke_check.py: PASS, real_two_layer_state=true
wiki lint: PASS
Auto Research project check: PASS
```

Honesty boundary: residual carry 0→1 is now real and checksum-gated, but layer 1 does not yet have an independent full-vector golden against a separate implementation. Layers 2–47, final norm, complete lm_head, tokenizer parity and identical greedy tokens remain open. The two-layer timing is not full-model decode throughput.

## 2026-08-17: one real token through all 48 layers

The existing full-MoE loop was already layer-generic. A new real-model regression gate now executes layers 0–47 and verifies every adjacent residual handoff without requiring a kernel change.

```text
layers_run: 48
kv_appends: 48
cache_readback_ok: true
all Q/K head norms present: true
all top-8 expert sets unique: true
all MoE output L2 values finite and positive: true
47/47 adjacent residual checksum links: PASS
first input checksum: 8017452295594298460
last output checksum: 675293441229675006
measured elapsed: ~8.50 s
```

Observed `(gate, up, down)` GGML type triples:

```text
(17, 17, 18)
(17, 17, 21)
(17, 17, 23)
(22, 22, 21)
(22, 22, 23)
```

Honesty boundary: this proves one-token state propagation and execution across all transformer layers. It does not include final RMSNorm, complete lm_head logits, reference-token equality or valid multi-token autoregression. The measured time is an instrumented probe duration, not token throughput.

## 2026-08-17: final RMSNorm and complete Q6_K output head

`state-loop-probe --full-moe --final-head` now consumes the real layer-47 residual, applies `output_norm.weight`, and computes all 151936 rows of `output.weight`.
Both tensor checksums are verified fail-closed before normalization or head evaluation.

```text
output_norm: F32 [2048]
output.weight: Q6_K type 14 [2048, 151936]
logits computed: 151936
argmax token: 1124
argmax logit: 11.739152169035485
final norm checksum: 1087599452263700755
logits F32 checksum: 17094101101096419516
logits RMS: 3.1986617278737643
warm measured probe: ~8.35 s
```

The independent gate recomputes final RMSNorm in Python and all vocabulary logits with llama.cpp's official `dequantize_row_q6_K`. It compares the FNV checksum of all 151936 F32 logits, top-5 and argmax. It initially failed with QX argmax `112567` versus reference `1124`, exposing a real Q6_K layout bug. The QX decoder was corrected to the official two-128-value/four-32-value layout; the complete checksum, top-5 and argmax then matched.

Honesty boundary: this validates the complete output head for QX's residual. It does not yet prove that every intermediate layer residual matches an external end-to-end Qwen3 implementation, and it does not implement tokenizer parity or valid multi-token autoregression.

## 2026-08-17: correct greedy multi-token state loop

`state-loop-probe --full-moe --final-head --steps 2` now feeds each selected token through its own `token_embd.weight` row at the next position while preserving per-layer dynamic INT8 KV. Every position executes all 48 transformer layers, final RMSNorm and all 151936 Q6_K output rows.

```text
input tokens: [42, 1124]
selected tokens: [1124, 29626]
positions: [0, 1]
layers_run: 96
kv_appends: 96
logits checksums: [17094101101096419516, 9438484627875866845]
```

The regression gate proves re-embedding by comparing the position-1 input checksum against an independent one-step run starting from token `1124`; it differs from the prior position's final residual. The second full-vocabulary checksum and argmax are independently recomputed with llama.cpp's official Q6_K decoder.

Honesty boundary: tokenizer/BPE is not implemented and QX's intermediate residuals are not yet matched end to end against another Qwen3 runtime. This is a correctness gate, not sustained decode throughput.
