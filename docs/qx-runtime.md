---
name: qx-mini-c-runtime
description: "QXF/GGUF tensor converter for Qwen3-30B-A3B models."
version: 0.1.0
author: Fernando Martinez (fjmn2392), Hermes Agent
license: MIT
platforms: [windows, linux, macos]
metadata:
  hermes:
    tags: [llm, tensor, format, qwen, converter]
    related_skills: [llama-cpp, huggingface-hub]
---

# QX Mini-C Runtime

## Overview

`qx-mini-c` is a minimal C runtime for creating and validating QXF tensor files from GGUF sources. Target: Qwen3-30B-A3B quantized models running on low-RAM/VRAM hardware (14GB RAM + 5GB VRAM).

## When to Use

- You need to convert GGUF quantized models to QXF format for local inference
- You're deploying Qwen3-30B-A3B on constrained hardware
- You need tensor-copy verification to ensure data integrity

## Key Tools

- `qxqxf` — CLI for QXF creation/inspection
- `create-from-gguf-copy` — Copy tensors from GGUF to QXF with 4096-byte alignment
- `inspect` — Validate QXF binary structure
- `inspect-gguf` — Parse GGUF header and tensor table

## Building

MSVC on Windows:

```batch
build_msvc.bat
```

Linux/macOS:

```bash
# Uses MSVC cross-compile or gcc/clang if available
```

## Usage

### Create QXF from GGUF (tensor copy)

```bash
./build/qxqxf.exe create-from-gguf-copy \
  --in tests/fixtures/qwen3-30b-a3b-mini.gguf \
  --model qwen3-30b-a3b \
  --quant q2 \
  --out models/qwen3-30b-a3b-copy.qxf
```

### Inspect QXF

```bash
./build/qxqxf.exe inspect --in models/qwen3-30b-a3b-copy.qxf
```

### Create synthetic test fixture

```bash
python scripts/make_synthetic_gguf.py
```

### Verify copy integrity

```bash
python scripts/verify_qxf_copy.py \
  tests/fixtures/qwen3-30b-a3b-mini.gguf \
  models/qwen3-30b-a3b-copy.qxf \
  32768
```

## QXF Format

Binary container for quantized tensors:

```
Offset 0-7:    Magic "QXF1" + version (uint32) + model_type (uint32)
Offset 8-15:   quant_type + layers + hidden + intermediate (all uint32)
Offset 16-23:  q_heads + kv_heads + head_dim + vocab (all uint32)
Offset 24-31:  tensor_count + dir_offset + data_offset + file_size (all uint32)
Offset 32+:    Tensor directory (name entries + data)
              Each tensor entry: name_len, name, rank, dims[N], type, offset (uint64)
              Tensors aligned to 4096 bytes
```

## Tensor Copy Workflow

1. Parse GGUF header to extract tensor table
2. Compute byte spans from GGUF offsets
3. Read GGUF tensor data
4. Align each tensor to 4096 bytes in QXF output
5. Write checksum (FNV-1a) back into QXF directory
6. Verify file_size matches expected

## Pitfalls

- GGUF tensor data layout differs from QXF alignment strategy
- Tensor checksum must be written AFTER data copy completes
- MSVC struct packing differs from GCC/Clang — use `#pragma pack(push, 1)` in headers
- Qwen3-30B-A3B MoE routing paths require specialized loader support

## References

- `references/gguf-spec.md` — GGUF format reference
- `references/qwen3-shapes.md` — Qwen3 tensor naming conventions