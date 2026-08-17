# QX-mini-MoE

Experimental C runtime and mmap-oriented model format for correctness-first local inference of **Qwen3-30B-A3B MoE**.

> Status: research runtime. Layer 0 executes real attention and top-8 MoE weights, but complete 48-layer token generation is not finished. Historical probe throughput is not full-model throughput.

## Goals

- Own C runtime rather than wrapping an inference framework.
- Convert a real GGUF into the QXF1 tensor-copy container.
- Validate every numerical slice independently before optimization.
- Run Qwen3-30B-A3B on constrained RAM/VRAM using expert slicing and caching.
- Keep Python outside the final inference path; use it for tests and references.

## Implemented path

```text
GGUF tensor bytes
→ QXF1 loader/checksum
→ embedding Q4_K
→ attention RMSNorm
→ Q/K/V IQ4_XS
→ Qwen3 split-half RoPE
→ dynamic INT8 KV
→ GQA 32Q/4KV + causal softmax
→ attention output 4096→2048
→ attention residual + FFN RMSNorm
→ F32 router + top-8
→ IQ2_XS gate/up + SwiGLU
→ IQ3_XXS down
→ weighted MoE sum + layer-0 residual
```

The next correctness gate is persistent real residual propagation across layers 0 and 1, then all 48 layers, final RMSNorm, full lm_head, tokenizer parity and identical greedy tokens.

## Honest performance state

Measured on the current scalar CPU path:

```text
real layer-0 probe median: ~0.2085 s/layer
48-layer lower-bound extrapolation: ~0.10 token/s
```

The extrapolation excludes final norm, complete lm_head and tokenizer. It is not completed inference. See [`wiki/concepts/performance-model.md`](wiki/concepts/performance-model.md).

CUDA is planned but **not implemented**.

## Repository map

```text
include/                 public C headers
src/                     QXF/GGUF/runtime implementation
scripts/                 model metadata, conversion and verification helpers
tests/                   synthetic and real-model numerical gates
docs/                    chronological bootstrap evidence
wiki/                    Obsidian/Karpathy-style project knowledge base
smoke_check.py            deterministic project smoke gate
```

Open `wiki/` as an Obsidian vault. Start at [`wiki/index.md`](wiki/index.md).

## Build

### Windows / MSVC

```bat
build_msvc.bat
```

### Make-compatible C toolchain

```bash
make
```

## Tests

```bash
python -m pytest tests -q
python smoke_check.py
```

Tests requiring the real 10+ GB model skip when local model files are absent. Synthetic fixtures remain in the repository.

An optional external decoder gate links a small test helper against a local llama.cpp build. Build it only when that checkout is available:

```bat
testsuild_ggml_reference.bat
```

## Model setup

Model weights are intentionally excluded from Git. Do not commit GGUF, QXF or provider credentials.

```bash
bash scripts/download_qwen30b_iq2m.sh
build/qxqxf.exe create-from-gguf-copy   --in models/Qwen3-30B-A3B-UD-IQ2_M.gguf   --model qwen3-30b-a3b   --quant q2   --out models/Qwen3-30B-A3B-UD-IQ2_M.qxf
```

Verify licenses and the source model card before redistributing model artifacts. This repository distributes code and documentation, not weights.

## Research method

The project uses a Karpathy-inspired Auto Research loop:

```text
question → falsifiable hypothesis → RED/baseline → minimal change
→ real execution → compare → keep/revert → document
```

See [`wiki/concepts/auto-research-loop.md`](wiki/concepts/auto-research-loop.md).

## Roadmap

1. Persistent real attention+MoE state loop.
2. Full 48-layer forward, final norm and lm_head.
3. Tokenizer parity and identical greedy-token gate.
4. QXF mmap and persistent scratch buffers.
5. Fused quant-dot, thread pool and AVX2 CPU kernels.
6. Hybrid CUDA backend with dense residency and expert cache.
7. Context 4K, RSS, quality and sustained thermal gates.

## License

MIT for repository code and documentation. Third-party model weights and external reference implementations retain their own licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
