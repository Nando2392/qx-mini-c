# QX-mini-MoE

Experimental C runtime and mmap-oriented model format for correctness-first local inference of **Qwen3-30B-A3B MoE**.

> Status: research runtime. Qwen3 GPT-2/Qwen2 BPE parity is GREEN for fixed ASCII, Unicode, whitespace and ChatML prompts. QXF1 now rejects malformed manifests, directories, dimensions, placements, overlaps and legacy truncated rows fail-closed. The C loop prefills IDs, then re-embeds greedy outputs with persistent per-layer INT8 KV across all 48 layers and the complete 151936-row Q6_K head. Exhaustive Unicode/chat-template coverage and external end-to-end residual parity are not finished. Probe timing is not conversational decode throughput.

## Goals

- Own C runtime rather than wrapping an inference framework.
- Convert a real GGUF into the QXF1 tensor-copy container.
- Validate every numerical slice independently before optimization.
- Run Qwen3-30B-A3B on constrained RAM/VRAM using expert slicing and caching.
- Keep Python outside the final inference path; use it for tests and references.

## Implemented path

```text
text prompt
→ QXT2 tokenizer sidecar: Qwen2 pre-tokenizer + GPT-2 byte BPE
→ exact token IDs
→ GGUF tensor bytes copied into QXF1
→ QXF1 loader/checksum
→ embedding Q4_K
→ attention RMSNorm
→ Q/K/V IQ4_XS
→ per-head Q/K RMSNorm
→ Qwen3 split-half RoPE
→ dynamic INT8 KV
→ GQA 32Q/4KV + causal softmax
→ attention output 4096→2048
→ attention residual + FFN RMSNorm
→ F32 router + top-8
→ IQ2_XS gate/up + SwiGLU
→ IQ3_XXS down
→ weighted MoE sum + residual propagation across layers 0–47
→ final RMSNorm F32
→ output.weight Q6_K, 2048→151936
→ complete logits + argmax/top-N
→ selected token embedding at the next position
→ persistent KV update and repeated 48-layer forward
```

The fixed-token greedy gate is GREEN for `42 → 1124 → 29626`. Fixed-prompt tokenizer parity against `llama-tokenize` is also GREEN. `Hello!` maps to `[9707, 0]`, prefills both positions, then produces two QX greedy tokens through the verified loop. External end-to-end residual/sequence comparison remains pending.

## Honest performance state

Measured on the current scalar CPU path:

```text
real layer-0 probe median: ~0.2085 s/layer
real one-token 48-layer state probe: ~8.50 s
real one-token 48-layer + complete output head probe: ~8.35 s warm run
```

The complete-head measurement includes final RMSNorm and all 151936 logits for one position. Multi-token execution is now correctness-gated but not benchmarked as sustained decode. It must not be reported as conversational tok/s. See [`wiki/concepts/performance-model.md`](wiki/concepts/performance-model.md).

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
tests\build_ggml_reference.bat
tests\build_llama_reference_oracle.bat
```

The second helper is a standalone llama.cpp oracle; it is not linked into the QX runtime. It writes selected layer-input residuals and final logits as lossless F32 sidecars. QX can write matching sidecars with `state-loop-probe --full-moe --dump-residuals <existing-dir>`, and `scripts/compare_residuals.py` reports max absolute error, RMSE, cosine similarity and the first divergent layer. Local model paths and sidecars must remain outside Git.

Current fixed-token baseline against llama.cpp commit `768d2a481a99cb75ec9a03b95dadbd35e7acf496`: token `42` has an exact layer-0 input residual and both runtimes select argmax `1124`, but the first residual divergence appears at layer 1. llama.cpp F16 and Q8_0 KV produce the same boundary, so exact end-to-end residual parity is not yet claimed.

## Model setup

Model weights are intentionally excluded from Git. Do not commit GGUF, QXF or provider credentials.

```bash
bash scripts/download_qwen30b_iq2m.sh
build/qxqxf.exe create-from-gguf-copy   --in models/Qwen3-30B-A3B-UD-IQ2_M.gguf   --model qwen3-30b-a3b   --quant q2   --out models/Qwen3-30B-A3B-UD-IQ2_M.qxf
python scripts/export_qwen3_tokenizer.py --gguf models/Qwen3-30B-A3B-UD-IQ2_M.gguf --out models/Qwen3-30B-A3B.qxt
build/qxqxf.exe tokenizer-encode --tokenizer models/Qwen3-30B-A3B.qxt --text-file prompt.txt
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

1. Compare residuals and greedy sequence end-to-end against an external Qwen3 runtime.
2. Expand tokenizer parity beyond the fixed prompt matrix and add chat-template application.
3. Add QXF mmap and persistent scratch buffers.
4. Add fused quant-dot, thread pool and AVX2 CPU kernels.
5. Add a hybrid CUDA backend with dense residency and expert cache.
6. Run context 4K, RSS, quality and sustained thermal gates.

## License

MIT for repository code and documentation. Third-party model weights and external reference implementations retain their own licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
