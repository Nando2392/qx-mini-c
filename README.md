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

The fixed-token QX greedy gate is GREEN for `42 → 1124 → 50853` after the Q5_K decoder/attention fix. Fixed-prompt tokenizer parity against `llama-tokenize` is also GREEN. `Hello!` maps to `[9707, 0]`, prefills both positions and generates `[358, 1184]`. These two sequences exactly match the pinned llama.cpp F16 oracle; broader exact residual/logit and prompt coverage remains gated.

## Honest performance state

Measured on the current scalar CPU path:

```text
real layer-0 probe median: ~0.2085 s/layer
real one-token 48-layer state probe: ~8.50 s
real one-token 48-layer + complete output head probe: ~8.35 s warm run
```

The complete-head measurement includes final RMSNorm and all 151936 logits for one position. Multi-token execution is now correctness-gated but not benchmarked as sustained decode. It must not be reported as conversational tok/s. See [`wiki/concepts/performance-model.md`](wiki/concepts/performance-model.md).

CUDA is planned but **not implemented**.

Long-context experimentation is opt-in and gated. The default remains
`--long-context-policy none`; `--long-context-policy ctx4k-smoke` is only an
admission/provenance gate and requires `--ctx >= 4096`. It records
`target_ctx_tokens=4096`. `--long-context-rss-limit-bytes N` is also opt-in and
defaults to `0` (disabled); the Python benchmark harness fails closed when its
sampled `peak_rss_bytes` exceeds `N`. `--long-context-kv-quality-checks` remains
`0` only and non-zero values fail closed until a real KV-quality sweep exists.
`--long-context-soak-seconds` also remains `0` only and non-zero values fail
closed until a real soak runner exists. This is not an OS hard limit and does
not run KV-quality checks, run soak tests, promote defaults or claim
speed/quality. Benchmark compact-run records preserve the validated
`long_context_profile` for each measured run so evidence keeps the active
contract attached to the timings; per-cell summaries preserve it only when all
measured runs agree, and the top-level report preserves it only when all cells
agree. The report also records a `long_context_measurement` gate with measured
ctx, cell count, run count and RSS-summary presence; this is measurement
metadata, not a throughput or stability claim.
The same metadata records `rss_limit_bytes` and whether that sampled RSS gate
was active; negative report-level RSS limits fail closed, and this does not
install an OS-level hard limit.
The inactive `none` policy also rejects non-zero report-level RSS limits before
an active sampled-RSS gate can be reported.
Report-level aggregation rejects benchmark profiles whose `enabled` flag is not
exactly `true` before deriving common profile or measurement metadata.
It also validates every profile's policy-specific `disabled_reason`: `none`
requires `none_policy`, while `ctx4k-smoke` requires null.
Missing or unsupported profile policies fail closed in the same per-cell
validation seam before profile equality or measurement derivation.
RSS summaries with non-positive sample counts also fail closed before reporting
measured runs.
RSS summaries that omit the sample count fail closed with an explicit validation
error too.
RSS summaries must also be objects before count validation runs.
Empty benchmark reports fail closed before `long_context_measurement` derives
cell or run counts.
Benchmark cells must be objects before profile or measurement metadata is read.
Benchmark cells must also include `summary` explicitly before profile or
measurement metadata is derived.
Benchmark cells must also include `summary.long_context_profile` explicitly
before profile or measurement metadata is derived.
Inactive future counters remain fail-closed at report level too: non-zero
KV-quality checks or soak seconds are rejected until real sweeps/runners exist.

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

The build creates standalone residual/logit and sequence llama.cpp oracles; neither is linked into the QX runtime. The residual oracle writes selected layer inputs, internal attention/MoE checkpoints and final logits as lossless F32 sidecars. The sequence oracle keeps a persistent context for token-ID prompts and multiple greedy steps. QX can write matching sidecars with `state-loop-probe --full-moe --dump-residuals <existing-dir>` and can replay a layer suffix with `--start-layer N --residual-in layer-N.f32`. `scripts/compare_residuals.py` compares accumulated checkpoints; `scripts/compare_hybrid_residual_replay.py` separates incoming accumulated error from suffix error with exact size/finite/metadata gates; `scripts/scaled_residual_replay.py` prepares, runs and revalidates one scaled suffix cell; `scripts/scaled_residual_matrix.py` runs and revalidates the token/activation/KV matrix while separating router order from top-k membership; `scripts/compare_layer_sensitivity.py` supports perturbation analysis and same-input chains; `scripts/compare_logits.py` compares the complete vocabulary and both argmax values. Local model paths and sidecars must remain outside Git.

Current fixed-token result against llama.cpp commit `768d2a481a99cb75ec9a03b95dadbd35e7acf496`: exact end-to-end numerical parity remains **refuted**. The backward same-input sweep closes every block materially, and the modal-equivalent F16 hybrid replay attributes the remaining trajectory to accumulated/global error plus strong downstream amplification of a microscopic layer-0 discrepancy. The scaled follow-up refutes a smooth scalar response: exact-direction scales `-1` and `+1` produce final deltas of `5.73359e-4` and `4.46778` from equal input L2. The 18-cell token/activation/KV extension shows that this response is token- and modality-dependent: all three `q8_k_compat` + F16 cells retain top-8 membership, while membership transitions appear in 14/18 cells across the broader F32/F32-KV/INT8 controls. No numerical fix is authorized. The opt-in final head uses `Q6_K × Q8_K`; F32 remains the default dequantized runtime path. See [`wiki/comparisons/scaled-residual-token-modality-matrix.md`](wiki/comparisons/scaled-residual-token-modality-matrix.md), [`wiki/comparisons/scaled-layer1-residual-sensitivity.md`](wiki/comparisons/scaled-layer1-residual-sensitivity.md), and [`wiki/comparisons/hybrid-residual-replay-accumulation.md`](wiki/comparisons/hybrid-residual-replay-accumulation.md).

The final pre-head residual is explicitly compared, not inferred. Post-Q6_K Q8_K/F32-KV gives max-abs `4.68347`, RMSE `0.172092`, cosine `0.999995600`. This is worse than the prior Q5_K-only global baseline despite the layer-46 same-input fix, and is documented as quantized-path sensitivity rather than claimed as a global improvement.

Two-token sequence parity is now GREEN for the fixed matrix. For token-ID prompt `[42]`, llama F16/Q8_0 and QX F32/INT8 generate `[1124, 50853]`. For `Hello!` (`[9707, 0]`), QX generates `[358, 1184]`, matching llama F16; llama Q8_0 remains `[358, 614]`. This is not exhaustive prompt or exact-logit parity.

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

1. Add an accumulated KV snapshot/replay seam before making multi-token perturbation claims.
2. Expand tokenizer parity beyond the fixed prompt matrix and add chat-template application.
3. Add QXF mmap and persistent scratch buffers.
4. Add fused quant-dot, thread pool and AVX2 CPU kernels.
5. Add a hybrid CUDA backend with dense residency and expert cache.
6. Run context 4K, RSS, quality and sustained thermal gates.

## License

MIT for repository code and documentation. Third-party model weights and external reference implementations retain their own licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
