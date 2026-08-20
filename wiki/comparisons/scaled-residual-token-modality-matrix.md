---
title: Scaled residual token and modality matrix
created: 2026-08-19
updated: 2026-08-19
type: comparison
tags: [qwen3-moe, residual, replay, sensitivity, routing, tokens, activation, kv]
sources: [state-loop-probe, llama.cpp]
confidence: high
---

# Scaled residual token and modality matrix

Issue: [#19](https://github.com/Nando2392/qx-mini-c/issues/19)

## Question

Does the layer-1 suffix response observed in [[scaled-layer1-residual-sensitivity]] persist across token IDs, activation modes, and KV formats?

## Verdict

The response is **token- and modality-dependent**. The full matrix completed and revalidated all 18 cells:

- 17/18 cells have at least one top-8 rank-order transition.
- 14/18 cells have at least one top-8 membership transition.
- The runtime-aligned `q8_k_compat` + KV F16 slice has **zero membership transitions in all three tokens**. Token `42` and token `0` have order-only transitions; token `9707` has no top-8 transition.
- Every F32-activation cell changes membership. Under `q8_k_compat`, membership appears in 0/3 F16 cells, 2/3 F32 cells, and 3/3 INT8 cells.
- Equal-magnitude `-1` and `+1` perturbations remain strongly asymmetric for some runtime-aligned cells: the final L2 ratio is `7792.3×` for token `42` and `105030×` for token `0`, while token `9707` is much closer at `1.54585×`.

This does not authorize a numerical runtime change. It narrows the result: the extreme suffix response is reproducible for some tokens, but membership instability is concentrated in activation/KV control modes rather than the three-cell `q8_k_compat` + F16 slice.

## Fixed experiment

Each cell is an independent one-token, position-zero run.

- Tokens: `42`, `9707`, `0`.
- Activation modes: `f32`, `q8_k_compat`.
- QX KV modes: `f16`, `f32`, `int8`.
- llama.cpp oracle KV mapping: `f16 → f16`, `f32 → f32`, `int8 → q8_0`.
- Perturbation scales: `-16,-8,-4,-2,-1,-0.5,-0.25,0,0.25,0.5,1,2,4,8,16`.
- Replay boundary: residual input to layer 1; suffix layers `1..47`.
- Direction per cell: QX layer-0 output minus llama.cpp layer-1 input.
- Control: scale zero is the per-cell suffix baseline.
- Full MoE, `expected_count=2048`, `ctx=4`, seed `7`.

The runner generated 9 llama.cpp oracles, 18 QX baselines, and 270 suffix replays. It records exact artifact hashes, commands, source-output hashes, cell-report hashes, response curves, routing-order transitions, and routing-membership transitions.

## Reproduction

Build the QX runtime and independent llama.cpp oracle:

```bat
build_msvc.bat
call tests\build_llama_reference_oracle.bat
```

Run the fixed matrix:

```bat
C:\Users\fjmn2\Dev\hermes-agent\.venv\Scripts\python.exe scripts\scaled_residual_matrix.py run ^
  --qxqxf build\qxqxf.exe ^
  --model models\Qwen3-30B-A3B-UD-IQ2_M.qxf ^
  --llama-oracle build\llama_reference_oracle.exe ^
  --gguf models\Qwen3-30B-A3B-UD-IQ2_M.gguf ^
  --experiment-dir %LOCALAPPDATA%\Temp\qx-scaled-matrix-issue19-v3 ^
  --tokens=42,9707,0 ^
  --activations=f32,q8_k_compat ^
  --kv-formats=f16,f32,int8 ^
  --scales=-16,-8,-4,-2,-1,-0.5,-0.25,0,0.25,0.5,1,2,4,8,16 ^
  --layers 48 --start-layer 1 --expected-count 2048 --ctx 4 --seed 7
```

Revalidate existing outputs fail-closed:

```bat
C:\Users\fjmn2\Dev\hermes-agent\.venv\Scripts\python.exe scripts\scaled_residual_matrix.py analyze ^
  --experiment-dir %LOCALAPPDATA%\Temp\qx-scaled-matrix-issue19-v3 ^
  --qxqxf build\qxqxf.exe ^
  --model models\Qwen3-30B-A3B-UD-IQ2_M.qxf ^
  --llama-oracle build\llama_reference_oracle.exe ^
  --gguf models\Qwen3-30B-A3B-UD-IQ2_M.gguf ^
  --output %LOCALAPPDATA%\Temp\qx-scaled-matrix-issue19-v3\matrix-report.json
```

The analyzer requires four explicit trusted artifact paths before reading the manifest-selected artifacts. It rejects path mismatches, non-object or non-finite JSON, JSON booleans in numeric fields, incomplete or duplicate cells/scales, changed artifact or report hashes, mismatched token/mode metadata, partial metric payloads, inconsistent routing counts, and membership transitions that are not also order transitions.

## Provenance

| Artifact | SHA-256 |
|---|---|
| `build/qxqxf.exe` | `57e9ead95ffd983b157195bb0ac094b83bb40177274498b8fa4fd2948aeba391` |
| QXF model | `5609589e45a610bee6699f336109f3231326850d8f1ca839c614667c2f439840` |
| `build/llama_reference_oracle.exe` | `350133f640e91c15cf41d184e15aec079852851a99de49f424107c31ea680cb9` |
| GGUF model | `c8c2dc330dd1ec0c72c31b12e318647e6f9e0c773b9123eccfc3d12d9acc6652` |
| `matrix-manifest.json` | `518c054b4ba2a80e226c8e7ff5274191775f00241087c54e28098e2d3ac23bf3` |
| Revalidated `matrix-report.json` | `c97e87b97b87f9baa02a4c160189b338ba403720b3108299b577da35a22be14e` |

The report and manifest are under `%LOCALAPPDATA%\Temp\qx-scaled-matrix-issue19-v3` and are intentionally not committed because they contain machine-local absolute paths.

## Results

`ratio` is `max(L2(-1), L2(+1)) / min(L2(-1), L2(+1))`. The last two columns count the tested scales with at least one transition; they do not count layers.

| Token | Activation | KV | Direction L2 | `-1` final L2 | `+1` final L2 | ratio | order scales | membership scales |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 42 | `f32` | `f16` | 0.0318493 | 2.74468 | 2.23623 | 1.22737 | 11 | 8 |
| 42 | `f32` | `f32` | 0.0318244 | 2.58913 | 2.38295 | 1.08652 | 11 | 8 |
| 42 | `f32` | `int8` | 0.0375749 | 5.86944 | 3.28652 | 1.78591 | 14 | 8 |
| 42 | `q8_k_compat` | `f16` | 6.64213e-07 | 0.000573359 | 4.46778 | 7792.3 | 10 | 0 |
| 42 | `q8_k_compat` | `f32` | 0.00638465 | 7.80632 | 7.9507 | 1.0185 | 12 | 0 |
| 42 | `q8_k_compat` | `int8` | 0.0242635 | 15.76 | 6.80371 | 2.31639 | 13 | 4 |
| 9707 | `f32` | `f16` | 0.0326592 | 2.99091 | 3.00462 | 1.00458 | 7 | 3 |
| 9707 | `f32` | `f32` | 0.0326675 | 2.97876 | 3.10453 | 1.04222 | 7 | 3 |
| 9707 | `f32` | `int8` | 0.034035 | 1.77924 | 3.274 | 1.84011 | 9 | 4 |
| 9707 | `q8_k_compat` | `f16` | 2.88978e-07 | 1.11001e-05 | 1.7159e-05 | 1.54585 | 0 | 0 |
| 9707 | `q8_k_compat` | `f32` | 0.0106756 | 10.3048 | 10.0947 | 1.02081 | 4 | 2 |
| 9707 | `q8_k_compat` | `int8` | 0.0201834 | 6.88774 | 5.23885 | 1.31474 | 10 | 3 |
| 0 | `f32` | `f16` | 0.0411977 | 1.74746 | 1.63802 | 1.06681 | 12 | 12 |
| 0 | `f32` | `f32` | 0.0411866 | 1.64737 | 1.64648 | 1.00054 | 12 | 12 |
| 0 | `f32` | `int8` | 0.0407679 | 2.19525 | 3.40586 | 1.55147 | 11 | 9 |
| 0 | `q8_k_compat` | `f16` | 9.88581e-07 | 2.05782 | 1.95927e-05 | 105030 | 3 | 0 |
| 0 | `q8_k_compat` | `f32` | 0.0158927 | 12.1374 | 18.4776 | 1.52237 | 13 | 11 |
| 0 | `q8_k_compat` | `int8` | 0.0288338 | 12.2111 | 10.4421 | 1.16941 | 14 | 11 |

### Runtime-aligned F16 slice

The `q8_k_compat` + F16 slice is the direct extension of Issue #18:

- Token `42` reproduces the prior direction L2 `6.64213e-7`, the `-1/+1` asymmetry, and order-only transitions. Transition scales are `-16,-8,-4,-2,0.25,1,2,4,8,16`; membership remains unchanged.
- Token `9707` has a smaller direction (`2.88978e-7`), no top-8 transition, and final L2 values `1.11001e-5` and `1.7159e-5` at `-1/+1`.
- Token `0` has direction L2 `9.88581e-7`. Scale `-1` reaches final L2 `2.05782`, while `+1` reaches only `1.95927e-5`; only scales `-16,-8,-4` produce order changes, at layer 47, and membership stays fixed.

Therefore, “no membership change” generalizes across this three-token runtime-aligned slice, but the response magnitude and rank-order behavior do not generalize uniformly by token.

### Modal controls

The controls materially change the direction being injected: direction L2 rises from roughly `1e-7..1e-6` in the runtime-aligned F16 slice to `0.00638..0.0412` in many F32/INT8 cells. Membership changes in all nine F32-activation cells, in both affected `q8_k_compat` + F32 cells, and in all three `q8_k_compat` + INT8 cells.

Those controls are evidence of modality sensitivity, not evidence that the normal F16 path has unstable top-8 membership. They must not be pooled into a single runtime verdict.

## Causal verdict and next gate

- **Universal smooth scalar response:** refuted; the `-1/+1` response ratio ranges from near `1` to over `100000` across the fixed cells.
- **Universal top-8 membership stability:** refuted across all controls, but retained for the tested `q8_k_compat` + F16 slice.
- **Token independence:** refuted; the three runtime-aligned tokens have different response and order-transition patterns.
- **Numerical runtime fix:** not authorized. The matrix separates modes but does not isolate a single defective kernel or threshold.
- **Multi-token inference claim:** not authorized. Every cell is position zero with fresh KV.

The next safe gate is an accumulated KV snapshot/replay seam, followed by multi-token perturbation tests that hold the prior-token KV state fixed. Related evidence: [[scaled-layer1-residual-sensitivity]], [[hybrid-residual-replay-accumulation]], and [[current-status-and-roadmap]].
