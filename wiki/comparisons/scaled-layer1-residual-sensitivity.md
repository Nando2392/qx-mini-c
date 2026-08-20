---
title: Scaled layer-1 residual suffix sensitivity
created: 2026-08-19
updated: 2026-08-19
type: comparison
tags: [qwen3-moe, residual, replay, sensitivity, routing, f16]
sources: [state-loop-probe, llama.cpp]
confidence: high
---

# Scaled layer-1 residual suffix sensitivity

## Question

Is the large one-token suffix response around `layer-1.f32` locally smooth, or does it coincide with a top-8 routing threshold?

The experiment perturbs the exact llama.cpp layer-1 input along the observed QX layer-0 error direction:

```text
direction = QX(step-0-layer-0-output) - llama.cpp(layer-1)
injected(scale) = llama.cpp(layer-1) + scale * direction
```

Every injected vector is rounded to F32, hashed, replayed through layers 1–47, and measured again after rounding. The report compares each final residual against both llama.cpp `l_out-47.f32` and the scale-zero suffix. Router order and top-8 membership are reported separately; a rank swap is not mislabeled as an expert-set transition.

## Fixed matrix and provenance

- Runtime source commit: `17c27d8fdd6ede23174323adbfe2a63edb0ff11f`.
- `qxqxf.exe` SHA-256: `16f74685b8865ceffdb4241146882514a89560fa5e13c4df5b9590f2e4496a80`.
- Oracle: llama.cpp `768d2a481a99cb75ec9a03b95dadbd35e7acf496`.
- Source GGUF SHA-256: `c8c2dc330dd1ec0c72c31b12e318647e6f9e0c773b9123eccfc3d12d9acc6652`.
- Token `42`, position `0`, one step, layers `1..47`.
- QX activation `q8_k_compat`, KV F16, temperature `0`, seed `7`, context `4`.
- Hidden width `2048`; scale grid `[-16,-8,-4,-2,-1,-0.5,-0.25,0,0.25,0.5,1,2,4,8,16]`.
- Oracle `layer-1.f32` SHA-256: `ab73249a11bac362e3f3375c394cd25c05775b76de02875140cc37dc857f3862`.
- QX baseline boundary SHA-256: `c485fa95708c3b184b0876cfa4f385d9d75db89df0eb1778f4b7239cdedac0b5`.
- Oracle `l_out-47.f32` SHA-256: `2aa780e4a7399ac335720c9a1427c000727f3096eca49f3c388ea46060bef59b`.
- Manifest SHA-256: `8174d092708196a22614fa87cebbe6d6f33c35f3d1d82ea5522c3e67eee0d5d5`.
- Report SHA-256: `a18b308fc105648a663fab7c8ea5eb0f3a04e5990937bd716963df65750248ae`.

The observed boundary direction has max-abs `2.38418579e-7`, RMSE `1.46771715e-8`, and L2 `6.64212962e-7`.

## Exact reproduction

First reproduce the modal-equivalent oracle and baseline described in [[hybrid-residual-replay-accumulation]]. Then run:

```bash
ROOT="$LOCALAPPDATA/Temp/qx-hybrid-f16"
OUT="$LOCALAPPDATA/Temp/qx-scaled-layer1"

python scripts/scaled_residual_replay.py run \
  --qxqxf build/qxqxf.exe \
  --model models/Qwen3-30B-A3B-UD-IQ2_M.qxf \
  --oracle-dir "$ROOT/oracle" \
  --baseline-dir "$ROOT/hybrids/start-0" \
  --experiment-dir "$OUT" \
  '--scales=-16,-8,-4,-2,-1,-0.5,-0.25,0,0.25,0.5,1,2,4,8,16' \
  --layers 48 --start-layer 1 --expected-count 2048 \
  --kv-format f16 --activation-format q8_k_compat \
  --prompt-token 42 --ctx 4 --seed 7
```

Re-analysis does not rerun the model and revalidates source hashes, every generated residual hash and formula, exact F32 size/finiteness, replay metadata, layer coverage, routing IDs/weights, and final sidecars:

```bash
python scripts/scaled_residual_replay.py analyze \
  --oracle-dir "$ROOT/oracle" \
  --baseline-dir "$ROOT/hybrids/start-0" \
  --experiment-dir "$OUT" \
  --output "$OUT/report.json"
```

## Result

`Final Δ L2` is measured against the scale-zero suffix, not against llama.cpp. `Projected scale` and `direction cosine` expose the actual F32 perturbation; half and quarter scales cannot be assumed to remain perfectly collinear after F32 rounding.

| Scale | Effective input L2 | Projected scale | Direction cosine | Final Δ L2 | Final vs oracle RMSE | Router-order layer | Top-8 membership layer |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -16 | 1.06274e-5 | -16 | 1 | 3.03984 | 6.71758e-2 | 46 | — |
| -8 | 5.31370e-6 | -8 | 1 | 2.56140 | 5.66012e-2 | 46 | — |
| -4 | 2.65685e-6 | -4 | 1 | 2.56114 | 5.65954e-2 | 46 | — |
| -2 | 1.32843e-6 | -2 | 1 | 2.56099 | 5.65921e-2 | 46 | — |
| -1 | 6.64213e-7 | -1 | 1 | 5.73359e-4 | 7.41343e-6 | — | — |
| -0.5 | 3.94860e-7 | -0.560085 | 0.942146 | 2.27830e-5 | 7.72939e-6 | — | — |
| -0.25 | 1.84344e-7 | -0.233849 | 0.842589 | 5.22268e-4 | 6.21681e-6 | — | — |
| 0 | 0 | 0 | 1 | 0 | 7.69131e-6 | — | — |
| 0.25 | 1.84344e-7 | 0.233849 | 0.842589 | 4.46775 | 9.87298e-2 | 46 | — |
| 0.5 | 3.94860e-7 | 0.560085 | 0.942146 | 1.46482e-5 | 7.68962e-6 | — | — |
| 1 | 6.64213e-7 | 1 | 1 | 4.46778 | 9.87305e-2 | 46 | — |
| 2 | 1.32843e-6 | 2 | 1 | 6.39856 | 1.41396e-1 | 28 | — |
| 4 | 2.65685e-6 | 4 | 1 | 6.27435 | 1.38651e-1 | 28 | — |
| 8 | 5.31370e-6 | 8 | 1 | 6.39680 | 1.41357e-1 | 28 | — |
| 16 | 1.06274e-5 | 16 | 1 | 6.39677 | 1.41356e-1 | 28 | — |

The response is not a smooth scalar amplification over this grid. Exact-direction scales `-1` and `+1` have equal input L2 but final deltas of `5.73359e-4` and `4.46778`, respectively. Positive `0.25`, `0.5`, and `1` also alternate between `4.46775`, `1.46482e-5`, and `4.46778` final L2.

No run changes top-8 membership. The only router changes are rank-order swaps inside the same selected set:

- layer 46 swaps experts `36` and `113` for scales `-16,-8,-4,-2,0.25,1`;
- layer 28 swaps experts `97` and `26` for scales `2,4,8,16`.

## Causal verdict

- **Smooth local scalar response:** refuted for this realized F32/Q8_K-compatible grid.
- **Top-8 membership threshold:** refuted in the tested range; membership is stable in all 15 runs.
- **Router rank-order threshold:** observed and correlated with high-response branches, but not proven causal. The selected expert set is unchanged, so rank crossing must not be promoted to an expert-membership claim.
- **Numerical runtime fix:** not authorized. The evidence does not isolate whether the discontinuity comes from routing-order-dependent F32 accumulation, activation quantization thresholds, or a downstream interaction.

[[scaled-residual-token-modality-matrix]] completes the additional-token and modality gate. It preserves top-8 membership across the three `q8_k_compat` + F16 cells, but shows token-dependent response magnitudes and membership changes in broader modal controls. Multi-token conclusions still require an accumulated KV snapshot/replay seam.

Related evidence: [[hybrid-residual-replay-accumulation]], [[layers0-40-same-input]], and [[current-status-and-roadmap]].
