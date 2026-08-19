---
title: Hybrid residual replay accumulation bisect
created: 2026-08-19
updated: 2026-08-19
type: comparison
tags: [qwen3-moe, residual, replay, accumulation, amplification, kv, f16]
sources: [llama.cpp, state-loop-probe]
confidence: high
---

# Hybrid residual replay accumulation bisect

## Question

Does the remaining fixed-token divergence come from another material local block error, from accumulated residual error, or from downstream amplification of a locally tiny error?

This experiment injects the exact llama.cpp `layer-N.f32` residual into QX at layer `N`, executes the suffix `N..47`, and compares the reconstructed `l_out-47` against llama.cpp. It is deliberately modal-equivalent: llama.cpp and QX both use F16 KV, QX uses `q8_k_compat`, token `42`, position `0`, and one forward step.

## Provenance and limitation

- **Injected residual:** lossless F32 `layer-N.f32` exported by the standalone llama.cpp oracle from `Qwen3-30B-A3B-UD-IQ2_M.gguf`.
- **Oracle version:** llama.cpp commit `768d2a481a99cb75ec9a03b95dadbd35e7acf496`; source GGUF SHA-256 `c8c2dc330dd1ec0c72c31b12e318647e6f9e0c773b9123eccfc3d12d9acc6652`.
- **KV used by the suffix:** recomputed by QX from the injected residual for the current token using F16 storage. Oracle KV bytes are **not** injected.
- **Important:** residual-only injection does not replace an accumulated KV cache. This one-token experiment has no prior-token KV history, but its conclusion must not be extrapolated to multi-token replay without an explicit KV snapshot/replay seam.
- **Runtime metadata:** each run records top-level `start_layer`, `residual_source="injected_f32_replay"`, nested `residual_replay={enabled:true,source:"f32_sidecar",values:2048}`, and `kv_format="f16"`.

## Exact reproduction

Build QX and the standalone oracle first:

```bash
cmd.exe /c build_msvc.bat
cmd.exe /c tests\\build_llama_reference_oracle.bat
```

Capture one modal-equivalent llama.cpp forward:

```bash
ROOT="$LOCALAPPDATA/Temp/qx-hybrid-f16"
rm -rf "$ROOT"
mkdir -p "$ROOT/oracle" "$ROOT/hybrids"
LAYERS=$(seq -s, 0 47)
build/llama_reference_oracle.exe \
  models/Qwen3-30B-A3B-UD-IQ2_M.gguf \
  "$ROOT/oracle" 42 "$LAYERS" f16 internals=47 \
  > "$ROOT/oracle-result.json"
```

Replay every residual boundary through the QX suffix:

```bash
for k in $(seq 0 47); do
  OUT="$ROOT/hybrids/start-$k"
  mkdir -p "$OUT"
  build/qxqxf.exe state-loop-probe \
    --in models/Qwen3-30B-A3B-UD-IQ2_M.qxf \
    --prompt-token 42 --steps 1 --layers 48 \
    --start-layer "$k" --residual-in "$ROOT/oracle/layer-$k.f32" \
    --ctx 4 --kv f16 --activation q8_k_compat \
    --temperature 0 --seed 7 --full-moe \
    --dump-residuals "$OUT" > "$OUT/result.json"
done
```

Generate the fail-closed report. The analyzer rejects absent/mismatched replay metadata, wrong F32 byte counts, and NaN/Inf:

```bash
python scripts/compare_hybrid_residual_replay.py \
  --oracle-dir "$ROOT/oracle" \
  --hybrid-dir "$ROOT/hybrids" \
  --layers 48 --expected-count 2048 --kv-format f16 \
  --output "$ROOT/hybrid-report.json"
```

## Modal-equivalent F16 result

`incoming RMSE` is the accumulated QX baseline error at the boundary before the selected layer. `final replay RMSE` is the suffix output error after replacing that incoming residual with the exact oracle residual.

| Start layer | Incoming accumulated RMSE | Incoming max-abs | Final replay RMSE | Final max-abs | Final cosine |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 9.87305e-2 | 2.49377 | 0.999998498 |
| 1 | 1.46772e-8 | 2.38419e-7 | 7.69131e-6 | 2.44141e-4 | ≈1 |
| 2 | 1.18152e-6 | 5.34058e-5 | 7.38140e-6 | 3.05176e-4 | ≈1 |
| 3 | 5.39878e-6 | 2.44141e-4 | 3.64231e-5 | 1.09863e-3 | ≈1 |
| 4 | 2.70389e-6 | 1.22070e-4 | 7.10186e-6 | 2.44141e-4 | ≈1 |
| 16 | 2.70303e-6 | 1.22070e-4 | 7.13944e-6 | 2.44141e-4 | ≈1 |
| 32 | 2.97103e-4 | 2.72751e-3 | 7.10231e-6 | 2.44141e-4 | ≈1 |
| 40 | 9.60344e-4 | 3.95775e-3 | 7.09798e-6 | 2.44141e-4 | ≈1 |
| 47 | 9.35561e-3 | 5.27344e-2 | 1.03862e-5 | 3.66211e-4 | ≈1 |

Replacing only the layer-1 input residual reduces final RMSE from `9.87305e-2` to `7.69131e-6`; the direct quotient of those two final-RMSE values is `12836.6×`. This is distinct from the removed, unstable suffix-gain-over-incoming-error metric. The incoming layer-1 discrepancy itself is only `1.46772e-8` RMSE. Therefore the observed fixed-token global error is not evidence of another material local seam: it is a microscopic layer-0 discrepancy that the downstream trajectory strongly amplifies.

Layer 3 is the largest suffix-only F16 outlier (`3.64231e-5` RMSE), still roughly 2700× below the unreplayed baseline and below a material global failure for this input.

## F32 control and modality effect

The same analysis under F32 KV does not close the suffix and must not be substituted for the F16 gate:

| KV mode | Start 0 final RMSE | Start 1 incoming RMSE | Start 1 final RMSE | Start 47 final RMSE |
|---|---:|---:|---:|---:|
| F16 | 9.87305e-2 | 1.46772e-8 | 7.69131e-6 | 1.03862e-5 |
| F32 | 2.09137e-1 | 1.41082e-4 | 9.05740e-2 | 6.14076e-3 |

This is a material **modality effect**. It confirms the existing Q5_K attention contract: the llama.cpp callback and QX comparison must both use F16 KV. F32 remains useful as a diagnostic control, not as proof of local closure.

## Causal verdict

- **Classification A — accumulated/global error:** confirmed.
- **Classification B — downstream amplification:** confirmed, beginning from a tiny layer-0 output discrepancy for the fixed input.
- **Classification C — modal mismatch:** confirmed for the F32 control and eliminated from the primary F16 replay.
- **Classification D — new material local seam:** refuted for this fixed-token F16 matrix; the same-input sweep and this suffix replay agree.

No runtime numerical fix is authorized by this evidence. The follow-up [[scaled-layer1-residual-sensitivity]] refutes a smooth scalar response over the tested grid, finds rank-order crossings but no top-8 membership change, and still does not authorize a numerical fix. Multi-token conclusions require a separate accumulated KV snapshot/replay seam.

Related evidence: [[layers0-40-same-input]], [[layer47-same-input]], and [[current-status-and-roadmap]].
