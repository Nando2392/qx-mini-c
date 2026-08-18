---
title: Layer 41 IQ3_S × Q8_K Same-Input Fix
created: 2026-08-18
updated: 2026-08-18
type: comparison
tags: [qwen3-moe, layer-41, iq3-s, q8-k, oracle]
sources: [src/qx_format.c, tests/ggml_reference_decode.c, tests/test_moe_stage_oracle.py, scripts/compare_layer_sensitivity.py]
confidence: high
---

# Layer 41 IQ3_S × Q8_K same-input fix

## Pregunta causal

Después de que [[layer47-same-input]] demostrara que el último bloque recibe la divergencia ya acumulada, ¿cuál es la capa más tardía anterior a 47 que deja de cerrar con exactamente el mismo residual del oracle?

La separación entre cierre local, forward acumulado y paridad global sigue el contrato de [[numerical-correctness]].

El bisect descendente regeneró sidecars reales para layers 44, 43, 42 y 41. Layers 44–42 cerraron materialmente. Layer 41 fue la primera capa que incumplió el gate.

## Frontera antes del fix

Con `layer-41.f32` exacto del oracle fijado:

```text
attention: GREEN
routing: [48,73,69,18,96,104,88,26] exacto
IQ2_S gate/up + SwiGLU: GREEN
down tensor: blk.41.ffn_down_exps.weight, ggml_type=21 (IQ3_S)
QX down kernel: dequant_f32
first divergence checkpoint: ffn_moe_down-41
```

El modo ggml CPU usa `IQ3_S × Q8_K`; QX estaba comparando pesos IQ3_S decodificados contra la activación F32 original. El mismatch no era un routing distinto ni una inferencia a partir del sweep acumulado.

Los valores pre-fix siguientes son el baseline histórico observado por el bisect antes del cambio. No se presentan como un gate que el runtime post-fix pueda recrear sin volver a la implementación anterior.

| Checkpoint pre-fix observado | Max abs | RMSE | Cosine |
|---|---:|---:|---:|
| `ffn_moe_weighted-41` | `1.25454e-3` | — | — |
| reconstructed `l_out-41` | `1.62544e-3` | `4.02635e-4` | `0.999999999905009` |

## Fix mínimo

El runtime C implementa un vec-dot escalar `IQ3_S × Q8_K` siguiendo el layout y las escalas del ggml fijado. `expert-q8-k-dot-probe` lo valida contra el traits CPU público de llama.cpp sobre filas reales de `blk.41.ffn_down_exps.weight`. El dispatch se activa sólo en `--activation q8_k_compat`; F32 continúa como default.

Metadata ejecutada:

```text
gate_up_projection_kernel=iq2_s_q8_k
down_projection_kernel=iq3_s_q8_k
```

Layer 24 también usa IQ3_S down. Una reproducción independiente confirmó tensor type 21, routing exacto y cierre de down (`max_abs=2.38419e-7`) antes de reemplazar su expectativa histórica de fallback.

## Resultado layer 41 post-fix

Comando reproducible:

```bash
python scripts/compare_layer_sensitivity.py \
  --oracle-dir <oracle> --attention-dir <attention> \
  --same-input-moe-dir <moe> \
  --expected-vcur-count 512 --expected-kqv-out-count 4096 \
  --layer 41
```

| Layer/checkpoint | Max abs | RMSE | Cosine |
|---|---:|---:|---:|
| `attn_norm-41` | `9.53674e-7` | `2.10813e-8` | ≈`1` |
| `Vcur-41` | `1.19209e-7` | `7.42027e-9` | `0.999999999999988` |
| `kqv_out-41` | `0` | `0` | `1` |
| `attn_out-41` | `1.16527e-5` | `2.59581e-7` | `0.999999999990445` |
| `ffn_inp-41` | `1.19209e-7` | `1.84042e-8` | `1` |
| router logits | `2.86102e-6` | `9.29148e-7` | `0.999999999999991` |
| normalized weights | `5.96046e-8` | `2.15724e-8` | ≈`1` |
| gate | `1.90735e-6` | `2.64453e-7` | `0.999999999999988` |
| up | `1.90735e-6` | `1.73687e-7` | `0.999999999999994` |
| SwiGLU | `1.90735e-6` | `1.46793e-7` | `0.999999999999990` |
| `ffn_moe_down-41` | `9.53674e-7` | `1.00230e-7` | `0.999999999999986` |
| weighted | `8.94070e-8` | `3.35602e-9` | `0.999999999999908` |
| reconstructed `ffn_moe_out-41` | `1.27940e-7` | `9.81110e-9` | `0.999999999999922` |
| reconstructed `l_out-41` | `4.76234e-5` | `1.05287e-6` | ≈`1` |

Top-8 exacto:

```text
[48, 73, 69, 18, 96, 104, 88, 26]
```

La microdiferencia de router excede el umbral diagnóstico histórico de `2e-6`, pero no cambia top-k y queda reducida a `5.96e-8` en pesos normalizados. No se ocultó: se reporta separadamente del primer fallo material, que era down.

## Provenance versionada y regenerable

`tests/test_moe_stage_oracle.py::test_backward_bisect_same_input_layers_close_with_reproducible_metrics` regenera el oracle fijado y los probes QX para layers 24, 41, 42, 43 y 44. Los sidecars viven sólo en `tmp_path`; el test ejecuta el comparador real y fija routing, kernels y métricas exactas de router/weighted/`l_out`. Así, el payload versiona los valores esperados como assertions ejecutables sin versionar `.f32` ni JSON experimental.

```bash
C:/Users/fjmn2/Dev/hermes-agent/.venv/Scripts/python.exe -m pytest \
  tests/test_moe_stage_oracle.py::test_backward_bisect_same_input_layers_close_with_reproducible_metrics -q
# 5 passed
```

| Layer | Routing exacto | Weighted max abs | `l_out` max abs | `l_out` RMSE |
|---:|---|---:|---:|---:|
| 24 | `[10,105,24,111,101,98,108,113]` | `1.78814e-7` | `4.57502e-5` | `1.01141e-6` |
| 41 | `[48,73,69,18,96,104,88,26]` | `8.94070e-8` | `4.76234e-5` | `1.05287e-6` |
| 42 | `[110,64,17,69,21,41,116,25]` | `1.19209e-7` | `8.47416e-7` | `4.00499e-8` |
| 43 | `[78,82,83,63,90,100,74,28]` | `3.57628e-7` | `4.68057e-5` | `1.03571e-6` |
| 44 | `[40,113,104,41,72,83,73,102]` | `2.98023e-8` | `7.81775e-6` | `1.76607e-7` |

## Veredicto y límite

- Layer 41: **GREEN same-input post-fix**.
- Primera divergencia interna observada en el bisect pre-fix: `ffn_moe_down-41`, causada por falta de `IQ3_S × Q8_K` en el modo opt-in.
- Esto no demuestra paridad acumulada, exactitud global de logits ni cobertura exhaustiva de prompts.
- F32 sigue siendo default; `q8_k_compat` sigue CPU-only y opt-in.
- El bisect supersedente [[layers0-40-same-input]] cierra materialmente las 41 capas restantes; el siguiente gate es acumulación con inyección híbrida, no otro kernel por analogía.
