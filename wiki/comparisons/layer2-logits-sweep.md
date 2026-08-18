---
title: Layer 2 to Logits Sweep and Layer 46 Q6_K
created: 2026-08-18
updated: 2026-08-18
type: comparison
tags: [qwen3-moe, attention, q6-k, q8-k, sensitivity, oracle]
sources: [comparisons/layer1-layer2-sensitivity.md, comparisons/moe-stage-bisect.md]
confidence: high
---

# Barrido layer 2 → logits y Q6_K de layer 46

> Estado: GREEN como bisect causal acotado. La paridad numérica global exacta sigue refutada.
> Fecha: 2026-08-18 | Issue: #12

## Pregunta

Después del fix Q5_K de [[layer1-layer2-sensitivity]], ¿dónde aparece la siguiente amplificación material entre layer 2 y logits, y corresponde a otro bug de kernel o a sensibilidad ante el residual ya perturbado?

## Contrato fijo

- runtime QX propio; llama.cpp sólo es oracle read-only;
- oracle: commit `768d2a481a99cb75ec9a03b95dadbd35e7acf496`;
- modelo local: `Qwen3-30B-A3B-UD-IQ2_M`;
- prompt token-ID `[42]`, un paso, 48 capas;
- QX: activación `q8_k_compat`, KV F32 para el sweep global;
- llama.cpp: KV F16;
- F32 sigue siendo el modo predeterminado; `q8_k_compat` sigue siendo CPU-only y opt-in;
- sidecars, modelos y reportes experimentales permanecen fuera de Git.

El comparador `scripts/compare_residuals.py` ahora reporta `delta_l2`, `gain_from_previous`, `first_material_amplification_layer` y acepta `--amplification-start-layer`. Para continuar desde layer 2 se usa umbral estricto `gain > 2.0` y `start=2`.

## Sweep inicial

El barrido de las 48 entradas de capa encontró la siguiente amplificación material en la transición layer 46 → 47:

| Checkpoint | Max abs | RMSE | Cosine | Delta L2 | Ganancia adyacente |
|---|---:|---:|---:|---:|---:|
| layer 46 input | 0.210571 | 0.00791186 | 0.999999975456 | 0.358050 | — |
| layer 47 input | 0.163246 | 0.0277259 | 0.999999195734 | 1.25473 | **3.50435×** |
| logits | 0.171505 | 0.0346769 | 0.999936486323 | 13.5167 | — |

La inspección del modelo real mostró:

```text
blk.46.attn_v.weight      ggml_type=13  Q5_K
blk.46.attn_output.weight ggml_type=14  Q6_K
```

`q8_k_compat` ejecutaba `Q5_K × Q8_K` para V, pero degradaba la proyección de salida Q6_K a `dequant_f32`. llama.cpp usa `Q6_K × Q8_K`; por tanto había una discrepancia real de contrato, aunque el fallback F32 fuera numéricamente más preciso de forma aislada.

## Fix y golden independiente

Se añadió un kernel escalar `Q6_K × Q8_K` que replica `ggml_vec_dot_q6_K_q8_K_generic`:

- bloque Q6_K de 210 bytes;
- 256 valores por bloque;
- high bits, low bits y 16 escalas con signo reconciliados con `dequantize_row_q6_K`;
- acumulación entera por ocho lanes y escala `d_q6 × d_q8`;
- filas incompletas o dimensiones incompatibles fallan cerrado.

Los tests comparan:

1. cuatro bloques Q6_K reales completos —256 floats cada uno— contra `dequantize_row_q6_K` público;
2. tres filas completas de `blk.46.attn_output.weight` contra el `vec_dot` público de ggml;
3. el vector completo `ffn_inp-46` contra el oracle same-input.

Metadata post-fix:

```text
projection_kernel:        q5_k_q6_k_q8_k
v_projection_kernel:      q5_k_q8_k
output_projection_kernel: q6_k_q8_k
state-loop families:      iq4_xs_q5_k_q6_k_q8_k
```

## Gate same-input

Usando exactamente el mismo `layer-46.f32` del oracle y KV F16:

| Checkpoint | Pre-fix max abs | Post-fix max abs | Post-fix RMSE | Post-fix cosine |
|---|---:|---:|---:|---:|
| `attn_norm-46` | — | 2.98e-8 | 2.17e-9 | ≈1 |
| `Vcur-46` | — | 6.41e-7 | 1.05e-7 | ≈1 |
| `kqv_out-46` | — | 1.53e-5 | 7.69e-7 | ≈1 |
| `ffn_inp-46` | 0.00445557 | **1.19e-7** | **2.54e-8** | ≈1 |

El max-abs de `ffn_inp-46` mejora **37,376×**. El kernel Q6_K queda cerrado para este tensor y este input.

## Resultado global post-fix

El resultado end-to-end no mejora; se reporta explícitamente:

| Checkpoint | Pre-fix RMSE | Post-fix RMSE | Post-fix max abs | Post-fix cosine |
|---|---:|---:|---:|---:|
| layer 46 input | 0.00791186 | 0.00778122 | 0.215820 | 0.999999977342 |
| layer 47 input | 0.0277259 | **0.0309398** | 0.765381 | 0.999999193349 |
| logits | 0.0346769 | **0.0393805** | 0.187882 | 0.999932288590 |

La ganancia layer 46 → 47 pasa de `3.50435×` a `3.97621×`. El argmax permanece `1124` y el smoke fijo de dos tokens permanece `[1124, 50853]`.

Esto no invalida el kernel: demuestra que al reproducir el temporal Q8_K de llama.cpp sobre un residual QX ya perturbado cambia la trayectoria. No se afirma mejora global ni paridad exacta.

## Bisect causal nominal vs perturbado

Se reejecutó layer 46 con:

- nominal: `layer-46.f32` del oracle;
- perturbado: input layer 46 del sweep QX;
- atención Q6_K×Q8_K y KV F16 en ambos;
- MoE Q8_K reejecutado sobre cada `ffn_inp`.

| Etapa | Delta L2 | Ganancia |
|---|---:|---:|
| layer input | 0.352138 | base |
| attention output derivado | 0.0539587 | **0.153232×** |
| FFN input | 0.359120 | 1.01983× |
| MoE output | 1.31557 | **3.66331×** |
| layer output | 1.40608 | **3.99298×** |

Primer checkpoint de amplificación: `moe_output`.

El top-8 permanece idéntico:

```text
[74, 96, 5, 44, 32, 3, 36, 113]
```

El experto 74 aporta `1.00086` delta L2, `76.0779%` del delta MoE. La reconstrucción nominal vs oracle queda en max-abs `4.08e-5`; perturbada vs QX queda en max-abs `2.87e-5`.

## Rendimiento

Benchmark post-fix, cinco runs warm, 48 capas, MoE completo y KV INT8:

| Modo | Mediana | Throughput equivalente del probe | Peak RSS mediano |
|---|---:|---:|---:|
| F32 default | 8.52569 s/token | 0.117293 token/s | 5,636,096 B |
| Q8_K opt-in | 2.34729 s/token | 0.426023 token/s | 5,640,192 B |

Speedup observado: `3.63214×`. Esta medición excluye RMSNorm final, lm_head completo y tokenización; es latencia del probe de 48 capas/MoE, no throughput end-to-end de generación.

## Veredicto

- **GREEN:** decoder Q6_K completo y `Q6_K × Q8_K` contra oracle público.
- **GREEN:** atención layer 46 same-input cierra hasta error F32.
- **GREEN:** el sweep global localiza la siguiente amplificación en layer 46 → 47.
- **CLASIFICADO:** la amplificación causal comienza en MoE, no en atención, sin cambio de top-8; es sensibilidad cuantizada acotada para esta matriz fija.
- **NO demostrado:** paridad global exacta, cobertura de prompts exhaustiva o que Q8_K deba ser el modo default.

## Siguiente gate

Bisect de layer 47 → `l_out-47` → RMSNorm final → logits usando input idéntico por etapa. No reabrir Q5_K/Q6_K ni MoE layer 46 salvo evidencia nueva que contradiga sus goldens.

Relacionado: [[layer1-layer2-sensitivity]], [[moe-stage-bisect]], [[llama-cpp-parity]], [[current-status-and-roadmap]].
