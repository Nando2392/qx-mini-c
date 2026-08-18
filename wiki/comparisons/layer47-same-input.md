---
title: Layer 47 Same-Input Attention and MoE
created: 2026-08-18
updated: 2026-08-18
type: comparison
tags: [qwen3-moe, layer-47, attention, moe, q8-k, oracle]
sources: [README.md, tests/test_moe_stage_oracle.py, scripts/compare_layer_sensitivity.py]
confidence: high
---

# Layer 47 same-input attention and MoE

## Pregunta

Tras cerrar [[final-head-q6k-q8k]], ¿la divergencia que entra al head nace dentro del último bloque o ya viene acumulada desde capas anteriores?

El punto de partida acumulado y la amplificación layer 46→47 están documentados en [[layer2-logits-sweep]]; las reglas para separar claims same-input de end-to-end viven en [[numerical-correctness]].

El gate inyecta exactamente el mismo `layer-47.f32` del oracle fijado en QX y llama.cpp, y separa:

```text
layer-47
  → attn_norm → V projection → F16 KV/context → Q6_K output projection → ffn_inp
  → ffn_norm → router/top-8 → IQ2_S gate/up → SwiGLU → IQ4_XS down
  → weighted experts → ffn_moe_out → l_out-47
```

Oracle read-only: llama.cpp `768d2a481a99cb75ec9a03b95dadbd35e7acf496`.

## Contrato reproducible

```bash
build/qxqxf.exe attention-stage-probe \
  --in models/Qwen3-30B-A3B-UD-IQ2_M.qxf \
  --layer 47 --layer-in <oracle>/layer-47.f32 \
  --out-dir <attention> --activation q8_k_compat --kv f16

build/qxqxf.exe moe-stage-probe \
  --in models/Qwen3-30B-A3B-UD-IQ2_M.qxf \
  --layer 47 --ffn-inp <attention>/ffn_inp-47.f32 \
  --out-dir <moe> --activation q8_k_compat

python scripts/compare_layer_sensitivity.py \
  --oracle-dir <oracle> --attention-dir <attention> \
  --same-input-moe-dir <moe> \
  --expected-vcur-count 512 --expected-kqv-out-count 4096 \
  --layer 47
```

El modo same-input exige counts explícitos para `Vcur` y `kqv_out`, y falla cerrado ante sidecars ausentes, truncados o sobredimensionados incluso si oracle y QX comparten el mismo tamaño incorrecto. También rechaza NaN/Inf, shapes contradictorios, routing incoherente, un `ffn_inp` que no sea exactamente la suma F32 `layer_input + attn_out`, argumentos parciales o mezcla ambigua con el modo perturbación.

## Atención same-input

| Checkpoint | Count | Max abs | RMSE | Cosine |
|---|---:|---:|---:|---:|
| `attn_norm-47` | 2048 | `4.76837e-7` | `1.82718e-8` | ≈`1` |
| `Vcur-47` | 512 | `1.46031e-6` | `3.55797e-7` | `0.999999999999948` |
| `kqv_out-47` | 4096 | `1.22070e-4` | `5.39550e-6` | `0.999999999987469` |
| `attn_out-47` derivado | 2048 | `6.10352e-5` | `1.41255e-6` | ≈`1` |
| `ffn_inp-47` | 2048 | `6.10352e-5` | `1.42349e-6` | `0.999999999999992` |

Metadata ejecutada:

- V: `q5_k_q8_k`.
- output projection: `q6_k_q8_k`.
- KV/context: F16, igual que el callback `kqv_out` del oracle.

## MoE con `ffn_inp-47` exacto del oracle

| Checkpoint | Max abs | RMSE | Cosine |
|---|---:|---:|---:|
| `ffn_norm` | `4.76837e-7` | `2.73699e-8` | `1` |
| router logits | `9.53674e-7` | `2.75814e-7` | ≈`1` |
| normalized weights | `2.08616e-7` | `8.15851e-8` | ≈`1` |
| gate | `1.90735e-6` | `2.39211e-7` | ≈`1` |
| up | `3.81470e-6` | `2.85854e-7` | ≈`1` |
| SwiGLU | `9.15527e-5` | `2.53703e-6` | ≈`1` |
| down | `2.44141e-4` | `3.69579e-6` | ≈`1` |
| weighted | `5.49316e-4` | `5.83641e-6` | ≈`1` |
| reconstructed `ffn_moe_out` | `5.13077e-4` | `1.59911e-5` | ≈`1` |
| reconstructed `l_out-47` | `5.74112e-4` | `1.69717e-5` | ≈`1` |

Top-8 exacto en ambos runtimes:

```text
[83, 3, 74, 119, 92, 28, 109, 101]
```

## Cadena QX attention → QX MoE

Cuando el MoE consume el `ffn_inp-47` producido por la atención QX same-input:

| Checkpoint reconstruido | Max abs | RMSE | Cosine |
|---|---:|---:|---:|
| `ffn_moe_out-47` | `2.34127e-4` | `7.98253e-6` | `0.999999999999997` |
| `l_out-47` | `2.57492e-4` | `8.01077e-6` | `0.999999999999996` |

La cadena completa queda dentro del gate local `max_abs ≤ 3e-4`, `RMSE ≤ 1e-5`.

## Atribución del delta restante

El mayor delta same-input no demuestra un decoder quant roto. La proyección F32 del router usa reducción escalar-double en QX, mientras ggml usa reducción SIMD F32 dependiente del backend. Los logits difieren como máximo `9.54e-7`; el peso normalizado del experto 83 difiere `2.09e-7`. Los outputs down alcanzan magnitud `2188`, por lo que esa diferencia minúscula se amplifica al multiplicar el experto.

Experimento de sustitución:

- QX down con pesos oracle: aggregate max-abs `2.44141e-4`.
- Oracle down con pesos QX: aggregate max-abs `4.88281e-4`.
- QX completo: aggregate max-abs `6.10352e-4` con suma F32, `5.13077e-4` con suma de alta precisión.

El routing pesa más que el error de down, pero el top-8 no cambia. Replicar dentro del runtime un orden SIMD específico de una CPU haría el resultado menos portable y no corregiría la divergencia acumulada anterior.

## Veredicto

- Layer 47 attention + MoE: **GREEN same-input** para este modelo/input/oracle.
- Q5_K×Q8_K, Q6_K×Q8_K, IQ2_S×Q8_K e IQ4_XS×Q8_K: no se reabren.
- F32 sigue default; `q8_k_compat` sigue CPU-only y opt-in.
- La divergencia global en `l_out-47` entra acumulada desde capas anteriores; exactitud global de residuales/logits sigue refutada.
- Siguiente trabajo: ampliar la matriz de tokenizer/prompts/greedy y optimizar sin convertir orden SIMD backend-specific en contrato público.
