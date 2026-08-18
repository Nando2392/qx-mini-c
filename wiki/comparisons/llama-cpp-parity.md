---
title: QX versus llama.cpp numerical parity
created: 2026-08-17
updated: 2026-08-18
type: comparison
tags: [validation, golden, quantization, kv-cache]
sources: [raw/project/project-state-2026-08-17.md]
confidence: high
---

# QX versus llama.cpp numerical parity

> **Baseline histórico superseded:** esta página conserva la evidencia pre-Q5_K que abrió los bisects posteriores. El estado actual está en [[layer1-layer2-sensitivity]]: la paridad numérica exacta sigue refutada, pero QX F32/INT8 ya coincide con llama F16 en `[1124,50853]` para `[42]` y `[358,1184]` para `Hello!`.

## Pregunta

¿Produce QX-mini-C los mismos residuals, logits y token greedy que llama.cpp al ejecutar el mismo GGUF Qwen3-30B-A3B desde token `42`?

## Oracle y contrato

- Oracle independiente: llama.cpp commit `768d2a481a99cb75ec9a03b95dadbd35e7acf496`.
- Modelo: mismo GGUF local, sin publicar pesos ni sidecars.
- GGUF SHA-256: `c8c2dc330dd1ec0c72c31b12e318647e6f9e0c773b9123eccfc3d12d9acc6652`.
- Dimensiones: hidden `2048`, layers `48`, vocab `151936`.
- Checkpoints F32 lossless: input, `Vcur`, contexto de attention, residual post-attention, salida MoE, salida de bloque y logits completos.
- Modos: QX INT8 versus llama Q8_0; QX diagnóstico F32 versus llama F16.
- Métricas: max-abs, RMSE, cosine y argmax.

El oracle es standalone y read-only respecto a QX. Véase [[numerical-correctness]] y [[architecture]].

## Bug material encontrado y corregido

QX seleccionaba los ocho expertos correctos pero usaba sus probabilidades softmax globales sin renormalizar. Qwen3MoE/llama.cpp divide los pesos top-8 por su suma. Antes del fix, layer 1 tenía max-abs aproximado `1.08`, RMSE `0.03916` y cosine `0.96442`.

Tests RED→GREEN exigen ahora:

- `router_norm_topk_prob=true`;
- ocho expertos únicos;
- suma de pesos top-8 igual a `1.0`;
- el mismo contrato en el probe de router y el loop real.

## Resultado por checkpoint de layer 0

| Checkpoint | QX/llama KV | max-abs | RMSE | cosine |
|---|---:|---:|---:|---:|
| input | F32/F16 | 0 | 0 | 1.0 |
| Vcur | F32/F16 | 0.000305031 | 0.000083164 | 0.999943484 |
| post-attention | F32/F16 | 0.00333905 | 0.000183834 | 0.999976473 |
| MoE output | F32/F16 | 0.00255167 | 0.000677590 | 0.999944290 |
| block output / layer 1 input | F32/F16 | 0.00589061 | 0.000703227 | 0.999961082 |
| block output / layer 1 input | INT8/Q8_0 | 0.0108175 | 0.000830295 | 0.999955836 |

`kqv_out` se captura en ambos lados, pero su orden físico no es semánticamente equivalente sin una transformación de layout; no se usa para declarar PASS/FAIL directo. `Vcur` sí es equivalente y demuestra que la primera diferencia aparece antes del append KV.

## Causa de la primera diferencia restante

QX calcula IQ4_XS decodificado × activación F32. La ruta CPU ggml empareja IQ4_XS con activación temporal Q8_K (`ggml_vec_dot_iq4_xs_q8_K`). Por tanto, bit-parity con llama.cpp no es esperable mientras QX conserve activaciones F32. Además, el cache normal QX cuantiza INT8 por vector, mientras llama Q8_0 usa bloques; el modo diagnóstico QX F32 separa ese efecto.

Esto es una diferencia de contrato numérico explícita, no evidencia de que el decoder IQ4_XS QX lea mal los pesos. Los goldens independientes de filas siguen validando el decoder.

## Baseline pre-Q5_K: propagación por capas después del fix de routing

| Layer input | F32/F16 max-abs | F32/F16 RMSE | F32/F16 cosine | INT8/Q8_0 max-abs | INT8/Q8_0 RMSE | INT8/Q8_0 cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 1.0 | 0 | 0 | 1.0 |
| 1 | 0.00589061 | 0.00070323 | 0.99996108 | 0.0108175 | 0.00083029 | 0.99995584 |
| 24 | 43.7548 | 0.971755 | 0.99999452 | 43.6355 | 0.969164 | 0.99999448 |
| 47 | 67.0054 | 1.85052 | 0.99829056 | 66.9384 | 1.85322 | 0.99826921 |

Los valores absolutos crecen con la escala del residual. Cosine se conserva alto, pero exactitud residual no existe.

## Baseline pre-Q5_K: residual final pre-head

`l_out-47` se captura directamente en llama.cpp y se compara contra `step-0-layer-47-output.f32` de QX:

| Modos | max-abs | RMSE | cosine |
|---|---:|---:|---:|
| F32/F16 | 1099.6502 | 32.9520 | 0.439279 |
| INT8/Q8_0 | 1100.9628 | 33.0204 | 0.440512 |

Por tanto, la similitud alta del **input** de layer 47 no puede extrapolarse a la salida final. El residual pre-head diverge materialmente y explica parte de la separación posterior de logits.

## Baseline pre-Q5_K: logits completos

| Modos | count | max-abs | RMSE | cosine | argmax QX | argmax llama |
|---|---:|---:|---:|---:|---:|---:|
| F32/F16 | 151936 | 9.09395 | 1.47826 | 0.874880 | 1124 | 1124 |
| INT8/Q8_0 | 151936 | 9.13947 | 1.48539 | 0.875121 | 1124 | 1124 |

Coincidencia greedy no implica paridad de logits.

## Baseline pre-Q5_K: secuencias greedy

| Prompt | llama F16 | llama Q8_0 | QX INT8 |
|---|---|---|---|
| token IDs `[42]` | `[1124, 50853]` | `[1124, 50853]` | `[1124, 11287]` |
| `Hello!` / `[9707, 0]` | `[358, 1184]` | `[358, 614]` | `[81379, 44707]` |

La comparación usa un oracle secuencial llama.cpp independiente con contexto/KV persistente. En este baseline, para `[42]` sólo coincidía el primer token y `Hello!` divergía desde el primero generado. La diferencia F16/Q8_0 del segundo token de `Hello!` demuestra además que el formato KV afecta la secuencia una vez existe contexto. El fix Q5_K posterior supersede únicamente los outputs QX de esta tabla.

## Reproducción

```bat
tests\build_llama_reference_oracle.bat
build\llama_reference_oracle.exe models\Qwen3-30B-A3B-UD-IQ2_M.gguf %TEMP%\llama-f16 42 0,1,24,47 f16 internals
build\llama_sequence_oracle.exe models\Qwen3-30B-A3B-UD-IQ2_M.gguf 42 2 f16
build\llama_sequence_oracle.exe models\Qwen3-30B-A3B-UD-IQ2_M.gguf 9707,0 2 q8_0
build\qxqxf.exe state-loop-probe --in models\Qwen3-30B-A3B-UD-IQ2_M.qxf --prompt-token 42 --steps 1 --layers 48 --ctx 4 --kv f32 --temperature 0 --seed 7 --full-moe --final-head --dump-residuals %TEMP%\qx-f32
python scripts\compare_residuals.py --qx-dir %TEMP%\qx-f32 --llama-dir %TEMP%\llama-f16 --layers 0,1,24,47 --phase input
python scripts\compare_residuals.py --qx-dir %TEMP%\qx-f32 --llama-dir %TEMP%\llama-f16 --layers 47 --phase output
python scripts\compare_logits.py --qx %TEMP%\qx-f32\step-0-logits.f32 --llama %TEMP%\llama-f16\logits.f32
```

## Veredicto

**Paridad numérica exacta: refutada, tanto en este baseline como post-Q5_K.**

**Estado post-Q5_K:** la matriz de dos prompts/dos tokens pasa contra llama F16; cobertura exhaustiva sigue pendiente.

Este milestone corrigió el bug de routing y dejó un baseline reproducible. Los milestones [[f32-vs-q8k-activation]], [[moe-stage-bisect]], [[iq2-s-iq4-xs-q8k]] y [[layer1-layer2-sensitivity]] contienen la evolución posterior.
