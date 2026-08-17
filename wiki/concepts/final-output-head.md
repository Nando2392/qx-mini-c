---
title: Final RMSNorm and Output Head
created: 2026-08-17
updated: 2026-08-17
type: concept
tags: [runtime, qwen3-moe, validation, quantization]
sources: [raw/project/project-state-2026-08-17.md]
confidence: high
---

# Final RMSNorm and output head

## Ruta implementada

```text
residual real después de layer 47, 2048 F32
→ RMSNorm con output_norm.weight, F32, epsilon 1e-6
→ output.weight Q6_K, shape 2048 × 151936
→ 151936 logits
→ argmax y top-N
```

`state-loop-probe --full-moe --final-head` falla cerrado salvo que se ejecuten entre 1 y 64 pasos, el manifest sea Qwen3 MoE `[48 layers, hidden 2048, vocab 151936]`, se usen todas las layers, el contexto esté entre 1 y 4096, la temperatura sea cero y no se active el benchmark histórico que reporta tok/s. Para varios pasos, cada argmax inicia el siguiente forward desde su propio embedding; ver [[autoregressive-loop]].

## Evidencia medida

- `logits_computed=151936` y `full_vocabulary=true`.
- Checksums QXF de `output_norm.weight` y `output.weight` verificados antes del cálculo.
- Argmax para el token de entrada 42 y seed 7: token `1124`.
- Checksum del vector completo de logits F32 después del fix de normalización top-8: `12662891110960910958`.
- Probe instrumentado completo medido: ~8.35 s en una corrida caliente.

## Golden independiente

El test externo:

1. recalcula RMSNorm en Python leyendo `output_norm.weight` directamente del QXF;
2. usa `dequantize_row_q6_K` enlazado desde `llama.cpp` para las 151936 filas;
3. compara el checksum FNV de las 151936 logits F32, top-5 y argmax.

Este gate descubrió y corrigió un decoder Q6_K propio con disposición de nibbles/bits altos incorrecta. Después del fix, el argmax QX y el helper oficial coinciden.

## Límite de honestidad

Validado: head completo para cada residual producido por QX. No validado todavía: paridad end-to-end del residual de todas las capas contra otra implementación ni tokenizer completo. Por eso el tiempo anterior no se presenta como tok/s de conversación.

Forward previo: [[moe-forward]]. Gates: [[numerical-correctness]]. Roadmap: [[current-status-and-roadmap]].
