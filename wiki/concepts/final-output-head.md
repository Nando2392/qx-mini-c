---
title: Final RMSNorm and Output Head
created: 2026-08-17
updated: 2026-08-18
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
  - default: dequant F32 × activación F32
  - q8_k_compat: Q6_K × Q8_K, CPU opt-in
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

## Gate same-input

`final-head-probe` acepta un residual F32 exacto y exporta RMSNorm final y logits completos. Con `l_out-47` del oracle fijado, RMSNorm queda en max-abs `3.81470e-6`, RMSE `1.17205e-7`; `Q6_K × Q8_K` queda en max-abs `2.38419e-6`, RMSE `4.91155e-7`, cosine `0.999999999999927`, argmax `1124`.

La activación normalizada se cuantiza una vez a Q8_K y el workspace se reutiliza para las 151936 filas. Metadata: `lm_head_kernel=q6_k_q8_k`, `activation_quantizations=1`. F32 conserva `dequant_f32` y cero cuantizaciones.

## Golden independiente

El test externo:

1. recalcula RMSNorm en Python leyendo `output_norm.weight` directamente del QXF;
2. usa `dequantize_row_q6_K` para el default y el vec-dot público `Q6_K × Q8_K` para el modo opt-in;
3. compara el checksum FNV de las 151936 logits F32, top-5 y argmax.

Este gate descubrió y corrigió un decoder Q6_K propio con disposición de nibbles/bits altos incorrecta. Después del fix, el argmax QX y el helper oficial coinciden.

## Límite de honestidad

Validado: head completo para residual QX y cierre same-input contra llama. No validado: paridad exacta de `l_out-47`, logits globales ni tokenizer exhaustivo. Evidencia: [[final-head-q6k-q8k]].

Forward previo: [[moe-forward]]. Gates: [[numerical-correctness]]. Roadmap: [[current-status-and-roadmap]].
