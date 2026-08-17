---
title: Greedy Autoregressive Loop
created: 2026-08-17
updated: 2026-08-17
type: concept
tags: [runtime, qwen3-moe, validation, kv-cache]
sources: [raw/project/project-state-2026-08-17.md]
confidence: high
---

# Greedy autoregressive loop

## Contrato implementado

Para cada posición:

```text
input token id
→ fila propia de token_embd.weight
→ layers 0–47
→ RoPE con posición actual
→ append K/V INT8 en cada layer
→ atención causal sobre posiciones 0..actual
→ top-8 MoE y residual
→ [[final-output-head]] completo
→ argmax global
→ token seleccionado como input de la posición siguiente
```

El residual final de una posición nunca se reutiliza como embedding del token siguiente.

## Evidencia medida

Con prompt token `42`, temperatura cero y seed `7`:

```text
input tokens: [42, 1124]
selected tokens: [1124, 11287]
positions: [0, 1]
layers_run: 96
kv_appends: 96
logits checksums: [12662891110960910958, 2895445711150549338]
```

Las 48 capas de la posición 0 reportan `attention_context_tokens=1`; las 48 de la posición 1 reportan `attention_context_tokens=2`. El checksum del embedding usado en posición 1 coincide con una ejecución independiente iniciada directamente desde token `1124`, y no coincide con el residual final de la posición anterior.

El segundo head recorre las 151936 filas. Su checksum F32 y argmax `11287` se recalculan con `dequantize_row_q6_K` oficial de llama.cpp.

## Fail-closed

El modo real exige:

- `--full-moe` y `--final-head`;
- entre 1 y 64 pasos;
- todas las 48 capas del manifest exacto;
- contexto entre 1 y 4096 y `steps <= ctx`;
- KV INT8, temperatura cero y benchmark desactivado;
- dimensiones Qwen3-30B-A3B exactas.

## Límite de honestidad

Validado: plumbing autoregresivo QX, re-embedding, posición, KV persistente, dos forwards completos, ambos heads y alimentación desde IDs producidos por [[qwen3-tokenizer]]. Pendiente: cobertura Unicode/chat-template exhaustiva y paridad externa end-to-end de todos los residuales y de la secuencia contra otro runtime Qwen3. Esta ejecución no es una medición de tok/s sostenido.

Forward: [[moe-forward]]. Head: [[final-output-head]]. Estado: [[current-status-and-roadmap]].
