---
title: Numerical Correctness
created: 2026-08-17
updated: 2026-08-17
type: concept
tags: [validation, golden, testing, quantization]
sources: [raw/project/project-state-2026-08-17.md]
confidence: high
---

# Numerical correctness

## Estrategia

Cada slice se valida antes de combinarlo:

1. Embedding Q4_K.
2. RMSNorm.
3. Q/K/V IQ4_XS.
4. RoPE split-half.
5. KV INT8 y GQA 32/4.
6. Softmax/contexto.
7. Attention output 4096→2048.
8. Residual y FFN RMSNorm.
9. Router/top-8.
10. Expert gate/up/down y SwiGLU.
11. [[final-output-head]] F32 + Q6_K sobre vocabulario completo.

## Referencias independientes

- Python lee bytes QXF directamente para Q4_K e IQ4_XS.
- Un helper enlazado con llama.cpp valida filas IQ2_XS/IQ3_XXS.
- El mismo helper usa `dequantize_row_q6_K` oficial para recalcular las 151936 logits y el argmax.
- El output C nunca se reutiliza como referencia esperada.

## Gates finales pendientes

```text
tokens greedy idénticos
comparación de logits/capas
Δppl KV INT8 <0.5%
contexto 4K sin fugas
RSS ±5%
8 h sostenidas y guarda térmica
```

## Baseline externo por capa

El oracle standalone de llama.cpp extrae sidecars F32 de inputs de capa sin reutilizar resultados QX. Para token `42`, QX y llama.cpp coinciden exactamente en el input de layer 0 (`max_abs=0`, cosine `1`) y ambos producen argmax `1124`. La primera divergencia está en el input de layer 1 (`max_abs≈1.08`, cosine≈`0.964`). Repetir llama.cpp con KV Q8_0 en vez de F16 no mueve esa frontera; el siguiente experimento debe separar attention output y MoE de layer 0 mediante tensors internos del oracle.

Este baseline localiza el error; no constituye PASS de paridad externa.

La política de investigación está en [[auto-research-loop]] y el avance en [[current-status-and-roadmap]].
