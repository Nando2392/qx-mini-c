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

La política de investigación está en [[auto-research-loop]] y el avance en [[current-status-and-roadmap]].
