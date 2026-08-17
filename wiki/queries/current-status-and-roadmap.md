---
title: Current Status and Roadmap
created: 2026-08-17
updated: 2026-08-17
type: query
tags: [roadmap, runtime, qwen3-moe, risk]
sources: [raw/project/project-state-2026-08-17.md]
confidence: high
---

# Current status and roadmap

## Entregado

- Conversión GGUF tensor-copy a [[qxf-format]].
- Loader, checksums y decoders quant necesarios.
- Attention real completa de layer 0.
- [[moe-forward]] real completo de layer 0.
- State loop real 0→47 para un token: residual, attention normalizada, Q/K RMSNorm, RoPE/GQA, KV INT8 y MoE top-8 en las 48 capas.
- Golden independientes para embedding, IQ4_XS e IQ2_XS/IQ3_XXS representativos.
- Smoke check y suite pytest.

## Gate activo

```text
state loop real layers 0–47: GREEN
→ final norm
→ lm_head completo
→ tokenizer
greedy tokens idénticos
```

`state-loop-probe --full-moe` ejecutó las 48 capas y verificó los 47 enlaces adyacentes de checksum. La medición de un token fue ~8.50 s; no incluye final norm, lm_head ni selección real del token siguiente.

## Después

1. Implementar final RMSNorm y lm_head completo.
2. Comparar logits y greedy token contra referencia externa.
3. Implementar el ciclo autoregresivo multi-token.
4. Aplicar [[optimization-priorities]] CPU.
5. Medir baseline de inferencia real.
6. Diseñar backend CUDA híbrido.
7. Gates 4K, RSS, calidad KV y 8 h.

## Riesgos

- Layout/padding de futuros GGUF.
- Tipos quant distintos por layer.
- Extrapolar probes parciales.
- Cache misses de expertos.
- Modelos locales grandes nunca deben entrar en Git.

Arquitectura: [[architecture]]. Evidencia: [[numerical-correctness]].
