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
- State loop real 0→1 para un token: residual, attention normalizada, Q/K RMSNorm, RoPE/GQA, KV INT8 y MoE top-8 en ambas capas.
- Golden independientes para embedding, IQ4_XS e IQ2_XS/IQ3_XXS representativos.
- Smoke check y suite pytest.

## Gate activo

```text
state loop real layers 0 y 1: GREEN
→ extender a 48 layers
→ final norm
→ lm_head completo
→ tokenizer
greedy tokens idénticos
```

`state-loop-probe --full-moe` propaga checksums reales entre layers 0 y 1. El checksum de salida de layer 0 coincide con el checksum de entrada de layer 1.

## Después

1. Completar correctitud multi-layer.
2. Implementar el ciclo autoregresivo multi-token con embedding nuevo por token.
3. Aplicar [[optimization-priorities]] CPU.
4. Medir baseline de inferencia real.
5. Diseñar backend CUDA híbrido.
6. Gates 4K, RSS, calidad KV y 8 h.

## Riesgos

- Layout/padding de futuros GGUF.
- Tipos quant distintos por layer.
- Extrapolar probes parciales.
- Cache misses de expertos.
- Modelos locales grandes nunca deben entrar en Git.

Arquitectura: [[architecture]]. Evidencia: [[numerical-correctness]].
