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
- Golden independientes para embedding, IQ4_XS e IQ2_XS/IQ3_XXS representativos.
- Smoke check y suite pytest.

## Gate activo

```text
state loop con residual real
→ layers 0 y 1
→ 48 layers
→ final norm
→ lm_head completo
→ tokenizer
greedy tokens idénticos
```

Existe un test RED para `state-loop-probe --full-moe`; debe permanecer documentado como trabajo en curso hasta quedar GREEN.

## Después

1. Completar correctitud multi-layer.
2. Aplicar [[optimization-priorities]] CPU.
3. Medir baseline de inferencia real.
4. Diseñar backend CUDA híbrido.
5. Gates 4K, RSS, calidad KV y 8 h.

## Riesgos

- Layout/padding de futuros GGUF.
- Tipos quant distintos por layer.
- Extrapolar probes parciales.
- Cache misses de expertos.
- Modelos locales grandes nunca deben entrar en Git.

Arquitectura: [[architecture]]. Evidencia: [[numerical-correctness]].
