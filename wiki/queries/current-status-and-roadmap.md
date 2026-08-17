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
- [[final-output-head]] completo: final RMSNorm, 151936 logits Q6_K, top-N y argmax.
- [[autoregressive-loop]] greedy multi-token: re-embedding, posición y KV persistente por layer.
- [[qwen3-tokenizer]] QXT2: paridad exacta para prompts fijos y prefill desde texto.
- Hardening QXF fail-closed: manifest, ABI, directorio, dims, nombres, placement, overflow y filas exactas.
- Golden independientes para embedding, IQ4_XS e IQ2_XS/IQ3_XXS representativos.
- Smoke check y suite pytest.

## Gate activo

```text
state loop real layers 0–47: GREEN
final norm + lm_head completo: GREEN
autoregresión multi-token correcta: GREEN
tokenizer parity para prompts fijos: GREEN
QXF corruption/legacy-clamp gate: GREEN
→ tokens greedy end-to-end idénticos
```

`state-loop-probe --full-moe --final-head --steps 2` produjo `42 → 1124 → 29626`. El token `1124` se re-embebe en posición 1, cada una de las 48 capas atiende dos posiciones mediante KV INT8 persistente y el segundo checksum de 151936 logits coincide con el helper Q6_K oficial. `prompt-state-loop-probe` tokenizó `Hello!` como `[9707, 0]`, hizo prefill y continuó greedy con inputs `[9707, 0, 117268]`.

## Después

1. Comparar tokens y residuales end-to-end contra referencia externa.
2. Ampliar tokenizer a cobertura Unicode/chat-template exhaustiva.
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
