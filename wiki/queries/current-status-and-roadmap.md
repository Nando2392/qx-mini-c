---
title: Current Status and Roadmap
created: 2026-08-17
updated: 2026-08-18
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
- Golden independientes para embedding, IQ4_XS y expertos IQ2_XS/IQ3_XXS/IQ2_S representativos.
- Smoke check y suite pytest.

## Gate activo

```text
state loop real layers 0–47: GREEN
final norm + lm_head completo: GREEN
autoregresión multi-token correcta: GREEN
tokenizer parity para prompts fijos: GREEN
QXF corruption/legacy-clamp gate: GREEN
modo Q8_K CPU compatible: GREEN, opt-in
F32 continúa default: DECIDIDO
bisect MoE layer 0 por etapa/experto: GREEN
Q8_K IQ2_XS/IQ3_XXS CPU opt-in: GREEN
Q8_K IQ2_S/IQ4_XS expertos CPU opt-in: GREEN
→ primera divergencia material actual: input layer 2
```

El issue GitHub #7 quedó cerrado como validación completada en el commit `42b3fd8b76acc26efdc7c53b6e7b427825b56b95`. GitHub Actions `32064105028` pasó build, tests y wiki lint. El cierre significa que la hipótesis de paridad fue probada y refutada de forma reproducible; no significa que QX sea numéricamente idéntico a llama.cpp.

El gate [[llama-cpp-parity]] aisló layer 0, encontró y corrigió la falta de renormalización de pesos top-8. Tras el fix, layer-1 cosine sube a `0.999961` (F32/F16), pero exactitud residual/logit queda refutada. La primera diferencia aparece en `Vcur`: QX usa activación F32 y ggml usa activación temporal Q8_K para IQ4_XS.

El gate [[f32-vs-q8k-activation]] implementó `q8_k_compat` como modo CPU explícito. En `Vcur-0` reduce max-abs de `3.05e-4` a `7.45e-9`, usa 4672 bytes de workspace y, en ese baseline anterior, fue ~7.4% más rápido. Ese gate identificó `ffn_moe_out-0` como siguiente objetivo; el resultado supersedente está en [[moe-stage-bisect]].

El gate [[moe-stage-bisect]] usa el mismo `ffn_inp-0` en QX/llama.cpp y cierra router, top-8, pesos, gate/up, SwiGLU, down y mezcla de layer 0 dentro de max-abs `1.20e-6`. El follow-up [[iq2-s-iq4-xs-q8k]] corrige el mapeo (`22=IQ2_S`, `23=IQ4_XS`) y valida layer 1 con el mismo input: mezcla final max-abs `4.35e-5`, RMSE `9.62e-7`, cosine ≈`1`. End-to-end, la perturbación previa de entrada se amplifica y `layer-2-input` mantiene max-abs `28.20` en Q8_K/F32-KV. El kernel queda cerrado; la sensibilidad layer 1→2 no.

`state-loop-probe --full-moe --final-head --steps 2` produce ahora `42 → 1124 → 11287`. El token `1124` se re-embebe en posición 1, cada una de las 48 capas atiende dos posiciones mediante KV INT8 persistente y ambos checksums de 151936 logits se validan con el helper Q6_K oficial.

La comparación externa secuencial está cerrada como refutación: llama F16/Q8_0 produce `[1124, 50853]` para `[42]`; para `Hello!`, llama produce `[358, 1184]`/`[358, 614]` frente a QX `[81379, 44707]`.

## Después

1. Bisectar la amplificación entre input layer 1 y input layer 2 usando el experto dominante 68, sin reabrir los vec-dot ya validados con input idéntico.
2. Ampliar tokenizer a cobertura Unicode/chat-template exhaustiva.
3. Aplicar [[optimization-priorities]] CPU manteniendo A/B F32/Q8_K.
4. Medir baseline de inferencia real.
5. Diseñar backend CUDA híbrido sin asumir temporal Q8_K CPU.
6. Gates 4K, RSS, calidad KV y 8 h.

## Riesgos

- Layout/padding de futuros GGUF.
- Tipos quant distintos por layer.
- Extrapolar probes parciales.
- Cache misses de expertos.
- Modelos locales grandes nunca deben entrar en Git.

Arquitectura: [[architecture]]. Evidencia: [[numerical-correctness]].
