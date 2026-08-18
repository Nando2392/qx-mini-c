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
- Golden independiente y dispatch CPU opt-in para experto down IQ3_S × Q8_K.
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
Q5_K decode + Q5_K×Q8_K atención CPU opt-in: GREEN
atención layer 1 con mismo attn_norm/KV F16: GREEN
sensibilidad layer 1→2 clasificada: GREEN
Q6_K decode + Q6_K×Q8_K atención CPU opt-in: GREEN
sweep layer 2→47 y sensibilidad layer 46 clasificada: GREEN
final RMSNorm + lm_head Q6_K×Q8_K same-input: GREEN
layer 47 attention + MoE same-input: GREEN
layers 44–42 attention + MoE same-input: GREEN
layer 41 IQ3_S down + bloque same-input: GREEN post-fix
layers 40–0 attention + MoE same-input: GREEN (41/41)
→ paridad global/logits/greedy: pendiente
```

El issue GitHub #7 quedó cerrado como validación completada en el commit `42b3fd8b76acc26efdc7c53b6e7b427825b56b95`. GitHub Actions `32064105028` pasó build, tests y wiki lint. El cierre significa que la hipótesis de paridad fue probada y refutada de forma reproducible; no significa que QX sea numéricamente idéntico a llama.cpp.

El gate [[llama-cpp-parity]] aisló layer 0, encontró y corrigió la falta de renormalización de pesos top-8. Tras el fix, layer-1 cosine sube a `0.999961` (F32/F16), pero exactitud residual/logit queda refutada. La primera diferencia aparece en `Vcur`: QX usa activación F32 y ggml usa activación temporal Q8_K para IQ4_XS.

El gate [[f32-vs-q8k-activation]] implementó `q8_k_compat` como modo CPU explícito. En `Vcur-0` reduce max-abs de `3.05e-4` a `7.45e-9`, usa 4672 bytes de workspace y, en ese baseline anterior, fue ~7.4% más rápido. Ese gate identificó `ffn_moe_out-0` como siguiente objetivo; el resultado supersedente está en [[moe-stage-bisect]].

El gate [[moe-stage-bisect]] usa el mismo `ffn_inp-0` en QX/llama.cpp y cierra router, top-8, pesos, gate/up, SwiGLU, down y mezcla de layer 0 dentro de max-abs `1.20e-6`. [[iq2-s-iq4-xs-q8k]] valida layer 1 con el mismo input: mezcla final max-abs `4.35e-5`, RMSE `9.62e-7`, cosine ≈`1`. [[layer1-layer2-sensitivity]] encontró un bug Q5_K real, añadió `Q5_K × Q8_K` y cerró atención layer 1 same-input. El sweep supersedente [[layer2-logits-sweep]] encuentra la siguiente amplificación en layer 46→47, añade `Q6_K × Q8_K` y reduce `ffn_inp-46` same-input de max-abs `0.00445557` a `1.19e-7`. End-to-end no mejora: layer-47 RMSE queda `0.0309398` y logits RMSE `0.0393805`; el bisect causal sitúa la amplificación en MoE (`3.663×`), con top-8 estable y experto 74 aportando `76.08%` del delta.

[[layer47-same-input]] cierra el último bloque con el `layer-47.f32` exacto: atención llega a `ffn_inp-47` con max-abs `6.10e-5`; la cadena attention→MoE reconstruye `l_out-47` con max-abs `2.57e-4`, RMSE `8.01e-6`, cosine ≈`1`, y top-8 exacto `[83,3,74,119,92,28,109,101]`. El delta dominante restante proviene de sensibilidad de reducción F32 del router multiplicada por outputs down grandes, no de un nuevo decoder quant roto. La divergencia global entra acumulada desde capas anteriores y no queda resuelta por este gate.

El bisect descendente [[layer41-iq3s-q8k]] cerró layers 44–42 y localizó en layer 41 el primer fallo material observado en esa ejecución: atención, routing y SwiGLU cerraban, pero `ffn_down_exps` IQ3_S seguía en `dequant_f32`. La matriz pytest versionada regenera sidecars y fija routing/métricas exactas para layers 24, 41, 42, 43 y 44. El golden real `IQ3_S × Q8_K` y el dispatch opt-in reducen down a max-abs `9.54e-7` y reconstruyen `l_out-41` con max-abs `4.76e-5`, RMSE `1.05e-6`, routing exacto `[48,73,69,18,96,104,88,26]`. El siguiente bisect empieza antes de layer 41; no se infiere todavía el origen acumulado global.

[[layers0-40-same-input]] completa ese intervalo: 41/41 bloques cierran con el residual exacto del oracle y routing exacto. Los máximos materiales son `Vcur=1.79e-7`, `kqv_out=1.91e-6`, weighted `1.83105e-4` y `l_out=2.32019e-4`/RMSE `5.12959e-6`; todos pasan. Treinta y siete capas conservan sólo warnings de router logits por encima de `2e-6`, sin cambio top-8 ni fallo downstream. La hipótesis de otro seam local material queda refutada para este input; la divergencia global requiere ahora un bisect de acumulación, no otro kernel especulativo.

`state-loop-probe --full-moe --final-head --steps 2` produce ahora `42 → 1124 → 50853`. El token `1124` se re-embebe en posición 1, cada una de las 48 capas atiende dos posiciones mediante KV INT8 persistente y ambos checksums de 151936 logits se validan con el helper Q6_K oficial.

La comparación externa secuencial fija queda GREEN post-Q5_K: QX F32/INT8 coincide con llama F16/Q8_0 en `[1124, 50853]` para `[42]`, y coincide con llama F16 en `[358, 1184]` para `Hello!`. Llama Q8_0 produce `[358, 614]`; cobertura exhaustiva y paridad exacta de logits siguen pendientes.

## Después

1. Medir acumulación adyacente con replay híbrido/inyección de residual sobre el forward real.
2. Ampliar tokenizer y matriz greedy a cobertura Unicode/chat-template y prompts múltiples.
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
