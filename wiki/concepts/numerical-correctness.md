---
title: Numerical Correctness
created: 2026-08-17
updated: 2026-08-18
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
tokens greedy multi-token idénticos
golden Q8_K para expertos tipos 22/23 después del bisect layer 0
Δppl KV INT8 <0.5%
contexto 4K sin fugas
RSS ±5%
8 h sostenidas y guarda térmica
```

## Resultado externo por capa

El oracle standalone de llama.cpp extrae sidecars F32 sin reutilizar resultados QX. La instrumentación interna localizó un defecto real: QX aplicaba las probabilidades globales de router a los expertos top-8 sin renormalizarlas. Qwen3MoE/llama.cpp normaliza los pesos seleccionados por su suma. Un test RED reprodujo el contrato incorrecto y el fix dejó la suma top-8 en `1.0` en las dos rutas QX.

Después del fix, token `42` conserva input layer 0 exacto y argmax `1124` en QX/llama. Layer 1 queda mucho más próximo pero no exacto: F32/F16 `max_abs=0.0058906`, RMSE `0.0007032`, cosine `0.999961`; INT8/Q8_0 `max_abs=0.0108175`, RMSE `0.0008303`, cosine `0.999956`. La primera diferencia medible aparece antes del KV en `Vcur`: `max_abs=0.0003050`, RMSE `8.316e-5`, cosine `0.999943`.

La causa de esa primera diferencia es un contrato de multiplicación distinto: QX usa pesos IQ4_XS decodificados × activación F32, mientras la ruta CPU ggml empareja IQ4_XS con activación temporal Q8_K. El cache también difiere: QX INT8 escala por vector y llama Q8_0 por bloques. El modo diagnóstico F32 QX separa ese efecto sin cambiar el runtime INT8 normal.

La comparación de 151936 logits refuta paridad completa: F32/F16 tiene max-abs `9.09395`, RMSE `1.47826`, cosine `0.874880`; INT8/Q8_0 tiene max-abs `9.13947`, RMSE `1.48539`, cosine `0.875121`. Ambos conservan argmax `1124`. Detalle y comandos: [[llama-cpp-parity]].

El residual final pre-head también se mide directamente: F32/F16 `max_abs=1099.6502`, RMSE `32.9520`, cosine `0.439279`; INT8/Q8_0 `max_abs=1100.9628`, RMSE `33.0204`, cosine `0.440512`.

El oracle secuencial también refuta paridad greedy a dos tokens. Para `[42]`, llama produce `[1124, 50853]` y QX `[1124, 11287]`. Para `Hello!` (`[9707, 0]`), llama F16/Q8_0 produce `[358, 1184]`/`[358, 614]`, mientras QX produce `[81379, 44707]`.

Este resultado es una refutación reproducible de paridad exacta, no un PASS numérico.

## Gate Q8_K compatible

**Implementado y medido:** `q8_k_compat` reproduce el temporal CPU de ggml para proyecciones IQ4_XS y mantiene fallback F32 explícito para Q5_K/Q6_K. F32 continúa siendo el default.

La serialización QX coincide byte por byte con `quantize_row_q8_K_ref` del oracle fijado para cuatro distribuciones: mixed, todo positivo, todo negativo y extremos alternos ±1. El gate cubre signo de escala, redondeo, clamp, `qs` y `bsums`; no se infiere equivalencia sólo de logits.

En `Vcur-0`, el modo reduce max-abs de `0.000305031` a `7.45e-9` y RMSE de `8.316e-5` a `1.158e-9`. En `kqv_out-0`, max-abs baja de `0.000305337` a `1.486e-5`. La mejora no cierra el forward: la siguiente diferencia material aparece en `ffn_moe_out-0`; logits Q8_K/F32-KV mantienen max-abs `9.09348`, RMSE `1.47465`, cosine `0.875527` y argmax `1124`.

El bisect posterior [[moe-stage-bisect]] amplió el modo opt-in a expertos IQ2_XS/IQ3_XXS. Con el mismo `ffn_inp`, router, top-8 y todas las etapas de layer 0 quedan dentro de max-abs `1.20e-6`. End-to-end `ffn_moe_out-0` mejora a max-abs `0.000701189`, RMSE `0.000140910`, cosine `0.999997595`. La primera divergencia material se desplaza a input layer 2 porque layer 1 usa tipos 22/23 y mantiene fallback F32.

El final norm también se captura desde la API pública de embeddings del oracle. Tras el bisect MoE, Q8_K/F32-KV frente a llama F16 mantiene max-abs `8.65504`, RMSE `1.07496`, cosine `0.810037`; por tanto, la normalización final no recupera la divergencia acumulada.

Las secuencias greedy siguen divergentes. La decisión, matriz completa, índices máximos y trade-offs están en [[f32-vs-q8k-activation]].

La política de investigación está en [[auto-research-loop]] y el avance en [[current-status-and-roadmap]].
