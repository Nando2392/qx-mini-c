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
- Un helper enlazado con llama.cpp valida filas IQ2_XS/IQ3_XXS/IQ2_S/IQ4_XS contra activaciones Q8_K.
- El mismo helper usa `dequantize_row_q6_K` y el vec-dot público `Q6_K × Q8_K`; el oracle completo valida las 151936 logits y el argmax.
- El output C nunca se reutiliza como referencia esperada.

## Gates finales pendientes

```text
tokens greedy multi-token idénticos
bisect layer 2 → logits
Δppl KV INT8 <0.5%
contexto 4K sin fugas
RSS ±5%
8 h sostenidas y guarda térmica
```

## Resultado externo por capa

El oracle standalone de llama.cpp extrae sidecars F32 sin reutilizar resultados QX. La instrumentación interna localizó un defecto real: QX aplicaba las probabilidades globales de router a los expertos top-8 sin renormalizarlas. Qwen3MoE/llama.cpp normaliza los pesos seleccionados por su suma. Un test RED reprodujo el contrato incorrecto y el fix dejó la suma top-8 en `1.0` en las dos rutas QX.

Después del fix, token `42` conserva input layer 0 exacto y argmax `1124` en QX/llama. Layer 1 queda mucho más próximo pero no exacto: F32/F16 `max_abs=0.0058906`, RMSE `0.0007032`, cosine `0.999961`; INT8/Q8_0 `max_abs=0.0108175`, RMSE `0.0008303`, cosine `0.999956`. La primera diferencia medible aparece antes del KV en `Vcur`: `max_abs=0.0003050`, RMSE `8.316e-5`, cosine `0.999943`.

La causa de esa primera diferencia es un contrato de multiplicación distinto: QX usa pesos IQ4_XS decodificados × activación F32, mientras la ruta CPU ggml empareja IQ4_XS con activación temporal Q8_K. El cache también difiere: QX INT8 escala por vector y llama Q8_0 por bloques. El modo diagnóstico F32 QX separa ese efecto sin cambiar el runtime INT8 normal.

La comparación post-Q5_K de 151936 logits aún refuta igualdad exacta: F32/F32-KV frente a llama F16 tiene max-abs `1.57686`, RMSE `0.203839`, cosine `0.998247`; Q8_K/F32-KV tiene max-abs `0.171505`, RMSE `0.0346769`, cosine `0.999936486`. Ambos conservan argmax `1124`. Detalle y comandos: [[layer1-layer2-sensitivity]].

El residual final pre-head también se mide directamente. Post-Q5_K, F32/F16 queda en `max_abs=199.862`, RMSE `6.67964`, cosine `0.997511`; Q8_K/F32-KV queda en `max_abs=1.09668`, RMSE `0.0932631`, cosine `0.999996986`.

El oracle secuencial pasa paridad greedy a dos tokens para la matriz fija post-Q5_K. Para `[42]`, llama F16/Q8_0 y QX F32/INT8 producen `[1124, 50853]`. Para `Hello!` (`[9707, 0]`), QX produce `[358, 1184]`, igual a llama F16; llama Q8_0 produce `[358, 614]`.

Este resultado prueba secuencia para dos prompts y dos tokens; no prueba paridad exacta de todos los logits/residuos ni cobertura exhaustiva de prompts.

## Gate Q8_K compatible

**Implementado y medido:** `q8_k_compat` reproduce el temporal CPU de ggml para proyecciones IQ4_XS, Q5_K y Q6_K, incluido el lm_head completo. Tipos sin golden conservan fallback F32 explícito. F32 continúa siendo el default.

La serialización QX coincide byte por byte con `quantize_row_q8_K_ref` del oracle fijado para cuatro distribuciones: mixed, todo positivo, todo negativo y extremos alternos ±1. El gate cubre signo de escala, redondeo, clamp, `qs` y `bsums`; no se infiere equivalencia sólo de logits.

En `Vcur-0`, el modo reduce max-abs de `0.000305031` a `7.45e-9` y RMSE de `8.316e-5` a `1.158e-9`. En `kqv_out-0`, max-abs baja de `0.000305337` a `1.486e-5`. La mejora no cierra el forward: la siguiente diferencia material aparece en `ffn_moe_out-0`; logits Q8_K/F32-KV mantienen max-abs `9.09348`, RMSE `1.47465`, cosine `0.875527` y argmax `1124`.

El bisect posterior [[moe-stage-bisect]] amplió el modo opt-in a expertos IQ2_XS/IQ3_XXS. [[iq2-s-iq4-xs-q8k]] añadió goldens separados para `22=IQ2_S` y `23=IQ4_XS`. [[layer1-layer2-sensitivity]] encontró y corrigió el decoder Q5_K y añadió `Q5_K × Q8_K`. Con el mismo `attn_norm-1` y KV F16, `attn_out-1` alcanza max-abs `6.33e-8`; con el mismo `ffn_inp-1`, la mezcla MoE mantiene max-abs `4.35e-5`. Post-fix, layer-2-input Q8_K/F32-KV queda en max-abs `0.294106`, RMSE `0.00660423`, cosine `0.999999906`. La diferencia restante es sensibilidad cuantizada: el top-8 no cambia y el experto 68 aporta `99.03%` del delta MoE.

El final norm también se captura desde la API pública de embeddings del oracle. Post-Q5_K, Q8_K/F32-KV frente a llama F16 queda en max-abs `0.0659037`, RMSE `0.00731730`, cosine `0.999986548`; mejora fuertemente, pero no convierte la comparación en igualdad exacta.

La matriz greedy fija queda GREEN post-Q5_K. La decisión, matriz numérica, índices máximos y trade-offs están en [[layer1-layer2-sensitivity]] y [[f32-vs-q8k-activation]].

[[final-head-q6k-q8k]] cierra `l_out-47 → RMSNorm final → logits` con input idéntico. Same-input logits quedan en max-abs `2.38419e-6`, RMSE `4.91155e-7`; globalmente, el kernel correcto mejora RMSE de `0.0393805` a `0.0257469`, pero `l_out-47` sigue divergente y la paridad exacta permanece refutada.

La política de investigación está en [[auto-research-loop]] y el avance en [[current-status-and-roadmap]].
