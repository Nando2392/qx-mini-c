---
title: Current Status and Roadmap
created: 2026-08-17
updated: 2026-08-23
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
- Issue #22 COMPLETED: matriz separada de Unicode, render Qwen3 chat-template y secuencias greedy para prompts múltiples, publicada en `f8facc600e4df708af22b5ea0e230dc1cf783ad1` con fix CI `333ee3df1b71a07b2912475ab423bc4bad12836f`; GitHub Actions `32409146859` terminó SUCCESS. La paridad demostrada sigue limitada a los casos y modalidades registrados; no implica paridad global de logits/modelo.
- [[cpu-inference-baseline]] fail-closed: A/B F32/Q8_K con startup/model-load, prefill, decode, total, peak RSS, provenance SHA-256 y outputs deterministas por modo.
- [[accumulated-kv-snapshot-replay]] fail-closed: captura/restaura K/V y escalas por layer/posición, con token de continuación y manifiesto SHA-256.
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
replay híbrido residual F16 layers 0–47: GREEN
clasificación acumulación/amplificación/modalidad: GREEN
perturbación escalada layer 1: GREEN; respuesta no suave, top-8 estable, cruces de orden observados
snapshot/replay de KV acumulado: GREEN; baseline 3 posiciones == captura 2 + replay 1
baseline CPU A/B F32/Q8_K: GREEN local; prefill/decode/total/RSS separados
QXF mmap read-only opt-in: GREEN local WSL2; gate 2x2 exacto por modalidad, buffered permanece default
scratch persistente opt-in: GREEN local; gate 2x2x2 exacto por modalidad/backend, ephemeral permanece default
policy provenance gates #27–#40: GREEN local/CI; Issue #41 long-context summary consistency en verificación local
→ paridad global/logits/greedy: pendiente
```

El issue GitHub #7 quedó cerrado como validación completada en el commit `42b3fd8b76acc26efdc7c53b6e7b427825b56b95`. GitHub Actions `32064105028` pasó build, tests y wiki lint. El cierre significa que la hipótesis de paridad fue probada y refutada de forma reproducible; no significa que QX sea numéricamente idéntico a llama.cpp.

El gate [[llama-cpp-parity]] aisló layer 0, encontró y corrigió la falta de renormalización de pesos top-8. Tras el fix, layer-1 cosine sube a `0.999961` (F32/F16), pero exactitud residual/logit queda refutada. La primera diferencia aparece en `Vcur`: QX usa activación F32 y ggml usa activación temporal Q8_K para IQ4_XS.

El gate [[f32-vs-q8k-activation]] implementó `q8_k_compat` como modo CPU explícito. En `Vcur-0` reduce max-abs de `3.05e-4` a `7.45e-9`, usa 4672 bytes de workspace y, en ese baseline anterior, fue ~7.4% más rápido. Ese gate identificó `ffn_moe_out-0` como siguiente objetivo; el resultado supersedente está en [[moe-stage-bisect]].

El gate [[moe-stage-bisect]] usa el mismo `ffn_inp-0` en QX/llama.cpp y cierra router, top-8, pesos, gate/up, SwiGLU, down y mezcla de layer 0 dentro de max-abs `1.20e-6`. [[iq2-s-iq4-xs-q8k]] valida layer 1 con el mismo input: mezcla final max-abs `4.35e-5`, RMSE `9.62e-7`, cosine ≈`1`. [[layer1-layer2-sensitivity]] encontró un bug Q5_K real, añadió `Q5_K × Q8_K` y cerró atención layer 1 same-input. El sweep supersedente [[layer2-logits-sweep]] encuentra la siguiente amplificación en layer 46→47, añade `Q6_K × Q8_K` y reduce `ffn_inp-46` same-input de max-abs `0.00445557` a `1.19e-7`. End-to-end no mejora: layer-47 RMSE queda `0.0309398` y logits RMSE `0.0393805`; el bisect causal sitúa la amplificación en MoE (`3.663×`), con top-8 estable y experto 74 aportando `76.08%` del delta.

[[layer47-same-input]] cierra el último bloque con el `layer-47.f32` exacto: atención llega a `ffn_inp-47` con max-abs `6.10e-5`; la cadena attention→MoE reconstruye `l_out-47` con max-abs `2.57e-4`, RMSE `8.01e-6`, cosine ≈`1`, y top-8 exacto `[83,3,74,119,92,28,109,101]`. El delta dominante restante proviene de sensibilidad de reducción F32 del router multiplicada por outputs down grandes, no de un nuevo decoder quant roto. La divergencia global entra acumulada desde capas anteriores y no queda resuelta por este gate.

El bisect descendente [[layer41-iq3s-q8k]] cerró layers 44–42 y localizó en layer 41 el primer fallo material observado en esa ejecución: atención, routing y SwiGLU cerraban, pero `ffn_down_exps` IQ3_S seguía en `dequant_f32`. La matriz pytest versionada regenera sidecars y fija routing/métricas exactas para layers 24, 41, 42, 43 y 44. El golden real `IQ3_S × Q8_K` y el dispatch opt-in reducen down a max-abs `9.54e-7` y reconstruyen `l_out-41` con max-abs `4.76e-5`, RMSE `1.05e-6`, routing exacto `[48,73,69,18,96,104,88,26]`. El siguiente bisect empieza antes de layer 41; no se infiere todavía el origen acumulado global.

[[layers0-40-same-input]] completa ese intervalo: 41/41 bloques cierran con el residual exacto del oracle y routing exacto. Los máximos materiales son `Vcur=1.79e-7`, `kqv_out=1.91e-6`, weighted `1.83105e-4` y `l_out=2.32019e-4`/RMSE `5.12959e-6`; todos pasan. Treinta y siete capas conservan sólo warnings de router logits por encima de `2e-6`, sin cambio top-8 ni fallo downstream. La hipótesis de otro seam local material queda refutada para este input; la divergencia global requiere ahora un bisect de acumulación, no otro kernel especulativo.

[[hybrid-residual-replay-accumulation]] completa el bisect de acumulación con KV F16 modal-equivalente. El baseline desde layer 0 termina con RMSE `9.87305e-2`; inyectar sólo el residual exacto de layer 1 reduce el final a `7.69131e-6` (`12836.6×`), aunque el error entrante original en layer 1 era apenas `1.46772e-8`. Esto prueba error acumulado/global y amplificación posterior de una discrepancia microscópica de layer 0, sin abrir otro seam local material. El control F32 no cierra (`9.05740e-2` tras replay de layer 1) y queda clasificado como efecto modal. La inyección sustituye sólo residual: el KV lo recalcula QX y no equivale a replay de KV acumulado para secuencias multi-token.

[[scaled-layer1-residual-sensitivity]] completa la perturbación escalada sobre la dirección observada de layer 0. La respuesta no es suave: escalas exactas `-1` y `+1` tienen el mismo L2 de entrada (`6.64213e-7`) pero producen deltas finales de `5.73359e-4` y `4.46778`. Ninguna de las 15 escalas cambia la membresía top-8. Sólo aparecen cruces de orden dentro del mismo set en layers 46 y 28; están correlacionados con ramas de respuesta alta, pero no prueban causalidad. No se autoriza un fix numérico: el siguiente gate debe separar orden de acumulación, thresholds de activación y modalidad sobre más tokens.

[[scaled-residual-token-modality-matrix]] completa ese gate con 18 celdas: tokens `42,9707,0`, activaciones F32/Q8_K-compatible y KV F16/F32/INT8. Diecisiete celdas cambian orden y catorce cambian membresía en alguna escala, pero las tres celdas runtime-aligned Q8_K-compatible + F16 conservan la membresía top-8. La asimetría `-1/+1` es extrema para tokens `42` (`7792.3×`) y `0` (`105030×`) en ese slice, mientras `9707` queda en `1.54585×` sin transición. El resultado separa sensibilidad de token/modalidad; no autoriza un fix ni una conclusión multi-token.

[[accumulated-kv-snapshot-replay]] cierra el seam previo al experimento multi-token. El payload nativo `QXKVSNP1` v2 conserva K/V, escalas, geometría, posición, seed y siguiente token, y el importador C verifica su trailer SHA-256 antes de consumir el cache. El manifiesto externo fija modelo/binario/revisión, SHA-256 y cobertura exacta por `(layer,position,kind)`. El control sintético exige igualdad exacta entre baseline de tres posiciones y captura de dos + replay de una. Truncación, bytes extra, magic incorrecto, mutación K/V de igual longitud y token de continuación distinto fallan cerrados. El gate se publicó en `0fc21c697994782b1f393dabd198855ef5ab939f` con CI `32370138054` SUCCESS; habilita experimentos posteriores, pero todavía no prueba paridad ni causalidad multi-token.

[[accumulated-kv-multi-token-perturbation-matrix]] usa ese seam sobre un QXF real y separa por KV F16/INT8 una dirección sintética F32 controlada en la continuación tras dos posiciones acumuladas. Baseline/replay y escala cero cierran exactamente en ambos modos. Las seis corridas conservan token `56`; F16 `+1` cambia orden de routing y las otras tres perturbaciones firmadas no. El resultado prueba mecánica reproducible en el slice registrado, no causalidad semántica ni equivalencia modal. Issue #21 fue publicado/completado en `4066538e88ccc0a18fafa213b606e9e619a3f9b5`; GitHub Actions `32390221256` terminó SUCCESS.

`state-loop-probe --full-moe --final-head --steps 2` produce ahora `42 → 1124 → 50853`. El token `1124` se re-embebe en posición 1, cada una de las 48 capas atiende dos posiciones mediante KV INT8 persistente y ambos checksums de 151936 logits se validan con el helper Q6_K oficial.

La comparación externa secuencial fija queda GREEN post-Q5_K: QX F32/INT8 coincide con llama F16/Q8_0 en `[1124, 50853]` para `[42]`, y coincide con llama F16 en `[358, 1184]` para `Hello!`. Llama Q8_0 produce `[358, 614]`; cobertura exhaustiva y paridad exacta de logits siguen pendientes.

[[qxf-mmap-io]] añade un backend QXF read-only explícitamente opt-in sin alterar kernels. El JSON final de Issue #24 (`wiki/evidence/issue-24-qxf-mmap-baseline.json`) supersede números antiguos: el gate 2×2 preserva exactamente outputs buffered/mmap dentro de F32 y `q8_k_compat`; observa ratios de wall-clock total `2.12548×` y `1.96456×`, respectivamente, mientras mmap añade aproximadamente `2.35 GB` de peak RSS y prefill/decode nativos no demuestran mejora material. Buffered sigue siendo default y la evidencia no autoriza inferencias de throughput global.

[[persistent-scratch-buffers]] completa Issue #25 para la prioridad CPU 2. La política `--scratch-policy persistent` queda opt-in; `ephemeral` sigue default/control. En la matriz 2×2×2 con 1 warm-up y 3 mediciones por celda, persistent reduce `480` malloc y `672` frees medianos por activación/backend, retiene `65,536` bytes de scratch, conserva outputs exactos por modalidad/backend y no muestra mejora wall-clock material. No se inicia prioridad 3 desde este resultado.

Issue #34 añade `--sampling-policy none` como contrato fail-closed de provenance para separar greedy determinístico de futuros samplers. `none` queda default, `sampling_profile` reporta `mode=greedy`, `stochastic_samples=0`, `top_p_evaluations=0` y `beam_width=1`; políticas no soportadas fallan antes de prompt/model/tokenizer I/O. No implementa top-p/min-p/beam, no promueve default y no afirma speedup/calidad.

Issue #35 añade `--long-context-policy none` como contrato fail-closed para separar el baseline actual de futuros gates 4K/RSS/KV-quality/soak. `none` queda default, `long_context_profile` reporta contadores inactivos en cero; políticas no soportadas fallan antes de prompt/model/tokenizer I/O. No ejecuta benchmark 4K, no aplica límite RSS, no mide calidad KV/8h y no afirma speedup/calidad.

Issue #36 añade `--long-context-policy ctx4k-smoke` como primer gate de admisión 4K: exige `--ctx >= 4096` antes de prompt/model/tokenizer I/O, reporta `target_ctx_tokens=4096` y conserva RSS/KV-quality/soak inactivos. `none` sigue default. No mide throughput 4K, no promueve default y no afirma speedup/calidad.

Issue #37 añade `--long-context-rss-limit-bytes` como gate RSS opt-in para los experimentos `ctx4k-smoke`: default `0` queda deshabilitado, valores no cero sólo son válidos con `ctx4k-smoke`, y el harness falla cerrado si el `peak_rss_bytes` muestreado excede el límite. No instala límite duro de OS, no cambia allocator, no mide calidad KV/8h y no afirma speedup/calidad.

Issue #38 añade `--long-context-kv-quality-checks` como contrato fail-closed para futuros sweeps de calidad KV: default `0` queda deshabilitado y valores non-zero fallan antes de prompt/model/tokenizer I/O. No ejecuta sweep KV, no corre soak, no promueve defaults y no afirma calidad.

Issue #39 añade `--long-context-soak-seconds` como contrato fail-closed para futuros runners de soak long-context: default `0` queda deshabilitado y valores non-zero fallan antes de prompt/model/tokenizer I/O. No ejecuta soak, no promueve defaults y no afirma estabilidad.

Issue #40 conserva el `long_context_profile` validado dentro de cada compact-run del harness de benchmark. Es sólo provenance por medición: no ejecuta benchmark 4K nuevo, no cambia defaults, no implementa quality sweep/soak y no afirma rendimiento, calidad o estabilidad.

Issue #41 conserva el `long_context_profile` común dentro de cada summary del harness y falla cerrado si las mediciones de una celda mezclan perfiles. Es sólo consistencia de provenance: no ejecuta benchmark 4K nuevo, no cambia defaults, no implementa quality sweep/soak y no afirma rendimiento, calidad o estabilidad.

## Después

1. Aplicar [[optimization-priorities]] CPU manteniendo A/B F32/Q8_K.
2. Diseñar backend CUDA híbrido sin asumir temporal Q8_K CPU.
3. Convertir contratos 4K/RSS/calidad KV/soak en mediciones reales sólo con gates reproducibles.

## Riesgos

- Layout/padding de futuros GGUF.
- Tipos quant distintos por layer.
- Extrapolar probes parciales.
- Cache misses de expertos.
- Modelos locales grandes nunca deben entrar en Git.

Arquitectura: [[architecture]]. Evidencia: [[numerical-correctness]].
