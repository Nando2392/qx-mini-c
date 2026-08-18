# Wiki Log

> Registro cronológico append-only.

## [2026-08-17] create | Bóveda inicializada

- Dominio: QX-mini-MoE y Qwen3-30B-A3B.
- Creada arquitectura de tres capas inspirada en la LLM Wiki de Karpathy.
- Ingeridos estado del proyecto, evidencia de tests, modelo de rendimiento y roadmap.
- La bóveda vive dentro del repositorio para que documentación y código evolucionen juntos.

## [2026-08-17] lint | Revisión inicial

- Wikilinks, frontmatter, índice y provenance creados.
- Claims de rendimiento separados entre medidos, estimados y no implementados.

## [2026-08-17] update | Hardening pre-publicación

- El loader QXF valida tamaño declarado, límites del directorio, rank, terminación de nombres y spans de tensor sin sumas vulnerables a overflow.
- Añadidos dos tests de archivos QXF malformados.
- Gate local: 24 passed, 1 xfailed documentado.

## [2026-08-17] update | State loop real 0→1

- `state-loop-probe --full-moe` ejecuta attention normalizada, Q/K RMSNorm por cabeza, RoPE/GQA, KV INT8 y top-8 MoE en layers 0 y 1.
- El golden Python independiente verifica pesos y orden Q/K RMSNorm antes de RoPE.
- Evidencia de carry: checksum salida layer 0 `15037121990391945191` = checksum entrada layer 1.
- Expertos layer 0: `[49, 89, 92, 48, 108, 58, 4, 38]`.
- Expertos layer 1: `[68, 13, 28, 73, 63, 32, 124, 114]`.
- Probe medido: ~0.34 s para dos layers; no es decode completo.

## [2026-08-17] update | State loop real 0→47

- El runtime ya estaba generalizado; el nuevo gate real de 48 capas pasó sin modificar el kernel.
- Un token ejecutó attention, Q/K RMSNorm, RoPE/GQA, KV INT8 y top-8 MoE en layers 0–47.
- Los 47 enlaces de carry adyacentes pasaron; `layers_run=48`, `kv_appends=48`, `cache_readback_ok=true`.
- Checksums extremos: entrada `8017452295594298460`, salida `675293441229675006`.
- Tiempo medido: ~8.50 s para el probe instrumentado; no es decode completo ni tok/s.
- Próximo gate: final RMSNorm y lm_head completo.

## [2026-08-17] update | Final RMSNorm y output head completo

- Añadido `--final-head` fail-closed sobre el forward real de 48 capas.
- Ejecutadas las 151936 filas Q6_K de `output.weight`; argmax medido: token `1124`.
- Golden independiente: RMSNorm Python + `dequantize_row_q6_K` oficial de llama.cpp para todo el vocabulario.
- El golden detectó un layout incorrecto en el decoder Q6_K propio; corregido contra la fuente oficial.
- Checksum logits F32: `17094101101096419516`; tiempo caliente observado: ~8.35 s.
- Límite: no demuestra todavía paridad end-to-end de todas las capas ni autoregresión multi-token.

## [2026-08-17] update | Autoregresión greedy multi-token

- `state-loop-probe --full-moe --final-head --steps 2` re-embebe cada token seleccionado y avanza posición/KV en las 48 capas.
- Secuencia fija medida: `42 → 1124 → 29626`; `layers_run=96`, `kv_appends=96`.
- El segundo checksum de 151936 logits F32 es `9438484627875866845` y coincide con el helper Q6_K oficial de llama.cpp.
- Límite: tokenizer/BPE y paridad externa end-to-end de residuales/secuencia siguen pendientes; no es benchmark de tok/s.

## [2026-08-17] update | Tokenizer Qwen3 y prefill desde texto

- Añadido sidecar binario QXT2 con metadata GPT-2/qwen2, vocabulario tipado, merges, flags y checksum fail-closed.
- Goldens `llama-tokenize` exactos para ASCII, Unicode, whitespace y ChatML; encode/decode C y adversariales cubiertos.
- `Hello!` produce IDs `[9707, 0]`; el loop hace prefill y dos generaciones con inputs `[9707, 0, 117268]`.
- Evidencia integrada: `layers_run=144`, `kv_appends=144`, tokens seleccionados `[117268, 69336]`.
- Límite: cobertura exhaustiva de Unicode/chat template y paridad residual externa siguen pendientes.

## [2026-08-17] update | Hardening global QXF fail-closed

- El loader valida manifest, ABI 272/208, offsets alineados, dimensiones/tamaños/traits canónicos, nombres únicos, tamaño físico exacto y rangos no solapados con aritmética checked.
- Metadata-only queda permitido sólo con spans de tensor cero; los QXF con datos exigen offsets alineados y tamaños no cero.
- Embedding, forward, logits, residual, token-forward, RMSNorm y matvec-stub ya no truncan spans ni rebobinan silenciosamente a la fila 0.
- Gate focal: 29 mutaciones/paths adversariales; suite local completa: 70 passed; smoke real: 96 layers y secuencia greedy `1124 → 29626`.

## [2026-08-17] baseline | Oracle externo de residuales

- Añadido helper standalone contra llama.cpp pinned y sidecars F32 opcionales en `state-loop-probe --full-moe`.
- Añadido comparador con max-abs, RMSE, cosine y primera capa divergente.
- Token `42`: input layer 0 exacto; input layer 1 diverge (`max_abs≈1.08`, cosine≈0.964); argmax externo `1124` coincide con QX.
- llama.cpp KV F16 y Q8_0 mantienen la misma primera divergencia, por lo que el siguiente gate debe separar attention y MoE dentro de layer 0.

## [2026-08-17] update | Gate #7 attention/MoE/logits

- Capturados `Vcur`, contexto attention, post-attention, salida MoE, salida de bloque y 151936 logits F32 en QX/llama.cpp.
- Corregido bug de routing: los pesos top-8 ahora se renormalizan a suma `1.0` en ambas rutas QX.
- Layer-1 max-abs bajó de ~`1.08` a `0.00589` (F32/F16) y `0.01082` (INT8/Q8_0).
- La primera diferencia restante aparece en `Vcur`: QX usa activación F32; ggml usa activación temporal Q8_K para IQ4_XS.
- Logits completos no son pares (cosine ~`0.875`), aunque argmax coincide en token `1124`.
- Añadidos comparadores reproducibles de checkpoints/logits y modo KV F32 diagnóstico.

## [2026-08-17] update | Goldens supersedidos por router normalizado

- Los goldens anteriores `42 → 1124 → 29626` pertenecían al router sin renormalización y quedan supersedidos, no borrados del historial.
- Secuencia QX corregida: `42 → 1124 → 11287`.
- Checksums logits corregidos: `[12662891110960910958, 2895445711150549338]`.
- El segundo argmax/checksum sigue recalculado por el helper Q6_K independiente.

## [2026-08-17] baseline | Oracle externo secuencial

- Añadido oracle llama.cpp standalone con contexto persistente, KV F16/Q8_0 y múltiples pasos greedy desde IDs.
- `[42]`: llama F16/Q8_0 `[1124, 50853]`; QX INT8 `[1124, 11287]`.
- `Hello!` tokeniza a `[9707, 0]`: llama F16 `[358, 1184]`, llama Q8_0 `[358, 614]`, QX INT8 `[81379, 44707]`.
- GGUF SHA-256 registrado: `c8c2dc330dd1ec0c72c31b12e318647e6f9e0c773b9123eccfc3d12d9acc6652`.

## [2026-08-17] baseline | Residual final pre-head

- Oracle ampliado con captura directa `l_out-47`; QX usa `step-0-layer-47-output.f32`.
- F32/F16: max-abs `1099.6502`, RMSE `32.9520`, cosine `0.439279`.
- INT8/Q8_0: max-abs `1100.9628`, RMSE `33.0204`, cosine `0.440512`.

## [2026-08-17] update | Cierre reproducible del issue #7

- Corregida la documentación stale de [[moe-forward]]: Qwen3MoE renormaliza los pesos top-8 seleccionados a suma `1.0`.
- Añadido gate automatizado real para layers `0,1,24,47`, residual final `l_out-47`, logits completos y secuencias `[42]`/`Hello!`.
- Suite local final: `83 passed`; smoke y wiki lint: PASS; revisión independiente focal: `11 passed`, sin blockers.
- Publicado commit `42b3fd8b76acc26efdc7c53b6e7b427825b56b95`; CI `32064105028`: PASS.
- Issue GitHub #7 cerrado como validación completada: paridad residual, de logits y de secuencia refutada de forma reproducible.

## [2026-08-17] decision | F32 default y Q8_K compatible opt-in

- Añadido `q8_k_compat`: cuantización temporal block Q8_K y dot IQ4_XS × Q8_K con workspace reutilizable de 4672 bytes; Q5_K/Q6_K conservan fallback F32 explícito.
- `Vcur-0` mejora de max-abs `3.05e-4` a `7.45e-9`; `kqv_out-0` mejora a `1.49e-5`.
- La siguiente divergencia aparece en `ffn_moe_out-0`; logits y greedy continúan sin paridad.
- Oracle ampliado con `result_norm.f32`: Q8_K/F32-KV mantiene max-abs `8.71117`, RMSE `1.07502`, cosine `0.809426`.
- Gate byte-for-byte contra `quantize_row_q8_K_ref`: mixed/positive/negative/edge PASS; metadata de kernel refleja ejecución real.
- Benchmark un token/48 capas/KV INT8: mediana F32 `8.22912 s`, Q8_K `7.62247 s`, cinco repeticiones; mismo peak RSS observado.
- Decisión: F32 sigue default; Q8_K queda modo compatibilidad/diagnóstico. Próximo gate: bisect MoE.

## [2026-08-18] update | Bisect MoE por etapa y expertos Q8_K

- Oracle fijado a llama.cpp `768d2a481a99cb75ec9a03b95dadbd35e7acf496`; 18 callbacks internos y sidecars F32 lossless.
- `moe-stage-probe` acepta el `ffn_inp` exacto, exporta 12 etapas y falla cerrado ante tamaño, NaN/Inf, capa, output, overflow y layout incompatibles.
- Router/top-8/renorm coinciden; gate/up/SwiGLU/down/weighted de layer 0 quedan dentro de max-abs `1.20e-6` con Q8_K.
- Añadidos vec-dot CPU IQ2_XS×Q8_K e IQ3_XXS×Q8_K verificados contra traits públicos ggml; F32 sigue default y otros tipos conservan fallback explícito.
- End-to-end `ffn_moe_out-0` mejora a RMSE `1.40910e-4`; primera divergencia material actual: input layer 2 por tipos 22/23 de layer 1.
- Greedy sigue sin paridad. Benchmark 5 warm, 48 capas/KV INT8: F32 `9.24408 s/token`, Q8_K `3.88918 s/token`, speedup `2.37687×`; RSS mediano no aumenta.

## [2026-08-18] update | Expertos IQ2_S/IQ4_XS × Q8_K

- Corregido el alcance de tipos: en el oracle fijado `21=IQ3_S`, `22=IQ2_S`, `23=IQ4_XS`; layers 1/47 usan `(22,22,23)`.
- Añadidos goldens por fila contra traits públicos ggml para `IQ2_S × Q8_K` e `IQ4_XS × Q8_K` y captura `internals=N` seleccionable.
- Con el mismo `ffn_inp-1`, layer 1 pasa por etapa: mezcla final max-abs `4.35e-5`, RMSE `9.62e-7`, cosine ≈`1`.
- End-to-end, `layer-1-input` Q8_K/F32-KV queda en max-abs `7.50e-4`, pero se amplifica a `28.20` en input layer 2; logits/greedy siguen sin paridad.
- Benchmark 5 warm, 48 capas/KV INT8: F32 `8.00310 s/token`, Q8_K `2.28223 s/token`, speedup `3.50670×`; RSS difiere sólo una página.
- F32 sigue default; metadata agrega familias realmente ejecutadas y conserva fallback explícito para tipos sin golden.
- Review independiente detectó que el string agregado no preservaba gate/up frente a down en todas las mezclas. Se añadieron campos exactos por rol y tests RED→GREEN para layers 0–1 y fallback layer 24.

## [2026-08-18] fix | Q5_K atención y sensibilidad layer 1→2

- El probe same-input reprodujo una divergencia antes del MoE de layer 1 y un golden independiente confirmó un bug real en el decoder Q5_K.
- Corregido el layout Q5_K y añadido `Q5_K × Q8_K` escalar contra ggml; metadata de atención/state-loop distingue `q5_k_q8_k`, `iq4_xs_q8_k` y combinaciones con fallback.
- Con el mismo `attn_norm-1` y KV F16, `attn_out-1` queda en max-abs `6.33e-8`, RMSE `1.51e-8`, cosine ≈`1`.
- Post-fix Q8_K/F32-KV: layer-2-input RMSE `0.00660423` (`94.91×` mejor), logits RMSE `0.0346769` (`42.71×` mejor), argmax `1124`.
- El top-8 permanece `[68,114,55,90,0,9,28,73]`; experto 68 aporta `99.03%` del delta MoE. Su SwiGLU coincide con `SiLU(gate)×up`, por lo que la diferencia restante se clasifica como sensibilidad cuantizada esperada, no otro bug de layout.
- El comparador falla cerrado si raw weights, probabilidades seleccionadas, suma o pesos normalizados se contradicen; delta real raw max-abs `0.000261843`, normalizado `0.000501402`.
- La matriz greedy fija pasa: QX F32/INT8 `[1124,50853]` para `[42]` y `[358,1184]` para `Hello!`, igual a llama F16; checksums logits `[10967348620636053936,14548714559300682082]`.
- Benchmark post-fix, 5 warm/48 capas/KV INT8: F32 `8.64031 s/token`, Q8_K `2.41209 s/token`, speedup `3.58208×`; RSS mediano idéntico `5,627,904 B`.

## [2026-08-18] fix | Q6_K atención y sweep layer 2→logits

- El sweep completo localizó la siguiente amplificación material en layer 46→47: delta L2 `0.358050 → 1.25473`, gain `3.50435×`.
- `blk.46.attn_output.weight` es Q6_K (`ggml_type=14`) y Q8_K caía a F32. Se añadió `Q6_K × Q8_K` escalar y metadata exacta de familias.
- Cuatro bloques Q6_K completos y tres filas Q6_K×Q8_K reales se comparan contra traits públicos ggml.
- Same-input layer 46, `ffn_inp` mejora de max-abs `0.00445557` a `1.19e-7` (`37,376×`).
- End-to-end no mejora: layer-47 RMSE `0.0309398`, logits RMSE `0.0393805`; argmax `1124` y secuencia `[1124,50853]` permanecen.
- Bisect causal: atención gain `0.153×`, MoE gain `3.663×`, layer output gain `3.993×`; top-8 estable y experto 74 aporta `76.08%` del delta MoE.
- Benchmark de probe 48 capas/MoE: F32 `8.52569 s/token`, Q8_K `2.34729 s/token`, speedup `3.63214×`; no incluye lm_head/tokenización.
- F32 permanece default; Q8_K permanece CPU-only opt-in. Siguiente gate: layer 47→RMSNorm final→logits con input idéntico.

## [2026-08-18] fix | Final lm_head Q6_K × Q8_K same-input

- Añadido `final-head-probe` con residual F32 exacto, sidecars de RMSNorm/logits completos y negativos fail-closed.
- La ruta opt-in cuantiza el norm una sola vez y reutiliza ocho bloques Q8_K para las 151936 filas; F32 conserva `dequant_f32` como default.
- Same-input: RMSNorm RMSE `1.17205e-7`; logits RMSE `4.91155e-7`, max-abs `2.38419e-6`, argmax `1124`.
- End-to-end: logits mejoran de RMSE `0.0393805` a `0.0257469` y max-abs `0.187882` a `0.132130`; no se afirma paridad global.
- La divergencia restante ya entra desde `l_out-47`; siguiente gate: layer 47 same-input por atención/MoE.
