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
