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
