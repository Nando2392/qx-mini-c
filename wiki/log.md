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
