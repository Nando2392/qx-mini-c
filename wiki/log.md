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
