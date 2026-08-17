# QX-mini-MoE Wiki Schema

## Domain

Runtime local-first escrito en C para ejecutar Qwen3-30B-A3B MoE desde GGUF mediante el formato QXF1, con correctitud numérica antes de optimización.

## Conventions

- Archivos en minúsculas y con guiones.
- Toda página de conocimiento usa frontmatter YAML.
- Mínimo dos `[[wikilinks]]` por página.
- Claims etiquetados como **medido**, **estimado**, **implementado**, **parcial** o **no implementado**.
- Los benchmarks parciales nunca se presentan como inferencia completa.
- `raw/` es inmutable y contiene evidencia o fuentes capturadas.
- Toda página nueva entra en `index.md`; toda acción entra en `log.md`.
- Actualizar `updated` al modificar una página.

## Frontmatter

```yaml
---
title: Título
created: 2026-08-17
updated: 2026-08-17
type: concept | comparison | query | summary
tags: [runtime, qwen3-moe]
sources: [raw/project/project-state-2026-08-17.md]
confidence: high | medium | low
---
```

## Tag taxonomy

- Proyecto: `runtime`, `qwen3-moe`, `qxf`, `roadmap`
- Correctitud: `validation`, `golden`, `quantization`, `testing`
- Rendimiento: `performance`, `cpu`, `cuda`, `memory`, `kv-cache`
- Operación: `auto-research`, `benchmark`, `risk`, `open-source`

## Evidence policy

1. **Medido** requiere comando y salida real.
2. **Implementado** requiere código ejecutado.
3. **Validado** requiere referencia independiente o test determinista.
4. **Estimado** debe mostrar supuestos y no puede promocionarse a benchmark.
5. **No implementado** no puede describirse en presente.

## Page thresholds

- Crear página si el tema es central al runtime o aparece en dos fuentes.
- Dividir páginas sobre ~200 líneas.
- Las contradicciones se conservan y se marcan; no se sobrescriben silenciosamente.
