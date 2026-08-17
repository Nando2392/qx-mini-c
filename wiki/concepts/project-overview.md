---
title: Project Overview
created: 2026-08-17
updated: 2026-08-17
type: summary
tags: [runtime, qwen3-moe, open-source]
sources: [raw/project/project-state-2026-08-17.md]
confidence: high
---

# Project overview

QX-mini-MoE es un runtime experimental propio en C para estudiar inferencia local de [[moe-forward|Qwen3-30B-A3B]] con formato [[qxf-format|QXF1]]. Prioriza equivalencia numérica antes que velocidad.

## Objetivo

```text
GGUF real → QXF mmap-friendly → forward MoE completo → tokens verificables
```

## Principios

- Correctitud antes de optimización.
- Python sólo como referencia/test, no en el path final.
- Modelos y pesos no se versionan.
- Todo claim de rendimiento sigue [[performance-model]].
- Cada fallo alimenta [[auto-research-loop]].

## No es todavía

- Un runtime de producción.
- Un backend CUDA.
- Inferencia completa de 48 capas.
- Evidencia de tokens idénticos a llama.cpp/Transformers.

Estado operativo: [[current-status-and-roadmap]]. Arquitectura: [[architecture]].
