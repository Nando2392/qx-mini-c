---
title: Auto Research Loop
created: 2026-08-17
updated: 2026-08-17
type: concept
tags: [auto-research, validation, benchmark, risk]
sources: [raw/project/project-state-2026-08-17.md]
confidence: high
---

# Auto Research loop

Aplicación al proyecto de un ciclo inspirado en el enfoque experimental de Karpathy:

```text
pregunta concreta
→ hipótesis falsable
→ test RED o benchmark baseline
→ cambio mínimo
→ ejecución real
→ comparación
→ conservar/revertir
→ registrar evidencia y siguiente hipótesis
```

## Reglas

- Una optimización sin baseline no se acepta.
- Un fallo numérico dispara investigación antes del siguiente fix.
- Un probe parcial no se presenta como tok/s de inferencia.
- Cada experimento registra hardware, comando, datos y limitaciones.
- Los resultados negativos se conservan: evitan repetir caminos fallidos.

## Scorecard por experimento

```text
correctness gate
wall latency
bytes leídos
allocations/hot path
RSS
p50/p95/p99
thermal drift
```

Este ciclo gobierna [[optimization-priorities]] y depende de [[numerical-correctness]].
