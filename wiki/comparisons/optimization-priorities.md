---
title: Optimization Priorities
created: 2026-08-17
updated: 2026-08-17
type: comparison
tags: [performance, cpu, cuda, memory, roadmap]
sources: [raw/project/project-state-2026-08-17.md]
confidence: medium
---

# Optimization priorities

| Prioridad | Cambio | Razón | Gate |
|---:|---|---|---|
| 1 | mmap QXF | elimina seek/read/copy por span | logits iguales |
| 2 | buffers persistentes | elimina malloc/free del hot path | RSS estable |
| 3 | dequant+dot fusionado | evita materializar 256 floats | golden quant |
| 4 | thread pool por filas/expertos | usa CPU disponible | deterministic output |
| 5 | AVX2/FMA | mejora kernel CPU | tolerancia numérica |
| 6 | expert cache | reduce I/O y prepara híbrido | hit-rate/bytes |
| 7 | CUDA híbrido | dense residency + cache expertos | full-layer golden |
| 8 | prefill GEMM | reduce TTFT | prompt golden |
| 9 | speculative/KV2 | sólo tras runtime estable | lossless/PPL |

## No priorizar todavía

- CUDA Graphs sin backend CUDA.
- EAGLE sin forward multi-token eficiente ni draft head compatible.
- KV 2-bit a contexto 4K: ahorro limitado frente al gap escalar actual.
- Persistent kernels/PTX antes de Nsight.

Base cuantitativa: [[performance-model]]. Disciplina: [[auto-research-loop]].
