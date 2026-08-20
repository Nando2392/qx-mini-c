---
title: Optimization Priorities
created: 2026-08-17
updated: 2026-08-20
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

El kernel scalar `IQ4_XS × Q8_K` ya existe como modo `q8_k_compat` y no pasa a default. El baseline reproducible [[cpu-inference-baseline]] preserva el A/B F32/Q8_K y separa startup, prefill, decode, total y RSS: en el slice fijado observa `4.76489×` en prefill, `3.77551×` en decode y `3.98929×` total. F32 selecciona `[358,1184]` y Q8_K `[358,614]`; por tanto no existe equivalencia cross-mode ni paridad global. Antes de SIMD/threading, mantener este gate fail-closed y exigir causalidad separada para cualquier cambio de kernel. Véase [[f32-vs-q8k-activation]].

Base cuantitativa: [[performance-model]]. Disciplina: [[auto-research-loop]].
