---
title: Optimization Priorities
created: 2026-08-17
updated: 2026-08-21
type: comparison
tags: [performance, cpu, cuda, memory, roadmap]
sources: [raw/project/project-state-2026-08-17.md]
confidence: medium
---

# Optimization priorities

| Prioridad | Cambio | Razón | Gate |
|---:|---|---|---|
| 1 | mmap QXF | elimina seek/read/copy por span; Issue #24 en gate 2x2 | outputs buffered/mmap iguales por modalidad |
| 2 | buffers persistentes | Issue #25 reduce allocation/free del scratch de atención/MoE sin cambiar outputs | opt-in; RSS/outputs/counters |
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

La prioridad 1 se ejecuta en [[qxf-mmap-io]] como backend read-only opt-in. Buffered sigue default/control; el gate exige igualdad exacta buffered/mmap dentro de F32 y dentro de `q8_k_compat`. El cambio no autoriza iniciar buffers persistentes, threading, SIMD ni CUDA en el mismo issue.

La prioridad 2 se ejecuta en [[persistent-scratch-buffers]] como política `--scratch-policy persistent` opt-in. El default sigue siendo `ephemeral`. En la matriz 2×2×2 de Issue #25 persistent reduce `480` malloc y `672` frees medianos por celda, retiene `65,536` bytes de scratch y conserva outputs exactos dentro de cada activación/backend. El wall-clock queda mixto/no material; no se promueve a default ni autoriza prioridad 3.

Base cuantitativa: [[performance-model]]. Disciplina: [[auto-research-loop]].
