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
| 3 | dequant+dot fusionado | Issue #26 reduce temporales final-head Q6_K con kernel opt-in | baseline/fused exactos; no default promotion |
| 4 | thread pool por filas/expertos | Issue #28 añade primer pool real opt-in sólo para filas del final-head F32; MoE/expertos siguen fuera de alcance | deterministic output; fail-closed policy |
| 5 | AVX2/FMA | Issue #29 añade primer gate opt-in `--simd-policy avx2-fma` para el dot F32 del final-head Q6_K, detrás de `--kernel-policy fused`, F32, thread serial y runtime CPU gates; la salida conserva reducción double determinística para equivalencia exacta | scalar default; logits checksum equivalence; no default promotion ni speedup no medido |
| 6 | expert cache | Issue #30 añade superficie fail-closed `--expert-cache-policy none` y `expert_cache_profile` para separar baseline sin cache de futuros hits residentes | no-cache default; fake hits rechazados; sin speedup no medido |
| 7 | CUDA híbrido | Issue #31 añade superficie fail-closed `--cuda-policy none` y `cuda_profile` para separar baseline CPU-only de futuros backends híbridos | CPU-only default; fake kernels/device bytes rechazados; sin speedup no medido |
| 8 | prefill GEMM | Issue #32 añade superficie fail-closed `--prefill-gemm-policy none` y `prefill_gemm_profile` para separar baseline escalar de futuros kernels GEMM de prefill | scalar/current prefill default; fake GEMM calls rechazados; sin speedup no medido |
| 9 | speculative/KV2 | Issue #33 añade superficies fail-closed `--speculative-policy none` y `--kv2-policy none` con provenance explícita antes de draft decode/KV2 real | greedy/current KV default; fake draft/KV2 counters rechazados; sin speedup no medido |

## No priorizar todavía

- CUDA Graphs sin backend CUDA.
- EAGLE sin forward multi-token eficiente ni draft head compatible.
- KV 2-bit a contexto 4K: ahorro limitado frente al gap escalar actual.
- Persistent kernels/PTX antes de Nsight.

El kernel scalar `IQ4_XS × Q8_K` ya existe como modo `q8_k_compat` y no pasa a default. El baseline reproducible [[cpu-inference-baseline]] preserva el A/B F32/Q8_K y separa startup, prefill, decode, total y RSS: en el slice fijado observa `4.76489×` en prefill, `3.77551×` en decode y `3.98929×` total. F32 selecciona `[358,1184]` y Q8_K `[358,614]`; por tanto no existe equivalencia cross-mode ni paridad global. Antes de SIMD/threading, mantener este gate fail-closed y exigir causalidad separada para cualquier cambio de kernel. Véase [[f32-vs-q8k-activation]].

La prioridad 1 se ejecuta en [[qxf-mmap-io]] como backend read-only opt-in. Buffered sigue default/control; el gate exige igualdad exacta buffered/mmap dentro de F32 y dentro de `q8_k_compat`. El cambio no autoriza iniciar buffers persistentes, threading, SIMD ni CUDA en el mismo issue.

La prioridad 2 se ejecuta en [[persistent-scratch-buffers]] como política `--scratch-policy persistent` opt-in. El default sigue siendo `ephemeral`. En la matriz 2×2×2 de Issue #25 persistent reduce `480` malloc y `672` frees medianos por celda, retiene `65,536` bytes de scratch y conserva outputs exactos dentro de cada activación/backend. El wall-clock queda mixto/no material; no se promueve a default ni autoriza prioridad 3.

La prioridad 3 se ejecutó en Issue #26 como `--kernel-policy fused` opt-in limitado al final-head Q6_K F32. El default sigue `baseline`; la evidencia está en `wiki/evidence/issue-26-fused-dequant-dot-baseline.json` y no autoriza fusionar projection/MoE ni promover default.

La prioridad 4 empezó en Issue #27 con contrato `--thread-policy serial --threads 1`, `thread_profile` y rechazo fail-closed. Issue #28 añade `--thread-policy pool --threads N` como opt-in limitado al row loop independiente de `output.weight` en final-head Q6_K F32. Serial sigue default; `q8_k_compat`, `threads < 2`, `threads > 64`, políticas no soportadas, y rutas sin `--final-head` fallan cerradas. La evidencia mínima está en `wiki/evidence/issue-27-thread-policy-serial-baseline.json` y `wiki/evidence/issue-28-thread-pool-final-head-baseline.json`. No autoriza paralelizar MoE/expertos ni promover defaults.

La prioridad 6 empieza en Issue #30 como contrato de provenance: `--expert-cache-policy none` es el único valor soportado, queda default, y el payload nativo expone `expert_cache_profile` con hits/misses/bytes en cero. Este slice no implementa cache residente ni autoriza speedup; sólo bloquea claims falsos y prepara el A/B futuro.

La prioridad 7 empieza en Issue #31 como contrato de provenance CUDA: `--cuda-policy none` es el único valor soportado, queda default, y el payload nativo expone `cuda_profile` con backend `none`, bytes de dispositivo/transferencias y launches en cero. Este slice no implementa CUDA, residency ni scheduler híbrido; sólo bloquea claims falsos y prepara el gate futuro.

La prioridad 8 empieza en Issue #32 como contrato de provenance prefill GEMM: `--prefill-gemm-policy none` es el único valor soportado, queda default, y el payload nativo expone `prefill_gemm_profile` con backend `none`, llamadas GEMM, tokens batched, filas fusionadas y bytes temporales en cero. Este slice no implementa GEMM, BLAS/CUDA ni batching de prefill; sólo bloquea claims falsos y prepara el gate futuro de TTFT.

La prioridad 9 empieza en Issue #33 como contrato de provenance speculative/KV2: `--speculative-policy none` y `--kv2-policy none` son los únicos valores soportados, quedan default, y el payload nativo expone `speculative_profile` y `kv2_profile` con backend/formato `none` y contadores cero. Este slice no implementa draft decode, aceptación speculative, compresión KV2 ni PPL/lossless gate; sólo bloquea claims falsos y prepara el contrato futuro.

Base cuantitativa: [[performance-model]]. Disciplina: [[auto-research-loop]].
