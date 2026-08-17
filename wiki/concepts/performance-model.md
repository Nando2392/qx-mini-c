---
title: Performance Model
created: 2026-08-17
updated: 2026-08-17
type: concept
tags: [performance, benchmark, cpu, cuda, memory, kv-cache]
sources: [raw/project/project-state-2026-08-17.md]
confidence: high
---

# Performance model

## Evidencia medida

Hardware observado:

```text
RTX 4070 Laptop: 8188 MiB VRAM
VRAM libre durante medición: ~6.6 GiB
RAM instalada: 31.67 GiB
RAM disponible: ~15.58 GiB
```

Forward real de layer 0:

```text
median: ~0.2085 s/layer
48-layer extrapolation: ~10.01 s/token
```

No incluye final norm/lm_head/tokenizer y no es benchmark final.

Comparación real F32/Q8_K, un token, 48 capas, KV INT8, cinco repeticiones warm:

| Activación | cold | mediana warm | MAD | peak RSS |
|---|---:|---:|---:|---:|
| F32 | 8.19598 s | 8.22912 s | 0.04541 s | 5,603,328 B |
| Q8_K compat | 7.57912 s | 7.62247 s | 0.02807 s | 5,603,328 B |

**Medido:** Q8_K fue ~7.4% más rápido en mediana dentro de este probe escalar. Usa 4672 bytes de workspace temporal. Es latencia total por token, no throughput; no incluye contexto largo y no sustituye profiling por proyección/layer. Evidencia: [[f32-vs-q8k-activation]].

## Bytes activos derivados del QXF

| Categoría | GiB/token |
|---|---:|
| Atención | 0.479 |
| Routers | 0.047 |
| Top-8 expert slices | 0.574 |
| Norms | 0.001 |
| lm_head | 0.238 |
| Total | 1.339 |

KV INT8 a 4K añade ~192 MiB.

## Consecuencia

El gap actual es implementación escalar/I/O, no KV. La prioridad es [[optimization-priorities]], no prometer techos GPU antes de tener backend CUDA.

Estado: [[current-status-and-roadmap]].
