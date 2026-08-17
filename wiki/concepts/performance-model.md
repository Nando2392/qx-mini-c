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
lower-bound estimate: ~0.10 token/s
```

No incluye final norm/lm_head/tokenizer y no es benchmark final.

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
